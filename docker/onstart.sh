#!/bin/bash
# Vast.ai onstart script for the unimatch-v2 image. Ported from
# ~/repos/VastAIOrchestration/onstart.sh, which has ~20 instances of production behind it.
#
#   vastai create instance <OFFER> --image yakovdan/unimatch-v2:<sha> --disk 60 \
#     --ssh --direct --cancel-unavail --label unimatch-92 \
#     --env '-e USE_WANDB=online -e SPLIT=92' \
#     --onstart docker/onstart.sh
#
# SSH launch mode replaces the image entrypoint, so training must be started from here or it
# never runs at all. (In args/entrypoint launch mode the opposite holds: the image's
# ENTRYPOINT runs and onstart is ignored entirely -- do not pass both and expect both.)
#
# Everything training-related is read from the environment, because --onstart takes no
# arguments. The B2_* credentials arrive automatically as account-level Vast env vars,
# injected into every instance this account creates -- and so does a WANDB_API_KEY that
# belongs to the wrong W&B account for this project. Pass -e WANDB_KEY=<key> to override
# it; see section 0b.

PROFILE=/etc/profile.d/00-vast-env.sh
ENVFILE=/etc/environment
LOG_DIR=${LOG_DIR:-/workspace/exp}

# ---- 0. repair /root/.ssh permissions -------------------------------------------------
# Vast provisions /root/.ssh/authorized_keys from the HOST into the container's overlay,
# before container init exists, so the container's umask never applies to it. On some
# machines it lands group/world-writable and sshd's StrictModes then refuses to read it:
# "Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys".
# The key is present and correct; sshd just will not trust the file containing it, and the
# instance reports `running` while being completely unreachable. Seen on 2 of ~20 instances
# (machines 58012, 36518). sshd re-reads the file per connection, so fixing modes here takes
# effect immediately, and costs nothing when they were already right.
if [ -d /root/.ssh ]; then
  chown -R root:root /root/.ssh
  chmod 700 /root/.ssh
  [ -f /root/.ssh/authorized_keys ] && chmod 600 /root/.ssh/authorized_keys
  echo "onstart: /root/.ssh perms -> $(stat -c '%A %U:%G' /root/.ssh) | authorized_keys $(stat -c '%A' /root/.ssh/authorized_keys 2>/dev/null || echo absent)"
fi

# ---- 0b. W&B identity ------------------------------------------------------------------
# Vast injects an ACCOUNT-LEVEL WANDB_API_KEY into every instance this account creates, and
# that key belongs to `yakovdan`. This project logs to a different account, so the per-run
# key is passed as WANDB_KEY (not an account-level name, hence no precedence ambiguity with
# the injected one) and exported over WANDB_API_KEY here, after injection.
#
# Note this OVERRIDES, where Calcium's entrypoint.sh bridges WANDB_KEY only as a fallback
# (`${WANDB_API_KEY:-${WANDB_KEY:-}}`). A fallback cannot help here: the wrong value is
# already set by the time onstart runs.
if [ -n "${WANDB_KEY:-}" ]; then
  export WANDB_API_KEY="$WANDB_KEY"
  echo "onstart: WANDB_API_KEY <- WANDB_KEY (overriding the account-level key)"
fi

# Resolve the key to a username before training starts. Six minutes into a run is a late
# and expensive moment to discover the wrong W&B account; `|| true` keeps a network hiccup
# from blocking the launch.
python - <<'PYIDENT' || true
import base64, json, os, urllib.request
key = os.environ.get("WANDB_API_KEY")
if not key:
    print("onstart: W&B identity: no key set"); raise SystemExit
req = urllib.request.Request("https://api.wandb.ai/graphql",
    data=json.dumps({"query": "{viewer{username entity}}"}).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()})
try:
    v = json.load(urllib.request.urlopen(req, timeout=20))["data"]["viewer"]
    print(f"onstart: W&B identity: username={v.get('username')} entity={v.get('entity')}")
except Exception as e:
    print(f"onstart: W&B identity check failed: {e}")
PYIDENT

# ---- 1. make the container env visible to SSH sessions -------------------------------
# SSH launch mode builds a clean env per session: without this, WANDB_API_KEY and B2_* are
# empty when you log in to rescue a checkpoint by hand.
: > "$PROFILE"
: > "$ENVFILE"
echo '# Written by onstart: container env for SSH sessions' >> "$PROFILE"

printenv | while IFS= read -r line; do
  key=${line%%=*}
  val=${line#*=}
  case "$key" in
    '' | *[!A-Za-z0-9_]* ) continue ;;
    PWD|OLDPWD|SHLVL|_|HOME|TERM|HOSTNAME|SHELL|USER|LOGNAME|LS_COLORS|MAIL ) continue ;;
  esac
  printf 'export %s=%q\n' "$key" "$val" >> "$PROFILE"
  printf '%s="%s"\n' "$key" "${val//\"/\\\"}" >> "$ENVFILE"
done
chmod 644 "$PROFILE" "$ENVFILE"
grep -q '00-vast-env' /root/.bashrc 2>/dev/null || \
  echo '[ -f /etc/profile.d/00-vast-env.sh ] && . /etc/profile.d/00-vast-env.sh' >> /root/.bashrc

# ---- 2. USE_WANDB means different things in different repos ---------------------------
# Calcium uses USE_WANDB as a 0/1 flag; unimatch_v2_1gpu.py uses it as a wandb MODE string
# and raises ValueError on anything but online/offline/disabled. The same account and the
# same .env feed both, so a copied `-e USE_WANDB=1` would otherwise kill this container
# seconds after boot. Translate loudly rather than dying.
case "${USE_WANDB:-}" in
  1) echo "onstart: WARNING USE_WANDB=1 is Calcium's flag form; using 'online'"; export USE_WANDB=online ;;
  0) echo "onstart: WARNING USE_WANDB=0 is Calcium's flag form; using 'disabled'"; export USE_WANDB=disabled ;;
esac

# ---- 3. diagnostics (these reach `vastai logs`) ---------------------------------------
mkdir -p "$LOG_DIR"
echo "=== onstart $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "GIT_SHA=${GIT_SHA:-unknown}  host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "  nvidia-smi unavailable"
# The image is CUDA 13.0: a machine whose driver predates r580 fails here, not later.
python -c 'import torch; print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())' \
  || echo "  torch import FAILED -- wrong driver for this cu130 image?"
echo "--- /dev/shm (dataloader workers; Vast sizes it from the GPU fraction) ---"
df -h /dev/shm || true

# Blackwell and newer (capability >= 10) have no fp32 xformers attention kernel in 0.0.35,
# and supervised.evaluate runs in fp32 -- so training completes epoch 0 and then dies at the
# first evaluation with "No operator found for memory_efficient_attention_forward". Decide
# it here from the actual GPU rather than trusting whoever writes the launch line to
# remember; an explicitly passed value still wins. See model/backbone/dinov2_layers/attention.py.
if [ -z "${XFORMERS_DISABLED:-}" ]; then
  CAP=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)
  if [ "${CAP:-0}" -ge 10 ]; then
    export XFORMERS_DISABLED=1
    echo "XFORMERS_DISABLED=1" >> /etc/environment
    echo "onstart: GPU capability ${CAP}.x -- exporting XFORMERS_DISABLED=1 (no fp32 xformers kernel above 9.0)"
  else
    echo "onstart: GPU capability ${CAP}.x -- keeping xformers enabled"
  fi
fi

# ---- 4. training arguments -------------------------------------------------------------
# GGR_MODE / GGR_DIM / GGR_SCOPE / GGR_REORTH_EVERY need no plumbing: unimatch_v2_1gpu.py
# reads them from the environment itself and they win over the command line.
SPLIT=${SPLIT:-92}
BACKBONE=${BACKBONE:-dinov2_small}
CONFIG=${CONFIG:-configs/pascal.yaml}
DATA_ROOT=${DATA_ROOT:-/opt/data/PASCAL}
RUN_NAME=${RUN_NAME:-${SPLIT}}
SAVE_PATH=${SAVE_PATH:-exp/pascal/unimatch_v2_1gpu/${BACKBONE}/${RUN_NAME}}

TRAIN_ARGS=(
  --config "$CONFIG"
  --labeled-id-path "splits/pascal/${SPLIT}/labeled.txt"
  --unlabeled-id-path "splits/pascal/${SPLIT}/unlabeled.txt"
  --save-path "$SAVE_PATH"
  --data-root "$DATA_ROOT"
  --backbone "$BACKBONE"
)
[ -n "${EPOCHS:-}" ] && TRAIN_ARGS+=(--epochs "$EPOCHS")
[ "${BF16:-1}" = "1" ] && TRAIN_ARGS+=(--bf16)
[ -n "${STOP_AFTER:-}" ] && TRAIN_ARGS+=(--stop-after "$STOP_AFTER")

echo "--- args: ${TRAIN_ARGS[*]} ---"
echo "--- env ---"
env | grep -E '^(USE_WANDB|WANDB_PROJECT|WANDB_ENTITY|SPLIT|BACKBONE|EPOCHS|BF16|STOP_AFTER|GGR_|SEED|DETERMINISTIC)' | sort
for v in WANDB_API_KEY B2_KEY_ID B2_APP_KEY B2_BUCKET_NAME B2_S3_ENDPOINT; do
  if [ -n "${!v:-}" ]; then echo "  $v=<set>"; else echo "  $v=<MISSING>"; fi
done

# ---- 5. start training ------------------------------------------------------------------
cd /workspace || { echo "FATAL: /workspace missing"; exit 1; }

# entrypoint.sh does the tee-to-timestamped-log and the exit-code line, so it is invoked
# here rather than python directly -- SSH mode skipped it as the ENTRYPOINT, not as a script.
#
# > /dev/null is not laziness: once onstart exits, writes to its stdout would hit a closed
# pipe and SIGPIPE the trainer. So `vastai logs` holds the boot diagnostics above and
# nothing after; training progress lives in $LOG_DIR/train_*.log and in W&B.
nohup /usr/local/bin/entrypoint.sh "${TRAIN_ARGS[@]}" > /dev/null 2>&1 &

echo "training launched in background, pid $!, log $LOG_DIR/train_*.log, save path $SAVE_PATH"
echo "=== onstart done ==="
