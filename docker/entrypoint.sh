#!/bin/bash
# Training entrypoint with durable logging.
#
# Why this exists: a bare `ENTRYPOINT ["python", ...]` writes its traceback to a stream
# nobody keeps. When the process dies early there is nothing left to read — no log, no
# exit code, just an empty save dir. This wrapper tees everything to a file and reports
# the exit status even on an uncaught exception.
#
# NOTE: this does NOT run under Vast.ai's --ssh or --jupyter launch modes. Those replace
# the image entrypoint with Vast's own setup scripts, so the container starts but training
# never begins. On Vast, invoke training from --onstart instead. This wrapper covers local
# `docker run` and Vast's entrypoint launch mode.

set -Eeuo pipefail

# Everything downstream is cwd-relative: the script loads ./pretrained/, configs/,
# splits/. Docker's WORKDIR only applies when Docker launches this entrypoint itself;
# Vast's onstart wrapper (and a manual SSH invocation) guarantee no particular cwd.
cd /workspace

LOG_DIR=${LOG_DIR:-/workspace/exp}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# Everything from here on lands in both the container log and $LOG. $LOG_DIR must be a
# mounted volume or the log dies with the container, which is the failure this file exists
# to prevent.
exec > >(tee -a "$LOG") 2>&1

# Fires on any exit path including an uncaught Python exception, so the exit code is always
# recorded. Without it, `set -e` would abort silently.
trap 'rc=$?; echo "=== EXIT rc=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="; exit $rc' EXIT

echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "GIT_SHA=${GIT_SHA:-unknown}  host=$(hostname)"
echo "--- gpu ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "  nvidia-smi unavailable"
# wandb.init() runs before the first batch and dies without a credential unless
# USE_WANDB is offline/disabled, so a missing key here predicts an immediate exit.
# Names only, never values: this log is world-readable via `vastai logs`.
echo "--- credentials present (values not shown) ---"
echo "  USE_WANDB=${USE_WANDB:-<unset, config decides>}"
for v in WANDB_API_KEY WANDB_PROJECT WANDB_RUN_ID; do
  if [ -n "${!v:-}" ]; then echo "  $v=<set>"; else echo "  $v=<MISSING>"; fi
done
# DataLoader workers hand batches over via /dev/shm; Docker's default is 64 MB and a
# too-small shm kills workers with a bus error minutes in. On Vast we cannot pass
# --ipc=host, so make the actual size visible in the boot log.
echo "--- /dev/shm (needs to be well above 64M for the dataloader workers) ---"
df -h /dev/shm || true
echo "--- args: $* ---"

# Deliberately NOT `exec python ...`: exec replaces this shell, and the EXIT trap above goes
# with it, so the rc line never printed on any normal exit path -- the one thing this file
# exists to guarantee. Running Python as a background job and waiting keeps the trap alive.
#
# The TERM/INT trap replaces what exec gave us for free: on container stop Docker signals
# PID 1 (this shell), which forwards to Python and waits for it to finish shutting down,
# instead of bash swallowing the signal and Python being SIGKILLed after the grace period.
python -u unimatch_v2_1gpu.py "$@" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"' TERM INT
wait "$pid"
