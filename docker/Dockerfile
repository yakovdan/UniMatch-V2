# syntax=docker/dockerfile:1

# 2.10.0 rather than newer, deliberately: it is the newest torch with a prebuilt
# xformers whose C++/CUDA extensions actually load (see requirements-docker.txt).
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# PIP_BREAK_SYSTEM_PACKAGES: this image carries Ubuntu's PEP 668 marker at
# /usr/lib/python3.12/EXTERNALLY-MANAGED, so pip otherwise refuses to install into the
# system interpreter ("error: externally-managed-environment"). There is no OS package
# manager to protect in a container, and torch itself already lives in the same
# /usr/local/lib/python3.12/dist-packages tree that pip writes to.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# util/dist_helper.py::setup_distributed reads RANK/WORLD_SIZE, and init_process_group's
# env:// rendezvous needs MASTER_ADDR/MASTER_PORT, even single-GPU (upstream launches via
# torchrun, which sets these). Baking them lets `python unimatch_v2_1gpu.py` run bare;
# the script asserts world_size == 1, so these are the only valid values for it anyway.
ENV RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500

# The dataset gets its own (biggest, least-churning) layer. Only PASCAL:
# data/checkpoints holds released eval-only checkpoints and SegmentationClass.zip is an
# unpacked leftover — neither is used by unimatch_v2_1gpu.py.
COPY data/PASCAL/ /opt/data/PASCAL/
WORKDIR /workspace

# DINOv2 weights: unimatch_v2_1gpu.py loads ./pretrained/{backbone}.pth relative to cwd
# (hardcoded), so these must live under WORKDIR. Both small and base ship so
# `--backbone dinov2_base` works without a rebuild (~415 MB total).
COPY pretrained/ ./pretrained/

# Two packages: the base image already provides torch/torchvision/numpy/pillow/pyyaml;
# wandb is the only import in the script's module closure it lacks (metric logging),
# and xformers enables the DINOv2 memory-efficient attention path (pinned to the cu130
# build that matches this image's torch — see requirements-docker.txt).
# image-constraints.txt pins the 76 distributions the base image
# ships. It installs
# nothing; it caps what pip may pick when a transitive dependency wants a different
# version, so a conflict is a hard error naming the culprit instead of a silent swap.
COPY docker/requirements-docker.txt docker/image-constraints.txt /tmp/
RUN python -m pip install \
    --no-cache-dir \
    -c /tmp/image-constraints.txt \
    -r /tmp/requirements-docker.txt \
 && rm -rf /tmp/requirements-docker.txt /tmp/image-constraints.txt

# Application code after dependencies so source edits don't reinstall anything.
COPY dataset/ ./dataset/
COPY model/ ./model/
COPY util/ ./util/
COPY configs/ ./configs/
COPY splits/ ./splits/
COPY supervised.py unimatch_v2_1gpu.py ./

# --data-root must appear in any CMD override too: configs/*.yaml carry the placeholder
# "Your/PASCAL/Path", so the config alone never resolves the dataset.
#
# Mount the save path — checkpoints, wandb run files and the entrypoint log all land
# there, and the script auto-resumes from latest.pth in it, so a restarted container
# continues training instead of starting over:
#
#   ./docker/docker_build.sh          # stamps GIT_SHA; PUSH=1 to publish + verify digest
#   docker run --gpus all --ipc=host \
#       -v "$PWD/exp:/workspace/exp" \
#       unimatch-v2:latest
#
# --ipc=host (or --shm-size=8g): DataLoader(num_workers=4) passes batches through
# /dev/shm, and Docker's default 64 MB shm kills the workers with a bus error.
#
# wandb: the script calls wandb.init at startup with mode from USE_WANDB env var >
# config wandb_mode > online. The configs default to online, so either forward the
# credential with `-e WANDB_API_KEY` (source it on the host first) or run without
# tracking via `-e USE_WANDB=offline` (sync later with `wandb sync`) or
# `-e USE_WANDB=disabled` — otherwise wandb.init dies in a credential-less container.
#
# Other splits/datasets/backbones are CMD overrides, e.g.:
#   docker run --gpus all --ipc=host -v "$PWD/exp:/workspace/exp" unimatch-v2:latest \
#       --config configs/pascal.yaml \
#       --labeled-id-path splits/pascal/92/labeled.txt \
#       --unlabeled-id-path splits/pascal/92/unlabeled.txt \
#       --save-path exp/pascal/unimatch_v2/dinov2_base/92 \
#       --data-root /opt/data/PASCAL --backbone dinov2_base
#
# entrypoint.sh wraps the python invocation and tees output to a timestamped log inside
# LOG_DIR (default /workspace/exp, i.e. the mount above), recording the exit code via a
# trap so an early crash leaves evidence instead of an empty directory.
#
# Beware: Vast.ai's --ssh and --jupyter launch modes REPLACE this entrypoint with their
# own setup scripts, so training never starts there. On Vast, drive training from
# --onstart instead: docker/onstart.sh calls this entrypoint itself, after exporting the
# container env to SSH sessions and repairing /root/.ssh. It is passed from the host
# (`--onstart docker/onstart.sh`), not baked in, so it can change without a rebuild.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--config", "configs/pascal.yaml", \
     "--labeled-id-path", "splits/pascal/366/labeled.txt", \
     "--unlabeled-id-path", "splits/pascal/366/unlabeled.txt", \
     "--save-path", "exp/pascal/unimatch_v2/dinov2_small/366", \
     "--data-root", "/opt/data/PASCAL"]

# Provenance stamp. Deliberately the LAST thing in the file: an ARG that changes on every
# commit would bust the cache for every layer after it, so nothing expensive may follow.
#
# Read it back from any image, local or pulled:
#   docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' IMAGE
# or from inside a running container:  echo $GIT_SHA
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="$GIT_SHA"
ENV GIT_SHA="$GIT_SHA"
