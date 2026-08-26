#!/usr/bin/env bash
# Build (and optionally push) the UniMatch V2 single-GPU training image.
#
#   ./docker/docker_build.sh            build only, three local tags
#   PUSH=1 ./docker/docker_build.sh     build, push, verify the registry digest matches
#
# Ported from ~/repos/Calcium/docker_build.sh. Unlike that one, PUSH defaults to 0: this
# repo has no image on Docker Hub yet, so pushing is an explicit decision, and Vast can
# only pull what has been pushed.
#
# Every build is stamped with the git commit it came from, so an image can always be traced
# back to source. See the provenance block at the end of docker/Dockerfile. Without this
# script the stamp is simply absent -- a plain `docker build` leaves GIT_SHA=unknown, which
# is what the current unimatch-v2:latest carries.
set -euo pipefail

IMAGE=${IMAGE:-yakovdan/unimatch-v2}
TAG=${TAG:-latest}
# Kept so plain `docker run unimatch-v2:latest ...` keeps working after a rebuild.
LOCAL_ALIAS=unimatch-v2:latest

cd "$(dirname "$0")/.."

# --- provenance -------------------------------------------------------------------------
# The build copies the working tree, not git HEAD, so a commit id alone can be a lie. Mark
# the build dirty if any TRACKED path the Dockerfile actually COPYs has uncommitted changes.
# Paths excluded by .dockerignore cannot affect the image and must not trip this check.
#
# -uno drops untracked files. Know what this costs here: data/PASCAL/ and pretrained/ are
# untracked in this repo and ARE copied into the image, so a clean tag means "every
# committed file in the image matches this commit", not "the image holds only committed
# files". The dataset and the DINOv2 weights are versioned by nothing but their content.
COPIED_PATHS=(dataset model util configs splits supervised.py unimatch_v2_1gpu.py docker)
RAW_SHA=$(git rev-parse HEAD)
SHORT_SHA=${RAW_SHA:0:12}
GIT_SHA=$RAW_SHA

if [ -n "$(git status --porcelain -uno -- "${COPIED_PATHS[@]}")" ]; then
    GIT_SHA="${RAW_SHA}-dirty"
    # Distinct tag too, so a dirty build never overwrites the immutable tag of a clean one.
    SHORT_SHA="${SHORT_SHA}-dirty"
    echo "WARNING: tracked files copied into the image have uncommitted changes."
    echo "         Tagging ${SHORT_SHA}; this build is NOT reproducible from git."
    git status --porcelain -uno -- "${COPIED_PATHS[@]}" | sed 's/^/           /'
fi

echo "==> building ${IMAGE}:${TAG}, ${IMAGE}:${SHORT_SHA}, ${LOCAL_ALIAS}  (revision ${GIT_SHA})"

docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  --build-arg GIT_SHA="${GIT_SHA}" \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:${SHORT_SHA}" \
  -t "${LOCAL_ALIAS}" \
  .

echo "==> built. recorded revision:"
docker image inspect \
  --format '    {{index .Config.Labels "org.opencontainers.image.revision"}}   (created {{.Created}})' \
  "${IMAGE}:${TAG}"

# --- push -------------------------------------------------------------------------------
if [ "${PUSH:-0}" != "1" ]; then
    echo "==> PUSH=0 (default), skipping push. Vast can only pull what is on the registry."
    exit 0
fi

echo "==> pushing (~5.5 GB compressed; the PASCAL layer is ~1.9 GB of it)"
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:${SHORT_SHA}"

# --- verify the remote actually matches -------------------------------------------------
# .RepoDigests is populated only once an image has been pushed or pulled, so an empty value
# here is itself proof the push did not land.
DIGEST_RE='^sha256:[0-9a-f]{64}$'

LOCAL=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${IMAGE}:${TAG}" \
        | grep "^${IMAGE}@" | head -1 | cut -d@ -f2 || true)

# Read the registry-side digest. Each strategy is validated to be a BARE digest before it is
# accepted: `imagetools inspect --format` is silently ignored by some buildx versions, which
# emit the full human-readable report instead -- capturing that blob and comparing it to a
# digest yields a guaranteed (and wrong) "out of sync".
remote_digest() {
    local ref=$1 out
    out=$(docker buildx imagetools inspect "$ref" --format '{{.Manifest.Digest}}' 2>/dev/null || true)
    [[ $out =~ $DIGEST_RE ]] && { printf '%s' "$out"; return 0; }

    out=$(docker buildx imagetools inspect "$ref" 2>/dev/null \
          | awk '/^Digest:[[:space:]]/ {print $2; exit}' || true)
    [[ $out =~ $DIGEST_RE ]] && { printf '%s' "$out"; return 0; }

    return 0   # nothing usable; caller reports it as "not found"
}
REMOTE=$(remote_digest "${IMAGE}:${TAG}")

echo "==> sync check"
echo "    local : ${LOCAL:-<none - never pushed>}"
echo "    remote: ${REMOTE:-<none - not found on registry>}"
if [ -n "$LOCAL" ] && [ "$LOCAL" = "$REMOTE" ]; then
    echo "    IN SYNC"
else
    echo "    OUT OF SYNC — the registry does not have this exact image"
    exit 1
fi
