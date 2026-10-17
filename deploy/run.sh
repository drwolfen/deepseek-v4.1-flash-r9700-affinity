#!/bin/bash
# Run the affinity server on two R9700s with the uncensored V4.1-Flash container.
# Adjust MODELS to where the .aff files live. Mirrors deploy/deepseek-v41-affinity.service.
set -eu

MODELS=${MODELS:-/models}
IMAGE=${IMAGE:-affinity:v41-vision}
NAME=${NAME:-deepseek-affinity}

exec docker run --rm --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --cpuset-cpus=0-23,48-71 \
  -e AFF_ENGRAM_WARM=1 \
  -e AFF_SPEC_BACKOFF=2 \
  -e AFF_ADAPTIVE_WIDTH=1 \
  -e AFF_BOUNDED_REPLAY=1 \
  -e AFF_RESIDENT_BIAS=0.008 \
  -e AFF_NOBUDGET_MAX_TOKENS=16384 \
  -p 8000:8000 \
  -v "$MODELS":"$MODELS" \
  "$IMAGE" \
    -m "$MODELS/model-uc.aff" \
    --tokenizer "$MODELS/tokenizer.json" \
    --host 0.0.0.0 --port 8000 \
    --gpu-headroom-mib 1024 \
    --dspark on \
    --dspark-model "$MODELS/model-uc.dspark.aff" \
    --engram "$MODELS/model-uc.engram" \
    --engram-tables "$MODELS" \
    --prefix-cache-dir "$MODELS/pcache" \
    --prefix-cache-gib 64 \
    --kv-size 524288 \
    --kv-commit 262144 \
    --host-pool-mib 200000 --host-pool-reserve-mib 1024 \
    --placement-strategy demand \
    --placement-save "$MODELS/placement.affprof" \
    --prefill-chunk-size 2816 \
    --default-effort 75 \
    --max-tokens-floor 0
