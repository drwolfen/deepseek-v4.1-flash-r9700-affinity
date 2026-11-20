#!/bin/bash
# Run the affinity server on two R9700s with DeepSeek-V4.1-Flash v5.3.0.
# Adjust MODELS to where the .aff files live. Mirrors deploy/deepseek-v41-affinity.service.
set -eu

MODELS=${MODELS:-/models/DeepSeek-V4.1-Flash-1017-MoonMath-aff}
TOKENIZER=${TOKENIZER:-/models/LLM-Models/DeepSeek-V4.1-Flash-1017-MoonMath-FP8/tokenizer.json}
ENGRAM_TABLES=${ENGRAM_TABLES:-/models/LLM-Models/DeepSeek-V4.1-Flash-1017-MoonMath-FP8}
STORAGE_DIR=${STORAGE_DIR:-/models}
IMAGE=${IMAGE:-deepseek-affinity:v5.3.0}
NAME=${NAME:-deepseek-affinity}

exec docker run --rm --name "$NAME" \
  --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --cpuset-cpus=0-23,48-71 \
  -e AFF_ENGRAM_WARM=1 \
  -e AFF_SPEC_BACKOFF=2 \
  -e AFF_ADAPTIVE_WIDTH=1 \
  -e AFF_BOUNDED_REPLAY=1 \
  -e AFF_FUSE_HC_TAP=1 \
  -e AFF_DRAFT_HEAD_INT4=1 \
  -e AFF_CACTUS_DELTA=0.35 \
  -e AFF_CACTUS_CAP_R=0.5 \
  -e AFF_CACTUS_ESCAPE=0.25 \
  -e AFF_RESIDENT_BIAS=0.008 \
  -e AFF_MOE_RECON_TILES=32 \
  -e AFF_PREFETCH_RING=0 \
  -e AFF_NOBUDGET_MAX_TOKENS=24576 \
  -e AFF_THINK_REPLY_RESERVE=4096 \
  -e AFF_THINK_LOOP_REPEATS=3 \
  -e AFF_THINK_FREEZE_BREAKER=1 \
  -e AFF_HC_FUSED_V2=1 \
  -e AFF_HC_CHAINED=1 \
  -e AFF_HC_CHAINED_MIN_ROWS=1 \
  -e AFF_SPARSE_INDEXER=1 \
  -e AFF_INDEXER_SCORE_BUDGET_MB=36 \
  -e AFF_DRAFT_SKIP_MASKED=1 \
  -e AFF_ATTN_LOOKAHEAD=0 \
  -e AFF_PCACHE_DELTA=1 \
  -p 8000:8000 \
  -v "$STORAGE_DIR":/models \
  "$IMAGE" \
    -m "$MODELS/model-moonmath-1017.aff" \
    --tokenizer "$TOKENIZER" \
    --host 0.0.0.0 --port 8000 \
    --raw-completions \
    --gpu-headroom-mib 384 \
    --dspark on \
    --dspark-model "$MODELS/model-moonmath-1017.dspark.aff" \
    --engram "$MODELS/model-moonmath-1017.engram" \
    --engram-tables "$ENGRAM_TABLES" \
    --prefix-cache-dir "$MODELS/pcache" \
    --prefix-cache-gib 64 \
    --persistent-prefix-cache \
    --kv-size 327680 \
    --kv-commit 262144 \
    --host-pool-mib 200000 --host-pool-reserve-mib 1024 \
    --placement-strategy demand \
    --placement-hysteresis 1.6 --placement-min-gain 3.0 \
    --placement-save "$MODELS/placement.affprof" \
    --prefill-chunk-size 2816 \
    --default-effort 60 \
    --repeat-penalty 1.1 \
    --repeat-last-n 96 \
    --dry-delta 40 \
    --max-tokens-floor 0
