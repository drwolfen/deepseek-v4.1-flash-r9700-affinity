#!/bin/bash
# post-encode.sh — runs after the distributed encode: DSpark draft experts + containers + checks.
set -u
B=${AFF_BUILD_DIR:-./build}; CK=${HF_MODEL_DIR:-/models/DeepSeek-V4.1-Flash}; CB=${CODEBOOK_RAW:-./v41cb.raw}
IM=${IMATRIX:-/models/imatrix-dsv41-262144.gguf}; OUT=${OUT_DIR:-./models/v41}
WORK_DIR=${WORK_DIR:-./work-v41}
ts(){ date +%T; }
echo "$(ts) waiting for 40 blobs"
until [ "$(ls "$WORK_DIR"/experts/*.affx 2>/dev/null | wc -l)" -ge 40 ]; do sleep 30; done
sleep 5; pgrep -f "[w]orker.sh" >/dev/null && { echo "$(ts) worker still running?"; sleep 60; }
echo "$(ts) 40 blobs; sizes: $(ls -l "$WORK_DIR"/experts/*.affx | awk "{print \$5}" | sort -u | tr "\n" " ")"
mkdir -p "$WORK_DIR"/experts-dspark
echo "$(ts) dspark experts"
nice -n 5 $B/aff-quantize --src $CK --dspark --codebook-raw $CB --experts-dir "$WORK_DIR"/experts-dspark --threads 32 > $OUT/dspark-experts.log 2>&1 || { echo "dspark experts FAILED"; tail -5 $OUT/dspark-experts.log; exit 1; }
echo "$(ts) dspark container"
$B/aff-quantize --src $CK --dspark --codebook-raw $CB --experts-from "$WORK_DIR"/experts-dspark --out $OUT/model.dspark.aff > $OUT/dspark-container.log 2>&1 || { echo "dspark container FAILED"; tail -5 $OUT/dspark-container.log; exit 1; }
grep -E "v4.1 map|done" $OUT/dspark-container.log
echo "$(ts) main container"
$B/aff-quantize --src $CK --codebook-raw $CB --imatrix $IM --experts-from "$WORK_DIR"/experts --out $OUT/model.aff > $OUT/main-container.log 2>&1 || { echo "main container FAILED"; tail -5 $OUT/main-container.log; exit 1; }
tr "\r" "\n" < $OUT/main-container.log | grep -E "v4.1 map|dense|profile|done"
echo "$(ts) fp8check"
$B/aff-fp8check $OUT/model.aff $CK > $OUT/fp8check.log 2>&1; tail -1 $OUT/fp8check.log
$B/aff-exphash $OUT/model.aff > $OUT/model.aff.exphash; echo "exphash: $(wc -l < $OUT/model.aff.exphash) experts"
echo "$(ts) POST-ENCODE DONE"
