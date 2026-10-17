#!/bin/bash
# finish-stage-a.sh — wait for the 40-layer expert encode, assemble the uncensored container,
# run the gates. Does NOT touch the service or the GPUs (swap/benchmark is a separate, explicit step).
set -u
D=/mnt/fast-storage/DeepSeek-V4.1-Flash-UNCENSORED-aff
CK=/mnt/fast-storage/LLM-Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
B=/mnt/fast-storage/affinity/build/aff-quantize
IM=$D/imatrix/imatrix-dsv41-262144.gguf
CB=$D/uc-v41cb.raw
log(){ echo "$(date +%T) $*"; }

log "waiting for 40 blobs and an idle encoder"
while :; do
  n=$(ls "$D"/experts/*.affx 2>/dev/null | wc -l)
  r=$(pgrep -x aff-quantize | wc -l)
  if [ "$n" -ge 40 ] && [ "$r" -eq 0 ]; then break; fi
  sleep 60
done
log "40 blobs present, encoder idle: $(ls -l "$D"/experts/*.affx | awk '{s+=$5} END{printf "%.1f GiB\n", s/1073741824}')"

log "assembling -> $D/model-uc.aff"
"$B" --src "$CK" --codebook-raw "$CB" --imatrix "$IM" \
     --experts-from "$D/experts" --out "$D/model-uc.aff" > "$D/assemble.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then log "ASSEMBLE FAILED rc=$rc"; tail -8 "$D/assemble.log"; exit "$rc"; fi
log "assemble ok: $(tail -1 "$D/assemble.log")"

{
  echo "=== aff-cfgcheck ==="
  /tmp/aff-cfgcheck "$D/model-uc.aff" 2>&1
  echo "=== aff-info (head) ==="
  /mnt/fast-storage/affinity/build/aff-info "$D/model-uc.aff" 2>&1 | head -12
  echo "=== aff-exphash unique (want 15360) ==="
  /tmp/aff-exphash "$D/model-uc.aff" 2>/dev/null | sort -u -k3 | wc -l
  echo "=== aff-fp8check ==="
  /tmp/aff-fp8check "$D/model-uc.aff" "$CK" 2>&1 | tail -3
} > "$D/gates.log" 2>&1
log "gates written -> $D/gates.log"
log "STAGE A READY: $D/model-uc.aff"
