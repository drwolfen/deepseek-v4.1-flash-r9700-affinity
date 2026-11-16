#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups/pre-exl3}"
AFFINITY_DIR="${AFFINITY_DIR:-$(pwd)}"
PROD_BIN_DIR="${PROD_BIN_DIR:-/opt/affinity/bin}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-${HOME}/.config/systemd/user/deepseek-v41-affinity.service}"
if [ ! -d "${BACKUP_DIR}" ]; then
  echo "ERROR: Backup directory ${BACKUP_DIR} not found!" >&2
  exit 1
fi

echo "[1/4] Restoring source files from pre-exl3-v1.1.0..."
git -C "${AFFINITY_DIR}" checkout pre-exl3-v1.1.0 -- \
  CMakeLists.txt \
  src/quant/hadamard.h \
  src/quant/hadamard.cpp \
  src/gpu/batch_kernels.h \
  src/gpu/batch_kernels.hip \
  src/gpu/expert_kernel_hip.h \
  src/gpu/expert_kernel.hip \
  src/gpu/moe_plan.h \
  src/gpu/moe_plan.hip \
  src/gpu/static_placement.h \
  src/gpu/static_placement.hip \
  tools/aff_quantize.cpp

echo "[2/4] Restoring production binary ${PROD_BIN_DIR}/affinity..."
if [ -f "${BACKUP_DIR}/affinity.v1.1.0" ]; then
  cp -a "${BACKUP_DIR}/affinity.v1.1.0" "${PROD_BIN_DIR}/affinity.tmp"
  mv -f "${PROD_BIN_DIR}/affinity.tmp" "${PROD_BIN_DIR}/affinity"
fi

echo "[3/4] Restoring systemd service..."
if [ -f "${BACKUP_DIR}/deepseek-v41-affinity.service.v1.1.0" ]; then
  cp -a "${BACKUP_DIR}/deepseek-v41-affinity.service.v1.1.0" \
        "${SYSTEMD_SERVICE}"
  systemctl --user daemon-reload
  systemctl --user restart deepseek-v41-affinity.service
fi

echo "[4/4] Waiting for health check on :8000..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ROLLBACK SUCCESSFUL: deepseek-v41-affinity.service is healthy on v1.1.0."
    exit 0
  fi
  sleep 2
done
echo "WARNING: Server still loading slabs; check: journalctl --user -u deepseek-v41-affinity.service -n 30"
