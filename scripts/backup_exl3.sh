#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/home/ydj/affinity-patched/backups/pre-exl3-20261027"
mkdir -p "${BACKUP_DIR}"

# 1. Backup live production binary and quantizer
cp -a /home/ydj/affinity-patched/affinity "${BACKUP_DIR}/affinity.v1.1.0"
if [ -f /mnt/fast-storage/affinity/build/aff-quantize ]; then
  cp -a /mnt/fast-storage/affinity/build/aff-quantize "${BACKUP_DIR}/aff-quantize.v1.1.0"
fi

# 2. Backup systemd user service
cp -a /home/ydj/.config/systemd/user/deepseek-v41-affinity.service \
      "${BACKUP_DIR}/deepseek-v41-affinity.service.v1.1.0"

# 3. Record git commit hashes and create rollback tag in both repos
git -C /mnt/fast-storage/affinity rev-parse HEAD > "${BACKUP_DIR}/affinity_src_head.txt"
git -C /mnt/fast-storage/affinity tag -f pre-exl3-v1.1.0
git -C /home/ydj/deepseek-v4.1-flash-r9700-affinity rev-parse HEAD > "${BACKUP_DIR}/github_repo_head.txt"

# 4. Archive modified/target source files
tar -czf "${BACKUP_DIR}/affinity-src-pre-exl3.tar.gz" \
  -C /mnt/fast-storage/affinity \
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

sha256sum "${BACKUP_DIR}"/* > "${BACKUP_DIR}/SHA256SUMS"
echo "BACKUP COMPLETE: ${BACKUP_DIR}"
cat "${BACKUP_DIR}/SHA256SUMS"
