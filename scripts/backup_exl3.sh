#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups/pre-exl3}"
AFFINITY_DIR="${AFFINITY_DIR:-$(pwd)}"
PROD_BIN_DIR="${PROD_BIN_DIR:-/opt/affinity/bin}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-${HOME}/.config/systemd/user/deepseek-v41-affinity.service}"
GITHUB_REPO_DIR="${GITHUB_REPO_DIR:-.}"
mkdir -p "${BACKUP_DIR}"

# 1. Backup live production binary and quantizer
if [ -f "${PROD_BIN_DIR}/affinity" ]; then
  cp -a "${PROD_BIN_DIR}/affinity" "${BACKUP_DIR}/affinity.v1.1.0"
fi
if [ -f "${AFFINITY_DIR}/build/aff-quantize" ]; then
  cp -a "${AFFINITY_DIR}/build/aff-quantize" "${BACKUP_DIR}/aff-quantize.v1.1.0"
fi

# 2. Backup systemd user service
if [ -f "${SYSTEMD_SERVICE}" ]; then
  cp -a "${SYSTEMD_SERVICE}" \
        "${BACKUP_DIR}/deepseek-v41-affinity.service.v1.1.0"
fi

# 3. Record git commit hashes and create rollback tag in both repos
git -C "${AFFINITY_DIR}" rev-parse HEAD > "${BACKUP_DIR}/affinity_src_head.txt"
git -C "${AFFINITY_DIR}" tag -f pre-exl3-v1.1.0
if [ -d "${GITHUB_REPO_DIR}/.git" ]; then
  git -C "${GITHUB_REPO_DIR}" rev-parse HEAD > "${BACKUP_DIR}/github_repo_head.txt"
fi

# 4. Archive modified/target source files
tar -czf "${BACKUP_DIR}/affinity-src-pre-exl3.tar.gz" \
  -C "${AFFINITY_DIR}" \
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
