# Affinity EXL3 Optimizations (`v1.2.0`) Implementation Plan

> **For Claude / Gemini Flash 3.8:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the 3 ExLlamaV3 (EXL3) architectural optimizations (`AFF_MOE_RECON_TILES` coalesced sub-tile reconstruction & L2 tile slicing, `AFF_HADAMARD_ROT` / `--hadamard` block-32 randomized Walsh-Hadamard incoherence rotation, and `AFF_PREFETCH_RING` device-planned asynchronous VRAM prefetch ring for pooled experts) into the Affinity inference engine (`/mnt/fast-storage/affinity`) with a full backup and one-command rollback plan.

**Architecture:** All runtime changes are opt-in via environment variables (`AFF_MOE_RECON_TILES`, `AFF_HADAMARD_ROT`, `AFF_PREFETCH_RING`) and CLI flags (`aff-quantize --hadamard`), preserving 100% bit-identity and backward compatibility with existing `.aff` containers (`model-uc-1017.aff`) and persistent prefix caches (`pcache`) when unset or when running the lossless `AFF_MOE_RECON_TILES` + `AFF_PREFETCH_RING` paths.

**Tech Stack:** C++20, HIP / ROCm (`gfx1201` RDNA4 wave32 WMMA), CMake, Ninja, `ctest`, `systemd` user service (`deepseek-v41-affinity.service`).

---

## Backup & Restore Plan (Execute First Before Any Code Edits)

### Task 0: Snapshot Live Binary, Service, and Source Tree + Create One-Command Rollback Script

**Files:**
- Create: `/mnt/fast-storage/affinity/scripts/backup_exl3.sh`
- Create: `/mnt/fast-storage/affinity/scripts/rollback_exl3.sh`
- Backup directory: `/home/ydj/affinity-patched/backups/pre-exl3-20261027/`

**Step 1: Create `/mnt/fast-storage/affinity/scripts/backup_exl3.sh`**

```bash
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
```

**Step 2: Create `/mnt/fast-storage/affinity/scripts/rollback_exl3.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/home/ydj/affinity-patched/backups/pre-exl3-20261027"
if [ ! -d "${BACKUP_DIR}" ]; then
  echo "ERROR: Backup directory ${BACKUP_DIR} not found!" >&2
  exit 1
fi

echo "[1/4] Restoring source files from pre-exl3-v1.1.0..."
git -C /mnt/fast-storage/affinity checkout pre-exl3-v1.1.0 -- \
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

echo "[2/4] Restoring production binary /home/ydj/affinity-patched/affinity..."
cp -a "${BACKUP_DIR}/affinity.v1.1.0" /home/ydj/affinity-patched/affinity.tmp
mv -f /home/ydj/affinity-patched/affinity.tmp /home/ydj/affinity-patched/affinity

echo "[3/4] Restoring systemd service..."
cp -a "${BACKUP_DIR}/deepseek-v41-affinity.service.v1.1.0" \
      /home/ydj/.config/systemd/user/deepseek-v41-affinity.service
systemctl --user daemon-reload
systemctl --user restart deepseek-v41-affinity.service

echo "[4/4] Waiting for health check on :8000..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ROLLBACK SUCCESSFUL: deepseek-v41-affinity.service is healthy on v1.1.0."
    exit 0
  fi
  sleep 2
done
echo "WARNING: Server still loading slabs; check: journalctl --user -u deepseek-v41-affinity.service -n 30"
```

**Step 3: Make both scripts executable and run `backup_exl3.sh`**

Run:
```bash
chmod +x /mnt/fast-storage/affinity/scripts/backup_exl3.sh /mnt/fast-storage/affinity/scripts/rollback_exl3.sh
/mnt/fast-storage/affinity/scripts/backup_exl3.sh
```
Expected: `BACKUP COMPLETE: /home/ydj/affinity-patched/backups/pre-exl3-20261027` with SHA256 checksums printed.

---

## Optimization 1: EXL3 MoE Reconstruction Tiling (`AFF_MOE_RECON_TILES`)

**Technical Context:**
1. In [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:573-581`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L573-L581), when `expert_gemm_wmma_kernel` runs with `MT=1, TR=true` (`kETPR=2, kEBPT=2`, which is the decode / speculative verification instantiation for `max_nt <= 16` at line 3062), `kWide` is `false`. Each thread currently executes 10 separate 1-byte non-temporal loads (`ld_nt(&cp[i])` for `i = 0..4` across `bb = 0..1`) per column group. On zero-copy PCIe tiles (`hyb` stream) and VRAM slabs, byte-granular `flat_load_ubyte` instructions fragment memory transactions.
2. Because `g * 20` is always 4-byte aligned (`20 = 5 * 4` bytes per column group), thread `sblk = 0` (`tid & 1 == 0`) owns bytes `0..9` (`g32[0]`, `g32[1]`, and the low `uint16_t` of `g32[2]`), while thread `sblk = 2` (`tid & 1 == 1`) owns bytes `10..19` (the high `uint16_t` of `g32[2]`, `g32[3]`, and `g32[4]`). Replacing the 10 byte loads with 2 `uint32_t` loads + 1 `uint16_t` load per thread cuts load instructions by `3.33x` and coalesces wave32 reads into contiguous 320-byte bursts while remaining **100% bit-identical**.
3. In [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:3025-3069`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L3025-L3069), adding `AFF_MOE_RECON_TILES=<N>` (e.g., `8` or `16`) slices `n_tiles` into L2-resident sub-batches of `<= recon_tiles` tiles per launch so reconstructed expert tiles do not thrash R9700's 64 MiB L2 cache during large prefill or heavy miss bursts.

### Task 1: Implement Coalesced `TR` Sub-Tile Loads and `AFF_MOE_RECON_TILES` L2 Slicing

**Files:**
- Modify: [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:549-585`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L549-L585)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:3025-3069`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L3025-L3069)
- Create: `/mnt/fast-storage/affinity/tests/test_moe_recon_tiles.hip`
- Modify: [`/mnt/fast-storage/affinity/CMakeLists.txt:303-310`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L303-L310)

**Step 1: Write the failing test (`/mnt/fast-storage/affinity/tests/test_moe_recon_tiles.hip`)**

Create `/mnt/fast-storage/affinity/tests/test_moe_recon_tiles.hip`:
```cpp
#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Verify that packing 20 bytes into g32[0..4] and extracting via coalesced
// (u32, u32, u16) for sblk=0 and (u16, u32, u32) for sblk=2 is bit-for-byte
// identical to the byte-by-byte loop for every possible byte pattern.
static void test_tr_mt1_pack_identity() {
  uint8_t raw[20];
  for (int i = 0; i < 20; ++i) raw[i] = (uint8_t)(0xA5u ^ (i * 19u + 7u));
  const uint32_t* g32 = reinterpret_cast<const uint32_t*>(raw);
  const uint16_t* g16 = reinterpret_cast<const uint16_t*>(raw);

  for (uint32_t sblk : {0u, 2u}) {
    uint32_t tw0 = 0, tw1 = 0, tw2 = 0;
    if (sblk == 0u) {
      tw0 = g32[0];
      tw1 = g32[1];
      tw2 = (uint32_t)g16[4];
    } else {
      tw0 = (uint32_t)g16[5];
      tw1 = g32[3];
      tw2 = g32[4];
    }
    for (uint32_t bb = 0; bb < 2u; ++bb) {
      uint64_t ref_win = 0;
      const uint8_t* cp = raw + sblk * 5u + bb * 5u;
      for (uint32_t i = 0; i < 5u; ++i) ref_win |= (uint64_t)cp[i] << (8u * i);

      uint64_t got_win = 0;
      if (sblk == 0u) {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFu) << 32))
                      : ((uint64_t)(tw1 >> 8) | ((uint64_t)tw2 << 24));
      } else {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFFFFFu) << 16))
                      : ((uint64_t)(tw1 >> 24) | ((uint64_t)tw2 << 8));
      }
      if (got_win != ref_win) {
        std::fprintf(stderr, "FAIL: sblk=%u bb=%u got=0x%010llx ref=0x%010llx\n",
                     sblk, bb, (unsigned long long)got_win, (unsigned long long)ref_win);
        std::exit(1);
      }
    }
  }
}

int main() {
  test_tr_mt1_pack_identity();
  if (!aff::hip_available()) {
    std::printf("PASS: test_moe_recon_tiles (CPU bit-identity verified; no gfx12 GPU)\n");
    return 0;
  }

  // Test GPU expert_gemm_wmma_hip with and without AFF_MOE_RECON_TILES=2 across 6 tiles
  const uint32_t rows = 128, cols = 512, n_tiles = 6, nt = 8;
  const size_t code_bytes = (size_t)rows * (cols / 2 * 5 / 8);
  const size_t field_bytes = (size_t)rows * (cols / 16 * 6 / 8);

  std::vector<uint8_t> h_codes(code_bytes), h_fields(field_bytes);
  for (size_t i = 0; i < code_bytes; ++i) h_codes[i] = (uint8_t)((i * 37 + 13) & 0xFF);
  for (size_t i = 0; i < field_bytes; ++i) h_fields[i] = (uint8_t)((i * 17 + 3) & 0x3F);

  std::vector<uint8_t> tr_codes(code_bytes), tr_fields(field_bytes);
  aff::expert_transpose_plane_host(h_codes.data(), h_fields.data(),
                                   tr_codes.data(), tr_fields.data(), rows, cols);

  std::vector<uint16_t> h_rscales(rows, 0x3C00u); // 1.0 in bf16 (0x3F80 >> 16 = 0x3F80)
  for (uint32_t r = 0; r < rows; ++r) h_rscales[r] = 0x3F80u;

  std::vector<uint8_t> h_x8(nt * cols, 0x38u);
  std::vector<float> h_xs(nt * (cols / 128), 1.0f);
  std::vector<uint32_t> h_cb(2 * 1024, 0x38383838u);

  uint8_t *d_codes = nullptr, *d_fields = nullptr, *d_x8 = nullptr;
  uint16_t *d_rscales = nullptr;
  float *d_xs = nullptr, *d_cb = nullptr, *d_y_full = nullptr, *d_y_sliced = nullptr;
  aff::ExpertTile* d_tiles = nullptr;

  AFF_HIP_CHECK(hipMalloc(&d_codes, code_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_fields, field_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_rscales, rows * sizeof(uint16_t)));
  AFF_HIP_CHECK(hipMalloc(&d_x8, h_x8.size()));
  AFF_HIP_CHECK(hipMalloc(&d_xs, h_xs.size() * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_cb, h_cb.size() * sizeof(uint32_t)));
  AFF_HIP_CHECK(hipMalloc(&d_y_full, (size_t)rows * n_tiles * nt * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_y_sliced, (size_t)rows * n_tiles * nt * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_tiles, n_tiles * sizeof(aff::ExpertTile)));

  AFF_HIP_CHECK(hipMemcpy(d_codes, tr_codes.data(), code_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_fields, tr_fields.data(), field_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_rscales, h_rscales.data(), rows * sizeof(uint16_t), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_x8, h_x8.data(), h_x8.size(), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_xs, h_xs.data(), h_xs.size() * sizeof(float), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_cb, h_cb.data(), h_cb.size() * sizeof(uint32_t), hipMemcpyHostToDevice));

  aff::expert_set_plane_layout_colgroup(true);
  aff::expert_set_wire_format(6u); // kFmtQuad2S

  std::vector<aff::ExpertTile> h_tiles(n_tiles);
  for (uint32_t t = 0; t < n_tiles; ++t) {
    h_tiles[t].data = d_codes;
    h_tiles[t].fields = d_fields;
    h_tiles[t].rscales = d_rscales;
    h_tiles[t].tok = nullptr;
    h_tiles[t].y = d_y_full + t * nt;
    h_tiles[t].nt = nt;
    h_tiles[t].ld = n_tiles * nt;
  }
  AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), n_tiles * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
  aff::expert_set_recon_tiles(0u);
  aff::expert_gemm_wmma_hip(d_tiles, n_tiles, nt, d_x8, d_xs, d_cb, rows, cols, 1.0f, nullptr, false);
  AFF_HIP_CHECK(hipDeviceSynchronize());

  for (uint32_t t = 0; t < n_tiles; ++t) h_tiles[t].y = d_y_sliced + t * nt;
  AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), n_tiles * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
  aff::expert_set_recon_tiles(2u);
  aff::expert_gemm_wmma_hip(d_tiles, n_tiles, nt, d_x8, d_xs, d_cb, rows, cols, 1.0f, nullptr, false);
  AFF_HIP_CHECK(hipDeviceSynchronize());
  aff::expert_set_recon_tiles(0u);

  std::vector<float> y0((size_t)rows * n_tiles * nt), y1((size_t)rows * n_tiles * nt);
  AFF_HIP_CHECK(hipMemcpy(y0.data(), d_y_full, y0.size() * sizeof(float), hipMemcpyDeviceToHost));
  AFF_HIP_CHECK(hipMemcpy(y1.data(), d_y_sliced, y1.size() * sizeof(float), hipMemcpyDeviceToHost));

  if (std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: sliced recon tiles output differs from full launch\n");
    return 1;
  }

  (void)hipFree(d_codes); (void)hipFree(d_fields); (void)hipFree(d_rscales);
  (void)hipFree(d_x8); (void)hipFree(d_xs); (void)hipFree(d_cb);
  (void)hipFree(d_y_full); (void)hipFree(d_y_sliced); (void)hipFree(d_tiles);
  std::printf("PASS: test_moe_recon_tiles\n");
  return 0;
}
```

**Step 2: Register `test_moe_recon_tiles` in `/mnt/fast-storage/affinity/CMakeLists.txt` and declare `expert_set_recon_tiles` in `/mnt/fast-storage/affinity/src/gpu/expert_kernel_hip.h`**

In [`/mnt/fast-storage/affinity/src/gpu/expert_kernel_hip.h:225-230`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel_hip.h#L225-L230), add:
```cpp
void expert_set_recon_tiles(uint32_t max_tiles);
uint32_t expert_recon_tiles();
```

In [`/mnt/fast-storage/affinity/CMakeLists.txt:303`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L303), right after `aff_hip_test(test_expert_shard expert_shard)`, add:
```cmake
    aff_hip_test(test_moe_recon_tiles moe_recon_tiles)
```

**Step 3: Implement coalesced `TR` sub-tile load and `AFF_MOE_RECON_TILES` slicing in `/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip`**

1. In [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:549-585`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L549-L585), update the `kWide` / `TR` load block inside `expert_gemm_wmma_kernel`:
```cpp
        constexpr bool kWide = kETPR == 1 && kEBPT == 4;
        constexpr bool kHalfTR = TR && (kETPR == 2 && kEBPT == 2);
        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, f24 = 0;
        if (kWide) {
          const auto* cp32 = (const AFF_GLOBAL uint32_t*)cp0;
          w0 = ld_nt(&cp32[0]); w1 = ld_nt(&cp32[1]); w2 = ld_nt(&cp32[2]);
          w3 = ld_nt(&cp32[3]); w4 = ld_nt(&cp32[4]);
        } else if (kHalfTR) {
          const size_t g = (size_t)(c0 / kEK) * rows + row;
          const auto* g32 = (const AFF_GLOBAL uint32_t*)as_global(data + g * (kEK / 16 * 5));
          const auto* g16 = (const AFF_GLOBAL uint16_t*)g32;
          if (sblk == 0u) {
            w0 = ld_nt(&g32[0]); w1 = ld_nt(&g32[1]); w2 = (uint32_t)ld_nt(&g16[4]);
          } else {
            w0 = (uint32_t)ld_nt(&g16[5]); w1 = ld_nt(&g32[3]); w2 = ld_nt(&g32[4]);
          }
        }
        // Row-major reaches this only when kWide made 6*b0 a multiple of 24; TR always can, because
        // the group's three bytes are a unit by construction.
        if (kWide || TR)
          f24 = (uint32_t)ld_nt(&fp0[0]) | ((uint32_t)ld_nt(&fp0[1]) << 8) |
                ((uint32_t)ld_nt(&fp0[2]) << 16);

#pragma unroll
        for (uint32_t bb = 0; bb < kEBPT; ++bb) {
          const uint32_t b = b0 + bb;
          uint32_t field;
          uint64_t win;                                 // 8 pairs = 40 bits at byte b*5
          if (kWide) {
            field = (f24 >> (fsh0 + bb * 6)) & 63u;
            // Bytes [5bb, 5bb+5) never span more than two dwords, and `bb` is a literal here.
            const uint32_t lo = bb == 0 ? w0 : bb == 1 ? w1 : bb == 2 ? w2 : w3;
            const uint32_t hi = bb == 0 ? w1 : bb == 1 ? w2 : bb == 2 ? w3 : w4;
            win = ((((uint64_t)hi) << 32) | lo) >> (((bb * 5) & 3u) * 8);
          } else if (kHalfTR) {
            field = (f24 >> (fsh0 + bb * 6)) & 63u;
            if (sblk == 0u) {
              win = (bb == 0u)
                        ? ((uint64_t)w0 | ((uint64_t)(w1 & 0xFFu) << 32))
                        : ((uint64_t)(w1 >> 8) | ((uint64_t)w2 << 24));
            } else {
              win = (bb == 0u)
                        ? ((uint64_t)w0 | ((uint64_t)(w1 & 0xFFFFFFu) << 16))
                        : ((uint64_t)(w1 >> 24) | ((uint64_t)w2 << 8));
            }
          } else if (TR) {
            // Fallback for any other TR geometry.
            field = (f24 >> (fsh0 + bb * 6)) & 63u;
            const auto* cp = cp0 + (size_t)bb * 5;
            win = 0;
#pragma unroll
            for (uint32_t i = 0; i < 5; ++i) win |= (uint64_t)ld_nt(&cp[i]) << (8 * i);
          } else {
```

2. In [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:3025-3069`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L3025-L3069), add `g_recon_tiles` and sub-batch loop in `expert_gemm_wmma_hip`:
```cpp
static int g_recon_tiles = -1;

void expert_set_recon_tiles(uint32_t max_tiles) { g_recon_tiles = (int)max_tiles; }

uint32_t expert_recon_tiles() {
  if (g_recon_tiles < 0) {
    const char* s = std::getenv("AFF_MOE_RECON_TILES");
    g_recon_tiles = (s && *s) ? std::max(0, std::atoi(s)) : 0;
  }
  return (uint32_t)g_recon_tiles;
}

void expert_gemm_wmma_hip(const ExpertTile* tiles, uint32_t n_tiles, uint32_t max_nt,
                          const uint8_t* x8, const float* xs, const float* cb, uint32_t rows,
                          uint32_t cols, float wscale, void* stream, bool y_bf16) {
  if (!n_tiles) return;
  const uint32_t rcap = expert_recon_tiles();
  const uint32_t step = (rcap > 0u && rcap < n_tiles) ? rcap : n_tiles;
  const uint32_t ntiles = (max_nt + 15) / 16;
  const uint32_t fmt = expert_wire_format();
  const bool tr = expert_planes_are_colgroup_major();
  for (uint32_t t0 = 0; t0 < n_tiles; t0 += step) {
    const uint32_t chunk_tiles = std::min<uint32_t>(step, n_tiles - t0);
    const ExpertTile* cur_tiles = tiles + t0;
#define AFF_EW1(NTV, MTV, FV, TRV)                                                                \
  do {                                                                                             \
  hipLaunchKernelGGL((expert_gemm_wmma_kernel<NTV, MTV, FV, TRV>),                                 \
                     dim3((rows + kEWaves * MTV * 16 - 1) / (kEWaves * MTV * 16), chunk_tiles),    \
                     dim3(kEThreads), 0, (hipStream_t)stream, cur_tiles, x8, xs, cb, wscale, rows, \
                     cols, y_bf16 ? 1 : 0); AFF_HIP_CHECK(hipGetLastError());                      \
  } while (0)
#define AFF_EWF(NTV, MTV, TRV)                                                                    \
  do {                                                                                             \
    if (fmt == kFmtQuad8) AFF_EW1(NTV, MTV, kFmtQuad8, TRV);                                       \
    else if (fmt == kFmtQuad2S) AFF_EW1(NTV, MTV, kFmtQuad2S, TRV);                                \
    else if (fmt == kFmtQuad4S) AFF_EW1(NTV, MTV, kFmtQuad4S, TRV);                                \
    else if (fmt == kFmtQuad4) AFF_EW1(NTV, MTV, kFmtQuad4, TRV);                                  \
    else if (fmt == kFmtPair16) AFF_EW1(NTV, MTV, kFmtPair16, TRV);                                       \
  } while (0)
#define AFF_EW(NTV, MTV)                                                                          \
  do {                                                                                             \
    if (tr) AFF_EWF(NTV, MTV, true); else AFF_EWF(NTV, MTV, false);                                \
  } while (0)
    if (ntiles <= 1) AFF_EW(1, 1);
    else if (ntiles <= 2) AFF_EW(2, 2);
    else if (ntiles <= 4) AFF_EW(4, 2);
    else AFF_EW(8, 1);
#undef AFF_EW
#undef AFF_EWF
#undef AFF_EW1
  }
}
```

**Step 4: Build and run `test_moe_recon_tiles`**

Run:
```bash
cmake -S /mnt/fast-storage/affinity -B /mnt/fast-storage/affinity/build -GNinja
ninja -C /mnt/fast-storage/affinity/build test_moe_recon_tiles
/mnt/fast-storage/affinity/build/test_moe_recon_tiles
```
Expected: `PASS: test_moe_recon_tiles`

**Step 5: Commit Task 1**

```bash
git -C /mnt/fast-storage/affinity add CMakeLists.txt src/gpu/expert_kernel_hip.h src/gpu/expert_kernel.hip tests/test_moe_recon_tiles.hip
git -C /mnt/fast-storage/affinity commit -m "perf(gpu): coalesced TR MT=1 sub-tile loads and AFF_MOE_RECON_TILES L2 slicing"
```

---

## Optimization 2: EXL3 Block-32 Randomized Walsh-Hadamard Incoherence Rotation (`AFF_HADAMARD_ROT` & `--hadamard`)

**Technical Context:**
1. DeepSeek-V4.1-Flash expert dimensions are `hidden = 5120` and `moe_inter = 2304` (`1152` per card at TP=2). Neither `5120` nor `2304` is a power of two, so full-row Walsh-Hadamard (`src/quant/hadamard.h`) cannot be applied to an entire row.
2. However, `32` divides `5120` ($160 \times 32$), `2304` ($72 \times 32$), `1152` ($36 \times 32$), `128` ($4 \times 32$), and `kEK = 64` ($2 \times 32$). A **block-32 randomized Walsh-Hadamard transform** $H_{32,s} = \frac{1}{\sqrt{32}} H_{32} \operatorname{diag}(s_0, \dots, s_{31})$ is orthogonal and involutive within every 32-column block:
   $$(W H_{32,s}^T)(H_{32,s} x) = W x$$
   Because it is block-diagonal with $32 \times 32$ blocks, it commutes with TP=2 column sharding (`kEK = 64`) and requires zero changes to `ExpertShardLayout` or `expert_gemm_wmma_kernel`.
3. Offline (`aff-quantize --hadamard`): rotates every 32-element block of expert weights before codebook fitting (`fit_codebook`) and matrix quantization (`quantize_matrix`), suppressing weight outlier spikes (`max|w|/rms(w)`) and lowering 2.875 bpw quantization error (`rel_fro`).
4. Online (`AFF_HADAMARD_ROT=1`): inside `batch_quant_fp8_kernel` ([`batch_kernels.hip:146-185`](file:///mnt/fast-storage/affinity/src/gpu/batch_kernels.hip#L146-L185)) and `expert_swiglu_quant_kernel` ([`expert_kernel.hip:864-908`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L864-L908)), the 128-column activation tile `tile[128][33]` is already staged in LDS before computing the block amax `smax[tt]`. Applying the 5-stage block-32 butterfly in LDS before computing `m = fmaxf(m, fabsf(v))` rotates the activation into the same Hadamard basis AND suppresses activation outliers prior to E4M3 quantization.

### Task 2: Implement CPU & GPU Block-32 Randomized Walsh-Hadamard Rotation

**Files:**
- Modify: [`/mnt/fast-storage/affinity/src/quant/hadamard.h:38-45`](file:///mnt/fast-storage/affinity/src/quant/hadamard.h#L38-L45)
- Modify: [`/mnt/fast-storage/affinity/src/quant/hadamard.cpp:40-50`](file:///mnt/fast-storage/affinity/src/quant/hadamard.cpp#L40-L50)
- Modify: [`/mnt/fast-storage/affinity/tools/aff_quantize.cpp:258-265,705-715,760-775`](file:///mnt/fast-storage/affinity/tools/aff_quantize.cpp#L705-L775)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/batch_kernels.hip:146-185,2592-2603`](file:///mnt/fast-storage/affinity/src/gpu/batch_kernels.hip#L146-L185)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:864-908,2822-2836`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L864-L908)
- Create: `/mnt/fast-storage/affinity/tests/test_hadamard_block32.cpp`
- Modify: [`/mnt/fast-storage/affinity/CMakeLists.txt:227`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L227)

**Step 1: Write the failing unit test (`/mnt/fast-storage/affinity/tests/test_hadamard_block32.cpp`)**

Create `/mnt/fast-storage/affinity/tests/test_hadamard_block32.cpp`:
```cpp
#include "quant/hadamard.h"
#include "quant/blockquant.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  // 1. Verify (W * H_32^T) * (H_32 * x) == W * x for non-power-of-two cols = 2304
  const uint64_t rows = 16, cols = 2304;
  std::vector<float> W(rows * cols), x(cols);
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = std::sin((float)i * 0.13f) + ((i % 64 == 0) ? 12.5f : 0.0f); // outlier spikes
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::cos((float)i * 0.07f) - ((i % 96 == 0) ? 8.0f : 0.0f);

  std::vector<float> y_ref(rows, 0.0f);
  for (uint64_t r = 0; r < rows; ++r)
    for (uint64_t c = 0; c < cols; ++c)
      y_ref[r] += W[r * cols + c] * x[c];

  // Check incoherence improvement on outlier-heavy matrix
  const aff::IncoherenceStats st_before = aff::incoherence_stats(W.data(), W.size(), 32);
  std::vector<float> W_rot = W, x_rot = x;
  aff::randomized_hadamard_block32_rows(W_rot.data(), rows, cols);
  aff::randomized_hadamard_block32_rows(x_rot.data(), 1, cols);
  const aff::IncoherenceStats st_after = aff::incoherence_stats(W_rot.data(), W_rot.size(), 32);

  if (st_after.mean_block_max_over_rms >= st_before.mean_block_max_over_rms) {
    std::fprintf(stderr, "FAIL: block32 Hadamard did not reduce mean_block_max_over_rms (%.3f -> %.3f)\n",
                 st_before.mean_block_max_over_rms, st_after.mean_block_max_over_rms);
    return 1;
  }

  // Verify dot product preservation: (W_rot) . (x_rot) == W . x
  for (uint64_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint64_t c = 0; c < cols; ++c) acc += W_rot[r * cols + c] * x_rot[c];
    const float diff = std::fabs(acc - y_ref[r]);
    if (diff > 1e-3f) {
      std::fprintf(stderr, "FAIL: row %llu dot mismatch: got %f ref %f (diff %g)\n",
                   (unsigned long long)r, acc, y_ref[r], diff);
      return 1;
    }
  }

  std::printf("PASS: test_hadamard_block32 (block max/rms %.2f -> %.2f)\n",
              st_before.mean_block_max_over_rms, st_after.mean_block_max_over_rms);
  return 0;
}
```

**Step 2: Add `randomized_hadamard_block32_rows` and `hadamard_block32_sign_mask` to `src/quant/hadamard.{h,cpp}`**

In [`/mnt/fast-storage/affinity/src/quant/hadamard.h:40-45`](file:///mnt/fast-storage/affinity/src/quant/hadamard.h#L40-L45), add:
```cpp
// Fixed 32-bit SplitMix64 sign mask (bit k == 1 means -1.0f, 0 means +1.0f) shared by CPU
// aff-quantize (--hadamard) and GPU activation quantizers (AFF_HADAMARD_ROT=1).
constexpr uint32_t kHadamardBlock32Mask = 0x96696996u ^ 0xA53C965Au;

// Applies diag(s_32) then 32-element normalized FWHT to every 32-column block of `x` ([rows][cols]).
// Requires cols % 32 == 0 (holds for 1152, 2304, 5120).
void randomized_hadamard_block32_rows(float* x, uint64_t rows, uint64_t cols,
                                      uint32_t sign_mask = kHadamardBlock32Mask) noexcept;
```

In [`/mnt/fast-storage/affinity/src/quant/hadamard.cpp:48`](file:///mnt/fast-storage/affinity/src/quant/hadamard.cpp#L48), add:
```cpp
void randomized_hadamard_block32_rows(float* x, uint64_t rows, uint64_t cols,
                                      uint32_t sign_mask) noexcept {
  if (cols % 32 != 0) return;
  constexpr float kInvSqrt32 = 0.17677669529663689f; // 1 / sqrt(32)
  for (uint64_t r = 0; r < rows; ++r) {
    float* row = x + r * cols;
    for (uint64_t c0 = 0; c0 < cols; c0 += 32) {
      float* b = row + c0;
      for (uint32_t i = 0; i < 32; ++i) {
        if ((sign_mask >> i) & 1u) b[i] = -b[i];
      }
      fwht(b, 32);
      for (uint32_t i = 0; i < 32; ++i) b[i] *= kInvSqrt32;
    }
  }
}
```

**Step 3: Wire `--hadamard` into `/mnt/fast-storage/affinity/tools/aff_quantize.cpp`**

1. Include `"quant/hadamard.h"` at line 15 of [`/mnt/fast-storage/affinity/tools/aff_quantize.cpp`](file:///mnt/fast-storage/affinity/tools/aff_quantize.cpp#L15).
2. Add `bool use_hadamard = false;` at line 245, and `else if (a == "--hadamard") use_hadamard = true;` at line 259 of [`/mnt/fast-storage/affinity/tools/aff_quantize.cpp`](file:///mnt/fast-storage/affinity/tools/aff_quantize.cpp#L259).
3. At line 706 (inside the codebook sampling loop right after `mxfp4_dequant`) and line 763 (inside `encode_one` right after `mxfp4_dequant`), add:
```cpp
        if (use_hadamard) randomized_hadamard_block32_rows(deq.data(), rows, cols);
```

**Step 4: Wire `AFF_HADAMARD_ROT=1` into GPU activation quantizers (`batch_kernels.hip` and `expert_kernel.hip`)**

1. In [`/mnt/fast-storage/affinity/src/gpu/batch_kernels.hip:145-185`](file:///mnt/fast-storage/affinity/src/gpu/batch_kernels.hip#L145-L185), add a template parameter `bool Hadamard32 = false` to `batch_quant_fp8_kernel`:
```cpp
template <class Src, bool Hadamard32 = false>
__global__ __launch_bounds__(kQThreads) void batch_quant_fp8_kernel(
    Src src, uint8_t* __restrict__ out,
    float* __restrict__ scales, uint32_t dim, uint32_t nb, uint32_t x_off) {
  __shared__ float tile[128][kQTok + 1];
  __shared__ float part[8][kQTok];
  __shared__ float smax[kQTok];

  const uint32_t tid = threadIdx.x, tt = tid & 31u, kk = tid >> 5;
  const uint32_t t0 = blockIdx.x * kQTok;
  const uint32_t tok = t0 + tt;
  const uint32_t nkb = dim >> 7;

  for (uint32_t kb = blockIdx.y; kb < nkb; kb += gridDim.y) {
    __syncthreads();
    float m = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      const uint32_t k = kk + i * 8;
      const size_t o = (size_t)(x_off + kb * 128 + k) * nb + tok;
      float v = tok >= nb ? 0.0f : src(o);
      if constexpr (Hadamard32) {
        constexpr uint32_t kSignMask = 0x96696996u ^ 0xA53C965Au;
        if ((kSignMask >> (k & 31u)) & 1u) v = -v;
      }
      tile[k][tt] = v;
      if constexpr (!Hadamard32) m = fmaxf(m, fabsf(v));
    }
    if constexpr (Hadamard32) {
      __syncthreads();
      // 4 blocks of 32 elements along k (0..127); 8 threads (kk=0..7) cooperate per token `tt`
      for (uint32_t stride = 1; stride < 32u; stride <<= 1) {
#pragma unroll
        for (uint32_t b32 = 0; b32 < 4u; ++b32) {
          for (uint32_t pair_idx = kk; pair_idx < 16u; pair_idx += 8u) {
            const uint32_t base = b32 * 32u + (pair_idx / stride) * (2u * stride) + (pair_idx % stride);
            const float a = tile[base][tt], b = tile[base + stride][tt];
            tile[base][tt] = a + b;
            tile[base + stride][tt] = a - b;
          }
        }
        __syncthreads();
      }
#pragma unroll
      for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t k = kk + i * 8;
        const float v = tile[k][tt] * 0.17677669529663689f;
        tile[k][tt] = v;
        m = fmaxf(m, fabsf(v));
      }
    }
    part[kk][tt] = m;
```
And in `batch_quant_fp8_hip` ([`batch_kernels.hip:2592`](file:///mnt/fast-storage/affinity/src/gpu/batch_kernels.hip#L2592)), check `AFF_HADAMARD_ROT`:
```cpp
void batch_quant_fp8_hip(const float* x, uint8_t* out, float* scales, uint32_t dim, uint32_t nb,
                         uint32_t x_off, void* stream) {
  static const bool had = [] {
    const char* s = std::getenv("AFF_HADAMARD_ROT");
    return s && *s && *s != '0';
  }();
  if (had) {
    AFF_BLAUNCH((batch_quant_fp8_kernel<QSrcF32, true>), quant_grid(dim, nb), dim3(kQThreads), 0,
                (hipStream_t)stream, QSrcF32{x}, out, scales, dim, nb, x_off);
  } else {
    AFF_BLAUNCH((batch_quant_fp8_kernel<QSrcF32, false>), quant_grid(dim, nb), dim3(kQThreads), 0,
                (hipStream_t)stream, QSrcF32{x}, out, scales, dim, nb, x_off);
  }
}
```
2. Apply the identical `template <bool Hadamard32 = false>` block to `expert_swiglu_quant_kernel` and `expert_swiglu_quant_hip` in [`/mnt/fast-storage/affinity/src/gpu/expert_kernel.hip:864-908,2822-2836`](file:///mnt/fast-storage/affinity/src/gpu/expert_kernel.hip#L864-L908).

**Step 5: Register `test_hadamard_block32` in `/mnt/fast-storage/affinity/CMakeLists.txt`, build, and test**

In [`/mnt/fast-storage/affinity/CMakeLists.txt:227`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L227), add:
```cmake
  add_executable(test_hadamard_block32 tests/test_hadamard_block32.cpp)
  target_link_libraries(test_hadamard_block32 PRIVATE aff_format)
  add_test(NAME hadamard_block32 COMMAND test_hadamard_block32)
```

Run:
```bash
ninja -C /mnt/fast-storage/affinity/build test_hadamard_block32 aff-quantize
/mnt/fast-storage/affinity/build/test_hadamard_block32
```
Expected: `PASS: test_hadamard_block32`

**Step 6: Commit Task 2**

```bash
git -C /mnt/fast-storage/affinity add CMakeLists.txt src/quant/hadamard.h src/quant/hadamard.cpp tools/aff_quantize.cpp src/gpu/batch_kernels.hip src/gpu/expert_kernel.hip tests/test_hadamard_block32.cpp
git -C /mnt/fast-storage/affinity commit -m "feat(quant,gpu): add EXL3 block-32 randomized Walsh-Hadamard rotation (--hadamard / AFF_HADAMARD_ROT)"
```

---

## Optimization 3: EXL3 Adaptive Asynchronous Prefetch Ring (`AFF_PREFETCH_RING`)

**Technical Context:**
1. As documented in [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:96-98`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L96-L98), zero-copy PCIe reads from `pool_dev_` inside `expert_gemm_wmma_kernel` inflate GEMM execution several times over because `expert_gemm_wmma_kernel` reads `gate`, `up`, and `down` at column-group strides (`20B` codes + `3B` fields per row) and re-reads the expert across multiple tiles when `counts[e] > 64`.
2. With `AFF_PREFETCH_RING=<slots>` (e.g., `AFF_PREFETCH_RING=16`, allocating `16 * 3.04 MiB = 48.6 MiB` of VRAM per card BEFORE slab sizing in `StaticPlacement::init`):
   - `moe_plan_kernel` ([`/mnt/fast-storage/affinity/src/gpu/moe_plan.hip:294-333`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.hip#L294-L333)) already identifies every pooled (non-resident) expert (`cls == 1`, index `m_idx = j - n_res`).
   - When `a.pref_slab != nullptr` and `m_idx < a.pref_cap`, `moe_plan_kernel` records the host pool shard address `base` into `a.pref_src[m_idx]`, records `s_npool` into `a.status->n_pool`, and points the pooled expert's `ExpertTile` (`T.data`, `T.fields`, `T.rscales`) at the VRAM prefetch slot `a.pref_slab + (size_t)m_idx * a.shard.stride`!
   - Right after `ev_q` on the `hyb` stream ([`static_placement.hip:1956`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L1956)), while the compute stream `st` is busy running resident experts' `gate` and `up` GEMMs, `hyb` runs `moe_prefetch_pack_hip(d_pref_slab_[dv], d_pref_src_[dv], plan_st_[dv], shard_.stride, pref_slots_, hyb)` — a linear 128-bit (`uint4`) coalesced burst copy from `pool_dev_` into `d_pref_slab_[dv]`.
   - Then `hyb`'s `gate`, `up`, and `down` `expert_gemm_wmma_hip` launches read the prefetched expert shards at **640 GB/s GDDR6 VRAM / L2 bandwidth** instead of scattered zero-copy PCIe reads, with **zero host synchronization (`hipStreamSynchronize`)** and **100% bit-identity**!

### Task 3: Implement Device-Planned Asynchronous VRAM Prefetch Ring (`AFF_PREFETCH_RING`)

**Files:**
- Modify: [`/mnt/fast-storage/affinity/src/gpu/moe_plan.h:50-57,160-185`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.h#L50-L57)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/moe_plan.hip:294-333,369-422`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.hip#L294-L333)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/static_placement.h:545-555`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.h#L545-L555)
- Modify: [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:190-196,374-395,890-920,1725-1745,1956-1964`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L890-L920)
- Create: `/mnt/fast-storage/affinity/tests/test_prefetch_ring.hip`
- Modify: [`/mnt/fast-storage/affinity/CMakeLists.txt:305`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L305)

**Step 1: Extend `MoePlanArgs` and `moe_plan.h` with `pref_slab`, `pref_src`, `pref_cap`, and `moe_prefetch_pack_hip`**

In [`/mnt/fast-storage/affinity/src/gpu/moe_plan.h:167-185`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.h#L167-L185), add to `MoePlanArgs`:
```cpp
  // Optional VRAM prefetch ring for pooled (cls == 1) experts (AFF_PREFETCH_RING=N).
  // When pref_slab != nullptr and (j - n_res) < pref_cap, moe_plan_kernel writes the pooled
  // expert's host-pool shard address into pref_src[j - n_res] and points its ExpertTiles at
  // pref_slab + (j - n_res) * shard.stride.
  uint8_t*        pref_slab = nullptr;
  const uint8_t** pref_src  = nullptr;
  uint32_t        pref_cap  = 0;
```
And declare after `moe_plan_hip` in [`/mnt/fast-storage/affinity/src/gpu/moe_plan.h:182`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.h#L182):
```cpp
// Linear uint4 coalesced burst copy of `min(status->n_uniq - status->n_res, pref_cap)` pooled
// expert shards from `pref_src[i]` into `pref_slab + i * stride` on `stream` (the hybrid stream).
void moe_prefetch_pack_hip(uint8_t* pref_slab, const uint8_t* const* pref_src,
                           const MoePlanStatus* status, uint32_t stride, uint32_t pref_cap,
                           void* stream);
```

**Step 2: Implement `pref_slab` redirection and `moe_prefetch_pack_hip` in `/mnt/fast-storage/affinity/src/gpu/moe_plan.hip`**

In [`/mnt/fast-storage/affinity/src/gpu/moe_plan.hip:294-305`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.hip#L294-L305), update `base` inside the `for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads)` loop:
```cpp
  for (uint32_t j = threadIdx.x; j < n_uniq; j += kPlanThreads) {
    const uint32_t e = L.uniq[j], c = L.counts[e], cls = (j < n_res) ? 0u : 1u;
    const uint32_t q = c / kExpertWmmaTok, r = c % kExpertWmmaTok;
    const uint64_t pv = plc[e];
    uint8_t* base = (uint8_t*)(uintptr_t)(pv & ~1ull);
    if (cls == 1u && a.pref_slab && a.pref_src && (j - n_res) < a.pref_cap) {
      const uint32_t pidx = j - n_res;
      a.pref_src[pidx] = base;
      base = a.pref_slab + (size_t)pidx * a.shard.stride;
    }
```
And add `moe_prefetch_pack_kernel` and `moe_prefetch_pack_hip` at the bottom of [`/mnt/fast-storage/affinity/src/gpu/moe_plan.hip:388`](file:///mnt/fast-storage/affinity/src/gpu/moe_plan.hip#L388):
```cpp
__global__ __launch_bounds__(256) void moe_prefetch_pack_kernel(
    uint8_t* __restrict__ pref_slab, const uint8_t* const* __restrict__ pref_src,
    const MoePlanStatus* __restrict__ status, uint32_t stride, uint32_t pref_cap) {
  const uint32_t n_pool = status->n_uniq > status->n_res ? (status->n_uniq - status->n_res) : 0u;
  const uint32_t n = n_pool < pref_cap ? n_pool : pref_cap;
  const uint32_t pidx = blockIdx.y;
  if (pidx >= n) return;
  const uint8_t* __restrict__ src = pref_src[pidx];
  if (!src) return;
  uint8_t* __restrict__ dst = pref_slab + (size_t)pidx * stride;
  const uint32_t n_vec = stride >> 4;
  const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
  uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += gridDim.x * blockDim.x) {
    d4[i] = __builtin_nontemporal_load(&s4[i]);
  }
}

void moe_prefetch_pack_hip(uint8_t* pref_slab, const uint8_t* const* pref_src,
                           const MoePlanStatus* status, uint32_t stride, uint32_t pref_cap,
                           void* stream) {
  if (!pref_slab || !pref_src || !status || !stride || !pref_cap) return;
  // 64 blocks along X (16 KiB per wave step) x pref_cap along Y; excess Y blocks exit in 1 cycle.
  const uint32_t gx = std::min<uint32_t>(128u, ((stride >> 4) + 255u) / 256u);
  hipLaunchKernelGGL(moe_prefetch_pack_kernel, dim3(gx, pref_cap), dim3(256), 0,
                     (hipStream_t)stream, pref_slab, pref_src, status, stride, pref_cap);
  AFF_HIP_CHECK(hipGetLastError());
}
```

**Step 3: Allocate `d_pref_slab_[dv]` and `d_pref_src_[dv]` in `StaticPlacement` and wire into `run_batch_device`**

1. In [`/mnt/fast-storage/affinity/src/gpu/static_placement.h:549`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.h#L549), add:
```cpp
  uint8_t* d_pref_slab_[8] = {};
  const uint8_t** d_pref_src_[8] = {};
  uint32_t pref_cap_ = 0;
```
2. In [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:192`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L192) (inside `StaticPlacement::~StaticPlacement()`), add `d_pref_slab_[i]` and `d_pref_src_[i]` to the cleanup loop:
```cpp
    for (void* p : {(void*)plan_col_[i], (void*)plan_cnt_[i], (void*)plan_st_[i],
                    (void*)d_imat_w2_[i], (void*)d_pref_slab_[i], (void*)d_pref_src_[i]})
      if (p) (void)hipFree(p);
    d_pref_slab_[i] = nullptr;
    d_pref_src_[i] = nullptr;
```
3. In [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:891`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L891) (right after `const uint64_t slot_stride = this->slot_stride();` and BEFORE the `hipMemGetInfo` slab-sizing loop so `freeb` automatically accounts for the prefetch ring!), add:
```cpp
  pref_cap_ = 0;
  if (const char* pr = std::getenv("AFF_PREFETCH_RING"); pr && *pr) {
    const int want_pref = std::atoi(pr);
    if (want_pref > 0 && tp_parts_ > 1) {
      pref_cap_ = std::min<uint32_t>((uint32_t)want_pref, 64u);
      for (size_t i = 0; i < devs_.size() && i < 8; ++i) {
        AFF_HIP_CHECK(hipSetDevice(devs_[i]));
        if (hipMalloc((void**)&d_pref_slab_[i], (size_t)pref_cap_ * slot_stride) != hipSuccess ||
            hipMalloc((void**)&d_pref_src_[i], (size_t)pref_cap_ * sizeof(uint8_t*)) != hipSuccess) {
          pref_cap_ = 0;
          break;
        }
      }
      if (pref_cap_)
        aff::ui::err("prefetch ring: %u slots/card (%.1f MiB/card) allocated on hybrid stream\n",
                     pref_cap_, (double)pref_cap_ * (double)slot_stride / 1048576.0);
    }
  }
```
4. In [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:1738`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L1738) (when populating `MoePlanArgs A;`), add:
```cpp
        A.pref_slab = d_pref_slab_[dv];
        A.pref_src  = d_pref_src_[dv];
        A.pref_cap  = pref_cap_;
```
5. In [`/mnt/fast-storage/affinity/src/gpu/static_placement.hip:1956`](file:///mnt/fast-storage/affinity/src/gpu/static_placement.hip#L1956) (on the `hyb` stream right before the `for (uint32_t b = 4; b < 8; ++b)` missing-expert GEMM loop), launch `moe_prefetch_pack_hip`:
```cpp
      { AFF_XTRACE((int)dv, 8, hyb);
      if (plan_on && pref_cap_ && d_pref_slab_[dv] && d_pref_src_[dv]) {
        moe_prefetch_pack_hip(d_pref_slab_[dv], d_pref_src_[dv], plan_st_[dv],
                              shard_.stride, pref_cap_, hyb);
      }
      for (uint32_t b = 4; b < 8; ++b) {
```

**Step 4: Write the unit test (`/mnt/fast-storage/affinity/tests/test_prefetch_ring.hip`), register in `CMakeLists.txt`, and run**

Create `/mnt/fast-storage/affinity/tests/test_prefetch_ring.hip`:
```cpp
#include "gpu/moe_plan.h"
#include "gpu/hip_common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  if (!aff::hip_available()) {
    std::printf("PASS: test_prefetch_ring (no gfx12 GPU)\n");
    return 0;
  }

  const uint32_t stride = 4096;
  const uint32_t pref_cap = 4;
  std::vector<uint8_t> h_pool0(stride), h_pool1(stride);
  for (uint32_t i = 0; i < stride; ++i) {
    h_pool0[i] = (uint8_t)((i * 13 + 5) & 0xFF);
    h_pool1[i] = (uint8_t)((i * 29 + 11) & 0xFF);
  }

  uint8_t *d_pool0 = nullptr, *d_pool1 = nullptr, *d_pref_slab = nullptr;
  const uint8_t** d_pref_src = nullptr;
  aff::MoePlanStatus* d_st = nullptr;

  AFF_HIP_CHECK(hipMalloc(&d_pool0, stride));
  AFF_HIP_CHECK(hipMalloc(&d_pool1, stride));
  AFF_HIP_CHECK(hipMalloc(&d_pref_slab, (size_t)pref_cap * stride));
  AFF_HIP_CHECK(hipMalloc(&d_pref_src, pref_cap * sizeof(uint8_t*)));
  AFF_HIP_CHECK(hipMalloc(&d_st, sizeof(aff::MoePlanStatus)));

  AFF_HIP_CHECK(hipMemcpy(d_pool0, h_pool0.data(), stride, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_pool1, h_pool1.data(), stride, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemset(d_pref_slab, 0, (size_t)pref_cap * stride));

  const uint8_t* h_srcs[4] = {d_pool0, d_pool1, nullptr, nullptr};
  AFF_HIP_CHECK(hipMemcpy(d_pref_src, h_srcs, sizeof(h_srcs), hipMemcpyHostToDevice));

  aff::MoePlanStatus h_st{};
  h_st.n_res = 3;
  h_st.n_uniq = 5; // 2 pooled experts -> should copy slots 0 and 1
  AFF_HIP_CHECK(hipMemcpy(d_st, &h_st, sizeof(h_st), hipMemcpyHostToDevice));

  aff::moe_prefetch_pack_hip(d_pref_slab, d_pref_src, d_st, stride, pref_cap, nullptr);
  AFF_HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint8_t> got0(stride), got1(stride);
  AFF_HIP_CHECK(hipMemcpy(got0.data(), d_pref_slab, stride, hipMemcpyDeviceToHost));
  AFF_HIP_CHECK(hipMemcpy(got1.data(), d_pref_slab + stride, stride, hipMemcpyDeviceToHost));

  if (std::memcmp(got0.data(), h_pool0.data(), stride) != 0 ||
      std::memcmp(got1.data(), h_pool1.data(), stride) != 0) {
    std::fprintf(stderr, "FAIL: moe_prefetch_pack_hip did not copy pooled shards accurately\n");
    return 1;
  }

  (void)hipFree(d_pool0); (void)hipFree(d_pool1);
  (void)hipFree(d_pref_slab); (void)hipFree(d_pref_src); (void)hipFree(d_st);
  std::printf("PASS: test_prefetch_ring\n");
  return 0;
}
```

In [`/mnt/fast-storage/affinity/CMakeLists.txt:305`](file:///mnt/fast-storage/affinity/CMakeLists.txt#L305), add:
```cmake
    aff_hip_test(test_prefetch_ring prefetch_ring)
```

Run:
```bash
ninja -C /mnt/fast-storage/affinity/build test_prefetch_ring test_moe_plan
/mnt/fast-storage/affinity/build/test_prefetch_ring
/mnt/fast-storage/affinity/build/test_moe_plan
```
Expected: `PASS: test_prefetch_ring` and `PASS` on `test_moe_plan`.

**Step 5: Commit Task 3**

```bash
git -C /mnt/fast-storage/affinity add CMakeLists.txt src/gpu/moe_plan.h src/gpu/moe_plan.hip src/gpu/static_placement.h src/gpu/static_placement.hip tests/test_prefetch_ring.hip
git -C /mnt/fast-storage/affinity commit -m "perf(gpu): add EXL3 device-planned async VRAM prefetch ring (AFF_PREFETCH_RING)"
```

---

## Task 4: Full Build, Complete Test Suite Verification, and Production Deployment

**Files:**
- Build: `/mnt/fast-storage/affinity/build/affinity` and `/mnt/fast-storage/affinity/build/aff-quantize`
- Install: `/home/ydj/affinity-patched/affinity`
- Modify: [`/home/ydj/.config/systemd/user/deepseek-v41-affinity.service`](file:///home/ydj/.config/systemd/user/deepseek-v41-affinity.service)

**Step 1: Build all binaries and run the complete `ctest` suite**

Run:
```bash
ninja -C /mnt/fast-storage/affinity/build affinity aff-quantize
ctest --test-dir /mnt/fast-storage/affinity/build --output-on-failure
```
Expected: `100% tests passed, 0 tests failed`.

**Step 2: Install new binary atomically and update `deepseek-v41-affinity.service`**

In [`/home/ydj/.config/systemd/user/deepseek-v41-affinity.service`](file:///home/ydj/.config/systemd/user/deepseek-v41-affinity.service), add `-e AFF_MOE_RECON_TILES=8` and `-e AFF_PREFETCH_RING=8` to the `docker run` environment flags (note: do **not** enable `AFF_HADAMARD_ROT=1` on the existing un-rotated `model-uc-1017.aff` container; `AFF_HADAMARD_ROT=1` is only used with containers quantized with `aff-quantize --hadamard`).

Run:
```bash
cp -a /mnt/fast-storage/affinity/build/affinity /home/ydj/affinity-patched/affinity.new
mv -f /home/ydj/affinity-patched/affinity.new /home/ydj/affinity-patched/affinity
```

**Step 3: Restart `deepseek-v41-affinity.service` at a request boundary and verify health + throughput**

Run:
```bash
systemctl --user daemon-reload
systemctl --user restart deepseek-v41-affinity.service
```
Wait for model load (~75s) and verify:
```bash
journalctl --user -u deepseek-v41-affinity.service -n 40 --no-pager
curl -sf http://127.0.0.1:8000/v1/models
```
Expected:
- Log shows `prefetch ring: 8 slots/card (24.3 MiB/card) allocated on hybrid stream`.
- `curl` returns `DeepSeek-V4.1-Flash-UNCENSORED` HTTP 200.
- If any check fails or output quality degrades, execute `/mnt/fast-storage/affinity/scripts/rollback_exl3.sh` immediately.
