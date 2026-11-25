#include "gpu/engram_gpu.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static int run_case(uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t hs, bool bf16) {
  const uint32_t out_dim = (n_hc + 1u) * n_embd;
  const uint32_t split = 2u * n_embd + n_embd / 2u;  // Asymmetric mid-band split (matches R=2 V4.1)
  const float eps = 1e-6f;
  const size_t hc_elems = (size_t)n_hc * n_embd * hs;
  const size_t kv_elems = (size_t)out_dim * nb;
  const size_t w_elems = (size_t)n_hc * n_embd;
  const size_t part_elems = (size_t)3u * n_hc * nb;

  std::vector<float> hc0_f32(hc_elems);
  std::vector<uint16_t> hc0_bf16(hc_elems);
  std::vector<float> kv_full(kv_elems);
  std::vector<float> w_host(w_elems);

  uint32_t seed = 1337u + nb * 17u + (bf16 ? 1u : 0u);
  auto nextf = [&]() -> float {
    seed = seed * 1664525u + 1013904223u;
    int32_t s = (int32_t)(seed >> 8) - (1 << 23);
    return (float)s * 1e-7f;
  };
  for (size_t i = 0; i < hc_elems; ++i) {
    float v = nextf();
    hc0_bf16[i] = f32_to_bf16(v);
    hc0_f32[i] = bf16 ? bf16_to_f32(hc0_bf16[i]) : v;
  }
  for (size_t i = 0; i < kv_elems; ++i) kv_full[i] = nextf();
  for (size_t i = 0; i < w_elems; ++i) w_host[i] = 0.5f + 0.5f * nextf();

  void* d_hc_ref = nullptr;
  void* d_hc_spl = nullptr;
  float *d_kv_full = nullptr, *d_kv_spl = nullptr, *d_w = nullptr;
  float *d_part0 = nullptr, *d_part1 = nullptr;
  const size_t hc_bytes = hc_elems * (bf16 ? 2u : 4u);
  (void)hipMalloc(&d_hc_ref, hc_bytes);
  (void)hipMalloc(&d_hc_spl, hc_bytes);
  (void)hipMalloc(&d_kv_full, kv_elems * sizeof(float));
  (void)hipMalloc(&d_kv_spl, kv_elems * sizeof(float));
  (void)hipMalloc(&d_w, w_elems * sizeof(float));
  (void)hipMalloc(&d_part0, part_elems * sizeof(float));
  (void)hipMalloc(&d_part1, part_elems * sizeof(float));

  const void* hc_src = bf16 ? (const void*)hc0_bf16.data() : (const void*)hc0_f32.data();
  (void)hipMemcpy(d_hc_ref, hc_src, hc_bytes, hipMemcpyHostToDevice);
  (void)hipMemcpy(d_hc_spl, hc_src, hc_bytes, hipMemcpyHostToDevice);
  (void)hipMemcpy(d_kv_full, kv_full.data(), kv_elems * sizeof(float), hipMemcpyHostToDevice);
  // Zero out half of key/value bands on each synthetic shard copy to ensure neither rank reads peer keys.
  std::vector<float> kv_rank0 = kv_full;
  std::vector<float> kv_rank1 = kv_full;
  for (size_t i = (size_t)split * nb; i < (size_t)n_hc * n_embd * nb; ++i) kv_rank0[i] = 777.0f;
  for (size_t i = 0; i < (size_t)split * nb; ++i) kv_rank1[i] = -777.0f;

  (void)hipMemcpy(d_w, w_host.data(), w_elems * sizeof(float), hipMemcpyHostToDevice);

  // Reference full gate.
  aff::engram_gate_hip(d_hc_ref, bf16, d_kv_full, d_w, n_embd, n_hc, nb, hs, eps, nullptr);

  // Split rank 0 keys [0, split)
  (void)hipMemcpy(d_kv_spl, kv_rank0.data(), kv_elems * sizeof(float), hipMemcpyHostToDevice);
  aff::engram_gate_partial_hip(d_hc_spl, bf16, d_kv_spl, d_w, d_part0, n_embd, n_hc, nb, hs, 0u,
                               split, nullptr);

  // Split rank 1 keys [split, out_dim)
  (void)hipMemcpy(d_kv_spl, kv_rank1.data(), kv_elems * sizeof(float), hipMemcpyHostToDevice);
  aff::engram_gate_partial_hip(d_hc_spl, bf16, d_kv_spl, d_w, d_part1, n_embd, n_hc, nb, hs,
                               split, out_dim, nullptr);

  // Apply using gathered value band [n_hc * n_embd, out_dim) + rank 0/1 partials.
  (void)hipMemcpy(d_kv_spl, kv_full.data(), kv_elems * sizeof(float), hipMemcpyHostToDevice);
  aff::engram_gate_apply_hip(d_hc_spl, bf16, d_kv_spl, d_part0, d_part1, n_embd, n_hc, nb, hs,
                             eps, nullptr);
  (void)hipDeviceSynchronize();

  float max_err = 0.0f;
  if (bf16) {
    std::vector<uint16_t> r_ref(hc_elems), r_spl(hc_elems);
    (void)hipMemcpy(r_ref.data(), d_hc_ref, hc_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(r_spl.data(), d_hc_spl, hc_bytes, hipMemcpyDeviceToHost);
    for (size_t i = 0; i < hc_elems; ++i) {
      float e = std::fabs(bf16_to_f32(r_ref[i]) - bf16_to_f32(r_spl[i]));
      if (e > max_err) max_err = e;
    }
  } else {
    std::vector<float> r_ref(hc_elems), r_spl(hc_elems);
    (void)hipMemcpy(r_ref.data(), d_hc_ref, hc_bytes, hipMemcpyDeviceToHost);
    (void)hipMemcpy(r_spl.data(), d_hc_spl, hc_bytes, hipMemcpyDeviceToHost);
    for (size_t i = 0; i < hc_elems; ++i) {
      float e = std::fabs(r_ref[i] - r_spl[i]);
      if (e > max_err) max_err = e;
    }
  }

  (void)hipFree(d_hc_ref);
  (void)hipFree(d_hc_spl);
  (void)hipFree(d_kv_full);
  (void)hipFree(d_kv_spl);
  (void)hipFree(d_w);
  (void)hipFree(d_part0);
  (void)hipFree(d_part1);

  if (max_err >= 1e-5f) {
    std::fprintf(stderr, "FAIL: nb=%u hs=%u bf16=%d max_err=%.6g\n", nb, hs, (int)bf16, max_err);
    return 1;
  }
  std::printf("PASS: engram split-gate nb=%u hs=%u bf16=%d max_err=%.3e\n", nb, hs, (int)bf16,
              max_err);
  return 0;
}

int main() {
  int dev_count = 0;
  hipError_t err = hipGetDeviceCount(&dev_count);
  if (err != hipSuccess || dev_count <= 0) {
    std::printf("PASS: test_engram_split_gate offline (no GPU devices in environment)\n");
    return 0;
  }
  if (run_case(512u, 4u, 1u, 1u, false)) return 1;
  if (run_case(512u, 4u, 16u, 16u, true)) return 1;
  if (run_case(512u, 4u, 128u, 256u, true)) return 1;
  if (run_case(512u, 4u, 128u, 256u, false)) return 1;
  return 0;
}
