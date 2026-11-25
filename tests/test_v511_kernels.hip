#include "gpu/attn_flash.h"
#include "gpu/batch_kernels.h"
#include "gpu/hip_common.h"
#include "gpu/indexer_gpu.h"
#include "quant/source_dtypes.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int check_lmhead_nt_parity() {
  const uint32_t rows = 12800u, cols = 4096u, nt = 6u;  // 50 MiB > 48 MiB NT threshold
  const uint32_t sc_cols = cols / 128u;
  std::vector<uint8_t> h_w((size_t)rows * cols);
  std::vector<uint16_t> h_s((size_t)rows * sc_cols);
  std::vector<float> h_x((size_t)nt * cols);
  uint32_t rng = 12345u;
  for (size_t i = 0; i < h_w.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    uint8_t b = (uint8_t)(rng >> 24);
    if ((b & 0x7Fu) == 0x7Fu) b = 0x38u;
    h_w[i] = b;
  }
  for (size_t i = 0; i < h_s.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_s[i] = aff::float_to_bf16(0.02f + 0.001f * (float)((rng >> 16) & 63u));
  }
  for (size_t i = 0; i < h_x.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_x[i] = ((float)(int32_t)(rng >> 8) - (float)(1 << 23)) * 1e-6f;
  }

  uint8_t* d_w = nullptr;
  uint16_t* d_s = nullptr;
  float *d_x = nullptr, *d_y0 = nullptr, *d_y1 = nullptr;
  if (hipMalloc(&d_w, h_w.size()) != hipSuccess ||
      hipMalloc(&d_s, h_s.size() * sizeof(uint16_t)) != hipSuccess ||
      hipMalloc(&d_x, h_x.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_y0, (size_t)nt * rows * sizeof(float)) != hipSuccess ||
      hipMalloc(&d_y1, (size_t)nt * rows * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "FAIL: check_lmhead_nt_parity hipMalloc\n");
    return 1;
  }
  (void)hipMemcpy(d_w, h_w.data(), h_w.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_s, h_s.data(), h_s.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice);

  ::setenv("AFF_LMHEAD_NT", "0", 1);
  if (!aff::matvec_fp8_g128_multix_hip(d_w, d_s, d_x, d_y0, rows, cols, nt, rows, 1, nullptr))
    return 1;
  ::setenv("AFF_LMHEAD_NT", "1", 1);
  if (!aff::matvec_fp8_g128_multix_hip(d_w, d_s, d_x, d_y1, rows, cols, nt, rows, 1, nullptr))
    return 1;
  ::unsetenv("AFF_LMHEAD_NT");
  (void)hipDeviceSynchronize();

  std::vector<float> h_y0((size_t)nt * rows), h_y1((size_t)nt * rows);
  (void)hipMemcpy(h_y0.data(), d_y0, h_y0.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(h_y1.data(), d_y1, h_y1.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipFree(d_w); (void)hipFree(d_s); (void)hipFree(d_x);
  (void)hipFree(d_y0); (void)hipFree(d_y1);

  if (std::memcmp(h_y0.data(), h_y1.data(), h_y0.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: AFF_LMHEAD_NT=0 vs 1 mismatch\n");
    return 1;
  }
  std::printf("PASS: check_lmhead_nt_parity (rows=%u cols=%u nt=%u)\n", rows, cols, nt);
  return 0;
}

int check_gemm_swizzle_and_tail_parity() {
  const uint32_t rows = 2560u, cols = 1024u;
  const uint32_t sc_cols = cols / 128u;
  for (uint32_t nb : {16u, 256u}) {
    std::vector<uint8_t> h_w((size_t)rows * cols), h_sw((size_t)(rows / 128u) * sc_cols);
    std::vector<uint8_t> h_x((size_t)nb * cols);
    std::vector<float> h_sx((size_t)nb * sc_cols);
    uint32_t rng = 777u + nb;
    for (size_t i = 0; i < h_w.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      uint8_t b = (uint8_t)(rng >> 24);
      h_w[i] = ((b & 0x7Fu) == 0x7Fu) ? 0x38u : b;
    }
    for (size_t i = 0; i < h_sw.size(); ++i) h_sw[i] = 127u + (uint8_t)(i % 3u);
    for (size_t i = 0; i < h_x.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      uint8_t b = (uint8_t)(rng >> 24);
      h_x[i] = ((b & 0x7Fu) == 0x7Fu) ? 0x34u : b;
    }
    for (size_t i = 0; i < h_sx.size(); ++i) h_sx[i] = 0.05f + 0.01f * (float)(i % 5u);

    uint8_t *d_w = nullptr, *d_wp = nullptr, *d_sw = nullptr, *d_x = nullptr;
    float *d_sx = nullptr, *d_y0 = nullptr, *d_y1 = nullptr;
    (void)hipMalloc(&d_w, h_w.size());
    (void)hipMalloc(&d_wp, h_w.size());
    (void)hipMalloc(&d_sw, h_sw.size());
    (void)hipMalloc(&d_x, h_x.size());
    (void)hipMalloc(&d_sx, h_sx.size() * sizeof(float));
    (void)hipMalloc(&d_y0, (size_t)rows * nb * sizeof(float));
    (void)hipMalloc(&d_y1, (size_t)rows * nb * sizeof(float));
    (void)hipMemcpy(d_w, h_w.data(), h_w.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_sw, h_sw.data(), h_sw.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_x, h_x.data(), h_x.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_sx, h_sx.data(), h_sx.size() * sizeof(float), hipMemcpyHostToDevice);
    aff::preshuffle_fp8_hip(d_w, d_wp, rows, cols, nullptr);

    ::setenv("AFF_GEMM_SWIZZLE", "0", 1);
    aff::batch_gemm_fp8_hip(d_wp, d_sw, d_x, d_sx, d_y0, rows, cols, nb, 0u, 0u, nullptr, true);
    ::setenv("AFF_GEMM_SWIZZLE", "1", 1);
    aff::batch_gemm_fp8_hip(d_wp, d_sw, d_x, d_sx, d_y1, rows, cols, nb, 0u, 0u, nullptr, true);
    ::unsetenv("AFF_GEMM_SWIZZLE");
    (void)hipDeviceSynchronize();

    std::vector<float> h_y0((size_t)rows * nb), h_y1((size_t)rows * nb);
    (void)hipMemcpy(h_y0.data(), d_y0, h_y0.size() * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_y1.data(), d_y1, h_y1.size() * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(d_w); (void)hipFree(d_wp); (void)hipFree(d_sw); (void)hipFree(d_x);
    (void)hipFree(d_sx); (void)hipFree(d_y0); (void)hipFree(d_y1);

    if (std::memcmp(h_y0.data(), h_y1.data(), h_y0.size() * sizeof(float)) != 0) {
      std::fprintf(stderr, "FAIL: AFF_GEMM_SWIZZLE nb=%u mismatch\n", nb);
      return 1;
    }
  }
  std::printf("PASS: check_gemm_swizzle_and_tail_parity\n");
  return 0;
}

int check_indexer_nt_store_parity() {
  const uint32_t n_keys = 512u, n_head = 64u, dim = aff::kIdxDim;
  for (uint32_t nb : {1u, 32u}) {
    std::vector<uint8_t> h_q((size_t)nb * n_head * dim), h_k((size_t)n_keys * dim);
    std::vector<float> h_qs(nb), h_ks(n_keys), h_hw((size_t)nb * n_head);
    uint32_t rng = 333u + nb;
    for (size_t i = 0; i < h_q.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_q[i] = ((rng >> 24) & 0x7Fu) == 0x7Fu ? 0x38u : (uint8_t)(rng >> 24);
    }
    for (size_t i = 0; i < h_k.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_k[i] = ((rng >> 24) & 0x7Fu) == 0x7Fu ? 0x38u : (uint8_t)(rng >> 24);
    }
    for (size_t i = 0; i < h_qs.size(); ++i) h_qs[i] = 0.05f + 0.01f * (float)i;
    for (size_t i = 0; i < h_ks.size(); ++i) h_ks[i] = 0.04f + 0.002f * (float)(i & 15u);
    for (size_t i = 0; i < h_hw.size(); ++i) h_hw[i] = 0.1f * (float)((int)(i & 7u) - 3);

    uint8_t *d_q = nullptr, *d_k = nullptr;
    float *d_qs = nullptr, *d_ks = nullptr, *d_hw = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
    (void)hipMalloc(&d_q, h_q.size());
    (void)hipMalloc(&d_k, h_k.size());
    (void)hipMalloc(&d_qs, h_qs.size() * sizeof(float));
    (void)hipMalloc(&d_ks, h_ks.size() * sizeof(float));
    (void)hipMalloc(&d_hw, h_hw.size() * sizeof(float));
    (void)hipMalloc(&d_o0, (size_t)nb * n_keys * sizeof(float));
    (void)hipMalloc(&d_o1, (size_t)nb * n_keys * sizeof(float));
    (void)hipMemcpy(d_q, h_q.data(), h_q.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_k, h_k.data(), h_k.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_qs, h_qs.data(), h_qs.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_hw, h_hw.data(), h_hw.size() * sizeof(float), hipMemcpyHostToDevice);

    ::setenv("AFF_IDX_SCORE_NT_BYTES", "0", 1);
    ::setenv("AFF_IDX_SCORE_NT_STORE", "0", 1);
    aff::indexer_scores_hip(d_q, d_qs, d_k, d_ks, d_hw, d_o0, nb, n_keys, n_head, nullptr);
    ::setenv("AFF_IDX_SCORE_NT_STORE", "1", 1);
    aff::indexer_scores_hip(d_q, d_qs, d_k, d_ks, d_hw, d_o1, nb, n_keys, n_head, nullptr);
    ::unsetenv("AFF_IDX_SCORE_NT_STORE");
    ::unsetenv("AFF_IDX_SCORE_NT_BYTES");
    (void)hipDeviceSynchronize();

    std::vector<float> h_o0((size_t)nb * n_keys), h_o1((size_t)nb * n_keys);
    (void)hipMemcpy(h_o0.data(), d_o0, h_o0.size() * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_o1.data(), d_o1, h_o1.size() * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(d_q); (void)hipFree(d_k); (void)hipFree(d_qs);
    (void)hipFree(d_ks); (void)hipFree(d_hw); (void)hipFree(d_o0); (void)hipFree(d_o1);

    if (std::memcmp(h_o0.data(), h_o1.data(), h_o0.size() * sizeof(float)) != 0) {
      std::fprintf(stderr, "FAIL: AFF_IDX_SCORE_NT_STORE nb=%u mismatch\n", nb);
      return 1;
    }
  }
  std::printf("PASS: check_indexer_nt_store_parity\n");
  return 0;
}

int check_flash_longctx_nt_parity() {
  const uint32_t n = 4u, n_head = 64u, n_comp = 128u, sliding = 64u, ring = 128u;
  const uint32_t n_raw = 64u, pos0 = 200u, raw_floor = 0u;
  std::vector<float> h_q((size_t)n * n_head * aff::kFlashWidth);
  std::vector<uint8_t> h_raw((size_t)ring * aff::kFlashRowBytes);
  std::vector<uint8_t> h_comp((size_t)n_comp * aff::kFlashRowBytes);
  std::vector<float> row_tmp(aff::kFlashWidth);
  uint32_t rng = 555u;
  for (size_t i = 0; i < h_q.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_q[i] = ((float)(int32_t)(rng >> 8) - (float)(1 << 23)) * 2e-6f;
  }
  for (uint32_t r = 0; r < ring; ++r) {
    for (uint32_t d = 0; d < aff::kFlashWidth; ++d) {
      rng = rng * 1664525u + 1013904223u;
      row_tmp[d] = ((float)(int32_t)(rng >> 8) - (float)(1 << 23)) * 2e-6f;
    }
    aff::kv_encode_fp8r_row(row_tmp.data(), h_raw.data() + (size_t)r * aff::kFlashRowBytes);
  }
  for (uint32_t c = 0; c < n_comp; ++c) {
    for (uint32_t d = 0; d < aff::kFlashWidth; ++d) {
      rng = rng * 1664525u + 1013904223u;
      row_tmp[d] = ((float)(int32_t)(rng >> 8) - (float)(1 << 23)) * 2e-6f;
    }
    aff::kv_encode_fp8r_row(row_tmp.data(), h_comp.data() + (size_t)c * aff::kFlashRowBytes);
  }
  (void)n_raw;

  float *d_q = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
  uint8_t *d_raw = nullptr, *d_comp = nullptr;
  (void)hipMalloc(&d_q, h_q.size() * sizeof(float));
  (void)hipMalloc(&d_raw, h_raw.size());
  (void)hipMalloc(&d_comp, h_comp.size());
  (void)hipMalloc(&d_o0, h_q.size() * sizeof(float));
  (void)hipMalloc(&d_o1, h_q.size() * sizeof(float));
  (void)hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_comp, h_comp.data(), h_comp.size(), hipMemcpyHostToDevice);

  const float scale = 1.0f / std::sqrt((float)aff::kFlashWidth);
  ::setenv("AFF_FLASH_NT_MIB", "0", 1);
  ::setenv("AFF_FLASH_LONGCTX_NT", "0", 1);
  aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, nullptr, nullptr, n_comp, nullptr, d_o0, 0u, n, n,
                            pos0, sliding, ring, n_comp, n_head, scale, nullptr, false, nullptr,
                            n_comp, nullptr, 0u, raw_floor, nullptr);
  ::setenv("AFF_FLASH_LONGCTX_NT", "1", 1);
  aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, nullptr, nullptr, n_comp, nullptr, d_o1, 0u, n, n,
                            pos0, sliding, ring, n_comp, n_head, scale, nullptr, false, nullptr,
                            n_comp, nullptr, 0u, raw_floor, nullptr);
  ::unsetenv("AFF_FLASH_LONGCTX_NT");
  ::unsetenv("AFF_FLASH_NT_MIB");
  (void)hipDeviceSynchronize();

  std::vector<float> h_o0(h_q.size()), h_o1(h_q.size());
  (void)hipMemcpy(h_o0.data(), d_o0, h_o0.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(h_o1.data(), d_o1, h_o1.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipFree(d_q); (void)hipFree(d_raw); (void)hipFree(d_comp);
  (void)hipFree(d_o0); (void)hipFree(d_o1);

  float max_abs = 0.0f;
  for (size_t i = 0; i < h_o0.size(); ++i) max_abs = std::fmax(max_abs, std::fabs(h_o0[i] - h_o1[i]));
  if (max_abs > 1e-5f) {
    std::fprintf(stderr, "FAIL: AFF_FLASH_LONGCTX_NT mismatch max_abs=%g\n", (double)max_abs);
    return 1;
  }
  std::printf("PASS: check_flash_longctx_nt_parity (max_abs=%g)\n", (double)max_abs);
  return 0;
}

}  // namespace

int main() {
  int dev_cnt = 0;
  if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) {
    std::fprintf(stderr, "No HIP devices found\n");
    return 1;
  }
  if (check_lmhead_nt_parity() != 0) return 1;
  if (check_gemm_swizzle_and_tail_parity() != 0) return 1;
  if (check_indexer_nt_store_parity() != 0) return 1;
  if (check_flash_longctx_nt_parity() != 0) return 1;
  std::printf("PASS: all v511 kernel tests passed.\n");
  return 0;
}
