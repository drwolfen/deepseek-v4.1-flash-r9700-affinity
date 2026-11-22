#include "gpu/attention_gpu.h"
#include "gpu/attn_flash.h"
#include "gpu/batch_kernels.h"
#include "gpu/compressor_gpu.h"
#include "gpu/dense_gpu.h"
#include "gpu/device_math.h"
#include "gpu/engram_gpu.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/indexer_gpu.h"
#include "gpu/moe_plan.h"
#include "gpu/sampler_gpu.h"
#include "gpu/tp_allreduce.h"
#include "engine/expert_kernel.h"
#include "quant/blockquant.h"
#include "format/aff_format.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
__global__ void test_e4m3_enc_kernel(const float* in, uint8_t* out_hw, uint8_t* out_sw,
                                     uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_hw[i] = aff::e4m3_enc(in[i]);
    out_sw[i] = aff::e4m3_enc_sw(in[i]);
  }
}

static float mxfp8_e4m3_to_f32_ref(uint8_t b) {
  const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
  const uint32_t ex = (b >> 3) & 0x0Fu;
  const uint32_t mn = b & 0x07u;
  if (ex == 15u && mn == 7u) return 0.0f;
  if (ex == 0u) {
    const float mag = (float)mn * (1.0f / 512.0f);
    return (b & 0x80u) ? -mag : mag;
  }
  const uint32_t u = sign | ((ex + 120u) << 23) | (mn << 20);
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
}  // namespace

static uint16_t f32_to_bf16_cpu(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32_cpu(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static float e4m3_round_ref(float v) {
  float a = std::fabs(v);
  if (a == 0.0f) return v;
  uint32_t bits;
  std::memcpy(&bits, &a, 4);
  int ea = std::max((int)((bits >> 23) & 0xFF) - 127, -6);
  uint32_t ulp_bits = (uint32_t)(ea - 3 + 127) << 23;
  float ulp;
  std::memcpy(&ulp, &ulp_bits, 4);
  float r = (a / ulp + 12582912.0f) - 12582912.0f;
  float q = r * ulp;
  return std::signbit(v) ? -q : q;
}

static void o_fp8_qdq_block32_ref(uint16_t* x, uint32_t dim, uint32_t nb) {
  for (uint32_t b = 0; b < nb; ++b) {
    for (uint32_t d0 = 0; d0 < dim; d0 += 32) {
      float amax = 1e-4f;
      for (uint32_t i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(bf16_to_f32_cpu(x[(size_t)(d0 + i) * nb + b])));
      float t = amax * (1.0f / 448.0f);
      uint32_t tbits;
      std::memcpy(&tbits, &t, 4);
      int e = (int)((tbits >> 23) & 0xFF) - 127 + ((tbits & 0x7FFFFFu) != 0 ? 1 : 0);
      uint32_t sbits = (uint32_t)(e + 127) << 23;
      float s;
      std::memcpy(&s, &sbits, 4);
      for (uint32_t i = 0; i < 32; ++i) {
        float v = bf16_to_f32_cpu(x[(size_t)(d0 + i) * nb + b]) / s;
        v = std::min(448.0f, std::max(-448.0f, v));
        x[(size_t)(d0 + i) * nb + b] = f32_to_bf16_cpu(e4m3_round_ref(v) * s);
      }
    }
  }
}

int main() {
  using namespace aff;
  // 1. Test batch_o_fp8_qdq_bf16_hip bit-exact vs CPU reference
  {
    const uint32_t dim = 512, nb = 6;
    std::vector<uint16_t> h_in((size_t)dim * nb);
    for (size_t i = 0; i < h_in.size(); ++i)
      h_in[i] = f32_to_bf16_cpu(std::sin((float)i * 0.13f) * 3.7f);
    std::vector<uint16_t> h_ref = h_in;
    o_fp8_qdq_block32_ref(h_ref.data(), dim, nb);

    uint16_t* d_x = nullptr;
    assert(hipMalloc(&d_x, h_in.size() * sizeof(uint16_t)) == hipSuccess);
    assert(hipMemcpy(d_x, h_in.data(), h_in.size() * sizeof(uint16_t), hipMemcpyHostToDevice) ==
           hipSuccess);
    batch_o_fp8_qdq_bf16_hip(d_x, dim / 32, nb, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<uint16_t> h_out(h_in.size());
    assert(hipMemcpy(h_out.data(), d_x, h_in.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) ==
           hipSuccess);
    (void)hipFree(d_x);
    for (size_t i = 0; i < h_ref.size(); ++i) {
      if (h_out[i] != h_ref[i]) {
        std::fprintf(stderr, "FAIL: batch_o_fp8_qdq mismatch at %zu: got %04x want %04x\n", i,
                     h_out[i], h_ref[i]);
        return 1;
      }
    }
  }

  // 2. Test fused batch_hc_expand_hip + tap bit-exact vs separate batch_hc_expand_hip + batch_hc_tap_hip
  {
    const uint32_t n_embd = 256, n_hc = 4, nb = 8;
    const uint32_t b_lo = 1, n_keep = 5, row0 = 0, rows_cap = 6, stride = n_embd * 3, slot_off = n_embd;
    std::vector<uint16_t> h_hc((size_t)n_hc * n_embd * nb);
    std::vector<float> h_block((size_t)n_embd * nb);
    std::vector<float> h_post((size_t)n_hc * nb);
    std::vector<float> h_comb((size_t)n_hc * n_hc * nb);
    for (size_t i = 0; i < h_hc.size(); ++i) h_hc[i] = f32_to_bf16_cpu(std::sin((float)i * 0.07f));
    for (size_t i = 0; i < h_block.size(); ++i) h_block[i] = std::cos((float)i * 0.05f);
    for (size_t i = 0; i < h_post.size(); ++i) h_post[i] = 0.25f + 0.01f * (float)(i % 7);
    for (size_t i = 0; i < h_comb.size(); ++i) h_comb[i] = 0.25f;

    uint16_t *d_hc = nullptr, *d_out1 = nullptr, *d_out2 = nullptr;
    float *d_blk1 = nullptr, *d_blk2 = nullptr, *d_post = nullptr, *d_comb = nullptr;
    float *d_tap1 = nullptr, *d_tap2 = nullptr;
    assert(hipMalloc(&d_hc, h_hc.size() * 2) == hipSuccess);
    assert(hipMalloc(&d_out1, h_hc.size() * 2) == hipSuccess);
    assert(hipMalloc(&d_out2, h_hc.size() * 2) == hipSuccess);
    assert(hipMalloc(&d_blk1, h_block.size() * 4) == hipSuccess);
    assert(hipMalloc(&d_blk2, h_block.size() * 4) == hipSuccess);
    assert(hipMalloc(&d_post, h_post.size() * 4) == hipSuccess);
    assert(hipMalloc(&d_comb, h_comb.size() * 4) == hipSuccess);
    assert(hipMalloc(&d_tap1, (size_t)rows_cap * stride * 4) == hipSuccess);
    assert(hipMalloc(&d_tap2, (size_t)rows_cap * stride * 4) == hipSuccess);

    (void)hipMemcpy(d_hc, h_hc.data(), h_hc.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_blk1, h_block.data(), h_block.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_blk2, h_block.data(), h_block.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_post, h_post.data(), h_post.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_comb, h_comb.data(), h_comb.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemset(d_tap1, 0, (size_t)rows_cap * stride * 4);
    (void)hipMemset(d_tap2, 0, (size_t)rows_cap * stride * 4);

    // Unfused reference
    batch_hc_expand_hip(d_out1, d_blk1, d_hc, d_post, d_comb, n_embd, n_hc, nb, true, nullptr);
    batch_hc_tap_hip(d_out1, d_tap1, n_embd, n_hc, nb, b_lo, n_keep, row0, rows_cap, stride,
                     slot_off, nullptr);
    // Fused
    batch_hc_expand_hip(d_out2, d_blk2, d_hc, d_post, d_comb, n_embd, n_hc, nb, true, nullptr,
                        d_tap2, b_lo, n_keep, row0, rows_cap, stride, slot_off);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> h_tap1((size_t)rows_cap * stride), h_tap2((size_t)rows_cap * stride);
    (void)hipMemcpy(h_tap1.data(), d_tap1, h_tap1.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_tap2.data(), d_tap2, h_tap2.size() * 4, hipMemcpyDeviceToHost);
    assert(std::memcmp(h_tap1.data(), h_tap2.data(), h_tap1.size() * 4) == 0);

    for (void* p : {(void*)d_hc, (void*)d_out1, (void*)d_out2, (void*)d_blk1, (void*)d_blk2,
                    (void*)d_post, (void*)d_comb, (void*)d_tap1, (void*)d_tap2})
      (void)hipFree(p);
  }

  // 3. Test quantize_bf16_to_w4_g128_hip + matvec_w4_multix_hip on GPU
  {
    const uint32_t rows = 128, cols = 512, nt = 5;
    std::vector<uint16_t> h_w((size_t)rows * cols);
    for (size_t i = 0; i < h_w.size(); ++i)
      h_w[i] = f32_to_bf16_cpu(std::sin((float)i * 0.0173f) * 0.08f +
                               std::cos((float)i * 0.113f) * 0.03f);
    std::vector<float> h_x((size_t)nt * cols);
    for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = std::sin((float)i * 0.031f);

    uint16_t *d_w = nullptr, *d_sc = nullptr;
    uint8_t* d_w4 = nullptr;
    float *d_x = nullptr, *d_out_tm = nullptr, *d_out_dm = nullptr;
    assert(hipMalloc(&d_w, h_w.size() * 2) == hipSuccess);
    assert(hipMalloc(&d_w4, (size_t)rows * (cols / 2)) == hipSuccess);
    assert(hipMalloc(&d_sc, (size_t)rows * (cols / 128) * 2) == hipSuccess);
    assert(hipMalloc(&d_x, h_x.size() * 4) == hipSuccess);
    assert(hipMalloc(&d_out_tm, (size_t)nt * rows * 4) == hipSuccess);
    assert(hipMalloc(&d_out_dm, (size_t)rows * 16 * 4) == hipSuccess);

    (void)hipMemcpy(d_w, h_w.data(), h_w.size() * 2, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_x, h_x.data(), h_x.size() * 4, hipMemcpyHostToDevice);
    quantize_bf16_to_w4_g128_hip(d_w, d_w4, d_sc, rows, cols, nullptr);
    assert(matvec_w4_multix_hip(d_w4, d_sc, d_x, d_out_tm, rows, cols, nt, rows, 1, nullptr));
    assert(matvec_w4_multix_hip(d_w4, d_sc, d_x, d_out_dm, rows, cols, nt, 16, 0, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> h_tm((size_t)nt * rows), h_dm((size_t)rows * 16);
    (void)hipMemcpy(h_tm.data(), d_out_tm, h_tm.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(h_dm.data(), d_out_dm, h_dm.size() * 4, hipMemcpyDeviceToHost);

    double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t t = 0; t < nt; ++t) {
        assert(h_tm[(size_t)t * rows + r] == h_dm[(size_t)r * 16 + t]);
        float ref = 0.0f;
        for (uint32_t c = 0; c < cols; ++c)
          ref += bf16_to_f32_cpu(h_w[(size_t)r * cols + c]) * h_x[(size_t)t * cols + c];
        const float got = h_tm[(size_t)t * rows + r];
        dot_ab += (double)ref * (double)got;
        norm_a += (double)ref * (double)ref;
        norm_b += (double)got * (double)got;
      }
    }
    const double cos_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
    assert(cos_sim > 0.996);
    for (void* p : {(void*)d_w, (void*)d_w4, (void*)d_sc, (void*)d_x, (void*)d_out_tm,
                    (void*)d_out_dm})
      (void)hipFree(p);
  }

  // 4. Test long prefill >65535-row hardware gridDim.y overflow guard (n = 70000, #1006)
  {
    const uint32_t n_tokens = 70000;
    const uint32_t n_embd = 64;
    std::vector<uint32_t> h_ids(n_tokens, 0u);
    h_ids[0] = 7u;
    h_ids[65535] = 42u;
    h_ids[69999] = 123u;

    const uint32_t vocab_size = 256;
    std::vector<uint16_t> h_table((size_t)vocab_size * n_embd);
    for (uint32_t v = 0; v < vocab_size; ++v) {
      for (uint32_t d = 0; d < n_embd; ++d) {
        h_table[(size_t)v * n_embd + d] = f32_to_bf16_cpu((float)(v * 100 + d) * 0.125f);
      }
    }

    uint16_t* d_table = nullptr;
    uint32_t* d_ids = nullptr;
    float* d_out = nullptr;
    assert(hipMalloc(&d_table, h_table.size() * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_ids, h_ids.size() * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_out, (size_t)n_tokens * n_embd * sizeof(float)) == hipSuccess);

    assert(hipMemcpy(d_table, h_table.data(), h_table.size() * sizeof(uint16_t),
                     hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(uint32_t),
                     hipMemcpyHostToDevice) == hipSuccess);

    embed_tokens_gather_hip(d_table, d_ids, d_out, n_tokens, n_embd, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);

    const uint32_t check_rows[] = {0u, 65535u, 69999u};
    for (uint32_t r : check_rows) {
      std::vector<float> h_row(n_embd);
      assert(hipMemcpy(h_row.data(), d_out + (size_t)r * n_embd, n_embd * sizeof(float),
                       hipMemcpyDeviceToHost) == hipSuccess);
      const uint32_t vid = h_ids[r];
      for (uint32_t d = 0; d < n_embd; ++d) {
        const float expected = bf16_to_f32_cpu(h_table[(size_t)vid * n_embd + d]);
        assert(std::fabs(h_row[d] - expected) < 1e-5f);
      }
    }

    for (void* p : {(void*)d_table, (void*)d_ids, (void*)d_out})
      (void)hipFree(p);
  }

  // 5. Test AFF_DRAFT_SKIP_MASKED DSpark ring-attention masked tile skip (#958)
  {
    const uint32_t width = 64, n_rot = 0, n_head = 4;
    const uint32_t ring = 64, n_raw = 16, n_comp = 32;
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t n_tokens = 4;

    std::vector<float> h_q((size_t)n_tokens * n_head * width);
    for (size_t i = 0; i < h_q.size(); ++i)
      h_q[i] = std::sin((float)i * 0.05f) * 0.2f;

    std::vector<float> h_raw((size_t)ring * width);
    for (size_t i = 0; i < h_raw.size(); ++i)
      h_raw[i] = std::cos((float)i * 0.07f) * 0.2f;

    std::vector<float> h_comp((size_t)n_comp * width);
    for (size_t i = 0; i < h_comp.size(); ++i)
      h_comp[i] = std::sin((float)i * 0.09f) * 0.2f;

    // Mask for compressed keys per token (n_tokens x n_comp)
    std::vector<uint8_t> h_allowed((size_t)n_tokens * n_comp, 0);
    // Token 0: partially unmasked (keys 0..7 active)
    for (uint32_t j = 0; j < 8; ++j) h_allowed[j] = 1;
    // Token 1: partially unmasked (keys 8..23 active)
    for (uint32_t j = 8; j < 24; ++j) h_allowed[1 * n_comp + j] = 1;
    // Token 2: all masked (dead row, #971 invariant: zeros, no NaNs)
    // (h_allowed remains 0 for token 2)
    // Token 3: fully unmasked
    for (uint32_t j = 0; j < n_comp; ++j) h_allowed[3 * n_comp + j] = 1;

    std::vector<uint32_t> h_ncomp_v = {n_comp, n_comp, n_comp, n_comp};

    float *d_q = nullptr, *d_raw = nullptr, *d_comp = nullptr;
    uint8_t* d_allowed = nullptr;
    uint32_t* d_ncomp_v = nullptr;
    float *d_out_skip = nullptr, *d_out_plain = nullptr;

    assert(hipMalloc(&d_q, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_raw, h_raw.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_comp, h_comp.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_allowed, h_allowed.size() * sizeof(uint8_t)) == hipSuccess);
    assert(hipMalloc(&d_ncomp_v, h_ncomp_v.size() * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_out_skip, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_out_plain, h_q.size() * sizeof(float)) == hipSuccess);

    assert(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_comp, h_comp.data(), h_comp.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_allowed, h_allowed.data(), h_allowed.size() * sizeof(uint8_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ncomp_v, h_ncomp_v.data(), h_ncomp_v.size() * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);

    const float scale = 1.0f / std::sqrt((float)width);

    // Run with AFF_DRAFT_SKIP_MASKED=0
    setenv("AFF_DRAFT_SKIP_MASKED", "0", 1);
    aff::mqa_attend_batch_hip(aff::KvDtype::F32, d_q, d_raw, d_comp, d_allowed, d_ncomp_v,
                              nullptr, d_out_plain, 0, n_tokens, n_tokens, 0, n_raw, ring,
                              n_comp, n_comp, n_head, width, n_rot, scale, nullptr, false,
                              n_raw, 0, 0);
    assert(hipDeviceSynchronize() == hipSuccess);

    // Run with AFF_DRAFT_SKIP_MASKED=1
    setenv("AFF_DRAFT_SKIP_MASKED", "1", 1);
    aff::mqa_attend_batch_hip(aff::KvDtype::F32, d_q, d_raw, d_comp, d_allowed, d_ncomp_v,
                              nullptr, d_out_skip, 0, n_tokens, n_tokens, 0, n_raw, ring,
                              n_comp, n_comp, n_head, width, n_rot, scale, nullptr, false,
                              n_raw, 0, 0);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> h_out_plain(h_q.size()), h_out_skip(h_q.size());
    assert(hipMemcpy(h_out_plain.data(), d_out_plain, h_q.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_out_skip.data(), d_out_skip, h_q.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_q.size(); ++i) {
      assert(!std::isnan(h_out_skip[i]));
      assert(!std::isinf(h_out_skip[i]));
      assert(!std::isnan(h_out_plain[i]));
      assert(!std::isinf(h_out_plain[i]));
      const float diff = std::fabs(h_out_skip[i] - h_out_plain[i]);
      if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-6f);

    for (void* p : {(void*)d_q, (void*)d_raw, (void*)d_comp, (void*)d_allowed, (void*)d_ncomp_v,
                    (void*)d_out_skip, (void*)d_out_plain})
      (void)hipFree(p);
  }

  // 6. Test AFF_ATTN_LOOKAHEAD RDNA4 decode K/V prefetch & Wave32 DPP reductions (#980, #992)
  {
    const uint32_t width = 64, n_rot = 0, n_head = 4;
    const std::vector<uint32_t> kv_lens = {64, 2049, 8192};

    std::vector<float> h_q((size_t)n_head * width);
    for (size_t i = 0; i < h_q.size(); ++i)
      h_q[i] = std::sin((float)i * 0.033f) * 0.25f;

    float* d_q = nullptr;
    float* d_out_ref = nullptr;
    float* d_out_lookahead = nullptr;
    assert(hipMalloc(&d_q, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_out_ref, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_out_lookahead, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

    for (uint32_t kv_len : kv_lens) {
      const uint32_t ring = kv_len + 16;
      std::vector<float> h_raw((size_t)ring * width);
      for (size_t i = 0; i < h_raw.size(); ++i)
        h_raw[i] = std::cos((float)i * 0.017f) * 0.2f;

      float* d_raw = nullptr;
      assert(hipMalloc(&d_raw, h_raw.size() * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

      const float scale = 1.0f / std::sqrt((float)width);

      // Run with AFF_ATTN_LOOKAHEAD=0 (untouched baseline path)
      setenv("AFF_ATTN_LOOKAHEAD", "0", 1);
      aff::mqa_attend_hip(aff::KvDtype::F32, d_q, d_raw, nullptr, nullptr, nullptr,
                          d_out_ref, n_head, 0, kv_len, ring, 0, width, n_rot, scale, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);

      // Run with AFF_ATTN_LOOKAHEAD=1 (lookahead + Wave32 DPP reductions)
      setenv("AFF_ATTN_LOOKAHEAD", "1", 1);
      aff::mqa_attend_hip(aff::KvDtype::F32, d_q, d_raw, nullptr, nullptr, nullptr,
                          d_out_lookahead, n_head, 0, kv_len, ring, 0, width, n_rot, scale, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<float> h_ref(h_q.size()), h_lookahead(h_q.size());
      assert(hipMemcpy(h_ref.data(), d_out_ref, h_q.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_lookahead.data(), d_out_lookahead, h_q.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      float max_diff = 0.0f;
      for (size_t i = 0; i < h_q.size(); ++i) {
        assert(!std::isnan(h_ref[i]));
        assert(!std::isinf(h_ref[i]));
        assert(!std::isnan(h_lookahead[i]));
        assert(!std::isinf(h_lookahead[i]));
        const float diff = std::fabs(h_lookahead[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
      }
      assert(max_diff < 1e-4f);

      (void)hipFree(d_raw);
    }

    for (void* p : {(void*)d_q, (void*)d_out_ref, (void*)d_out_lookahead})
      (void)hipFree(p);
  }

  // 7. Test MoE multi-row shared-expert deduplication and stream join invariants (#983, #984)
  {
    const uint32_t nb_tok = 16;
    const uint32_t k = 4;
    const uint32_t n_expert = 64;
    const uint32_t taken = nb_tok * k;
    const uint32_t cap = aff::moe_plan_tile_cap(nb_tok, k, n_expert);
    const uint32_t pref_cap = 16;

    std::vector<uint32_t> h_sel(taken);
    std::vector<float> h_wt(taken, 0.25f);
    std::vector<uint64_t> h_plc(n_expert, 0ull);

    // Expert 0 is resident (bit 0 = 1)
    h_plc[0] = 0x10000000ull | 1ull;
    // Experts 1, 2, 3, 4 are pooled (bit 0 = 0, non-null address)
    for (uint32_t e = 1; e <= 4; ++e) {
      h_plc[e] = 0x20000000ull + (uint64_t)e * 0x100000ull;
    }

    // All 16 tokens select the exact same 4 experts: expert 0 (resident), experts 1, 2, 3 (pooled)
    for (uint32_t t = 0; t < nb_tok; ++t) {
      h_sel[t * k + 0] = 0; // resident
      h_sel[t * k + 1] = 1; // pooled
      h_sel[t * k + 2] = 2; // pooled
      h_sel[t * k + 3] = 3; // pooled
    }

    uint32_t* d_sel = nullptr;
    float* d_wt = nullptr;
    uint64_t* d_plc = nullptr;
    uint32_t* d_tok = nullptr;
    float* d_wt_out = nullptr;
    uint32_t* d_ord = nullptr;
    uint32_t* d_run = nullptr;
    uint32_t* d_slot_col = nullptr;
    aff::ExpertTile* d_tiles = nullptr;
    uint32_t* d_counts = nullptr;
    aff::MoePlanStatus* d_status = nullptr;
    uint16_t* d_bg = nullptr;
    uint16_t* d_bu = nullptr;
    uint16_t* d_bydn = nullptr;
    uint8_t* d_pref_slab = nullptr;
    const uint8_t** d_pref_src = nullptr;

    assert(hipMalloc(&d_sel, h_sel.size() * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wt, h_wt.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_plc, h_plc.size() * sizeof(uint64_t)) == hipSuccess);
    assert(hipMalloc(&d_tok, 2 * taken * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wt_out, taken * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_ord, taken * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_run, (nb_tok + 1) * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_slot_col, taken * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_tiles, 3 * 8 * cap * sizeof(aff::ExpertTile)) == hipSuccess);
    assert(hipMalloc(&d_counts, 8 * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_status, sizeof(aff::MoePlanStatus)) == hipSuccess);
    assert(hipMalloc(&d_bg, taken * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_bu, taken * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_bydn, taken * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_pref_slab, pref_cap * 1024) == hipSuccess);
    assert(hipMalloc(&d_pref_src, pref_cap * sizeof(uint8_t*)) == hipSuccess);

    assert(hipMemcpy(d_sel, h_sel.data(), h_sel.size() * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_wt, h_wt.data(), h_wt.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_plc, h_plc.data(), h_plc.size() * sizeof(uint64_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemset(d_status, 0, sizeof(aff::MoePlanStatus)) == hipSuccess);
    assert(hipMemset(d_pref_src, 0, pref_cap * sizeof(uint8_t*)) == hipSuccess);

    aff::MoePlanArgs A;
    A.sel = d_sel;
    A.wt = d_wt;
    A.plc = d_plc;
    A.layer = 0;
    A.nb_tok = nb_tok;
    A.k = k;
    A.n_expert = n_expert;
    A.tok = d_tok;
    A.wt_out = d_wt_out;
    A.ord = d_ord;
    A.run = d_run;
    A.slot_col = d_slot_col;
    A.tiles = d_tiles;
    A.counts = d_counts;
    A.status = d_status;
    A.act = nullptr;
    A.shard.stride = 1024;
    A.bg = d_bg;
    A.bu = d_bu;
    A.bydn = d_bydn;
    A.tile_cap = cap;
    A.pref_slab = d_pref_slab;
    A.pref_src = d_pref_src;
    A.pref_cap = pref_cap;

    bool plan_ok = aff::moe_plan_hip(A, nullptr);
    assert(plan_ok);
    assert(hipDeviceSynchronize() == hipSuccess);

    aff::MoePlanStatus h_status{};
    assert(hipMemcpy(&h_status, d_status, sizeof(aff::MoePlanStatus), hipMemcpyDeviceToHost) == hipSuccess);
    assert(h_status.bad == 0);
    assert(h_status.over == 0);
    assert(h_status.taken == taken);
    assert(h_status.n_res == 1);   // Only expert 0 is resident
    assert(h_status.n_uniq == 4);  // 1 resident + 3 unique pooled experts

    std::vector<const uint8_t*> h_pref_src(pref_cap, nullptr);
    assert(hipMemcpy(h_pref_src.data(), d_pref_src, pref_cap * sizeof(uint8_t*), hipMemcpyDeviceToHost) == hipSuccess);

    // Verify #984 invariant: exactly 1 unique read entry per pooled expert in pref_src
    const uint32_t pooled_count = h_status.n_uniq - h_status.n_res;
    assert(pooled_count == 3);
    for (uint32_t i = 0; i < pooled_count; ++i) {
      assert(h_pref_src[i] != nullptr);
      const uint64_t expected_addr = 0x20000000ull + (uint64_t)(i + 1) * 0x100000ull;
      assert((uint64_t)(uintptr_t)h_pref_src[i] == expected_addr);
    }
    // Verify remaining pref_src slots were not overwritten
    for (uint32_t i = pooled_count; i < pref_cap; ++i) {
      assert(h_pref_src[i] == nullptr);
    }

    // Verify AFF_MOE_SINGLE_JOIN runtime toggle safety
    setenv("AFF_MOE_SINGLE_JOIN", "1", 1);
    const char* env_val = std::getenv("AFF_MOE_SINGLE_JOIN");
    assert(env_val && std::strcmp(env_val, "1") == 0);
    setenv("AFF_MOE_SINGLE_JOIN", "0", 1);

    for (void* p : {(void*)d_sel, (void*)d_wt, (void*)d_plc, (void*)d_tok, (void*)d_wt_out,
                    (void*)d_ord, (void*)d_run, (void*)d_slot_col, (void*)d_tiles,
                    (void*)d_counts, (void*)d_status, (void*)d_bg, (void*)d_bu,
                    (void*)d_bydn, (void*)d_pref_slab, (void*)d_pref_src})
      (void)hipFree(p);
  }

  // 8. Test multi-token sparse indexer candidate stride & <=k invalid-key guard
  {
    const uint32_t nt = 6, n_keys = 1024, block = 8, nb_blocks = n_keys / block;
    const uint32_t topk_blocks = 16, topk = 64, adm_stride = 128;
    std::vector<uint32_t> h_nk = {900, 960, 1000, 1024, 912, 999};
    std::vector<uint32_t> h_nkb(nt);
    for (uint32_t t = 0; t < nt; ++t) h_nkb[t] = (h_nk[t] + block - 1u) / block;

    std::vector<float> h_sc_src((size_t)nt * n_keys, -INFINITY);
    std::vector<float> h_sc_tgt((size_t)nt * n_keys, -INFINITY);
    for (uint32_t t = 0; t < nt; ++t) {
      for (uint32_t s = 0; s < h_nk[t]; ++s) {
        const uint32_t b = s / block;
        // Distinct winning candidate blocks per token plus lower-ranked blocks with huge target
        // scores so any OOB stride mismatch immediately admits wrong indices.
        const bool hot_src = ((b + t * 5u) % 7u) == 0u;
        h_sc_src[(size_t)t * n_keys + s] =
            (s % 13u == 0u) ? -INFINITY : (hot_src ? (50.0f - (float)b * 0.1f) : 0.25f);
        h_sc_tgt[(size_t)t * n_keys + s] =
            hot_src ? (10.0f + std::sin((float)(s + t * 19u)) * 2.0f) : (200.0f - (float)s * 0.01f);
      }
    }

    // Exact CPU reference for two-level candidate filtering
    std::vector<std::vector<uint32_t>> ref_adm(nt);
    for (uint32_t t = 0; t < nt; ++t) {
      const uint32_t nk = h_nk[t], nblk = h_nkb[t];
      std::vector<std::pair<float, uint32_t>> bmax(nblk);
      for (uint32_t b = 0; b < nblk; ++b) {
        float m = -INFINITY;
        const uint32_t lo = b * block, hi = std::min(lo + block, nk);
        for (uint32_t s = lo; s < hi; ++s) m = std::fmax(m, h_sc_src[(size_t)t * n_keys + s]);
        if (b == (nk - 1u) / block) m = INFINITY;
        bmax[b] = {m, b};
      }
      std::stable_sort(bmax.begin(), bmax.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
      });
      std::vector<uint8_t> cand_blk(nblk, 0u);
      for (uint32_t i = 0; i < std::min(topk_blocks, nblk); ++i) {
        if (bmax[i].first > -INFINITY) cand_blk[bmax[i].second] = 1u;
      }
      std::vector<std::pair<float, uint32_t>> valid;
      for (uint32_t s = 0; s < nk; ++s) {
        if (!cand_blk[s / block]) continue;
        const float v = h_sc_tgt[(size_t)t * n_keys + s];
        if (v > -INFINITY) valid.push_back({v, s});
      }
      std::stable_sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
      });
      const uint32_t take = std::min<uint32_t>(topk, (uint32_t)valid.size());
      for (uint32_t i = 0; i < take; ++i) ref_adm[t].push_back(valid[i].second);
      std::sort(ref_adm[t].begin(), ref_adm[t].end());
    }

    float *d_sc_src = nullptr, *d_sc_tgt = nullptr, *d_bsc = nullptr;
    uint8_t *d_cand = nullptr, *d_mask = nullptr;
    uint32_t *d_nk = nullptr, *d_nkb = nullptr, *d_adm = nullptr, *d_admn = nullptr;
    assert(hipMalloc(&d_sc_src, (size_t)nt * n_keys * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_sc_tgt, (size_t)nt * n_keys * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_bsc, (size_t)nt * nb_blocks * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_cand, (size_t)nt * nb_blocks) == hipSuccess);
    assert(hipMalloc(&d_mask, (size_t)nt * n_keys) == hipSuccess);
    assert(hipMalloc(&d_nk, nt * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_nkb, nt * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_adm, (size_t)nt * adm_stride * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_admn, nt * sizeof(uint32_t)) == hipSuccess);

    assert(hipMemcpy(d_sc_src, h_sc_src.data(), (size_t)nt * n_keys * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_sc_tgt, h_sc_tgt.data(), (size_t)nt * n_keys * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_nk, h_nk.data(), nt * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_nkb, h_nkb.data(), nt * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemset(d_cand, 0, (size_t)nt * nb_blocks) == hipSuccess);
    assert(hipMemset(d_mask, 0, (size_t)nt * n_keys) == hipSuccess);

    aff::idx_block_max_hip(d_sc_src, d_bsc, d_nk, nt, n_keys, nb_blocks, nb_blocks, block, n_keys, nullptr);
    aff::idx_topk_mask_hip(d_bsc, d_cand, d_nkb, nt, nb_blocks, nb_blocks, nb_blocks, topk_blocks, nullptr);
    aff::idx_apply_cand_hip(d_sc_tgt, d_cand, d_nk, nt, n_keys, nb_blocks, n_keys, block, n_keys, nullptr);
    aff::idx_topk_mask_hip(d_sc_tgt, d_mask, d_nk, nt, n_keys, n_keys, n_keys, topk, nullptr,
                           nullptr, d_adm, d_admn, adm_stride);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<uint8_t> h_mask((size_t)nt * n_keys);
    std::vector<uint32_t> h_adm((size_t)nt * adm_stride), h_admn(nt);
    assert(hipMemcpy(h_mask.data(), d_mask, h_mask.size(), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_adm.data(), d_adm, h_adm.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_admn.data(), d_admn, nt * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

    for (uint32_t t = 0; t < nt; ++t) {
      assert(h_admn[t] == (uint32_t)ref_adm[t].size());
      for (uint32_t i = 0; i < h_admn[t]; ++i) {
        assert(h_adm[(size_t)t * adm_stride + i] == ref_adm[t][i]);
        assert(h_mask[(size_t)t * n_keys + ref_adm[t][i]] == 1u);
      }
      uint32_t cnt = 0;
      for (uint32_t s = 0; s < n_keys; ++s) cnt += h_mask[(size_t)t * n_keys + s] ? 1u : 0u;
      assert(cnt == h_admn[t]);
    }
    for (void* p : {(void*)d_sc_src, (void*)d_sc_tgt, (void*)d_bsc, (void*)d_cand, (void*)d_mask,
                    (void*)d_nk, (void*)d_nkb, (void*)d_adm, (void*)d_admn})
      (void)hipFree(p);

    // Spread TopkScratch path with n_keys <= k and -INFINITY entries
    const uint32_t sp_nb = 2, sp_stride = 4096, sp_k = 512, sp_adm_stride = 512;
    std::vector<uint32_t> h_sp_nk = {300, 512};
    std::vector<float> h_sp_sc((size_t)sp_nb * sp_stride, -INFINITY);
    std::vector<std::vector<uint32_t>> ref_sp(sp_nb);
    for (uint32_t t = 0; t < sp_nb; ++t) {
      for (uint32_t s = 0; s < h_sp_nk[t]; ++s) {
        if ((s + t) % 3u != 0u) {
          h_sp_sc[(size_t)t * sp_stride + s] = 1.0f + (float)s * 0.01f;
          ref_sp[t].push_back(s);
        }
      }
    }
    uint64_t state_b = 0, tie_b = 0;
    const uint64_t hist_b = aff::idx_topk_scratch_bytes(4u, &state_b, &tie_b);
    aff::TopkScratch sp{};
    sp.tok_cap = 4u;
    float* d_sp_sc = nullptr;
    uint8_t* d_sp_mask = nullptr;
    uint32_t *d_sp_nk = nullptr, *d_sp_adm = nullptr, *d_sp_admn = nullptr;
    assert(hipMalloc(&sp.hist, hist_b) == hipSuccess);
    assert(hipMalloc(&sp.state, state_b) == hipSuccess);
    assert(hipMalloc(&sp.tie, tie_b) == hipSuccess);
    assert(hipMemset(sp.hist, 0, hist_b) == hipSuccess);
    assert(hipMemset(sp.state, 0, state_b) == hipSuccess);
    assert(hipMemset(sp.tie, 0, tie_b) == hipSuccess);
    assert(hipMalloc(&d_sp_sc, (size_t)sp_nb * sp_stride * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_sp_mask, (size_t)sp_nb * sp_stride) == hipSuccess);
    assert(hipMalloc(&d_sp_nk, sp_nb * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_sp_adm, (size_t)sp_nb * sp_adm_stride * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_sp_admn, sp_nb * sizeof(uint32_t)) == hipSuccess);

    assert(hipMemcpy(d_sp_sc, h_sp_sc.data(), (size_t)sp_nb * sp_stride * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_sp_nk, h_sp_nk.data(), sp_nb * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemset(d_sp_mask, 0, (size_t)sp_nb * sp_stride) == hipSuccess);
    assert(hipMemset(d_sp_admn, 0, sp_nb * sizeof(uint32_t)) == hipSuccess);

    aff::idx_topk_mask_hip(d_sp_sc, d_sp_mask, d_sp_nk, sp_nb, sp_stride, sp_stride, sp_stride,
                           sp_k, nullptr, &sp, d_sp_adm, d_sp_admn, sp_adm_stride);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<uint8_t> h_sp_mask((size_t)sp_nb * sp_stride);
    std::vector<uint32_t> h_sp_adm((size_t)sp_nb * sp_adm_stride), h_sp_admn(sp_nb);
    assert(hipMemcpy(h_sp_mask.data(), d_sp_mask, h_sp_mask.size(), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_sp_adm.data(), d_sp_adm, h_sp_adm.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_sp_admn.data(), d_sp_admn, sp_nb * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

    for (uint32_t t = 0; t < sp_nb; ++t) {
      assert(h_sp_admn[t] == (uint32_t)ref_sp[t].size());
      for (uint32_t i = 0; i < h_sp_admn[t]; ++i) {
        assert(h_sp_adm[(size_t)t * sp_adm_stride + i] == ref_sp[t][i]);
      }
      for (uint32_t s = 0; s < sp_stride; ++s) {
        const uint8_t want = (s < h_sp_nk[t] && h_sp_sc[(size_t)t * sp_stride + s] > -INFINITY) ? 1u : 0u;
        assert(h_sp_mask[(size_t)t * sp_stride + s] == want);
      }
    }
    for (void* p : {(void*)sp.hist, (void*)sp.state, (void*)sp.tie, (void*)d_sp_sc,
                    (void*)d_sp_mask, (void*)d_sp_nk, (void*)d_sp_adm, (void*)d_sp_admn})
      (void)hipFree(p);
  }

  // 9. Test kFKNarrow=128 / kFKNarrowWide=256 decode WMMA specializations vs reference BF16 matmul
  {
    const uint32_t nb = 6;
    const float kWVals[] = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f};
    const uint8_t kWCode[] = {0x00u, 0x30u, 0xb0u, 0x38u, 0xb8u, 0x3cu, 0xbcu};
    // Max 3.5f gives dynamic FP8 scale 3.5/448 = 1/128, so every element in kXVals maps exactly to E4M3.
    const float kXVals[] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.5f};
    const uint8_t kSwCode[] = {0x7eu, 0x7fu, 0x80u}; // UE8M0 scales: 0.5, 1.0, 2.0
    const float kSwVal[] = {0.5f, 1.0f, 2.0f};

    for (uint32_t cols : {1152u, 1280u}) {
      const uint32_t sc_cols = (cols + 127u) / 128u;
      std::vector<float> h_x_dm((size_t)cols * nb); // dim-major [cols][nb]
      for (uint32_t c = 0; c < cols; ++c) {
        for (uint32_t b = 0; b < nb; ++b) {
          h_x_dm[(size_t)c * nb + b] = kXVals[(c * 13u + b * 7u) % 8u];
        }
      }

      uint8_t* d_x8 = nullptr;
      float *d_x = nullptr, *d_xs = nullptr;
      assert(hipMalloc(&d_x, h_x_dm.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_x8, (size_t)nb * cols) == hipSuccess);
      assert(hipMalloc(&d_xs, (size_t)nb * sc_cols * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_x, h_x_dm.data(), h_x_dm.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      batch_quant_fp8_hip(d_x, d_x8, d_xs, cols, nb, 0u, nullptr);

      for (uint32_t rows : {80u, 256u, 512u}) {
        const uint32_t sw_rows = (rows + 127u) / 128u;
        std::vector<uint8_t> h_sw((size_t)sw_rows * sc_cols);
        for (uint32_t sr = 0; sr < sw_rows; ++sr) {
          for (uint32_t sc = 0; sc < sc_cols; ++sc) {
            h_sw[(size_t)sr * sc_cols + sc] = kSwCode[(sr * 3u + sc) % 3u];
          }
        }

        std::vector<uint8_t> h_w8((size_t)rows * cols);
        std::vector<uint16_t> h_w16((size_t)rows * cols);
        std::vector<float> h_wf32((size_t)rows * cols);
        for (uint32_t r = 0; r < rows; ++r) {
          for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t idx = (r * 17u + c * 31u + 5u) % 7u;
            const uint32_t sidx = ((r >> 7) * 3u + (c >> 7)) % 3u;
            const float w_scaled = kWVals[idx] * kSwVal[sidx];
            h_w8[(size_t)r * cols + c] = kWCode[idx];
            h_w16[(size_t)r * cols + c] = f32_to_bf16_cpu(w_scaled);
            h_wf32[(size_t)r * cols + c] = w_scaled;
          }
        }

        uint8_t *d_w8 = nullptr, *d_w8_ps = nullptr, *d_sw = nullptr;
        uint16_t* d_w16 = nullptr;
        float *d_y_bf16 = nullptr, *d_y_rm = nullptr, *d_y_ps = nullptr;
        assert(hipMalloc(&d_w8, h_w8.size()) == hipSuccess);
        assert(hipMalloc(&d_w8_ps, h_w8.size()) == hipSuccess);
        assert(hipMalloc(&d_sw, h_sw.size()) == hipSuccess);
        assert(hipMalloc(&d_w16, h_w16.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_y_bf16, (size_t)rows * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_rm, (size_t)rows * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_ps, (size_t)rows * nb * sizeof(float)) == hipSuccess);

        assert(hipMemcpy(d_w8, h_w8.data(), h_w8.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw, h_sw.data(), h_sw.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_w16, h_w16.data(), h_w16.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);

        preshuffle_fp8_hip(d_w8, d_w8_ps, rows, cols, nullptr);
        batch_gemm_hip(AFF_BF16, d_w16, nullptr, d_x, d_y_bf16, rows, cols, nb, nullptr);
        batch_gemm_fp8_hip(d_w8, d_sw, d_x8, d_xs, d_y_rm, rows, cols, nb, 0u, 0u, nullptr,
                           /*preshuffled=*/false, 0u, 0u, /*accum=*/false, 0u);
        batch_gemm_fp8_hip(d_w8_ps, d_sw, d_x8, d_xs, d_y_ps, rows, cols, nb, 0u, 0u, nullptr,
                           /*preshuffled=*/true, 0u, 0u, /*accum=*/false, 0u);
        assert(hipDeviceSynchronize() == hipSuccess);

        std::vector<float> h_bf16((size_t)rows * nb), h_rm((size_t)rows * nb), h_ps((size_t)rows * nb);
        assert(hipMemcpy(h_bf16.data(), d_y_bf16, h_bf16.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(h_rm.data(), d_y_rm, h_rm.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(h_ps.data(), d_y_ps, h_ps.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

        std::vector<float> h_init((size_t)rows * nb);
        for (uint32_t r = 0; r < rows; ++r) {
          for (uint32_t b = 0; b < nb; ++b) {
            h_init[(size_t)r * nb + b] = 0.25f * (float)((r + b) % 5u + 1u);
          }
        }
        assert(hipMemcpy(d_y_rm, h_init.data(), h_init.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_y_ps, h_init.data(), h_init.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        batch_gemm_fp8_hip(d_w8, d_sw, d_x8, d_xs, d_y_rm, rows, cols, nb, 0u, 0u, nullptr,
                           /*preshuffled=*/false, 0u, 0u, /*accum=*/true, 0u);
        batch_gemm_fp8_hip(d_w8_ps, d_sw, d_x8, d_xs, d_y_ps, rows, cols, nb, 0u, 0u, nullptr,
                           /*preshuffled=*/true, 0u, 0u, /*accum=*/true, 0u);
        assert(hipDeviceSynchronize() == hipSuccess);

        std::vector<float> h_rm_acc((size_t)rows * nb), h_ps_acc((size_t)rows * nb);
        assert(hipMemcpy(h_rm_acc.data(), d_y_rm, h_rm_acc.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(h_ps_acc.data(), d_y_ps, h_ps_acc.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

        for (uint32_t r = 0; r < rows; ++r) {
          for (uint32_t b = 0; b < nb; ++b) {
            float cpu_acc = 0.0f;
            for (uint32_t c = 0; c < cols; ++c)
              cpu_acc += h_wf32[(size_t)r * cols + c] * h_x_dm[(size_t)c * nb + b];
            const size_t idx = (size_t)r * nb + b;
            assert(!std::isnan(h_rm[idx]) && !std::isnan(h_ps[idx]));
            assert(std::fabs(h_rm[idx] - cpu_acc) < 1e-3f);
            assert(std::fabs(h_ps[idx] - cpu_acc) < 1e-3f);
            assert(std::fabs(h_rm[idx] - h_bf16[idx]) < 1e-3f);
            assert(std::fabs(h_ps[idx] - h_bf16[idx]) < 1e-3f);
            assert(std::fabs(h_rm_acc[idx] - (h_init[idx] + cpu_acc)) < 1e-3f);
            assert(std::fabs(h_ps_acc[idx] - (h_init[idx] + cpu_acc)) < 1e-3f);
          }
        }

        for (void* p : {(void*)d_w8, (void*)d_w8_ps, (void*)d_sw, (void*)d_w16, (void*)d_y_bf16,
                        (void*)d_y_rm, (void*)d_y_ps})
          (void)hipFree(p);
      }

      for (void* p : {(void*)d_x, (void*)d_x8, (void*)d_xs})
        (void)hipFree(p);
    }
  }

  // 10. Test batch_attn_out_prep_hip (AFF_FUSED_ATTN_OUT_PREP=1) vs unfused pipeline
  {
    const uint32_t n_head = 32, head_dim = 512, n_rot = 64, pos0 = 37;
    const uint32_t n = n_head * head_dim;
    const RopeDerived rope{1.0f, std::pow(10000.0f, -2.0f / (float)n_rot), 1.0f, 1.0f, 1.0f};

    for (uint32_t nb : {6u, 130u}) {
      std::vector<float> h_src((size_t)nb * n);
      for (size_t i = 0; i < h_src.size(); ++i)
        h_src[i] = std::sin((float)(i % 997u) * 0.031f) * 2.5f + std::cos((float)(i % 251u) * 0.07f);

      float* d_src = nullptr;
      uint16_t *d_unfused = nullptr, *d_fused = nullptr;
      assert(hipMalloc(&d_src, h_src.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_unfused, h_src.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_fused, h_src.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

      for (bool o_qdq : {false, true}) {
        assert(hipMemset(d_unfused, 0, h_src.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMemset(d_fused, 0, h_src.size() * sizeof(uint16_t)) == hipSuccess);

        batch_transpose_bf16_hip(d_src, d_unfused, n, nb, 0, nullptr);
        batch_rope_tail_bf16_hip(d_unfused, n_head, head_dim, n_rot, pos0, rope, -1.0f, nb, nullptr);
        if (o_qdq) batch_o_fp8_qdq_bf16_hip(d_unfused, n / 32u, nb, nullptr);

        const bool ok = batch_attn_out_prep_hip(d_src, d_fused, n_head, head_dim, n_rot, pos0, rope,
                                                -1.0f, nb, o_qdq, nullptr);
        assert(ok);
        assert(hipDeviceSynchronize() == hipSuccess);

        std::vector<uint16_t> h_u(h_src.size()), h_f(h_src.size());
        assert(hipMemcpy(h_u.data(), d_unfused, h_u.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(h_f.data(), d_fused, h_f.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);

        float max_diff = 0.0f;
        for (size_t i = 0; i < h_u.size(); ++i) {
          const float u = bf16_to_f32_cpu(h_u[i]);
          const float f = bf16_to_f32_cpu(h_f[i]);
          assert(!std::isnan(u) && !std::isnan(f));
          const float diff = std::fabs(u - f);
          if (diff > max_diff) max_diff = diff;
        }
        assert(max_diff < 1e-3f);
      }

      for (void* p : {(void*)d_src, (void*)d_unfused, (void*)d_fused})
        (void)hipFree(p);
    }
  }

  // 11. Test o_fp8_qdq_f32_hip & batch_o_fp8_qdq_bf16_hip reciprocal path across edge exponents
  {
    auto qdq_scale_ref = [](float amax) -> float {
      float t = amax * (1.0f / 448.0f);
      uint32_t tbits;
      std::memcpy(&tbits, &t, 4);
      int e = (int)((tbits >> 23) & 0xFF) - 127 + ((tbits & 0x7FFFFFu) != 0 ? 1 : 0);
      uint32_t sbits = (uint32_t)(e + 127) << 23;
      float s;
      std::memcpy(&s, &sbits, 4);
      return s;
    };
    const float edge_vals[] = {0.0f, -0.0f, 1e-6f, -1e-5f, 1e-4f, 0.015625f, -0.5f, 1.0f,
                               -16.0f, 447.5f, 448.0f, -448.0f, 1024.0f, -2048.0f, 32768.0f, -65024.0f};
    const uint32_t dim = 256u;
    std::vector<float> h_f32(dim), h_f32_ref(dim);
    for (uint32_t i = 0; i < dim; ++i) {
      const float base = edge_vals[i % 16u];
      const float scale_blk = (i < 32u) ? 1e-3f : (i < 64u) ? 1.0f : (i < 128u) ? 16.0f : 64.0f;
      h_f32[i] = base * (scale_blk / 64.0f);
    }
    h_f32_ref = h_f32;
    for (uint32_t d0 = 0; d0 < dim; d0 += 32u) {
      float amax = 1e-4f;
      for (uint32_t i = 0; i < 32u; ++i) amax = std::max(amax, std::fabs(h_f32_ref[d0 + i]));
      const float s = qdq_scale_ref(amax);
      for (uint32_t i = 0; i < 32u; ++i) {
        const float v = std::min(448.0f, std::max(-448.0f, h_f32_ref[d0 + i] / s));
        h_f32_ref[d0 + i] = e4m3_round_ref(v) * s;
      }
    }
    float* d_f32 = nullptr;
    assert(hipMalloc(&d_f32, dim * sizeof(float)) == hipSuccess);
    assert(hipMemcpy(d_f32, h_f32.data(), dim * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    o_fp8_qdq_f32_hip(d_f32, dim / 32u, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<float> h_f32_got(dim);
    assert(hipMemcpy(h_f32_got.data(), d_f32, dim * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    (void)hipFree(d_f32);
    for (uint32_t i = 0; i < dim; ++i) {
      assert(std::fabs(h_f32_got[i] - h_f32_ref[i]) <= 1e-6f * std::max(1.0f, std::fabs(h_f32_ref[i])));
    }

    for (uint32_t nb : {1u, 6u, 32u}) {
      std::vector<uint16_t> h_bf((size_t)dim * nb);
      for (size_t i = 0; i < h_bf.size(); ++i)
        h_bf[i] = f32_to_bf16_cpu(edge_vals[i % 16u] * (0.25f + 0.0625f * (float)((i / 32u) % 7u)));
      std::vector<uint16_t> h_bf_ref = h_bf;
      o_fp8_qdq_block32_ref(h_bf_ref.data(), dim, nb);
      uint16_t* d_bf = nullptr;
      assert(hipMalloc(&d_bf, h_bf.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMemcpy(d_bf, h_bf.data(), h_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      batch_o_fp8_qdq_bf16_hip(d_bf, dim / 32u, nb, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      std::vector<uint16_t> h_bf_got(h_bf.size());
      assert(hipMemcpy(h_bf_got.data(), d_bf, h_bf.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
      (void)hipFree(d_bf);
      for (size_t i = 0; i < h_bf_ref.size(); ++i) assert(h_bf_got[i] == h_bf_ref[i]);
    }
  }

  // 12. Test engram_gate_hip parity at nb = 1, 6, 64 (both f32 and bf16 hc)
  {
    const uint32_t n_embd = 256u, n_hc = 4u;
    const float eps = 1e-6f;
    for (uint32_t nb : {1u, 6u, 64u}) {
      const uint32_t hs = (nb == 6u) ? 8u : nb;
      std::vector<float> h_kv((size_t)(n_hc + 1u) * n_embd * nb);
      std::vector<float> h_w((size_t)n_hc * n_embd);
      for (size_t i = 0; i < h_kv.size(); ++i) h_kv[i] = std::sin((float)i * 0.031f) * 1.5f;
      for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = std::cos((float)i * 0.017f) * 0.7f;

      float *d_kv = nullptr, *d_w = nullptr;
      assert(hipMalloc(&d_kv, h_kv.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_w, h_w.size() * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_kv, h_kv.data(), h_kv.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

      for (bool bf16 : {false, true}) {
        std::vector<float> h_hc_f((size_t)n_hc * n_embd * hs, 0.0f);
        std::vector<uint16_t> h_hc_b((size_t)n_hc * n_embd * hs, 0u);
        for (size_t band_i = 0; band_i < (size_t)n_hc * n_embd; ++band_i) {
          for (uint32_t t = 0; t < nb; ++t) {
            const float val = std::sin((float)(band_i * 7u + t * 3u) * 0.019f) * 2.0f;
            const uint16_t bv = f32_to_bf16_cpu(val);
            h_hc_b[band_i * hs + t] = bv;
            h_hc_f[band_i * hs + t] = bf16 ? bf16_to_f32_cpu(bv) : val;
          }
        }
        std::vector<float> h_ref = h_hc_f;
        const size_t vband = (size_t)n_hc * n_embd;
        for (uint32_t t = 0; t < nb; ++t) {
          for (uint32_t c = 0; c < n_hc; ++c) {
            const size_t band = (size_t)c * n_embd;
            double sh = 0.0, sk = 0.0, dot = 0.0;
            for (uint32_t i = 0; i < n_embd; ++i) {
              const double h = h_hc_f[(band + i) * hs + t];
              const double k = h_kv[(band + i) * nb + t];
              sh += h * h;
              sk += k * k;
              dot += h * (double)h_w[band + i] * k;
            }
            const double rstd =
                (1.0 / std::sqrt(sh / n_embd + eps)) * (1.0 / std::sqrt(sk / n_embd + eps));
            const double d = dot * rstd / std::sqrt((double)n_embd);
            const double s = std::copysign(std::sqrt(std::max(std::fabs(d), 1e-6)), d);
            const double gate = 1.0 / (1.0 + std::exp(-s));
            for (uint32_t i = 0; i < n_embd; ++i) {
              const size_t off = (band + i) * hs + t;
              const float nv = (float)(h_hc_f[off] + gate * (double)h_kv[(vband + i) * nb + t]);
              h_ref[off] = bf16 ? bf16_to_f32_cpu(f32_to_bf16_cpu(nv)) : nv;
            }
          }
        }

        void* d_hc = nullptr;
        const size_t hc_bytes = (size_t)n_hc * n_embd * hs * (bf16 ? 2u : 4u);
        assert(hipMalloc(&d_hc, hc_bytes) == hipSuccess);
        assert(hipMemcpy(d_hc, bf16 ? (const void*)h_hc_b.data() : (const void*)h_hc_f.data(),
                         hc_bytes, hipMemcpyHostToDevice) == hipSuccess);
        engram_gate_hip(d_hc, bf16, d_kv, d_w, n_embd, n_hc, nb, hs, eps, nullptr);
        assert(hipDeviceSynchronize() == hipSuccess);

        float max_diff = 0.0f;
        if (bf16) {
          std::vector<uint16_t> got(h_hc_b.size());
          assert(hipMemcpy(got.data(), d_hc, hc_bytes, hipMemcpyDeviceToHost) == hipSuccess);
          for (size_t i = 0; i < got.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(bf16_to_f32_cpu(got[i]) - h_ref[i]));
          assert(max_diff <= 2e-2f);
        } else {
          std::vector<float> got(h_hc_f.size());
          assert(hipMemcpy(got.data(), d_hc, hc_bytes, hipMemcpyDeviceToHost) == hipSuccess);
          for (size_t i = 0; i < got.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(got[i] - h_ref[i]));
          assert(max_diff <= 1e-5f);
        }
        (void)hipFree(d_hc);
      }
      (void)hipFree(d_kv);
      (void)hipFree(d_w);
    }
  }

  // 13. Test tp_ar_fp8_kernel (unstaged & >= 4 MiB staged, mixed Exact/Fp8, alias, reset) at TP=2
  {
    int ndev = 0, can01 = 0, can10 = 0;
    if (hipGetDeviceCount(&ndev) == hipSuccess && ndev >= 2 &&
        hipDeviceCanAccessPeer(&can01, 0, 1) == hipSuccess && can01 &&
        hipDeviceCanAccessPeer(&can10, 1, 0) == hipSuccess && can10) {
      TpContext ctx;
      std::string err;
      const size_t max_n = 2097152u;  // 8 MiB: exercises >= 4 MiB AFF_TP_AR_STAGED path
      assert(tp_init(&ctx, max_n, &err));
      float *d_in[2] = {}, *d_out[2] = {};
      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        assert(hipMalloc(&d_in[r], max_n * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_out[r], max_n * sizeof(float)) == hipSuccess);
      }
      struct TpCase { size_t n; TpReduce mode; bool alias; bool reset_after; };
      const TpCase cases[] = {
          {32768u, TpReduce::Fp8, false, false},
          {65536u, TpReduce::Fp8, false, false},
          {262144u, TpReduce::Fp8, false, false},
          {2563u, TpReduce::Fp8, false, false},
          {1048576u, TpReduce::Fp8, false, false},  // 4 MiB staged boundary
          {2048u, TpReduce::Exact, false, false},   // interleaved Exact on shared seq/flags
          {2097152u, TpReduce::Fp8, false, false},  // 8 MiB staged
          {1049003u, TpReduce::Fp8, true, true},    // > 4 MiB odd-tail staged in-place alias + reset
          {1048576u, TpReduce::Fp8, false, false},  // post-tp_reset_seq staged verify
      };
      for (const TpCase& tc : cases) {
        const size_t n = tc.n;
        std::vector<float> h0(n), h1(n), ref(n);
        for (size_t i = 0; i < n; ++i) {
          h0[i] = std::sin((float)(i % 331u) * 0.043f) * 3.0f;
          h1[i] = std::cos((float)(i % 277u) * 0.059f) * 2.5f;
        }
        if (tc.mode == TpReduce::Exact) {
          for (size_t i = 0; i < n; ++i) ref[i] = h0[i] + h1[i];
        } else {
          // CPU FP8 group-128 QDQ wire reference
          for (size_t g0 = 0; g0 < n; g0 += 128u) {
            const size_t ge = std::min(n, g0 + 128u);
            float a0 = 0.0f, a1 = 0.0f;
            for (size_t i = g0; i < ge; ++i) {
              a0 = std::max(a0, std::fabs(h0[i]));
              a1 = std::max(a1, std::fabs(h1[i]));
            }
            const float s0 = a0 > 0.0f ? a0 / 448.0f : 1.0f;
            const float s1 = a1 > 0.0f ? a1 / 448.0f : 1.0f;
            for (size_t i = g0; i < ge; ++i) {
              const float q0 = e4m3_round_ref(std::min(448.0f, std::max(-448.0f, h0[i] / s0))) * s0;
              const float q1 = e4m3_round_ref(std::min(448.0f, std::max(-448.0f, h1[i] / s1))) * s1;
              ref[i] = q0 + q1;
            }
          }
        }
        assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
        assert(hipMemcpy(d_in[0], h0.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
        assert(hipMemcpy(d_in[1], h1.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        float* const* dst_ptrs = tc.alias ? d_in : d_out;
        tp_all_reduce(ctx, d_in, dst_ptrs, n, tc.mode);
        tp_sync(ctx);
        tp_check(ctx);

        std::vector<float> got0(n), got1(n);
        assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
        assert(hipMemcpy(got0.data(), dst_ptrs[0], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
        assert(hipMemcpy(got1.data(), dst_ptrs[1], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(std::memcmp(got0.data(), got1.data(), n * sizeof(float)) == 0);
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i) max_diff = std::max(max_diff, std::fabs(got0[i] - ref[i]));
        assert(max_diff <= (tc.mode == TpReduce::Exact ? 1e-5f : 1e-2f));
        if (tc.reset_after) {
          tp_reset_seq(ctx);
          tp_sync(ctx);
        }
      }
      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        (void)hipFree(d_in[r]);
        (void)hipFree(d_out[r]);
      }
      tp_free(&ctx);

      // 13b. Staged AR LDS-cache parity (AFF_TP_AR_STAGED_CACHE=1 vs =0, bit-identical)
      {
        const char* prev_env = std::getenv("AFF_TP_AR_STAGED_CACHE");
        std::string prev_val = prev_env ? prev_env : "";
        TpContext ctx_p;
        std::string perr;
        const size_t max_np = 2097152u;
        assert(tp_init(&ctx_p, max_np, &perr));
        float *p_in[2] = {}, *p_out0[2] = {}, *p_out1[2] = {};
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx_p.dev[r]) == hipSuccess);
          assert(hipMalloc(&p_in[r], max_np * sizeof(float)) == hipSuccess);
          assert(hipMalloc(&p_out0[r], max_np * sizeof(float)) == hipSuccess);
          assert(hipMalloc(&p_out1[r], max_np * sizeof(float)) == hipSuccess);
        }
        // Covers <= 30 groups/stage (all LDS cached: 1048576), > 30 groups/stage (2097152),
        // and non-aligned tail (1049003).
        const size_t cache_sizes[] = {1048576u, 1049003u, 2097152u};
        for (size_t n : cache_sizes) {
          std::vector<float> h0(n), h1(n);
          for (size_t i = 0; i < n; ++i) {
            h0[i] = std::sin((float)(i % 331u) * 0.043f) * 3.0f;
            h1[i] = std::cos((float)(i % 277u) * 0.059f) * 2.5f;
          }
          for (int r = 0; r < 2; ++r) {
            assert(hipSetDevice(ctx_p.dev[r]) == hipSuccess);
            assert(hipMemcpy(p_in[r], (r == 0 ? h0 : h1).data(), n * sizeof(float),
                             hipMemcpyHostToDevice) == hipSuccess);
          }
          setenv("AFF_TP_AR_STAGED_CACHE", "0", 1);
          tp_all_reduce(ctx_p, p_in, p_out0, n, TpReduce::Fp8);
          tp_sync(ctx_p);
          tp_check(ctx_p);

          setenv("AFF_TP_AR_STAGED_CACHE", "1", 1);
          tp_all_reduce(ctx_p, p_in, p_out1, n, TpReduce::Fp8);
          tp_sync(ctx_p);
          tp_check(ctx_p);

          std::vector<float> g0(n), g1(n), g1_peer(n);
          assert(hipSetDevice(ctx_p.dev[0]) == hipSuccess);
          assert(hipMemcpy(g0.data(), p_out0[0], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(g1.data(), p_out1[0], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipSetDevice(ctx_p.dev[1]) == hipSuccess);
          assert(hipMemcpy(g1_peer.data(), p_out1[1], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          assert(std::memcmp(g0.data(), g1.data(), n * sizeof(float)) == 0);
          assert(std::memcmp(g1.data(), g1_peer.data(), n * sizeof(float)) == 0);
        }
        if (prev_env) setenv("AFF_TP_AR_STAGED_CACHE", prev_val.c_str(), 1);
        else unsetenv("AFF_TP_AR_STAGED_CACHE");
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx_p.dev[r]) == hipSuccess);
          (void)hipFree(p_in[r]);
          (void)hipFree(p_out0[r]);
          (void)hipFree(p_out1[r]);
        }
        tp_free(&ctx_p);
      }

      // 13c. TpReduce::Coded (R2 lossless coded BF16 AR + raw 0xFF fallback + AFF_TP_AR_CODED=1)
      {
        auto save_env = [](const char* k, bool& has, std::string& val) {
          const char* v = std::getenv(k);
          has = (v != nullptr);
          val = v ? v : "";
        };
        auto restore_env = [](const char* k, bool has, const std::string& val) {
          if (has) setenv(k, val.c_str(), 1);
          else unsetenv(k);
        };
        bool had_coded = false, had_mode = false, had_wake = false, had_reg = false;
        std::string prev_coded, prev_mode, prev_wake, prev_reg;
        save_env("AFF_TP_AR_CODED", had_coded, prev_coded);
        save_env("AFF_TP_AR_MODE", had_mode, prev_mode);
        save_env("AFF_AR_WAKE_STORE", had_wake, prev_wake);
        save_env("AFF_AR_REG_FORM", had_reg, prev_reg);
        unsetenv("AFF_TP_AR_MODE");
        setenv("AFF_TP_AR_CODED", "1", 1);

        TpContext ctx_c;
        std::string cerr;
        const size_t max_nc = 2097152u;
        assert(tp_init(&ctx_c, max_nc, &cerr));
        float *c_in[2] = {}, *c_out[2] = {};
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx_c.dev[r]) == hipSuccess);
          assert(hipMalloc(&c_in[r], max_nc * sizeof(float)) == hipSuccess);
          assert(hipMalloc(&c_out[r], max_nc * sizeof(float)) == hipSuccess);
        }
        struct CodedCase {
          size_t n;
          TpReduce mode;
          bool alias;
          bool reset_after;
          int wake_store;
          int reg_form;
        };
        const CodedCase ccases[] = {
            {2563u, TpReduce::Coded, false, false, 1, 1},
            {32768u, TpReduce::Auto, false, false, 1, 1},
            {262144u, TpReduce::Coded, false, false, 0, 0},
            {1048576u, TpReduce::Coded, false, false, 1, 0},
            {1048576u, TpReduce::Auto, false, false, 0, 1},
            {1049003u, TpReduce::Coded, true, true, 1, 1},
            {2097152u, TpReduce::Coded, false, false, 1, 1},
            {2097152u, TpReduce::Auto, true, false, 0, 0},
        };
        for (const CodedCase& tc : ccases) {
          const size_t n = tc.n;
          std::vector<float> h0(n), h1(n), ref(n);
          for (size_t i = 0; i < n; ++i) {
            h0[i] = std::sin((float)(i % 331u) * 0.043f) * 3.0f;
            h1[i] = std::cos((float)(i % 277u) * 0.059f) * 2.5f;
          }
          // Group 0: +0.0 and -0.0
          h0[0] = 0.0f;  h1[0] = 0.0f;
          h0[1] = -0.0f; h1[1] = 0.0f;
          h0[2] = -0.0f; h1[2] = -0.0f;
          // Group 1 (64..127): subnormals -> 0xFF fallback
          h0[64] = bf16_to_f32_cpu(0x0001u); h1[64] = bf16_to_f32_cpu(0x0002u);
          h0[65] = bf16_to_f32_cpu(0x8042u); h1[65] = bf16_to_f32_cpu(0x8001u);
          // Group 2 (128..191): +-Inf -> 0xFF fallback
          h0[128] = bf16_to_f32_cpu(0x7F80u); h1[128] = bf16_to_f32_cpu(0x7F80u);
          h0[129] = bf16_to_f32_cpu(0xFF80u); h1[129] = bf16_to_f32_cpu(0xFF80u);
          // Group 3 (192..255): wide exponent span > 14 -> 0xFF fallback
          h0[192] = 1.0f;                     h1[192] = std::ldexp(1.0f, -20);
          h0[193] = std::ldexp(1.0f, -20);    h1[193] = 1.0f;
          // Odd tail / late-stage boundary fallback cases
          if (n > 2048u) {
            const size_t tail = n - 1u;
            h0[tail] = bf16_to_f32_cpu(0x7F80u); h1[tail] = 1.0f;
            h0[n - 11u] = -0.0f;                 h1[n - 11u] = -0.0f;
            const size_t mid = (n / 2u) & ~size_t(63);
            h0[mid] = 1.0f;                      h1[mid] = 2.0f;
            h0[mid + 1u] = std::ldexp(1.0f, -24); h1[mid + 1u] = bf16_to_f32_cpu(0x0015u);
          }
          for (size_t i = 0; i < n; ++i)
            ref[i] = bf16_to_f32_cpu(f32_to_bf16_cpu(h0[i])) +
                     bf16_to_f32_cpu(f32_to_bf16_cpu(h1[i]));

          setenv("AFF_AR_WAKE_STORE", tc.wake_store ? "1" : "0", 1);
          setenv("AFF_AR_REG_FORM", tc.reg_form ? "1" : "0", 1);
          assert(hipSetDevice(ctx_c.dev[0]) == hipSuccess);
          assert(hipMemcpy(c_in[0], h0.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipSetDevice(ctx_c.dev[1]) == hipSuccess);
          assert(hipMemcpy(c_in[1], h1.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          float* const* dst_ptrs = tc.alias ? c_in : c_out;
          tp_all_reduce(ctx_c, c_in, dst_ptrs, n, tc.mode);
          tp_sync(ctx_c);
          tp_check(ctx_c);

          std::vector<float> got0(n), got1(n);
          assert(hipSetDevice(ctx_c.dev[0]) == hipSuccess);
          assert(hipMemcpy(got0.data(), dst_ptrs[0], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipSetDevice(ctx_c.dev[1]) == hipSuccess);
          assert(hipMemcpy(got1.data(), dst_ptrs[1], n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          assert(std::memcmp(got0.data(), got1.data(), n * sizeof(float)) == 0);
          assert(std::memcmp(got0.data(), ref.data(), n * sizeof(float)) == 0);
          if (tc.reset_after) {
            tp_reset_seq(ctx_c);
            tp_sync(ctx_c);
          }
        }
        restore_env("AFF_TP_AR_CODED", had_coded, prev_coded);
        restore_env("AFF_TP_AR_MODE", had_mode, prev_mode);
        restore_env("AFF_AR_WAKE_STORE", had_wake, prev_wake);
        restore_env("AFF_AR_REG_FORM", had_reg, prev_reg);
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx_c.dev[r]) == hipSuccess);
          (void)hipFree(c_in[r]);
          (void)hipFree(c_out[r]);
        }
        tp_free(&ctx_c);
      }
      assert(hipSetDevice(0) == hipSuccess);
    } else {
      std::printf("SKIP: TP=2 fp8 all-reduce test (%d device(s))\n", ndev);
    }
  }

  // 14. Test moe_plan_hip Wave32 scan parity vs scalar CPU reference (nb = 6, 64)
  {
    const uint32_t ne = 256u, k = 6u;
    std::vector<uint64_t> h_plc(ne);
    for (uint32_t e = 0; e < ne; ++e) {
      const uint64_t base = 0x100000ull + (uint64_t)e * 4096ull;
      h_plc[e] = (e % 3u != 0u) ? (base | 1ull) : base; // ~2/3 resident, ~1/3 pooled
    }
    ExpertShardLayout sh{};
    sh.stride = 4096u;
    for (int pl = 0; pl < 3; ++pl) {
      sh.data_off[pl] = 256u * (uint32_t)(pl + 1);
      sh.scale_off[pl] = 1024u + 256u * (uint32_t)pl;
      sh.rscale_off[pl] = 2048u + 256u * (uint32_t)pl;
    }

    for (uint32_t nb : {6u, 64u}) {
      const uint32_t total = nb * k;
      const uint32_t cap = moe_plan_tile_cap(nb, k, ne);
      std::vector<uint32_t> h_sel(total);
      std::vector<float> h_wt(total);
      for (uint32_t t = 0; t < nb; ++t) {
        for (uint32_t s = 0; s < k; ++s) {
          // Ensure distinct experts within each token
          h_sel[t * k + s] = (t * 13u + s * 29u + 5u) % ne;
          h_wt[t * k + s] = 0.1f + 0.05f * (float)s;
        }
      }

      // CPU reference for uniq, start, tok, wt_out, ord, run, counts
      std::vector<uint32_t> counts_e(ne, 0u), tok_cnt(nb, 0u);
      for (uint32_t i = 0; i < total; ++i) {
        counts_e[h_sel[i]]++;
        tok_cnt[i / k]++;
      }
      std::vector<uint32_t> uniq;
      for (uint32_t e = 0; e < ne; ++e)
        if (counts_e[e] && (h_plc[e] & 1ull)) uniq.push_back(e);
      const uint32_t n_res = (uint32_t)uniq.size();
      for (uint32_t e = 0; e < ne; ++e)
        if (counts_e[e] && !(h_plc[e] & 1ull)) uniq.push_back(e);

      std::vector<uint32_t> start_e(ne, 0u), fill_e(ne, 0u);
      uint32_t acc = 0u;
      for (uint32_t e : uniq) {
        start_e[e] = acc;
        acc += counts_e[e];
      }
      std::vector<uint32_t> ref_tok(2u * total), ref_ord(total), ref_run(nb + 1u);
      std::vector<float> ref_wt(total);
      ref_run[0] = 0u;
      for (uint32_t t = 0; t < nb; ++t) ref_run[t + 1u] = ref_run[t] + tok_cnt[t];
      std::vector<uint32_t> tfill(nb, 0u);
      for (uint32_t i = 0; i < total; ++i) {
        const uint32_t e = h_sel[i];
        const uint32_t at = start_e[e] + fill_e[e]++;
        ref_tok[at] = i / k;
        ref_tok[total + at] = at;
        ref_wt[at] = h_wt[i];
        ref_ord[ref_run[i / k] + tfill[i / k]++] = at;
      }
      for (uint32_t t = 0; t < nb; ++t)
        std::sort(ref_ord.begin() + ref_run[t], ref_ord.begin() + ref_run[t + 1u]);
      uint32_t ref_gcnt[8] = {};
      for (uint32_t j = 0; j < uniq.size(); ++j) {
        const uint32_t c = counts_e[uniq[j]];
        const uint32_t cls = (j < n_res) ? 0u : 1u;
        const uint32_t q = c / 64u, r = c % 64u;
        ref_gcnt[cls * 4u + 2u] += q;
        if (r) {
          const uint32_t b = (r <= 16u) ? 0u : (r <= 32u) ? 1u : 2u;
          ref_gcnt[cls * 4u + b] += 1u;
        }
      }

      uint32_t *d_sel = nullptr, *d_tok = nullptr, *d_ord = nullptr, *d_run = nullptr,
               *d_col = nullptr, *d_cnt = nullptr;
      uint64_t* d_plc = nullptr;
      float* d_wt = nullptr;
      float* d_wto = nullptr;
      uint16_t *d_bg = nullptr, *d_bu = nullptr, *d_bydn = nullptr;
      ExpertTile* d_tiles = nullptr;
      MoePlanStatus* d_st = nullptr;
      assert(hipMalloc(&d_sel, total * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_wt, total * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_plc, ne * sizeof(uint64_t)) == hipSuccess);
      assert(hipMalloc(&d_tok, 2u * total * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_wto, total * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_ord, total * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_run, (nb + 1u) * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_col, total * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_tiles, (size_t)3u * 8u * cap * sizeof(ExpertTile)) == hipSuccess);
      assert(hipMalloc(&d_cnt, 8u * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_st, sizeof(MoePlanStatus)) == hipSuccess);
      assert(hipMalloc(&d_bg, 64u) == hipSuccess);
      assert(hipMalloc(&d_bu, 64u) == hipSuccess);
      assert(hipMalloc(&d_bydn, 64u) == hipSuccess);
      assert(hipMemset(d_st, 0, sizeof(MoePlanStatus)) == hipSuccess);
      assert(hipMemcpy(d_sel, h_sel.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_wt, h_wt.data(), total * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_plc, h_plc.data(), ne * sizeof(uint64_t), hipMemcpyHostToDevice) == hipSuccess);

      MoePlanArgs A{};
      A.sel = d_sel; A.wt = d_wt; A.plc = d_plc;
      A.layer = 0; A.nb_tok = nb; A.k = k; A.n_expert = ne;
      A.tok = d_tok; A.wt_out = d_wto; A.ord = d_ord; A.run = d_run; A.slot_col = d_col;
      A.tiles = d_tiles; A.counts = d_cnt; A.status = d_st;
      A.shard = sh; A.bg = d_bg; A.bu = d_bu; A.bydn = d_bydn; A.tile_cap = cap;
      assert(moe_plan_hip(A, nullptr));
      assert(hipDeviceSynchronize() == hipSuccess);

      MoePlanStatus st{};
      std::vector<uint32_t> g_tok(2u * total), g_ord(total), g_run(nb + 1u), g_cnt(8u);
      std::vector<float> g_wt(total);
      assert(hipMemcpy(&st, d_st, sizeof(st), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_tok.data(), d_tok, 2u * total * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_wt.data(), d_wto, total * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_ord.data(), d_ord, total * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_run.data(), d_run, (nb + 1u) * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_cnt.data(), d_cnt, 8u * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

      assert(st.taken == total && st.bad == 0u && st.over == 0u);
      assert(st.n_res == n_res && st.n_uniq == uniq.size());
      assert(g_run == ref_run);
      assert(g_tok == ref_tok);
      assert(g_ord == ref_ord);
      for (size_t i = 0; i < total; ++i) assert(g_wt[i] == ref_wt[i]);
      const uint32_t lg = moe_plan_last_group(nb);
      for (uint32_t g = 0; g < 8u; ++g) {
        assert(g_cnt[g] == ref_gcnt[g]);
        if ((g & 3u) > lg) assert(g_cnt[g] == 0u);
      }
      for (void* p : {(void*)d_sel, (void*)d_wt, (void*)d_plc, (void*)d_tok, (void*)d_wto,
                      (void*)d_ord, (void*)d_run, (void*)d_col, (void*)d_tiles, (void*)d_cnt,
                      (void*)d_st, (void*)d_bg, (void*)d_bu, (void*)d_bydn})
        (void)hipFree(p);
    }
  }

  // 15. Test expert_gemv_hip 6-wave (cols = 1152) parity vs CPU expert_gemv_scaled
  {
    const uint32_t rows = 64u, cols = 1152u;
    const QuantSpec spec = expert_quant_spec(AFF_Q2P875);
    std::vector<float> h_w((size_t)rows * cols);
    for (size_t i = 0; i < h_w.size(); ++i)
      h_w[i] = std::sin((float)(i % 389u) * 0.041f) * 0.8f;
    std::vector<float> norm_samps;
    collect_normalized_samples(h_w.data(), rows, cols, spec, &norm_samps);
    Codebook cb = fit_codebook(norm_samps.data(), norm_samps.size(), spec, nullptr, 15, 7);
    QuantizedMatrix qm;
    std::string qerr;
    assert(quantize_matrix(h_w.data(), rows, cols, spec, cb, &qm, &qerr));

    std::vector<float> h_x(cols), h_y_cpu(rows, 0.0f);
    for (uint32_t c = 0; c < cols; ++c) h_x[c] = std::cos((float)c * 0.029f) * 1.3f;
    const float router_w = 0.75f;
    ExpertMatrixView mv{qm.data.data(), qm.scales.data(), qm.rscales.data(), rows, cols, spec};
    expert_gemv_scaled(mv, cb, h_x.data(), router_w, h_y_cpu.data());

    std::vector<uint8_t> cg_data(qm.data.size()), cg_scales(qm.scales.size());
    expert_transpose_plane_host(qm.data.data(), qm.scales.data(), cg_data.data(), cg_scales.data(),
                                rows, cols);
    expert_set_plane_layout_colgroup(true);
    expert_set_wire_format(expert_wire_format_for_quant(AFF_Q2P875));

    uint8_t *d_data = nullptr, *d_scales = nullptr;
    uint16_t* d_rscales = nullptr;
    float *d_cb = nullptr, *d_x = nullptr, *d_y = nullptr;
    assert(hipMalloc(&d_data, cg_data.size()) == hipSuccess);
    assert(hipMalloc(&d_scales, cg_scales.size()) == hipSuccess);
    assert(hipMalloc(&d_rscales, qm.rscales.size()) == hipSuccess);
    assert(hipMalloc(&d_cb, cb.v.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_x, cols * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_y, rows * sizeof(float)) == hipSuccess);
    assert(hipMemcpy(d_data, cg_data.data(), cg_data.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_scales, cg_scales.data(), cg_scales.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_rscales, qm.rscales.data(), qm.rscales.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_cb, cb.v.data(), cb.v.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_x, h_x.data(), cols * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemset(d_y, 0, rows * sizeof(float)) == hipSuccess);

    DeviceMatrix dm{d_data, d_scales, d_rscales, rows, cols};
    expert_gemv_hip(dm, d_cb, d_x, router_w, d_y, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<float> h_y_gpu(rows);
    assert(hipMemcpy(h_y_gpu.data(), d_y, rows * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    float max_diff = 0.0f;
    for (uint32_t r = 0; r < rows; ++r)
      max_diff = std::max(max_diff, std::fabs(h_y_gpu[r] - h_y_cpu[r]));
    assert(max_diff < 1e-4f);
    for (void* p : {(void*)d_data, (void*)d_scales, (void*)d_rscales, (void*)d_cb, (void*)d_x, (void*)d_y})
      (void)hipFree(p);
  }

  // 16. Test idx_topk_mask_hip with mask == nullptr (spread + single-block + ties)
  {
    auto cpu_topk_adm = [](const float* sc, uint32_t nk, uint32_t k) -> std::vector<uint32_t> {
      std::vector<uint32_t> ord;
      for (uint32_t i = 0; i < nk; ++i)
        if (sc[i] > -1e29f) ord.push_back(i);
      if (ord.size() > k) {
        std::nth_element(ord.begin(), ord.begin() + k, ord.end(), [&](uint32_t a, uint32_t b) {
          return (sc[a] != sc[b]) ? (sc[a] > sc[b]) : (a < b);
        });
        ord.resize(k);
      }
      std::sort(ord.begin(), ord.end());
      return ord;
    };

    // (a) nb=1 spread path (stride >= 4096) with ties at boundary
    {
      const uint32_t nb = 1u, stride = 8192u, nkeys_v = 6000u, k = 128u, adm_stride = 128u;
      std::vector<float> h_sc(stride, -1e30f);
      for (uint32_t i = 0; i < nkeys_v; ++i)
        h_sc[i] = ((i % 97u) == 0u) ? 5.0f : std::sin((float)i * 0.13f);
      uint64_t sb = 0, tb = 0;
      const uint64_t hb = idx_topk_scratch_bytes(nb, &sb, &tb);
      TopkScratch sp{};
      sp.tok_cap = nb;
      float* d_sc = nullptr;
      uint32_t *d_nk = nullptr, *d_adm = nullptr, *d_admn = nullptr;
      assert(hipMalloc(&d_sc, stride * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_nk, sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm, adm_stride * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn, sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&sp.hist, hb) == hipSuccess);
      assert(hipMalloc(&sp.state, sb) == hipSuccess);
      assert(hipMalloc(&sp.tie, tb) == hipSuccess);
      assert(hipMemset(sp.hist, 0, hb) == hipSuccess);
      assert(hipMemset(sp.state, 0, sb) == hipSuccess);
      assert(hipMemset(sp.tie, 0, tb) == hipSuccess);
      assert(hipMemcpy(d_sc, h_sc.data(), stride * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_nk, &nkeys_v, sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);

      idx_topk_mask_hip(d_sc, nullptr, d_nk, nb, stride, 0u, nkeys_v, k, nullptr, &sp,
                        d_adm, d_admn, adm_stride);
      assert(hipDeviceSynchronize() == hipSuccess);

      uint32_t got_n = 0u;
      std::vector<uint32_t> got_adm(adm_stride);
      assert(hipMemcpy(&got_n, d_admn, sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(got_adm.data(), d_adm, adm_stride * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      const std::vector<uint32_t> ref = cpu_topk_adm(h_sc.data(), nkeys_v, k);
      assert(got_n == ref.size());
      got_adm.resize(got_n);
      assert(got_adm == ref);
      for (void* p : {(void*)d_sc, (void*)d_nk, (void*)d_adm, (void*)d_admn,
                      (void*)sp.hist, (void*)sp.state, (void*)sp.tie})
        (void)hipFree(p);
    }

    // (b) nb=4 single-block path mixing <=k and >k rows + ties
    {
      const uint32_t nb = 4u, stride = 512u, k = 128u, adm_stride = 128u;
      const uint32_t h_nk[4] = {100u, 300u, 512u, 40u};
      std::vector<float> h_sc((size_t)nb * stride);
      for (uint32_t t = 0; t < nb; ++t)
        for (uint32_t i = 0; i < stride; ++i)
          h_sc[(size_t)t * stride + i] = ((i % 23u) == 0u) ? 2.0f : std::cos((float)(t * 511u + i) * 0.07f);
      float* d_sc = nullptr;
      uint32_t *d_nk = nullptr, *d_adm = nullptr, *d_admn = nullptr;
      assert(hipMalloc(&d_sc, h_sc.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_nk, nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm, (size_t)nb * adm_stride * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn, nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMemcpy(d_sc, h_sc.data(), h_sc.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_nk, h_nk, nb * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);

      idx_topk_mask_hip(d_sc, nullptr, d_nk, nb, stride, 0u, stride, k, nullptr, nullptr,
                        d_adm, d_admn, adm_stride);
      assert(hipDeviceSynchronize() == hipSuccess);

      uint32_t got_n[4] = {};
      std::vector<uint32_t> got_adm((size_t)nb * adm_stride);
      assert(hipMemcpy(got_n, d_admn, nb * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(got_adm.data(), d_adm, got_adm.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      for (uint32_t t = 0; t < nb; ++t) {
        const std::vector<uint32_t> ref = cpu_topk_adm(h_sc.data() + (size_t)t * stride, h_nk[t], k);
        assert(got_n[t] == ref.size());
        std::vector<uint32_t> slice(got_adm.begin() + (size_t)t * adm_stride,
                                    got_adm.begin() + (size_t)t * adm_stride + got_n[t]);
        assert(slice == ref);
      }
      for (void* p : {(void*)d_sc, (void*)d_nk, (void*)d_adm, (void*)d_admn}) (void)hipFree(p);
    }
  }

  // 17. Test attn_flash (FP8R split-KV + s_ps + rcpf + uint2) & attention_gpu (FP8R + s_q + DPP)
  {
    auto decode_fp8r_row = [](const uint8_t* row_b, float* dst, uint32_t w) {
      float sc = 1.0f;
      std::memcpy(&sc, row_b + w, 4);
      for (uint32_t i = 0; i < w; ++i) {
        const uint8_t b = row_b[i];
        const float sign = (b & 0x80u) ? -1.0f : 1.0f;
        const int exp = (b >> 3) & 0xFu;
        const int mant = b & 0x7u;
        float mag = 0.0f;
        if (exp == 0) mag = std::ldexp((float)mant, -9);
        else if (!(exp == 15 && mant == 7)) mag = std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
        dst[i] = sign * mag * sc;
      }
    };
    const uint32_t width = kFlashWidth, n_head = 32u, ring = 128u, n_comp = 192u, sliding = 64u;
    const uint32_t nb = 4u, pos0 = 64u, n_raw = 64u;
    const float scale = 1.0f / std::sqrt((float)width);
    std::vector<uint8_t> h_raw(ring * kFlashRowBytes, 0u), h_comp(n_comp * kFlashRowBytes, 0u);
    std::vector<float> kv_dec((n_raw + n_comp) * width);
    std::vector<float> tmp_row(width);
    for (uint32_t r = 0; r < ring; ++r) {
      for (uint32_t d = 0; d < width; ++d)
        tmp_row[d] = std::sin((float)(r * 31u + d * 7u) * 0.017f) * 1.4f;
      kv_encode_fp8r_row(tmp_row.data(), h_raw.data() + (size_t)r * kFlashRowBytes);
    }
    for (uint32_t c = 0; c < n_comp; ++c) {
      for (uint32_t d = 0; d < width; ++d)
        tmp_row[d] = std::cos((float)(c * 23u + d * 11u) * 0.021f) * 1.2f;
      kv_encode_fp8r_row(tmp_row.data(), h_comp.data() + (size_t)c * kFlashRowBytes);
    }
    std::vector<float> h_sinks(n_head), h_q((size_t)nb * n_head * width);
    std::vector<uint16_t> h_qb(h_q.size());
    for (uint32_t h = 0; h < n_head; ++h) h_sinks[h] = -0.5f + 0.05f * (float)h;
    for (size_t i = 0; i < h_q.size(); ++i) {
      const float v = std::sin((float)(i % 997u) * 0.019f) * 1.1f;
      h_qb[i] = f32_to_bf16_cpu(v);
      h_q[i] = bf16_to_f32_cpu(h_qb[i]);
    }
    std::vector<uint8_t> h_allow((size_t)nb * n_comp, 0u);
    for (uint32_t t = 0; t < nb; ++t)
      for (uint32_t c = 0; c < n_comp; ++c)
        if ((c + t) % 3u != 0u) h_allow[(size_t)t * n_comp + c] = 1u;

    uint8_t *d_raw = nullptr, *d_comp = nullptr, *d_allow = nullptr;
    float *d_sinks = nullptr, *d_qf = nullptr, *d_out_f = nullptr, *d_out_a = nullptr;
    uint16_t* d_qb = nullptr;
    uint32_t* d_ncv = nullptr;
    assert(hipMalloc(&d_raw, h_raw.size()) == hipSuccess);
    assert(hipMalloc(&d_comp, h_comp.size()) == hipSuccess);
    assert(hipMalloc(&d_allow, h_allow.size()) == hipSuccess);
    assert(hipMalloc(&d_sinks, h_sinks.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_qf, h_q.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_qb, h_qb.size() * 2u) == hipSuccess);
    assert(hipMalloc(&d_out_f, h_q.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_out_a, h_q.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_ncv, nb * 4u) == hipSuccess);
    std::vector<uint32_t> h_ncv(nb, n_comp);
    assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_comp, h_comp.data(), h_comp.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_allow, h_allow.data(), h_allow.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_qf, h_q.data(), h_q.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_qb, h_qb.data(), h_qb.size() * 2u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ncv, h_ncv.data(), nb * 4u, hipMemcpyHostToDevice) == hipSuccess);

    FlashSplit fsp{};
    uint64_t ml_bytes = 0;
    const uint64_t o_bytes = flash_split_bytes(nb, n_head, 16u, &ml_bytes);
    fsp.tok_cap = nb; fsp.slots = 16u;
    assert(hipMalloc(&fsp.o, o_bytes) == hipSuccess);
    assert(hipMalloc(&fsp.ml, ml_bytes) == hipSuccess);

    mqa_attend_flash_hip(d_qf, d_raw, d_comp, d_allow, d_ncv, 0u, d_sinks, d_out_f, 0u, nb, nb,
                         pos0, sliding, ring, n_comp, n_head, scale, nullptr, false, &fsp);
    mqa_attend_batch_hip(KvDtype::FP8R, d_qb, d_raw, d_comp, d_allow, d_ncv, d_sinks, d_out_a,
                         0u, nb, nb, pos0, sliding, ring, n_comp, n_comp, n_head, width, 64u,
                         scale, nullptr, true);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<float> out_f(h_q.size()), out_a(h_q.size());
    assert(hipMemcpy(out_f.data(), d_out_f, out_f.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(out_a.data(), d_out_a, out_a.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);

    // CPU FP8R attention ref for token 0
    for (uint32_t j = 0; j < n_raw; ++j) {
      const uint32_t rs = ((pos0 + 1u - n_raw) + j) % ring;
      decode_fp8r_row(h_raw.data() + (size_t)rs * kFlashRowBytes, &kv_dec[j * width], width);
    }
    for (uint32_t c = 0; c < n_comp; ++c)
      decode_fp8r_row(h_comp.data() + (size_t)c * kFlashRowBytes, &kv_dec[(n_raw + c) * width], width);
    for (uint32_t h = 0; h < n_head; ++h) {
      std::vector<float> scores(n_raw + n_comp, -1e30f);
      float mx = h_sinks[h];
      for (uint32_t j = 0; j < n_raw + n_comp; ++j) {
        if (j >= n_raw && !h_allow[j - n_raw]) continue;
        double s = 0.0;
        for (uint32_t d = 0; d < width; ++d)
          s += (double)h_q[(size_t)h * width + d] * (double)kv_dec[(size_t)j * width + d];
        scores[j] = (float)s * scale;
        mx = std::max(mx, scores[j]);
      }
      double den = std::exp((double)h_sinks[h] - (double)mx);
      std::vector<double> wv(n_raw + n_comp, 0.0);
      for (uint32_t j = 0; j < n_raw + n_comp; ++j) {
        if (j >= n_raw && !h_allow[j - n_raw]) continue;
        wv[j] = std::exp((double)scores[j] - (double)mx);
        den += wv[j];
      }
      for (uint32_t d = 0; d < width; ++d) {
        double acc = 0.0;
        for (uint32_t j = 0; j < n_raw + n_comp; ++j)
          acc += wv[j] * (double)kv_dec[(size_t)j * width + d];
        const float ref_v = (float)(acc / den);
        const size_t idx = (size_t)h * width + d;
        assert(std::fabs(out_a[idx] - ref_v) < 1e-4f);
        assert(std::fabs(out_f[idx] - ref_v) < 2e-2f);
      }
    }
    // Single-token decode mqa_attend_hip(FP8R, nullptr allowed)
    std::vector<float> out_s(n_head * width);
    mqa_attend_hip(KvDtype::FP8R, d_qf, d_raw, d_comp, nullptr, d_sinks, d_out_a, n_head,
                   (pos0 + 1u - n_raw) % ring, n_raw, ring, n_comp, width, 64u, scale, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);
    assert(hipMemcpy(out_s.data(), d_out_a, out_s.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);
    for (float v : out_s) assert(!std::isnan(v));

    for (void* p : {(void*)d_raw, (void*)d_comp, (void*)d_allow, (void*)d_sinks, (void*)d_qf,
                    (void*)d_qb, (void*)d_out_f, (void*)d_out_a, (void*)d_ncv,
                    (void*)fsp.o, (void*)fsp.ml})
      (void)hipFree(p);
  }

  // 18. Test compressor_rows_hip & idx_key_rows_hip Wave32 DPP reductions
  {
    const uint32_t hd = 512u, ratio = 4u, n = 8u, n_out = 2u, n_rot = 64u;
    const uint32_t w = comp_width(ratio, hd), hist = comp_hist(ratio);
    const RopeDerived rope{1.0f, std::pow(10000.0f, -2.0f / (float)n_rot), 1.0f, 1.0f, 1.0f};
    std::vector<float> h_kv((size_t)n * w), h_sc((size_t)n * w), h_ape(ratio * w), h_nrm(hd);
    for (size_t i = 0; i < h_kv.size(); ++i) h_kv[i] = std::sin((float)i * 0.037f);
    for (size_t i = 0; i < h_sc.size(); ++i) h_sc[i] = std::cos((float)i * 0.023f) * 0.5f;
    for (size_t i = 0; i < h_ape.size(); ++i) h_ape[i] = 0.01f * std::sin((float)i * 0.1f);
    for (uint32_t j = 0; j < hd; ++j) h_nrm[j] = 0.8f + 0.4f * (float)j / (float)hd;

    float *d_kv = nullptr, *d_sc = nullptr, *d_hkv = nullptr, *d_hsc = nullptr,
          *d_ape = nullptr, *d_nrm = nullptr, *d_lat = nullptr;
    uint8_t* d_out = nullptr;
    assert(hipMalloc(&d_kv, h_kv.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_sc, h_sc.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_hkv, (size_t)hist * w * 4u) == hipSuccess);
    assert(hipMalloc(&d_hsc, (size_t)hist * w * 4u) == hipSuccess);
    assert(hipMalloc(&d_ape, h_ape.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_nrm, hd * 4u) == hipSuccess);
    assert(hipMalloc(&d_lat, (size_t)n_out * hd * 4u) == hipSuccess);
    assert(hipMalloc(&d_out, (size_t)n_out * kFlashRowBytes) == hipSuccess);
    assert(hipMemcpy(d_kv, h_kv.data(), h_kv.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_sc, h_sc.data(), h_sc.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ape, h_ape.data(), h_ape.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_nrm, h_nrm.data(), hd * 4u, hipMemcpyHostToDevice) == hipSuccess);
    compressor_hist_reset_hip(d_hkv, d_hsc, w, hist, nullptr);

    CompressJob j{};
    j.kv = d_kv; j.sc = d_sc; j.hist_kv = d_hkv; j.hist_sc = d_hsc;
    j.ape = d_ape; j.norm = d_nrm; j.out = d_out; j.out_stride = kFlashRowBytes;
    j.slot0 = 0u; j.head_dim = hd; j.lat = d_lat;
    compressor_rows_hip(0, j, n, 0u, ratio, n_out, n_rot, rope, 1e-6f, nullptr);

    // Test idx_key_rows_hip (128-wide) consuming generated latent prefix
    const uint32_t idx_hd = 128u;
    std::vector<float> h_knrm(idx_hd, 1.0f);
    float *d_knrm = nullptr, *d_ksc = nullptr;
    uint8_t* d_kout = nullptr;
    assert(hipMalloc(&d_knrm, idx_hd * 4u) == hipSuccess);
    assert(hipMalloc(&d_ksc, n_out * 4u) == hipSuccess);
    assert(hipMalloc(&d_kout, (size_t)n_out * idx_hd) == hipSuccess);
    assert(hipMemcpy(d_knrm, h_knrm.data(), idx_hd * 4u, hipMemcpyHostToDevice) == hipSuccess);
    idx_key_rows_hip(d_lat, d_knrm, d_kout, d_ksc, idx_hd, 0u, n_out, idx_hd, ratio, 64u, rope,
                     1e-6f, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> g_lat((size_t)n_out * hd), g_ksc(n_out);
    assert(hipMemcpy(g_lat.data(), d_lat, g_lat.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(g_ksc.data(), d_ksc, g_ksc.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);
    for (float v : g_lat) assert(std::isfinite(v));
    for (float v : g_ksc) assert(std::isfinite(v) && v > 0.0f);

    // Check latent[g][d] matches CPU pool+RMSNorm for group g=1 (window pos 0..7)
    double ss = 0.0;
    std::vector<double> pooled(hd, 0.0);
    for (uint32_t d = 0; d < hd; ++d) {
      double m = -1e30;
      for (uint32_t r = 0; r < ratio; ++r) {
        m = std::max(m, (double)h_sc[r * w + d] + (double)h_ape[r * w + d]);
        m = std::max(m, (double)h_sc[(r + ratio) * w + hd + d] + (double)h_ape[r * w + hd + d]);
      }
      double den = 0.0, sum = 0.0;
      for (uint32_t r = 0; r < ratio; ++r) {
        const double wp = std::exp((double)h_sc[r * w + d] + (double)h_ape[r * w + d] - m);
        const double wc = std::exp((double)h_sc[(r + ratio) * w + hd + d] + (double)h_ape[r * w + hd + d] - m);
        den += wp + wc;
        sum += wp * h_kv[r * w + d] + wc * h_kv[(r + ratio) * w + hd + d];
      }
      pooled[d] = sum / den;
      ss += pooled[d] * pooled[d];
    }
    const double inv_rms = 1.0 / std::sqrt(ss / hd + 1e-6);
    for (uint32_t d = 0; d < hd; ++d) {
      const float ref = (float)(pooled[d] * inv_rms * h_nrm[d]);
      assert(std::fabs(g_lat[(size_t)d * n_out + 1u] - ref) < 1e-5f);
    }
    for (void* p : {(void*)d_kv, (void*)d_sc, (void*)d_hkv, (void*)d_hsc, (void*)d_ape,
                    (void*)d_nrm, (void*)d_lat, (void*)d_out, (void*)d_knrm, (void*)d_ksc,
                    (void*)d_kout})
      (void)hipFree(p);
  }

  // 19. Test sampler_gpu (greedy tie-break, ballot top_k/top_p compacted & uncompacted, tm layout)
  {
    const uint32_t vocab = 4099u, n = 2u;
    std::vector<float> h_logits_tm((size_t)n * vocab);
    for (uint32_t pos = 0; pos < n; ++pos)
      for (uint32_t v = 0; v < vocab; ++v)
        h_logits_tm[(size_t)pos * vocab + v] = -4.0f + 0.002f * (float)((v * 17u + pos * 53u) % 200u);
    // Inject exact tie at max logit on pos 0 (ids 300 < 2500 -> winner must be 300)
    h_logits_tm[2500] = 12.5f;
    h_logits_tm[300] = 12.5f;
    h_logits_tm[vocab + 777] = 11.25f;

    float* d_logits = nullptr;
    uint32_t* d_q = nullptr;
    float* d_u = nullptr;
    SpecDraw* d_out = nullptr;
    assert(hipMalloc(&d_logits, h_logits_tm.size() * 4u) == hipSuccess);
    assert(hipMalloc(&d_q, n * 4u) == hipSuccess);
    assert(hipMalloc(&d_u, 2u * n * 4u) == hipSuccess);
    assert(hipMalloc(&d_out, n * sizeof(SpecDraw)) == hipSuccess);
    const uint32_t h_q[2] = {300u, 777u};
    const float h_u[4] = {0.35f, 0.72f, 0.15f, 0.85f};
    assert(hipMemcpy(d_logits, h_logits_tm.data(), h_logits_tm.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_q, h_q, n * 4u, hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_u, h_u, 2u * n * 4u, hipMemcpyHostToDevice) == hipSuccess);

    SampleParams sp_g{};
    sp_g.temperature = 0.0f; sp_g.top_p = 1.0f;
    assert(batch_sample_hip(d_logits, vocab, n, 1u, vocab, d_q, d_u, sp_g, d_out, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);
    SpecDraw got[2] = {};
    assert(hipMemcpy(got, d_out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess);
    assert(got[0].tok == 300u && got[0].p_query == 1.0f);
    assert(got[1].tok == 777u && got[1].p_query == 1.0f);

    // Stochastic compacted + uncompacted (flat near-uniform logits trigger > kCandCap)
    SampleParams sp_c{};
    sp_c.temperature = 0.8f; sp_c.top_k = 64u; sp_c.top_p = 0.9f; sp_c.min_p = 0.05f;
    assert(batch_sample_hip(d_logits, vocab, n, 1u, vocab, d_q, d_u, sp_c, d_out, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);
    assert(hipMemcpy(got, d_out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess);
    // On pos 0, logits[300] == logits[2500] == 12.5f, so p(300) ~ 0.5, tok_excl == 2500.
    // On pos 1, logits[777] == 11.25f is uniquely dominant, so p_query > 0.99f and tok == 777.
    assert(got[0].tok == 300u && got[0].tok_excl == 2500u &&
           std::fabs(got[0].p_query - 0.5f) < 1e-3f);
    assert(got[1].tok == 777u && got[1].tok_excl != 777u && got[1].p_query > 0.99f);

    // Also test apply_repeat_penalty_hip (pos 0: token 300 divided by 2.0 -> 2500 becomes dominant)
    uint32_t* d_pen = nullptr;
    assert(hipMalloc(&d_pen, sizeof(uint32_t)) == hipSuccess);
    assert(hipMemcpy(d_pen, h_q, sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(apply_repeat_penalty_hip(d_logits, vocab, n, 1u, vocab, d_pen, 1u, 2.0f, nullptr));
    assert(batch_sample_hip(d_logits, vocab, n, 1u, vocab, d_q, d_u, sp_g, d_out, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);
    assert(hipMemcpy(got, d_out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess);
    assert(got[0].tok == 2500u && got[1].tok == 777u);
    (void)hipFree(d_pen);

    std::vector<float> flat_logits((size_t)n * vocab, 1.0f);
    assert(hipMemcpy(d_logits, flat_logits.data(), flat_logits.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
    SampleParams sp_u{};
    sp_u.temperature = 1.0f; sp_u.top_k = 0u; sp_u.top_p = 0.95f; sp_u.min_p = 0.0f;
    assert(batch_sample_hip(d_logits, vocab, n, 1u, vocab, d_q, d_u, sp_u, d_out, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);
    assert(hipMemcpy(got, d_out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess);
    assert(got[0].tok < vocab && got[0].tok_excl < vocab && got[0].tok_excl != h_q[0]);

    for (void* p : {(void*)d_logits, (void*)d_q, (void*)d_u, (void*)d_out}) (void)hipFree(p);
  }

  // 20. Test idx_quant_rows_hip (dim=128 single-wave & dim=8192 reg-cached) + indexer_scores_hip
  {
    for (uint32_t dim : {128u, 8192u}) {
      const uint32_t nrows = 4u;
      std::vector<float> h_src((size_t)nrows * dim);
      for (size_t i = 0; i < h_src.size(); ++i)
        h_src[i] = std::sin((float)(i % 509u) * 0.027f) * (1.0f + (float)(i / dim));
      std::vector<uint8_t> ref_b(h_src.size());
      std::vector<float> ref_sc(nrows);
      idx_encode_rows(h_src.data(), ref_b.data(), ref_sc.data(), nrows, dim);

      float *d_src = nullptr, *d_sc = nullptr;
      uint8_t* d_dst = nullptr;
      assert(hipMalloc(&d_src, h_src.size() * 4u) == hipSuccess);
      assert(hipMalloc(&d_dst, h_src.size()) == hipSuccess);
      assert(hipMalloc(&d_sc, nrows * 4u) == hipSuccess);
      assert(hipMemcpy(d_src, h_src.data(), h_src.size() * 4u, hipMemcpyHostToDevice) == hipSuccess);
      idx_quant_rows_hip(d_src, d_dst, d_sc, nrows, dim, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      std::vector<uint8_t> got_b(h_src.size());
      std::vector<float> got_sc(nrows);
      assert(hipMemcpy(got_b.data(), d_dst, got_b.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(got_sc.data(), d_sc, got_sc.size() * 4u, hipMemcpyDeviceToHost) == hipSuccess);
      for (uint32_t r = 0; r < nrows; ++r) assert(std::fabs(got_sc[r] - ref_sc[r]) <= 1e-6f);
      for (size_t i = 0; i < got_b.size(); ++i) assert(got_b[i] == ref_b[i]);
      (void)hipFree(d_src); (void)hipFree(d_dst); (void)hipFree(d_sc);
    }
  }

  // 21. Test hybrid_tile128 bucket 3 (NT=8, max_nt=128) in moe_plan_hip + moe_plan_group_max_nt
  {
    assert(moe_plan_group_max_nt(0u) == 16u && moe_plan_group_max_nt(1u) == 32u &&
           moe_plan_group_max_nt(2u) == 64u && moe_plan_group_max_nt(3u) == 128u);
    assert(moe_plan_group_max_nt(4u) == 16u && moe_plan_group_max_nt(5u) == 32u &&
           moe_plan_group_max_nt(6u) == 64u && moe_plan_group_max_nt(7u) == 128u);

    const uint32_t ne = 256u, k = 2u, nb = 140u;
    const uint32_t total = nb * k;
    const uint32_t cap = moe_plan_tile_cap(nb, k, ne);
    std::vector<uint64_t> h_plc(ne, 0x300000ull);
    h_plc[0] = 0x100000ull | 1ull; // resident
    h_plc[1] = 0x200000ull;        // pooled
    std::vector<uint32_t> h_sel(total);
    std::vector<float> h_wt(total, 0.5f);
    for (uint32_t t = 0; t < nb; ++t) {
      h_sel[t * k + 0u] = 0u;
      h_sel[t * k + 1u] = 1u;
    }
    ExpertShardLayout sh{};
    sh.stride = 4096u;
    for (int pl = 0; pl < 3; ++pl) {
      sh.data_off[pl] = 256u * (uint32_t)(pl + 1);
      sh.scale_off[pl] = 1024u + 256u * (uint32_t)pl;
      sh.rscale_off[pl] = 2048u + 256u * (uint32_t)pl;
    }

    uint32_t *d_sel = nullptr, *d_tok = nullptr, *d_ord = nullptr, *d_run = nullptr,
             *d_col = nullptr, *d_cnt = nullptr;
    uint64_t* d_plc = nullptr;
    float *d_wt = nullptr, *d_wto = nullptr;
    uint16_t *d_bg = nullptr, *d_bu = nullptr, *d_bydn = nullptr;
    ExpertTile* d_tiles = nullptr;
    MoePlanStatus* d_st = nullptr;
    assert(hipMalloc(&d_sel, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wt, total * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_plc, ne * sizeof(uint64_t)) == hipSuccess);
    assert(hipMalloc(&d_tok, 2u * total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wto, total * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_ord, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_run, (nb + 1u) * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_col, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_tiles, (size_t)3u * 8u * cap * sizeof(ExpertTile)) == hipSuccess);
    assert(hipMalloc(&d_cnt, 8u * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_st, sizeof(MoePlanStatus)) == hipSuccess);
    assert(hipMalloc(&d_bg, 64u) == hipSuccess);
    assert(hipMalloc(&d_bu, 64u) == hipSuccess);
    assert(hipMalloc(&d_bydn, 64u) == hipSuccess);
    assert(hipMemset(d_st, 0, sizeof(MoePlanStatus)) == hipSuccess);
    assert(hipMemcpy(d_sel, h_sel.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_wt, h_wt.data(), total * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_plc, h_plc.data(), ne * sizeof(uint64_t), hipMemcpyHostToDevice) == hipSuccess);

    MoePlanArgs A{};
    A.sel = d_sel; A.wt = d_wt; A.plc = d_plc;
    A.layer = 0; A.nb_tok = nb; A.k = k; A.n_expert = ne;
    A.tok = d_tok; A.wt_out = d_wto; A.ord = d_ord; A.run = d_run; A.slot_col = d_col;
    A.tiles = d_tiles; A.counts = d_cnt; A.status = d_st;
    A.shard = sh; A.bg = d_bg; A.bu = d_bu; A.bydn = d_bydn; A.tile_cap = cap;
    A.hybrid_tile128 = true;
    assert(moe_plan_hip(A, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);

    MoePlanStatus st{};
    std::vector<uint32_t> g_cnt(8u);
    std::vector<ExpertTile> g_tiles((size_t)3u * 8u * cap);
    assert(hipMemcpy(&st, d_st, sizeof(st), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(g_cnt.data(), d_cnt, 8u * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(g_tiles.data(), d_tiles, g_tiles.size() * sizeof(ExpertTile), hipMemcpyDeviceToHost) == hipSuccess);
    assert(st.taken == total && st.bad == 0u && st.over == 0u);
    assert(g_cnt[0] == 1u && g_cnt[1] == 0u && g_cnt[2] == 2u && g_cnt[3] == 0u);
    assert(g_cnt[4] == 1u && g_cnt[5] == 0u && g_cnt[6] == 0u && g_cnt[7] == 1u);
    assert(g_tiles[0u * cap + 0u].nt == 12u);
    assert(g_tiles[2u * cap + 0u].nt == 64u && g_tiles[2u * cap + 1u].nt == 64u);
    assert(g_tiles[4u * cap + 0u].nt == 12u);
    assert(g_tiles[7u * cap + 0u].nt == 128u);

    for (void* p : {(void*)d_sel, (void*)d_wt, (void*)d_plc, (void*)d_tok, (void*)d_wto,
                    (void*)d_ord, (void*)d_run, (void*)d_col, (void*)d_tiles, (void*)d_cnt,
                    (void*)d_st, (void*)d_bg, (void*)d_bu, (void*)d_bydn})
      (void)hipFree(p);
  }

  // 22. Test DenseGpu::kv_part_save_async / kv_part_load_async / kv_sync_all / pin_host / unpin_host
  {
    assert(!DenseGpu::pin_host(nullptr, 0));
    DenseGpu::unpin_host(nullptr);
    assert(hipGetLastError() == hipSuccess);

    DenseGpu dg;
    std::string dg_err;
    assert(dg.init(64, 64, &dg_err, 1));
    uint32_t comp_rows[1] = {16};
    assert(dg.attn_init(1, 16, 64, 16, 16, comp_rows, KvDtype::FP8R, 64, 64, &dg_err));
    uint32_t ratios[1] = {4};
    assert(dg.reserve_runtime(1, ratios, 64, kIdxDim, 4, &dg_err));

    const DenseGpu::KvPart parts[] = {
        DenseGpu::KvPart::Raw,  DenseGpu::KvPart::Comp, DenseGpu::KvPart::Idx8,
        DenseGpu::KvPart::IdxScale, DenseGpu::KvPart::Hist, DenseGpu::KvPart::IdxHist,
    };
    for (DenseGpu::KvPart part : parts) {
      uint64_t rb = 0, rows = 0;
      if (!dg.kv_part_geometry(part, 0, &rb, &rows)) continue;
      const uint64_t nrows = std::min<uint64_t>(rows, 8);
      const size_t bytes = (size_t)(nrows * rb);
      const size_t alloc_bytes = (bytes + 4095u) & ~(size_t)4095u;

      void *raw_src = nullptr, *raw_sync = nullptr, *raw_async = nullptr;
      assert(::posix_memalign(&raw_src, 4096, alloc_bytes) == 0);
      assert(::posix_memalign(&raw_sync, 4096, alloc_bytes) == 0);
      assert(::posix_memalign(&raw_async, 4096, alloc_bytes) == 0);

      auto* src = static_cast<uint8_t*>(raw_src);
      auto* dst_sync = static_cast<uint8_t*>(raw_sync);
      auto* dst_async = static_cast<uint8_t*>(raw_async);
      for (size_t i = 0; i < bytes; ++i)
        src[i] = (uint8_t)(((i * 53 + (uint8_t)part * 19 + 7) ^ (i >> 4)) & 0xFF);
      std::memset(dst_sync, 0, bytes);
      std::memset(dst_async, 0, bytes);

      const bool p_src = DenseGpu::pin_host(src, alloc_bytes);
      const bool p_dst = DenseGpu::pin_host(dst_async, alloc_bytes);

      assert(dg.kv_part_load(part, 0, 0, 0, nrows, src));
      assert(dg.kv_part_save(part, 0, 0, 0, nrows, dst_sync));
      assert(std::memcmp(src, dst_sync, bytes) == 0);

      // Overwrite device rows with zeros synchronously, then reload via async + kv_sync_all.
      std::vector<uint8_t> zeros(bytes, 0);
      assert(dg.kv_part_load(part, 0, 0, 0, nrows, zeros.data()));
      assert(dg.kv_part_load_async(part, 0, 0, 0, nrows, src));
      assert(dg.kv_sync_all());
      assert(dg.kv_part_save_async(part, 0, 0, 0, nrows, dst_async));
      assert(dg.kv_sync_rank(0));
      assert(std::memcmp(dst_sync, dst_async, bytes) == 0);

      if (p_src) DenseGpu::unpin_host(src);
      if (p_dst) DenseGpu::unpin_host(dst_async);
      std::free(raw_src);
      std::free(raw_sync);
      std::free(raw_async);
    }
  }

  // 23. F7: e4m3_enc HW vs SW parity + subnormal boundary [7.5*2^-9, 8*2^-9) + host KV encode
  {
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> inputs = {
        +0.0f, -0.0f, 0.015f, -0.015f,
        7.5f / 512.0f, -7.5f / 512.0f, 7.4f / 512.0f,
        0.001f, 0.001953125f, 0x1p-6f,
        448.0f, -448.0f, 500.0f, -500.0f, inf, -inf,
    };
    for (int i = 0; i < 256; ++i) {
      inputs.push_back((float)i * (1.0f / 1024.0f));
      inputs.push_back(-(float)i * (1.0f / 1024.0f));
      inputs.push_back((float)(i - 128) * 4.25f);
    }
    const uint32_t n = (uint32_t)inputs.size();
    float* d_in = nullptr;
    uint8_t *d_hw = nullptr, *d_sw = nullptr;
    assert(hipMalloc(&d_in, n * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_hw, n) == hipSuccess);
    assert(hipMalloc(&d_sw, n) == hipSuccess);
    assert(hipMemcpy(d_in, inputs.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    hipLaunchKernelGGL(test_e4m3_enc_kernel, dim3((n + 255u) / 256u), dim3(256u), 0, nullptr,
                       d_in, d_hw, d_sw, n);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<uint8_t> hw(n), sw(n);
    assert(hipMemcpy(hw.data(), d_hw, n, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(sw.data(), d_sw, n, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t i = 0; i < n; ++i) assert(hw[i] == sw[i]);
    assert(hw[0] == 0x00u);   // +0.0
    assert(hw[1] == 0x80u);   // -0.0
    assert(hw[2] == 0x08u);   // 0.015f in [7.5*2^-9, 8*2^-9)
    assert(hw[3] == 0x88u);   // -0.015f
    assert(hw[4] == 0x08u);   // exact 7.5 * 2^-9 tie-to-even
    assert(hw[10] == 0x7Eu);  // +448
    assert(hw[11] == 0xFEu);  // -448
    assert(hw[12] == 0x7Eu);  // +500 clamped
    assert(hw[13] == 0xFEu);  // -500 clamped
    assert(hw[14] == 0x7Eu);  // +Inf clamped
    assert(hw[15] == 0xFEu);  // -Inf clamped
    (void)hipFree(d_in);
    (void)hipFree(d_hw);
    (void)hipFree(d_sw);

    // Host kv_encode_row(FP8) & kv_encode_fp8r_row with unit scale (amax = 448.0f per block)
    std::vector<float> row(kFlashWidth, 0.0f);
    for (uint32_t i = 0; i < kFlashWidth; ++i) {
      if ((i % 64u) == 0u) row[i] = 448.0f;
      else if ((i % 64u) == 1u) row[i] = -448.0f;
      else if ((i % 64u) == 2u) row[i] = 0.015f;
      else if ((i % 64u) == 3u) row[i] = -0.015f;
      else if ((i % 64u) == 4u) row[i] = -0.0f;
      else row[i] = std::sin((float)i * 0.37f) * 20.0f;
    }
    float* d_r = nullptr;
    uint8_t *d_rhw = nullptr, *d_rsw = nullptr;
    assert(hipMalloc(&d_r, kFlashWidth * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_rhw, kFlashWidth) == hipSuccess);
    assert(hipMalloc(&d_rsw, kFlashWidth) == hipSuccess);
    assert(hipMemcpy(d_r, row.data(), kFlashWidth * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);
    hipLaunchKernelGGL(test_e4m3_enc_kernel, dim3(2), dim3(256), 0, nullptr, d_r, d_rhw, d_rsw,
                       kFlashWidth);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<uint8_t> r_hw(kFlashWidth);
    assert(hipMemcpy(r_hw.data(), d_rhw, kFlashWidth, hipMemcpyDeviceToHost) == hipSuccess);
    (void)hipFree(d_r);
    (void)hipFree(d_rhw);
    (void)hipFree(d_rsw);

    std::vector<uint8_t> dst_fp8(kv_row_bytes(KvDtype::FP8, kFlashWidth, 0u), 0);
    std::vector<uint8_t> dst_fp8r(kFlashRowBytes, 0);
    kv_encode_row(KvDtype::FP8, row.data(), kFlashWidth, 0u, dst_fp8.data());
    kv_encode_fp8r_row(row.data(), dst_fp8r.data());
    assert(std::memcmp(dst_fp8.data(), r_hw.data(), kFlashWidth) == 0);
    assert(std::memcmp(dst_fp8r.data(), r_hw.data(), kFlashWidth) == 0);
  }

  // 24. F4: tp_ar_fp8 clamping (>448, +-Inf) & cross-rank memcmp == 0
  {
    int ndev = 0, can01 = 0, can10 = 0;
    if (hipGetDeviceCount(&ndev) == hipSuccess && ndev >= 2 &&
        hipDeviceCanAccessPeer(&can01, 0, 1) == hipSuccess && can01 &&
        hipDeviceCanAccessPeer(&can10, 1, 0) == hipSuccess && can10) {
      TpContext ctx;
      std::string err;
      const size_t max_n = 1049003u;
      assert(tp_init(&ctx, max_n, &err));
      float *d_in[2] = {}, *d_out[2] = {};
      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        assert(hipMalloc(&d_in[r], max_n * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_out[r], max_n * sizeof(float)) == hipSuccess);
      }
      const float inf = std::numeric_limits<float>::infinity();
      for (size_t n : {size_t(16384u), max_n}) {
        // (a) finite > 448 values: outputs must stay finite, accurate, and bit-identical across ranks
        std::vector<float> h0(n), h1(n);
        for (size_t i = 0; i < n; ++i) {
          h0[i] = std::sin((float)(i % 251u) * 0.07f) * 3.0f;
          h1[i] = std::cos((float)(i % 199u) * 0.05f) * 2.5f;
        }
        h0[0] = 500.0f;    h1[0] = 600.0f;
        h0[129] = -700.0f; h1[129] = -550.0f;
        h0[n - 1] = 12000.0f; h1[n - 1] = -900.0f;
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
          assert(hipMemcpy(d_in[r], (r == 0 ? h0 : h1).data(), n * sizeof(float),
                           hipMemcpyHostToDevice) == hipSuccess);
        }
        tp_all_reduce(ctx, d_in, d_out, n, TpReduce::Fp8);
        tp_sync(ctx);
        tp_check(ctx);
        std::vector<float> out0(n), out1(n);
        assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
        assert(hipMemcpy(out0.data(), d_out[0], n * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess);
        assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
        assert(hipMemcpy(out1.data(), d_out[1], n * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess);
        assert(std::memcmp(out0.data(), out1.data(), n * sizeof(float)) == 0);
        for (size_t i : {size_t(0), size_t(129), n - 1}) {
          assert(std::isfinite(out0[i]));
          const float ref = h0[i] + h1[i];
          assert(std::fabs(out0[i] - ref) <= std::max(0.5f, 0.1f * std::fabs(ref)));
        }

        // (b) +-Inf values: E4M3 codes must clamp finite & cross-rank outputs stay bit-identical
        h0[3] = inf;  h1[3] = inf;
        h0[256] = -inf; h1[256] = -inf;
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
          assert(hipMemcpy(d_in[r], (r == 0 ? h0 : h1).data(), n * sizeof(float),
                           hipMemcpyHostToDevice) == hipSuccess);
        }
        tp_all_reduce(ctx, d_in, d_out, n, TpReduce::Fp8);
        tp_sync(ctx);
        tp_check(ctx);
        assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
        assert(hipMemcpy(out0.data(), d_out[0], n * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess);
        assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
        assert(hipMemcpy(out1.data(), d_out[1], n * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess);
        assert(std::memcmp(out0.data(), out1.data(), n * sizeof(float)) == 0);
      }
      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        (void)hipFree(d_in[r]);
        (void)hipFree(d_out[r]);
      }
      tp_free(&ctx);
    }
  }

  // 25. F5: engram_mxfp8_requant_fp8_hip bit-exact vs gather + batch_quant_tm_fp8_hip
  {
    assert(!engram_mxfp8_requant_fp8_hip(nullptr, nullptr, nullptr, 1u, 6000u, nullptr));
    assert(engram_mxfp8_requant_fp8_hip(nullptr, nullptr, nullptr, 0u, 6144u, nullptr));
    const uint32_t in_dim = 6144u;
    const uint32_t sc_per_tok = in_dim / 32u;
    const uint32_t qs_per_tok = in_dim / 128u;
    const uint32_t stride = in_dim + sc_per_tok;
    for (uint32_t nt : {1u, 6u, 64u}) {
      std::vector<uint8_t> h_raw((size_t)nt * stride, 0u);
      std::vector<float> h_f32((size_t)nt * in_dim, 0.0f);
      for (uint32_t t = 0; t < nt; ++t) {
        uint8_t* w = h_raw.data() + (size_t)t * stride;
        uint8_t* sc = w + in_dim;
        float* dst = h_f32.data() + (size_t)t * in_dim;
        // Last token (when nt > 1) exercises OOR all-zero row (w=0x00, sc=0xFF)
        if (nt > 1u && t == nt - 1u) {
          std::memset(w, 0x00, in_dim);
          std::memset(sc, 0xFF, sc_per_tok);
          continue;
        }
        for (uint32_t b = 0; b < sc_per_tok; ++b) {
          uint8_t e = (uint8_t)(0x79u + ((t * 13u + b) % 10u));
          if (b == 0u) e = 0x00u;                    // 2^-127 smallest scale
          else if (b == 5u || b == 6u) e = 0xFFu;    // invalid/OOR block -> 0.0f
          sc[b] = e;
          const float s = (e == 0xFFu) ? 0.0f : std::ldexp(1.0f, (int)e - 127);
          for (uint32_t k = 0; k < 32u; ++k) {
            uint8_t byte = (uint8_t)((t * 37u + b * 29u + k * 11u) & 0xFFu);
            if ((byte & 0x7Fu) == 0x7Fu) byte = (uint8_t)(byte ^ 0x01u);  // keep finite E4M3
            w[b * 32u + k] = byte;
            dst[b * 32u + k] = mxfp8_e4m3_to_f32_ref(byte) * s;
          }
        }
      }

      uint8_t *d_raw = nullptr, *d_q8_ref = nullptr, *d_q8_got = nullptr;
      float *d_f32 = nullptr, *d_qs_ref = nullptr, *d_qs_got = nullptr;
      assert(hipMalloc(&d_raw, h_raw.size()) == hipSuccess);
      assert(hipMalloc(&d_f32, h_f32.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_q8_ref, (size_t)nt * in_dim) == hipSuccess);
      assert(hipMalloc(&d_q8_got, (size_t)nt * in_dim) == hipSuccess);
      assert(hipMalloc(&d_qs_ref, (size_t)nt * qs_per_tok * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_qs_got, (size_t)nt * qs_per_tok * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_f32, h_f32.data(), h_f32.size() * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);

      batch_quant_tm_fp8_hip(d_f32, d_q8_ref, d_qs_ref, in_dim, nt, nullptr);
      assert(engram_mxfp8_requant_fp8_hip(d_raw, d_q8_got, d_qs_got, nt, in_dim, nullptr));
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<uint8_t> q8_ref((size_t)nt * in_dim), q8_got((size_t)nt * in_dim);
      std::vector<float> qs_ref((size_t)nt * qs_per_tok), qs_got((size_t)nt * qs_per_tok);
      assert(hipMemcpy(q8_ref.data(), d_q8_ref, q8_ref.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(q8_got.data(), d_q8_got, q8_got.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(qs_ref.data(), d_qs_ref, qs_ref.size() * sizeof(float),
                       hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(qs_got.data(), d_qs_got, qs_got.size() * sizeof(float),
                       hipMemcpyDeviceToHost) == hipSuccess);
      assert(std::memcmp(q8_got.data(), q8_ref.data(), q8_ref.size()) == 0);
      assert(std::memcmp(qs_got.data(), qs_ref.data(), qs_ref.size() * sizeof(float)) == 0);

      for (void* p : {(void*)d_raw, (void*)d_f32, (void*)d_q8_ref, (void*)d_q8_got,
                      (void*)d_qs_ref, (void*)d_qs_got})
        (void)hipFree(p);
    }
  }

  std::printf("PASS: test_gpu_ports on gfx1201\n");
  return 0;
}
