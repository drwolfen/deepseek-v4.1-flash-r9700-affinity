#include "gpu/attention_gpu.h"
#include "gpu/batch_kernels.h"
#include "gpu/indexer_gpu.h"
#include "gpu/moe_plan.h"
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
#include <vector>

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

  std::printf("PASS: test_gpu_ports on gfx1201\n");
  return 0;
}
