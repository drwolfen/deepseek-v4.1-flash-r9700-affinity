#include "gpu/attention_gpu.h"
#include "gpu/batch_kernels.h"
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

  std::printf("PASS: test_gpu_ports on gfx1201\n");
  return 0;
}
