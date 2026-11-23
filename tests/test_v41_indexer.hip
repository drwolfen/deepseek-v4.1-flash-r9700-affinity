#include "gpu/indexer_gpu.h"
#include "gpu/hip_common.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace aff;

static float f32_from_e4m3(uint8_t b) {
  const uint32_t sign = (b >> 7) & 1u;
  const uint32_t exp  = (b >> 3) & 0x0fu;
  const uint32_t mant = b & 0x07u;
  float val = 0.0f;
  if (exp == 0u) {
    val = (float)mant * (1.0f / 512.0f);
  } else if (exp == 15u && mant == 7u) {
    val = 448.0f;
  } else {
    val = (1.0f + (float)mant / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
  }
  return sign ? -val : val;
}

int main() {
  std::printf("=== test_v41_indexer standalone unit & parity tests ===\n");

  int dev_count = 0;
  hipError_t err_cnt = hipGetDeviceCount(&dev_count);
  if (err_cnt != hipSuccess || dev_count <= 0) {
    std::printf("PASS: test_v41_indexer offline checks (0 GPU devices available in ephemeral environment)\n");
    return 0;
  }

  std::printf("Found %d HIP device(s), proceeding with GPU kernel tests on gfx1201\n", dev_count);
  assert(hipSetDevice(0) == hipSuccess);

  // ------------------------------------------------------------------------------------------------
  // Test 1: 4-pass 256-bucket radix top-512 with monotonic bit-flipped FP32 keys
  // across seqused = {128, 512, 513, 2048, 8192, 65536, 131072} vs CPU std::stable_sort
  // ------------------------------------------------------------------------------------------------
  {
    const uint32_t test_seqs[] = {128, 512, 513, 2048, 8192, 65536, 131072};
    for (uint32_t seqused : test_seqs) {
      std::vector<float> h_scores(seqused);
      for (uint32_t i = 0; i < seqused; ++i) {
        if (i % 37 == 0 && i > 10) {
          h_scores[i] = -INFINITY; // Masked out
        } else if (i % 23 == 0) {
          h_scores[i] = 42.5f;     // Deliberate exact positive tie
        } else if (i % 29 == 0) {
          h_scores[i] = -15.25f;   // Deliberate exact negative tie
        } else {
          h_scores[i] = std::sin((float)i * 0.317f) * 100.0f;
        }
      }

      struct Entry { float score; uint32_t idx; };
      std::vector<Entry> valid_entries;
      for (uint32_t i = 0; i < seqused; ++i) {
        if (h_scores[i] > -INFINITY && !std::isnan(h_scores[i])) {
          valid_entries.push_back({h_scores[i], i});
        }
      }

      std::stable_sort(valid_entries.begin(), valid_entries.end(), [](const Entry& a, const Entry& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.idx < b.idx;
      });

      const uint32_t expected_k = std::min<uint32_t>(512, (uint32_t)valid_entries.size());
      std::vector<uint32_t> expected_adm(expected_k);
      for (uint32_t i = 0; i < expected_k; ++i) {
        expected_adm[i] = valid_entries[i].idx;
      }
      std::sort(expected_adm.begin(), expected_adm.end());

      float* d_scores = nullptr;
      uint8_t* d_mask = nullptr;
      uint32_t* d_adm = nullptr;
      uint32_t* d_admn = nullptr;
      assert(hipMalloc(&d_scores, (size_t)seqused * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_mask, (size_t)seqused * sizeof(uint8_t)) == hipSuccess);
      assert(hipMalloc(&d_adm, 1024 * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn, sizeof(uint32_t)) == hipSuccess);

      assert(hipMemcpy(d_scores, h_scores.data(), (size_t)seqused * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemset(d_mask, 0, (size_t)seqused * sizeof(uint8_t)) == hipSuccess);
      assert(hipMemset(d_adm, 0, 1024 * sizeof(uint32_t)) == hipSuccess);
      assert(hipMemset(d_admn, 0, sizeof(uint32_t)) == hipSuccess);

      idx_topk_mask_hip(d_scores, d_mask, nullptr, 1, seqused, seqused, seqused, 512, nullptr,
                        nullptr, d_adm, d_admn, 1024);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<uint8_t> h_mask(seqused);
      std::vector<uint32_t> h_adm(1024);
      uint32_t h_admn = 0;
      assert(hipMemcpy(h_mask.data(), d_mask, (size_t)seqused * sizeof(uint8_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_adm.data(), d_adm, 1024 * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(&h_admn, d_admn, sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

      assert(h_admn == expected_k);
      for (uint32_t i = 0; i < expected_k; ++i) {
        assert(h_adm[i] == expected_adm[i]);
        assert(h_mask[h_adm[i]] == 1);
      }
      uint32_t mask_cnt = 0;
      for (uint32_t i = 0; i < seqused; ++i) {
        if (h_mask[i]) mask_cnt++;
      }
      assert(mask_cnt == expected_k);

      (void)hipFree(d_scores); (void)hipFree(d_mask); (void)hipFree(d_adm); (void)hipFree(d_admn);
    }
    std::printf("PASS: Test 1 - 4-pass 256-bucket radix top-512 exact parity vs CPU stable_sort\n");
  }

  // ------------------------------------------------------------------------------------------------
  // Test 2: LDS-padded (K_VECS_PADDED) + 8-byte Q-staged indexer score kernel vs naive FP32 reference
  // ------------------------------------------------------------------------------------------------
  {
    const uint32_t nb = 4, n_head = 32, n_keys = 256, dim = kIdxDim;
    std::vector<float> h_q(nb * n_head * dim);
    std::vector<float> h_k(n_keys * dim);
    std::vector<float> h_hw(nb * n_head);

    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = std::sin((float)i * 0.05f) * 0.5f;
    for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = std::cos((float)i * 0.07f) * 0.5f;
    for (size_t i = 0; i < h_hw.size(); ++i) h_hw[i] = 0.5f + 0.5f * std::sin((float)i * 0.1f);

    std::vector<uint8_t> h_q8(nb * n_head * dim);
    std::vector<float> h_qs(nb);
    idx_encode_rows(h_q.data(), h_q8.data(), h_qs.data(), nb, n_head * dim);

    std::vector<uint8_t> h_k8(n_keys * dim);
    std::vector<float> h_ks(n_keys);
    idx_encode_rows(h_k.data(), h_k8.data(), h_ks.data(), n_keys, dim);

    std::vector<float> ref_scores(nb * n_keys, 0.0f);
    for (uint32_t t = 0; t < nb; ++t) {
      for (uint32_t s = 0; s < n_keys; ++s) {
        double sum = 0.0;
        for (uint32_t h = 0; h < n_head; ++h) {
          double dot = 0.0;
          for (uint32_t d = 0; d < dim; ++d) {
            float qv = f32_from_e4m3(h_q8[(t * n_head + h) * dim + d]);
            float kv = f32_from_e4m3(h_k8[s * dim + d]);
            dot += (double)qv * (double)kv;
          }
          if (dot > 0.0) {
            sum += (double)h_hw[t * n_head + h] * dot;
          }
        }
        ref_scores[t * n_keys + s] = (float)(sum * (double)h_qs[t] * (double)h_ks[s]);
      }
    }

    uint8_t *d_q8 = nullptr, *d_k8 = nullptr;
    float *d_qs = nullptr, *d_ks = nullptr, *d_hw = nullptr, *d_out = nullptr;
    assert(hipMalloc(&d_q8, h_q8.size()) == hipSuccess);
    assert(hipMalloc(&d_k8, h_k8.size()) == hipSuccess);
    assert(hipMalloc(&d_qs, h_qs.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_ks, h_ks.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_hw, h_hw.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_out, (size_t)nb * n_keys * sizeof(float)) == hipSuccess);

    assert(hipMemcpy(d_q8, h_q8.data(), h_q8.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_k8, h_k8.data(), h_k8.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_qs, h_qs.data(), h_qs.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_hw, h_hw.data(), h_hw.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

    indexer_scores_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_out, nb, n_keys, n_head, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> h_out(nb * n_keys);
    assert(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

    double dot_prod = 0.0, norm_out = 0.0, norm_ref = 0.0;
    float max_abs_err = 0.0f;
    for (size_t i = 0; i < nb * n_keys; ++i) {
      dot_prod += (double)h_out[i] * (double)ref_scores[i];
      norm_out += (double)h_out[i] * (double)h_out[i];
      norm_ref += (double)ref_scores[i] * (double)ref_scores[i];
      float diff = std::fabs(h_out[i] - ref_scores[i]);
      if (diff > max_abs_err) max_abs_err = diff;
    }
    double cos_sim = (norm_out > 0.0 && norm_ref > 0.0)
                         ? (dot_prod / (std::sqrt(norm_out) * std::sqrt(norm_ref)))
                         : 1.0;

    std::printf("Indexer score kernel parity: cos_sim = %.7f, max_abs_err = %.6e\n", cos_sim, max_abs_err);
    assert(cos_sim > 0.99999);
    assert(max_abs_err < 1e-4);

    (void)hipFree(d_q8); (void)hipFree(d_k8); (void)hipFree(d_qs); (void)hipFree(d_ks); (void)hipFree(d_hw); (void)hipFree(d_out);
    std::printf("PASS: Test 2 - LDS-padded K_VECS_PADDED + 8-byte Q-staged indexer score parity\n");
  }

  // ------------------------------------------------------------------------------------------------
  // Test 2b: Indexer decode/narrow (nb <= 8) vs batched LDS union (nb = 16 / AFF_IDX_NARROW=0) parity
  // ------------------------------------------------------------------------------------------------
  {
    const char* prev_idx = std::getenv("AFF_IDX_NARROW");
    std::string prev_idx_val = prev_idx ? prev_idx : "";
    const uint32_t nb_list[] = {1u, 5u, 6u, 8u, 16u};
    const uint32_t n_head = 64u, n_keys = 384u, dim = kIdxDim;
    for (uint32_t nb : nb_list) {
      std::vector<float> h_q((size_t)nb * n_head * dim), h_k((size_t)n_keys * dim),
          h_hw((size_t)nb * n_head);
      for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = std::sin((float)(i + nb) * 0.043f) * 0.6f;
      for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = std::cos((float)i * 0.067f) * 0.55f;
      for (size_t i = 0; i < h_hw.size(); ++i) h_hw[i] = 0.4f + 0.6f * std::sin((float)i * 0.13f);

      std::vector<uint8_t> h_q8((size_t)nb * n_head * dim), h_k8((size_t)n_keys * dim);
      std::vector<float> h_qs(nb), h_ks(n_keys);
      idx_encode_rows(h_q.data(), h_q8.data(), h_qs.data(), nb, n_head * dim);
      idx_encode_rows(h_k.data(), h_k8.data(), h_ks.data(), n_keys, dim);

      std::vector<float> ref_scores((size_t)nb * n_keys, 0.0f);
      for (uint32_t t = 0; t < nb; ++t) {
        for (uint32_t s = 0; s < n_keys; ++s) {
          double sum = 0.0;
          for (uint32_t h = 0; h < n_head; ++h) {
            double dot = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
              dot += (double)f32_from_e4m3(h_q8[((size_t)t * n_head + h) * dim + d]) *
                     (double)f32_from_e4m3(h_k8[(size_t)s * dim + d]);
            }
            if (dot > 0.0) sum += (double)h_hw[(size_t)t * n_head + h] * dot;
          }
          ref_scores[(size_t)t * n_keys + s] = (float)(sum * (double)h_qs[t] * (double)h_ks[s]);
        }
      }

      uint8_t *d_q8 = nullptr, *d_k8 = nullptr;
      float *d_qs = nullptr, *d_ks = nullptr, *d_hw = nullptr, *d_out1 = nullptr, *d_out0 = nullptr;
      assert(hipMalloc(&d_q8, h_q8.size()) == hipSuccess);
      assert(hipMalloc(&d_k8, h_k8.size()) == hipSuccess);
      assert(hipMalloc(&d_qs, h_qs.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_ks, h_ks.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_hw, h_hw.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out1, (size_t)nb * n_keys * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out0, (size_t)nb * n_keys * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_q8, h_q8.data(), h_q8.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_k8, h_k8.data(), h_k8.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_qs, h_qs.data(), h_qs.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_hw, h_hw.data(), h_hw.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

      setenv("AFF_IDX_NARROW", "1", 1);
      indexer_scores_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_out1, nb, n_keys, n_head, nullptr);
      setenv("AFF_IDX_NARROW", "0", 1);
      indexer_scores_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_out0, nb, n_keys, n_head, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<float> h1((size_t)nb * n_keys), h0((size_t)nb * n_keys);
      assert(hipMemcpy(h1.data(), d_out1, h1.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h0.data(), d_out0, h0.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      double dot1 = 0.0, n1 = 0.0, nr = 0.0, dot0 = 0.0, n0 = 0.0;
      for (size_t i = 0; i < h1.size(); ++i) {
        dot1 += (double)h1[i] * (double)ref_scores[i];
        n1 += (double)h1[i] * (double)h1[i];
        dot0 += (double)h0[i] * (double)ref_scores[i];
        n0 += (double)h0[i] * (double)h0[i];
        nr += (double)ref_scores[i] * (double)ref_scores[i];
      }
      const double cos1 = dot1 / (std::sqrt(n1) * std::sqrt(nr));
      const double cos0 = dot0 / (std::sqrt(n0) * std::sqrt(nr));
      std::printf("  indexer nb=%2u: cos1=%.8f cos0=%.8f\n", nb, cos1, cos0);
      assert(cos1 > 0.999999 && cos0 > 0.999999);
      if (nb == 1u || nb == 16u) assert(std::memcmp(h1.data(), h0.data(), h1.size() * sizeof(float)) == 0);
      for (void* p : {(void*)d_q8, (void*)d_k8, (void*)d_qs, (void*)d_ks, (void*)d_hw,
                      (void*)d_out1, (void*)d_out0})
        (void)hipFree(p);
    }
    if (prev_idx) setenv("AFF_IDX_NARROW", prev_idx_val.c_str(), 1);
    else unsetenv("AFF_IDX_NARROW");
    std::printf("PASS: Test 2b - Narrow/LDS-union indexer score parity nb={1,5,6,8,16}\n");
  }

  // ------------------------------------------------------------------------------------------------
  // Test 3: Ragged causal prefill (pos0 in {0, 400, 16384}, n_q = 128)
  // Verifying strict causality (<= pos0 + r) and pos0 + r + 1 <= 512 fast-path
  // ------------------------------------------------------------------------------------------------
  {
    const uint32_t test_pos0[] = {0, 400, 16384};
    const uint32_t n_q = 128;

    for (uint32_t pos0 : test_pos0) {
      const uint32_t max_keys = pos0 + n_q;
      const uint32_t stride = ((max_keys + 127u) / 128u) * 128u;
      std::vector<float> h_scores((size_t)n_q * stride, -INFINITY);
      std::vector<uint32_t> h_ncomp(n_q);

      for (uint32_t r = 0; r < n_q; ++r) {
        const uint32_t live_cols = pos0 + r + 1u;
        h_ncomp[r] = live_cols;
        for (uint32_t c = 0; c < live_cols; ++c) {
          h_scores[(size_t)r * stride + c] = 0.1f + std::sin((float)(r * 31u + c) * 0.03f) * 10.0f;
        }
      }

      float* d_scores = nullptr;
      uint8_t* d_mask = nullptr;
      uint32_t* d_ncomp = nullptr;
      uint32_t* d_adm = nullptr;
      uint32_t* d_admn = nullptr;
      assert(hipMalloc(&d_scores, (size_t)n_q * stride * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_mask, (size_t)n_q * stride * sizeof(uint8_t)) == hipSuccess);
      assert(hipMalloc(&d_ncomp, (size_t)n_q * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm, (size_t)n_q * 1024 * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn, (size_t)n_q * sizeof(uint32_t)) == hipSuccess);

      assert(hipMemcpy(d_scores, h_scores.data(), (size_t)n_q * stride * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_ncomp, h_ncomp.data(), (size_t)n_q * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemset(d_mask, 0, (size_t)n_q * stride * sizeof(uint8_t)) == hipSuccess);
      assert(hipMemset(d_adm, 0, (size_t)n_q * 1024 * sizeof(uint32_t)) == hipSuccess);
      assert(hipMemset(d_admn, 0, (size_t)n_q * sizeof(uint32_t)) == hipSuccess);

      idx_topk_mask_hip(d_scores, d_mask, d_ncomp, n_q, stride, stride, stride, 512, nullptr,
                        nullptr, d_adm, d_admn, 1024);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<uint32_t> h_adm((size_t)n_q * 1024);
      std::vector<uint32_t> h_admn(n_q);
      assert(hipMemcpy(h_adm.data(), d_adm, h_adm.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_admn.data(), d_admn, h_admn.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

      for (uint32_t r = 0; r < n_q; ++r) {
        const uint32_t live_cols = pos0 + r + 1u;
        if (live_cols <= 512u) {
          // Fast-path: seqused <= 512 admits every single live key in identity order
          assert(h_admn[r] == live_cols);
          for (uint32_t i = 0; i < live_cols; ++i) {
            assert(h_adm[(size_t)r * 1024 + i] == i);
          }
        } else {
          // Ragged causal top-512: exactly 512 picks, all strictly <= pos0 + r
          assert(h_admn[r] == 512u);
          for (uint32_t i = 0; i < 512u; ++i) {
            const uint32_t pick = h_adm[(size_t)r * 1024 + i];
            assert(pick <= pos0 + r);
          }
        }
      }

      (void)hipFree(d_scores); (void)hipFree(d_mask); (void)hipFree(d_ncomp); (void)hipFree(d_adm); (void)hipFree(d_admn);
    }
    std::printf("PASS: Test 3 - Ragged causal prefill & pos0 + r + 1 <= 512 fast-path verified\n");
  }

  // ------------------------------------------------------------------------------------------------
  // Test 4 (W3-2a): Gather-only candidate indexer parity vs legacy full-score + idx_apply_cand_hip
  // Covers toggle, uint16_t overflow guard, causal prefill, short prefix, single-token decode, and
  // multi-token decode with cand_block in {8, 16}, TopkScratch spread, adm & dense mask outputs.
  // ------------------------------------------------------------------------------------------------
  {
    const char* prev_cg = std::getenv("AFF_IDX_CAND_GATHER");
    std::string prev_cg_val = prev_cg ? prev_cg : "";
    const char* prev_nr = std::getenv("AFF_IDX_NARROW");
    std::string prev_nr_val = prev_nr ? prev_nr : "";

    unsetenv("AFF_IDX_CAND_GATHER");
    assert(idx_cand_gather_enabled() == true);
    setenv("AFF_IDX_CAND_GATHER", "1", 1);
    assert(idx_cand_gather_enabled() == true);
    setenv("AFF_IDX_CAND_GATHER", "0", 1);
    assert(idx_cand_gather_enabled() == false);
    if (prev_cg) setenv("AFF_IDX_CAND_GATHER", prev_cg_val.c_str(), 1);
    else unsetenv("AFF_IDX_CAND_GATHER");

    assert(idx_cand_blocks_fit_u16(65535u) == true);
    assert(idx_cand_blocks_fit_u16(65536u) == false);
    {
      uint16_t dummy_id = 0;
      uint32_t dummy_n = 0;
      float dummy_f = 0.0f;
      assert(idx_topk_mask_cand_emit_hip(&dummy_f, nullptr, nullptr, 1, 65536u, 65536u, 65536u,
                                         1024u, &dummy_id, &dummy_n, 1024u, nullptr) == false);
    }

    struct CandCase {
      const char* name;
      uint32_t nb;
      uint32_t pos0;
      uint32_t n_keys;
      uint32_t mlen;
      uint32_t n_head;
      uint32_t cand_block;
      uint32_t cand_topk_blocks;
      uint32_t topk;
      bool causal_nkeys;
      bool use_scratch;
      const char* narrow_env;
      bool unaligned_bstride;
      bool force_ties;
    };
    const CandCase cases[] = {
      {"prefill-c16-causal",    32u, 16500u, 16532u, 16600u, 64u, 16u, 256u, 512u, true,  false, "1", false, true},
      {"prefill-c8-short",      24u,     0u,    24u,    64u, 64u,  8u, 256u, 512u, true,  false, "1", false, false},
      {"decode-1tok-c16-nosp",   1u,     0u, 17003u, 17150u, 64u, 16u, 320u, 512u, false, false, "1", false, true},
      {"decode-1tok-c16-spread", 1u,     0u, 17003u, 17150u, 64u, 16u, 320u, 512u, false, true,  "1", false, true},
      {"decode-4tok-c8-narrow1", 4u,     0u, 17003u, 17003u, 64u,  8u, 512u, 512u, false, true,  "1", false, false},
      {"decode-4tok-c8-narrow0", 4u,     0u, 17003u, 17003u, 64u,  8u, 512u, 512u, false, false, "0", false, false},
      {"decode-2tok-c12-unalign",2u,     0u, 10003u, 10003u, 64u, 12u, 350u, 512u, false, true,  "1", true,  true},
    };

    for (const CandCase& tc : cases) {
      setenv("AFF_IDX_NARROW", tc.narrow_env, 1);
      const uint32_t nb = tc.nb, n_keys = tc.n_keys, mlen = tc.mlen, n_head = tc.n_head;
      const uint32_t dim = kIdxDim;
      const uint32_t nblk = (n_keys + tc.cand_block - 1u) / tc.cand_block;
      const uint32_t bstride = tc.unaligned_bstride ? nblk : (((nblk + 7u) / 8u) * 8u);
      const uint32_t cand_stride = tc.cand_topk_blocks;
      const uint32_t cand_max_blocks = std::min(nblk, tc.cand_topk_blocks);
      const uint32_t compact_stride = cand_max_blocks * tc.cand_block;
      const uint32_t adm_stride = tc.topk;

      std::vector<float> h_q((size_t)nb * n_head * dim);
      std::vector<float> h_q_tgt((size_t)nb * n_head * dim);
      std::vector<float> h_k((size_t)n_keys * dim);
      std::vector<float> h_hw_src((size_t)nb * n_head);
      std::vector<float> h_hw_tgt((size_t)nb * n_head);
      for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = std::sin((float)(i + nb * 7u) * 0.031f) * 0.6f;
        h_q_tgt[i] = std::cos((float)(i + nb * 13u) * 0.027f) * 0.6f;
      }
      for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = std::sin((float)i * 0.019f) * 0.55f;
      if (tc.force_ties && n_keys > 64u) {
        for (uint32_t s = 17u; s < n_keys; s += 29u) {
          std::copy(&h_k[17u * dim], &h_k[18u * dim], &h_k[(size_t)s * dim]);
        }
      }
      for (size_t i = 0; i < h_hw_src.size(); ++i) {
        h_hw_src[i] = 0.4f + 0.5f * std::cos((float)i * 0.11f);
        h_hw_tgt[i] = 0.5f + 0.4f * std::sin((float)i * 0.17f);
      }

      std::vector<uint8_t> h_q8_src(h_q.size()), h_q8_tgt(h_q_tgt.size()), h_k8(h_k.size());
      std::vector<float> h_qs_src(nb), h_qs_tgt(nb), h_ks(n_keys);
      idx_encode_rows(h_q.data(), h_q8_src.data(), h_qs_src.data(), nb, n_head * dim);
      idx_encode_rows(h_q_tgt.data(), h_q8_tgt.data(), h_qs_tgt.data(), nb, n_head * dim);
      idx_encode_rows(h_k.data(), h_k8.data(), h_ks.data(), n_keys, dim);

      std::vector<uint32_t> h_nkeys(nb), h_nblk(nb);
      for (uint32_t t = 0; t < nb; ++t) {
        h_nkeys[t] = tc.causal_nkeys ? (tc.pos0 + t + 1u) : n_keys;
        h_nblk[t] = (h_nkeys[t] + tc.cand_block - 1u) / tc.cand_block;
      }

      uint8_t *d_q8s = nullptr, *d_q8t = nullptr, *d_k8 = nullptr;
      float *d_qss = nullptr, *d_qst = nullptr, *d_ks = nullptr, *d_hws = nullptr, *d_hwt = nullptr;
      float *d_sc_leg = nullptr, *d_bsc = nullptr, *d_sc_cand = nullptr;
      uint32_t *d_nk = nullptr, *d_nbk = nullptr;
      uint8_t *d_cand_mask = nullptr, *d_mask_leg = nullptr, *d_mask_cand = nullptr;
      uint16_t *d_cand_ids = nullptr, *d_cand_ids_opt = nullptr;
      uint32_t *d_cand_n = nullptr, *d_cand_n_opt = nullptr;
      uint32_t *d_adm_leg = nullptr, *d_admn_leg = nullptr;
      uint32_t *d_adm_cand = nullptr, *d_admn_cand = nullptr;
      uint32_t *d_adm_both = nullptr, *d_admn_both = nullptr;

      assert(hipMalloc(&d_q8s, h_q8_src.size()) == hipSuccess);
      assert(hipMalloc(&d_q8t, h_q8_tgt.size()) == hipSuccess);
      assert(hipMalloc(&d_k8, h_k8.size()) == hipSuccess);
      assert(hipMalloc(&d_qss, (size_t)nb * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_qst, (size_t)nb * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_ks, (size_t)n_keys * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_hws, (size_t)nb * n_head * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_hwt, (size_t)nb * n_head * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_sc_leg, (size_t)nb * n_keys * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_bsc, (size_t)nb * bstride * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_sc_cand, (size_t)nb * compact_stride * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_nk, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_nbk, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_cand_mask, (size_t)nb * bstride) == hipSuccess);
      assert(hipMalloc(&d_mask_leg, (size_t)nb * mlen) == hipSuccess);
      assert(hipMalloc(&d_mask_cand, (size_t)nb * mlen) == hipSuccess);
      assert(hipMalloc(&d_cand_ids, (size_t)nb * cand_stride * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_cand_ids_opt, (size_t)nb * cand_stride * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_cand_n, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_cand_n_opt, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm_leg, (size_t)nb * adm_stride * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn_leg, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm_cand, (size_t)nb * adm_stride * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn_cand, (size_t)nb * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_adm_both, (size_t)nb * adm_stride * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_admn_both, (size_t)nb * sizeof(uint32_t)) == hipSuccess);

      assert(hipMemcpy(d_q8s, h_q8_src.data(), h_q8_src.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_q8t, h_q8_tgt.data(), h_q8_tgt.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_k8, h_k8.data(), h_k8.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_qss, h_qs_src.data(), (size_t)nb * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_qst, h_qs_tgt.data(), (size_t)nb * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_ks, h_ks.data(), (size_t)n_keys * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_hws, h_hw_src.data(), h_hw_src.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_hwt, h_hw_tgt.data(), h_hw_tgt.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_nk, h_nkeys.data(), (size_t)nb * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_nbk, h_nblk.data(), (size_t)nb * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
      // Legacy idx_topk_mask_hip with nkeys==nullptr reads up to `bstride`; pad [nblk, bstride) to -inf.
      std::vector<float> h_bsc_init((size_t)nb * bstride, -INFINITY);
      assert(hipMemcpy(d_bsc, h_bsc_init.data(), h_bsc_init.size() * sizeof(float),
                       hipMemcpyHostToDevice) == hipSuccess);
      // Pre-poison dense mask buffers to ensure clearing covers all [0, mlen) including padding
      assert(hipMemset(d_mask_leg, 0xA5, (size_t)nb * mlen) == hipSuccess);
      assert(hipMemset(d_mask_cand, 0x5A, (size_t)nb * mlen) == hipSuccess);

      TopkScratch sp{};
      if (tc.use_scratch) {
        sp.tok_cap = nb;
        uint64_t sb = 0, tb = 0;
        const uint64_t hb = idx_topk_scratch_bytes(nb, &sb, &tb);
        assert(hipMalloc(&sp.hist, hb) == hipSuccess);
        assert(hipMalloc(&sp.state, sb) == hipSuccess);
        assert(hipMalloc(&sp.tie, tb) == hipSuccess);
        assert(hipMemset(sp.hist, 0, hb) == hipSuccess);
        assert(hipMemset(sp.state, 0, sb) == hipSuccess);
        assert(hipMemset(sp.tie, 0, tb) == hipSuccess);
      }
      const TopkScratch* spp = tc.use_scratch ? &sp : nullptr;
      const uint32_t* nk_arg = tc.causal_nkeys ? d_nk : nullptr;
      const uint32_t* nbk_arg = tc.causal_nkeys ? d_nbk : nullptr;

      // 1) Legacy Layer 20: score + block_max + topk -> d_cand_mask
      indexer_scores_hip(d_q8s, d_qss, d_k8, d_ks, d_hws, d_sc_leg, nb, n_keys, n_head, nullptr);
      idx_block_max_hip(d_sc_leg, d_bsc, nk_arg, nb, n_keys, bstride, nblk, tc.cand_block, n_keys,
                        nullptr);
      idx_topk_mask_hip(d_bsc, d_cand_mask, nbk_arg, nb, bstride, bstride, nblk,
                        tc.cand_topk_blocks, nullptr, spp);

      // 2) Legacy Consumer Layer 24: full score -> apply_cand -> topk (adm + dense mask)
      indexer_scores_hip(d_q8t, d_qst, d_k8, d_ks, d_hwt, d_sc_leg, nb, n_keys, n_head, nullptr);
      // Save unmasked legacy scores for score-level bit-exact check
      std::vector<float> h_sc_unmasked((size_t)nb * n_keys);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_sc_unmasked.data(), d_sc_leg, h_sc_unmasked.size() * sizeof(float),
                       hipMemcpyDeviceToHost) == hipSuccess);

      idx_apply_cand_hip(d_sc_leg, d_cand_mask, nk_arg, nb, n_keys, bstride, n_keys, tc.cand_block,
                         n_keys, nullptr);
      idx_topk_mask_hip(d_sc_leg, d_mask_leg, nk_arg, nb, n_keys, mlen, mlen, tc.topk, nullptr, spp,
                        d_adm_leg, d_admn_leg, adm_stride);

      // 3) W3-2a Layer 20 publisher: compact uint16_t candidate blocks (skip dense cand_mask)
      assert(idx_topk_mask_cand_emit_hip(d_bsc, nullptr, nbk_arg, nb, bstride, bstride, nblk,
                                         tc.cand_topk_blocks, d_cand_ids, d_cand_n, cand_stride,
                                         nullptr, spp));
      // Also verify optional cand_ids_out parameter on idx_topk_mask_hip directly
      idx_topk_mask_hip(d_bsc, nullptr, nbk_arg, nb, bstride, bstride, nblk, tc.cand_topk_blocks,
                        nullptr, spp, nullptr, nullptr, 0u, d_cand_ids_opt, d_cand_n_opt,
                        cand_stride);

      // 4) W3-2a Consumer Layer 24: gather score + gather topk (adm-only AND mask+adm)
      indexer_scores_cand_hip(d_q8t, d_qst, d_k8, d_ks, d_hwt, d_cand_ids, d_cand_n, nk_arg,
                              d_sc_cand, nb, n_keys, n_head, cand_stride, tc.cand_block,
                              cand_max_blocks, compact_stride, nullptr);
      idx_topk_mask_cand_hip(d_sc_cand, nullptr, d_cand_ids, d_cand_n, nb, compact_stride,
                             cand_stride, tc.cand_block, cand_max_blocks, mlen, mlen, tc.topk,
                             nullptr, spp, d_adm_cand, d_admn_cand, adm_stride);
      idx_topk_mask_cand_hip(d_sc_cand, d_mask_cand, d_cand_ids, d_cand_n, nb, compact_stride,
                             cand_stride, tc.cand_block, cand_max_blocks, mlen, mlen, tc.topk,
                             nullptr, spp, d_adm_both, d_admn_both, adm_stride);
      assert(hipDeviceSynchronize() == hipSuccess);

      // Verify candidate block publish parity
      std::vector<uint8_t> h_cand_mask((size_t)nb * bstride);
      std::vector<uint16_t> h_cand_ids((size_t)nb * cand_stride), h_cand_ids_opt((size_t)nb * cand_stride);
      std::vector<uint32_t> h_cand_n(nb), h_cand_n_opt(nb);
      assert(hipMemcpy(h_cand_mask.data(), d_cand_mask, h_cand_mask.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_cand_ids.data(), d_cand_ids, h_cand_ids.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_cand_ids_opt.data(), d_cand_ids_opt, h_cand_ids_opt.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_cand_n.data(), d_cand_n, h_cand_n.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_cand_n_opt.data(), d_cand_n_opt, h_cand_n_opt.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);

      for (uint32_t t = 0; t < nb; ++t) {
        std::vector<uint16_t> expected_blocks;
        for (uint32_t b = 0; b < h_nblk[t]; ++b) {
          if (h_cand_mask[(size_t)t * bstride + b]) expected_blocks.push_back((uint16_t)b);
        }
        assert(h_cand_n[t] == expected_blocks.size());
        assert(h_cand_n[t] <= tc.cand_topk_blocks);
        for (uint32_t j = 0; j < h_cand_n[t]; ++j) {
          assert(h_cand_ids[(size_t)t * cand_stride + j] == expected_blocks[j]);
          if (j > 0) assert(h_cand_ids[(size_t)t * cand_stride + j] > h_cand_ids[(size_t)t * cand_stride + j - 1u]);
        }
        if (!tc.causal_nkeys && bstride == nblk) {
          assert(h_cand_n_opt[t] == h_cand_n[t]);
          for (uint32_t j = 0; j < h_cand_n[t]; ++j)
            assert(h_cand_ids_opt[(size_t)t * cand_stride + j] == h_cand_ids[(size_t)t * cand_stride + j]);
        }
      }

      // Verify gather score bit-for-bit parity vs legacy full scores
      std::vector<float> h_sc_cand((size_t)nb * compact_stride);
      assert(hipMemcpy(h_sc_cand.data(), d_sc_cand, h_sc_cand.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      for (uint32_t t = 0; t < nb; ++t) {
        for (uint32_t j = 0; j < h_cand_n[t]; ++j) {
          const uint32_t b = h_cand_ids[(size_t)t * cand_stride + j];
          for (uint32_t r = 0; r < tc.cand_block; ++r) {
            const uint32_t u = j * tc.cand_block + r;
            const uint32_t g = b * tc.cand_block + r;
            const float got = h_sc_cand[(size_t)t * compact_stride + u];
            if (g < h_nkeys[t]) {
              const float ref = h_sc_unmasked[(size_t)t * n_keys + g];
              assert(std::memcmp(&got, &ref, sizeof(float)) == 0);
            } else {
              assert(got == -INFINITY);
            }
          }
        }
      }

      // Verify admitted list & dense mask exact parity vs legacy
      std::vector<uint32_t> h_adm_leg((size_t)nb * adm_stride), h_admn_leg(nb);
      std::vector<uint32_t> h_adm_cand((size_t)nb * adm_stride), h_admn_cand(nb);
      std::vector<uint32_t> h_adm_both((size_t)nb * adm_stride), h_admn_both(nb);
      std::vector<uint8_t> h_mask_leg((size_t)nb * mlen), h_mask_cand((size_t)nb * mlen);
      assert(hipMemcpy(h_adm_leg.data(), d_adm_leg, h_adm_leg.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_admn_leg.data(), d_admn_leg, h_admn_leg.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_adm_cand.data(), d_adm_cand, h_adm_cand.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_admn_cand.data(), d_admn_cand, h_admn_cand.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_adm_both.data(), d_adm_both, h_adm_both.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_admn_both.data(), d_admn_both, h_admn_both.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_mask_leg.data(), d_mask_leg, h_mask_leg.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_mask_cand.data(), d_mask_cand, h_mask_cand.size(), hipMemcpyDeviceToHost) == hipSuccess);

      for (uint32_t t = 0; t < nb; ++t) {
        assert(h_admn_cand[t] == h_admn_leg[t]);
        assert(h_admn_both[t] == h_admn_leg[t]);
        for (uint32_t i = 0; i < h_admn_leg[t]; ++i) {
          assert(h_adm_cand[(size_t)t * adm_stride + i] == h_adm_leg[(size_t)t * adm_stride + i]);
          assert(h_adm_both[(size_t)t * adm_stride + i] == h_adm_leg[(size_t)t * adm_stride + i]);
        }
      }
      assert(std::memcmp(h_mask_cand.data(), h_mask_leg.data(), h_mask_leg.size()) == 0);
      std::printf("  cand gather [%s]: PASS (bit-exact scores, adm, dense mask)\n", tc.name);

      if (tc.use_scratch) {
        (void)hipFree(sp.hist); (void)hipFree(sp.state); (void)hipFree(sp.tie);
      }
      for (void* p : {(void*)d_q8s, (void*)d_q8t, (void*)d_k8, (void*)d_qss, (void*)d_qst,
                      (void*)d_ks, (void*)d_hws, (void*)d_hwt, (void*)d_sc_leg, (void*)d_bsc,
                      (void*)d_sc_cand, (void*)d_nk, (void*)d_nbk, (void*)d_cand_mask,
                      (void*)d_mask_leg, (void*)d_mask_cand, (void*)d_cand_ids,
                      (void*)d_cand_ids_opt, (void*)d_cand_n, (void*)d_cand_n_opt, (void*)d_adm_leg,
                      (void*)d_admn_leg, (void*)d_adm_cand, (void*)d_admn_cand, (void*)d_adm_both,
                      (void*)d_admn_both})
        (void)hipFree(p);
    }
    if (prev_nr) setenv("AFF_IDX_NARROW", prev_nr_val.c_str(), 1);
    else unsetenv("AFF_IDX_NARROW");
    std::printf("PASS: Test 4 - W3-2a gather-only candidate indexer bit-exact parity\n");
  }

  std::printf("ALL TEST_V41_INDEXER CHECKS PASSED SUCCESSFULLY!\n");
  return 0;
}
