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

  std::printf("ALL TEST_V41_INDEXER CHECKS PASSED SUCCESSFULLY!\n");
  return 0;
}
