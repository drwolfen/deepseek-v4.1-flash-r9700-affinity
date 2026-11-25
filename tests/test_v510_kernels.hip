#include "gpu/attention_gpu.h"
#include "gpu/indexer_gpu.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int check_fuse_idx_q() {
  for (bool had : {false, true}) {
    for (uint32_t n : {1u, 16u, 128u}) {
      const uint32_t nh = 64u, dim = aff::kIdxDim, n_rot = 64u;
      const size_t q_cnt = (size_t)n * nh * dim;
      std::vector<float> h_q(q_cnt);
      uint32_t s = 42u + n + (had ? 7u : 0u);
      for (size_t i = 0; i < q_cnt; ++i) {
        s = s * 1664525u + 1013904223u;
        h_q[i] = ((float)(int32_t)(s >> 8) - (float)(1 << 23)) * 1e-6f;
      }
      float *d_q_ref = nullptr, *d_qs_ref = nullptr, *d_qs_fus = nullptr;
      const float* d_q_const = nullptr;
      float* d_q_src = nullptr;
      uint8_t *d_q8_ref = nullptr, *d_q8_fus = nullptr;
      (void)hipMalloc(&d_q_ref, q_cnt * sizeof(float));
      (void)hipMalloc(&d_q_src, q_cnt * sizeof(float));
      (void)hipMalloc(&d_q8_ref, q_cnt);
      (void)hipMalloc(&d_q8_fus, q_cnt);
      (void)hipMalloc(&d_qs_ref, n * sizeof(float));
      (void)hipMalloc(&d_qs_fus, n * sizeof(float));
      (void)hipMemcpy(d_q_ref, h_q.data(), q_cnt * sizeof(float), hipMemcpyHostToDevice);
      (void)hipMemcpy(d_q_src, h_q.data(), q_cnt * sizeof(float), hipMemcpyHostToDevice);
      d_q_const = d_q_src;

      aff::idx_rope_qat_hip(d_q_ref, n, nh, n_rot, 1024u, 1.0f, 0.9f, 0.5f, 1.0f, 32.0f, nullptr,
                            had);
      aff::idx_quant_rows_hip(d_q_ref, d_q8_ref, d_qs_ref, n, nh * dim, nullptr);
      bool ok = aff::idx_rope_qat_quant_hip(d_q_const, d_q8_fus, d_qs_fus, n, nh, n_rot, 1024u,
                                            1.0f, 0.9f, 0.5f, 1.0f, 32.0f, nullptr, had);
      (void)hipDeviceSynchronize();
      if (!ok) return 1;

      std::vector<uint8_t> h_q8_ref(q_cnt), h_q8_fus(q_cnt);
      std::vector<float> h_qs_ref(n), h_qs_fus(n);
      (void)hipMemcpy(h_q8_ref.data(), d_q8_ref, q_cnt, hipMemcpyDeviceToHost);
      (void)hipMemcpy(h_q8_fus.data(), d_q8_fus, q_cnt, hipMemcpyDeviceToHost);
      (void)hipMemcpy(h_qs_ref.data(), d_qs_ref, n * sizeof(float), hipMemcpyDeviceToHost);
      (void)hipMemcpy(h_qs_fus.data(), d_qs_fus, n * sizeof(float), hipMemcpyDeviceToHost);

      (void)hipFree(d_q_ref); (void)hipFree(d_q_src);
      (void)hipFree(d_q8_ref); (void)hipFree(d_q8_fus);
      (void)hipFree(d_qs_ref); (void)hipFree(d_qs_fus);

      if (std::memcmp(h_q8_ref.data(), h_q8_fus.data(), q_cnt) != 0 ||
          std::memcmp(h_qs_ref.data(), h_qs_fus.data(), n * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL: fuse_idx_q n=%u had=%d mismatch\n", n, (int)had);
        return 1;
      }
    }
  }
  std::printf("PASS: AFF_FUSE_IDX_Q bit-exact oracle\n");
  return 0;
}

static int check_kv_commit_coalesced() {
  const uint32_t width = 512u, n_rot = 64u, ring = 384u, pos0 = 250u;
  const uint64_t rb = aff::kv_row_bytes(aff::KvDtype::FP8R, width, n_rot);
  for (uint32_t nb : {16u, 256u}) {
    for (uint32_t n : {16u, 128u}) {
      if (n > nb) continue;
      std::vector<float> h_lat((size_t)width * nb), h_nw(width);
      uint32_t s = 9001u + nb + n;
      for (size_t i = 0; i < h_lat.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        h_lat[i] = ((float)(int32_t)(s >> 8) - (float)(1 << 23)) * 1e-6f;
      }
      for (uint32_t i = 0; i < width; ++i) h_nw[i] = 0.8f + 0.001f * (float)i;

      float *d_lat = nullptr, *d_nw = nullptr;
      uint8_t *d_raw0 = nullptr, *d_raw1 = nullptr;
      const size_t raw_bytes = (size_t)ring * rb;
      (void)hipMalloc(&d_lat, h_lat.size() * sizeof(float));
      (void)hipMalloc(&d_nw, width * sizeof(float));
      (void)hipMalloc(&d_raw0, raw_bytes);
      (void)hipMalloc(&d_raw1, raw_bytes);
      (void)hipMemset(d_raw0, 0xA5, raw_bytes);
      (void)hipMemset(d_raw1, 0xA5, raw_bytes);
      (void)hipMemcpy(d_lat, h_lat.data(), h_lat.size() * sizeof(float), hipMemcpyHostToDevice);
      (void)hipMemcpy(d_nw, h_nw.data(), width * sizeof(float), hipMemcpyHostToDevice);

      setenv("AFF_KV_COMMIT_COALESCED", "0", 1);
      bool ok0 = aff::kv_norm_commit_batch_hip(aff::KvDtype::FP8R, d_lat, d_nw, 1e-6f, nb, 0u, n,
                                               d_raw0, pos0, ring, width, n_rot, 1.0f, 0.9f, 0.5f,
                                               1.0f, 32.0f, nullptr);
      setenv("AFF_KV_COMMIT_COALESCED", "1", 1);
      bool ok1 = aff::kv_norm_commit_batch_hip(aff::KvDtype::FP8R, d_lat, d_nw, 1e-6f, nb, 0u, n,
                                               d_raw1, pos0, ring, width, n_rot, 1.0f, 0.9f, 0.5f,
                                               1.0f, 32.0f, nullptr);
      (void)hipDeviceSynchronize();
      if (!ok0 || !ok1) return 1;

      std::vector<uint8_t> h0(raw_bytes), h1(raw_bytes);
      (void)hipMemcpy(h0.data(), d_raw0, raw_bytes, hipMemcpyDeviceToHost);
      (void)hipMemcpy(h1.data(), d_raw1, raw_bytes, hipMemcpyDeviceToHost);
      (void)hipFree(d_lat); (void)hipFree(d_nw); (void)hipFree(d_raw0); (void)hipFree(d_raw1);

      if (std::memcmp(h0.data(), h1.data(), raw_bytes) != 0) {
        std::fprintf(stderr, "FAIL: kv_commit_coalesced nb=%u n=%u mismatch\n", nb, n);
        return 1;
      }
    }
  }
  std::printf("PASS: AFF_KV_COMMIT_COALESCED bit-exact oracle\n");
  return 0;
}

static int check_cand_scores_kpt2() {
  const uint32_t nb = 32u, n_keys = 4096u, nh = 64u, dim = aff::kIdxDim;
  const uint32_t cand_blk = 8u, max_blks = 256u, cand_stride = 256u, out_stride = 2048u;
  std::vector<uint8_t> h_q8((size_t)nb * nh * dim), h_k8((size_t)n_keys * dim);
  std::vector<float> h_qs(nb, 0.25f), h_ks(n_keys, 0.5f), h_hw((size_t)nb * nh, 0.125f);
  std::vector<uint16_t> h_cids((size_t)nb * cand_stride, 0);
  std::vector<uint32_t> h_cn(nb, 250u), h_nk(nb, 3995u);
  uint32_t s = 12345u;
  for (size_t i = 0; i < h_q8.size(); ++i) { s = s * 1664525u + 1013904223u; h_q8[i] = (uint8_t)((s >> 24) & 0x7Eu); }
  for (size_t i = 0; i < h_k8.size(); ++i) { s = s * 1664525u + 1013904223u; h_k8[i] = (uint8_t)((s >> 24) & 0x7Eu); }
  for (uint32_t t = 0; t < nb; ++t)
    for (uint32_t j = 0; j < max_blks; ++j) h_cids[(size_t)t * cand_stride + j] = (uint16_t)(j * 2u);

  uint8_t *d_q8 = nullptr, *d_k8 = nullptr;
  float *d_qs = nullptr, *d_ks = nullptr, *d_hw = nullptr, *d_out0 = nullptr, *d_out1 = nullptr;
  uint16_t* d_cids = nullptr;
  uint32_t *d_cn = nullptr, *d_nk = nullptr;
  const size_t out_bytes = (size_t)nb * out_stride * sizeof(float);
  (void)hipMalloc(&d_q8, h_q8.size()); (void)hipMalloc(&d_k8, h_k8.size());
  (void)hipMalloc(&d_qs, nb * sizeof(float)); (void)hipMalloc(&d_ks, n_keys * sizeof(float));
  (void)hipMalloc(&d_hw, h_hw.size() * sizeof(float));
  (void)hipMalloc(&d_cids, h_cids.size() * sizeof(uint16_t));
  (void)hipMalloc(&d_cn, nb * sizeof(uint32_t)); (void)hipMalloc(&d_nk, nb * sizeof(uint32_t));
  (void)hipMalloc(&d_out0, out_bytes); (void)hipMalloc(&d_out1, out_bytes);
  (void)hipMemcpy(d_q8, h_q8.data(), h_q8.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_k8, h_k8.data(), h_k8.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_qs, h_qs.data(), nb * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_ks, h_ks.data(), n_keys * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_hw, h_hw.data(), h_hw.size() * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_cids, h_cids.data(), h_cids.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_cn, h_cn.data(), nb * sizeof(uint32_t), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_nk, h_nk.data(), nb * sizeof(uint32_t), hipMemcpyHostToDevice);

  setenv("AFF_IDX_CAND_MULTITOK", "0", 1);
  aff::indexer_scores_cand_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_cids, d_cn, d_nk, d_out0, nb,
                               n_keys, nh, cand_stride, cand_blk, max_blks, out_stride, nullptr);
  setenv("AFF_IDX_CAND_MULTITOK", "1", 1);
  aff::indexer_scores_cand_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_cids, d_cn, d_nk, d_out1, nb,
                               n_keys, nh, cand_stride, cand_blk, max_blks, out_stride, nullptr);
  (void)hipDeviceSynchronize();

  std::vector<float> o0((size_t)nb * out_stride), o1((size_t)nb * out_stride);
  (void)hipMemcpy(o0.data(), d_out0, out_bytes, hipMemcpyDeviceToHost);
  (void)hipMemcpy(o1.data(), d_out1, out_bytes, hipMemcpyDeviceToHost);
  (void)hipFree(d_q8); (void)hipFree(d_k8); (void)hipFree(d_qs); (void)hipFree(d_ks);
  (void)hipFree(d_hw); (void)hipFree(d_cids); (void)hipFree(d_cn); (void)hipFree(d_nk);
  (void)hipFree(d_out0); (void)hipFree(d_out1);

  if (std::memcmp(o0.data(), o1.data(), out_bytes) != 0) {
    std::fprintf(stderr, "FAIL: indexer_scores_cand KPT2 mismatch\n");
    return 1;
  }
  std::printf("PASS: AFF_IDX_CAND_MULTITOK bit-exact oracle\n");
  return 0;
}

static int check_topk_and_bmax_v2() {
  const uint32_t nb = 4u, n_keys = 8192u, sstride = 8192u, bstride = 1024u, nblk = 1024u;
  const uint32_t k = 512u, mstride = 8192u;
  std::vector<float> h_sc((size_t)nb * sstride);
  std::vector<uint32_t> h_nk = {61u, 64u, 8192u, 1u};
  uint32_t s = 777u;
  for (uint32_t t = 0; t < nb; ++t) {
    for (uint32_t i = 0; i < sstride; ++i) {
      if ((i % 7u) == 0u) h_sc[(size_t)t * sstride + i] = -INFINITY;
      else if ((i % 13u) == 0u) h_sc[(size_t)t * sstride + i] = 0.42f; // deliberate multi-way ties
      else {
        s = s * 1664525u + 1013904223u;
        h_sc[(size_t)t * sstride + i] = ((float)(int32_t)(s >> 8) - (float)(1 << 23)) * 1e-7f;
      }
    }
  }
  float *d_sc = nullptr, *d_bsc0 = nullptr, *d_bsc1 = nullptr;
  uint32_t *d_nk = nullptr, *d_adm0 = nullptr, *d_adm1 = nullptr, *d_an0 = nullptr, *d_an1 = nullptr;
  uint8_t *d_m0 = nullptr, *d_m1 = nullptr;
  (void)hipMalloc(&d_sc, h_sc.size() * sizeof(float));
  (void)hipMalloc(&d_nk, nb * sizeof(uint32_t));
  (void)hipMalloc(&d_bsc0, (size_t)nb * bstride * sizeof(float));
  (void)hipMalloc(&d_bsc1, (size_t)nb * bstride * sizeof(float));
  (void)hipMalloc(&d_m0, (size_t)nb * mstride);
  (void)hipMalloc(&d_m1, (size_t)nb * mstride);
  (void)hipMalloc(&d_adm0, (size_t)nb * k * sizeof(uint32_t));
  (void)hipMalloc(&d_adm1, (size_t)nb * k * sizeof(uint32_t));
  (void)hipMalloc(&d_an0, nb * sizeof(uint32_t));
  (void)hipMalloc(&d_an1, nb * sizeof(uint32_t));
  (void)hipMemcpy(d_sc, h_sc.data(), h_sc.size() * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(d_nk, h_nk.data(), nb * sizeof(uint32_t), hipMemcpyHostToDevice);

  // 1) Block-max b=8
  setenv("AFF_IDX_TOPK_V2", "0", 1);
  aff::idx_block_max_hip(d_sc, d_bsc0, d_nk, nb, sstride, bstride, nblk, 8u, n_keys, nullptr);
  setenv("AFF_IDX_TOPK_V2", "1", 1);
  aff::idx_block_max_hip(d_sc, d_bsc1, d_nk, nb, sstride, bstride, nblk, 8u, n_keys, nullptr);

  // 2) Single-block & spread TopK
  aff::TopkScratch sp{};
  uint64_t sb = 0, tb = 0;
  uint64_t hb = aff::idx_topk_scratch_bytes(16u, &sb, &tb);
  (void)hipMalloc(&sp.hist, hb); (void)hipMalloc(&sp.state, sb); (void)hipMalloc(&sp.tie, tb);
  (void)hipMemset(sp.hist, 0, hb); (void)hipMemset(sp.state, 0, sb); (void)hipMemset(sp.tie, 0, tb);
  sp.tok_cap = 16u;

  for (int use_sp = 0; use_sp <= 1; ++use_sp) {
    const aff::TopkScratch* spp = use_sp ? &sp : nullptr;
    setenv("AFF_IDX_TOPK_V2", "0", 1);
    aff::idx_topk_mask_hip(d_sc, d_m0, d_nk, nb, sstride, mstride, n_keys, k, nullptr, spp, d_adm0, d_an0, k);
    setenv("AFF_IDX_TOPK_V2", "1", 1);
    aff::idx_topk_mask_hip(d_sc, d_m1, d_nk, nb, sstride, mstride, n_keys, k, nullptr, spp, d_adm1, d_an1, k);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> m0((size_t)nb * mstride), m1((size_t)nb * mstride);
    std::vector<uint32_t> an0(nb), an1(nb), a0((size_t)nb * k), a1((size_t)nb * k);
    (void)hipMemcpy(m0.data(), d_m0, m0.size(), hipMemcpyDeviceToHost);
    (void)hipMemcpy(m1.data(), d_m1, m1.size(), hipMemcpyDeviceToHost);
    (void)hipMemcpy(an0.data(), d_an0, nb * sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(an1.data(), d_an1, nb * sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(a0.data(), d_adm0, a0.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
    (void)hipMemcpy(a1.data(), d_adm1, a1.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (an0 != an1 || m0 != m1 || a0 != a1) {
      std::fprintf(stderr, "FAIL: topk v2 mismatch use_sp=%d\n", use_sp);
      return 1;
    }
  }

  // 3) Gather-cand topk (cand_block = 8, single-block & spread)
  {
    const uint32_t cb = 8u, max_b = 512u, cs = 512u, c_sstride = 4096u;
    std::vector<uint16_t> h_cids((size_t)nb * cs);
    std::vector<uint32_t> h_cn(nb, max_b);
    for (uint32_t t = 0; t < nb; ++t)
      for (uint32_t j = 0; j < max_b; ++j) h_cids[(size_t)t * cs + j] = (uint16_t)(j * 2u);
    uint16_t* d_cids = nullptr;
    uint32_t* d_cn = nullptr;
    (void)hipMalloc(&d_cids, h_cids.size() * sizeof(uint16_t));
    (void)hipMalloc(&d_cn, nb * sizeof(uint32_t));
    (void)hipMemcpy(d_cids, h_cids.data(), h_cids.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_cn, h_cn.data(), nb * sizeof(uint32_t), hipMemcpyHostToDevice);
    for (int use_sp = 0; use_sp <= 1; ++use_sp) {
      const aff::TopkScratch* spp = use_sp ? &sp : nullptr;
      setenv("AFF_IDX_TOPK_V2", "0", 1);
      aff::idx_topk_mask_cand_hip(d_sc, d_m0, d_cids, d_cn, nb, c_sstride, cs, cb, max_b, mstride,
                                  n_keys, k, nullptr, spp, d_adm0, d_an0, k);
      setenv("AFF_IDX_TOPK_V2", "1", 1);
      aff::idx_topk_mask_cand_hip(d_sc, d_m1, d_cids, d_cn, nb, c_sstride, cs, cb, max_b, mstride,
                                  n_keys, k, nullptr, spp, d_adm1, d_an1, k);
      (void)hipDeviceSynchronize();
      std::vector<uint8_t> m0((size_t)nb * mstride), m1((size_t)nb * mstride);
      std::vector<uint32_t> an0(nb), an1(nb), a0((size_t)nb * k), a1((size_t)nb * k);
      (void)hipMemcpy(m0.data(), d_m0, m0.size(), hipMemcpyDeviceToHost);
      (void)hipMemcpy(m1.data(), d_m1, m1.size(), hipMemcpyDeviceToHost);
      (void)hipMemcpy(an0.data(), d_an0, nb * sizeof(uint32_t), hipMemcpyDeviceToHost);
      (void)hipMemcpy(an1.data(), d_an1, nb * sizeof(uint32_t), hipMemcpyDeviceToHost);
      (void)hipMemcpy(a0.data(), d_adm0, a0.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
      (void)hipMemcpy(a1.data(), d_adm1, a1.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
      if (an0 != an1 || m0 != m1 || a0 != a1) {
        std::fprintf(stderr, "FAIL: topk cand v2 mismatch use_sp=%d\n", use_sp);
        return 1;
      }
    }
    (void)hipFree(d_cids);
    (void)hipFree(d_cn);
  }
  std::vector<float> b0((size_t)nb * bstride), b1((size_t)nb * bstride);
  (void)hipMemcpy(b0.data(), d_bsc0, b0.size() * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(b1.data(), d_bsc1, b1.size() * sizeof(float), hipMemcpyDeviceToHost);

  (void)hipFree(sp.hist); (void)hipFree(sp.state); (void)hipFree(sp.tie);
  (void)hipFree(d_sc); (void)hipFree(d_nk); (void)hipFree(d_bsc0); (void)hipFree(d_bsc1);
  (void)hipFree(d_m0); (void)hipFree(d_m1); (void)hipFree(d_adm0); (void)hipFree(d_adm1);
  (void)hipFree(d_an0); (void)hipFree(d_an1);

  if (std::memcmp(b0.data(), b1.data(), b0.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: idx_block_max b8 v2 mismatch\n");
    return 1;
  }
  std::printf("PASS: AFF_IDX_TOPK_V2 block-max + single/spread 11-bit top-k bit-exact oracle\n");
  return 0;
}

int main() {
  uint64_t sb = 0, tb = 0;
  const uint64_t hb = aff::idx_topk_scratch_bytes(16u, &sb, &tb);
  if (hb != 16ull * 2048u * 4u || sb != 16ull * 16u || tb != 16ull * 1025u * 4u) return 1;
  if (aff::kv_row_bytes(aff::KvDtype::FP8R, 512u, 64u) != 528ull) return 1;

  int dev_count = 0;
  hipError_t err = hipGetDeviceCount(&dev_count);
  if (err != hipSuccess || dev_count <= 0) {
    std::printf("PASS: test_v510_kernels offline checks (no GPU devices in environment)\n");
    return 0;
  }
  if (check_fuse_idx_q()) return 1;
  if (check_kv_commit_coalesced()) return 1;
  if (check_cand_scores_kpt2()) return 1;
  if (check_topk_and_bmax_v2()) return 1;
  return 0;
}
