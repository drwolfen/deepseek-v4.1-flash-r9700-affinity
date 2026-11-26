#include "gpu/batch_kernels.h"
#include "gpu/dense_gpu.h"
#include "gpu/engram_gpu.h"
#include "gpu/hip_common.h"
#include "gpu/indexer_gpu.h"
#include "gpu/moe_plan.h"
#include "quant/source_dtypes.h"

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
#include <string>
#include <vector>

namespace {

uint16_t f32_to_bf16_cpu(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float bf16_to_f32_cpu(uint16_t h) {
  const uint32_t u = (uint32_t)h << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

int check_batch_slice_tail_parity() {
  const uint32_t n_embd = 256u, n_hc = 4u, n_head = 16u, width = 128u, n_rot = 64u, qr_dim = 128u;
  const uint32_t n_live = 272u;  // batch_round(272) == 384 -> compact tail keep=128 (nb=128)
  const uint32_t off = 144u, keep = 128u;
  const uint32_t pos0 = 100u;

  std::vector<uint16_t> w_qa(qr_dim * n_embd), w_qb(n_head * width * qr_dim),
      w_kv(width * n_embd), w_fn((n_hc * 2u + n_hc * n_hc) * n_embd);
  auto fill_w = [](std::vector<uint16_t>& w, float scale, uint32_t seed) {
    uint32_t rng = seed;
    for (size_t i = 0; i < w.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      w[i] = f32_to_bf16_cpu(scale * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f));
    }
  };
  fill_w(w_qa, 0.02f, 11u);
  fill_w(w_qb, 0.02f, 22u);
  fill_w(w_kv, 0.02f, 33u);
  fill_w(w_fn, 0.01f, 44u);

  std::vector<float> v_sv(n_hc * 2u + n_hc * n_hc, 0.0f), v_bv(n_hc * 2u + n_hc * n_hc, 0.0f);
  for (uint32_t i = 0; i < n_hc; ++i) {
    v_bv[i] = 1.0f;
    v_bv[n_hc + i] = 0.25f;
    v_bv[2u * n_hc + i * n_hc + i] = 1.0f;
  }
  std::vector<float> v_nw(n_embd, 1.0f), v_qn(qr_dim, 1.0f), v_kvn(width, 1.0f);
  std::vector<float> emb((size_t)n_live * n_embd);
  for (size_t i = 0; i < emb.size(); ++i) {
    emb[i] = 0.2f * std::sin((float)i * 0.013f) + 0.05f * std::cos((float)(i % 97u) * 0.17f);
  }

  auto run_once = [&](bool do_slice, std::vector<float>* out_hc, std::vector<float>* out_norm) {
    aff::DenseGpu dg;
    std::string err;
    assert(dg.init(n_head * width, n_embd, &err, 1));
    assert(dg.batch_init(n_live, n_embd, n_hc, n_head, width, 2048u, 256u, 32u, &err));
    aff::Model::DenseOps dops = dg.ops();
    aff::Model::BatchOps bops = dg.batch_ops();

    const int32_t h_qa = dops.reg(dops.ctx, "blk.0.attn_q_a", w_qa.data(), nullptr, aff::AFF_BF16,
                                  qr_dim, n_embd);
    const int32_t h_qb = dops.reg(dops.ctx, "blk.0.attn_q_b", w_qb.data(), nullptr, aff::AFF_BF16,
                                  n_head * width, qr_dim);
    const int32_t h_kv = dops.reg(dops.ctx, "blk.0.attn_kv", w_kv.data(), nullptr, aff::AFF_BF16,
                                  width, n_embd);
    const int32_t h_fn = dops.reg(dops.ctx, "blk.0.hc_fn", w_fn.data(), nullptr, aff::AFF_BF16,
                                  n_hc * 2u + n_hc * n_hc, n_embd);
    const int32_t h_sv = dops.reg_vec(dops.ctx, v_sv.data(), v_sv.size());
    const int32_t h_bv = dops.reg_vec(dops.ctx, v_bv.data(), v_bv.size());
    const int32_t h_nw = dops.reg_vec(dops.ctx, v_nw.data(), v_nw.size());
    const int32_t h_qn = dops.reg_vec(dops.ctx, v_qn.data(), v_qn.size());
    const int32_t h_kvn = dops.reg_vec(dops.ctx, v_kvn.data(), v_kvn.size());
    assert(h_qa >= 0 && h_qb >= 0 && h_kv >= 0 && h_fn >= 0 && h_sv >= 0 && h_bv >= 0 &&
           h_nw >= 0 && h_qn >= 0 && h_kvn >= 0);

    aff::RopeDerived rope{1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    if (do_slice) {
      assert(bops.begin(bops.ctx, n_live, pos0, emb.data(), n_embd, n_hc));
      assert(dg.batch_nb() == 384u);
      assert(bops.hc_pre(bops.ctx, h_fn, h_sv, h_bv, h_nw, n_embd, n_hc, 1u, 1e-6f, 1e-6f,
                         nullptr));
      assert(bops.attn_qkv(bops.ctx, h_qa, h_qb, h_qn, h_kv, h_kvn, n_embd, qr_dim, n_head, width,
                           n_rot, rope, 1e-6f, nullptr, false));
      assert(aff::DenseGpu::slice_tail_cb(&dg, off, keep, n_embd, n_hc, n_head, width, qr_dim));
      assert(dg.batch_nb() == 128u);
    } else {
      assert(bops.begin(bops.ctx, keep, pos0 + off, emb.data() + (size_t)off * n_embd, n_embd,
                        n_hc));
      assert(dg.batch_nb() == 128u);
      assert(bops.hc_pre(bops.ctx, h_fn, h_sv, h_bv, h_nw, n_embd, n_hc, 1u, 1e-6f, 1e-6f,
                         nullptr));
      assert(bops.attn_qkv(bops.ctx, h_qa, h_qb, h_qn, h_kv, h_kvn, n_embd, qr_dim, n_head, width,
                           n_rot, rope, 1e-6f, nullptr, false));
    }
    out_norm->assign((size_t)keep * n_embd, 0.0f);
    std::vector<float> tmp_norm((size_t)keep * n_embd, 0.0f);
    assert(bops.norm_fetch(bops.ctx, tmp_norm.data()));
    *out_norm = tmp_norm;

    assert(bops.hc_post(bops.ctx, true, n_embd, n_hc));
    out_hc->assign((size_t)keep * n_embd, 0.0f);
    for (uint32_t b = 0; b < keep; ++b) {
      assert(bops.hc_read(bops.ctx, b, out_hc->data() + (size_t)b * n_embd, n_embd, n_hc));
    }
  };

  std::vector<float> hc_slice, norm_slice, hc_ref, norm_ref;
  run_once(true, &hc_slice, &norm_slice);
  run_once(false, &hc_ref, &norm_ref);

  float max_norm_diff = 0.0f, max_hc_diff = 0.0f;
  for (size_t i = 0; i < norm_ref.size(); ++i)
    max_norm_diff = std::max(max_norm_diff, std::fabs(norm_slice[i] - norm_ref[i]));
  for (size_t i = 0; i < hc_ref.size(); ++i)
    max_hc_diff = std::max(max_hc_diff, std::fabs(hc_slice[i] - hc_ref[i]));
  assert(max_norm_diff <= 1e-5f);
  assert(max_hc_diff <= 1e-5f);
  std::printf("PASS: batch_slice_tail parity (nb 384->128, max_norm=%.3e max_hc=%.3e)\n",
              max_norm_diff, max_hc_diff);
  return 0;
}

int check_engram_gate_wide_tok_parity() {
  const uint32_t n_embd = 256u, n_hc = 4u;
  const float eps = 1e-6f;
  for (uint32_t nb : {512u, 1408u}) {
    const uint32_t hc_stride = nb;
    const size_t hc_elems = (size_t)n_hc * n_embd * hc_stride;
    const size_t kv_elems = (size_t)(n_hc + 1u) * n_embd * nb;
    const size_t part_elems = (size_t)3u * n_hc * nb;

    std::vector<uint16_t> h_hc(hc_elems);
    std::vector<float> h_kv(kv_elems), h_w((size_t)n_hc * n_embd);
    uint32_t rng = 9001u + nb;
    for (size_t i = 0; i < hc_elems; ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_hc[i] = f32_to_bf16_cpu((((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 0.5f);
    }
    for (size_t i = 0; i < kv_elems; ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_kv[i] = (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 0.4f;
    }
    for (size_t i = 0; i < h_w.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_w[i] = 0.25f + 0.1f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
    }

    uint16_t *d_hc_ref = nullptr, *d_hc_split = nullptr;
    float *d_kv = nullptr, *d_w = nullptr, *d_pa = nullptr, *d_pb = nullptr;
    assert(hipMalloc(&d_hc_ref, hc_elems * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_hc_split, hc_elems * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_kv, kv_elems * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_w, h_w.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_pa, part_elems * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_pb, part_elems * sizeof(float)) == hipSuccess);

    assert(hipMemcpy(d_hc_ref, h_hc.data(), hc_elems * sizeof(uint16_t), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_hc_split, h_hc.data(), hc_elems * sizeof(uint16_t), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_kv, h_kv.data(), kv_elems * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);

    aff::engram_gate_hip(d_hc_ref, true, d_kv, d_w, n_embd, n_hc, nb, hc_stride, eps, nullptr);
    const uint32_t split_row = (n_hc * n_embd) / 2u;
    aff::engram_gate_partial_hip(d_hc_split, true, d_kv, d_w, d_pa, n_embd, n_hc, nb, hc_stride,
                                 0u, split_row, nullptr);
    aff::engram_gate_partial_hip(d_hc_split, true, d_kv, d_w, d_pb, n_embd, n_hc, nb, hc_stride,
                                 split_row, n_hc * n_embd, nullptr);
    aff::engram_gate_apply_hip(d_hc_split, true, d_kv, d_pa, d_pb, n_embd, n_hc, nb, hc_stride, eps,
                               nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<uint16_t> out_ref(hc_elems), out_split(hc_elems);
    assert(hipMemcpy(out_ref.data(), d_hc_ref, hc_elems * sizeof(uint16_t),
                     hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(out_split.data(), d_hc_split, hc_elems * sizeof(uint16_t),
                     hipMemcpyDeviceToHost) == hipSuccess);
    for (void* p : {(void*)d_hc_ref, (void*)d_hc_split, (void*)d_kv, (void*)d_w, (void*)d_pa,
                    (void*)d_pb})
      (void)hipFree(p);

    float max_diff = 0.0f;
    for (size_t i = 0; i < hc_elems; ++i)
      max_diff =
          std::max(max_diff, std::fabs(bf16_to_f32_cpu(out_ref[i]) - bf16_to_f32_cpu(out_split[i])));
    assert(max_diff <= 1e-3f);
    std::printf("PASS: wide engram_gate_tok parity (nb=%u, max_diff=%.3e)\n", nb, max_diff);
  }
  return 0;
}

int check_indexer_causal_skip_parity() {
  const uint32_t nb = 128u, n_keys = 512u, n_head = 16u, topk = 32u;
  const uint32_t kDim = aff::kIdxDim;  // 128
  std::vector<uint8_t> h_q8((size_t)nb * n_head * kDim), h_k8((size_t)n_keys * kDim);
  std::vector<float> h_qs(nb), h_ks(n_keys), h_hw((size_t)nb * n_head);
  std::vector<uint32_t> h_nc(nb);
  uint32_t rng = 4242u;
  auto gen_fp8 = [&](std::vector<uint8_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      uint8_t b = (uint8_t)(rng >> 24);
      v[i] = ((b & 0x7Fu) == 0x7Fu) ? 0x38u : b;
    }
  };
  gen_fp8(h_q8);
  gen_fp8(h_k8);
  for (uint32_t t = 0; t < nb; ++t) {
    rng = rng * 1664525u + 1013904223u;
    h_qs[t] = 0.05f + 0.002f * (float)(rng & 31u);
    h_nc[t] = 16u + t * 3u;  // strictly < 400 < n_keys so tail blocks skip
  }
  for (uint32_t k = 0; k < n_keys; ++k) {
    rng = rng * 1664525u + 1013904223u;
    h_ks[k] = 0.04f + 0.002f * (float)(rng & 31u);
  }
  for (size_t i = 0; i < h_hw.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_hw[i] = 0.5f + 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
  }

  uint8_t *d_q8 = nullptr, *d_k8 = nullptr;
  float *d_qs = nullptr, *d_ks = nullptr, *d_hw = nullptr, *d_out_ref = nullptr,
        *d_out_skip = nullptr;
  uint32_t *d_nc = nullptr, *d_adm_ref = nullptr, *d_adm_skip = nullptr, *d_admn_ref = nullptr,
           *d_admn_skip = nullptr;
  const size_t sc_cnt = (size_t)nb * n_keys;
  assert(hipMalloc(&d_q8, h_q8.size()) == hipSuccess);
  assert(hipMalloc(&d_k8, h_k8.size()) == hipSuccess);
  assert(hipMalloc(&d_qs, h_qs.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_ks, h_ks.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_hw, h_hw.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_nc, nb * sizeof(uint32_t)) == hipSuccess);
  assert(hipMalloc(&d_out_ref, sc_cnt * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_out_skip, sc_cnt * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_adm_ref, (size_t)nb * topk * sizeof(uint32_t)) == hipSuccess);
  assert(hipMalloc(&d_adm_skip, (size_t)nb * topk * sizeof(uint32_t)) == hipSuccess);
  assert(hipMalloc(&d_admn_ref, nb * sizeof(uint32_t)) == hipSuccess);
  assert(hipMalloc(&d_admn_skip, nb * sizeof(uint32_t)) == hipSuccess);

  assert(hipMemcpy(d_q8, h_q8.data(), h_q8.size(), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_k8, h_k8.data(), h_k8.size(), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_qs, h_qs.data(), h_qs.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_ks, h_ks.data(), h_ks.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_hw, h_hw.data(), h_hw.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_nc, h_nc.data(), nb * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemset(d_out_ref, 0, sc_cnt * sizeof(float)) == hipSuccess);
  assert(hipMemset(d_out_skip, 0x7f, sc_cnt * sizeof(float)) == hipSuccess);  // poison skipped area

  aff::indexer_scores_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_out_ref, nb, n_keys, n_head, nullptr,
                          nullptr);
  aff::indexer_scores_hip(d_q8, d_qs, d_k8, d_ks, d_hw, d_out_skip, nb, n_keys, n_head, nullptr,
                          d_nc);
  aff::idx_topk_mask_hip(d_out_ref, nullptr, d_nc, nb, n_keys, n_keys, n_keys, topk, nullptr,
                         nullptr, d_adm_ref, d_admn_ref, topk);
  aff::idx_topk_mask_hip(d_out_skip, nullptr, d_nc, nb, n_keys, n_keys, n_keys, topk, nullptr,
                         nullptr, d_adm_skip, d_admn_skip, topk);
  assert(hipDeviceSynchronize() == hipSuccess);

  std::vector<float> sc_ref(sc_cnt), sc_skip(sc_cnt);
  std::vector<uint32_t> adm_ref((size_t)nb * topk), adm_skip((size_t)nb * topk), admn_ref(nb),
      admn_skip(nb);
  assert(hipMemcpy(sc_ref.data(), d_out_ref, sc_cnt * sizeof(float), hipMemcpyDeviceToHost) ==
         hipSuccess);
  assert(hipMemcpy(sc_skip.data(), d_out_skip, sc_cnt * sizeof(float), hipMemcpyDeviceToHost) ==
         hipSuccess);
  assert(hipMemcpy(adm_ref.data(), d_adm_ref, adm_ref.size() * sizeof(uint32_t),
                   hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(adm_skip.data(), d_adm_skip, adm_skip.size() * sizeof(uint32_t),
                   hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(admn_ref.data(), d_admn_ref, admn_ref.size() * sizeof(uint32_t),
                   hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(admn_skip.data(), d_admn_skip, admn_skip.size() * sizeof(uint32_t),
                   hipMemcpyDeviceToHost) == hipSuccess);

  for (void* p : {(void*)d_q8, (void*)d_k8, (void*)d_qs, (void*)d_ks, (void*)d_hw, (void*)d_nc,
                  (void*)d_out_ref, (void*)d_out_skip, (void*)d_adm_ref, (void*)d_adm_skip,
                  (void*)d_admn_ref, (void*)d_admn_skip})
    (void)hipFree(p);

  for (uint32_t t = 0; t < nb; ++t) {
    for (uint32_t k = 0; k < h_nc[t]; ++k) {
      assert(sc_ref[(size_t)t * n_keys + k] == sc_skip[(size_t)t * n_keys + k]);
    }
    assert(admn_ref[t] == admn_skip[t]);
    for (uint32_t i = 0; i < admn_ref[t]; ++i) {
      assert(adm_ref[(size_t)t * topk + i] == adm_skip[(size_t)t * topk + i]);
    }
  }
  std::printf("PASS: indexer causal skip parity (nb=%u, n_keys=%u)\n", nb, n_keys);
  return 0;
}

int check_prefill_fuse_ar_hc_parity() {
  int nd = 0;
  if (hipGetDeviceCount(&nd) != hipSuccess || nd < 2) {
    std::printf("SKIP: check_prefill_fuse_ar_hc_parity (requires >=2 GPUs, found %d)\n", nd);
    return 0;
  }
  const uint32_t n_embd = 512u, inter = 256u, n_hc = 4u, n_live = 128u;
  std::vector<uint16_t> w_g(inter * n_embd), w_u(inter * n_embd), w_d(n_embd * inter);
  auto fill_bf16 = [](std::vector<uint16_t>& w, float scale, uint32_t seed) {
    uint32_t rng = seed;
    for (size_t i = 0; i < w.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      w[i] = f32_to_bf16_cpu(scale * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f));
    }
  };
  fill_bf16(w_g, 0.03f, 101u);
  fill_bf16(w_u, 0.03f, 202u);
  fill_bf16(w_d, 0.03f, 303u);

  std::vector<float> emb((size_t)n_live * n_embd);
  for (size_t i = 0; i < emb.size(); ++i) emb[i] = 0.4f * std::sin((float)i * 0.031f);

  auto run_arm = [&](const char* fuse_env, std::vector<float>* out) {
    ::setenv("AFF_PREFILL_FUSE_AR_HC", fuse_env, 1);
    aff::DenseGpu dg;
    std::string err;
    assert(dg.init(n_embd, n_embd, &err, 2));
    assert(dg.batch_init(n_live, n_embd, n_hc, 16u, 128u, 512u, 256u, 32u, &err));
    aff::Model::DenseOps dops = dg.ops();
    aff::Model::BatchOps bops = dg.batch_ops();
    const int32_t hg =
        dops.reg(dops.ctx, "blk.0.shexp_gate", w_g.data(), nullptr, aff::AFF_BF16, inter, n_embd);
    const int32_t hu =
        dops.reg(dops.ctx, "blk.0.shexp_up", w_u.data(), nullptr, aff::AFF_BF16, inter, n_embd);
    const int32_t hd =
        dops.reg(dops.ctx, "blk.0.shexp_down", w_d.data(), nullptr, aff::AFF_BF16, n_embd, inter);
    assert(hg >= 0 && hu >= 0 && hd >= 0);

    assert(bops.begin(bops.ctx, n_live, 0u, emb.data(), n_embd, n_hc));
    assert(bops.hc_pre(bops.ctx, -1, -1, -1, -1, n_embd, n_hc, 1u, 1e-6f, 1e-6f, nullptr));
    assert(bops.shexp(bops.ctx, hg, hu, hd, n_embd, inter, 0.0f));
    assert(bops.block_reduce(bops.ctx, n_embd));
    assert(bops.hc_post(bops.ctx, false, n_embd, n_hc));
    out->assign((size_t)n_live * n_embd, 0.0f);
    for (uint32_t b = 0; b < n_live; ++b) {
      assert(bops.hc_read(bops.ctx, b, out->data() + (size_t)b * n_embd, n_embd, n_hc));
    }
  };

  std::vector<float> out_ref, out_fused;
  run_arm("0", &out_ref);
  run_arm("1", &out_fused);
  ::unsetenv("AFF_PREFILL_FUSE_AR_HC");

  float max_diff = 0.0f;
  for (size_t i = 0; i < out_ref.size(); ++i)
    max_diff = std::max(max_diff, std::fabs(out_ref[i] - out_fused[i]));
  assert(max_diff <= 0.25f);
  std::printf("PASS: prefill AFF_PREFILL_FUSE_AR_HC parity (nb=%u, max_diff=%.3e)\n", n_live,
              max_diff);
  return 0;
}

int check_moe_plan_large_tok_lds() {
  const uint32_t ne = 384u, k = 8u, nb = 5632u;
  const size_t lds_5632 = aff::moe_plan_lds_bytes(ne, nb);
  const size_t lds_8192 = aff::moe_plan_lds_bytes(ne, 8192u);
  assert(lds_5632 <= 65536u);
  assert(lds_8192 <= 65536u);

  const uint32_t total = nb * k;
  std::vector<uint64_t> h_plc(ne, 0x100000ull | 1ull);  // all resident for cls=0 tile128 check
  std::vector<uint32_t> h_sel(total);
  std::vector<float> h_wt(total, 1.0f / (float)k);
  for (uint32_t t = 0; t < nb; ++t) {
    for (uint32_t j = 0; j < k; ++j) h_sel[(size_t)t * k + j] = j + (t & 15u) * k;
  }

  aff::ExpertShardLayout sh{};
  sh.stride = 4096u;
  for (int pl = 0; pl < 3; ++pl) {
    sh.data_off[pl] = 256u * (uint32_t)(pl + 1);
    sh.scale_off[pl] = 1024u + 256u * (uint32_t)pl;
    sh.rscale_off[pl] = 2048u + 256u * (uint32_t)pl;
  }

  for (bool res_128 : {false, true}) {
    uint32_t gcap[8] = {}, goff[8] = {};
    const uint32_t pstride = aff::moe_plan_bucket_caps(nb, k, ne, false, gcap, res_128);
    for (uint32_t g = 1; g < 8u; ++g) goff[g] = goff[g - 1u] + gcap[g - 1u];

    uint32_t *d_sel = nullptr, *d_tok = nullptr, *d_ord = nullptr, *d_run = nullptr,
             *d_col = nullptr, *d_cnt = nullptr;
    uint64_t* d_plc = nullptr;
    float *d_wt = nullptr, *d_wto = nullptr;
    uint16_t *d_bg = nullptr, *d_bu = nullptr, *d_bydn = nullptr;
    aff::ExpertTile* d_tiles = nullptr;
    aff::MoePlanStatus* d_st = nullptr;

    assert(hipMalloc(&d_sel, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wt, total * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_plc, ne * sizeof(uint64_t)) == hipSuccess);
    assert(hipMalloc(&d_tok, 2u * total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wto, total * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_ord, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_run, (nb + 1u) * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_col, total * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_tiles, (size_t)3u * pstride * sizeof(aff::ExpertTile)) == hipSuccess);
    assert(hipMalloc(&d_cnt, 8u * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_st, sizeof(aff::MoePlanStatus)) == hipSuccess);
    assert(hipMalloc(&d_bg, 64u) == hipSuccess);
    assert(hipMalloc(&d_bu, 64u) == hipSuccess);
    assert(hipMalloc(&d_bydn, 64u) == hipSuccess);

    assert(hipMemset(d_st, 0, sizeof(aff::MoePlanStatus)) == hipSuccess);
    assert(hipMemcpy(d_sel, h_sel.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_wt, h_wt.data(), total * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_plc, h_plc.data(), ne * sizeof(uint64_t), hipMemcpyHostToDevice) ==
           hipSuccess);

    aff::MoePlanArgs A{};
    A.sel = d_sel;
    A.wt = d_wt;
    A.plc = d_plc;
    A.layer = 0;
    A.nb_tok = nb;
    A.k = k;
    A.n_expert = ne;
    A.tok = d_tok;
    A.wt_out = d_wto;
    A.ord = d_ord;
    A.run = d_run;
    A.slot_col = d_col;
    A.tiles = d_tiles;
    A.counts = d_cnt;
    A.status = d_st;
    A.shard = sh;
    A.bg = d_bg;
    A.bu = d_bu;
    A.bydn = d_bydn;
    A.plane_stride = pstride;
    for (uint32_t g = 0; g < 8u; ++g) {
      A.group_off[g] = goff[g];
      A.group_cap[g] = gcap[g];
    }
    A.res_tile128 = res_128;
    assert(aff::moe_plan_hip(A, nullptr));
    assert(hipDeviceSynchronize() == hipSuccess);

    aff::MoePlanStatus st{};
    uint32_t h_cnt[8] = {};
    assert(hipMemcpy(&st, d_st, sizeof(st), hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipMemcpy(h_cnt, d_cnt, sizeof(h_cnt), hipMemcpyDeviceToHost) == hipSuccess);
    assert(st.taken == total && st.bad == 0u && st.over == 0u);
    if (res_128) assert(h_cnt[3] > 0u);
    else assert(h_cnt[3] == 0u);

    for (void* p : {(void*)d_sel, (void*)d_wt, (void*)d_plc, (void*)d_tok, (void*)d_wto,
                    (void*)d_ord, (void*)d_run, (void*)d_col, (void*)d_tiles, (void*)d_cnt,
                    (void*)d_st, (void*)d_bg, (void*)d_bu, (void*)d_bydn})
      (void)hipFree(p);
  }
  std::printf("PASS: moe_plan large-token LDS & res_tile128 (nb=5632 lds=%zu, nb=8192 lds=%zu)\n",
              lds_5632, lds_8192);
  return 0;
}

}  // namespace

int main() {
  int n_dev = 0;
  if (hipGetDeviceCount(&n_dev) != hipSuccess || n_dev < 1) {
    std::fprintf(stderr, "No HIP devices available\n");
    return 1;
  }
  if (check_batch_slice_tail_parity() != 0) return 1;
  if (check_engram_gate_wide_tok_parity() != 0) return 1;
  if (check_indexer_causal_skip_parity() != 0) return 1;
  if (check_prefill_fuse_ar_hc_parity() != 0) return 1;
  if (check_moe_plan_large_tok_lds() != 0) return 1;
  std::printf("PASS: test_v512_kernels\n");
  return 0;
}
