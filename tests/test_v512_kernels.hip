#include "gpu/attention_gpu.h"
#include "gpu/attn_flash.h"
#include "gpu/batch_kernels.h"
#include "gpu/dense_gpu.h"
#include "gpu/engram_gpu.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"
#include "gpu/indexer_gpu.h"
#include "gpu/moe_plan.h"
#include "gpu/static_placement.h"
#include "gpu/wmma.h"
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
#include <sys/mman.h>

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
      w_kv(width * n_embd), w_fn((n_hc * 2u + n_hc * n_hc) * (n_embd * n_hc));
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
    assert(dg.init(n_head * width, n_embd * n_hc, &err, 1));
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
                                  n_hc * 2u + n_hc * n_hc, n_embd * n_hc);
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
    out_hc->assign((size_t)keep * n_embd * n_hc, 0.0f);
    for (uint32_t b = 0; b < keep; ++b) {
      assert(bops.hc_read(bops.ctx, b, out_hc->data() + (size_t)b * n_embd * n_hc, n_embd, n_hc));
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
    assert(dg.init(n_embd * n_hc, n_embd * n_hc, &err, 2));
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
    out->assign((size_t)n_live * n_embd * n_hc, 0.0f);
    for (uint32_t b = 0; b < n_live; ++b) {
      assert(bops.hc_read(bops.ctx, b, out->data() + (size_t)b * n_embd * n_hc, n_embd, n_hc));
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

__global__ void test_bf16x2_rne_kernel(const float* in_lo, const float* in_hi, uint32_t* out,
                                       uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = aff::wmma::bf16x2_rne(in_lo[i], in_hi[i]);
}

int check_bf16x2_rne_parity() {
  auto bits_to_f32 = [](uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };
  std::vector<float> lo, hi;
  const uint32_t specials[] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u,
      0x3f807fffu, 0x3f808001u, 0x00008000u, 0x00018000u, 0x007f8000u, 0x7f7fffffu, 0xff7fffffu,
      0x7f800000u, 0xff800000u, 0x7f800001u, 0x7fc00001u, 0xffffffffu, 0x42488000u, 0x42498000u,
  };
  for (uint32_t a : specials) {
    for (uint32_t b : specials) {
      lo.push_back(bits_to_f32(a));
      hi.push_back(bits_to_f32(b));
    }
  }
  uint32_t rng = 0x12345678u;
  for (int i = 0; i < 4096; ++i) {
    rng = rng * 1664525u + 1013904223u;
    lo.push_back(bits_to_f32(rng));
    rng = rng * 1664525u + 1013904223u;
    hi.push_back(bits_to_f32(rng));
  }
  const uint32_t n = (uint32_t)lo.size();
  float *d_lo = nullptr, *d_hi = nullptr;
  uint32_t* d_out = nullptr;
  assert(hipMalloc(&d_lo, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_hi, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_out, n * sizeof(uint32_t)) == hipSuccess);
  assert(hipMemcpy(d_lo, lo.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_hi, hi.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  hipLaunchKernelGGL(test_bf16x2_rne_kernel, dim3((n + 255u) / 256u), dim3(256), 0, nullptr,
                     d_lo, d_hi, d_out, n);
  assert(hipGetLastError() == hipSuccess);
  assert(hipDeviceSynchronize() == hipSuccess);
  std::vector<uint32_t> got(n);
  assert(hipMemcpy(got.data(), d_out, n * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess);
  (void)hipFree(d_lo);
  (void)hipFree(d_hi);
  (void)hipFree(d_out);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t exp =
        (uint32_t)f32_to_bf16_cpu(lo[i]) | ((uint32_t)f32_to_bf16_cpu(hi[i]) << 16);
    assert(got[i] == exp);
  }
  std::printf("PASS: bf16x2_rne bit-exact parity (n=%u)\n", n);
  return 0;
}

int check_router_topk_v2_parity() {
  struct Case {
    uint32_t ne, live, nb, k;
    int norm;
    float scale, res_bias;
    bool use_res, has_prot, sparse_res;
  };
  const Case cases[] = {
      {256u, 8u, 16u, 8u, 1, 2.5f, 0.0f, false, false, false},
      {256u, 8u, 8u, 8u, 1, 2.5f, 0.0f, true, false, false},
      {256u, 4u, 8u, 8u, 1, 1.0f, 0.25f, true, false, false},
      {256u, 4u, 8u, 8u, 1, 1.0f, 0.25f, true, true, false},
      {256u, 2u, 4u, 8u, 0, 1.5f, 0.0f, true, false, true},
      {384u, 4u, 8u, 8u, 1, 2.0f, 0.2f, true, false, false},
      {512u, 4u, 8u, 8u, 1, 2.0f, 0.0f, false, false, false},
  };
  uint32_t rng = 777u;
  for (const Case& c : cases) {
    std::vector<float> logits((size_t)c.ne * c.nb), bias(c.ne);
    std::vector<uint8_t> res(c.ne, 1u);
    for (uint32_t e = 0; e < c.ne; ++e) {
      rng = rng * 1664525u + 1013904223u;
      bias[e] = 0.05f * (float)((int)(rng & 7u) - 3);
      if (c.use_res) res[e] = (e & 1u) ? 1u : 0u;
    }
    if (c.sparse_res) {
      std::fill(res.begin(), res.end(), 0u);
      res[5] = 1u;
      res[17] = 1u;
      res[103] = 1u;
    }
    if (c.has_prot) res[12] = 3u;
    for (size_t i = 0; i < logits.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      logits[i] = 0.1f * (float)((int)(rng & 15u) - 8);
    }
    // Inject exact score ties across multiple expert IDs to verify lower-ID tie break.
    for (uint32_t b = 0; b < c.live; ++b) {
      for (uint32_t e : {3u, 19u, 64u, 133u, 255u}) {
        if (e < c.ne) {
          logits[(size_t)e * c.nb + b] = (e == 19u) ? -0.0f : 0.0f;
          bias[e] = 0.25f;
        }
      }
      if (c.has_prot) {
        logits[(size_t)12u * c.nb + b] = 25.0f;
        bias[12] = 1.0f;
      }
    }

    float *d_log = nullptr, *d_bias = nullptr, *d_wt0 = nullptr, *d_wt1 = nullptr;
    uint8_t* d_res = nullptr;
    uint32_t *d_sel0 = nullptr, *d_sel1 = nullptr, *d_ns0 = nullptr, *d_ns1 = nullptr;
    assert(hipMalloc(&d_log, logits.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_bias, bias.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_res, res.size()) == hipSuccess);
    assert(hipMalloc(&d_sel0, (size_t)c.live * c.k * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_sel1, (size_t)c.live * c.k * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_wt0, (size_t)c.live * c.k * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_wt1, (size_t)c.live * c.k * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_ns0, c.live * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_ns1, c.live * sizeof(uint32_t)) == hipSuccess);
    assert(hipMemcpy(d_log, logits.data(), logits.size() * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_res, res.data(), res.size(), hipMemcpyHostToDevice) == hipSuccess);

    ::setenv("AFF_ROUTER_TOPK_V2", "0", 1);
    aff::router_topk_hip(d_log, d_bias, c.use_res ? d_res : nullptr, d_sel0, d_wt0, d_ns0, c.ne,
                         c.nb, c.live, c.k, c.norm, c.scale, nullptr, c.res_bias);
    ::setenv("AFF_ROUTER_TOPK_V2", "1", 1);
    aff::router_topk_hip(d_log, d_bias, c.use_res ? d_res : nullptr, d_sel1, d_wt1, d_ns1, c.ne,
                         c.nb, c.live, c.k, c.norm, c.scale, nullptr, c.res_bias);
    ::unsetenv("AFF_ROUTER_TOPK_V2");
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<uint32_t> s0((size_t)c.live * c.k), s1((size_t)c.live * c.k), ns0(c.live),
        ns1(c.live);
    std::vector<float> w0((size_t)c.live * c.k), w1((size_t)c.live * c.k);
    assert(hipMemcpy(s0.data(), d_sel0, s0.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(s1.data(), d_sel1, s1.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(w0.data(), d_wt0, w0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(w1.data(), d_wt1, w1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(ns0.data(), d_ns0, ns0.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(ns1.data(), d_ns1, ns1.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
           hipSuccess);

    for (void* p : {(void*)d_log, (void*)d_bias, (void*)d_res, (void*)d_sel0, (void*)d_sel1,
                    (void*)d_wt0, (void*)d_wt1, (void*)d_ns0, (void*)d_ns1})
      (void)hipFree(p);

    assert(ns0 == ns1);
    for (uint32_t b = 0; b < c.live; ++b) {
      for (uint32_t i = 0; i < ns0[b]; ++i) {
        const size_t idx = (size_t)b * c.k + i;
        assert(s0[idx] == s1[idx]);
        assert(std::memcmp(&w0[idx], &w1[idx], sizeof(float)) == 0);
      }
    }
  }
  std::printf("PASS: router_topk_v2 bit-exact parity (ties, res_bias, prot, ne=256/384/512)\n");
  return 0;
}

int check_mhc_mix_fast_parity() {
  const uint32_t n_embd = 5120u, n_hc = 4u, nb_n = aff::kNarrowTok;
  const uint32_t n_mix = 2u * n_hc + n_hc * n_hc;
  const size_t hc_sz = (size_t)n_hc * n_embd * nb_n;
  std::vector<uint16_t> h_hc(hc_sz);
  std::vector<float> h_mix((size_t)n_mix * nb_n), h_scale(3u, 0.25f), h_base(n_mix, 0.1f),
      h_nw(n_embd, 1.0f), h_pre_in((size_t)n_hc * nb_n, 0.25f);
  uint32_t rng = 31337u;
  for (size_t i = 0; i < hc_sz; ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_hc[i] = f32_to_bf16_cpu((((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 0.5f);
  }
  for (size_t i = 0; i < h_mix.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_mix[i] = (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 0.2f;
  }

  uint16_t* d_hc = nullptr;
  float *d_mix = nullptr, *d_sc = nullptr, *d_base = nullptr, *d_nw = nullptr, *d_pin = nullptr;
  float *d_pre0 = nullptr, *d_post0 = nullptr, *d_comb0 = nullptr, *d_cur0 = nullptr,
        *d_norm0 = nullptr, *d_part0 = nullptr, *d_qs0 = nullptr;
  float *d_pre1 = nullptr, *d_post1 = nullptr, *d_comb1 = nullptr, *d_cur1 = nullptr,
        *d_norm1 = nullptr, *d_part1 = nullptr, *d_qs1 = nullptr;
  uint8_t *d_q80 = nullptr, *d_q81 = nullptr;
  const size_t nkb = n_embd / 128u;
  assert(hipMalloc(&d_hc, hc_sz * sizeof(uint16_t)) == hipSuccess);
  assert(hipMalloc(&d_mix, h_mix.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_sc, h_scale.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_base, h_base.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_nw, h_nw.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_pin, h_pre_in.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_pre0, h_pre_in.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_pre1, h_pre_in.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_post0, h_pre_in.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_post1, h_pre_in.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_comb0, (size_t)n_hc * n_hc * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_comb1, (size_t)n_hc * n_hc * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_cur0, (size_t)n_embd * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_cur1, (size_t)n_embd * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_norm0, (size_t)n_embd * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_norm1, (size_t)n_embd * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_part0, (size_t)2u * aff::kHcnChunks * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_part1, (size_t)2u * aff::kHcnChunks * nb_n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_q80, (size_t)nb_n * n_embd) == hipSuccess);
  assert(hipMalloc(&d_q81, (size_t)nb_n * n_embd) == hipSuccess);
  assert(hipMalloc(&d_qs0, (size_t)nb_n * nkb * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_qs1, (size_t)nb_n * nkb * sizeof(float)) == hipSuccess);

  assert(hipMemcpy(d_hc, h_hc.data(), hc_sz * sizeof(uint16_t), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_mix, h_mix.data(), h_mix.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_sc, h_scale.data(), h_scale.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_base, h_base.data(), h_base.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_nw, h_nw.data(), h_nw.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_pin, h_pre_in.data(), h_pre_in.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);

  for (const float* pin : {(const float*)d_pin, (const float*)nullptr}) {
    ::setenv("AFF_MHC_MIX_FAST", "0", 1);
    aff::batch_hc_fused_narrow_hip(d_hc, d_mix, d_sc, d_base, d_nw, d_pre0, d_post0, d_comb0,
                                   d_cur0, d_norm0, d_part0, d_q80, d_qs0, n_embd, n_hc, nb_n,
                                   20u, 1e-6f, 1e-6f, 1, nullptr, pin);
    ::setenv("AFF_MHC_MIX_FAST", "1", 1);
    aff::batch_hc_fused_narrow_hip(d_hc, d_mix, d_sc, d_base, d_nw, d_pre1, d_post1, d_comb1,
                                   d_cur1, d_norm1, d_part1, d_q81, d_qs1, n_embd, n_hc, nb_n,
                                   20u, 1e-6f, 1e-6f, 1, nullptr, pin);
    ::unsetenv("AFF_MHC_MIX_FAST");
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> c0((size_t)n_embd * nb_n), c1((size_t)n_embd * nb_n),
        n0((size_t)n_embd * nb_n), n1((size_t)n_embd * nb_n),
        cb0((size_t)n_hc * n_hc * nb_n), cb1((size_t)n_hc * n_hc * nb_n);
    assert(hipMemcpy(c0.data(), d_cur0, c0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(c1.data(), d_cur1, c1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(n0.data(), d_norm0, n0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(n1.data(), d_norm1, n1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(cb0.data(), d_comb0, cb0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(cb1.data(), d_comb1, cb1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(std::memcmp(c0.data(), c1.data(), c0.size() * sizeof(float)) == 0);
    for (size_t i = 0; i < n0.size(); ++i) assert(std::fabs(n0[i] - n1[i]) <= 1e-5f);
    for (size_t i = 0; i < cb0.size(); ++i) assert(std::fabs(cb0[i] - cb1[i]) <= 1e-5f);
  }

  for (void* p : {(void*)d_hc, (void*)d_mix, (void*)d_sc, (void*)d_base, (void*)d_nw, (void*)d_pin,
                  (void*)d_pre0, (void*)d_pre1, (void*)d_post0, (void*)d_post1, (void*)d_comb0,
                  (void*)d_comb1, (void*)d_cur0, (void*)d_cur1, (void*)d_norm0, (void*)d_norm1,
                  (void*)d_part0, (void*)d_part1, (void*)d_q80, (void*)d_q81, (void*)d_qs0,
                  (void*)d_qs1})
    (void)hipFree(p);
  std::printf("PASS: mhc_mix_fast parity (20-WG split-K + Kahan, cur exact, norm/comb <=1e-5)\n");
  return 0;
}

int check_flash_exp2_klate_parity() {
  const uint32_t n = 2u, n_head = 32u, n_raw = 16u, n_comp = 192u, ring = 64u, sliding = 64u;
  const uint32_t width = aff::kFlashWidth;
  std::vector<float> h_q((size_t)n * n_head * width), h_sinks(n_head);
  std::vector<uint8_t> h_raw((size_t)ring * aff::kFlashRowBytes),
      h_comp((size_t)n_comp * aff::kFlashRowBytes), h_allow((size_t)n * n_comp, 1u);
  std::vector<float> row(width);
  uint32_t rng = 5150u;
  for (size_t i = 0; i < h_q.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_q[i] = 0.3f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
  }
  for (uint32_t h = 0; h < n_head; ++h) h_sinks[h] = -0.5f + 0.03f * (float)h;
  for (uint32_t r = 0; r < ring; ++r) {
    for (uint32_t d = 0; d < width; ++d) {
      rng = rng * 1664525u + 1013904223u;
      row[d] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
    }
    aff::kv_encode_fp8r_row(row.data(), h_raw.data() + (size_t)r * aff::kFlashRowBytes);
  }
  for (uint32_t r = 0; r < n_comp; ++r) {
    for (uint32_t d = 0; d < width; ++d) {
      rng = rng * 1664525u + 1013904223u;
      row[d] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
    }
    aff::kv_encode_fp8r_row(row.data(), h_comp.data() + (size_t)r * aff::kFlashRowBytes);
  }
  for (size_t i = 0; i < h_allow.size(); ++i)
    if ((i % 7u) == 0u) h_allow[i] = 0u;

  float *d_q = nullptr, *d_sinks = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
  uint8_t *d_raw = nullptr, *d_comp = nullptr, *d_allow = nullptr;
  assert(hipMalloc(&d_q, h_q.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_sinks, h_sinks.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_raw, h_raw.size()) == hipSuccess);
  assert(hipMalloc(&d_comp, h_comp.size()) == hipSuccess);
  assert(hipMalloc(&d_allow, h_allow.size()) == hipSuccess);
  assert(hipMalloc(&d_o0, h_q.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_o1, h_q.size() * sizeof(float)) == hipSuccess);
  assert(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * sizeof(float),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_comp, h_comp.data(), h_comp.size(), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_allow, h_allow.data(), h_allow.size(), hipMemcpyHostToDevice) == hipSuccess);

  const uint32_t pos0 = n_raw - 1u;
  const float scale = 1.0f / std::sqrt((float)width);
  ::setenv("AFF_FLASH_EXP2", "0", 1);
  ::setenv("AFF_FLASH_KLATE", "0", 1);
  aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, d_allow, nullptr, n_comp, d_sinks, d_o0, 0u, n, n,
                            pos0, sliding, ring, n_comp, n_head, scale, nullptr);
  ::setenv("AFF_FLASH_EXP2", "1", 1);
  ::setenv("AFF_FLASH_KLATE", "1", 1);
  aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, d_allow, nullptr, n_comp, d_sinks, d_o1, 0u, n, n,
                            pos0, sliding, ring, n_comp, n_head, scale, nullptr);
  ::unsetenv("AFF_FLASH_EXP2");
  ::unsetenv("AFF_FLASH_KLATE");
  assert(hipDeviceSynchronize() == hipSuccess);

  std::vector<float> o0(h_q.size()), o1(h_q.size());
  assert(hipMemcpy(o0.data(), d_o0, o0.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(o1.data(), d_o1, o1.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  for (void* p :
       {(void*)d_q, (void*)d_sinks, (void*)d_raw, (void*)d_comp, (void*)d_allow, (void*)d_o0,
        (void*)d_o1})
    (void)hipFree(p);
  float max_d = 0.0f;
  for (size_t i = 0; i < o0.size(); ++i) max_d = std::max(max_d, std::fabs(o0[i] - o1[i]));
  assert(max_d <= 5e-5f);
  std::printf("PASS: flash_exp2_klate parity (max_diff=%.3e)\n", max_d);
  return 0;
}

int check_expert_had32_dpp_parity() {
  const uint32_t dim = 512u, total = 20u, nkb = dim / 128u;
  std::vector<uint16_t> h_g((size_t)dim * total), h_u((size_t)dim * total);
  std::vector<float> h_w(total);
  uint32_t rng = 8888u;
  for (size_t i = 0; i < h_g.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_g[i] = f32_to_bf16_cpu((((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 2.0f);
    rng = rng * 1664525u + 1013904223u;
    h_u[i] = f32_to_bf16_cpu((((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f) * 2.0f);
  }
  for (uint32_t s = 0; s < total; ++s) h_w[s] = 0.5f + 0.05f * (float)s;

  uint16_t *d_g = nullptr, *d_u = nullptr;
  float *d_w = nullptr, *d_s0 = nullptr, *d_s1 = nullptr;
  uint8_t *d_o0 = nullptr, *d_o1 = nullptr;
  assert(hipMalloc(&d_g, h_g.size() * sizeof(uint16_t)) == hipSuccess);
  assert(hipMalloc(&d_u, h_u.size() * sizeof(uint16_t)) == hipSuccess);
  assert(hipMalloc(&d_w, h_w.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_o0, (size_t)total * dim) == hipSuccess);
  assert(hipMalloc(&d_o1, (size_t)total * dim) == hipSuccess);
  assert(hipMalloc(&d_s0, (size_t)total * nkb * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_s1, (size_t)total * nkb * sizeof(float)) == hipSuccess);
  assert(hipMemcpy(d_g, h_g.data(), h_g.size() * sizeof(uint16_t), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_u, h_u.data(), h_u.size() * sizeof(uint16_t), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice) ==
         hipSuccess);

  ::setenv("AFF_HADAMARD_ROT", "1", 1);
  ::setenv("AFF_EXPERT_HAD32_DPP", "0", 1);
  aff::expert_swiglu_quant_hip(d_g, d_u, d_w, d_o0, d_s0, dim, total, 7.0f, nullptr);
  ::setenv("AFF_EXPERT_HAD32_DPP", "1", 1);
  aff::expert_swiglu_quant_hip(d_g, d_u, d_w, d_o1, d_s1, dim, total, 7.0f, nullptr);
  ::unsetenv("AFF_HADAMARD_ROT");
  ::unsetenv("AFF_EXPERT_HAD32_DPP");
  assert(hipDeviceSynchronize() == hipSuccess);

  std::vector<uint8_t> o0((size_t)total * dim), o1((size_t)total * dim);
  std::vector<float> s0((size_t)total * nkb), s1((size_t)total * nkb);
  assert(hipMemcpy(o0.data(), d_o0, o0.size(), hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(o1.data(), d_o1, o1.size(), hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(s0.data(), d_s0, s0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
         hipSuccess);
  assert(hipMemcpy(s1.data(), d_s1, s1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
         hipSuccess);
  for (void* p :
       {(void*)d_g, (void*)d_u, (void*)d_w, (void*)d_o0, (void*)d_o1, (void*)d_s0, (void*)d_s1})
    (void)hipFree(p);
  assert(o0 == o1);
  assert(std::memcmp(s0.data(), s1.data(), s0.size() * sizeof(float)) == 0);
  std::printf("PASS: expert_had32_dpp bit-exact parity\n");
  return 0;
}

int check_moe_plan_dpp_parity() {
  aff::ExpertShardLayout sh{};
  sh.stride = 4096u;
  for (int pl = 0; pl < 3; ++pl) {
    sh.data_off[pl] = 256u * (uint32_t)(pl + 1);
    sh.scale_off[pl] = 1024u + 256u * (uint32_t)pl;
    sh.rscale_off[pl] = 2048u + 256u * (uint32_t)pl;
  }
  for (uint32_t ne : {256u, 512u}) {
    for (uint32_t nb : {16u, 128u}) {
      const uint32_t k = 8u, total = nb * k;
      uint32_t gcap[8] = {}, goff[8] = {};
      const uint32_t pstride = aff::moe_plan_bucket_caps(nb, k, ne, false, gcap, false);
      for (uint32_t g = 1u; g < 8u; ++g) goff[g] = goff[g - 1u] + gcap[g - 1u];
      std::vector<uint32_t> h_sel(total);
      std::vector<float> h_wt(total);
      std::vector<uint64_t> h_plc(ne, 0x100000ull | 1ull);
      uint32_t rng = 424242u + ne + nb;
      for (uint32_t b = 0; b < nb; ++b) {
        for (uint32_t j = 0; j < k; ++j) {
          rng = rng * 1664525u + 1013904223u;
          h_sel[(size_t)b * k + j] = (j * 29u + (rng & 31u)) % ne;
          h_wt[(size_t)b * k + j] = 0.1f + 0.05f * (float)j;
        }
      }
      uint32_t *d_sel = nullptr, *d_tok = nullptr, *d_ord = nullptr, *d_run = nullptr,
               *d_col = nullptr, *d_cnt = nullptr;
      float *d_wt = nullptr, *d_wto = nullptr;
      uint64_t* d_plc = nullptr;
      aff::ExpertTile* d_tiles = nullptr;
      aff::MoePlanStatus* d_st = nullptr;
      uint16_t *d_bg = nullptr, *d_bu = nullptr, *d_bydn = nullptr;
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
      assert(hipMemcpy(d_wt, h_wt.data(), total * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);
      assert(hipMemcpy(d_plc, h_plc.data(), ne * sizeof(uint64_t), hipMemcpyHostToDevice) ==
             hipSuccess);
      aff::MoePlanArgs A{};
      A.sel = d_sel;
      A.wt = d_wt;
      A.plc = d_plc;
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
      assert(aff::moe_plan_hip(A, nullptr));
      assert(hipDeviceSynchronize() == hipSuccess);
      std::vector<uint32_t> h_run(nb + 1u), h_ord(total), h_cnt(8u);
      aff::MoePlanStatus st{};
      assert(hipMemcpy(h_run.data(), d_run, (nb + 1u) * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
             hipSuccess);
      assert(hipMemcpy(h_ord.data(), d_ord, total * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
             hipSuccess);
      assert(hipMemcpy(h_cnt.data(), d_cnt, 8u * sizeof(uint32_t), hipMemcpyDeviceToHost) ==
             hipSuccess);
      assert(hipMemcpy(&st, d_st, sizeof(st), hipMemcpyDeviceToHost) == hipSuccess);
      for (void* p : {(void*)d_sel, (void*)d_wt, (void*)d_plc, (void*)d_tok, (void*)d_wto,
                      (void*)d_ord, (void*)d_run, (void*)d_col, (void*)d_tiles, (void*)d_cnt,
                      (void*)d_st, (void*)d_bg, (void*)d_bu, (void*)d_bydn})
        (void)hipFree(p);
      assert(st.taken == total && st.bad == 0u && st.over == 0u);
      for (uint32_t b = 0; b <= nb; ++b) assert(h_run[b] == b * k);
      for (uint32_t b = 0; b < nb; ++b)
        for (uint32_t m = 1u; m < k; ++m)
          assert(h_ord[(size_t)b * k + m - 1u] < h_ord[(size_t)b * k + m]);
    }
  }
  std::printf("PASS: moe_plan_dpp prefix scan parity (ne=256/512, nb=16/128)\n");
  return 0;
}

int check_flash_flat_win_parity() {
  const uint32_t n = 5u, n_head = 32u, width = aff::kFlashWidth, n_rot = 64u;
  const uint32_t ring = 256u, sliding = 128u, flat_win = 133u, flat_raw0 = 180u;
  std::vector<uint16_t> h_q((size_t)n * n_head * width);
  std::vector<float> h_sinks(n_head), row(width);
  std::vector<uint8_t> h_raw((size_t)ring * aff::kFlashRowBytes);
  uint32_t rng = 4040u;
  for (size_t i = 0; i < h_q.size(); ++i) {
    rng = rng * 1664525u + 1013904223u;
    h_q[i] = f32_to_bf16_cpu(0.3f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f));
  }
  for (uint32_t h = 0; h < n_head; ++h) h_sinks[h] = -0.4f + 0.02f * (float)h;
  for (uint32_t r = 0; r < ring; ++r) {
    for (uint32_t d = 0; d < width; ++d) {
      rng = rng * 1664525u + 1013904223u;
      row[d] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
    }
    aff::kv_encode_fp8r_row(row.data(), h_raw.data() + (size_t)r * aff::kFlashRowBytes);
  }

  uint16_t* d_q = nullptr;
  float *d_sinks = nullptr, *d_ob = nullptr, *d_of = nullptr;
  uint8_t* d_raw = nullptr;
  assert(hipMalloc(&d_q, h_q.size() * sizeof(uint16_t)) == hipSuccess);
  assert(hipMalloc(&d_sinks, h_sinks.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_raw, h_raw.size()) == hipSuccess);
  assert(hipMalloc(&d_ob, h_q.size() * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_of, h_q.size() * sizeof(float)) == hipSuccess);
  assert(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(uint16_t), hipMemcpyHostToDevice) ==
         hipSuccess);
  assert(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * sizeof(float),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice) == hipSuccess);

  const float scale = 1.0f / std::sqrt((float)width);
  for (int with_sink = 0; with_sink < 2; ++with_sink) {
    const float* sp = with_sink ? d_sinks : nullptr;
    aff::mqa_attend_batch_hip(aff::KvDtype::FP8R, d_q, d_raw, nullptr, nullptr, nullptr, sp, d_ob,
                              0u, n, n, 100u, sliding, ring, 0u, 0u, n_head, width, n_rot, scale,
                              nullptr, /*q_bf16=*/true, flat_win, flat_raw0);
    aff::mqa_attend_flash_hip(d_q, d_raw, nullptr, nullptr, nullptr, 0u, sp, d_of, 0u, n, n, 100u,
                              sliding, ring, 0u, n_head, scale, nullptr, /*q_bf16=*/true, nullptr,
                              0u, nullptr, 0u, 0u, nullptr, flat_win, flat_raw0);
    assert(hipDeviceSynchronize() == hipSuccess);
    std::vector<float> hb(h_q.size()), hf(h_q.size());
    assert(hipMemcpy(hb.data(), d_ob, hb.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(hf.data(), d_of, hf.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    float max_err = 0.0f;
    for (size_t i = 0; i < hb.size(); ++i) {
      assert(std::isfinite(hf[i]));
      max_err = std::max(max_err, std::fabs(hb[i] - hf[i]));
    }
    assert(max_err <= 2.5e-2f);
  }
  for (void* p : {(void*)d_q, (void*)d_sinks, (void*)d_raw, (void*)d_ob, (void*)d_of})
    (void)hipFree(p);
  std::printf("PASS: mqa_attend_flash_hip flat_win parity vs mqa_attend_batch_hip (q_bf16=true)\n");
  return 0;
}

int check_flash_lowm_nosplit_parity() {
  const uint32_t n_head = 32u, ring = 128u, sliding = 64u, width = aff::kFlashWidth;
  struct Case { uint32_t n_tok, n_raw, n_comp; };
  const Case cases[] = {{2u, 64u, 64u}, {4u, 64u, 192u}};  // tiles=2 and n_tok=4,tiles=4
  const float scale = 1.0f / std::sqrt((float)width);
  float worst_diff = 0.0f;
  for (const Case& c : cases) {
    std::vector<float> h_q((size_t)c.n_tok * n_head * width), h_sinks(n_head), row(width);
    std::vector<uint8_t> h_raw((size_t)ring * aff::kFlashRowBytes),
        h_comp((size_t)c.n_comp * aff::kFlashRowBytes);
    uint32_t rng = 7700u + c.n_tok;
    for (size_t i = 0; i < h_q.size(); ++i) {
      rng = rng * 1664525u + 1013904223u;
      h_q[i] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
    }
    for (uint32_t h = 0; h < n_head; ++h) h_sinks[h] = -0.3f + 0.01f * (float)h;
    for (uint32_t r = 0; r < ring; ++r) {
      for (uint32_t d = 0; d < width; ++d) {
        rng = rng * 1664525u + 1013904223u;
        row[d] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
      }
      aff::kv_encode_fp8r_row(row.data(), h_raw.data() + (size_t)r * aff::kFlashRowBytes);
    }
    for (uint32_t r = 0; r < c.n_comp; ++r) {
      for (uint32_t d = 0; d < width; ++d) {
        rng = rng * 1664525u + 1013904223u;
        row[d] = 0.25f * (((float)(int32_t)(rng >> 16) / 32768.0f) - 1.0f);
      }
      aff::kv_encode_fp8r_row(row.data(), h_comp.data() + (size_t)r * aff::kFlashRowBytes);
    }

    const uint32_t split_slots = 16u;
    const size_t split_o_bytes = (size_t)split_slots * c.n_tok * n_head * width * sizeof(float);
    const size_t split_ml_bytes = (size_t)split_slots * c.n_tok * n_head * 2u * sizeof(float);
    float *d_q = nullptr, *d_sinks = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
    float *d_po = nullptr, *d_pml = nullptr;
    uint8_t *d_raw = nullptr, *d_comp = nullptr;
    assert(hipMalloc(&d_q, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_sinks, h_sinks.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_raw, h_raw.size()) == hipSuccess);
    assert(hipMalloc(&d_comp, h_comp.size()) == hipSuccess);
    assert(hipMalloc(&d_o0, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_o1, h_q.size() * sizeof(float)) == hipSuccess);
    assert(hipMalloc(&d_po, split_o_bytes) == hipSuccess);
    assert(hipMalloc(&d_pml, split_ml_bytes) == hipSuccess);
    assert(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess);
    assert(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * sizeof(float),
                     hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_raw, h_raw.data(), h_raw.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_comp, h_comp.data(), h_comp.size(), hipMemcpyHostToDevice) == hipSuccess);

    aff::FlashSplit fs{};
    fs.o = d_po;
    fs.ml = d_pml;
    fs.tok_cap = c.n_tok;
    fs.slots = split_slots;
    const uint32_t pos0 = c.n_raw - 1u;

    ::setenv("AFF_FLASH_LOWM_NOSPLIT", "0", 1);
    ::setenv("AFF_FLASH_SPLIT_V2", "1", 1);
    ::setenv("AFF_FLASH_MIN_TILES", "1", 1);
    aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, nullptr, nullptr, c.n_comp, d_sinks, d_o0, 0u,
                              c.n_tok, c.n_tok, pos0, sliding, ring, c.n_comp, n_head, scale,
                              nullptr, false, &fs);
    // Poison split scratch so AFF_FLASH_LOWM_NOSPLIT=1 proves no split-K touch.
    assert(hipMemset(d_po, 0x7f, split_o_bytes) == hipSuccess);
    ::setenv("AFF_FLASH_LOWM_NOSPLIT", "1", 1);
    aff::mqa_attend_flash_hip(d_q, d_raw, d_comp, nullptr, nullptr, c.n_comp, d_sinks, d_o1, 0u,
                              c.n_tok, c.n_tok, pos0, sliding, ring, c.n_comp, n_head, scale,
                              nullptr, false, &fs);
    ::unsetenv("AFF_FLASH_LOWM_NOSPLIT");
    ::unsetenv("AFF_FLASH_SPLIT_V2");
    ::unsetenv("AFF_FLASH_MIN_TILES");
    assert(hipDeviceSynchronize() == hipSuccess);

    std::vector<float> h0(h_q.size()), h1(h_q.size());
    std::vector<uint8_t> po_check(64u);
    assert(hipMemcpy(po_check.data(), d_po, po_check.size(), hipMemcpyDeviceToHost) == hipSuccess);
    for (uint8_t b : po_check) assert(b == 0x7fu);
    assert(hipMemcpy(h0.data(), d_o0, h0.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(hipMemcpy(h1.data(), d_o1, h1.size() * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess);
    float max_diff = 0.0f;
    for (size_t i = 0; i < h0.size(); ++i) {
      assert(std::isfinite(h1[i]));
      max_diff = std::max(max_diff, std::fabs(h0[i] - h1[i]));
    }
    assert(max_diff <= 2e-3f);
    worst_diff = std::max(worst_diff, max_diff);
    for (void* p : {(void*)d_q, (void*)d_sinks, (void*)d_raw, (void*)d_comp, (void*)d_o0,
                    (void*)d_o1, (void*)d_po, (void*)d_pml})
      (void)hipFree(p);
  }
  std::printf("PASS: AFF_FLASH_LOWM_NOSPLIT parity vs split (max_diff=%.3e <= 2e-3f, scratch untouched)\n",
              worst_diff);
  return 0;
}

int check_hip_graph_capture_replay_parity() {
  const uint32_t dim = 256u, nb = 16u;
  const size_t n = (size_t)dim * nb;
  std::vector<float> h_x(n), h_blk(n), h_w(dim, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    h_x[i] = 0.05f * (float)((int)(i % 29u) - 14);
    h_blk[i] = 0.01f * (float)((int)(i % 13u) - 6);
  }
  float *d_x0 = nullptr, *d_x1 = nullptr, *d_b0 = nullptr, *d_b1 = nullptr, *d_y0 = nullptr,
        *d_y1 = nullptr, *d_w = nullptr;
  assert(hipMalloc(&d_x0, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_x1, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_b0, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_b1, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_y0, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_y1, n * sizeof(float)) == hipSuccess);
  assert(hipMalloc(&d_w, dim * sizeof(float)) == hipSuccess);
  assert(hipMemcpy(d_x0, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_x1, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_b0, h_blk.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_b1, h_blk.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

  hipStream_t s0 = nullptr, s1 = nullptr;
  hipEvent_t ev_fork = nullptr, ev_join = nullptr;
  assert(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking) == hipSuccess);
  assert(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking) == hipSuccess);
  assert(hipEventCreateWithFlags(&ev_fork, hipEventDisableTiming) == hipSuccess);
  assert(hipEventCreateWithFlags(&ev_join, hipEventDisableTiming) == hipSuccess);

  auto launch_seq = [&](float* dx, float* db, float* dy) {
    aff::batch_rms_norm_hip(dx, d_w, nullptr, dy, dim, nb, 1e-5f, s0);
    assert(hipEventRecord(ev_fork, s0) == hipSuccess);
    assert(hipStreamWaitEvent(s1, ev_fork, 0) == hipSuccess);
    aff::batch_hc_add_hip(db, dy, n, s1);
    assert(hipEventRecord(ev_join, s1) == hipSuccess);
    assert(hipStreamWaitEvent(s0, ev_join, 0) == hipSuccess);
    aff::batch_rms_norm_hip(db, d_w, nullptr, dx, dim, nb, 1e-5f, s0);
  };

  for (int it = 0; it < 3; ++it) launch_seq(d_x0, d_b0, d_y0);

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  assert(hipStreamBeginCapture(s0, hipStreamCaptureModeRelaxed) == hipSuccess);
  launch_seq(d_x1, d_b1, d_y1);
  assert(hipStreamEndCapture(s0, &graph) == hipSuccess);
  assert(graph != nullptr);
  assert(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess);
  assert(hipGraphDestroy(graph) == hipSuccess);
  for (int it = 0; it < 3; ++it) assert(hipGraphLaunch(exec, s0) == hipSuccess);
  assert(hipStreamSynchronize(s0) == hipSuccess);
  assert(hipGraphExecDestroy(exec) == hipSuccess);

  std::vector<float> r0(n), r1(n);
  assert(hipMemcpy(r0.data(), d_x0, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  assert(hipMemcpy(r1.data(), d_x1, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  assert(std::memcmp(r0.data(), r1.data(), n * sizeof(float)) == 0);
  (void)hipEventDestroy(ev_fork);
  (void)hipEventDestroy(ev_join);
  (void)hipStreamDestroy(s0);
  (void)hipStreamDestroy(s1);
  for (void* p : {(void*)d_x0, (void*)d_x1, (void*)d_b0, (void*)d_b1, (void*)d_y0, (void*)d_y1,
                  (void*)d_w})
    (void)hipFree(p);
  std::printf("PASS: HIP Graph capture + multi-replay fork/join parity (bit-exact)\n");
  return 0;
}

int check_vram_loan_ranges_and_reclaim_invariants() {
  aff::DenseGpu dg;
  std::string err;
  const uint32_t cap = 2048u, n_embd = 2048u, n_hc = 4u, n_head = 32u, width = 512u,
                 max_mid = 4096u, vocab = 1024u, idx_topk = 64u;
  assert(dg.init(max_mid, n_embd, &err, 1u));
  assert(dg.batch_init(cap, n_embd, n_hc, n_head, width, max_mid, vocab, idx_topk, &err));

  ::setenv("AFF_VRAM_LOAN", "0", 1);
  std::vector<aff::DenseGpu::VramLoanRange> ranges;
  dg.vram_loan_ranges(0u, &ranges);
  assert(ranges.empty());

  ::setenv("AFF_VRAM_LOAN", "1", 1);
  dg.vram_loan_ranges(0u, &ranges);
  assert(!ranges.empty());
  for (size_t i = 0; i < ranges.size(); ++i) {
    assert(ranges[i].ptr != nullptr);
    assert(ranges[i].bytes > 0u);
    assert((reinterpret_cast<uintptr_t>(ranges[i].ptr) & 255u) == 0u);
    const uintptr_t ia = reinterpret_cast<uintptr_t>(ranges[i].ptr);
    const uintptr_t ib = ia + ranges[i].bytes;
    for (size_t j = i + 1u; j < ranges.size(); ++j) {
      const uintptr_t ja = reinterpret_cast<uintptr_t>(ranges[j].ptr);
      const uintptr_t jb = ja + ranges[j].bytes;
      assert(!(ia < jb && ib > ja));
    }
  }

  struct CbState { int pre = 0, post = 0; } cbs{};
  dg.set_prefill_reclaim_cb(
      [](void* ctx) { ++static_cast<CbState*>(ctx)->pre; },
      [](void* ctx) { ++static_cast<CbState*>(ctx)->post; }, &cbs);

  const auto bops = dg.batch_ops();
  std::vector<float> emb_narrow((size_t)5 * n_embd, 0.1f);
  assert(bops.begin(bops.ctx, 5u, 0u, emb_narrow.data(), n_embd, n_hc));
  assert(cbs.pre == 0 && cbs.post == 1);

  std::vector<float> emb_wide((size_t)256u * n_embd, 0.1f);
  assert(bops.begin(bops.ctx, 256u, 5u, emb_wide.data(), n_embd, n_hc));
  assert(cbs.pre == 1 && cbs.post == 1);

  aff::StaticPlacement sp;
  assert(sp.loan_slot_count(0u) == 0u);
  assert(sp.loan_active_count(0u) == 0u);
  assert(sp.prefill_stage_slot_count() == 0u);
  assert(!sp.vram_loan_eligible());
  sp.mark_vram_loan_eligible(true);
  assert(!sp.vram_loan_eligible());
  sp.reclaim_vram_loan();
  assert(!sp.vram_loan_eligible());

  // Verify cap=4096 external arenas (`ext_bydn` / `ext_ba8`) are completely disjoint from
  // DenseGpu::vram_loan_ranges (including `[u_res, u_res + 4 MiB)` decode tail on `b_qtm`).
  {
    aff::DenseGpu dg4k;
    std::string err4k;
    assert(dg4k.init(max_mid, n_embd, &err4k, 1u));
    assert(dg4k.batch_init(4096u, n_embd, n_hc, n_head, width, max_mid, vocab, idx_topk, &err4k));
    void *bydn = nullptr, *ba8 = nullptr;
    size_t bydn_b = 0, ba8_b = 0;
    dg4k.moe_ext_arenas(0u, &bydn, &bydn_b, &ba8, &ba8_b);
    assert(bydn != nullptr && bydn_b > 0u);
    assert(ba8 != nullptr && ba8_b > 0u);
    std::vector<aff::DenseGpu::VramLoanRange> r4k;
    dg4k.vram_loan_ranges(0u, &r4k);
    assert(!r4k.empty());
    const uintptr_t bydn_a = reinterpret_cast<uintptr_t>(bydn);
    const uintptr_t bydn_e = bydn_a + std::min<size_t>(bydn_b, 8u << 20);
    const uintptr_t ba8_a = reinterpret_cast<uintptr_t>(ba8);
    const uintptr_t ba8_e = ba8_a + std::min<size_t>(ba8_b, 4u << 20);
    for (const auto& lr : r4k) {
      const uintptr_t la = reinterpret_cast<uintptr_t>(lr.ptr);
      const uintptr_t lb = la + lr.bytes;
      assert(!(bydn_a < lb && bydn_e > la));
      assert(!(ba8_a < lb && ba8_e > la));
    }
  }

  ::unsetenv("AFF_VRAM_LOAN");
  std::printf("PASS: DenseGpu::vram_loan_ranges + reclaim callback + StaticPlacement invariants\n");
  return 0;
}

int check_pinchunk_boundary_memcpy_parity() {
  const auto devs = aff::gfx12_devices();
  assert(!devs.empty());
  assert(hipSetDevice(devs[0]) == hipSuccess);

  constexpr size_t kHalf = 1u << 20;
  constexpr size_t kSpan = 512u << 10;
  constexpr size_t kTestBytes = 2u * kSpan;
  const size_t map_len = aff::kHipPinChunkBytes + (2u << 20);
  void* map = ::mmap(nullptr, map_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(map != MAP_FAILED);

  const uintptr_t map_lo = reinterpret_cast<uintptr_t>(map);
  const uintptr_t bnd = (map_lo + kHalf + aff::kHipPinChunkBytes - 1u) &
                        ~(aff::kHipPinChunkBytes - 1u);
  assert(bnd >= map_lo + kHalf);
  assert(bnd + kHalf <= map_lo + map_len);

  uint8_t* const lo_chunk = reinterpret_cast<uint8_t*>(bnd - kHalf);
  uint8_t* const hi_chunk = reinterpret_cast<uint8_t*>(bnd);
  assert(::mprotect(lo_chunk, 2u * kHalf, PROT_READ | PROT_WRITE) == 0);

  uint8_t* const host_span = reinterpret_cast<uint8_t*>(bnd - kSpan);
  std::vector<uint8_t> ref(kTestBytes);
  for (size_t i = 0; i < kTestBytes; ++i) {
    const uint8_t v = (uint8_t)((i * 131u + 17u) ^ (i >> 8));
    ref[i] = v;
    host_span[i] = v;
  }

  const hipError_t reg_lo = hipHostRegister(lo_chunk, kHalf, hipHostRegisterDefault);
  const hipError_t reg_hi = hipHostRegister(hi_chunk, kHalf, hipHostRegisterDefault);
  if (reg_lo != hipSuccess || reg_hi != hipSuccess) {
    if (reg_hi == hipSuccess) (void)hipHostUnregister(hi_chunk);
    if (reg_lo == hipSuccess) (void)hipHostUnregister(lo_chunk);
    (void)::munmap(map, map_len);
    std::fprintf(stderr, "FAIL: hipHostRegister failed reg_lo=%d reg_hi=%d\n", (int)reg_lo,
                 (int)reg_hi);
    return 1;
  }

  uint8_t* d_buf = nullptr;
  hipStream_t st = nullptr;
  assert(hipMalloc(&d_buf, kTestBytes) == hipSuccess);
  assert(hipStreamCreateWithFlags(&st, hipStreamNonBlocking) == hipSuccess);

  const hipError_t e_h2d = aff::hip_memcpy_h2d_pinchunk_async(d_buf, host_span, kTestBytes, st);
  assert(e_h2d == hipSuccess);
  assert(hipStreamSynchronize(st) == hipSuccess);

  std::memset(host_span, 0, kTestBytes);
  const hipError_t e_d2h = aff::hip_memcpy_d2h_pinchunk_async(host_span, d_buf, kTestBytes, st);
  assert(e_d2h == hipSuccess);
  assert(hipStreamSynchronize(st) == hipSuccess);

  const bool match = (std::memcmp(host_span, ref.data(), kTestBytes) == 0);
  (void)hipStreamDestroy(st);
  (void)hipFree(d_buf);
  (void)hipHostUnregister(hi_chunk);
  (void)hipHostUnregister(lo_chunk);
  (void)::munmap(map, map_len);
  assert(hipGetLastError() == hipSuccess);
  assert(match);
  std::printf("PASS: 1 GiB piecewise hipHostRegister boundary H2D/D2H round-trip parity\n");
  return 0;
}

int check_draft_graph_per_stage_warmup_and_abort() {
  ::setenv("AFF_DSPARK_HIP_GRAPH", "1", 1);
  aff::DenseGpu dg;
  std::string err;
  const uint32_t cap = 64u, n_embd = 256u, n_hc = 4u, n_head = 16u, width = 192u,
                 max_mid = 256u, vocab = 128u, idx_topk = 16u;
  assert(dg.init(max_mid, n_embd, &err, 1u));
  assert(dg.batch_init(cap, n_embd, n_hc, n_head, width, max_mid, vocab, idx_topk, &err));

  const auto bops = dg.batch_ops();
  std::vector<float> emb((size_t)5 * n_embd, 0.25f);
  assert(bops.begin(bops.ctx, 5u, 0u, emb.data(), n_embd, n_hc));

  // Warming stage 2 must not cause stage 0 to enter capture on its first dev_plan pass.
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 2u, true));
  assert(bops.draft_graph_end(bops.ctx, 5u, 2u, true, true));

  // First stage 0 pass: must warm cleanly without capture (`draft_graph_end` returns true).
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 0u, false));
  assert(bops.draft_graph_end(bops.ctx, 5u, 0u, false, true));

  // Second stage 0 pass: begins capture; body_ok=false must abort capture, return false,
  // mark failed, and leave future stage 0 calls falling back uncaptured cleanly.
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 0u, false));
  assert(!bops.draft_graph_end(bops.ctx, 5u, 0u, false, false));
  assert(hipGetLastError() == hipSuccess);

  assert(!bops.draft_graph_begin(bops.ctx, 5u, 0u, false));
  assert(bops.draft_graph_end(bops.ctx, 5u, 0u, false, true));

  // Verify split-K workspace growth bumps batch_splitk_epoch_hip and invalidates cached graphs.
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 1u, false));
  assert(bops.draft_graph_end(bops.ctx, 5u, 1u, false, true));

  void* d_dummy = nullptr;
  assert(hipMalloc(&d_dummy, 256u) == hipSuccess);
  hipStream_t st = static_cast<hipStream_t>(dg.stream(0u));
  assert(st != nullptr);
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 1u, false));
  assert(hipMemsetAsync(d_dummy, 1, 256u, st) == hipSuccess);
  assert(bops.draft_graph_end(bops.ctx, 5u, 1u, false, true));
  assert(bops.draft_graph_begin(bops.ctx, 5u, 1u, false));

  const uint64_t ep_before = aff::batch_splitk_epoch_hip();
  assert(aff::batch_splitk_reserve_hip(2u << 20, st));
  assert(aff::batch_splitk_epoch_hip() > ep_before);
  assert(!bops.draft_graph_begin(bops.ctx, 5u, 1u, false));
  assert(bops.draft_graph_end(bops.ctx, 5u, 1u, false, true));
  assert(hipStreamSynchronize(st) == hipSuccess);
  (void)hipFree(d_dummy);

  ::unsetenv("AFF_DSPARK_HIP_GRAPH");
  std::printf("PASS: DenseGpu DSpark per-stage HIP Graph warmup + abort + split-K invalidation\n");
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
  if (check_bf16x2_rne_parity() != 0) return 1;
  if (check_router_topk_v2_parity() != 0) return 1;
  if (check_mhc_mix_fast_parity() != 0) return 1;
  if (check_flash_exp2_klate_parity() != 0) return 1;
  if (check_expert_had32_dpp_parity() != 0) return 1;
  if (check_moe_plan_dpp_parity() != 0) return 1;
  if (check_flash_flat_win_parity() != 0) return 1;
  if (check_flash_lowm_nosplit_parity() != 0) return 1;
  if (check_hip_graph_capture_replay_parity() != 0) return 1;
  if (check_vram_loan_ranges_and_reclaim_invariants() != 0) return 1;
  if (check_pinchunk_boundary_memcpy_parity() != 0) return 1;
  if (check_draft_graph_per_stage_warmup_and_abort() != 0) return 1;
  std::printf("PASS: test_v512_kernels\n");
  return 0;
}
