#include "gpu/engram_gpu.h"
#include "gpu/device_math.h"
#include "gpu/hip_common.h"
#include "gpu/wmma.h"

#include <hip/hip_runtime.h>

namespace aff {
namespace {

// The project's own bf16 conversion, not the HIP intrinsic: `b_hc` is written by batch_hc_seed and
// by hc_post through `wmma::f32_to_bf16`, which rounds to nearest even, and a second writer that
// truncated instead would bias this one contribution toward zero against every other.
__device__ __forceinline__ float ld_hc(const void* p, size_t i, bool bf16) {
  return bf16 ? wmma::bf16_to_f32(((const uint16_t*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ void st_hc(void* p, size_t i, float v, bool bf16) {
  if (bf16) ((uint16_t*)p)[i] = wmma::f32_to_bf16(v);
  else      ((float*)p)[i] = v;
}

constexpr uint32_t kEngThreads = 256u;
constexpr uint32_t kEngWaves = kEngThreads / 32u;
constexpr uint32_t kEngMaxHc = 8u;

// 256-thread-per-token kernel for narrow batches (nb <= 64: decode & DSpark verify), eliminating
// the 1-6 thread/WG stall while loading the shared `kv` value vector once across all `n_hc` copies.
template <bool BF16>
__global__ __launch_bounds__(kEngThreads) void engram_gate_tok_kernel(
    void* __restrict__ hc, const float* __restrict__ kv, const float* __restrict__ w,
    uint32_t n_embd, uint32_t n_hc, uint32_t nb, uint32_t hs, float eps) {
  const uint32_t t = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wave = tid >> 5;
  if (t >= nb || n_hc > kEngMaxHc) return;

  __shared__ float s_sh[kEngMaxHc][kEngWaves];
  __shared__ float s_sk[kEngMaxHc][kEngWaves];
  __shared__ float s_dot[kEngMaxHc][kEngWaves];
  __shared__ float s_gate[kEngMaxHc];

  const size_t vband = (size_t)n_hc * n_embd;

  for (uint32_t c = 0; c < n_hc; ++c) {
    const size_t band = (size_t)c * n_embd;
    float sh = 0.0f, sk = 0.0f, dot = 0.0f;
    for (uint32_t i = tid; i < n_embd; i += kEngThreads) {
      const float h = ld_hc(hc, (band + i) * hs + t, BF16);
      const float k = kv[(band + i) * nb + t];
      sh = fmaf(h, h, sh);
      sk = fmaf(k, k, sk);
      dot = fmaf(h * w[band + i], k, dot);
    }
    sh = aff::wave_reduce_add_dpp(sh);
    sk = aff::wave_reduce_add_dpp(sk);
    dot = aff::wave_reduce_add_dpp(dot);
    if (lane == 0u) {
      s_sh[c][wave] = sh;
      s_sk[c][wave] = sk;
      s_dot[c][wave] = dot;
    }
  }
  aff::lds_barrier();

  if (tid < n_hc) {
    float fsh = 0.0f, fsk = 0.0f, fdot = 0.0f;
#pragma unroll
    for (uint32_t wp = 0; wp < kEngWaves; ++wp) {
      fsh += s_sh[tid][wp];
      fsk += s_sk[tid][wp];
      fdot += s_dot[tid][wp];
    }
    const float rstd = rsqrtf(fsh / (float)n_embd + eps) * rsqrtf(fsk / (float)n_embd + eps);
    const float d = fdot * rstd * rsqrtf((float)n_embd);
    const float s = copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d);
    s_gate[tid] = 1.0f / (1.0f + __expf(-s));
  }
  aff::lds_barrier();

  for (uint32_t i = tid; i < n_embd; i += kEngThreads) {
    const float val = kv[(vband + i) * nb + t];
    for (uint32_t c = 0; c < n_hc; ++c) {
      const size_t off = ((size_t)c * n_embd + i) * hs + t;
      st_hc(hc, off, ld_hc(hc, off, BF16) + s_gate[c] * val, BF16);
    }
  }
}

// One block a (hc copy, token tile); one thread a token. BF16 is a template parameter so the branch
// is resolved at compile time rather than inside two loops over `dim`.
template <bool BF16>
__global__ void engram_gate_kernel(void* __restrict__ hc, const float* __restrict__ kv,
                                   const float* __restrict__ w, uint32_t n_embd, uint32_t n_hc,
                                   uint32_t nb, uint32_t hs, float eps) {
  const uint32_t c = blockIdx.x;
  const uint32_t t = blockIdx.y * blockDim.x + threadIdx.x;
  if (t >= nb || c >= n_hc) return;

  const size_t band = (size_t)c * n_embd;
  const size_t vband = (size_t)n_hc * n_embd;          // the shared value's band in `kv`

  // Pass one: the two RMS terms and the weighted dot, in f32 whatever the state is stored as —
  // the reference does this in float and casts only on the way back.
  float sh = 0.0f, sk = 0.0f, dot = 0.0f;
  for (uint32_t i = 0; i < n_embd; ++i) {
    const float h = ld_hc(hc, (band + i) * hs + t, BF16);
    const float k = kv[(band + i) * nb + t];
    sh = fmaf(h, h, sh);
    sk = fmaf(k, k, sk);
    dot = fmaf(h * w[band + i], k, dot);
  }
  const float rstd = rsqrtf(sh / (float)n_embd + eps) * rsqrtf(sk / (float)n_embd + eps);
  const float d = dot * rstd * rsqrtf((float)n_embd);
  // Signed sqrt before the sigmoid, matching the training kernel: the magnitude is clamped away
  // from zero first, then the sign is put back.
  const float s = copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d);
  const float gate = 1.0f / (1.0f + __expf(-s));

  // Pass two: the write. ONE value for every copy, each scaled by its own gate.
  for (uint32_t i = 0; i < n_embd; ++i) {
    const size_t off = (band + i) * hs + t;
    st_hc(hc, off, ld_hc(hc, off, BF16) + gate * kv[(vband + i) * nb + t], BF16);
  }
}

__global__ __launch_bounds__(32) void engram_mxfp8_requant_fp8_kernel(
    const uint8_t* __restrict__ raw_mxfp8, uint8_t* __restrict__ out_q8,
    float* __restrict__ out_scales, uint32_t nkb, uint32_t in_dim, uint32_t raw_stride) {
  const uint32_t kb = blockIdx.x;
  const uint32_t tok = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  const uint32_t sb = lane >> 3;

  const uint8_t* const tok_base = raw_mxfp8 + (size_t)tok * raw_stride;
  const uint8_t* const w_ptr = tok_base + (size_t)kb * 128u + lane * 4u;
  const uint8_t* const s_ptr = tok_base + in_dim + (size_t)kb * 4u;

  uint32_t w4;
  uint32_t s4;
  if ((reinterpret_cast<uintptr_t>(raw_mxfp8) & 3u) == 0u) {
    w4 = *reinterpret_cast<const uint32_t*>(w_ptr);
    s4 = *reinterpret_cast<const uint32_t*>(s_ptr);
  } else {
    w4 = (uint32_t)w_ptr[0] | ((uint32_t)w_ptr[1] << 8) |
         ((uint32_t)w_ptr[2] << 16) | ((uint32_t)w_ptr[3] << 24);
    s4 = (uint32_t)s_ptr[0] | ((uint32_t)s_ptr[1] << 8) |
         ((uint32_t)s_ptr[2] << 16) | ((uint32_t)s_ptr[3] << 24);
  }

  const uint32_t e = (s4 >> (sb * 8u)) & 0xFFu;
  // Match engram.cpp:247 E8M0 decode bit-for-bit: 0xFF -> 0.0f, 0 -> 2^-127 subnormal, else 2^(e-127).
  const float s = (e == 0xFFu) ? 0.0f
                : (e == 0u)    ? __uint_as_float(0x00400000u)
                               : __uint_as_float(e << 23);

  const aff_f32x2 d01 = e4m3_decode_pk<false>(w4);
  const aff_f32x2 d23 = e4m3_decode_pk<true>(w4);
  const float v0 = d01.x * s;
  const float v1 = d01.y * s;
  const float v2 = d23.x * s;
  const float v3 = d23.y * s;

  float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  amax = aff::wave_reduce_max_dpp(amax);
  const float sc128 = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
  if (lane == 0u) out_scales[(size_t)tok * nkb + kb] = sc128;

  // Match batch_quant_tm_fp8_kernel: multiply by `1.0f / sc128` rather than dividing by `sc128`.
  const float inv = 1.0f / sc128;
  const float c0 = fminf(448.0f, fmaxf(-448.0f, v0 * inv));
  const float c1 = fminf(448.0f, fmaxf(-448.0f, v1 * inv));
  const float c2 = fminf(448.0f, fmaxf(-448.0f, v2 * inv));
  const float c3 = fminf(448.0f, fmaxf(-448.0f, v3 * inv));
  int pk = 0;
  pk = __builtin_amdgcn_cvt_pk_fp8_f32(c0, c1, pk, false);
  pk = __builtin_amdgcn_cvt_pk_fp8_f32(c2, c3, pk, true);

  const size_t out_off = (size_t)tok * in_dim + (size_t)kb * 128u;
  if ((reinterpret_cast<uintptr_t>(out_q8) & 3u) == 0u) {
    reinterpret_cast<uint32_t*>(out_q8 + out_off)[lane] = (uint32_t)pk;
  } else {
    uint8_t* dst = out_q8 + out_off + lane * 4u;
    dst[0] = (uint8_t)((uint32_t)pk & 0xFFu);
    dst[1] = (uint8_t)(((uint32_t)pk >> 8) & 0xFFu);
    dst[2] = (uint8_t)(((uint32_t)pk >> 16) & 0xFFu);
    dst[3] = (uint8_t)(((uint32_t)pk >> 24) & 0xFFu);
  }
}

}  // namespace

void engram_gate_hip(void* hc, bool hc_bf16, const float* kv, const float* w, uint32_t n_embd,
                     uint32_t n_hc, uint32_t nb, uint32_t hc_stride, float eps, void* stream) {
  if (!hc || !kv || !w || !n_embd || !n_hc || !nb || hc_stride < nb) return;
  if (nb <= 64u && n_hc <= kEngMaxHc) {
    const dim3 g(nb), b(kEngThreads);
    if (hc_bf16)
      hipLaunchKernelGGL(engram_gate_tok_kernel<true>, g, b, 0, (hipStream_t)stream, hc, kv, w,
                         n_embd, n_hc, nb, hc_stride, eps);
    else
      hipLaunchKernelGGL(engram_gate_tok_kernel<false>, g, b, 0, (hipStream_t)stream, hc, kv, w,
                         n_embd, n_hc, nb, hc_stride, eps);
    AFF_HIP_CHECK(hipGetLastError());
    return;
  }
  const uint32_t thr = nb < 256u ? nb : 256u;
  const dim3 g(n_hc, (nb + thr - 1u) / thr), b(thr);
  if (hc_bf16)
    hipLaunchKernelGGL(engram_gate_kernel<true>, g, b, 0, (hipStream_t)stream, hc, kv, w, n_embd,
                       n_hc, nb, hc_stride, eps);
  else
    hipLaunchKernelGGL(engram_gate_kernel<false>, g, b, 0, (hipStream_t)stream, hc, kv, w, n_embd,
                       n_hc, nb, hc_stride, eps);
  AFF_HIP_CHECK(hipGetLastError());
}

bool engram_mxfp8_requant_fp8_hip(const uint8_t* raw_mxfp8, uint8_t* out_q8, float* out_scales,
                                  uint32_t nt, uint32_t in_dim, hipStream_t stream) {
  if ((in_dim % 128u) != 0u) return false;
  if (nt == 0u) return true;
  if (in_dim == 0u || !raw_mxfp8 || !out_q8 || !out_scales) return false;
  const uint32_t nkb = in_dim >> 7;
  const uint32_t raw_stride = in_dim + (in_dim >> 5);
  hipLaunchKernelGGL(engram_mxfp8_requant_fp8_kernel, dim3(nkb, nt), dim3(32), 0, stream,
                     raw_mxfp8, out_q8, out_scales, nkb, in_dim, raw_stride);
  AFF_HIP_CHECK(hipGetLastError());
  return true;
}

}  // namespace aff
