#include "expert_kernel.h"
#include "engine/cpu_dispatch.h"
#include "engine/threadpool.h"
#include "quant/source_dtypes.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#if AFF_CPU_X86
#include <immintrin.h>
#endif

namespace aff {

// CPU kernel variant. 1 is the scalar reference; 2 adds two accumulators and one field load per
// block pair (default on AVX1/AVX2/AVX-512 hosts without VBMI, e.g. Skylake-SP); 4 is the default
// on VBMI hosts and cuts instructions with AVX512_VBMI, bit-identical to v2. 3 and 5
// are intermediate points kept as controls rather than as attributions — both are unstable against
// code layout, so the only claim this file makes is v2 -> v4. All are retained so the set stays
// comparable after a change to the surrounding code (`aff-kbench`).
//
// The kernel is COMPUTE-bound, not memory-bound (see the header), which is why every variant below
// chases operation count and nothing else.
//
// -1 auto-selects 4 on VBMI hosts and 2 otherwise; `cpu_kernel_variant(v)` overrides explicitly.
static int g_cpu_variant = -1;
void cpu_kernel_variant(int v) { g_cpu_variant = v; }
int  cpu_kernel_variant() {
  if (g_cpu_variant >= 0) return g_cpu_variant;
  return (active_cpu_isa() >= CpuIsa::Avx512Vbmi) ? 4 : 2;
}

namespace {

// Codes are 5 bits and therefore not byte-aligned. Rather than a bit-extract per pair, we walk a
// running 64-bit window: for block=16 there are 8 pairs = 40 bits, so one 64-bit load per block
// covers it with room to spare. This keeps the inner loop free of shifts across byte boundaries.
inline uint64_t load_bits(const uint8_t* p, uint64_t bitpos) noexcept {
  uint64_t v;
  std::memcpy(&v, p + (bitpos >> 3), sizeof(v));
  return v >> (bitpos & 7);
}

inline float rscale_of(const uint8_t* rscales, uint64_t row, bool f32) noexcept {
  if (f32) { float v; std::memcpy(&v, rscales + row * 4, 4); return v; }
  uint16_t h; std::memcpy(&h, rscales + row * 2, 2);
  return bf16_to_float(h);
}

#if AFF_CPU_X86
AFF_TARGET_AVX1 void expert_gemv_run_avx1(
    const uint8_t* data, const uint8_t* scales, const uint8_t* rscales, uint64_t rows,
    uint64_t nblk, uint64_t codes_row, uint64_t fbits, uint32_t scale_bits, uint32_t smask,
    uint32_t fmask, bool row_scale_f32, const float (*Lp)[32], const float (*Hp)[32],
    const float* xe_p, const float* xo_p, float w, float* y) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const float rs = rscale_of(rscales, r, row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    uint64_t b = 0;
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(scales, fbase + b * fbits);
      const uint32_t f0 = (uint32_t)(fw & fmask);
      const uint32_t f1 = (uint32_t)((fw >> fbits) & fmask);
      const float bs0 = rs * (float)(f0 & smask);
      const float bs1 = rs * (float)(f1 & smask);
      if (bs0 != 0.0f) {
        const uint32_t va = f0 >> scale_bits;
        const uint32_t v0 = va < 4u ? va : 0u;
        const uint64_t w0 = load_bits(data, cbase + b * 40u);
        alignas(32) float vl[8], vh[8];
        const float* lp = Lp[v0];
        const float* hp = Hp[v0];
        for (int i = 0; i < 8; ++i) {
          const uint32_t c = (uint32_t)((w0 >> (5 * i)) & 31u);
          vl[i] = lp[c];
          vh[i] = hp[c];
        }
        const __m256 part = _mm256_add_ps(
            _mm256_mul_ps(_mm256_load_ps(vl), _mm256_loadu_ps(xe_p + b * 8u)),
            _mm256_mul_ps(_mm256_load_ps(vh), _mm256_loadu_ps(xo_p + b * 8u)));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_broadcast_ss(&bs0), part));
      }
      if (bs1 != 0.0f) {
        const uint32_t vb = f1 >> scale_bits;
        const uint32_t v1 = vb < 4u ? vb : 0u;
        const uint64_t w1 = load_bits(data, cbase + (b + 1u) * 40u);
        alignas(32) float vl[8], vh[8];
        const float* lp = Lp[v1];
        const float* hp = Hp[v1];
        for (int i = 0; i < 8; ++i) {
          const uint32_t c = (uint32_t)((w1 >> (5 * i)) & 31u);
          vl[i] = lp[c];
          vh[i] = hp[c];
        }
        const __m256 part = _mm256_add_ps(
            _mm256_mul_ps(_mm256_load_ps(vl), _mm256_loadu_ps(xe_p + (b + 1u) * 8u)),
            _mm256_mul_ps(_mm256_load_ps(vh), _mm256_loadu_ps(xo_p + (b + 1u) * 8u)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_broadcast_ss(&bs1), part));
      }
    }
    const __m256 sum = _mm256_add_ps(acc0, acc1);
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    y[r] += _mm_cvtss_f32(h1);
  }
}

AFF_TARGET_AVX2 void expert_gemv_run_avx2(
    const uint8_t* data, const uint8_t* scales, const uint8_t* rscales, uint64_t rows,
    uint64_t nblk, uint64_t codes_row, uint64_t fbits, uint32_t scale_bits, uint32_t smask,
    uint32_t fmask, bool row_scale_f32, const float (*Lp)[32], const float (*Hp)[32],
    const float* xe_p, const float* xo_p, float w, float* y) noexcept {
  const __m128i sh4 = _mm_setr_epi32(0, 5, 10, 15);
  const __m128i m31 = _mm_set1_epi32(31);
  for (uint64_t r = 0; r < rows; ++r) {
    const float rs = rscale_of(rscales, r, row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    uint64_t b = 0;
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(scales, fbase + b * fbits);
      const uint32_t f0 = (uint32_t)(fw & fmask);
      const uint32_t f1 = (uint32_t)((fw >> fbits) & fmask);
      const float bs0 = rs * (float)(f0 & smask);
      const float bs1 = rs * (float)(f1 & smask);
      if (bs0 != 0.0f) {
        const uint32_t va = f0 >> scale_bits;
        const uint32_t v0 = va < 4u ? va : 0u;
        const uint64_t w0 = load_bits(data, cbase + b * 40u);
        const __m128i c_lo = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)(uint32_t)w0), sh4), m31);
        const __m128i c_hi =
            _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)(uint32_t)(w0 >> 20)), sh4), m31);
        const __m256i vidx = _mm256_inserti128_si256(_mm256_castsi128_si256(c_lo), c_hi, 1);
        const __m256 vlo = _mm256_i32gather_ps(Lp[v0], vidx, 4);
        const __m256 vhi = _mm256_i32gather_ps(Hp[v0], vidx, 4);
        const __m256 part = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xe_p + b * 8u),
                                            _mm256_mul_ps(vhi, _mm256_loadu_ps(xo_p + b * 8u)));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(bs0), part, acc0);
      }
      if (bs1 != 0.0f) {
        const uint32_t vb = f1 >> scale_bits;
        const uint32_t v1 = vb < 4u ? vb : 0u;
        const uint64_t w1 = load_bits(data, cbase + (b + 1u) * 40u);
        const __m128i c_lo = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)(uint32_t)w1), sh4), m31);
        const __m128i c_hi =
            _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int)(uint32_t)(w1 >> 20)), sh4), m31);
        const __m256i vidx = _mm256_inserti128_si256(_mm256_castsi128_si256(c_lo), c_hi, 1);
        const __m256 vlo = _mm256_i32gather_ps(Lp[v1], vidx, 4);
        const __m256 vhi = _mm256_i32gather_ps(Hp[v1], vidx, 4);
        const __m256 part = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(xe_p + (b + 1u) * 8u),
                                            _mm256_mul_ps(vhi, _mm256_loadu_ps(xo_p + (b + 1u) * 8u)));
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(bs1), part, acc1);
      }
    }
    const __m256 sum = _mm256_add_ps(acc0, acc1);
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    y[r] += _mm_cvtss_f32(h1);
  }
}

AFF_TARGET_AVX512 void expert_gemv_run_avx512(
    const uint8_t* data, const uint8_t* scales, const uint8_t* rscales, uint64_t rows,
    uint64_t nblk, uint64_t codes_row, uint64_t fbits, uint32_t scale_bits, uint32_t smask,
    uint32_t fmask, bool row_scale_f32, const float (*Lp)[32], const float (*Hp)[32],
    const float* xe_p, const float* xo_p, float w, float* y) noexcept {
  __m512 lo0[4], lo1[4], hi0[4], hi1[4];
  const __m512i vshift = _mm512_setr_epi64(0, 5, 10, 15, 20, 25, 30, 35);
  const __m512i vmask = _mm512_set1_epi64(31);
  for (uint32_t v = 0; v < 4u; ++v) {
    lo0[v] = _mm512_load_ps(Lp[v]);
    lo1[v] = _mm512_load_ps(Lp[v] + 16);
    hi0[v] = _mm512_load_ps(Hp[v]);
    hi1[v] = _mm512_load_ps(Hp[v] + 16);
  }
  const __mmask16 hiHalf = 0xFF00;
  for (uint64_t r = 0; r < rows; ++r) {
    const float rs = rscale_of(rscales, r, row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    uint64_t b = 0;
    for (; b + 3 < nblk; b += 4) {
      const uint64_t fw0 = load_bits(scales, fbase + b * fbits);
      const uint64_t fw1 = load_bits(scales, fbase + (b + 2) * fbits);
      const uint32_t f0 = (uint32_t)(fw0 & fmask), f1 = (uint32_t)((fw0 >> fbits) & fmask);
      const uint32_t f2 = (uint32_t)(fw1 & fmask), f3 = (uint32_t)((fw1 >> fbits) & fmask);
      const float bs0 = rs * (float)(f0 & smask), bs1 = rs * (float)(f1 & smask);
      const float bs2 = rs * (float)(f2 & smask), bs3 = rs * (float)(f3 & smask);
      if (bs0 != 0.0f || bs1 != 0.0f) {
        const uint32_t va = f0 >> scale_bits, vb = f1 >> scale_bits;
        const uint32_t vA = va < 4u ? va : 0u, vB = vb < 4u ? vb : 0u;
        const uint64_t w0 = load_bits(data, cbase + b * 8 * 5);
        const uint64_t w1 = load_bits(data, cbase + (b + 1) * 8 * 5);
        const __m256i c0 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
        const __m256i c1 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
        const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
        const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
        const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bs0), _mm512_set1_ps(bs1));
        acc0 = _mm512_fmadd_ps(vs,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + b * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + b * 8))), acc0);
      }
      if (bs2 != 0.0f || bs3 != 0.0f) {
        const uint32_t va = f2 >> scale_bits, vb = f3 >> scale_bits;
        const uint32_t vA = va < 4u ? va : 0u, vB = vb < 4u ? vb : 0u;
        const uint64_t w0 = load_bits(data, cbase + (b + 2) * 8 * 5);
        const uint64_t w1 = load_bits(data, cbase + (b + 3) * 8 * 5);
        const __m256i c0 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
        const __m256i c1 = _mm512_cvtepi64_epi32(
            _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
        const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
        const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
        const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bs2), _mm512_set1_ps(bs3));
        acc1 = _mm512_fmadd_ps(vs,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + (b + 2) * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + (b + 2) * 8))), acc1);
      }
    }
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(scales, fbase + b * fbits);
      const uint32_t fA = (uint32_t)(fw & fmask), fB = (uint32_t)((fw >> fbits) & fmask);
      const float bsA = rs * (float)(fA & smask), bsB = rs * (float)(fB & smask);
      if (bsA == 0.0f && bsB == 0.0f) continue;
      const uint32_t va = fA >> scale_bits, vb = fB >> scale_bits;
      const uint32_t vA = va < 4u ? va : 0u, vB = vb < 4u ? vb : 0u;
      const uint64_t w0 = load_bits(data, cbase + b * 8 * 5);
      const uint64_t w1 = load_bits(data, cbase + (b + 1) * 8 * 5);
      const __m256i c0 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w0), vshift), vmask));
      const __m256i c1 = _mm512_cvtepi64_epi32(
          _mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long)w1), vshift), vmask));
      const __m512i vidx = _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
      const __m512 vlo = _mm512_mask_blend_ps(hiHalf,
          _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
          _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
      const __m512 vhi = _mm512_mask_blend_ps(hiHalf,
          _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
          _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
      const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bsA), _mm512_set1_ps(bsB));
      acc0 = _mm512_fmadd_ps(vs, _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + b * 8),
                                                 _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + b * 8))),
                             acc0);
    }
    y[r] += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  }
}

AFF_TARGET_AVX512VBMI void expert_gemv_run_avx512vbmi(
    const uint8_t* data, const uint8_t* scales, const uint8_t* rscales, uint64_t rows,
    uint64_t nblk, uint64_t codes_row, uint64_t fbits, uint32_t scale_bits, uint32_t smask,
    uint32_t fmask, bool row_scale_f32, const float (*Lp)[32], const float (*Hp)[32],
    const float* xe_p, const float* xo_p, float w, float* y, int variant) noexcept {
  __m512 lo0[4], lo1[4], hi0[4], hi1[4];
  for (uint32_t v = 0; v < 4u; ++v) {
    lo0[v] = _mm512_load_ps(Lp[v]);
    lo1[v] = _mm512_load_ps(Lp[v] + 16);
    hi0[v] = _mm512_load_ps(Hp[v]);
    hi1[v] = _mm512_load_ps(Hp[v] + 16);
  }
  const __m512i kCodeCtl = _mm512_broadcast_i32x4(
      _mm_setr_epi8(0, 5, 10, 15, 20, 25, 30, 35, 0, 5, 10, 15, 20, 25, 30, 35));
  const __m512i kFieldCtl = _mm512_castsi128_si512(
      _mm_setr_epi8(0, 6, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  const __m512i kCode31 = _mm512_set1_epi32(31);
  const __m512i kFmask = _mm512_set1_epi32((int)fmask);
  const __m512i kSmask = _mm512_set1_epi32((int)smask);
  const __m512i kBcast01 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
  const __m512i kBcast23 = _mm512_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
  const __mmask16 hiHalf = 0xFF00;
  const bool v4 = (variant == 4);
  const bool v5 = (variant == 5);

  for (uint64_t r = 0; r < rows; ++r) {
    const float rs = rscale_of(rscales, r, row_scale_f32) * w;
    const uint64_t cbase = r * codes_row * 5;
    const uint64_t fbase = r * nblk * fbits;
    const uint8_t* cp = data + (cbase >> 3);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    uint64_t b = 0;
    for (; b + 3 < nblk; b += 4) {
      uint32_t vr[4];
      __m512 vs01, vs23;
      if (v4) {
        uint64_t fw;
        std::memcpy(&fw, scales + ((fbase + b * fbits) >> 3), 8);
        const __m512i f4 = _mm512_and_si512(
            _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(
                kFieldCtl, _mm512_castsi128_si512(_mm_cvtsi64_si128((long long)fw))))),
            kFmask);
        const __m512 bs4 = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_and_si512(f4, kSmask)), _mm512_set1_ps(rs));
        alignas(16) uint32_t vtmp[4];
        _mm_store_si128((__m128i*)vtmp,
                        _mm512_castsi512_si128(_mm512_srli_epi32(f4, (int)scale_bits)));
        vr[0] = vtmp[0] < 4u ? vtmp[0] : 0u;
        vr[1] = vtmp[1] < 4u ? vtmp[1] : 0u;
        vr[2] = vtmp[2] < 4u ? vtmp[2] : 0u;
        vr[3] = vtmp[3] < 4u ? vtmp[3] : 0u;
        vs01 = _mm512_permutexvar_ps(kBcast01, bs4);
        vs23 = _mm512_permutexvar_ps(kBcast23, bs4);
      } else {
        const uint64_t fw0 = load_bits(scales, fbase + b * fbits);
        const uint64_t fw1 = load_bits(scales, fbase + (b + 2) * fbits);
        const uint32_t f0 = (uint32_t)(fw0 & fmask), f1 = (uint32_t)((fw0 >> fbits) & fmask);
        const uint32_t f2 = (uint32_t)(fw1 & fmask), f3 = (uint32_t)((fw1 >> fbits) & fmask);
        const float b0 = rs * (float)(f0 & smask), b1 = rs * (float)(f1 & smask);
        const float b2 = rs * (float)(f2 & smask), b3 = rs * (float)(f3 & smask);
        vr[0] = (f0 >> scale_bits) < 4u ? (f0 >> scale_bits) : 0u;
        vr[1] = (f1 >> scale_bits) < 4u ? (f1 >> scale_bits) : 0u;
        vr[2] = (f2 >> scale_bits) < 4u ? (f2 >> scale_bits) : 0u;
        vr[3] = (f3 >> scale_bits) < 4u ? (f3 >> scale_bits) : 0u;
        if (b0 == 0.0f && b1 == 0.0f && b2 == 0.0f && b3 == 0.0f) continue;
        vs01 = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(b0), _mm512_set1_ps(b1));
        vs23 = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(b2), _mm512_set1_ps(b3));
      }
      {
        uint64_t cw0, cw1;
        std::memcpy(&cw0, cp + b * 5, 8);
        std::memcpy(&cw1, cp + b * 5 + 5, 8);
        const __m512i cd = _mm512_castsi128_si512(_mm_set_epi64x((long long)cw1, (long long)cw0));
        const __m512i vidx = _mm512_and_si512(
            _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(kCodeCtl, cd))),
            kCode31);
        __m512 vlo, vhi;
        if (v5 && vr[0] == vr[1]) {
          vlo = _mm512_permutex2var_ps(lo0[vr[0]], vidx, lo1[vr[0]]);
          vhi = _mm512_permutex2var_ps(hi0[vr[0]], vidx, hi1[vr[0]]);
        } else {
          vlo = _mm512_mask_blend_ps(hiHalf,
              _mm512_permutex2var_ps(lo0[vr[0]], vidx, lo1[vr[0]]),
              _mm512_permutex2var_ps(lo0[vr[1]], vidx, lo1[vr[1]]));
          vhi = _mm512_mask_blend_ps(hiHalf,
              _mm512_permutex2var_ps(hi0[vr[0]], vidx, hi1[vr[0]]),
              _mm512_permutex2var_ps(hi0[vr[1]], vidx, hi1[vr[1]]));
        }
        acc0 = _mm512_fmadd_ps(vs01,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + b * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + b * 8))), acc0);
      }
      {
        uint64_t cw0, cw1;
        std::memcpy(&cw0, cp + (b + 2) * 5, 8);
        std::memcpy(&cw1, cp + (b + 2) * 5 + 5, 8);
        const __m512i cd = _mm512_castsi128_si512(_mm_set_epi64x((long long)cw1, (long long)cw0));
        const __m512i vidx = _mm512_and_si512(
            _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(kCodeCtl, cd))),
            kCode31);
        __m512 vlo, vhi;
        if (v5 && vr[2] == vr[3]) {
          vlo = _mm512_permutex2var_ps(lo0[vr[2]], vidx, lo1[vr[2]]);
          vhi = _mm512_permutex2var_ps(hi0[vr[2]], vidx, hi1[vr[2]]);
        } else {
          vlo = _mm512_mask_blend_ps(hiHalf,
              _mm512_permutex2var_ps(lo0[vr[2]], vidx, lo1[vr[2]]),
              _mm512_permutex2var_ps(lo0[vr[3]], vidx, lo1[vr[3]]));
          vhi = _mm512_mask_blend_ps(hiHalf,
              _mm512_permutex2var_ps(hi0[vr[2]], vidx, hi1[vr[2]]),
              _mm512_permutex2var_ps(hi0[vr[3]], vidx, hi1[vr[3]]));
        }
        acc1 = _mm512_fmadd_ps(vs23,
            _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + (b + 2) * 8),
                            _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + (b + 2) * 8))), acc1);
      }
    }
    for (; b + 1 < nblk; b += 2) {
      const uint64_t fw = load_bits(scales, fbase + b * fbits);
      const uint32_t fA = (uint32_t)(fw & fmask), fB = (uint32_t)((fw >> fbits) & fmask);
      const float bsA = rs * (float)(fA & smask), bsB = rs * (float)(fB & smask);
      if (bsA == 0.0f && bsB == 0.0f) continue;
      const uint32_t va = fA >> scale_bits, vb = fB >> scale_bits;
      const uint32_t vA = va < 4u ? va : 0u, vB = vb < 4u ? vb : 0u;
      const __m512 vs = _mm512_mask_blend_ps(hiHalf, _mm512_set1_ps(bsA), _mm512_set1_ps(bsB));
      uint64_t cw0, cw1;
      std::memcpy(&cw0, cp + b * 5, 8);
      std::memcpy(&cw1, cp + b * 5 + 5, 8);
      const __m512i cd = _mm512_castsi128_si512(_mm_set_epi64x((long long)cw1, (long long)cw0));
      const __m512i vidx = _mm512_and_si512(
          _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_multishift_epi64_epi8(kCodeCtl, cd))),
          kCode31);
      __m512 vlo, vhi;
      if (v5 && vA == vB) {
        vlo = _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]);
        vhi = _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]);
      } else {
        vlo = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(lo0[vA], vidx, lo1[vA]),
            _mm512_permutex2var_ps(lo0[vB], vidx, lo1[vB]));
        vhi = _mm512_mask_blend_ps(hiHalf,
            _mm512_permutex2var_ps(hi0[vA], vidx, hi1[vA]),
            _mm512_permutex2var_ps(hi0[vB], vidx, hi1[vB]));
      }
      acc0 = _mm512_fmadd_ps(vs,
          _mm512_fmadd_ps(vlo, _mm512_loadu_ps(xe_p + b * 8),
                          _mm512_mul_ps(vhi, _mm512_loadu_ps(xo_p + b * 8))), acc0);
    }
    y[r] += _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  }
}
#endif
} // namespace

ExpertMatrixView view_of(const QuantizedMatrix& q) noexcept {
  ExpertMatrixView v;
  v.data = q.data.data();
  v.scales = q.scales.data();
  v.rscales = q.rscales.data();
  v.rows = q.rows; v.cols = q.cols; v.spec = q.spec;
  return v;
}

uint64_t expert_stream_bytes(const ExpertMatrixView& m) noexcept {
  const QuantSpec& s = m.spec;
  const uint64_t codes = m.rows * (m.cols / s.pair) * s.code_bits;
  const uint64_t blocks = m.rows * (m.cols / s.block) * s.block_field_bits();
  return (codes + 7) / 8 + (blocks + 7) / 8 + m.rows * (s.row_scale_f32 ? 4 : 2);
}

void expert_gemv_reference(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float* y) noexcept {
  const QuantSpec& s = m.spec;
  const uint64_t nblk = m.cols / s.block;
  const uint64_t ipb = s.block / s.pair;           // pairs per block
  const uint32_t smask = (1u << s.scale_bits) - 1u;
  const uint64_t codes_row = m.cols / s.pair;

  for (uint64_t r = 0; r < m.rows; ++r) {
    const float rs = rscale_of(m.rscales, r, s.row_scale_f32);
    double acc = 0.0;
    for (uint64_t b = 0; b < nblk; ++b) {
      const uint64_t fbit = (r * nblk + b) * s.block_field_bits();
      const uint32_t field = (uint32_t)(load_bits(m.scales, fbit) & ((1u << s.block_field_bits()) - 1u));
      // LSB-first: scale, sign, rotation, variant — the same order the GPU reads. Spelled out
      // rather than assumed because both the integer-scale pair format and the pow2 quad format
      // with its two symmetry bits reach here, and getting the shift wrong is not a crash: it
      // reads the rotation bit as the variant and returns numbers with no relationship to the
      // weights at all.
      const uint32_t g = (field >> s.scale_bits) & (s.transforms() - 1u);
      const uint32_t var = field >> (s.scale_bits + s.sign_bits + s.rot_bits);
      const float bs = rs * block_scale_from_code(field & smask, s) * ((g & 1u) ? -1.0f : 1.0f);
      const uint32_t rh = (g & 2u) ? s.pair / 2u : 0u;
      const float* lut = cb.entry(var < cb.variants ? var : 0, 0);
      for (uint64_t i = 0; i < ipb; ++i) {
        const uint64_t cbit = (r * codes_row + b * ipb + i) * s.code_bits;
        const uint32_t code = (uint32_t)(load_bits(m.data, cbit) & ((1u << s.code_bits) - 1u));
        const float* e = lut + (size_t)code * s.pair;
        const uint64_t c0 = b * s.block + i * s.pair;
        // The rotation is on the RECONSTRUCTION: position p takes entry (p + rh) % pair, the same
        // permutation the kernel gets from one funnel shift of the packed dword.
        for (uint32_t p = 0; p < s.pair; ++p)
          acc += (double)(bs * e[(p + rh) % s.pair]) * x[c0 + p];
      }
    }
    y[r] += (float)acc;
  }
}

void expert_gemv(const ExpertMatrixView& m, const Codebook& cb,
                 const float* x, float* y) noexcept {
  expert_gemv_scaled(m, cb, x, 1.0f, y);
}

ExpertMatrixView expert_row_slice(const ExpertMatrixView& m, uint64_t lo, uint64_t hi) noexcept {
  ExpertMatrixView s = m;
  s.rows = hi - lo;
  s.rscales = m.rscales + lo * (m.spec.row_scale_f32 ? 4 : 2);
  const uint64_t codes_row_bits = m.cols / m.spec.pair * m.spec.code_bits;
  const uint64_t field_row_bits = m.cols / m.spec.block * m.spec.block_field_bits();
  s.data    = m.data + lo * codes_row_bits / 8;
  s.scales  = m.scales + lo * field_row_bits / 8;
  return s;
}

void expert_gemv_scaled_mt(const ExpertMatrixView& m, const Codebook& cb,
                           const float* x, float w, float* y) {
  parallel_for(m.rows, 256, [&](uint64_t lo, uint64_t hi) {
    expert_gemv_scaled(expert_row_slice(m, lo, hi), cb, x, w, y + lo);
  });
}

void expert_gemv_scaled(const ExpertMatrixView& m, const Codebook& cb,
                        const float* x, float w, float* y) noexcept {
  const QuantSpec& s = m.spec;
  // Fast path is specialised for the locked format (D11): block 16, pair 2, 5-bit codes.
  // Anything else falls back to the reference — correctness never depends on the fast path.
  const bool fast = (s.block == 16 && s.pair == 2 && s.code_bits == 5 &&
                     s.block_field_bits() <= 8 && cb.pair == 2 && cb.k == 32 &&
                     (m.cols % 32) == 0);
  if (!fast) {
    std::vector<float> tmp(m.rows, 0.0f);
    expert_gemv_reference(m, cb, x, tmp.data());
    for (uint64_t r = 0; r < m.rows; ++r) y[r] += w * tmp[r];
    return;
  }

  const uint64_t nblk  = m.cols / 16;
  const uint32_t smask = (1u << s.scale_bits) - 1u;
  const uint32_t fmask = (1u << s.block_field_bits()) - 1u;
  const uint64_t fbits = s.block_field_bits();
  const uint64_t codes_row = m.cols / 2;
  const uint64_t npair = m.cols / 2;

  // Split the activation into even/odd planes ONCE per matrix. The codebook is 2-D (a code
  // decodes a PAIR), so with x pre-split the pair components line up with contiguous lanes and
  // no cross-lane interleave is needed in the inner loop. Cost is one pass over `x` (4096 floats)
  // against ~2.9 MiB of weights — free.
  std::vector<float> xe(npair), xo(npair);
  for (uint64_t i = 0; i < npair; ++i) { xe[i] = x[2 * i]; xo[i] = x[2 * i + 1]; }

  const CpuIsa isa = active_cpu_isa();
  const int eff_var = (g_cpu_variant >= 0) ? g_cpu_variant
                                           : ((isa >= CpuIsa::Avx512Vbmi) ? 4 : 2);

  if (eff_var == 1 || isa == CpuIsa::Scalar) {
    for (uint64_t r = 0; r < m.rows; ++r) {
      const float rs = rscale_of(m.rscales, r, s.row_scale_f32) * w;
      const uint64_t cbase = r * codes_row * 5;
      const uint64_t fbase = r * nblk * fbits;
      float acc = 0.0f;
      for (uint64_t b = 0; b < nblk; ++b) {
        const uint32_t field = (uint32_t)(load_bits(m.scales, fbase + b * fbits) & fmask);
        const float bs = rs * (float)(field & smask);
        if (bs == 0.0f) continue;
        const uint32_t var = field >> s.scale_bits;
        const float* lut = cb.v.data() + (size_t)(var < cb.variants ? var : 0u) * cb.k * 2;
        uint64_t bits = load_bits(m.data, cbase + b * 8 * 5);
        float part = 0.0f;
        for (int i = 0; i < 8; ++i) {
          const float* e = lut + (size_t)((bits >> (5 * i)) & 31u) * 2;
          part += e[0] * xe[b * 8 + i] + e[1] * xo[b * 8 + i];
        }
        acc += bs * part;
      }
      y[r] += acc;
    }
    return;
  }

#if AFF_CPU_X86
  alignas(64) float L[4][32];
  alignas(64) float H[4][32];
  for (uint32_t v = 0; v < 4u; ++v) {
    const uint32_t vv = v < cb.variants ? v : 0u;
    for (uint32_t c = 0; c < 32u; ++c) {
      L[v][c] = cb.entry(vv, c)[0];
      H[v][c] = cb.entry(vv, c)[1];
    }
  }

  if (isa >= CpuIsa::Avx512Vbmi && eff_var >= 3 && eff_var <= 5 && (m.cols % 64) == 0) {
    expert_gemv_run_avx512vbmi(m.data, m.scales, m.rscales, m.rows, nblk, codes_row, fbits,
                               s.scale_bits, smask, fmask, s.row_scale_f32, L, H, xe.data(),
                               xo.data(), w, y, eff_var);
    return;
  }
  if (isa >= CpuIsa::Avx512) {
    expert_gemv_run_avx512(m.data, m.scales, m.rscales, m.rows, nblk, codes_row, fbits,
                           s.scale_bits, smask, fmask, s.row_scale_f32, L, H, xe.data(),
                           xo.data(), w, y);
    return;
  }
  if (isa == CpuIsa::Avx2) {
    expert_gemv_run_avx2(m.data, m.scales, m.rscales, m.rows, nblk, codes_row, fbits, s.scale_bits,
                         smask, fmask, s.row_scale_f32, L, H, xe.data(), xo.data(), w, y);
    return;
  }
  if (isa == CpuIsa::Avx1) {
    expert_gemv_run_avx1(m.data, m.scales, m.rscales, m.rows, nblk, codes_row, fbits, s.scale_bits,
                         smask, fmask, s.row_scale_f32, L, H, xe.data(), xo.data(), w, y);
    return;
  }
#endif
}

} // namespace aff
