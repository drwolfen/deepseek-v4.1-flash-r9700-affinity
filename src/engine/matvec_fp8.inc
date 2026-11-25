// FP8-E4M3 dense matvec — the checkpoint's own dense format, decoded in the kernel.
//
// 92% of the dense parameters ship as F8_E4M3 with one UE8M0 scale per 128x128 tile. The quantiser
// used to dequantise them to bf16 and store bf16, which doubled the single largest traffic source
// in the engine (12.60 GB/token dense against 2.33 for the routed experts) and bought nothing: the
// model was TRAINED in this format, so bf16 carries no extra information. Storing the bytes
// verbatim is bit-exact with the checkpoint and halves that traffic.
//
// The container holds exactly what the safetensors held, no repacking:
//   data   row-major [rows, cols] E4M3, one byte per weight
//   scales row-major [rows/128, cols/128] E8M0, one byte per tile
//
// The scale is constant across a 128-wide run of a row, so it multiplies ONCE per tile rather than
// per weight — the inner loop is a plain dot product and the scale is a single FMA at the end of it.
// That is why the tile is the unit here and not the individual weight.

#pragma once

namespace aff {

struct Fp8Lut {
  float v[256];
  Fp8Lut() noexcept { for (int i = 0; i < 256; ++i) v[i] = fp8_e4m3_to_float((uint8_t)i); }
};
inline const Fp8Lut& fp8_lut() noexcept { static const Fp8Lut t; return t; }

namespace detail {

inline void matvec_fp8_rows_scalar(const uint8_t* W, const uint8_t* S, uint64_t row0,
                                   uint64_t nrows, uint64_t cols, const float* x, float* out,
                                   const float* lut) noexcept {
  const uint64_t sc_cols = (cols + 127) / 128;
  for (uint64_t r = 0; r < nrows; ++r) {
    const uint8_t* w = W + (row0 + r) * cols;
    const uint8_t* srow = S + ((row0 + r) >> 7) * sc_cols;
    float acc = 0.0f;
    for (uint64_t t = 0; t < sc_cols; ++t) {
      const uint64_t c0 = t * 128;
      const uint64_t c1 = (c0 + 128 < cols) ? (c0 + 128) : cols;
      const float scale = e8m0_to_float(srow[t]);
      float sub = 0.0f;
      for (uint64_t c = c0; c < c1; ++c) sub += lut[w[c]] * x[c];
      acc += sub * scale;
    }
    out[r] = acc;
  }
}

#if AFF_CPU_X86
AFF_TARGET_AVX1 inline void matvec_fp8_rows_avx1(const uint8_t* W, const uint8_t* S, uint64_t row0,
                                                 uint64_t nrows, uint64_t cols, const float* x,
                                                 float* out, const float* lut) noexcept {
  const uint64_t sc_cols = (cols + 127) / 128;
  const __m128i zero = _mm_setzero_si128();
  const __m128i k_sign = _mm_set1_epi32(0x80);
  const __m128i k_exp = _mm_set1_epi32(0x78);
  const __m128i k_man = _mm_set1_epi32(0x07);
  const __m128i k_120 = _mm_set1_epi32(120);
  const __m128 k_sub = _mm_set1_ps(1.0f / 512.0f);

  for (uint64_t r = 0; r < nrows; ++r) {
    const uint8_t* w = W + (row0 + r) * cols;
    const uint8_t* srow = S + ((row0 + r) >> 7) * sc_cols;
    float acc = 0.0f;
    for (uint64_t t = 0; t < sc_cols; ++t) {
      const uint64_t c0 = t * 128;
      const uint64_t c1 = (c0 + 128 < cols) ? (c0 + 128) : cols;
      const float scale = e8m0_to_float(srow[t]);
      __m256 a = _mm256_setzero_ps();
      uint64_t c = c0;
      for (; c + 8 <= c1; c += 8) {
        uint32_t u0, u1;
        std::memcpy(&u0, w + c, 4);
        std::memcpy(&u1, w + c + 4, 4);
        const __m128i b0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128((int)u0), zero), zero);
        const __m128i b1 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128((int)u1), zero), zero);

        const __m128i s0 = _mm_slli_epi32(_mm_and_si128(b0, k_sign), 24);
        const __m128i e0 = _mm_srli_epi32(_mm_and_si128(b0, k_exp), 3);
        const __m128i m0 = _mm_and_si128(b0, k_man);
        const __m128i nrm0 = _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e0, k_120), 23), _mm_slli_epi32(m0, 20));
        const __m128i sbn0 = _mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(m0), k_sub));
        const __m128i sub0 = _mm_cmpeq_epi32(e0, zero);
        const __m128 wlo = _mm_castsi128_ps(_mm_or_si128(
            _mm_or_si128(_mm_and_si128(sub0, sbn0), _mm_andnot_si128(sub0, nrm0)), s0));

        const __m128i s1 = _mm_slli_epi32(_mm_and_si128(b1, k_sign), 24);
        const __m128i e1 = _mm_srli_epi32(_mm_and_si128(b1, k_exp), 3);
        const __m128i m1 = _mm_and_si128(b1, k_man);
        const __m128i nrm1 = _mm_or_si128(_mm_slli_epi32(_mm_add_epi32(e1, k_120), 23), _mm_slli_epi32(m1, 20));
        const __m128i sbn1 = _mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(m1), k_sub));
        const __m128i sub1 = _mm_cmpeq_epi32(e1, zero);
        const __m128 whi = _mm_castsi128_ps(_mm_or_si128(
            _mm_or_si128(_mm_and_si128(sub1, sbn1), _mm_andnot_si128(sub1, nrm1)), s1));

        const __m256 wv = _mm256_insertf128_ps(_mm256_castps128_ps256(wlo), whi, 1);
        a = _mm256_add_ps(a, _mm256_mul_ps(wv, _mm256_loadu_ps(x + c)));
      }
      const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
      const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
      const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
      float sub = _mm_cvtss_f32(h1);
      for (; c < c1; ++c) sub += lut[w[c]] * x[c];
      acc += sub * scale;
    }
    out[r] = acc;
  }
}

AFF_TARGET_AVX2 inline void matvec_fp8_rows_avx2(const uint8_t* W, const uint8_t* S, uint64_t row0,
                                                 uint64_t nrows, uint64_t cols, const float* x,
                                                 float* out, const float* lut) noexcept {
  const uint64_t sc_cols = (cols + 127) / 128;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i k_sign = _mm256_set1_epi32(0x80);
  const __m256i k_exp = _mm256_set1_epi32(0x78);
  const __m256i k_man = _mm256_set1_epi32(0x07);
  const __m256i k_120 = _mm256_set1_epi32(120);
  const __m256 k_sub = _mm256_set1_ps(1.0f / 512.0f);

  for (uint64_t r = 0; r < nrows; ++r) {
    const uint8_t* w = W + (row0 + r) * cols;
    const uint8_t* srow = S + ((row0 + r) >> 7) * sc_cols;
    float acc = 0.0f;
    for (uint64_t t = 0; t < sc_cols; ++t) {
      const uint64_t c0 = t * 128;
      const uint64_t c1 = (c0 + 128 < cols) ? (c0 + 128) : cols;
      const float scale = e8m0_to_float(srow[t]);
      __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
      uint64_t c = c0;
      for (; c + 16 <= c1; c += 16) {
        for (int h = 0; h < 2; ++h) {
          const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + c + h * 8));
          const __m256i b = _mm256_cvtepu8_epi32(raw);
          const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, k_sign), 24);
          const __m256i ex = _mm256_srli_epi32(_mm256_and_si256(b, k_exp), 3);
          const __m256i mn = _mm256_and_si256(b, k_man);
          const __m256i nrm = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(ex, k_120), 23),
                                              _mm256_slli_epi32(mn, 20));
          const __m256 sbn = _mm256_mul_ps(_mm256_cvtepi32_ps(mn), k_sub);
          const __m256 is_sub = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ex, zero));
          const __m256 mag = _mm256_blendv_ps(_mm256_castsi256_ps(nrm), sbn, is_sub);
          const __m256 wv = _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sign));
          if (h == 0) a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), a0);
          else        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c + 8), a1);
        }
      }
      for (; c + 8 <= c1; c += 8) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + c));
        const __m256i b = _mm256_cvtepu8_epi32(raw);
        const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, k_sign), 24);
        const __m256i ex = _mm256_srli_epi32(_mm256_and_si256(b, k_exp), 3);
        const __m256i mn = _mm256_and_si256(b, k_man);
        const __m256i nrm = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(ex, k_120), 23),
                                            _mm256_slli_epi32(mn, 20));
        const __m256 sbn = _mm256_mul_ps(_mm256_cvtepi32_ps(mn), k_sub);
        const __m256 is_sub = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ex, zero));
        const __m256 mag = _mm256_blendv_ps(_mm256_castsi256_ps(nrm), sbn, is_sub);
        const __m256 wv = _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sign));
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), a0);
      }
      const __m256 s = _mm256_add_ps(a0, a1);
      const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
      const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
      const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
      float sub = _mm_cvtss_f32(h1);
      for (; c < c1; ++c) sub += lut[w[c]] * x[c];
      acc += sub * scale;
    }
    out[r] = acc;
  }
}

AFF_TARGET_AVX512 inline void matvec_fp8_rows_avx512(const uint8_t* W, const uint8_t* S,
                                                     uint64_t row0, uint64_t nrows, uint64_t cols,
                                                     const float* x, float* out,
                                                     const float* lut) noexcept {
  const uint64_t sc_cols = (cols + 127) / 128;
  const __m512i k_sign = _mm512_set1_epi32(0x80);
  const __m512i k_exp = _mm512_set1_epi32(0x78);
  const __m512i k_man = _mm512_set1_epi32(0x07);
  const __m512i k_120 = _mm512_set1_epi32(120);
  const __m512 k_sub = _mm512_set1_ps(1.0f / 512.0f);
  const __m512i zero = _mm512_setzero_si512();

  for (uint64_t r = 0; r < nrows; ++r) {
    const uint8_t* w = W + (row0 + r) * cols;
    const uint8_t* srow = S + ((row0 + r) >> 7) * sc_cols;
    float acc = 0.0f;
    for (uint64_t t = 0; t < sc_cols; ++t) {
      const uint64_t c0 = t * 128;
      const uint64_t c1 = (c0 + 128 < cols) ? (c0 + 128) : cols;
      const float scale = e8m0_to_float(srow[t]);
      __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
      uint64_t c = c0;
      for (; c + 32 <= c1; c += 32) {
        for (int h = 0; h < 2; ++h) {
          const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + h * 16));
          const __m512i b = _mm512_cvtepu8_epi32(raw);
          const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, k_sign), 24);
          const __m512i ex = _mm512_srli_epi32(_mm512_and_si512(b, k_exp), 3);
          const __m512i mn = _mm512_and_si512(b, k_man);
          const __m512i nrm = _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(ex, k_120), 23),
                                              _mm512_slli_epi32(mn, 20));
          const __m512 sbn = _mm512_mul_ps(_mm512_cvtepi32_ps(mn), k_sub);
          const __mmask16 is_sub = _mm512_cmpeq_epi32_mask(ex, zero);
          const __m512 mag = _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(nrm), sbn);
          const __m512 wv = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(mag), sign));
          if (h == 0) a0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c), a0);
          else        a1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c + 16), a1);
        }
      }
      float sub = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
      for (; c < c1; ++c) sub += lut[w[c]] * x[c];
      acc += sub * scale;
    }
    out[r] = acc;
  }
}
#endif

}  // namespace detail

inline void matvec_fp8_rows(const uint8_t* W, const uint8_t* S, uint64_t row0, uint64_t nrows,
                            uint64_t cols, const float* x, float* out) noexcept {
  const float* lut = fp8_lut().v;
#if AFF_CPU_X86
  const CpuIsa isa = active_cpu_isa();
  if (isa >= CpuIsa::Avx512) { detail::matvec_fp8_rows_avx512(W, S, row0, nrows, cols, x, out, lut); return; }
  if (isa == CpuIsa::Avx2)   { detail::matvec_fp8_rows_avx2(W, S, row0, nrows, cols, x, out, lut); return; }
  if (isa == CpuIsa::Avx1)   { detail::matvec_fp8_rows_avx1(W, S, row0, nrows, cols, x, out, lut); return; }
#endif
  detail::matvec_fp8_rows_scalar(W, S, row0, nrows, cols, x, out, lut);
}

}  // namespace aff
