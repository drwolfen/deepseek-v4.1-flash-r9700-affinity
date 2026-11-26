// bf16 dense matvec — the hottest kernel in the engine by BYTES MOVED. Storage is not traffic: only
// six of 256 experts are read per token, while EVERY dense weight is, so the dense side is several
// times the routed-expert traffic despite being a fraction of the container.
//
// Single-threaded this kernel already runs at 83-90% of the single-thread DRAM wall. Unlike the
// expert kernel, which unpacks 2.875-bit codes through a codebook and is genuinely decode-limited,
// this one only shifts bf16 left by 16 — there is no arithmetic to hide behind, so no arithmetic
// change can buy anything. Variant 2's four accumulators are the default because the spread is
// tighter, not because they are faster.
//
// THE LEVER IS THE FORMAT, NOT THE KERNEL — and not by quantising further. Most of these dense
// parameters ship as F8_E4M3 with a per-128x128 UE8M0 scale, and dequantising them to bf16 at
// quantise time doubles the largest traffic source in the engine for nothing. Storing FP8 as FP8 is
// bit-exact with the checkpoint and halves it, and the VRAM it returns is residency budget.

#pragma once

namespace aff {
namespace detail {

inline void matvec_bf16_v1_scalar(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                                  float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    float acc = 0.0f;
    for (uint64_t c = 0; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

#if AFF_CPU_X86
AFF_TARGET_AVX1 inline void matvec_bf16_v1_avx1(const uint16_t* W, const float* x, uint64_t rows,
                                                uint64_t cols, float* out) noexcept {
  const __m128i zero = _mm_setzero_si128();
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m256 a = _mm256_setzero_ps();
    uint64_t c = 0;
    for (; c + 8 <= cols; c += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      const __m128 lo = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h));
      const __m128 hi = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h));
      const __m256 wv = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
      a = _mm256_add_ps(a, _mm256_mul_ps(wv, _mm256_loadu_ps(x + c)));
    }
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    float acc = _mm_cvtss_f32(h1);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

AFF_TARGET_AVX2 inline void matvec_bf16_v1_avx2(const uint16_t* W, const float* x, uint64_t rows,
                                                uint64_t cols, float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m256 a = _mm256_setzero_ps();
    uint64_t c = 0;
    for (; c + 8 <= cols; c += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      const __m256 wv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
      a = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), a);
    }
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    float acc = _mm_cvtss_f32(h1);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

AFF_TARGET_AVX512 inline void matvec_bf16_v1_avx512(const uint16_t* W, const float* x, uint64_t rows,
                                                    uint64_t cols, float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m512 a = _mm512_setzero_ps();
    uint64_t c = 0;
    for (; c + 16 <= cols; c += 16) {
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      const __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
      a = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + c), a);
    }
    float acc = _mm512_reduce_add_ps(a);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

AFF_TARGET_AVX1 inline void matvec_bf16_v2_avx1(const uint16_t* W, const float* x, uint64_t rows,
                                                uint64_t cols, float* out) noexcept {
  const __m128i zero = _mm_setzero_si128();
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    uint64_t c = 0;
    for (; c + 32 <= cols; c += 32) {
      const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      const __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 8));
      const __m128i h2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16));
      const __m128i h3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 24));
      const __m256 w0 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, h0))),
          _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h0)), 1);
      const __m256 w1 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1))),
          _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h1)), 1);
      const __m256 w2 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2))),
          _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h2)), 1);
      const __m256 w3 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, h3))),
          _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h3)), 1);
      a0 = _mm256_add_ps(a0, _mm256_mul_ps(w0, _mm256_loadu_ps(x + c)));
      a1 = _mm256_add_ps(a1, _mm256_mul_ps(w1, _mm256_loadu_ps(x + c + 8)));
      a2 = _mm256_add_ps(a2, _mm256_mul_ps(w2, _mm256_loadu_ps(x + c + 16)));
      a3 = _mm256_add_ps(a3, _mm256_mul_ps(w3, _mm256_loadu_ps(x + c + 24)));
    }
    for (; c + 8 <= cols; c += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      const __m256 wv = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_castsi128_ps(_mm_unpacklo_epi16(zero, h))),
          _mm_castsi128_ps(_mm_unpackhi_epi16(zero, h)), 1);
      a0 = _mm256_add_ps(a0, _mm256_mul_ps(wv, _mm256_loadu_ps(x + c)));
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    float acc = _mm_cvtss_f32(h1);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

AFF_TARGET_AVX2 inline void matvec_bf16_v2_avx2(const uint16_t* W, const float* x, uint64_t rows,
                                                uint64_t cols, float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    uint64_t c = 0;
    for (; c + 32 <= cols; c += 32) {
      const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      const __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 8));
      const __m128i h2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16));
      const __m128i h3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 24));
      a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16)),
                           _mm256_loadu_ps(x + c), a0);
      a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16)),
                           _mm256_loadu_ps(x + c + 8), a1);
      a2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16)),
                           _mm256_loadu_ps(x + c + 16), a2);
      a3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h3), 16)),
                           _mm256_loadu_ps(x + c + 24), a3);
    }
    for (; c + 8 <= cols; c += 8) {
      const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c));
      a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)),
                           _mm256_loadu_ps(x + c), a0);
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    const __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    const __m128 h2 = _mm_add_ps(h4, _mm_movehl_ps(h4, h4));
    const __m128 h1 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1));
    float acc = _mm_cvtss_f32(h1);
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}

AFF_TARGET_AVX512 inline void matvec_bf16_v2_avx512(const uint16_t* W, const float* x, uint64_t rows,
                                                    uint64_t cols, float* out) noexcept {
  for (uint64_t r = 0; r < rows; ++r) {
    const uint16_t* w = W + r * cols;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    uint64_t c = 0;
    for (; c + 64 <= cols; c += 64) {
      const __m256i h0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      const __m256i h1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 16));
      const __m256i h2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 32));
      const __m256i h3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c + 48));
      a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h0), 16)),
                           _mm512_loadu_ps(x + c), a0);
      a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h1), 16)),
                           _mm512_loadu_ps(x + c + 16), a1);
      a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h2), 16)),
                           _mm512_loadu_ps(x + c + 32), a2);
      a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h3), 16)),
                           _mm512_loadu_ps(x + c + 48), a3);
    }
    for (; c + 16 <= cols; c += 16) {
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
      a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)),
                           _mm512_loadu_ps(x + c), a0);
    }
    float acc = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; c < cols; ++c) acc += bf16_to_float(w[c]) * x[c];
    out[r] = acc;
  }
}
#endif

}  // namespace detail

inline void matvec_bf16_v1(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out, CpuIsa isa) noexcept {
#if AFF_CPU_X86
  if (isa >= CpuIsa::Avx512) { detail::matvec_bf16_v1_avx512(W, x, rows, cols, out); return; }
  if (isa == CpuIsa::Avx2)   { detail::matvec_bf16_v1_avx2(W, x, rows, cols, out); return; }
  if (isa == CpuIsa::Avx1)   { detail::matvec_bf16_v1_avx1(W, x, rows, cols, out); return; }
#else
  (void)isa;
#endif
  detail::matvec_bf16_v1_scalar(W, x, rows, cols, out);
}

inline void matvec_bf16_v1(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out) noexcept {
  matvec_bf16_v1(W, x, rows, cols, out, active_cpu_isa());
}

inline void matvec_bf16_v2(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out, CpuIsa isa) noexcept {
#if AFF_CPU_X86
  if (isa >= CpuIsa::Avx512) { detail::matvec_bf16_v2_avx512(W, x, rows, cols, out); return; }
  if (isa == CpuIsa::Avx2)   { detail::matvec_bf16_v2_avx2(W, x, rows, cols, out); return; }
  if (isa == CpuIsa::Avx1)   { detail::matvec_bf16_v2_avx1(W, x, rows, cols, out); return; }
#else
  (void)isa;
#endif
  detail::matvec_bf16_v1_scalar(W, x, rows, cols, out);
}

inline void matvec_bf16_v2(const uint16_t* W, const float* x, uint64_t rows, uint64_t cols,
                           float* out) noexcept {
  matvec_bf16_v2(W, x, rows, cols, out, active_cpu_isa());
}

}  // namespace aff
