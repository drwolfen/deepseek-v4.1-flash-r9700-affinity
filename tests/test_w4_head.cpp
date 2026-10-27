#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static void quantize_bf16_to_w4_g128_cpu(const uint16_t* src, uint8_t* w4, uint16_t* scales,
                                         uint32_t rows, uint32_t cols) {
  const uint32_t groups_per_row = cols / 128;
  const uint32_t total_groups = rows * groups_per_row;
  for (uint32_t blk = 0; blk < total_groups; ++blk) {
    const uint16_t* gptr = src + (size_t)blk * 128;
    float amax = 0.0f;
    for (uint32_t i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(gptr[i])));
    float best_s = 0.0f;
    float best_err = 1e30f;
    if (amax > 0.0f) {
      for (int step = 0; step <= 8; ++step) {
        const float alpha = 0.80f + 0.025f * (float)step;
        const float s_cand = bf16_to_f32(f32_to_bf16((amax * alpha) / 7.0f));
        if (s_cand <= 0.0f) continue;
        const float inv_s = 1.0f / s_cand;
        float err = 0.0f;
        for (uint32_t i = 0; i < 128; ++i) {
          const float f = bf16_to_f32(gptr[i]);
          const float q = std::min(7.0f, std::max(-7.0f, std::nearbyintf(f * inv_s)));
          const float d = f - q * s_cand;
          err += d * d;
        }
        if (err < best_err) {
          best_err = err;
          best_s = s_cand;
        }
      }
    }
    scales[blk] = f32_to_bf16(best_s);
    const float inv_best = best_s > 0.0f ? (1.0f / best_s) : 0.0f;
    for (uint32_t lane = 0; lane < 32; ++lane) {
      uint32_t q[4];
      for (uint32_t k = 0; k < 4; ++k) {
        const float f = bf16_to_f32(gptr[lane * 4 + k]);
        const int qi = best_s > 0.0f
                           ? (int)std::min(7.0f, std::max(-7.0f, std::nearbyintf(f * inv_best)))
                           : 0;
        q[k] = (uint32_t)(qi + 8);
      }
      const uint16_t packed2 = (uint16_t)(q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12));
      ((uint16_t*)w4)[(size_t)blk * 32 + lane] = packed2;
    }
  }
}

int main() {
  const uint32_t rows = 64, cols = 512, nt = 5;
  std::vector<uint16_t> w_bf16((size_t)rows * cols);
  for (size_t i = 0; i < w_bf16.size(); ++i) {
    float v = std::sin((float)i * 0.0173f) * 0.08f + std::cos((float)i * 0.113f) * 0.03f;
    w_bf16[i] = f32_to_bf16(v);
  }
  std::vector<float> x((size_t)nt * cols);
  for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)i * 0.031f);

  std::vector<uint8_t> w4((size_t)rows * (cols / 2));
  std::vector<uint16_t> scales((size_t)rows * (cols / 128));
  quantize_bf16_to_w4_g128_cpu(w_bf16.data(), w4.data(), scales.data(), rows, cols);

  double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (uint32_t r = 0; r < rows; ++r) {
    const uint8_t* w4_row = w4.data() + (size_t)r * (cols / 2);
    const uint16_t* sc_row = scales.data() + (size_t)r * (cols / 128);
    for (uint32_t t = 0; t < nt; ++t) {
      float ref = 0.0f;
      for (uint32_t c = 0; c < cols; ++c)
        ref += bf16_to_f32(w_bf16[(size_t)r * cols + c]) * x[(size_t)t * cols + c];

      float got = 0.0f;
      for (uint32_t c0 = 0; c0 < cols; c0 += 256) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          const uint32_t c = c0 + lane * 8;
          uint32_t p;
          std::memcpy(&p, w4_row + (c >> 1), 4);
          const float s = bf16_to_f32(sc_row[c >> 7]);
          float sub = 0.0f;
          for (uint32_t k = 0; k < 8; ++k) {
            const float wk = (float)(int)((p >> (4 * k)) & 0xFu) - 8.0f;
            sub += wk * x[(size_t)t * cols + c + k];
          }
          got += sub * s;
        }
      }
      dot_ab += (double)ref * (double)got;
      norm_a += (double)ref * (double)ref;
      norm_b += (double)got * (double)got;
    }
  }
  const double cos_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
  assert(cos_sim > 0.996);
  std::printf("PASS: test_w4_head (cos_sim=%.6f)\n", cos_sim);
  return 0;
}
