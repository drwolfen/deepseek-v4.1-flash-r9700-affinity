#include <algorithm>
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

// Reference 1017 act_quant(x, 32, "ue8m0", inplace=True) in FP32/E4M3 grid:
static float e4m3_round_ref(float v) {
  float a = std::fabs(v);
  if (a == 0.0f) return v;
  uint32_t bits;
  std::memcpy(&bits, &a, 4);
  int ea = std::max((int)((bits >> 23) & 0xFF) - 127, -6);
  uint32_t ulp_bits = (uint32_t)(ea - 3 + 127) << 23;
  float ulp;
  std::memcpy(&ulp, &ulp_bits, 4);
  float r = a / ulp;
  r = (r + 12582912.0f) - 12582912.0f;
  float q = r * ulp;
  return std::signbit(v) ? -q : q;
}

static void o_fp8_qdq_block32_ref(uint16_t* x, uint32_t dim, uint32_t nb) {
  for (uint32_t b = 0; b < nb; ++b) {
    for (uint32_t d0 = 0; d0 < dim; d0 += 32) {
      float amax = 1e-4f;
      for (uint32_t i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(bf16_to_f32(x[(size_t)(d0 + i) * nb + b])));
      float t = amax * (1.0f / 448.0f);
      uint32_t tbits;
      std::memcpy(&tbits, &t, 4);
      int e = (int)((tbits >> 23) & 0xFF) - 127 + ((tbits & 0x7FFFFFu) != 0 ? 1 : 0);
      uint32_t sbits = (uint32_t)(e + 127) << 23;
      float s;
      std::memcpy(&s, &sbits, 4);
      for (uint32_t i = 0; i < 32; ++i) {
        float v = bf16_to_f32(x[(size_t)(d0 + i) * nb + b]) / s;
        v = std::min(448.0f, std::max(-448.0f, v));
        x[(size_t)(d0 + i) * nb + b] = f32_to_bf16(e4m3_round_ref(v) * s);
      }
    }
  }
}

int main() {
  std::vector<uint16_t> buf(64 * 2);
  for (size_t i = 0; i < buf.size(); ++i)
    buf[i] = f32_to_bf16((float)((int)i - 30) * 0.137f);
  std::vector<uint16_t> ref = buf;
  o_fp8_qdq_block32_ref(ref.data(), 64, 2);
  bool changed = false;
  for (size_t i = 0; i < buf.size(); ++i) {
    if (ref[i] != buf[i]) changed = true;
    // Also verify idempotence: applying block-32 E4M3 QDQ a second time must be a no-op
  }
  if (!changed) {
    std::fprintf(stderr, "FAIL: reference QDQ did not quantize\n");
    return 1;
  }
  std::vector<uint16_t> ref2 = ref;
  o_fp8_qdq_block32_ref(ref2.data(), 64, 2);
  for (size_t i = 0; i < ref.size(); ++i) {
    if (ref2[i] != ref[i]) {
      std::fprintf(stderr, "FAIL: QDQ not idempotent at %zu: %04x vs %04x\n", i, ref[i], ref2[i]);
      return 1;
    }
  }
  std::printf("PASS: test_o_fp8_qdq\n");
  return 0;
}
