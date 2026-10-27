#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

static inline double cactus_threshold(double q, uint32_t generated, double delta, double cap_r,
                                      uint32_t fade_start, uint32_t fade_end) {
  if (delta <= 0.0 || q <= 0.0 || q >= 1.0) return q;
  double fade = 1.0;
  if (fade_end > fade_start) {
    fade = std::clamp((double)((int64_t)fade_end - (int64_t)generated) /
                          (double)(fade_end - fade_start),
                      0.0, 1.0);
  }
  const double delta_eff = delta * fade;
  const double h_raw = std::min(1.0, q * (1.0 + 2.0 * delta_eff * (1.0 - q)));
  if (cap_r > 0.0) {
    const double h_cap = 1.0 - cap_r * (1.0 - q);
    return std::max(q, std::min(h_raw, h_cap));
  }
  return h_raw;
}

int main() {
  // At generated=0, q=0.5, delta=0.5, cap_r=0.5 -> h_raw=0.75, h_cap=0.75 -> 0.75 > 0.5
  double h0 = cactus_threshold(0.5, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h0 - 0.75) < 1e-6);
  // At generated=16384 (past fade_end), h == q exactly (stock rejection rule)
  double h_end = cactus_threshold(0.5, 16384, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_end - 0.5) < 1e-6);
  // At q=0.8, delta=0.5, cap_r=0.5 -> h_raw = 0.96, h_cap = 0.90 -> capped at 0.90
  double h_cap = cactus_threshold(0.8, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_cap - 0.90) < 1e-6);
  std::printf("PASS: test_cactus_accept\n");
  return 0;
}
