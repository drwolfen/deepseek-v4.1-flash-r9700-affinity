#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace aff {

struct CactusParams {
  double delta = 0.0;
  double cap_r = 0.5;
  uint32_t fade_start = 4096;
  uint32_t fade_end = 16384;
  double escape = 0.25;

  static CactusParams from_env() {
    CactusParams p;
    if (const char* e = std::getenv("AFF_CACTUS_DELTA"); e && *e)
      p.delta = std::strtod(e, nullptr);
    if (const char* e = std::getenv("AFF_CACTUS_CAP_R"); e && *e)
      p.cap_r = std::strtod(e, nullptr);
    if (const char* e = std::getenv("AFF_CACTUS_FADE_START"); e && *e)
      p.fade_start = (uint32_t)std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("AFF_CACTUS_FADE_END"); e && *e)
      p.fade_end = (uint32_t)std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("AFF_CACTUS_ESCAPE"); e && *e)
      p.escape = std::strtod(e, nullptr);
    p.escape = std::clamp(p.escape, 0.0, 1.0);
    return p;
  }
};

inline double cactus_threshold(double q, uint32_t generated, double delta, double cap_r,
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

// Convex mixture: (1 - eps) * min(1, h / p) + eps * min(1, q / p)
// Eliminates the temp=0.6 reasoning repetition lock-in (q >= 0.5 => h = 1)
// by guaranteeing every token retains at least eps of the stock rejection chance.
inline double cactus_accept_prob(double q, double p, uint32_t generated, const CactusParams& params) {
  if (q <= 0.0) return 0.0;
  const double p_safe = (p > 0.0) ? p : 1.0;
  const double a_stock = std::min(1.0, q / p_safe);
  if (params.delta <= 0.0) return a_stock;

  const double h = cactus_threshold(q, generated, params.delta, params.cap_r,
                                    params.fade_start, params.fade_end);
  const double a_cactus = std::min(1.0, h / p_safe);
  const double eps = std::clamp(params.escape, 0.0, 1.0);
  return (1.0 - eps) * a_cactus + eps * a_stock;
}

}  // namespace aff
