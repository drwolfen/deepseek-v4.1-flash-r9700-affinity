#include "engine/cactus.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main() {
  using namespace aff;

  // 1. Legacy equivalence when eps=0.0
  CactusParams p0{};
  p0.delta = 0.5;
  p0.cap_r = 0.5;
  p0.fade_start = 4096;
  p0.fade_end = 16384;
  p0.escape = 0.0;

  // At generated=0, q=0.5, delta=0.5, cap_r=0.5 -> h_raw=0.75, h_cap=0.75 -> 0.75
  double h0 = cactus_threshold(0.5, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h0 - 0.75) < 1e-6);
  double a0 = cactus_accept_prob(0.5, 1.0, 0, p0);
  assert(std::fabs(a0 - 0.75) < 1e-6);

  // At generated=16384 (past fade_end), h == q exactly
  double h_end = cactus_threshold(0.5, 16384, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_end - 0.5) < 1e-6);
  double a_end = cactus_accept_prob(0.5, 1.0, 16384, p0);
  assert(std::fabs(a_end - 0.5) < 1e-6);

  // At q=0.8, delta=0.5, cap_r=0.5 -> h_raw = 0.96, h_cap = 0.90 -> capped at 0.90
  double h_cap = cactus_threshold(0.8, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_cap - 0.90) < 1e-6);
  double a_cap = cactus_accept_prob(0.8, 1.0, 0, p0);
  assert(std::fabs(a_cap - 0.90) < 1e-6);

  // 2. Default eps=0.25 escape out of q >= 0.5 => h = 1 lock-in corner
  // Consider delta=1.0 (or high delta) where h saturates at 1.0, e.g. q=0.6, p=1.0:
  CactusParams p_esc{};
  p_esc.delta = 1.0;
  p_esc.cap_r = 0.0; // uncapped to allow h=1.0
  p_esc.fade_start = 4096;
  p_esc.fade_end = 16384;
  p_esc.escape = 0.25;

  double h_sat = cactus_threshold(0.6, 0, p_esc.delta, p_esc.cap_r, p_esc.fade_start, p_esc.fade_end);
  // h_raw = min(1.0, 0.6 * (1 + 2 * 1.0 * 0.4)) = min(1.0, 0.6 * 1.8) = 1.0
  assert(std::fabs(h_sat - 1.0) < 1e-6);
  // With legacy rule (eps=0), prob = 1.0 (lock-in)
  p_esc.escape = 0.0;
  double a_lock = cactus_accept_prob(0.6, 1.0, 0, p_esc);
  assert(std::fabs(a_lock - 1.0) < 1e-6);
  // With escape mixture (eps=0.25): (1 - 0.25) * 1.0 + 0.25 * 0.6 = 0.75 + 0.15 = 0.90 < 1.0
  p_esc.escape = 0.25;
  double a_esc = cactus_accept_prob(0.6, 1.0, 0, p_esc);
  assert(std::fabs(a_esc - 0.90) < 1e-6);
  // Guarantees escape probability = 1.0 - 0.90 = 0.10 (10% rejection chance)
  assert(a_esc < 1.0);

  // 3. Probability bounds [0, 1] across broad spectrum of q, p, eps
  for (double q = 0.0; q <= 1.0; q += 0.05) {
    for (double p = 0.1; p <= 1.0; p += 0.1) {
      for (double eps = 0.0; eps <= 1.0; eps += 0.25) {
        CactusParams pt{};
        pt.delta = 0.5;
        pt.cap_r = 0.5;
        pt.escape = eps;
        double pr = cactus_accept_prob(q, p, 0, pt);
        assert(pr >= 0.0 && pr <= 1.0);
      }
    }
  }

  // 4. AFF_CACTUS_ESCAPE env parsing
  ::unsetenv("AFF_CACTUS_ESCAPE");
  CactusParams def_p = CactusParams::from_env();
  assert(std::fabs(def_p.escape - 0.25) < 1e-6);

  ::setenv("AFF_CACTUS_ESCAPE", "0.5", 1);
  CactusParams set_p = CactusParams::from_env();
  assert(std::fabs(set_p.escape - 0.5) < 1e-6);

  ::setenv("AFF_CACTUS_ESCAPE", "1.5", 1); // should clamp to 1.0
  CactusParams clamp_hi = CactusParams::from_env();
  assert(std::fabs(clamp_hi.escape - 1.0) < 1e-6);

  ::setenv("AFF_CACTUS_ESCAPE", "-0.2", 1); // should clamp to 0.0
  CactusParams clamp_lo = CactusParams::from_env();
  assert(std::fabs(clamp_lo.escape - 0.0) < 1e-6);

  ::unsetenv("AFF_CACTUS_ESCAPE");

  std::printf("PASS: test_cactus_accept\n");
  return 0;
}
