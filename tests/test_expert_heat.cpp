#include "engine/expert_heat.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

namespace {

using namespace aff;

void test_basic_tie_break_and_guards() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 0.98f;
  p.hysteresis = 1.25f;
  p.min_gain = 1.5f;
  eh.init(/*n_layer=*/2, /*n_expert=*/4, p);

  // Layer 0 holds 3 resident experts; layer 1 holds 1 resident expert.
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Vram);
  eh.set_tier(2, ExpertTier::Vram);
  eh.set_tier(3, ExpertTier::Pool);

  eh.set_tier(4, ExpertTier::Vram);
  eh.set_tier(5, ExpertTier::Pool);
  eh.set_tier(6, ExpertTier::Pool);
  eh.set_tier(7, ExpertTier::Pool);

  const uint32_t sel_e = 1; // expert 1 in layer 1 -> global idx 5
  const uint32_t cnt = 2;
  eh.observe(1, &sel_e, 1, &cnt);

  uint32_t prom = 0, dem = 0;
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 5u);
  // Victim at heat==0 comes from layer 0 (most resident experts = 3).
  CHECK(dem < 4u);

  std::printf("PASS: test_basic_tie_break_and_guards\n");
}

void test_max_block_credit() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 0.992f;
  p.hysteresis = 1.25f;
  p.min_gain = 3.0f;
  p.max_block_credit = 2;
  eh.init(/*n_layer=*/1, /*n_expert=*/4, p);

  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Pool);

  // Single-block sighting with count=6 must be capped at +2.0 (< min_gain 3.0).
  const uint32_t e = 1;
  const uint32_t count = 6;
  eh.observe(0, &e, 1, &count);
  CHECK(eh.heat(1) == 2.0f);

  uint32_t prom = 0, dem = 0;
  CHECK(!eh.pick(&prom, &dem));

  eh.note_missing(0, &e, 1);
  CHECK(!eh.pick_demand(&prom, &dem));

  // Second block sighting reaches 2.0 * 0.992 + 2.0 = 3.984 >= 3.0 -> passes.
  eh.observe(0, &e, 1, &count);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 1u && dem == 0u);

  eh.note_missing(0, &e, 1);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 1u && dem == 0u);

  std::printf("PASS: test_max_block_credit\n");
}

void test_cooldown_steps() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 1.0f;
  p.hysteresis = 1.0f;
  p.min_gain = 1.0f;
  p.cooldown_steps = 16;
  const uint32_t n_layer = 3;
  const uint32_t n_expert = 4;
  eh.init(n_layer, n_expert, p);

  // Initial tier setup (ticks == 0) must not arm cooldown.
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Vram);
  eh.set_tier(2, ExpertTier::Pool);
  eh.set_tier(3, ExpertTier::Pool);

  // Advance 1 tick so ticks_ > 0, and heat up pooled expert 2.
  const uint32_t e2 = 2;
  const uint32_t c5 = 5;
  eh.observe(0, &e2, 1, &c5);
  CHECK(eh.ticks() == 1);

  uint32_t prom = 0, dem = 0;
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 2u);

  // Publish swap: 2 -> Vram, 0 -> Pool (both enter cooldown for 16 * n_layer = 48 ticks).
  eh.set_tier(2, ExpertTier::Vram);
  eh.set_tier(0, ExpertTier::Pool);

  // Even if expert 0 becomes hotter than expert 1, cooldown blocks re-promotion/demotion
  // for 16 * n_layer - 1 ticks.
  const uint32_t e0 = 0;
  const uint32_t c10 = 10;
  for (uint32_t t = 0; t < p.cooldown_steps * n_layer - 1; ++t) {
    eh.observe(0, &e0, 1, &c10);
    // Expert 0 (on cooldown) cannot be promoted; expert 2 (on cooldown) cannot be demoted,
    // though expert 1 is not on cooldown so only 0 is blocked as candidate.
    CHECK(!eh.pick(&prom, &dem));
    eh.note_missing(0, &e0, 1);
    CHECK(!eh.pick_demand(&prom, &dem));
  }

  // On the 16 * n_layer-th tick, cooldown expires and expert 0 can be promoted again.
  eh.observe(0, &e0, 1, &c10);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 0u && dem == 1u);

  eh.note_missing(0, &e0, 1);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 0u && dem == 1u);

  // Verify cooldown also protects a recently promoted Vram victim from immediate demotion.
  // Publish the (0 -> Vram, 1 -> Pool) swap, then place idx 4 in Vram (all at ticks > 0 -> cooldown).
  // Expert 3 was placed in Pool at ticks==0 (not on cooldown).
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Pool);
  eh.set_tier(4, ExpertTier::Vram);
  const uint32_t e_l0_2 = 2;
  const uint32_t e_l0_3 = 3;
  eh.observe(0, &e_l0_2, 1, &c10);
  eh.observe(0, &e_l0_3, 1, &c10);
  // Now Vram experts: 0 (high heat, on cooldown), 2 (cooldown expired, heat=15), 4 (ON COOLDOWN, heat=0).
  // Pool expert 3 (not on cooldown, heat=10): fails against 2 (10 < 15+1) and must skip 4 due to cooldown.
  CHECK(!eh.pick(&prom, &dem));
  eh.note_missing(0, &e_l0_3, 1);
  CHECK(!eh.pick_demand(&prom, &dem));

  // Advance cooldown_steps * n_layer ticks on layer 2 (leaves layer 0 & 1 heats untouched).
  const uint32_t e_l2_0 = 0;
  for (uint32_t t = 0; t < p.cooldown_steps * n_layer; ++t) eh.observe(2, &e_l2_0, 1, &c5);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 3u && dem == 4u);

  std::printf("PASS: test_cooldown_steps\n");
}

}  // namespace

int main() {
  test_basic_tie_break_and_guards();
  test_max_block_credit();
  test_cooldown_steps();
  std::printf("PASS: all expert_heat tests passed.\n");
  return 0;
}
