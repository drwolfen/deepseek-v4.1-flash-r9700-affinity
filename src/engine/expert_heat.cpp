#include "engine/expert_heat.h"

#include <algorithm>

namespace aff {

void ExpertHeat::init(uint32_t n_layer, uint32_t n_expert, const HeatParams& p) {
  params_ = p;
  n_layer_ = n_layer;
  n_expert_ = n_expert;
  ticks_ = 0;
  const size_t n = (size_t)n_layer * n_expert;
  heat_.assign(n, 0.0f);
  tier_.assign(n, ExpertTier::Ssd);
  busy_.assign(n, 0u);
  protected_.assign(n, 0u);
  cooldown_until_.assign(n, 0ull);
  res_count_.assign(n_layer, 0u);
  demand_.clear();
  demand_.reserve(n);
  demand_seen_.assign(n, 0u);
}

void ExpertHeat::set_tier(uint32_t idx, ExpertTier t) {
  const uint32_t l = idx / n_expert_;
  if (tier_[idx] == ExpertTier::Vram && t != ExpertTier::Vram) --res_count_[l];
  else if (tier_[idx] != ExpertTier::Vram && t == ExpertTier::Vram) ++res_count_[l];
  if (ticks_ > 0 && params_.cooldown_steps > 0 && tier_[idx] != t)
    cooldown_until_[idx] = ticks_ + (uint64_t)params_.cooldown_steps * n_layer_;
  tier_[idx] = t;
}

void ExpertHeat::observe(uint32_t layer, const uint32_t* experts, uint32_t n,
                         const uint32_t* counts) {
  if (layer >= n_layer_) return;
  float* row = heat_.data() + (size_t)layer * n_expert_;
  // Decay first, then credit. The other order would give this dispatch's experts a decayed count,
  // which makes `min_gain = 1.0` mean 1/decay activations rather than one and is the kind of
  // off-by-a-constant that only shows up as a threshold that does not do what it says.
  for (uint32_t e = 0; e < n_expert_; ++e) row[e] *= params_.decay;
  for (uint32_t i = 0; i < n; ++i) {
    if (experts[i] < n_expert_) {
      float credit = counts ? (float)counts[i] : 1.0f;
      if (params_.max_block_credit > 0 && credit > (float)params_.max_block_credit)
        credit = (float)params_.max_block_credit;
      row[experts[i]] += credit;
    }
  }
  ++ticks_;
}

// A FULL SCAN of the whole heat plane, called once per move rather than once per tick — several
// scans a layer dispatch, which looks like something to fold into one pass keeping the top-N of each
// end. It is not: quadrupling the move budget quadruples the scanning and costs nothing, because the
// host ends every layer blocked in a stream drain and this runs inside that slack. Folding the scans
// would buy nothing and cost the property that makes this easy to reason about — that picking N
// pairs is exactly picking one pair N times.
bool ExpertHeat::pick(uint32_t* promote, uint32_t* demote) const {
  if (!warm() || !promote || !demote) return false;
  // One pass, both ends. `busy` excludes an expert with a move in flight — its tier still says
  // where dispatch will look, which is the old tier, so without this the same expert is picked
  // every tick until its move retires and the engine issues the same transfer several times.
  uint32_t hot = UINT32_MAX, cold = UINT32_MAX;
  float hot_h = 0.0f, cold_h = 0.0f;
  uint32_t hot_res = 0, cold_res = 0;             // resident count of the candidate's layer
  for (uint32_t l = 0; l < n_layer_; ++l) {
    const uint32_t rc = res_count_[l];
    const size_t base = (size_t)l * n_expert_;
    for (uint32_t e = 0; e < n_expert_; ++e) {
      const size_t i = base + e;
      if (busy_[i] || ticks_ < cooldown_until_[i]) continue;
      const float h = heat_[i];
      if (tier_[i] == ExpertTier::Pool) {
        // Hotter wins; on an exact tie, the layer with the FEWEST resident experts wins, because
        // that is the layer whose tokens are missing most often.
        if (hot == UINT32_MAX || h > hot_h || (h == hot_h && rc < hot_res)) {
          hot = (uint32_t)i; hot_h = h; hot_res = rc;
        }
      } else if (tier_[i] == ExpertTier::Vram && !protected_[i]) {
        // Colder wins; on an exact tie, take from the layer holding the MOST, which is the one that
        // can spare it. Both halves of this are the same statement about where capacity should sit.
        if (cold == UINT32_MAX || h < cold_h || (h == cold_h && rc > cold_res)) {
          cold = (uint32_t)i; cold_h = h; cold_res = rc;
        }
      }
      // ExpertTier::Ssd is skipped on purpose: reaching one needs a streaming dispatch path that
      // does not exist yet, and a frequency counter is not evidence enough to put an expert there.
      // Both directions are the predictive strategy's business.
    }
  }
  if (hot == UINT32_MAX || cold == UINT32_MAX) return false;
  // BOTH guards, and they answer different failures. The ratio stops a swap that trades one warm
  // expert for another; the additive floor stops the degenerate case where the victim's heat is
  // zero, which every ratio passes and which describes most of the resident set right after warmup.
  if (hot_h < cold_h * params_.hysteresis + params_.min_gain) return false;
  *promote = hot;
  *demote = cold;
  return true;
}

void ExpertHeat::clear_demand() {
  demand_.clear();
  std::fill(demand_seen_.begin(), demand_seen_.end(), 0u);
}

void ExpertHeat::note_missing(uint32_t layer, const uint32_t* experts, uint32_t n, bool append) {
  if (!append) clear_demand();
  if (layer >= n_layer_ || !experts) return;
  if (demand_seen_.size() != heat_.size()) demand_seen_.assign(heat_.size(), 0u);
  const size_t base = (size_t)layer * n_expert_;
  // In the order the dispatch found them, which is ascending by expert id. The engine takes them
  // from the FRONT so the list is served in that order rather than reversed; with a budget below
  // the miss count the choice is arbitrary either way, and being deterministic is what makes an
  // A/B reproducible.
  for (uint32_t i = 0; i < n; ++i) {
    if (experts[i] < n_expert_) {
      const uint32_t idx = (uint32_t)(base + experts[i]);
      if (!demand_seen_[idx]) {
        demand_seen_[idx] = 1u;
        demand_.push_back(idx);
      }
    }
  }
}

bool ExpertHeat::pick_demand(uint32_t* promote, uint32_t* demote) {
  if (!warm() || !promote || !demote || demand_.empty()) return false;
  uint32_t cold = UINT32_MAX, cold_res = 0;
  float cold_h = 0.0f;
  for (uint32_t l = 0; l < n_layer_; ++l) {
    const uint32_t rc = res_count_[l];
    const size_t base = (size_t)l * n_expert_;
    for (uint32_t e = 0; e < n_expert_; ++e) {
      const size_t i = base + e;
      if (busy_[i] || protected_[i] || ticks_ < cooldown_until_[i] || tier_[i] != ExpertTier::Vram)
        continue;
      const float h = heat_[i];
      // Identical to `pick`'s victim half, tie-break included: on an exact tie take from the layer
      // holding the MOST resident experts. At the residencies this runs at `heat == 0.0f` is the
      // common case, so whatever breaks that tie IS the policy for most of a run (§4.1).
      if (cold == UINT32_MAX || h < cold_h || (h == cold_h && rc > cold_res)) {
        cold = (uint32_t)i; cold_h = h; cold_res = rc;
      }
    }
  }
  if (cold == UINT32_MAX) return false;
  size_t consumed = 0;
  bool found = false;
  while (consumed < demand_.size()) {
    const uint32_t idx = demand_[consumed++];
    if (idx < demand_seen_.size()) demand_seen_[idx] = 0u;
    // Not `Vram`: between issue and publish a promoted expert is still read out of the pool, so the
    // dispatch reports it as missing for one more layer. Its `busy` flag is what stops it being
    // admitted twice. Not `Ssd` either — there is no dispatch path for one, so promoting out of it
    // would name bytes nothing can read.
    if (busy_[idx] || ticks_ < cooldown_until_[idx] || tier_[idx] != ExpertTier::Pool) continue;
    // ---- the guard, and it is doing a DIFFERENT job here than in `pick` -------------------------
    //
    // In `pick` the guard stops two ends of a ranking trading places on noise. Here the candidate is
    // not a ranking's guess, it is an expert that was demonstrably needed — so the question is not
    // "is this the right expert" but "is one sighting worth two shards".
    //
    // TWO shards, and that is the whole reason a guard is needed at all. The two PCIe directions
    // share one budget, so the demotion's write-back is charged at exactly the rate a miss is
    // streamed at: a swap costs a promotion plus a demotion against a miss's single shard.
    //
    // Unguarded, this trigger cuts missing experts a dispatch, RAISES total link traffic, and is
    // slower end to end. The guard is what buys the miss reduction without the traffic.
    if (heat_[idx] < cold_h * params_.hysteresis + params_.min_gain) continue;
    *promote = idx;
    *demote = cold;
    found = true;
    break;
  }
  if (consumed > 0) demand_.erase(demand_.begin(), demand_.begin() + (std::ptrdiff_t)consumed);
  return found;
}

bool ExpertHeat::pick_hint(uint32_t idx, uint32_t* demote, const uint32_t* protect_list,
                           uint32_t n_protect) const {
  if (!ready() || !demote || idx >= heat_.size()) return false;
  if (busy_[idx] || ticks_ < cooldown_until_[idx] || tier_[idx] != ExpertTier::Pool) return false;
  uint32_t cold = UINT32_MAX, cold_res = 0;
  float cold_h = 0.0f;
  for (uint32_t l = 0; l < n_layer_; ++l) {
    const uint32_t rc = res_count_[l];
    const size_t base = (size_t)l * n_expert_;
    for (uint32_t e = 0; e < n_expert_; ++e) {
      const size_t i = base + e;
      if (busy_[i] || protected_[i] || ticks_ < cooldown_until_[i] || tier_[i] != ExpertTier::Vram)
        continue;
      if (protect_list && n_protect) {
        bool skip = false;
        for (uint32_t p = 0; p < n_protect; ++p) {
          if (protect_list[p] == (uint32_t)i) { skip = true; break; }
        }
        if (skip) continue;
      }
      const float h = heat_[i];
      if (cold == UINT32_MAX || h < cold_h || (h == cold_h && rc > cold_res)) {
        cold = (uint32_t)i;
        cold_h = h;
        cold_res = rc;
      }
    }
  }
  if (cold == UINT32_MAX) return false;
  *demote = cold;
  return true;
}

void ExpertHeat::resident_per_layer(std::vector<uint32_t>* out) const {
  if (!out) return;
  out->assign(n_layer_, 0u);
  for (uint32_t l = 0; l < n_layer_; ++l) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < n_expert_; ++e)
      if (tier_[(size_t)l * n_expert_ + e] == ExpertTier::Vram) ++n;
    (*out)[l] = n;
  }
}

} // namespace aff
