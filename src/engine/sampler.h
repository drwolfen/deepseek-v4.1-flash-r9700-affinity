// Token sampling: temperature, top-k, top-p, min-p, and repetition penalty.
#pragma once
#include <cstdint>
#include <vector>

namespace aff {

struct SamplerConfig {
  float    temperature = 1.0f;
  uint32_t top_k = 0;          // 0 = disabled
  float    top_p = 1.0f;
  float    min_p = 0.0f;       // keep tokens scoring >= min_p * top
  float    repeat_penalty = 1.0f;
  uint32_t repeat_last_n = 64;
  // DRY. `dry_delta` is SUBTRACTED from the logit of the single token that would continue a
  // detected cycle, so 0 disables the pass. `dry_ngram` is the comparison window: the tail and the
  // tail one period earlier must match for this many tokens. `dry_window` is the longest period
  // considered, and `dry_min_repeats` how many consecutive periods make a loop rather than a
  // coincidence.
  float    dry_delta = 0.0f;
  uint32_t dry_ngram = 8;
  uint32_t dry_window = 512;
  uint32_t dry_min_repeats = 2;
  uint64_t seed = 0;
};

class Sampler {
public:
  explicit Sampler(const SamplerConfig& c) : cfg_(c), rng_(c.seed ? c.seed : 0x243F6A8885A308D3ull) {}
  // Samples from `logits` (modified in place), honouring `history` for the repetition penalty.
  uint32_t sample(std::vector<float>& logits, const std::vector<uint32_t>& history);
  uint32_t argmax(const std::vector<float>& logits) const;
  const SamplerConfig& config() const { return cfg_; }

private:
  SamplerConfig cfg_;
  uint64_t rng_;
  double next_uniform();
};

} // namespace aff
