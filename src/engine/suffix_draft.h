#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aff {

// Incremental suffix-matching and empirical transition-scoring draft index
// (SuffixDecoding, Oliaro et al. arXiv:2411.04975).
class SuffixDraftIndex {
public:
  enum { kExtra = 3, kMaxMinMatch = 12 };

  struct Proposal {
    int n = 0;
    int match_len = 0;
    float score = 0.0f;
    float expect = 0.0f;
  };

  void reset(uint64_t owner = 0) {
    owner_ = owner;
    filled_ = 0;
    min_match_ = 0;
    for (int k = 0; k <= kExtra; ++k) {
      pat_[k].clear();
      roll_h1_[k] = 0;
      roll_h2_[k] = 0;
      drop1_[k] = 1;
      drop2_[k] = 1;
    }
    ctx_total_.clear();
    ctx_tok_.clear();
  }

  uint64_t owner() const { return owner_; }
  size_t filled() const { return filled_; }

  void roll_back_to(size_t pos) {
    if (pos < filled_) {
      reset(owner_);
    }
  }

  Proposal propose(const std::vector<uint32_t>& seq, uint64_t owner,
                   int min_match, int max_draft, float min_score,
                   int32_t* out_tokens) {
    if (min_match <= 0) return Proposal{};
    if (min_match > kMaxMinMatch) min_match = kMaxMinMatch;

    if (owner != owner_ || min_match != min_match_ || seq.size() < filled_) {
      reset(owner);
      min_match_ = min_match;
      init_drops(min_match);
    }

    const size_t n = seq.size();
    if (filled_ < n) {
      index_range(seq, filled_, n);
      filled_ = n;
    }

    if (max_draft <= 0 || out_tokens == nullptr ||
        n <= static_cast<size_t>(min_match_)) {
      return Proposal{};
    }

    size_t match_end = 0;
    int match_len = 0;
    for (int k = kExtra; k >= 0; --k) {
      const int L = min_match_ + k;
      if (n <= static_cast<size_t>(L)) continue;
      const uint64_t h64 = finalize_hash(roll_h1_[k], static_cast<uint64_t>(L));
      const uint32_t tag = finalize_tag(roll_h2_[k], static_cast<uint64_t>(L));
      bool found = false;
      const uint32_t pos = pat_[k].find_pos(h64, tag, &found);
      if (!found) continue;
      const size_t end = static_cast<size_t>(pos) + static_cast<size_t>(L);
      if (end >= n) continue;
      const size_t cur_start = n - static_cast<size_t>(L);
      bool eq = true;
      for (size_t t = 0; t < static_cast<size_t>(L); ++t) {
        if (seq[static_cast<size_t>(pos) + t] != seq[cur_start + t]) {
          eq = false;
          break;
        }
      }
      if (!eq) continue;
      match_end = end;
      match_len = L;
      break;
    }

    if (match_len == 0) return Proposal{};

    const size_t m = static_cast<size_t>(min_match_);
    std::array<uint32_t, kMaxMinMatch> ctx_ring{};
    const size_t ctx_start = n - m;
    for (size_t t = 0; t < m; ++t) {
      ctx_ring[t] = seq[ctx_start + t];
    }

    uint64_t ctx_h1 = roll_h1_[0];
    uint64_t ctx_h2 = roll_h2_[0];
    const uint64_t ctx_salt = static_cast<uint64_t>(m);
    const uint64_t tok_salt = static_cast<uint64_t>(m) + kTokSalt;

    const size_t limit = std::min(static_cast<size_t>(max_draft), n - match_end);
    int n_kept = 0;
    float D = 1.0f;
    float last_D = 0.0f;
    float expect = 0.0f;

    for (size_t j = 0; j < limit; ++j) {
      const uint32_t tok = seq[match_end + j];
      const uint64_t tot_h = finalize_hash(ctx_h1, ctx_salt);
      const uint32_t tot_t = finalize_tag(ctx_h2, ctx_salt);
      const uint32_t tot_cnt = ctx_total_.get_count(tot_h, tot_t);
      if (tot_cnt == 0) break;

      const uint64_t ch1 = ctx_h1 * kBase1 + mix64_1(tok);
      const uint64_t ch2 = ctx_h2 * kBase2 + mix64_2(tok);
      const uint64_t tok_h = finalize_hash(ch1, tok_salt);
      const uint32_t tok_t = finalize_tag(ch2, tok_salt);
      const uint32_t tok_cnt = ctx_tok_.get_count(tok_h, tok_t);
      if (tok_cnt == 0) break;

      const float p = static_cast<float>(tok_cnt) / static_cast<float>(tot_cnt);
      D *= p;
      if (D < min_score) break;

      out_tokens[n_kept++] = static_cast<int32_t>(tok);
      last_D = D;
      expect += D;

      const size_t ring_idx = j % m;
      const uint32_t old_tok = ctx_ring[ring_idx];
      ctx_ring[ring_idx] = tok;
      ctx_h1 = (ctx_h1 - mix64_1(old_tok) * drop1_[0]) * kBase1 + mix64_1(tok);
      ctx_h2 = (ctx_h2 - mix64_2(old_tok) * drop2_[0]) * kBase2 + mix64_2(tok);
    }

    return Proposal{n_kept, match_len, last_D, expect};
  }

private:
  static constexpr uint64_t kBase1 = 0x9e3779b97f4a7c15ULL;
  static constexpr uint64_t kBase2 = 0xbf58476d1ce4e5b9ULL;
  static constexpr uint64_t kTokSalt = 0x9e3779b97f4a7c15ULL;

  static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  static inline uint64_t mix64_1(uint32_t tok) noexcept {
    return splitmix64(static_cast<uint64_t>(tok) + 0x517cc1b727220a95ULL);
  }

  static inline uint64_t mix64_2(uint32_t tok) noexcept {
    return splitmix64(static_cast<uint64_t>(tok) ^ 0xd6e8feb86659fd93ULL);
  }

  static inline uint64_t finalize_hash(uint64_t h1, uint64_t salt) noexcept {
    return splitmix64(h1 ^ salt);
  }

  static inline uint32_t finalize_tag(uint64_t h2, uint64_t salt) noexcept {
    return static_cast<uint32_t>(splitmix64(h2 ^ salt) >> 32);
  }

  struct Slot {
    uint64_t hash = 0;
    uint32_t tag = 0;
    uint32_t val = 0;  // 0 = empty; pat: pos + 1; count: count
  };

  class OpenTable {
  public:
    static constexpr size_t kInitCap = 256;
    static constexpr size_t kMaxRetainCap = 65536;

    void clear() {
      size_ = 0;
      if (slots_.size() > kMaxRetainCap) {
        std::vector<Slot>().swap(slots_);
      } else if (!slots_.empty()) {
        std::fill(slots_.begin(), slots_.end(), Slot{});
      }
    }

    void set_latest(uint64_t hash, uint32_t tag, uint32_t pos) {
      if (pos == UINT32_MAX) return;
      if (slots_.empty()) rehash(kInitCap);
      const uint32_t stored = pos + 1;
      for (;;) {
        const size_t mask = slots_.size() - 1;
        size_t idx = static_cast<size_t>(hash) & mask;
        for (;;) {
          Slot& s = slots_[idx];
          if (s.val == 0) {
            if ((size_ + 1) * 100 > slots_.size() * 65) {
              rehash(slots_.size() * 2);
              break;
            }
            s.hash = hash;
            s.tag = tag;
            s.val = stored;
            ++size_;
            return;
          }
          if (s.hash == hash && s.tag == tag) {
            s.val = stored;
            return;
          }
          idx = (idx + 1) & mask;
        }
      }
    }

    void increment(uint64_t hash, uint32_t tag) {
      if (slots_.empty()) rehash(kInitCap);
      for (;;) {
        const size_t mask = slots_.size() - 1;
        size_t idx = static_cast<size_t>(hash) & mask;
        for (;;) {
          Slot& s = slots_[idx];
          if (s.val == 0) {
            if ((size_ + 1) * 100 > slots_.size() * 65) {
              rehash(slots_.size() * 2);
              break;
            }
            s.hash = hash;
            s.tag = tag;
            s.val = 1;
            ++size_;
            return;
          }
          if (s.hash == hash && s.tag == tag) {
            if (s.val < UINT32_MAX) ++s.val;
            return;
          }
          idx = (idx + 1) & mask;
        }
      }
    }

    uint32_t find_pos(uint64_t hash, uint32_t tag, bool* found) const {
      if (slots_.empty()) {
        *found = false;
        return 0;
      }
      const size_t mask = slots_.size() - 1;
      size_t idx = static_cast<size_t>(hash) & mask;
      for (;;) {
        const Slot& s = slots_[idx];
        if (s.val == 0) {
          *found = false;
          return 0;
        }
        if (s.hash == hash && s.tag == tag) {
          *found = true;
          return s.val - 1;
        }
        idx = (idx + 1) & mask;
      }
    }

    uint32_t get_count(uint64_t hash, uint32_t tag) const {
      if (slots_.empty()) return 0;
      const size_t mask = slots_.size() - 1;
      size_t idx = static_cast<size_t>(hash) & mask;
      for (;;) {
        const Slot& s = slots_[idx];
        if (s.val == 0) return 0;
        if (s.hash == hash && s.tag == tag) return s.val;
        idx = (idx + 1) & mask;
      }
    }

  private:
    void rehash(size_t new_cap) {
      std::vector<Slot> next(new_cap);
      const size_t mask = new_cap - 1;
      for (const Slot& s : slots_) {
        if (s.val == 0) continue;
        size_t idx = static_cast<size_t>(s.hash) & mask;
        while (next[idx].val != 0) idx = (idx + 1) & mask;
        next[idx] = s;
      }
      slots_.swap(next);
    }

    std::vector<Slot> slots_;
    size_t size_ = 0;
  };

  void init_drops(int min_match) {
    for (int k = 0; k <= kExtra; ++k) {
      const int L = min_match + k;
      uint64_t d1 = 1;
      uint64_t d2 = 1;
      for (int i = 1; i < L; ++i) {
        d1 *= kBase1;
        d2 *= kBase2;
      }
      drop1_[k] = d1;
      drop2_[k] = d2;
    }
  }

  void index_range(const std::vector<uint32_t>& seq, size_t start, size_t end) {
    const uint64_t m = static_cast<uint64_t>(min_match_);
    const uint64_t ctx_salt = m;
    const uint64_t tok_salt = m + kTokSalt;
    for (size_t i = start; i < end; ++i) {
      const uint32_t tok = seq[i];
      const uint64_t t1 = mix64_1(tok);
      const uint64_t t2 = mix64_2(tok);

      if (i >= static_cast<size_t>(min_match_)) {
        const uint64_t ctx_h1 = roll_h1_[0];
        const uint64_t ctx_h2 = roll_h2_[0];
        ctx_total_.increment(finalize_hash(ctx_h1, ctx_salt),
                             finalize_tag(ctx_h2, ctx_salt));
        const uint64_t ch1 = ctx_h1 * kBase1 + t1;
        const uint64_t ch2 = ctx_h2 * kBase2 + t2;
        ctx_tok_.increment(finalize_hash(ch1, tok_salt),
                           finalize_tag(ch2, tok_salt));
      }

      for (int k = 0; k <= kExtra; ++k) {
        const size_t L = static_cast<size_t>(min_match_ + k);
        if (i >= L) {
          const size_t pos = i - L;
          if (pos < static_cast<size_t>(UINT32_MAX)) {
            const uint64_t h64 = finalize_hash(roll_h1_[k], static_cast<uint64_t>(L));
            const uint32_t tag = finalize_tag(roll_h2_[k], static_cast<uint64_t>(L));
            pat_[k].set_latest(h64, tag, static_cast<uint32_t>(pos));
          }
          roll_h1_[k] -= mix64_1(seq[pos]) * drop1_[k];
          roll_h2_[k] -= mix64_2(seq[pos]) * drop2_[k];
        }
        roll_h1_[k] = roll_h1_[k] * kBase1 + t1;
        roll_h2_[k] = roll_h2_[k] * kBase2 + t2;
      }
    }
  }

  uint64_t owner_ = 0;
  size_t filled_ = 0;
  int min_match_ = 0;

  OpenTable pat_[kExtra + 1];
  OpenTable ctx_total_;
  OpenTable ctx_tok_;

  uint64_t roll_h1_[kExtra + 1]{};
  uint64_t roll_h2_[kExtra + 1]{};
  uint64_t drop1_[kExtra + 1]{};
  uint64_t drop2_[kExtra + 1]{};
};

// Utility- and acceptance-aware gate controlling whether SuffixDecoding preempts
// the GPU DSpark drafter on a block. Tracks empirical suffix acceptance EMA and
// exponential miss cooldown so prose streams stay on DSpark while code/structured
// continuations still trigger suffix drafting immediately on long exact matches.
struct SuffixGate {
  float acc_ema = 0.90f;
  uint32_t miss_streak = 0;
  uint32_t cooldown = 0;
  uint32_t steps = 0;

  void reset() noexcept {
    acc_ema = 0.90f;
    miss_streak = 0;
    cooldown = 0;
    steps = 0;
  }

  bool should_preempt(const SuffixDraftIndex::Proposal& prop, int min_match, float min_expect,
                      bool dspark_active, float dspark_exp_acc, float dspark_cost_ms,
                      float sfx_cost_ms, float min_gain = 1.05f) noexcept {
    if (prop.n < min_match || prop.expect < min_expect) return false;
    if (dspark_active) {
      // Never let bare 4-token (min_match) or cold 5-token collisions preempt GPU DSpark;
      // require a strong exact anchor (>= min_match + 2) to enter suffix mode, or
      // >= min_match + 1 once a proven high-acceptance streak is active.
      const int req_match = (steps > 0 && miss_streak == 0 && acc_ema >= 0.80f)
                                ? (min_match + 1)
                                : (min_match + 2);
      if (prop.match_len < req_match) {
        if (cooldown > 0) --cooldown;
        return false;
      }
    }
    if (cooldown > 0) {
      if (prop.match_len >= min_match + 2 && prop.n >= min_match + 2 && acc_ema >= 0.65f) {
        cooldown = 0;
      } else {
        --cooldown;
        return false;
      }
    }
    const float eff_expect = prop.expect * acc_ema;
    if (eff_expect < min_expect) return false;
    if (!dspark_active) return true;
    if (eff_expect <= dspark_exp_acc) return false;
    const float sfx_u = (1.0f + eff_expect) / std::max(sfx_cost_ms, 1.0f);
    const float dsp_u = (1.0f + std::max(dspark_exp_acc, 0.0f)) / std::max(dspark_cost_ms, 1.0f);
    return sfx_u >= dsp_u * min_gain;
  }

  void observe(uint32_t nd, uint32_t k) noexcept {
    if (nd == 0) return;
    const float rate = static_cast<float>(k) / static_cast<float>(nd);
    acc_ema = 0.55f * acc_ema + 0.45f * rate;
    ++steps;
    if (k < 2u || (k * 2u < nd)) {
      ++miss_streak;
      cooldown = std::min<uint32_t>(16u, 2u << std::min<uint32_t>(miss_streak - 1u, 3u));
    } else {
      miss_streak = 0;
      cooldown = 0;
    }
  }
};

}  // namespace aff
