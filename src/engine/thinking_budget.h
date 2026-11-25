#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aff {

// Thinking-budget configuration for one request. When the remaining
// generation window falls to hard_limit_remaining while the model is still
// thinking, or when repetition/executive-freeze churn is detected,
// close_token_ids replace the next emitted tokens in order so reusable KV
// state sees the replacement.
struct BudgetHook {
  // The sequence the budget injects (e.g. "</think>\n\n"). Empty disables the hook.
  std::vector<int32_t> close_token_ids;
  // How the model itself ends thinking, e.g. "</think>". A natural close
  // is recognized only by this marker, never by close_token_ids.
  std::vector<int32_t> marker_token_ids;
  int hard_limit_remaining = 0;
  // Thinking that goes round in circles: once one span of loop_span tokens has
  // appeared loop_repeats times, force-close <think>. 0 or <=1 repeats disables.
  int loop_span = 64;
  int loop_repeats = 0;
  // Executive-Freeze Churn Breaker: after freeze_min_tokens inside <think>, if the
  // trailing freeze_window tokens contain >= freeze_nl_thresh newline-group tokens
  // and <= freeze_distinct_max distinct token IDs, force-close <think>.
  bool freeze_enabled = true;
  int freeze_min_tokens = 512;
  int freeze_window = 256;
  int freeze_nl_thresh = 64;
  int freeze_distinct_max = 36;
  std::vector<int32_t> freeze_nl_token_ids;
  // Optional pointer to a pre-sorted vocabulary-wide list of newline-bearing token IDs; avoids
  // copying large per-vocab vectors per request and enables O(log V) binary_search.
  const std::vector<int32_t>* freeze_nl_sorted = nullptr;
  // Short-scale repeat detector (0 / <=1 disables so legacy defaults stay unchanged).
  int short_loop_span = 0;
  int short_loop_repeats = 0;
  int short_loop_window = 1024;
  // Trailing bigram diversity detector (0 disables so legacy defaults stay unchanged).
  int novelty_window = 0;
  int novelty_min_tokens = 1024;
  float novelty_min_ratio = 0.28f;
  // Short-window bigram diversity detector for rapid mantra loops (<512 tokens; 0 disables).
  int short_novelty_window = 0;
  int short_novelty_min_tokens = 160;
  float short_novelty_min_ratio = 0.20f;
};

// The reply reserve comes out of the request's output cap. When the cap is at
// most twice the configured reserve, the reply gets half of it and thinking
// the other half; the full reserve would leave no room to think at all.
inline int reply_reserve_within(int reserve, int output_cap) {
  const int half = output_cap / 2;
  return reserve < half ? reserve : half;
}

// One controller per generation, shared across speculative blocks and tails.
// The decoder supplies ACCEPTED output candidates exactly once, before
// committing them to history/KV or emitting them.
class ThinkingBudget {
public:
  struct Decision {
    int32_t token;
    // End a speculative block here, discard its suffix and repair KV.
    // Starting a forced close cuts even when the candidate already equals
    // close[0]. Later matching forced drafts can be retained.
    bool cut;
  };

  ThinkingBudget() = default;
  ThinkingBudget(const BudgetHook& hook, int n_gen)
      : ThinkingBudget(&hook, n_gen) {}
  ThinkingBudget(const BudgetHook* hook, int n_gen)
      : hook_(hook), remaining_(n_gen) {
    if (hook_ && !hook_->close_token_ids.empty() &&
        hook_->marker_token_ids.size() > 1) {
      marker_tail_.reserve(hook_->marker_token_ids.size() - 1);
    }
    if (hook_ && !hook_->close_token_ids.empty() &&
        hook_->loop_repeats > 1 && hook_->loop_span > 0) {
      loop_ring_.assign((size_t)hook_->loop_span, 0);
      loop_drop_ = 1;
      for (int i = 1; i < hook_->loop_span; ++i) loop_drop_ *= kLoopBase;
      loop_windows_.reserve(128);
    }
    if (hook_ && !hook_->close_token_ids.empty() &&
        hook_->freeze_enabled && hook_->freeze_window > 0) {
      freeze_ring_.assign((size_t)hook_->freeze_window, 0);
      freeze_counts_.reserve((size_t)hook_->freeze_window);
    }
    if (hook_ && !hook_->close_token_ids.empty() &&
        hook_->short_loop_span > 0 && hook_->short_loop_repeats > 1 &&
        hook_->short_loop_window > 0) {
      short_token_ring_.assign((size_t)hook_->short_loop_span, 0);
      short_loop_ring_.assign((size_t)hook_->short_loop_window, 0);
      short_loop_drop_ = 1;
      for (int i = 1; i < hook_->short_loop_span; ++i) short_loop_drop_ *= kLoopBase;
      short_loop_counts_.reserve((size_t)hook_->short_loop_window);
    }
    if (hook_ && !hook_->close_token_ids.empty() && hook_->novelty_window > 0) {
      novelty_ring_.assign((size_t)hook_->novelty_window, 0);
      novelty_counts_.reserve((size_t)hook_->novelty_window);
    }
    if (hook_ && !hook_->close_token_ids.empty() && hook_->short_novelty_window > 0) {
      short_novelty_ring_.assign((size_t)hook_->short_novelty_window, 0);
      short_novelty_counts_.reserve((size_t)hook_->short_novelty_window);
    }
  }
  ThinkingBudget(BudgetHook&&, int) = delete;

  int remaining() const { return remaining_; }
  int thought_tokens() const { return thought_tokens_; }
  bool naturally_closed() const { return closed_; }
  bool forced_close() const { return forced_; }
  bool force_closed() const { return forced_; }
  bool freeze_triggered() const { return freeze_triggered_; }
  bool short_loop_triggered() const { return short_loop_triggered_; }
  bool novelty_triggered() const { return novelty_triggered_; }
  bool short_novelty_triggered() const { return short_novelty_triggered_; }
  const char* close_reason() const { return close_reason_; }
  bool injecting() const {
    return forced_ && hook_ && inject_pos_ < hook_->close_token_ids.size();
  }
  bool settled() const {
    return !hook_ || hook_->close_token_ids.empty() || closed_ ||
           (forced_ && !injecting());
  }

  bool needs_ar_tail(int max_block_tokens) const {
    return !settled() &&
           (looping_ || freeze_triggered_ || short_loop_triggered_ || novelty_triggered_ ||
            short_novelty_triggered_ ||
            remaining_ - hook_->hard_limit_remaining <= max_block_tokens);
  }

  bool looping() const { return looping_ && !settled(); }
  void arm_loop_close() {
    if (looping() || freeze_triggered_ || short_loop_triggered_ || novelty_triggered_ ||
        short_novelty_triggered_)
      loop_armed_ = true;
  }

  std::size_t pending_close_tokens() const {
    return injecting() ? hook_->close_token_ids.size() - inject_pos_ : 0;
  }
  int32_t pending_close_token(std::size_t offset) const {
    return hook_->close_token_ids[inject_pos_ + offset];
  }

  // Includes this candidate in output accounting, even after the hook has
  // settled. The returned token, not the original candidate, is the token
  // the decoder must emit and use for its next forward pass.
  Decision apply(int32_t candidate) {
    const int remaining = remaining_--;
    if (settled()) return {candidate, false};
    if (forced_) {
      const int32_t token = hook_->close_token_ids[inject_pos_++];
      return {token, token != candidate};
    }
    if (completes_marker(candidate)) {
      closed_ = true;
      marker_tail_.clear();
      return {candidate, false};
    }
    if (remaining <= hook_->hard_limit_remaining || loop_armed_) {
      const char* reason = "hard_limit";
      if (loop_armed_) {
        if (freeze_triggered_) reason = "freeze_churn";
        else if (short_loop_triggered_) reason = "short_loop";
        else if (short_novelty_triggered_) reason = "short_low_novelty";
        else if (novelty_triggered_) reason = "low_novelty";
        else reason = "loop_repeat";
      }
      return force_close(reason);
    }
    remember(candidate);
    ++thought_tokens_;
    if (!loop_ring_.empty() && repeats(candidate)) {
      looping_ = true;
      return force_close("loop_repeat");
    }
    if (!freeze_ring_.empty() && freeze_check(candidate)) {
      freeze_triggered_ = true;
      return force_close("freeze_churn");
    }
    if (!short_loop_ring_.empty() && short_loop_check(candidate)) {
      short_loop_triggered_ = true;
      return force_close("short_loop");
    }
    if (!short_novelty_ring_.empty() && short_novelty_check(candidate)) {
      short_novelty_triggered_ = true;
      return force_close("short_low_novelty");
    }
    if (!novelty_ring_.empty() && novelty_check(candidate)) {
      novelty_triggered_ = true;
      return force_close("low_novelty");
    }
    return {candidate, false};
  }

  // Walk an accepted block in emit order. A cut replaces its last retained
  // token and excludes the suffix from both output and budget accounting.
  int apply_block(int32_t* tokens, int n) {
    for (int i = 0; i < n; ++i) {
      if (settled()) {
        remaining_ -= n - i;
        return n;
      }
      const Decision decision = apply(tokens[i]);
      tokens[i] = decision.token;
      if (decision.cut) return i + 1;
    }
    return n;
  }

private:
  Decision force_close(const char* reason) {
    forced_ = true;
    inject_pos_ = 1;
    close_reason_ = reason;
    marker_tail_.clear();
    return {hook_->close_token_ids.front(), true};
  }

  bool completes_marker(int32_t token) const {
    const auto& marker = hook_->marker_token_ids;
    if (marker.empty() || token != marker.back()) return false;
    const std::size_t need = marker.size() - 1;
    if (marker_tail_.size() < need) return false;
    for (std::size_t i = 0; i < need; ++i) {
      if (marker_tail_[i] != marker[i]) return false;
    }
    return true;
  }

  // Open-addressed flat counter map with backward-shift deletion (zero per-token heap churn).
  template <typename Key>
  class FlatCountMap {
  public:
    void reserve(size_t expected_items) {
      size_t want = 16;
      const size_t min_slots = std::max<size_t>(16, expected_items * 2);
      while (want < min_slots) want <<= 1;
      if (want > slots_.size()) rehash(want);
    }

    int inc(Key key) {
      if (slots_.empty()) rehash(64);
      if ((size_ + 1) * 10 > slots_.size() * 6) rehash(slots_.size() * 2);
      const size_t mask = slots_.size() - 1;
      size_t idx = hash_key(key) & mask;
      for (;;) {
        Slot& s = slots_[idx];
        if (s.count == 0) {
          s.key = key;
          s.count = 1;
          ++size_;
          return 1;
        }
        if (s.key == key) return ++s.count;
        idx = (idx + 1) & mask;
      }
    }

    void dec_erase(Key key) {
      if (slots_.empty()) return;
      const size_t mask = slots_.size() - 1;
      size_t idx = hash_key(key) & mask;
      for (;;) {
        Slot& s = slots_[idx];
        if (s.count == 0) return;
        if (s.key == key) {
          if (--s.count > 0) return;
          // Backward-shift deletion preserves probe chains without tombstones.
          size_t empty = idx;
          size_t j = (idx + 1) & mask;
          while (slots_[j].count > 0) {
            const size_t natural = hash_key(slots_[j].key) & mask;
            if (((j - natural) & mask) >= ((j - empty) & mask)) {
              slots_[empty] = slots_[j];
              empty = j;
            }
            j = (j + 1) & mask;
          }
          slots_[empty].count = 0;
          --size_;
          return;
        }
        idx = (idx + 1) & mask;
      }
    }

    size_t size() const noexcept { return size_; }

  private:
    struct Slot {
      Key key{};
      int count = 0;  // 0 = empty
    };

    static inline size_t hash_key(uint64_t x) noexcept {
      x += 0x9e3779b97f4a7c15ULL;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return static_cast<size_t>(x ^ (x >> 31));
    }

    static inline size_t hash_key(int32_t k) noexcept {
      return hash_key(static_cast<uint64_t>(static_cast<uint32_t>(k)));
    }

    void rehash(size_t new_cap) {
      std::vector<Slot> next(new_cap);
      const size_t mask = new_cap - 1;
      for (const Slot& s : slots_) {
        if (s.count == 0) continue;
        size_t idx = hash_key(s.key) & mask;
        while (next[idx].count > 0) idx = (idx + 1) & mask;
        next[idx] = s;
      }
      slots_.swap(next);
    }

    std::vector<Slot> slots_;
    size_t size_ = 0;
  };

  // Counts each loop_span-token window of thinking by a rolling hash.
  bool repeats(int32_t token) {
    const size_t span = loop_ring_.size();
    const size_t at = (size_t)(loop_seen_ % span);
    if (loop_seen_ >= (long long)span)
      loop_hash_ -= (uint64_t)loop_ring_[at] * loop_drop_;
    loop_hash_ = loop_hash_ * kLoopBase + (uint64_t)(uint32_t)token;
    loop_ring_[at] = token;
    if (++loop_seen_ < (long long)span) return false;
    return loop_windows_.inc(loop_hash_) >= hook_->loop_repeats;
  }

  bool is_nl_token(int32_t token) const {
    if (hook_->freeze_nl_sorted && !hook_->freeze_nl_sorted->empty()) {
      return std::binary_search(hook_->freeze_nl_sorted->begin(),
                                hook_->freeze_nl_sorted->end(), token);
    }
    return std::find(hook_->freeze_nl_token_ids.begin(),
                     hook_->freeze_nl_token_ids.end(),
                     token) != hook_->freeze_nl_token_ids.end();
  }

  bool freeze_check(int32_t token) {
    const size_t w = freeze_ring_.size();
    const size_t at = freeze_seen_ % w;
    if (freeze_seen_ >= w) {
      const int32_t old = freeze_ring_[at];
      if (is_nl_token(old)) --freeze_nl_count_;
      freeze_counts_.dec_erase(old);
    }
    freeze_ring_[at] = token;
    ++freeze_seen_;
    if (is_nl_token(token)) ++freeze_nl_count_;
    freeze_counts_.inc(token);
    return thought_tokens_ >= hook_->freeze_min_tokens &&
           freeze_seen_ >= w &&
           freeze_nl_count_ >= hook_->freeze_nl_thresh &&
           (int)freeze_counts_.size() <= hook_->freeze_distinct_max;
  }

  // Counts finished short-span hashes inside a bounded sliding window.
  bool short_loop_check(int32_t token) {
    const size_t span = short_token_ring_.size();
    const size_t sat = short_token_seen_ % span;
    if (short_token_seen_ >= span)
      short_token_hash_ -= (uint64_t)(uint32_t)short_token_ring_[sat] * short_loop_drop_;
    short_token_hash_ = short_token_hash_ * kLoopBase + (uint64_t)(uint32_t)token;
    short_token_ring_[sat] = token;
    if (++short_token_seen_ < span) return false;

    const size_t w = short_loop_ring_.size();
    const size_t wat = short_loop_seen_ % w;
    if (short_loop_seen_ >= w) {
      short_loop_counts_.dec_erase(short_loop_ring_[wat]);
    }
    short_loop_ring_[wat] = short_token_hash_;
    ++short_loop_seen_;
    return short_loop_counts_.inc(short_token_hash_) >= hook_->short_loop_repeats;
  }

  // Tracks distinct token bigrams over a sliding ring buffer.
  static bool update_bigram_novelty(int32_t token, std::vector<uint64_t>& ring,
                                    FlatCountMap<uint64_t>& counts, size_t& seen,
                                    int32_t& prev_tok, bool& has_prev, int min_tokens,
                                    float min_ratio, int thought_tokens) {
    if (!has_prev) {
      prev_tok = token;
      has_prev = true;
      return false;
    }
    const uint64_t bg = ((uint64_t)(uint32_t)prev_tok << 32) | (uint32_t)token;
    prev_tok = token;
    const size_t w = ring.size();
    const size_t at = seen % w;
    if (seen >= w) {
      counts.dec_erase(ring[at]);
    }
    ring[at] = bg;
    ++seen;
    counts.inc(bg);
    return thought_tokens >= min_tokens &&
           seen >= w &&
           (float)counts.size() < min_ratio * (float)w;
  }

  bool novelty_check(int32_t token) {
    return update_bigram_novelty(token, novelty_ring_, novelty_counts_, novelty_seen_, prev_tok_,
                                 has_prev_tok_, hook_->novelty_min_tokens, hook_->novelty_min_ratio,
                                 thought_tokens_);
  }

  bool short_novelty_check(int32_t token) {
    return update_bigram_novelty(token, short_novelty_ring_, short_novelty_counts_,
                                 short_novelty_seen_, short_prev_tok_, short_has_prev_tok_,
                                 hook_->short_novelty_min_tokens, hook_->short_novelty_min_ratio,
                                 thought_tokens_);
  }

  void remember(int32_t token) {
    const auto size = hook_->marker_token_ids.size();
    if (size < 2) return;
    if (marker_tail_.size() == size - 1) {
      marker_tail_.erase(marker_tail_.begin());
    }
    marker_tail_.push_back(token);
  }

  const BudgetHook* hook_ = nullptr;
  int remaining_ = 0;
  int thought_tokens_ = 0;
  bool closed_ = false;
  bool forced_ = false;
  bool freeze_triggered_ = false;
  bool short_loop_triggered_ = false;
  bool novelty_triggered_ = false;
  bool short_novelty_triggered_ = false;
  const char* close_reason_ = "";
  std::size_t inject_pos_ = 0;
  std::vector<int32_t> marker_tail_;
  static constexpr uint64_t kLoopBase = 0x9E3779B97F4A7C15ull;
  std::vector<int32_t> loop_ring_;
  FlatCountMap<uint64_t> loop_windows_;
  uint64_t loop_hash_ = 0;
  uint64_t loop_drop_ = 0;
  long long loop_seen_ = 0;
  bool looping_ = false;
  bool loop_armed_ = false;
  std::vector<int32_t> freeze_ring_;
  FlatCountMap<int32_t> freeze_counts_;
  size_t freeze_seen_ = 0;
  int freeze_nl_count_ = 0;
  std::vector<int32_t> short_token_ring_;
  std::vector<uint64_t> short_loop_ring_;
  FlatCountMap<uint64_t> short_loop_counts_;
  uint64_t short_token_hash_ = 0;
  uint64_t short_loop_drop_ = 0;
  size_t short_token_seen_ = 0;
  size_t short_loop_seen_ = 0;
  std::vector<uint64_t> novelty_ring_;
  FlatCountMap<uint64_t> novelty_counts_;
  size_t novelty_seen_ = 0;
  int32_t prev_tok_ = 0;
  bool has_prev_tok_ = false;
  std::vector<uint64_t> short_novelty_ring_;
  FlatCountMap<uint64_t> short_novelty_counts_;
  size_t short_novelty_seen_ = 0;
  int32_t short_prev_tok_ = 0;
  bool short_has_prev_tok_ = false;
};

}  // namespace aff
