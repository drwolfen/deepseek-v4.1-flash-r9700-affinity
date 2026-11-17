#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace aff {

// Thinking-budget configuration for one request. When the remaining
// generation window falls to hard_limit_remaining while the model is still
// thinking, or when repetition/executive-freeze churn is detected,
// close_token_ids replace the next emitted tokens in order so reusable KV
// state sees the replacement.
struct BudgetHook {
  // The sequence the budget injects (e.g. "\n</think>\n\n"). Empty disables the hook.
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
    }
    if (hook_ && !hook_->close_token_ids.empty() &&
        hook_->freeze_enabled && hook_->freeze_window > 0) {
      freeze_ring_.assign((size_t)hook_->freeze_window, 0);
      freeze_counts_.reserve((size_t)hook_->freeze_window * 2);
    }
  }
  ThinkingBudget(BudgetHook&&, int) = delete;

  int remaining() const { return remaining_; }
  int thought_tokens() const { return thought_tokens_; }
  bool naturally_closed() const { return closed_; }
  bool forced_close() const { return forced_; }
  bool force_closed() const { return forced_; }
  bool freeze_triggered() const { return freeze_triggered_; }
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
           (looping_ || freeze_triggered_ ||
            remaining_ - hook_->hard_limit_remaining <= max_block_tokens);
  }

  bool looping() const { return looping_ && !settled(); }
  void arm_loop_close() {
    if (looping() || freeze_triggered_) loop_armed_ = true;
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
      return force_close(loop_armed_ ? (freeze_triggered_ ? "freeze_churn" : "loop_repeat")
                                     : "hard_limit");
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

  // Counts each loop_span-token window of thinking by a rolling hash.
  bool repeats(int32_t token) {
    const size_t span = loop_ring_.size();
    const size_t at = (size_t)(loop_seen_ % span);
    if (loop_seen_ >= (long long)span)
      loop_hash_ -= (uint64_t)loop_ring_[at] * loop_drop_;
    loop_hash_ = loop_hash_ * kLoopBase + (uint64_t)(uint32_t)token;
    loop_ring_[at] = token;
    if (++loop_seen_ < (long long)span) return false;
    return ++loop_windows_[loop_hash_] >= hook_->loop_repeats;
  }

  bool is_nl_token(int32_t token) const {
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
      auto it = freeze_counts_.find(old);
      if (it != freeze_counts_.end()) {
        if (--(it->second) <= 0) freeze_counts_.erase(it);
      }
    }
    freeze_ring_[at] = token;
    ++freeze_seen_;
    if (is_nl_token(token)) ++freeze_nl_count_;
    ++freeze_counts_[token];
    return thought_tokens_ >= hook_->freeze_min_tokens &&
           freeze_seen_ >= w &&
           freeze_nl_count_ >= hook_->freeze_nl_thresh &&
           (int)freeze_counts_.size() <= hook_->freeze_distinct_max;
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
  const char* close_reason_ = "";
  std::size_t inject_pos_ = 0;
  std::vector<int32_t> marker_tail_;
  static constexpr uint64_t kLoopBase = 0x9E3779B97F4A7C15ull;
  std::vector<int32_t> loop_ring_;
  std::unordered_map<uint64_t, int> loop_windows_;
  uint64_t loop_hash_ = 0;
  uint64_t loop_drop_ = 0;
  long long loop_seen_ = 0;
  bool looping_ = false;
  bool loop_armed_ = false;
  std::vector<int32_t> freeze_ring_;
  std::unordered_map<int32_t, int> freeze_counts_;
  size_t freeze_seen_ = 0;
  int freeze_nl_count_ = 0;
};

}  // namespace aff
