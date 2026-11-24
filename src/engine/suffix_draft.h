#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#ifdef AFF_WITH_RUST
struct AffSam;
extern "C" {
AffSam* aff_sam_new();
void aff_sam_free(AffSam* sam);
void aff_sam_clear(AffSam* sam);
size_t aff_sam_len(const AffSam* sam);
void aff_sam_push(AffSam* sam, uint32_t tok);
void aff_sam_push_slice(AffSam* sam, const uint32_t* ptr, size_t len);
bool aff_sam_rollback_to(AffSam* sam, size_t target_len);
size_t aff_sam_match_len(const AffSam* sam);
bool aff_sam_propose(const AffSam* sam, const uint32_t* ctx_ptr, size_t ctx_len,
                     size_t max_draft, size_t min_match, float min_score,
                     uint32_t* out_toks, size_t* out_len, float* out_score,
                     size_t* out_match);
bool aff_sam_propose_ex(const AffSam* sam, const uint32_t* ctx_ptr, size_t ctx_len,
                        size_t max_draft, size_t min_match, float min_score,
                        uint32_t* out_toks, size_t* out_len, float* out_score,
                        float* out_expect, size_t* out_match);
bool aff_sam_contains_ngram(const AffSam* sam, const uint32_t* ngram_ptr, size_t ngram_len);
size_t aff_sam_first_endpos(const AffSam* sam, const uint32_t* ngram_ptr, size_t ngram_len);
bool aff_sam_matches_prefix(const AffSam* sam, const uint32_t* ptr, size_t len);
}  // extern "C"
#endif

namespace aff {

inline bool rust_sam_env_enabled() noexcept {
#ifdef AFF_WITH_RUST
  const char* v = std::getenv("AFF_RUST_SAM");
  return !(v && *v && std::strcmp(v, "0") == 0);
#else
  return false;
#endif
}

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

  SuffixDraftIndex() : use_rust_(rust_sam_env_enabled()) {}

  explicit SuffixDraftIndex(bool force_rust) : explicit_mode_(true) {
#ifdef AFF_WITH_RUST
    use_rust_ = force_rust;
#else
    (void)force_rust;
    use_rust_ = false;
#endif
  }

  ~SuffixDraftIndex() {
#ifdef AFF_WITH_RUST
    if (sam_) {
      aff_sam_free(sam_);
      sam_ = nullptr;
    }
#endif
  }

  SuffixDraftIndex(const SuffixDraftIndex&) = delete;
  SuffixDraftIndex& operator=(const SuffixDraftIndex&) = delete;

  SuffixDraftIndex(SuffixDraftIndex&& other) noexcept
      : owner_(other.owner_),
        filled_(other.filled_),
        min_match_(other.min_match_),
        last_match_len_(other.last_match_len_),
        use_rust_(other.use_rust_),
        explicit_mode_(other.explicit_mode_),
#ifdef AFF_WITH_RUST
        sam_(other.sam_),
#endif
        ctx_total_(std::move(other.ctx_total_)),
        ctx_tok_(std::move(other.ctx_tok_)),
        cpp_seq_(std::move(other.cpp_seq_)) {
    for (int k = 0; k <= kExtra; ++k) {
      pat_[k] = std::move(other.pat_[k]);
      roll_h1_[k] = other.roll_h1_[k];
      roll_h2_[k] = other.roll_h2_[k];
      drop1_[k] = other.drop1_[k];
      drop2_[k] = other.drop2_[k];
      other.roll_h1_[k] = 0;
      other.roll_h2_[k] = 0;
      other.drop1_[k] = 1;
      other.drop2_[k] = 1;
    }
#ifdef AFF_WITH_RUST
    other.sam_ = nullptr;
#endif
    other.owner_ = 0;
    other.filled_ = 0;
    other.min_match_ = 0;
    other.last_match_len_ = 0;
  }

  SuffixDraftIndex& operator=(SuffixDraftIndex&& other) noexcept {
    if (this == &other) return *this;
#ifdef AFF_WITH_RUST
    if (sam_) aff_sam_free(sam_);
    sam_ = other.sam_;
    other.sam_ = nullptr;
#endif
    owner_ = other.owner_;
    filled_ = other.filled_;
    min_match_ = other.min_match_;
    last_match_len_ = other.last_match_len_;
    use_rust_ = other.use_rust_;
    explicit_mode_ = other.explicit_mode_;
    for (int k = 0; k <= kExtra; ++k) {
      pat_[k] = std::move(other.pat_[k]);
      roll_h1_[k] = other.roll_h1_[k];
      roll_h2_[k] = other.roll_h2_[k];
      drop1_[k] = other.drop1_[k];
      drop2_[k] = other.drop2_[k];
      other.roll_h1_[k] = 0;
      other.roll_h2_[k] = 0;
      other.drop1_[k] = 1;
      other.drop2_[k] = 1;
    }
    ctx_total_ = std::move(other.ctx_total_);
    ctx_tok_ = std::move(other.ctx_tok_);
    cpp_seq_ = std::move(other.cpp_seq_);
    other.owner_ = 0;
    other.filled_ = 0;
    other.min_match_ = 0;
    other.last_match_len_ = 0;
    return *this;
  }

  bool rust_sam_active() const noexcept {
#ifdef AFF_WITH_RUST
    return use_rust_;
#else
    return false;
#endif
  }

  void reset(uint64_t owner = 0) {
    if (!explicit_mode_) use_rust_ = rust_sam_env_enabled();
    owner_ = owner;
    filled_ = 0;
    min_match_ = 0;
    last_match_len_ = 0;
#ifdef AFF_WITH_RUST
    if (sam_) {
      if (use_rust_) {
        aff_sam_clear(sam_);
      } else {
        aff_sam_free(sam_);
        sam_ = nullptr;
      }
    }
#endif
    for (int k = 0; k <= kExtra; ++k) {
      pat_[k].clear();
      roll_h1_[k] = 0;
      roll_h2_[k] = 0;
      drop1_[k] = 1;
      drop2_[k] = 1;
    }
    ctx_total_.clear();
    ctx_tok_.clear();
    cpp_seq_.clear();
  }

  uint64_t owner() const noexcept { return owner_; }
  size_t filled() const noexcept { return filled_; }

  size_t match_len() const noexcept {
#ifdef AFF_WITH_RUST
    if (use_rust_ && sam_) return aff_sam_match_len(sam_);
#endif
    return last_match_len_;
  }

  // Destructive reset helper preserved for backward compatibility when prefix tokens change.
  void roll_back_to(size_t pos) {
    if (pos < filled_) {
      reset(owner_);
    }
  }

  // Non-destructive rollback reverting state mutations since `pos` while keeping prefix [0..pos).
  void rollback_to(size_t pos) {
    if (pos >= filled_) return;
    if (pos == 0) {
      reset(owner_);
      return;
    }
#ifdef AFF_WITH_RUST
    if (use_rust_) {
      if (sam_ && aff_sam_rollback_to(sam_, pos)) {
        filled_ = pos;
        return;
      }
      reset(owner_);
      return;
    }
#endif
    if (cpp_seq_.size() >= pos) {
      std::vector<uint32_t> saved(cpp_seq_.begin(), cpp_seq_.begin() + static_cast<std::ptrdiff_t>(pos));
      const int saved_mm = min_match_ > 0 ? min_match_ : 4;
      reset(owner_);
      min_match_ = saved_mm;
      init_drops(saved_mm);
      cpp_seq_ = std::move(saved);
      index_range_ptr(cpp_seq_.data(), 0, pos);
      filled_ = pos;
    } else {
      reset(owner_);
    }
  }

  void append_delta(const uint32_t* tokens, size_t new_len) {
    if (new_len < filled_) {
      rollback_to(new_len);
      return;
    }
    if (new_len == filled_ || tokens == nullptr) return;
#ifdef AFF_WITH_RUST
    if (use_rust_) {
      if (!sam_) sam_ = aff_sam_new();
      if (sam_) {
        aff_sam_push_slice(sam_, tokens + filled_, new_len - filled_);
        filled_ = new_len;
      }
      return;
    }
#endif
    if (min_match_ == 0) {
      min_match_ = 4;
      init_drops(min_match_);
    }
    cpp_seq_.insert(cpp_seq_.end(), tokens + filled_, tokens + new_len);
    index_range_ptr(cpp_seq_.data(), filled_, new_len);
    filled_ = new_len;
  }

  bool contains_ngram(const uint32_t* ptr, size_t len) const {
    if (!ptr || len == 0 || filled_ < len) return false;
#ifdef AFF_WITH_RUST
    if (use_rust_) {
      return sam_ != nullptr && aff_sam_contains_ngram(sam_, ptr, len);
    }
#endif
    return std::search(cpp_seq_.begin(), cpp_seq_.end(), ptr, ptr + len) != cpp_seq_.end();
  }

  // Returns true iff `ptr[0..len)` occurred with 1-based end position <= `max_endpos`.
  bool contains_ngram_before(const uint32_t* ptr, size_t len, size_t max_endpos) const {
    if (!ptr || len == 0 || len > max_endpos || filled_ < len) return false;
#ifdef AFF_WITH_RUST
    if (use_rust_) {
      if (!sam_) return false;
      const size_t ep = aff_sam_first_endpos(sam_, ptr, len);
      return ep > 0 && ep <= max_endpos;
    }
#endif
    const size_t lim = std::min(filled_, max_endpos);
    return std::search(cpp_seq_.begin(), cpp_seq_.begin() + static_cast<std::ptrdiff_t>(lim),
                       ptr, ptr + len) !=
           cpp_seq_.begin() + static_cast<std::ptrdiff_t>(lim);
  }

  Proposal propose(const std::vector<uint32_t>& seq, uint64_t owner,
                   int min_match, int max_draft, float min_score,
                   int32_t* out_tokens) {
    if (min_match <= 0) return Proposal{};
    if (!rust_sam_active() && min_match > kMaxMinMatch) min_match = kMaxMinMatch;

    if (owner != owner_) {
      reset(owner);
    } else if (min_match != min_match_) {
#ifdef AFF_WITH_RUST
      if (!use_rust_) {
        reset(owner);
      } else {
        const size_t check_len = std::min(seq.size(), filled_);
        if (check_len > 0 && (!sam_ || !aff_sam_matches_prefix(sam_, seq.data(), check_len))) {
          reset(owner);
        } else if (seq.size() < filled_) {
          rollback_to(seq.size());
        }
        min_match_ = min_match;
      }
#else
      reset(owner);
#endif
    } else if (seq.size() < filled_) {
#ifdef AFF_WITH_RUST
      if (!use_rust_ || (sam_ && aff_sam_matches_prefix(sam_, seq.data(), seq.size()))) {
        rollback_to(seq.size());
      } else {
        reset(owner);
      }
#else
      rollback_to(seq.size());
#endif
    }

    if (min_match_ == 0) {
      min_match_ = min_match;
      if (!rust_sam_active()) init_drops(min_match);
    }

    const size_t n = seq.size();
    if (filled_ < n) {
      append_delta(seq.data(), n);
    }

    if (max_draft <= 0 || out_tokens == nullptr || n <= static_cast<size_t>(min_match)) {
      return Proposal{};
    }

#ifdef AFF_WITH_RUST
    if (use_rust_) {
      if (!sam_) return Proposal{};
      const size_t cur_mlen = aff_sam_match_len(sam_);
      if (cur_mlen < static_cast<size_t>(min_match)) return Proposal{};
      // Passing only the relevant tail slice avoids scanning a 140K context in ends_with().
      const size_t tail_len = std::min(n, std::max<size_t>(64, cur_mlen + 1));
      const uint32_t* ctx_ptr = seq.data() + (n - tail_len);

      std::array<uint32_t, 64> stack_toks{};
      std::vector<uint32_t> heap_toks;
      uint32_t* buf = stack_toks.data();
      if (static_cast<size_t>(max_draft) > stack_toks.size()) {
        heap_toks.resize(static_cast<size_t>(max_draft));
        buf = heap_toks.data();
      }
      size_t out_len = 0;
      size_t out_match = 0;
      float out_score = 0.0f;
      float out_expect = 0.0f;
      if (!aff_sam_propose_ex(sam_, ctx_ptr, tail_len, static_cast<size_t>(max_draft),
                              static_cast<size_t>(min_match), min_score, buf, &out_len,
                              &out_score, &out_expect, &out_match)) {
        return Proposal{};
      }
      for (size_t i = 0; i < out_len; ++i) {
        out_tokens[i] = static_cast<int32_t>(buf[i]);
      }
      return Proposal{static_cast<int>(out_len), static_cast<int>(out_match), out_score,
                      out_expect};
    }
#endif

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

    last_match_len_ = static_cast<size_t>(match_len);
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

  void index_range_ptr(const uint32_t* seq, size_t start, size_t end) {
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
  size_t last_match_len_ = 0;
  bool use_rust_ = false;
  bool explicit_mode_ = false;

#ifdef AFF_WITH_RUST
  AffSam* sam_ = nullptr;
#endif

  OpenTable pat_[kExtra + 1];
  OpenTable ctx_total_;
  OpenTable ctx_tok_;
  std::vector<uint32_t> cpp_seq_;

  uint64_t roll_h1_[kExtra + 1]{};
  uint64_t roll_h2_[kExtra + 1]{};
  uint64_t drop1_[kExtra + 1]{};
  uint64_t drop2_[kExtra + 1]{};
};

// Thread-safe 8-entry LRU LCP session pool of SuffixDraftIndex instances (mirrors
// Tokenizer LCP reuse) so multi-turn conversations only append delta tokens.
class SuffixSessionPool {
public:
  static constexpr size_t kCapacity = 8;

  struct Stats {
    uint64_t hits = 0;
    uint64_t exact_hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
  };

  // mode: -1 = env (AFF_RUST_SAM), 0 = C++ fallback, 1 = Rust SAM
  explicit SuffixSessionPool(int mode = -1) : mode_(mode) {}

  class Lease {
  public:
    Lease() = default;
    ~Lease() { release_uncommitted(); }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : pool_(other.pool_),
          slot_(other.slot_),
          lcp_(other.lcp_),
          idx_(other.idx_),
          ephemeral_(std::move(other.ephemeral_)) {
      other.pool_ = nullptr;
      other.slot_ = SIZE_MAX;
      other.lcp_ = 0;
      other.idx_ = nullptr;
    }

    Lease& operator=(Lease&& other) noexcept {
      if (this == &other) return *this;
      release_uncommitted();
      pool_ = other.pool_;
      slot_ = other.slot_;
      lcp_ = other.lcp_;
      idx_ = other.idx_;
      ephemeral_ = std::move(other.ephemeral_);
      other.pool_ = nullptr;
      other.slot_ = SIZE_MAX;
      other.lcp_ = 0;
      other.idx_ = nullptr;
      return *this;
    }

    SuffixDraftIndex& index() noexcept { return *idx_; }
    size_t lcp() const noexcept { return lcp_; }
    size_t slot() const noexcept { return slot_; }

    // Commit final token history (prompt + emitted reply) and return entry to the pool.
    void release(const std::vector<uint32_t>& final_seq) {
      if (!idx_) return;
      if (!pool_ || slot_ == SIZE_MAX) {
        ephemeral_.reset();
        idx_ = nullptr;
        pool_ = nullptr;
        return;
      }
      idx_->append_delta(final_seq.data(), final_seq.size());
      pool_->finish_release(slot_, &final_seq);
      pool_ = nullptr;
      slot_ = SIZE_MAX;
      idx_ = nullptr;
    }

  private:
    friend class SuffixSessionPool;

    Lease(SuffixSessionPool* pool, size_t slot, size_t lcp, SuffixDraftIndex* idx)
        : pool_(pool), slot_(slot), lcp_(lcp), idx_(idx) {}

    Lease(size_t lcp, std::unique_ptr<SuffixDraftIndex> eph)
        : lcp_(lcp), idx_(eph.get()), ephemeral_(std::move(eph)) {}

    void release_uncommitted() {
      if (!idx_) return;
      if (pool_ && slot_ != SIZE_MAX) {
        pool_->finish_release(slot_, nullptr);
      }
      ephemeral_.reset();
      pool_ = nullptr;
      slot_ = SIZE_MAX;
      idx_ = nullptr;
    }

    SuffixSessionPool* pool_ = nullptr;
    size_t slot_ = SIZE_MAX;
    size_t lcp_ = 0;
    SuffixDraftIndex* idx_ = nullptr;
    std::unique_ptr<SuffixDraftIndex> ephemeral_;
  };

  Lease acquire(const std::vector<uint32_t>& seq, uint64_t owner = 1) {
    size_t chosen = SIZE_MAX;
    size_t best_lcp = 0;
    SuffixDraftIndex* idx_ptr = nullptr;
    bool need_reset = false;
    size_t prev_filled = 0;

    {
      std::lock_guard<std::mutex> lk(mu_);
      uint64_t best_lru_use = 0;
      size_t lru_idx = SIZE_MAX;
      uint64_t min_use = UINT64_MAX;

      for (size_t i = 0; i < count_; ++i) {
        const Entry& e = entries_[i];
        if (e.in_use) continue;
        if (e.last_use < min_use) {
          min_use = e.last_use;
          lru_idx = i;
        }
        const size_t lcp = common_prefix_len(e.tokens, seq);
        if (lcp > best_lcp || (lcp > 0 && lcp == best_lcp && e.last_use > best_lru_use)) {
          best_lcp = lcp;
          best_lru_use = e.last_use;
          chosen = i;
        }
      }

      if (best_lcp == 0) {
        if (count_ < kCapacity) {
          chosen = count_++;
          entries_[chosen].index = make_index();
        } else {
          chosen = lru_idx;
        }
      }

      if (chosen == SIZE_MAX) {
        ++stats_.misses;
      } else {
        Entry& e = entries_[chosen];
        e.in_use = true;
        e.last_use = ++tick_;
        idx_ptr = e.index.get();
        prev_filled = idx_ptr->filled();
        if (best_lcp > 0 && idx_ptr->owner() == owner) {
          best_lcp = std::min(best_lcp, prev_filled);
          ++stats_.hits;
          if (best_lcp == seq.size() && best_lcp == e.tokens.size()) {
            ++stats_.exact_hits;
          }
        } else {
          if (prev_filled > 0) ++stats_.evictions;
          ++stats_.misses;
          best_lcp = 0;
          need_reset = true;
        }
      }
    }

    if (chosen == SIZE_MAX) {
      auto eph = make_index();
      eph->reset(owner);
      eph->append_delta(seq.data(), seq.size());
      return Lease(0, std::move(eph));
    }

    // Mutate the leased entry outside mu_ so concurrent slots never block on delta indexing.
    if (need_reset || best_lcp == 0) {
      idx_ptr->reset(owner);
    } else if (best_lcp < prev_filled) {
      idx_ptr->rollback_to(best_lcp);
    }
    idx_ptr->append_delta(seq.data(), seq.size());
    entries_[chosen].tokens = seq;
    return Lease(this, chosen, best_lcp, idx_ptr);
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return count_;
  }

private:
  struct Entry {
    std::unique_ptr<SuffixDraftIndex> index;
    std::vector<uint32_t> tokens;
    uint64_t last_use = 0;
    bool in_use = false;
  };

  std::unique_ptr<SuffixDraftIndex> make_index() const {
    if (mode_ == 0) return std::make_unique<SuffixDraftIndex>(false);
    if (mode_ == 1) return std::make_unique<SuffixDraftIndex>(true);
    return std::make_unique<SuffixDraftIndex>();
  }

  static size_t common_prefix_len(const std::vector<uint32_t>& a,
                                  const std::vector<uint32_t>& b) noexcept {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i + 1 < n) {
      uint64_t va = 0, vb = 0;
      std::memcpy(&va, a.data() + i, 8);
      std::memcpy(&vb, b.data() + i, 8);
      if (va != vb) break;
      i += 2;
    }
    while (i < n && a[i] == b[i]) ++i;
    return i;
  }

  void finish_release(size_t slot, const std::vector<uint32_t>* final_seq) {
    Entry& e = entries_[slot];
    if (final_seq) {
      const size_t keep = std::min(final_seq->size(), e.index->filled());
      e.tokens.assign(final_seq->begin(), final_seq->begin() + static_cast<std::ptrdiff_t>(keep));
    } else if (e.index->filled() != e.tokens.size()) {
      const size_t target = std::min(e.tokens.size(), e.index->filled());
      e.index->rollback_to(target);
      e.tokens.resize(std::min(e.tokens.size(), e.index->filled()));
    }
    std::lock_guard<std::mutex> lk(mu_);
    e.last_use = ++tick_;
    e.in_use = false;
  }

  mutable std::mutex mu_;
  // Fixed-capacity array prevents element reallocation while a concurrent Lease accesses Entry::tokens.
  std::array<Entry, kCapacity> entries_{};
  size_t count_ = 0;
  uint64_t tick_ = 0;
  int mode_ = -1;
  Stats stats_{};
};

inline bool suffix_gate_v2_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("AFF_SUFFIX_GATE_V2");
    return !s || !*s || std::strcmp(s, "0") != 0;
  }();
  return on;
}

// Utility- and acceptance-aware gate controlling whether SuffixDecoding preempts
// the GPU DSpark drafter on a block. Tracks empirical suffix acceptance EMA and
// exponential miss cooldown so prose streams stay on DSpark while code/structured
// continuations still trigger suffix drafting immediately on long exact matches.
struct SuffixGate {
  float acc_ema = 0.90f;
  uint32_t miss_streak = 0;
  uint32_t cooldown = 0;
  uint32_t steps = 0;
  uint32_t good_steps = 0;
  bool last_rejected = false;

  void reset() noexcept {
    acc_ema = 0.90f;
    miss_streak = 0;
    cooldown = 0;
    steps = 0;
    good_steps = 0;
    last_rejected = false;
  }

  void observe_dspark() noexcept { last_rejected = false; }

  bool should_preempt(const SuffixDraftIndex::Proposal& prop, int min_match, float min_expect,
                      bool dspark_active, float dspark_exp_acc, float dspark_cost_ms,
                      float sfx_cost_ms, float min_gain = 1.05f) noexcept {
    if (prop.n < min_match || prop.expect < min_expect) return false;
    const bool v2 = suffix_gate_v2_enabled();
    if (dspark_active) {
      // Never let bare 4-token (min_match) or cold 5-token collisions preempt GPU DSpark;
      // require a strong exact anchor (>= min_match + 2) to enter suffix mode, or
      // >= min_match + 1 once a proven high-acceptance streak is active.
      // After a rejected suffix block (edit boundary), require a stronger >= min_match + 6 anchor
      // so short collisions skip cleanly onto one DSpark block before verbatim matching resumes.
      int req_match = (steps > 0 && miss_streak == 0 && acc_ema >= 0.80f)
                          ? (min_match + 1)
                          : (min_match + 2);
      if (v2 && last_rejected) req_match = std::max(req_match, min_match + 6);
      if (prop.match_len < req_match) {
        if (cooldown > 0) --cooldown;
        return false;
      }
    }
    const bool proven_verbatim_relock =
        v2 && good_steps >= 2u && prop.match_len >= 12 && prop.n >= 8 && prop.expect >= 8.0f;
    if (cooldown > 0) {
      if (proven_verbatim_relock ||
          (prop.match_len >= min_match + 2 && prop.n >= min_match + 2 && acc_ema >= 0.65f)) {
        cooldown = 0;
      } else {
        --cooldown;
        return false;
      }
    }
    const float eff_acc = proven_verbatim_relock ? std::max(acc_ema, 0.85f) : acc_ema;
    const float eff_expect = prop.expect * eff_acc;
    if (eff_expect < min_expect) return false;
    if (!dspark_active) return true;
    if (eff_expect <= dspark_exp_acc) return false;
    const float sfx_u = (1.0f + eff_expect) / std::max(sfx_cost_ms, 1.0f);
    const float dsp_u = (1.0f + std::max(dspark_exp_acc, 0.0f)) / std::max(dspark_cost_ms, 1.0f);
    return sfx_u >= dsp_u * min_gain;
  }

  void observe(uint32_t nd, uint32_t k) noexcept {
    if (nd == 0) return;
    const bool v2 = suffix_gate_v2_enabled();
    const float rate = static_cast<float>(k) / static_cast<float>(nd);
    const float alpha = (v2 && good_steps >= 3u && miss_streak == 0u) ? 0.25f : 0.45f;
    acc_ema = (1.0f - alpha) * acc_ema + alpha * rate;
    ++steps;
    last_rejected = (k < nd && k * 2u >= nd);
    if (k >= 4u && (k * 2u >= nd)) ++good_steps;
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
