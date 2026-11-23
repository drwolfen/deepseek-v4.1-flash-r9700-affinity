// Byte-level BPE tokenizer, loading a HuggingFace tokenizer.json.
//
// DeepSeek-V4 uses byte-level BPE with a GPT-2-style byte<->unicode mapping. Merges are applied
// by rank; ties are broken by leftmost position.
//
// Pre-tokenisation matters as much as the merges. DeepSeek-V4-Flash ships a three-stage Sequence:
// isolate runs of 1-3 digits, isolate CJK runs, then split on a GPT-4-style pattern. Feeding the
// merge table whitespace-delimited words instead produces ids the model never saw in training —
// the text still round-trips, so the damage is invisible until generation quality is measured.
#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aff {

class Tokenizer {
public:
  bool load(const std::string& tokenizer_json_path, std::string* err);
  std::vector<uint32_t> encode(const std::string& text, bool add_bos = false) const;
  std::string decode(const std::vector<uint32_t>& ids) const;
  const std::string& decode_one(uint32_t id) const;
  uint32_t vocab_size() const { return (uint32_t)id_to_tok_.size(); }
  int32_t bos() const { return bos_; }
  int32_t eos() const { return eos_; }

private:
  std::unordered_map<std::string, uint32_t> tok_to_id_;
  std::vector<std::string> id_to_tok_;
  std::vector<std::string> id_to_decoded_;                 // precomputed O(1) decode_one()
  std::unordered_map<std::string, uint32_t> merge_rank_;   // "a b" -> rank
  std::unordered_map<uint32_t, int> special_;
  int32_t bos_ = -1, eos_ = -1;
  std::vector<std::string> byte_to_unicode_;               // 256 entries
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
  void build_byte_maps();
  void build_fast_bpe();
  struct Span {
    uint32_t off = 0;
    uint32_t len = 0;
  };
  static void pretokenize_spans(std::string_view text, std::vector<Span>& out,
                                std::vector<Span>& stage1, std::vector<Span>& stage2);
  bool bpe_fast(std::string_view piece, std::vector<uint32_t>& out) const;
  std::vector<std::string> bpe(const std::string& piece) const;
  std::vector<std::string> pretokenize(const std::string& text) const;
  // added_tokens, longest-match-first so multi-byte markers like <｜User｜> map to their own id
  // instead of being byte-BPE'd into a dozen pieces.
  std::vector<std::pair<std::string, uint32_t>> added_;
  uint8_t added_lead_[256] = {};
  uint8_t added_len1_[256] = {};
  uint64_t added_lead2_[1024] = {};   // 64K-bit (8 KiB) 2-byte lead filter for added_tokens
  size_t max_added_len_ = 0;

  // Flat open-addressed merge hash table: ((uint64_t)left << 32) | right -> (rank, merged_id).
  static constexpr uint32_t kNoTokId = UINT32_MAX;
  struct MergeSlot {
    uint64_t key = UINT64_MAX;
    uint32_t rank = UINT32_MAX;
    uint32_t merged = UINT32_MAX;
  };
  std::vector<MergeSlot> merge_table_;
  uint32_t merge_mask_ = 0;
  std::array<uint32_t, 256> byte_to_id_{};
  uint32_t synth_base_ = 0;
  bool fast_bpe_ready_ = false;

  const MergeSlot* find_merge(uint32_t left, uint32_t right) const noexcept;

  // Bounded thread-safe per-piece BPE memoization cache (piece -> token IDs).
  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
      return std::hash<std::string_view>{}(sv);
    }
    size_t operator()(const std::string& s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };
  static constexpr size_t kBpeCacheMax = 32768;
  mutable std::shared_mutex bpe_mu_;
  mutable std::unordered_map<std::string, std::vector<uint32_t>, StringHash, std::equal_to<>> bpe_cache_;
  mutable std::deque<std::string> bpe_fifo_;

  // Bounded turn-boundary LCP prefix token cache.
  struct LcpEntry {
    bool add_bos = false;
    std::string text;
    std::vector<uint32_t> ids;
    std::vector<std::pair<size_t, size_t>> boundaries; // (byte_end, token_count)
    uint64_t last_use = 0;
  };
  static constexpr size_t kLcpCacheMax = 8;
  mutable std::mutex lcp_mu_;
  mutable std::vector<LcpEntry> lcp_cache_;
  mutable uint64_t lcp_tick_ = 0;
};

} // namespace aff
