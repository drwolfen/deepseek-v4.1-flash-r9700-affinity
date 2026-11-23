#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aff {

namespace {
// GPT-2 byte<->unicode: maps all 256 bytes to printable codepoints so the BPE alphabet is text.
void utf8_append(std::string& s, uint32_t cp) {
  if (cp < 0x80) s.push_back((char)cp);
  else if (cp < 0x800) { s.push_back((char)(0xC0 | (cp >> 6))); s.push_back((char)(0x80 | (cp & 0x3F))); }
  else { s.push_back((char)(0xE0 | (cp >> 12))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
         s.push_back((char)(0x80 | (cp & 0x3F))); }
}
std::string json_unescape(const std::string& in) {
  std::string o; o.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] != '\\') { o.push_back(in[i]); continue; }
    if (++i >= in.size()) break;
    switch (in[i]) {
      case 'n': o.push_back('\n'); break;
      case 't': o.push_back('\t'); break;
      case 'r': o.push_back('\r'); break;
      case 'b': o.push_back('\b'); break;
      case 'f': o.push_back('\f'); break;
      case '"': o.push_back('"'); break;
      case '\\': o.push_back('\\'); break;
      case '/': o.push_back('/'); break;
      case 'u': {
        if (i + 4 < in.size()) {
          const uint32_t cp = (uint32_t)std::strtoul(in.substr(i + 1, 4).c_str(), nullptr, 16);
          utf8_append(o, cp); i += 4;
        }
        break;
      }
      default: o.push_back(in[i]);
    }
  }
  return o;
}
// Finds the closing quote of a JSON string that opens at `q0`. A quote is escaped only when
// preceded by an ODD number of backslashes: the naive `j[q1-1] != '\\'` test treats the closing
// quote of a token ending in a backslash as escaped, runs off the end of the entry, and silently
// truncates the vocabulary to a fraction of its real size.
size_t json_string_end(const std::string& j, size_t q0, size_t limit) {
  for (size_t i = q0 + 1; i < limit; ++i) {
    if (j[i] != '"') continue;
    size_t bs = 0;
    while (bs < i && j[i - 1 - bs] == '\\') ++bs;
    if ((bs & 1) == 0) return i;
  }
  return std::string::npos;
}

// Finds the matching close for the bracket at `open`, skipping over string contents. The naive
// version counts brackets everywhere, so the vocabulary token "}" ends the vocab object early —
// which is how a 128000-entry vocab silently loaded as 1778 entries.
size_t json_match(const std::string& j, size_t open) {
  const char oc = j[open], cc = (oc == '{') ? '}' : ']';
  size_t depth = 0;
  for (size_t i = open; i < j.size(); ++i) {
    if (j[i] == '"') { i = json_string_end(j, i, j.size()); if (i == std::string::npos) break; continue; }
    if (j[i] == oc) ++depth;
    else if (j[i] == cc && --depth == 0) return i;
  }
  return std::string::npos;
}

// --- pre-tokeniser character classes (UTF-8, ASCII-exact) --------------------------------------
inline size_t utf8_len(unsigned char c) {
  if (c >= 0xF0) return 4;
  if (c >= 0xE0) return 3;
  if (c >= 0xC0) return 2;
  return 1;
}
inline uint32_t utf8_cp(std::string_view s, size_t i, size_t len) {
  const unsigned char c = (unsigned char)s[i];
  if (len == 1) return c;
  uint32_t cp = c & (0xFFu >> (len + 1));
  for (size_t k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
  return cp;
}
inline bool is_digit_cp(uint32_t c)  { return c >= '0' && c <= '9'; }
inline bool is_cjk_cp(uint32_t c) {
  return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}
inline bool is_space_cp(uint32_t c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
                                             c == 0x0B || c == 0x0C || c == 0xA0; }
inline bool is_nl_cp(uint32_t c)    { return c == '\r' || c == '\n'; }
// Letters: ASCII alphabetics plus everything non-ASCII that is not a space. Non-ASCII punctuation
// is rare enough in practice that lumping it with letters costs less than a Unicode table would.
inline bool is_letter_cp(uint32_t c) {
  if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  return !is_space_cp(c);
}
inline bool is_punct_cp(uint32_t c) {
  return c < 0x80 && !is_space_cp(c) && !is_digit_cp(c) && !is_letter_cp(c);
}
inline constexpr auto kAsciiPunctLut = []() {
  std::array<uint8_t, 256> t{};
  const char* s = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";
  for (size_t i = 0;; ++i) {
    t[(unsigned char)s[i]] = 1;
    if (s[i] == '\0') break;
  }
  return t;
}();
inline bool is_ascii_punct_class(char c) {   // the explicit set in rule A (including '\0' parity with strchr)
  return kAsciiPunctLut[(unsigned char)c] != 0;
}
inline uint32_t hash_pair(uint64_t k) noexcept {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return (uint32_t)k;
}

} // namespace

void Tokenizer::build_byte_maps() {
  byte_to_unicode_.assign(256, "");
  std::vector<int> cps(256, -1);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    if (printable) cps[b] = b;
  }
  for (int b = 0; b < 256; ++b) if (cps[b] < 0) cps[b] = 256 + n++;
  for (int b = 0; b < 256; ++b) {
    std::string s; utf8_append(s, (uint32_t)cps[b]);
    byte_to_unicode_[b] = s;
    unicode_to_byte_[s] = (uint8_t)b;
  }
}

bool Tokenizer::load(const std::string& path, std::string* err) {
  tok_to_id_.clear();
  id_to_tok_.clear();
  id_to_decoded_.clear();
  merge_rank_.clear();
  special_.clear();
  added_.clear();
  std::memset(added_lead_, 0, sizeof(added_lead_));
  std::memset(added_len1_, 0, sizeof(added_len1_));
  std::memset(added_lead2_, 0, sizeof(added_lead2_));
  max_added_len_ = 0;
  merge_table_.clear();
  merge_mask_ = 0;
  byte_to_id_.fill(kNoTokId);
  synth_base_ = 0;
  fast_bpe_ready_ = false;
  bos_ = -1;
  eos_ = -1;
  {
    std::unique_lock<std::shared_mutex> lk(bpe_mu_);
    bpe_cache_.clear();
    bpe_fifo_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(lcp_mu_);
    lcp_cache_.clear();
    lcp_tick_ = 0;
  }

  build_byte_maps();
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { if (err) *err = "cannot open " + path; return false; }
  std::string j; char buf[1 << 16]; size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) j.append(buf, n);
  std::fclose(f);

  // --- vocab ---
  size_t p = j.find("\"vocab\"");
  if (p == std::string::npos) { if (err) *err = "no vocab in tokenizer.json"; return false; }
  p = j.find('{', p);
  const size_t end = json_match(j, p);
  if (end == std::string::npos) { if (err) *err = "unterminated vocab object"; return false; }
  {
    size_t i = p + 1;
    while (i < end) {
      const size_t q0 = j.find('"', i); if (q0 == std::string::npos || q0 >= end) break;
      const size_t q1 = json_string_end(j, q0, end);
      if (q1 == std::string::npos) break;
      const std::string tok = json_unescape(j.substr(q0 + 1, q1 - q0 - 1));
      const size_t c = j.find(':', q1); if (c == std::string::npos) break;
      char* vend = nullptr;
      const uint32_t id = (uint32_t)std::strtoul(j.c_str() + c + 1, &vend, 10);
      if (id >= id_to_tok_.size()) id_to_tok_.resize(id + 1);
      id_to_tok_[id] = tok;
      tok_to_id_[tok] = id;
      // Advance past the VALUE, not to the next comma anywhere: a token containing a comma would
      // otherwise resync in the middle of the next key.
      i = (size_t)(vend - j.c_str());
    }
  }

  // --- merges ---
  p = j.find("\"merges\"");
  if (p != std::string::npos) {
    p = j.find('[', p);
    const size_t e = json_match(j, p);
    if (e == std::string::npos) { if (err) *err = "unterminated merges array"; return false; }
    uint32_t rank = 0;
    size_t i = p + 1;
    while (i < e) {
      const size_t q0 = j.find('"', i); if (q0 == std::string::npos || q0 >= e) break;
      const size_t q1 = json_string_end(j, q0, e);
      if (q1 == std::string::npos) break;
      merge_rank_[json_unescape(j.substr(q0 + 1, q1 - q0 - 1))] = rank++;
      i = q1 + 1;
    }
  }

  // --- added_tokens: the chat markers and control tokens live here, NOT in `vocab` ---
  // Without these, render_chat_prompt's <｜User｜> would be byte-BPE'd into a dozen ordinary
  // tokens and the model would never see the turn boundary it was trained on.
  {
    const size_t ap = j.find("\"added_tokens\"");
    if (ap != std::string::npos) {
      const size_t a0 = j.find('[', ap);
      const size_t a1 = a0 == std::string::npos ? std::string::npos : json_match(j, a0);
      for (size_t i = a0 + 1; a1 != std::string::npos && i < a1; ) {
        const size_t ob = j.find('{', i);
        if (ob == std::string::npos || ob >= a1) break;
        const size_t oe = json_match(j, ob);
        if (oe == std::string::npos || oe > a1) break;
        const std::string ent = j.substr(ob, oe - ob + 1);
        const size_t ip = ent.find("\"id\"");
        const size_t cp = ent.find("\"content\"");
        if (ip != std::string::npos && cp != std::string::npos) {
          const uint32_t id = (uint32_t)std::strtoul(ent.c_str() + ent.find(':', ip) + 1, nullptr, 10);
          const size_t q0 = ent.find('"', ent.find(':', cp));
          const size_t q1 = json_string_end(ent, q0, ent.size());
          if (q0 != std::string::npos && q1 != std::string::npos) {
            const std::string content = json_unescape(ent.substr(q0 + 1, q1 - q0 - 1));
            if (id >= id_to_tok_.size()) id_to_tok_.resize(id + 1);
            id_to_tok_[id] = content;
            added_.push_back({content, id});
            special_[id] = 1;
          }
        }
        i = oe + 1;
      }
      // Longest first so <｜end▁of▁sentence｜> wins over any prefix of itself.
      std::sort(added_.begin(), added_.end(),
                [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
      for (const auto& [content, id] : added_) {
        (void)id;
        if (!content.empty()) {
          const uint8_t b0 = (uint8_t)content[0];
          added_lead_[b0] = 1;
          if (content.size() == 1) {
            added_len1_[b0] = 1;
            for (uint32_t b1 = 0; b1 < 256; ++b1) {
              const uint32_t pair = ((uint32_t)b0 << 8) | b1;
              added_lead2_[pair >> 6] |= (1ULL << (pair & 63));
            }
          } else {
            const uint32_t pair = ((uint32_t)b0 << 8) | (uint8_t)content[1];
            added_lead2_[pair >> 6] |= (1ULL << (pair & 63));
          }
          if (content.size() > max_added_len_) max_added_len_ = content.size();
        }
      }
    }
  }

  // Precompute O(1) decode table for all tokens.
  id_to_decoded_.resize(id_to_tok_.size());
  for (size_t id = 0; id < id_to_tok_.size(); ++id) {
    const std::string& t = id_to_tok_[id];
    if (special_.count((uint32_t)id)) {
      id_to_decoded_[id] = t;
      continue;
    }
    std::string out;
    out.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
      size_t len = 1;
      const unsigned char c = (unsigned char)t[i];
      if (c >= 0xE0) len = 3;
      else if (c >= 0xC0) len = 2;
      auto it = unicode_to_byte_.find(t.substr(i, len));
      if (it != unicode_to_byte_.end()) out.push_back((char)it->second);
      else out.append(t, i, len);
      i += len;
    }
    id_to_decoded_[id] = std::move(out);
  }

  // --- special tokens: pick up bos/eos if the file names them ---
  auto find_named = [&](std::initializer_list<const char*> names) -> int32_t {
    for (const char* name : names) {
      for (const auto& [content, id] : added_) if (content == name) return (int32_t)id;
      auto it = tok_to_id_.find(name);
      if (it != tok_to_id_.end()) return (int32_t)it->second;
    }
    return -1;
  };
  bos_ = find_named({"<｜begin▁of▁sentence｜>", "<s>", "<|begin_of_text|>"});
  eos_ = find_named({"<｜end▁of▁sentence｜>", "</s>", "<|end_of_text|>", "<|eot_id|>"});
  build_fast_bpe();
  return !id_to_tok_.empty();
}

void Tokenizer::build_fast_bpe() {
  fast_bpe_ready_ = false;
  merge_table_.clear();
  merge_mask_ = 0;
  byte_to_id_.fill(kNoTokId);
  synth_base_ = (uint32_t)id_to_tok_.size();
  if (id_to_tok_.size() >= (size_t)(kNoTokId - 1000000u)) return;

  for (uint32_t b = 0; b < 256; ++b) {
    auto it = tok_to_id_.find(byte_to_unicode_[b]);
    if (it == tok_to_id_.end() || it->second >= synth_base_) return;
    byte_to_id_[b] = it->second;
  }

  if (merge_rank_.empty()) {
    fast_bpe_ready_ = true;
    return;
  }

  std::unordered_map<std::string, uint32_t> synth_map;
  uint32_t next_synth = synth_base_;
  auto resolve_sym = [&](const std::string& s) -> uint32_t {
    auto it = tok_to_id_.find(s);
    if (it != tok_to_id_.end()) return it->second;
    auto sit = synth_map.find(s);
    if (sit != synth_map.end()) return sit->second;
    if (next_synth >= kNoTokId - 1u) return kNoTokId;
    const uint32_t sid = next_synth++;
    synth_map.emplace(s, sid);
    return sid;
  };

  size_t cap = 16;
  while (cap < merge_rank_.size() * 2) cap <<= 1;
  merge_table_.assign(cap, MergeSlot{});
  merge_mask_ = (uint32_t)(cap - 1);

  for (const auto& [pair_str, rank] : merge_rank_) {
    const size_t sp = pair_str.find(' ');
    if (sp == std::string::npos || sp == 0 || sp + 1 >= pair_str.size()) {
      merge_table_.clear();
      return;
    }
    size_t chosen_sp = sp;
    if (pair_str.find(' ', sp + 1) != std::string::npos) {
      bool found = false;
      for (size_t pos = sp; pos != std::string::npos; pos = pair_str.find(' ', pos + 1)) {
        if (pos == 0 || pos + 1 >= pair_str.size()) continue;
        if (tok_to_id_.count(pair_str.substr(0, pos)) &&
            tok_to_id_.count(pair_str.substr(pos + 1))) {
          chosen_sp = pos;
          found = true;
          break;
        }
      }
      if (!found) {
        merge_table_.clear();
        return;
      }
    }
    const std::string left_s = pair_str.substr(0, chosen_sp);
    const std::string right_s = pair_str.substr(chosen_sp + 1);
    const uint32_t lid = resolve_sym(left_s);
    const uint32_t rid = resolve_sym(right_s);
    const uint32_t mid = resolve_sym(left_s + right_s);
    if (lid == kNoTokId || rid == kNoTokId || mid == kNoTokId) {
      merge_table_.clear();
      return;
    }
    const uint64_t key = ((uint64_t)lid << 32) | (uint64_t)rid;
    uint32_t idx = hash_pair(key) & merge_mask_;
    while (merge_table_[idx].key != UINT64_MAX && merge_table_[idx].key != key) {
      idx = (idx + 1) & merge_mask_;
    }
    merge_table_[idx] = MergeSlot{key, rank, mid};
  }
  fast_bpe_ready_ = true;
}

const Tokenizer::MergeSlot* Tokenizer::find_merge(uint32_t left, uint32_t right) const noexcept {
  if (merge_table_.empty()) return nullptr;
  const uint64_t key = ((uint64_t)left << 32) | (uint64_t)right;
  uint32_t idx = hash_pair(key) & merge_mask_;
  for (;;) {
    const MergeSlot& s = merge_table_[idx];
    if (s.key == key) return &s;
    if (s.key == UINT64_MAX) return nullptr;
    idx = (idx + 1) & merge_mask_;
  }
}

bool Tokenizer::bpe_fast(std::string_view piece, std::vector<uint32_t>& out) const {
  if (!fast_bpe_ready_) return false;
  const size_t n = piece.size();
  if (n == 0) return true;
  if (n == 1) {
    const uint32_t id = byte_to_id_[(uint8_t)piece[0]];
    if (id == kNoTokId) return false;
    out.push_back(id);
    return true;
  }
  if (n == 2) {
    const uint32_t id0 = byte_to_id_[(uint8_t)piece[0]];
    const uint32_t id1 = byte_to_id_[(uint8_t)piece[1]];
    if (id0 == kNoTokId || id1 == kNoTokId) return false;
    if (const MergeSlot* ms = find_merge(id0, id1)) {
      if (ms->merged >= synth_base_) return false;
      out.push_back(ms->merged);
    } else {
      out.push_back(id0);
      out.push_back(id1);
    }
    return true;
  }
  if (n >= 0xFFFFFFFEu) return false;

  struct BpeNode {
    uint32_t sym;
    uint32_t ver;
    uint32_t prev;
    uint32_t next;
  };
  struct BpeCand {
    uint32_t rank;
    uint32_t left;
    uint32_t right;
    uint32_t ver_l;
    uint32_t ver_r;
    uint32_t merged;
    bool operator>(const BpeCand& o) const noexcept {
      // Lower rank wins; identical ranks break leftmost-first via strictly monotonic left index.
      if (rank != o.rank) return rank > o.rank;
      return left > o.left;
    }
  };
  struct BpeWorkspace {
    std::vector<BpeNode> nodes;
    std::vector<BpeCand> heap;
  };
  thread_local BpeWorkspace ws;

  const uint32_t m = (uint32_t)n;
  ws.nodes.resize(m);
  for (uint32_t i = 0; i < m; ++i) {
    const uint32_t id = byte_to_id_[(uint8_t)piece[i]];
    if (id == kNoTokId) return false;
    ws.nodes[i] = BpeNode{
        id,
        0u,
        i == 0 ? kNoTokId : i - 1,
        i + 1 == m ? kNoTokId : i + 1
    };
  }

  ws.heap.clear();
  for (uint32_t i = 0; i + 1 < m; ++i) {
    if (const MergeSlot* ms = find_merge(ws.nodes[i].sym, ws.nodes[i + 1].sym)) {
      ws.heap.push_back(BpeCand{ms->rank, i, i + 1, 0u, 0u, ms->merged});
    }
  }
  std::make_heap(ws.heap.begin(), ws.heap.end(), std::greater<>{});

  while (!ws.heap.empty()) {
    std::pop_heap(ws.heap.begin(), ws.heap.end(), std::greater<>{});
    const BpeCand top = ws.heap.back();
    ws.heap.pop_back();
    BpeNode& nl = ws.nodes[top.left];
    BpeNode& nr = ws.nodes[top.right];
    if (nl.ver != top.ver_l || nr.ver != top.ver_r) continue;

    // Absorb right node into left so surviving node indices stay strictly increasing left-to-right.
    nl.sym = top.merged;
    ++nl.ver;
    nr.ver = kNoTokId;
    const uint32_t next_r = nr.next;
    nl.next = next_r;
    if (next_r != kNoTokId) ws.nodes[next_r].prev = top.left;

    if (nl.prev != kNoTokId) {
      if (const MergeSlot* ms = find_merge(ws.nodes[nl.prev].sym, nl.sym)) {
        ws.heap.push_back(BpeCand{ms->rank, nl.prev, top.left, ws.nodes[nl.prev].ver, nl.ver, ms->merged});
        std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<>{});
      }
    }
    if (nl.next != kNoTokId) {
      if (const MergeSlot* ms = find_merge(nl.sym, ws.nodes[nl.next].sym)) {
        ws.heap.push_back(BpeCand{ms->rank, top.left, nl.next, nl.ver, ws.nodes[nl.next].ver, ms->merged});
        std::push_heap(ws.heap.begin(), ws.heap.end(), std::greater<>{});
      }
    }
  }

  const size_t base_out = out.size();
  for (uint32_t cur = 0; cur != kNoTokId; cur = ws.nodes[cur].next) {
    const uint32_t sym = ws.nodes[cur].sym;
    if (sym >= synth_base_) {
      out.resize(base_out);
      return false;
    }
    out.push_back(sym);
  }
  return true;
}

std::vector<std::string> Tokenizer::bpe(const std::string& piece) const {
  // Start from single byte-chars, then repeatedly merge the lowest-ranked adjacent pair.
  std::vector<std::string> sym;
  for (unsigned char ch : piece) sym.push_back(byte_to_unicode_[ch]);
  if (sym.size() < 2) return sym;
  for (;;) {
    uint32_t best = UINT32_MAX; size_t at = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
      if (it != merge_rank_.end() && it->second < best) { best = it->second; at = i; }
    }
    if (best == UINT32_MAX) break;
    sym[at] += sym[at + 1];
    sym.erase(sym.begin() + (long)at + 1);
  }
  return sym;
}

// DeepSeek-V4-Flash's pre-tokeniser Sequence, in order:
//   1. Split on \p{N}{1,3}, Isolated  -> digit runs become chunks of at most three, left to right
//   2. Split on CJK/Hiragana/Katakana, Isolated
//   3. Split on a GPT-4-style pattern, Isolated
// The character classes are exact for ASCII; non-ASCII non-space is treated as a letter, which
// matches the pattern's behaviour for every script that actually reaches stage 3 (CJK having been
// removed by stage 2).
void Tokenizer::pretokenize_spans(std::string_view text, std::vector<Span>& out,
                                  std::vector<Span>& stage1, std::vector<Span>& stage2) {
  out.clear();
  stage1.clear();
  stage2.clear();
  if (text.empty()) return;

  // Stage 1: isolate digit runs in groups of <= 3.
  for (size_t i = 0; i < text.size(); ) {
    if (is_digit_cp((unsigned char)text[i])) {
      size_t j = i;
      while (j < text.size() && is_digit_cp((unsigned char)text[j])) ++j;
      for (size_t k = i; k < j; k += 3) {
        stage1.push_back(Span{(uint32_t)k, (uint32_t)std::min<size_t>(3, j - k)});
      }
      i = j;
    } else {
      size_t j = i;
      while (j < text.size() && !is_digit_cp((unsigned char)text[j])) ++j;
      stage1.push_back(Span{(uint32_t)i, (uint32_t)(j - i)});
      i = j;
    }
  }

  // Stage 2: isolate CJK runs.
  for (const Span sp : stage1) {
    const std::string_view seg = text.substr(sp.off, sp.len);
    size_t i = 0;
    while (i < seg.size()) {
      const size_t len = utf8_len((unsigned char)seg[i]);
      const bool cjk = is_cjk_cp(utf8_cp(seg, i, len));
      size_t j = i;
      while (j < seg.size()) {
        const size_t l2 = utf8_len((unsigned char)seg[j]);
        if (is_cjk_cp(utf8_cp(seg, j, l2)) != cjk) break;
        j += l2;
      }
      const size_t j_end = std::min(j, seg.size());
      stage2.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(j_end - i)});
      i = j;
    }
  }

  // Stage 3: the main pattern. HuggingFace's Split/Isolated emits the spans BETWEEN matches as
  // pieces too, so unmatched characters must coalesce into one piece rather than one piece each.
  // The pattern has no digit alternative, so a "123" segment from stage 1 has to arrive at the
  // merge table whole -- splitting it per digit is why 12345 tokenised as five ids instead of two.
  for (const Span sp : stage2) {
    const std::string_view seg = text.substr(sp.off, sp.len);
    size_t i = 0, pending = 0;
    auto flush_pending = [&](size_t upto) {
      if (upto > pending) out.push_back(Span{sp.off + (uint32_t)pending, (uint32_t)(upto - pending)});
    };
    while (i < seg.size()) {
      const size_t start = i;
      const size_t len = utf8_len((unsigned char)seg[i]);
      const uint32_t c = utf8_cp(seg, i, len);

      // A: one of an explicit punctuation set immediately followed by ASCII letters ("'s", "(x").
      if (len == 1 && is_ascii_punct_class(seg[i]) && i + 1 < seg.size() &&
          is_letter_cp((unsigned char)seg[i + 1]) && (unsigned char)seg[i + 1] < 0x80) {
        size_t j = i + 1;
        while (j < seg.size() && (unsigned char)seg[j] < 0x80 && is_letter_cp((unsigned char)seg[j])) ++j;
        flush_pending(i);
        out.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(j - i)});
        i = pending = j;
        continue;
      }
      // B: an optional single non-letter, non-punct, non-newline lead (typically a space) then letters.
      {
        size_t j = i;
        if (!is_nl_cp(c) && !is_letter_cp(c) && !is_punct_cp(c)) j += len;
        size_t k = j;
        while (k < seg.size()) {
          const size_t l2 = utf8_len((unsigned char)seg[k]);
          if (!is_letter_cp(utf8_cp(seg, k, l2))) break;
          k += l2;
        }
        if (k > j) {
          const size_t k_end = std::min(k, seg.size());
          flush_pending(i);
          out.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(k_end - i)});
          i = pending = k;
          continue;
        }
      }
      // C: an optional leading space then a punctuation/symbol run, then any newlines.
      {
        size_t j = i;
        if (c == ' ') j += 1;
        size_t k = j;
        while (k < seg.size() && is_punct_cp((unsigned char)seg[k])) ++k;
        if (k > j) {
          while (k < seg.size() && is_nl_cp((unsigned char)seg[k])) ++k;
          flush_pending(i);
          out.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(k - i)});
          i = pending = k;
          continue;
        }
      }
      // D: whitespace ending in newlines.
      {
        size_t j = i;
        while (j < seg.size() && is_space_cp((unsigned char)seg[j]) && !is_nl_cp((unsigned char)seg[j])) ++j;
        size_t k = j;
        while (k < seg.size() && is_nl_cp((unsigned char)seg[k])) ++k;
        if (k > j) {
          flush_pending(i);
          out.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(k - i)});
          i = pending = k;
          continue;
        }
      }
      // E/F: a whitespace run. When more text follows, the pattern's (?!\S) lookahead leaves the
      // final space attached to the NEXT segment, which is what makes " word" a single token.
      {
        size_t j = i;
        while (j < seg.size() && is_space_cp((unsigned char)seg[j])) ++j;
        if (j > i) {
          if (j < seg.size() && j - i > 1) --j;
          flush_pending(i);
          out.push_back(Span{sp.off + (uint32_t)i, (uint32_t)(j - i)});
          i = pending = j;
          continue;
        }
      }
      i = start + len;                        // no rule matched: keep accumulating
    }
    flush_pending(seg.size());
  }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
  std::vector<Span> spans, stage1, stage2;
  pretokenize_spans(text, spans, stage1, stage2);
  std::vector<std::string> out;
  out.reserve(spans.size());
  for (const Span sp : spans) out.emplace_back(text, sp.off, sp.len);
  return out;
}

namespace {
size_t common_prefix_len(const std::string& a, const std::string& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i + 8 <= n) {
    uint64_t va = 0, vb = 0;
    std::memcpy(&va, a.data() + i, 8);
    std::memcpy(&vb, b.data() + i, 8);
    if (va != vb) break;
    i += 8;
  }
  while (i < n && a[i] == b[i]) ++i;
  return i;
}
} // namespace

std::vector<uint32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<uint32_t> out;
  std::vector<std::pair<size_t, size_t>> boundaries;
  size_t resume_byte = 0;
  size_t matched_lcp_idx = SIZE_MAX;

  if (text.size() >= 64) {
    std::lock_guard<std::mutex> lk(lcp_mu_);
    size_t best_tok = 0;
    size_t best_bcount = 0;
    size_t best_idx = SIZE_MAX;
    for (size_t idx = 0; idx < lcp_cache_.size(); ++idx) {
      const LcpEntry& e = lcp_cache_[idx];
      if (e.add_bos != add_bos) continue;
      if (e.text.size() == text.size() && e.text == text) {
        lcp_cache_[idx].last_use = ++lcp_tick_;
        return e.ids;
      }
      if (e.text.size() < 64 || std::memcmp(e.text.data(), text.data(), 64) != 0) continue;
      const size_t lcp = common_prefix_len(e.text, text);
      if (lcp <= max_added_len_ || e.boundaries.empty()) continue;
      const size_t safe_limit = lcp - max_added_len_;
      auto it = std::upper_bound(
          e.boundaries.begin(), e.boundaries.end(), safe_limit,
          [](size_t lim, const std::pair<size_t, size_t>& b) { return lim < b.first; });
      if (it == e.boundaries.begin()) continue;
      --it;
      if (it->second > best_tok) {
        best_tok = it->second;
        resume_byte = it->first;
        best_bcount = (size_t)(it - e.boundaries.begin()) + 1;
        best_idx = idx;
      }
    }
    if (best_idx != SIZE_MAX) {
      const LcpEntry& e = lcp_cache_[best_idx];
      out.assign(e.ids.begin(), e.ids.begin() + (std::ptrdiff_t)best_tok);
      boundaries.assign(e.boundaries.begin(), e.boundaries.begin() + (std::ptrdiff_t)best_bcount);
      lcp_cache_[best_idx].last_use = ++lcp_tick_;
      matched_lcp_idx = best_idx;
    }
  }

  if (resume_byte == 0 && add_bos && bos_ >= 0) out.push_back((uint32_t)bos_);

  const char* fast_bpe_env = std::getenv("AFF_FAST_BPE");
  const bool use_fast_bpe = (!fast_bpe_env || !*fast_bpe_env || std::strcmp(fast_bpe_env, "0") != 0) &&
                            fast_bpe_ready_;

  // Added tokens are matched on the RAW text before pre-tokenisation, longest first, and the
  // spans between them go through the normal path.
  std::vector<uint32_t> pids;
  auto encode_piece = [&](std::string_view piece) {
    if (use_fast_bpe && piece.size() == 1) {
      out.push_back(byte_to_id_[(uint8_t)piece[0]]);
      return;
    }
    if (piece.size() <= 128) {
      std::shared_lock<std::shared_mutex> slk(bpe_mu_);
      auto it = bpe_cache_.find(piece);
      if (it != bpe_cache_.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
      }
    }
    pids.clear();
    if (!use_fast_bpe || !bpe_fast(piece, pids)) {
      pids.clear();
      for (const std::string& sym : bpe(std::string(piece))) {
        auto it = tok_to_id_.find(sym);
        if (it != tok_to_id_.end()) { pids.push_back(it->second); continue; }
        // The byte alphabet covers all 256 values, so this only fires on a truncated vocab.
        for (unsigned char ch : sym) {
          auto b = tok_to_id_.find(byte_to_unicode_[ch]);
          if (b != tok_to_id_.end()) pids.push_back(b->second);
        }
      }
    }
    out.insert(out.end(), pids.begin(), pids.end());
    if (piece.size() <= 128) {
      std::unique_lock<std::shared_mutex> ulk(bpe_mu_);
      if (bpe_cache_.find(piece) == bpe_cache_.end()) {
        while (bpe_cache_.size() >= kBpeCacheMax && !bpe_fifo_.empty()) {
          bpe_cache_.erase(bpe_fifo_.front());
          bpe_fifo_.pop_front();
        }
        std::string key(piece);
        bpe_fifo_.push_back(key);
        bpe_cache_.emplace(std::move(key), pids);
      }
    }
  };

  size_t at = resume_byte;
  std::vector<Span> s1, s2, spans;
  auto flush = [&](size_t from, size_t to) {
    if (to <= from) return;
    if (to - from >= 0xFFFFFFFEu) {
      for (const std::string& piece : pretokenize(text.substr(from, to - from))) {
        encode_piece(piece);
      }
      return;
    }
    const std::string_view chunk(text.data() + from, to - from);
    pretokenize_spans(chunk, spans, s1, s2);
    for (const Span sp : spans) encode_piece(chunk.substr(sp.off, sp.len));
  };
  const size_t tlen = text.size();
  size_t i = resume_byte;
  while (i < tlen) {
    size_t hit_len = 0;
    uint32_t hit_id = 0;
    const uint8_t b0 = (uint8_t)text[i];
    if (added_lead_[b0]) {
      bool pass2 = false;
      if (i + 1 < tlen) {
        const uint32_t pair = ((uint32_t)b0 << 8) | (uint8_t)text[i + 1];
        pass2 = ((added_lead2_[pair >> 6] >> (pair & 63)) & 1ULL) != 0;
      } else {
        pass2 = added_len1_[b0] != 0;
      }
      if (pass2) {
        const size_t rem = tlen - i;
        for (const auto& [content, id] : added_) {
          if (!content.empty() && content.size() <= rem && content[0] == text[i] &&
              (content.size() == 1 || content[1] == text[i + 1]) &&
              std::memcmp(text.data() + i, content.data(), content.size()) == 0) {
            hit_len = content.size(); hit_id = id; break;      // added_ is sorted longest-first
          }
        }
      }
    }
    if (hit_len) {
      flush(at, i);
      out.push_back(hit_id);
      i += hit_len;
      at = i;
      boundaries.emplace_back(i, out.size());
    } else {
      ++i;
    }
  }
  flush(at, text.size());

  if (text.size() >= 64) {
    std::lock_guard<std::mutex> lk(lcp_mu_);
    const uint64_t tick = ++lcp_tick_;
    size_t slot = matched_lcp_idx;
    if (slot >= lcp_cache_.size()) {
      if (lcp_cache_.size() < kLcpCacheMax) {
        lcp_cache_.push_back(LcpEntry{});
        slot = lcp_cache_.size() - 1;
      } else {
        slot = 0;
        for (size_t k = 1; k < lcp_cache_.size(); ++k) {
          if (lcp_cache_[k].last_use < lcp_cache_[slot].last_use) slot = k;
        }
      }
    }
    LcpEntry& e = lcp_cache_[slot];
    e.add_bos = add_bos;
    e.text = text;
    e.ids = out;
    e.boundaries = std::move(boundaries);
    e.last_use = tick;
  }

  return out;
}

const std::string& Tokenizer::decode_one(uint32_t id) const {
  static const std::string kEmpty;
  return id < id_to_decoded_.size() ? id_to_decoded_[id] : kEmpty;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
  std::string out;
  for (uint32_t id : ids) out += decode_one(id);
  return out;
}

} // namespace aff
