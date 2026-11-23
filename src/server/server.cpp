#include "server.h"
#include "engine/model.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace aff {

namespace {
const char* skip_ws(const char* p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p; return p; }

struct JsonSpan {
  size_t start = 0;
  size_t end = 0;
};

// Scans a top-level JSON object in a single pass, recording byte spans for every top-level key.
// Rejects non-object inputs, unclosed strings, and unbalanced braces/brackets.
bool scan_top_level_object(const std::string& s,
                           std::vector<std::pair<std::string, JsonSpan>>* fields) {
  fields->clear();
  auto skip_w = [&](size_t p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\t' || s[p] == '\r')) ++p;
    return p;
  };
  auto scan_str_end = [&](size_t q0) -> size_t {
    for (size_t k = q0 + 1; k < s.size(); ++k) {
      if (s[k] == '\\') {
        if (k + 1 >= s.size()) return std::string::npos;
        ++k;
      } else if (s[k] == '"') {
        return k;
      }
    }
    return std::string::npos;
  };

  size_t i = skip_w(0);
  if (i >= s.size() || s[i] != '{') return false;
  i = skip_w(i + 1);
  if (i < s.size() && s[i] == '}') return skip_w(i + 1) == s.size();

  while (i < s.size()) {
    i = skip_w(i);
    if (i >= s.size() || s[i] != '"') return false;
    const size_t k1 = scan_str_end(i);
    if (k1 == std::string::npos) return false;
    std::string key;
    key.reserve(k1 - i - 1);
    bool simple = true;
    for (size_t k = i + 1; k < k1; ++k) {
      if (s[k] == '\\') { simple = false; break; }
    }
    if (simple) {
      key.assign(s, i + 1, k1 - i - 1);
    } else {
      for (size_t k = i + 1; k < k1; ++k) {
        if (s[k] == '\\' && k + 1 < k1) {
          ++k;
          switch (s[k]) {
            case 'n': key.push_back('\n'); break;
            case 't': key.push_back('\t'); break;
            case 'r': key.push_back('\r'); break;
            case '"': key.push_back('"'); break;
            case '\\': key.push_back('\\'); break;
            default: key.push_back(s[k]); break;
          }
        } else {
          key.push_back(s[k]);
        }
      }
    }
    i = skip_w(k1 + 1);
    if (i >= s.size() || s[i] != ':') return false;
    i = skip_w(i + 1);
    if (i >= s.size()) return false;
    const size_t vstart = i;
    size_t vend = i;
    if (s[i] == '"') {
      const size_t s1 = scan_str_end(i);
      if (s1 == std::string::npos) return false;
      vend = s1 + 1;
    } else if (s[i] == '{' || s[i] == '[') {
      size_t depth = 0;
      bool in_str = false, esc = false, closed = false;
      for (size_t k = i; k < s.size(); ++k) {
        const char c = s[k];
        if (in_str) {
          if (esc) esc = false;
          else if (c == '\\') esc = true;
          else if (c == '"') in_str = false;
          continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') {
          if (--depth == 0) { vend = k + 1; closed = true; break; }
        }
      }
      if (!closed) return false;
    } else {
      size_t k = i;
      while (k < s.size() && s[k] != ',' && s[k] != '}' &&
             s[k] != ' ' && s[k] != '\n' && s[k] != '\t' && s[k] != '\r') {
        ++k;
      }
      if (k == i) return false;
      vend = k;
    }
    fields->emplace_back(std::move(key), JsonSpan{vstart, vend});
    i = skip_w(vend);
    if (i >= s.size()) return false;
    if (s[i] == '}') return skip_w(i + 1) == s.size();
    if (s[i] != ',') return false;
    ++i;
  }
  return false;
}

const JsonSpan* find_field(const std::vector<std::pair<std::string, JsonSpan>>& fields,
                           const char* key) {
  for (const auto& kv : fields) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

// A key at the TOP level of `s`. The plain search matches the same name anywhere, nested objects
// included: a get_weather tool with a "temperature" property would otherwise be read as the
// sampler's temperature, parse to 0, and pin the whole server to greedy decoding.
const char* find_top_key(const std::string& s, const char* key) {
  const std::string pat = std::string("\"") + key + "\"";
  size_t depth = 0;
  bool in_str = false, esc = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      // Only a key is followed by a colon; a value that happens to read the same is not one.
      if (depth == 1 && s.compare(i, pat.size(), pat) == 0) {
        const char* q = skip_ws(s.c_str() + i + pat.size());
        if (*q == ':') return q + 1;
      }
      in_str = true;
      continue;
    }
    if (c == '{' || c == '[') ++depth;
    else if (c == '}' || c == ']') { if (depth) --depth; }
  }
  return nullptr;
}

std::string read_json_string(const char*& p) {
  p = skip_ws(p);
  if (*p != '"') return "";
  ++p;
  const char* scan = p;
  while (*scan && *scan != '"') {
    if (*scan == '\\' && scan[1]) scan += 2;
    else ++scan;
  }
  std::string out;
  out.reserve((size_t)(scan - p));
  const auto hex4 = [](const char* q) {
    for (int i = 0; i < 4; ++i)
      if (!q[i] || !std::isxdigit((unsigned char)q[i])) return false;
    return true;
  };
  while (*p && *p != '"') {
    const char* chunk = p;
    while (*p && *p != '"' && *p != '\\') ++p;
    if (p > chunk) out.append(chunk, (size_t)(p - chunk));
    if (*p != '\\') continue;
    if (!p[1]) {
      out.push_back('\\');
      ++p;
      break;
    }
    ++p;
    switch (*p) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case 'u': {
        if (hex4(p + 1)) {
          char hx[5] = {p[1], p[2], p[3], p[4], 0};
          uint32_t cp = (uint32_t)std::strtoul(hx, nullptr, 16);
          p += 4;
          // A character outside the BMP is escaped as a surrogate PAIR, which is what any client
          // sending ASCII-safe JSON emits — Python's json.dumps does by default. Decoding the halves
          // separately gives two sequences that are not valid UTF-8, so an emoji in a tool result
          // reaches the model as mojibake and comes back out the same way.
          if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u' && hex4(p + 3)) {
            char lhx[5] = {p[3], p[4], p[5], p[6], 0};
            const uint32_t lo = (uint32_t)std::strtoul(lhx, nullptr, 16);
            if (lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              p += 6;
            }
          }
          if (cp < 0x80) out.push_back((char)cp);
          else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
          else if (cp < 0x10000) { out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F)));
                 out.push_back((char)(0x80|(cp&0x3F))); }
          else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F)));
                 out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
        } else {
          out.push_back('u');
        }
        break;
      }
      default: out.push_back(*p);
    }
    ++p;
  }
  if (*p == '"') ++p;
  return out;
}
bool send_all(int fd, const char* d, size_t n) {
  while (n) {
    const ssize_t w = ::send(fd, d, n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (w == 0) return false;
    d += w; n -= (size_t)w;
  }
  return true;
}
constexpr const char* kUtf8Replacement = "\xEF\xBF\xBD";

// Validate one UTF-8 codepoint starting at s[i..n). Returns valid byte length (1..4),
// or 0 if ill-formed and sets *skip to the maximal ill-formed subpart length (1..3).
size_t utf8_next(const char* s, size_t i, size_t n, size_t* skip) {
  const uint8_t b0 = (uint8_t)s[i];
  if (b0 < 0x80) return 1;
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    if (i + 1 < n && (((uint8_t)s[i + 1] & 0xC0) == 0x80)) return 2;
    *skip = 1;
    return 0;
  }
  if (b0 >= 0xE0 && b0 <= 0xEF) {
    if (i + 1 < n) {
      const uint8_t b1 = (uint8_t)s[i + 1];
      const bool b1_ok = (b0 == 0xE0) ? (b1 >= 0xA0 && b1 <= 0xBF)
                       : (b0 == 0xED) ? (b1 >= 0x80 && b1 <= 0x9F)
                                      : (b1 >= 0x80 && b1 <= 0xBF);
      if (b1_ok) {
        if (i + 2 < n && (((uint8_t)s[i + 2] & 0xC0) == 0x80)) return 3;
        *skip = 2;
        return 0;
      }
    }
    *skip = 1;
    return 0;
  }
  if (b0 >= 0xF0 && b0 <= 0xF4) {
    if (i + 1 < n) {
      const uint8_t b1 = (uint8_t)s[i + 1];
      const bool b1_ok = (b0 == 0xF0) ? (b1 >= 0x90 && b1 <= 0xBF)
                       : (b0 == 0xF4) ? (b1 >= 0x80 && b1 <= 0x8F)
                                      : (b1 >= 0x80 && b1 <= 0xBF);
      if (b1_ok) {
        if (i + 2 < n && (((uint8_t)s[i + 2] & 0xC0) == 0x80)) {
          if (i + 3 < n && (((uint8_t)s[i + 3] & 0xC0) == 0x80)) return 4;
          *skip = 3;
          return 0;
        }
        *skip = 2;
        return 0;
      }
    }
    *skip = 1;
    return 0;
  }
  *skip = 1;
  return 0;
}
} // namespace

size_t utf8_safe_end(const std::string& s, size_t limit) {
  if (limit > s.size()) limit = s.size();
  if (limit == 0) return 0;
  if ((uint8_t)s[limit - 1] < 0x80) return limit;

  size_t lead = limit - 1;
  size_t cont = 0;
  while (lead > 0 && cont < 3 && (((uint8_t)s[lead] & 0xC0) == 0x80)) {
    --lead;
    ++cont;
  }
  const uint8_t b0 = (uint8_t)s[lead];
  if ((b0 & 0xC0) == 0x80 || b0 < 0xC2 || b0 > 0xF4) return limit;

  const size_t have = limit - lead;
  const size_t need = b0 >= 0xF0 ? 4 : b0 >= 0xE0 ? 3 : 2;
  if (have >= need) return limit;

  if (have >= 2) {
    const uint8_t b1 = (uint8_t)s[lead + 1];
    if (b0 == 0xE0 && (b1 < 0xA0 || b1 > 0xBF)) return limit;
    if (b0 == 0xED && (b1 < 0x80 || b1 > 0x9F)) return limit;
    if (b0 == 0xF0 && (b1 < 0x90 || b1 > 0xBF)) return limit;
    if (b0 == 0xF4 && (b1 < 0x80 || b1 > 0x8F)) return limit;
  }
  return lead;
}

std::string sanitize_utf8(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    size_t skip = 1;
    const size_t len = utf8_next(s.data(), i, n, &skip);
    if (len > 0) {
      o.append(s, i, len);
      i += len;
    } else {
      o += kUtf8Replacement;
      i += skip;
    }
  }
  return o;
}

#ifdef __AVX2__
// Scans clean 7-bit ASCII [0x20, 0x7F] excluding '"' and '\\' in 32-byte YMM chunks (no 512-bit ZMM).
static inline size_t avx2_scan_clean_ascii(const char* p, size_t i, size_t n) {
  const __m256i v_1f = _mm256_set1_epi8(0x1F);
  const __m256i v_quote = _mm256_set1_epi8('"');
  const __m256i v_bslash = _mm256_set1_epi8('\\');
  while (i + 32 <= n) {
    const __m256i chunk = _mm256_loadu_si256((const __m256i*)(p + i));
    const __m256i in_range = _mm256_cmpgt_epi8(chunk, v_1f);
    const __m256i bad_ch = _mm256_or_si256(_mm256_cmpeq_epi8(chunk, v_quote),
                                           _mm256_cmpeq_epi8(chunk, v_bslash));
    const __m256i clean = _mm256_andnot_si256(bad_ch, in_range);
    const uint32_t clean_mask = (uint32_t)_mm256_movemask_epi8(clean);
    if (clean_mask != 0xFFFFFFFFu) {
      i += (size_t)__builtin_ctz(~clean_mask);
      break;
    }
    i += 32;
  }
  return i;
}
#endif

std::string json_escape(const std::string& s) {
  const size_t n = s.size();
  size_t i = 0;
#ifdef __AVX2__
  i = avx2_scan_clean_ascii(s.data(), 0, n);
  if (i == n) return s;
#endif
  std::string o;
  o.reserve(n + 16);
  if (i > 0) o.append(s.data(), i);
  while (i < n) {
#ifdef __AVX2__
    if (i + 32 <= n) {
      const size_t start = i;
      i = avx2_scan_clean_ascii(s.data(), i, n);
      if (i > start) {
        o.append(s.data() + start, i - start);
        if (i >= n) break;
      }
    }
#endif
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
          if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
          else o.push_back((char)c);
      }
      ++i;
      continue;
    }
    size_t skip = 1;
    const size_t len = utf8_next(s.data(), i, n, &skip);
    if (len > 0) {
      o.append(s, i, len);
      i += len;
    } else {
      o += kUtf8Replacement;
      i += skip;
    }
  }
  return o;
}


static std::string usage_json(const GenResult& g) {
  const uint64_t t = (uint64_t)g.prompt_tokens + g.completion_tokens;
  return "\"usage\":{\"prompt_tokens\":" + std::to_string(g.prompt_tokens) +
         ",\"completion_tokens\":" + std::to_string(g.completion_tokens) +
         ",\"total_tokens\":" + std::to_string(t) +
         ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(g.cached_tokens) + "}" +
         ",\"completion_tokens_details\":{\"reasoning_tokens\":" +
         std::to_string(g.reasoning_tokens) + "}}";
}

// The raw text of the value at `key`, braces and all. The parser above reads scalars; tools and
// tool_calls are structures that get handed on verbatim, so they need their span rather than their
// meaning.
static std::string raw_value(const std::string& s, const char* key) {
  const char* p = find_top_key(s, key);
  if (!p) return {};
  const char* q = skip_ws(p);
  const size_t start = (size_t)(q - s.c_str());
  if (*q != '{' && *q != '[') return {};
  const char open = *q, close = open == '{' ? '}' : ']';
  size_t depth = 0;
  bool in_str = false, esc = false;
  for (size_t i = start; i < s.size(); ++i) {
    const char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') { in_str = true; continue; }
    if (c == open) ++depth;
    else if (c == close && --depth == 0) return s.substr(start, i - start + 1);
  }
  return {};
}

// Top-level elements of a JSON array in s[lo..hi), as raw text.
static std::vector<std::string> raw_elements_range(const std::string& arr, size_t lo, size_t hi) {
  std::vector<std::string> out;
  size_t depth = 0, start = std::string::npos;
  bool in_str = false, esc = false;
  for (size_t i = lo; i < hi; ++i) {
    const char c = arr[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') { in_str = true; continue; }
    if (c == '{' || c == '[') { if (depth++ == 1) start = i; }
    else if (c == '}' || c == ']') {
      if (--depth == 1 && start != std::string::npos) { out.push_back(arr.substr(start, i - start + 1)); start = std::string::npos; }
    }
  }
  return out;
}

bool parse_completion_request(const std::string& body, bool chat, CompletionRequest* out) {
  *out = CompletionRequest{};
  out->raw_prompt = !chat;

  std::vector<std::pair<std::string, JsonSpan>> fields;
  if (!scan_top_level_object(body, &fields)) return false;

  auto field_ptr = [&](const char* key) -> const char* {
    const JsonSpan* sp = find_field(fields, key);
    return sp ? (body.c_str() + sp->start) : nullptr;
  };
  auto field_str = [&](const char* key) -> std::string {
    const JsonSpan* sp = find_field(fields, key);
    if (!sp) return {};
    const char c = body[sp->start];
    if (c != '{' && c != '[') return {};
    return body.substr(sp->start, sp->end - sp->start);
  };

  if (const char* p = field_ptr("ignore_eos")) out->ignore_eos = std::strncmp(p, "true", 4) == 0;
  if (const char* p = field_ptr("model")) { const char* q = p; out->model = read_json_string(q); }
  // `max_completion_tokens` is the current OpenAI spelling and `max_tokens` the deprecated one;
  // clients send either, and reading only one silently ignores the caller's limit.
  for (const char* k : {"max_tokens", "max_completion_tokens"})
    if (const char* p = field_ptr(k)) {
      out->max_tokens = (uint32_t)std::strtoul(p, nullptr, 10);
      out->has_max_tokens = true;
    }
  if (const char* p = field_ptr("temperature")) { out->temperature = std::strtof(p, nullptr); out->has_temperature = true; }
  if (const char* p = field_ptr("top_p")) { out->top_p = std::strtof(p, nullptr); out->has_top_p = true; }
  if (const char* p = field_ptr("top_k")) out->top_k = (uint32_t)std::strtoul(p, nullptr, 10);
  if (const char* p = field_ptr("min_p")) out->min_p = std::strtof(p, nullptr);
  if (const char* p = field_ptr("repeat_penalty")) out->repeat_penalty = std::strtof(p, nullptr);
  else if (const char* p = field_ptr("repetition_penalty")) out->repeat_penalty = std::strtof(p, nullptr);
  if (const char* p = field_ptr("repeat_last_n")) out->repeat_last_n = (uint32_t)std::strtoul(p, nullptr, 10);
  if (const char* p = field_ptr("dry_delta")) out->dry_delta = std::strtof(p, nullptr);
  if (const char* p = field_ptr("seed")) out->seed = std::strtoull(p, nullptr, 10);
  if (const char* p = field_ptr("n")) out->n = (uint32_t)std::strtoul(p, nullptr, 10);
  if (const char* p = field_ptr("stream")) out->stream = std::strncmp(p, "true", 4) == 0;
  // include_usage lives inside stream_options, so it is one level down and the top-level search
  // cannot see it. Pull the sub-object out and search that.
  const std::string stream_options = field_str("stream_options");
  if (!stream_options.empty()) {
    if (const char* p = find_top_key(stream_options, "include_usage")) {
      p = skip_ws(p);
      out->include_usage = std::strncmp(p, "true", 4) == 0;
    }
  }
  if (const char* p = field_ptr("reasoning_effort")) {
    const char* q = p;
    if (*q == '"') out->reasoning_effort = read_json_string(q);
    else if (std::isdigit((unsigned char)*q)) out->reasoning_effort = std::to_string(std::strtoul(q, nullptr, 10));
  }
  if (const char* p = field_ptr("thinking_mode")) { const char* q = p; out->thinking_mode = read_json_string(q); }
  const std::string ctk = field_str("chat_template_kwargs");
  if (out->reasoning_effort.empty() && !ctk.empty()) {
    if (const char* p = find_top_key(ctk, "reasoning_effort")) {
      const char* q = skip_ws(p);
      if (*q == '"') out->reasoning_effort = read_json_string(q);
      else if (std::isdigit((unsigned char)*q)) out->reasoning_effort = std::to_string(std::strtoul(q, nullptr, 10));
    }
  }
  if (out->thinking_mode.empty()) {
    if (!ctk.empty()) {
      for (const char* k : {"enable_thinking", "thinking"}) {
        if (const char* p = find_top_key(ctk, k)) {
          p = skip_ws(p);
          if (std::strncmp(p, "false", 5) == 0) out->thinking_mode = "chat";
          else if (std::strncmp(p, "true", 4) == 0) out->thinking_mode = "thinking";
        }
      }
    }
    if (out->thinking_mode.empty()) {
      for (const char* k : {"enable_thinking", "thinking"}) {
        if (const char* p = field_ptr(k)) {
          if (std::strncmp(p, "false", 5) == 0) out->thinking_mode = "chat";
          else if (std::strncmp(p, "true", 4) == 0) out->thinking_mode = "thinking";
        }
      }
    }
    if (out->reasoning_effort == "none" || out->reasoning_effort == "off" || out->reasoning_effort == "0") {
      if (out->thinking_mode == "thinking") {
        out->suppress_final_think = true;
      } else {
        out->thinking_mode = "chat";
      }
    }
  }
  if (const char* p = field_ptr("tool_choice")) { const char* q = p; out->tool_choice = read_json_string(q); }
  out->response_format = field_str("response_format");

  // tools[] -> the `function` object of each, verbatim. The wrapper is dropped here because that is
  // what the container's encoder renders (`tools_from_openai_format`). If the outer wrapper carries
  // a 1017 `namespace` field, inject it into `fn` so `chat_encode` qualifies `name` as `ns::name`.
  if (const JsonSpan* tsp = find_field(fields, "tools")) {
    if (body[tsp->start] == '[') {
      for (const std::string& t : raw_elements_range(body, tsp->start, tsp->end)) {
        std::string fn = raw_value(t, "function");
        const std::string outer_ns = raw_value(t, "namespace");
        if (fn.empty()) {
          fn = t;                  // already unwrapped, which some clients send
        } else if (!outer_ns.empty() && raw_value(fn, "namespace").empty()) {
          const size_t rbrace = fn.rfind('}');
          if (rbrace != std::string::npos) {
            fn = fn.substr(0, rbrace) + ",\"namespace\":" + outer_ns + "}";
          }
        }
        out->tools.push_back(std::move(fn));
      }
    }
  }

  if (const char* p = field_ptr("stop")) {
    const char* q = p;
    if (*q == '[') { ++q; while (*q && *q != ']') { q = skip_ws(q); if (*q == '"') out->stop.push_back(read_json_string(q)); else ++q; if (*q == ',') ++q; } }
    else if (*q == '"') out->stop.push_back(read_json_string(q));
  }

  if (chat) {
    const JsonSpan* mspan = find_field(fields, "messages");
    if (!mspan || body[mspan->start] != '[') return false;
    for (const std::string& sub : raw_elements_range(body, mspan->start, mspan->end)) {
      if (sub.empty() || sub.front() != '{') continue;
      ChatMessage m;
      std::vector<std::pair<std::string, JsonSpan>> mfields;
      if (scan_top_level_object(sub, &mfields)) {
        auto mptr = [&](const char* key) -> const char* {
          const JsonSpan* sp = find_field(mfields, key);
          return sp ? (sub.c_str() + sp->start) : nullptr;
        };
        if (const char* r = mptr("role")) { const char* z = r; m.role = read_json_string(z); }
        if (m.role == "developer") m.role = "system";
        else if (m.role == "function") m.role = "tool";
        if (const JsonSpan* csp = find_field(mfields, "content")) {
          const char* z = sub.c_str() + csp->start;
          if (*z == '[') {
            std::string joined;
            for (const std::string& blk : raw_elements_range(sub, csp->start, csp->end)) {
              if (const char* t = find_top_key(blk, "text")) {
                const char* w = t;
                const std::string piece = read_json_string(w);
                if (!piece.empty()) {
                  if (!joined.empty()) joined += "\n\n";
                  joined += piece;
                }
              }
              if (m.role == "user") {
                if (const char* img = find_top_key(blk, "image_url")) {
                  (void)img;
                  std::string img_obj = raw_value(blk, "image_url");
                  if (img_obj.empty()) img_obj = blk;
                  if (const char* u = find_top_key(img_obj, "url")) {
                    const char* w = u;
                    m.image_data = read_json_string(w);
                    m.has_image = !m.image_data.empty();
                  }
                }
              }
            }
            m.content = joined;
          } else if (*z == '"') {
            const char* w = z;
            m.content = read_json_string(w);
            size_t sent_pos = m.role == "user" ? m.content.find("<｜deepseek_image｜>") : std::string::npos;
            if (sent_pos != std::string::npos) {
              size_t b64_pos = m.content.find("data:image/", sent_pos);
              if (b64_pos != std::string::npos) {
                size_t end_pos = m.content.find_first_of(" \t\r\n)]}\"", b64_pos);
                m.image_data = m.content.substr(b64_pos, end_pos == std::string::npos ? std::string::npos : (end_pos - b64_pos));
                m.has_image = !m.image_data.empty();
              }
            }
          }
        }
        if (const char* c = mptr("reasoning_content")) { const char* z = c; m.reasoning_content = read_json_string(z); }
        if (const char* c = mptr("tool_call_id")) { const char* z = c; m.tool_call_id = read_json_string(z); }
        if (const JsonSpan* tcsp = find_field(mfields, "tool_calls")) {
          if (sub[tcsp->start] == '[') {
            for (const std::string& tc : raw_elements_range(sub, tcsp->start, tcsp->end)) {
              ToolCallIO c2;
              if (const char* z = find_top_key(tc, "id")) { const char* w = z; c2.id = read_json_string(w); }
              if (const char* z = find_top_key(tc, "namespace")) { const char* w = z; c2.ns = read_json_string(w); }
              const std::string fn = raw_value(tc, "function");
              const std::string& src = fn.empty() ? tc : fn;
              if (c2.ns.empty()) {
                if (const char* z = find_top_key(src, "namespace")) { const char* w = z; c2.ns = read_json_string(w); }
              }
              if (const char* z = find_top_key(src, "name")) { const char* w = z; c2.name = read_json_string(w); }
              if (const char* z = find_top_key(src, "arguments")) { const char* w = z; c2.arguments = read_json_string(w); }
              const size_t sep = c2.name.find("::");
              if (sep != std::string::npos) {
                if (c2.ns.empty()) c2.ns = c2.name.substr(0, sep);
                c2.name = c2.name.substr(sep + 2);
              }
              if (!c2.name.empty()) m.tool_calls.push_back(std::move(c2));
            }
          }
        }
      }
      if (!m.role.empty()) out->messages.push_back(std::move(m));
    }
    return !out->messages.empty();
  }
  if (const char* p = field_ptr("prompt")) { const char* q = p; out->prompt = read_json_string(q); }
  return !out->prompt.empty();
}

namespace {
struct ChunkedDecoder {
  size_t pos = 0;
  bool done = false;
  bool error = false;
  std::string out;

  // Advances over complete chunks in `buf` starting at `pos`.
  // Returns true if the terminal 0-length chunk + CRLF has been consumed.
  bool feed(const std::string& buf) {
    while (!done && pos < buf.size()) {
      const size_t eol = buf.find("\r\n", pos);
      if (eol == std::string::npos) break;
      const char* line = buf.c_str() + pos;
      char* endptr = nullptr;
      const unsigned long len = std::strtoul(line, &endptr, 16);
      if (endptr == line) { error = true; return false; }
      // Allow optional whitespace or ;chunk-ext after hex digits up to CRLF.
      for (const char* q = endptr; q < buf.c_str() + eol; ++q) {
        if (*q == ';') break;
        if (*q != ' ' && *q != '\t') { error = true; return false; }
      }
      if (len == 0) {
        if (buf.size() >= eol + 4 && buf.compare(eol + 2, 2, "\r\n") == 0) {
          pos = eol + 4;
          done = true;
          return true;
        }
        const size_t tr_end = buf.find("\r\n\r\n", eol);
        if (tr_end != std::string::npos) {
          pos = tr_end + 4;
          done = true;
          return true;
        }
        break;
      }
      if (buf.size() < eol + 2 + len + 2) break;
      if (buf[eol + 2 + len] != '\r' || buf[eol + 2 + len + 1] != '\n') {
        error = true;
        return false;
      }
      out.append(buf, eol + 2, len);
      pos = eol + 2 + len + 2;
    }
    return done;
  }
};
} // namespace

bool fd_alive(int fd) {
  if (fd < 0) return false;
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = POLLRDHUP;
  int pr = 0;
  do {
    pr = ::poll(&pfd, 1, 0);
  } while (pr < 0 && errno == EINTR);
  if (pr < 0) return false;
  if ((pfd.revents & (POLLERR | POLLHUP | POLLRDHUP)) != 0) return false;
  char b = 0;
  ssize_t n = 0;
  do {
    n = ::recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
  } while (n < 0 && errno == EINTR);
  if (n == 0) return false;
  if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) return false;
  return true;
}

std::string make_sse_keepalive_chunk(const std::string& envelope, const std::string& obj,
                                     const std::string& model_name, bool is_chat) {
  return "data: {" + envelope + obj + "\",\"model\":\"" + json_escape(model_name) +
         "\",\"choices\":[{\"index\":0," +
         std::string(is_chat ? "\"delta\":{}" : "\"text\":\"\"") + "}]}\n\n";
}

bool dechunk_http_body(const std::string& in, std::string* out) {
  ChunkedDecoder dec;
  const bool complete = dec.feed(in);
  if (dec.error) return false;
  *out = std::move(dec.out);
  return complete;
}

// Strict FIFO waiter queue waking every 1s to drop disconnected sockets and emit
// OpenAI-SDK-visible empty-delta SSE keepalive chunks every 15s for streaming waiters.
int32_t Server::acquire_slot(int fd, bool stream, const std::string& envelope,
                             const std::string& obj, bool is_chat,
                             std::chrono::steady_clock::time_point* last_send) {
  const uint64_t all_busy = n_slots_ >= 64 ? ~0ull : ((1ull << n_slots_) - 1);
  constexpr auto kWakeInterval = std::chrono::seconds(1);
  constexpr auto kKeepAlive = std::chrono::seconds(15);

  std::unique_lock<std::mutex> lk(slot_mu_);
  const uint64_t ticket = next_ticket_++;
  wait_queue_.push_back(ticket);

  auto remove_ticket = [&]() {
    auto it = std::find(wait_queue_.begin(), wait_queue_.end(), ticket);
    if (it != wait_queue_.end()) wait_queue_.erase(it);
    slot_cv_.notify_all();
  };

  for (;;) {
    if (!running_) {
      remove_ticket();
      return -1;
    }
    if (fd >= 0 && !fd_alive(fd)) {
      remove_ticket();
      return -1;
    }
    if (!wait_queue_.empty() && wait_queue_.front() == ticket && slot_busy_ != all_busy) {
      for (uint32_t i = 0; i < n_slots_; ++i) {
        if (!(slot_busy_ & (1ull << i))) {
          slot_busy_ |= (1ull << i);
          wait_queue_.pop_front();
          slot_cv_.notify_all();
          return (int32_t)i;
        }
      }
    }
    if (stream && fd >= 0 && last_send) {
      const auto now = std::chrono::steady_clock::now();
      if (now - *last_send >= kKeepAlive) {
        const std::string ka = make_sse_keepalive_chunk(envelope, obj, model_name_, is_chat);
        lk.unlock();
        const bool ok = send_all(fd, ka.data(), ka.size());
        lk.lock();
        if (!ok) {
          remove_ticket();
          return -1;
        }
        *last_send = now;
        continue;
      }
    }
    slot_cv_.wait_for(lk, kWakeInterval, [&] {
      return !running_ ||
             (!wait_queue_.empty() && wait_queue_.front() == ticket && slot_busy_ != all_busy);
    });
  }
}

void Server::release_slot(uint32_t s) {
  { std::lock_guard<std::mutex> lk(slot_mu_); slot_busy_ &= ~(1ull << s); }
  slot_cv_.notify_all();
}


void Server::serve_one(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  std::string req;
  char buf[8192];
  size_t header_end = std::string::npos, content_len = 0;
  bool chunked = false;
  ChunkedDecoder chunk_dec;
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      return;
    }
    if (n == 0) { ::close(fd); return; }
    req.append(buf, (size_t)n);
    if (header_end == std::string::npos) {
      header_end = req.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        // Header names are case-insensitive (RFC 7230). Lowercasing preserves offsets.
        std::string h = req.substr(0, header_end);
        for (char& c : h) c = (char)std::tolower((unsigned char)c);
        const size_t cl = h.find("\r\ncontent-length:");
        if (cl != std::string::npos)
          content_len = (size_t)std::strtoul(req.c_str() + cl + 17, nullptr, 10);
        const size_t te = h.find("\r\ntransfer-encoding:");
        if (te != std::string::npos) {
          const size_t te_end = h.find("\r\n", te + 20);
          const size_t chunk_at = h.find("chunked", te + 20);
          chunked = chunk_at != std::string::npos && (te_end == std::string::npos || chunk_at < te_end);
        }
        if (chunked) chunk_dec.pos = header_end + 4;
      }
    }
    if (header_end == std::string::npos) continue;
    if (chunked) {
      if (chunk_dec.feed(req)) break;
      if (chunk_dec.error) { ::close(fd); return; }
      continue;
    }
    if (req.size() >= header_end + 4 + content_len) break;
  }
  const std::string head = req.substr(0, header_end);
  const std::string body = chunked ? std::move(chunk_dec.out) : req.substr(header_end + 4);
  const bool is_chat = head.find("/v1/chat/completions") != std::string::npos;
  const bool is_comp = head.find("/v1/completions") != std::string::npos;
  const bool is_models = head.find("/v1/models") != std::string::npos;
  const bool is_health = head.find("/health") != std::string::npos;

  // A browser sends OPTIONS before any cross-origin POST and will not send the request itself
  // unless the preflight is answered. Without this every web client fails before it starts.
  if (head.rfind("OPTIONS ", 0) == 0) {
    const char* h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(fd, h, std::strlen(h));
    ::close(fd);
    return;
  }

  auto reply = [&](const std::string& ctype, const std::string& payload) {
    char h[512];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", ctype.c_str(), payload.size());
    send_all(fd, h, (size_t)n);
    send_all(fd, payload.data(), payload.size());
  };

  auto send_400 = [&](const std::string& msg) {
    const std::string e = "{\"error\":{\"message\":\"" + json_escape(msg) +
                          "\",\"type\":\"invalid_request_error\"}}";
    char h[256];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", e.size());
    send_all(fd, h, (size_t)n);
    send_all(fd, e.data(), e.size());
  };

  if (is_health) {
    std::string vram_json = "[";
    if (ctx_.get_free_vram_mib) {
      const std::vector<uint64_t> vrams = ctx_.get_free_vram_mib();
      for (size_t i = 0; i < vrams.size(); ++i) {
        if (i > 0) vram_json += ",";
        vram_json += std::to_string(vrams[i]);
      }
    }
    vram_json += "]";

    const int64_t resident = ctx_.get_resident_experts ? ctx_.get_resident_experts() : 0;
    const std::string ver = ctx_.version.empty() ? AFF_VERSION : ctx_.version;
    const std::string sha = ctx_.git_sha.empty() ? AFF_GIT_SHA : ctx_.git_sha;
    const std::string stamp = ctx_.build_stamp.empty() ? AFF_BUILD_STAMP : ctx_.build_stamp;
    const std::string st = ctx_.status.empty() ? "ok" : ctx_.status;

    const std::string j = "{\"status\":\"" + st + "\""
                          ",\"version\":\"" + ver + "\""
                          ",\"git_sha\":\"" + sha + "\""
                          ",\"build_stamp\":\"" + stamp + "\""
                          ",\"resident_experts\":" + std::to_string(resident) +
                          ",\"free_vram_mib\":" + vram_json + "}";
    reply("application/json", j);
    ::close(fd);
    return;
  }

  if (is_models) {
    const std::string j = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_name_ +
                          "\",\"object\":\"model\",\"created\":" +
                          std::to_string((long long)::time(nullptr)) +
                          ",\"owned_by\":\"affinity\"}]}";
    reply("application/json", j);
    ::close(fd);
    return;
  }
  if (!is_chat && !is_comp) {
    const std::string j = "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}";
    char h[256];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", j.size());
    send_all(fd, h, (size_t)n); send_all(fd, j.data(), j.size());
    ::close(fd);
    return;
  }

  CompletionRequest r;
  if (!parse_completion_request(body, is_chat, &r)) {
    send_400("bad request");
    ::close(fd);
    return;
  }
  if (ctx_.preflight_check) {
    const std::string err_msg = ctx_.preflight_check(r);
    if (!err_msg.empty()) {
      send_400(err_msg);
      ::close(fd);
      return;
    }
  }

  // `id` and `created` are required by the OpenAI schema on both the completion and every chunk of a
  // stream, and a client that validates its responses rejects the whole turn without them. One id
  // per request, repeated on each chunk, is what identifies the chunks as one completion.
  static std::atomic<uint64_t> seq{0};
  const std::string cmpl_id = "chatcmpl-" + std::to_string(seq.fetch_add(1)) + "-" +
                              std::to_string((uint64_t)::time(nullptr));
  const std::string envelope = "\"id\":\"" + cmpl_id + "\",\"created\":" +
                               std::to_string((long long)::time(nullptr)) + ",\"object\":\"";

  if (r.stream) {
    const std::string obj = is_chat ? "chat.completion.chunk" : "text_completion";
    const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    if (!send_all(fd, h, std::strlen(h))) {
      ::close(fd);
      return;
    }
    bool alive = true;
    GenResult res;
    constexpr auto kKeepAlive = std::chrono::seconds(15);
    auto last_send = std::chrono::steady_clock::now();
    // Held for the whole request: the slot owns this sequence's KV until the turn is done. Step-
    // level exclusion is the generator's business, not the server's.
    const int32_t slot_i = acquire_slot(fd, true, envelope, obj, is_chat, &last_send);
    if (slot_i < 0) {
      ::close(fd);
      return;
    }
    const uint32_t slot = (uint32_t)slot_i;
    struct Rel { Server* s; uint32_t i; ~Rel() { s->release_slot(i); } } rel{this, slot};
    // The first chat delta of a stream carries the role and nothing else to say it opens an
    // assistant message; clients key the new message on it, and OpenAI itself sends it.
    bool opened = false;
    gen_(r, [&](const std::string& piece, Delta chan, bool done) -> bool {
      if (!alive || done) return alive;
      if (piece.empty()) {
        if (!fd_alive(fd)) {
          alive = false;
          return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_send >= kKeepAlive) {
          const std::string ka = make_sse_keepalive_chunk(envelope, obj, model_name_, is_chat);
          alive = send_all(fd, ka.data(), ka.size());
          last_send = now;
        }
        return alive;
      }
      last_send = std::chrono::steady_clock::now();
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + json_escape(model_name_) +
                      "\",\"choices\":[{\"index\":0,";
      if (is_chat) {
        const std::string role = opened ? "" : "\"role\":\"assistant\",";
        opened = true;
        j += chan == Delta::Reasoning
                 ? "\"delta\":{" + role + "\"reasoning_content\":\"" + json_escape(piece) + "\"}"
                 : "\"delta\":{" + role + "\"content\":\"" + json_escape(piece) + "\"}";
        j += ",\"finish_reason\":null";
      } else {
        // The legacy endpoint has no channel to put reasoning in, so it carries the answer only.
        if (chan == Delta::Reasoning) return alive;
        j += "\"text\":\"" + json_escape(piece) + "\",\"finish_reason\":null";
      }
      j += "}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
      return alive;
    }, &res, slot);
    // The 200 and its headers are long gone by the time generation reports a problem, so a stream
    // carries the refusal as an `error` event — which is how OpenAI reports a mid-stream failure.
    if (alive && !res.error.empty()) {
      const std::string e = "data: {\"error\":{\"message\":\"" + json_escape(res.error) +
                            "\",\"type\":\"invalid_request_error\"}}\n\n";
      send_all(fd, e.data(), e.size());
      send_all(fd, "data: [DONE]\n\n", 14);
      ::close(fd);
      return;
    }
    // Tool calls arrive as one delta at the end rather than as text: they are not content, and a
    // client that concatenated them as content would have to re-parse markup this server already
    // parsed. The final chunk also carries the finish_reason, which is what tells the client a tool
    // round trip is expected rather than a finished answer.
    if (alive && is_chat && !res.tool_calls.empty()) {
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + json_escape(model_name_) +
                      "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[";
      for (size_t i = 0; i < res.tool_calls.size(); ++i) {
        if (i) j += ",";
        j += "{\"index\":" + std::to_string(i) + ",\"id\":\"" + json_escape(res.tool_calls[i].id) +
             "\",\"type\":\"function\"";
        if (!res.tool_calls[i].ns.empty())
          j += ",\"namespace\":\"" + json_escape(res.tool_calls[i].ns) + "\"";
        j += ",\"function\":{\"name\":\"" +
             json_escape(res.tool_calls[i].name) + "\",\"arguments\":\"" +
             json_escape(res.tool_calls[i].arguments) + "\"}}";
      }
      j += "]},\"finish_reason\":null}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    if (alive) {
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + json_escape(model_name_) +
                      "\",\"choices\":[{\"index\":0,";
      j += is_chat ? "\"delta\":{}" : "\"text\":\"\"";
      j += ",\"finish_reason\":\"" + res.finish_reason + "\"}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    // OpenAI puts usage in one extra chunk with an EMPTY choices array, only when asked for it.
    if (alive && r.include_usage) {
      const std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + json_escape(model_name_) +
                            "\",\"choices\":[]," + usage_json(res) + "}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    if (alive) send_all(fd, "data: [DONE]\n\n", 14);
    ::close(fd);
    if (res.error.empty() && res.post_send) res.post_send();
    return;
  }

  GenResult res;
  const int32_t slot_i = acquire_slot(fd, false, envelope, "", is_chat, nullptr);
  if (slot_i < 0) {
    ::close(fd);
    return;
  }
  const uint32_t slot = (uint32_t)slot_i;
  struct Rel { Server* s; uint32_t i; ~Rel() { s->release_slot(i); } } rel{this, slot};
  gen_(r, [fd](const std::string&, Delta, bool) -> bool { return fd_alive(fd); }, &res, slot);
  if (!res.error.empty()) {
    send_400(res.error);
    ::close(fd);
    return;
  }
  std::string j = "{" + envelope + std::string(is_chat ? "chat.completion" : "text_completion") +
                  "\",\"model\":\"" + json_escape(model_name_) + "\",\"choices\":[{\"index\":0,";
  if (is_chat) {
    j += "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(res.content) + "\"";
    // Alongside content, as DeepSeek's own API returns it. Not in the OpenAI schema, and clients
    // that do not know it ignore it; a client that does gets the reasoning without having to strip
    // <think> out of the text itself.
    if (!res.reasoning_content.empty())
      j += ",\"reasoning_content\":\"" + json_escape(res.reasoning_content) + "\"";
    if (!res.tool_calls.empty()) {
      j += ",\"tool_calls\":[";
      for (size_t i = 0; i < res.tool_calls.size(); ++i) {
        if (i) j += ",";
        j += "{\"id\":\"" + json_escape(res.tool_calls[i].id) +
             "\",\"type\":\"function\"";
        if (!res.tool_calls[i].ns.empty())
          j += ",\"namespace\":\"" + json_escape(res.tool_calls[i].ns) + "\"";
        j += ",\"function\":{\"name\":\"" +
             json_escape(res.tool_calls[i].name) + "\",\"arguments\":\"" +
             json_escape(res.tool_calls[i].arguments) + "\"}}";
      }
      j += "]";
    }
    j += "}";
  } else {
    j += "\"text\":\"" + json_escape(res.content) + "\"";
  }
  j += ",\"finish_reason\":\"" + res.finish_reason + "\"}]," + usage_json(res) + "}";
  reply("application/json", j);
  ::close(fd);
  if (res.post_send) res.post_send();
}

bool Server::start(const std::string& host, uint16_t port, GenerateFn gen, std::string* err) {
  stop();
  gen_ = std::move(gen);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) { if (err) *err = "socket failed"; return false; }
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = (host == "0.0.0.0") ? INADDR_ANY : inet_addr(host.c_str());
  if (::bind(listen_fd_, (sockaddr*)&a, sizeof(a)) != 0) {
    if (err) *err = "bind failed";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::listen(listen_fd_, 64) != 0) {
    if (err) *err = "listen failed";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  socklen_t sl = sizeof(a);
  ::getsockname(listen_fd_, (sockaddr*)&a, &sl);
  port_ = ntohs(a.sin_port);
  running_ = true;
  accept_thread_ = std::thread([this] {
    (void)pin_thread_numa0();
    while (running_) {
      const int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        if (!running_) break;
        if (errno == EINTR) continue;
        break;
      }
      {
        std::lock_guard<std::mutex> lk(active_mu_);
        ++active_workers_;
      }
      // One thread per connection. With a latency-first scheduler at B<=4 the connection count is
      // small by design, so a thread pool would be complexity without benefit.
      std::thread([this, fd] {
        (void)pin_thread_numa0();
        struct WorkerGuard {
          Server* s;
          ~WorkerGuard() {
            std::lock_guard<std::mutex> lk(s->active_mu_);
            if (--s->active_workers_ == 0) s->active_cv_.notify_all();
          }
        } wg{this};
        serve_one(fd);
      }).detach();
    }
  });
  return true;
}

void Server::stop() {
  running_ = false;
  slot_cv_.notify_all();
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (accept_thread_.joinable()) accept_thread_.join();
  std::unique_lock<std::mutex> lk(active_mu_);
  active_cv_.wait(lk, [this] { return active_workers_ == 0; });
}

} // namespace aff
