#include "engine/chat_encode.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>

namespace aff {

namespace {

const char* const kBos        = "<｜begin▁of▁sentence｜>";
const char* const kUser       = "<｜User｜>";
const char* const kAssistant  = "<｜Assistant｜>";
const char* const kReminder   = "<｜latest_reminder｜>";
const char* const kThinkStart = "<think>";
const char* const kDsml       = "｜DSML｜";
const char* const kSystem     = "<｜System｜>";           // V4.1

// Verbatim from REASONING_EFFORT_PROMPTS. The trailing blank line is part of the prefix.
const char* const kEffortHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const char* const kEffortMax =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

// TOOLS_TEMPLATE with dsml_token/think tokens already substituted. Every space and newline here is
// load-bearing: the fixtures compare byte for byte.
std::string tools_block(const std::vector<ToolDef>& tools, bool v41 = false);
std::string response_format_block(const std::string& schema_json);

// ---- JSON, only as much as the format needs -----------------------------------------------------
//
// The reference emits tool schemas with `json.dumps(ensure_ascii=False)`, which keeps insertion
// order and uses ", " and ": " as separators. Rebuilding that from a parsed value model would mean
// re-spelling every number and re-escaping every string the way Python happens to, and any
// disagreement is a byte the fixture catches.
//
// So this REFORMATS the caller's JSON text instead: it walks the tokens, drops whitespace between
// them, and puts back exactly Python's separators. Key order, string escapes and the literal text of
// every number survive untouched, because none of them is ever re-derived.
//
// The one thing it cannot reproduce is Python renormalising a number's spelling (1e5 -> 100000.0).

// Tool schemas are objects, strings and arrays; no fixture contains a float, and a schema that did
// would differ only in the digits of a number the model reads as text.
std::string json_compact(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  bool in_str = false, esc = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char ch = s[i];
    if (in_str) {
      out += ch;
      if (esc)            esc = false;
      else if (ch == '\\') esc = true;
      else if (ch == '"')  in_str = false;
      continue;
    }
    switch (ch) {
      case ' ': case '\t': case '\n': case '\r': continue;   // whitespace between tokens
      case '"': in_str = true; out += ch; break;
      case ',': out += ", "; break;
      case ':': out += ": "; break;
      default:  out += ch; break;
    }
  }
  return out;
}

// Whether a JSON value's text is a string literal, which is what decides `string="true|false"`.
bool json_is_string(const std::string& v) {
  size_t i = 0;
  while (i < v.size() && (v[i] == ' ' || v[i] == '\t' || v[i] == '\n' || v[i] == '\r')) ++i;
  return i < v.size() && v[i] == '"';
}

// The text of a JSON string literal, unescaped. Only the escapes the format can produce.
std::string json_unquote(const std::string& v) {
  size_t i = v.find('"');
  if (i == std::string::npos) return v;
  std::string out;
  for (++i; i < v.size(); ++i) {
    if (v[i] == '"') break;
    if (v[i] != '\\') { out += v[i]; continue; }
    if (++i >= v.size()) break;
    switch (v[i]) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'u': {
        // \uXXXX. Only the BMP, and surrogate pairs are joined, which is what the encoder's own
        // ensure_ascii=False output round-trips to.
        if (i + 4 >= v.size()) return out;
        unsigned cp = (unsigned)std::stoul(v.substr(i + 1, 4), nullptr, 16);
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < v.size() && v[i + 1] == '\\' && v[i + 2] == 'u') {
          const unsigned lo = (unsigned)std::stoul(v.substr(i + 3, 4), nullptr, 16);
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          i += 6;
        }
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
          out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
          out += (char)(0x80 | (cp & 0x3F));
        } else {
          out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
          out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: out += v[i]; break;
    }
  }
  return out;
}

// A JSON string literal for `s`, matching json.dumps(ensure_ascii=False): only the mandatory
// escapes, UTF-8 passed through.
std::string json_quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char ch : s) {
    switch (ch) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\t': out += "\\t";  break;
      case '\r': out += "\\r";  break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      default:
        if (ch < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00"; out += hex[(ch >> 4) & 0xF]; out += hex[ch & 0xF];
        } else {
          out += (char)ch;
        }
    }
  }
  return out + "\"";
}

// Split a JSON object's top-level members into (key, raw value text) in order. Not a validator: the
// arguments it is handed came from the model or from a client, and a malformed one is reported by
// the caller rather than diagnosed here.
std::vector<std::pair<std::string, std::string>> json_members(const std::string& obj) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = obj.find('{');
  if (i == std::string::npos) return out;
  ++i;
  auto skip_ws = [&] { while (i < obj.size() && (obj[i]==' '||obj[i]=='\t'||obj[i]=='\n'||obj[i]=='\r')) ++i; };
  while (i < obj.size()) {
    skip_ws();
    if (i < obj.size() && obj[i] == '}') break;
    if (i >= obj.size() || obj[i] != '"') break;
    const size_t ks = i;
    bool esc = false;
    for (++i; i < obj.size(); ++i) {
      if (esc) { esc = false; continue; }
      if (obj[i] == '\\') { esc = true; continue; }
      if (obj[i] == '"') break;
    }
    if (i >= obj.size()) break;
    const std::string key = json_unquote(obj.substr(ks, ++i - ks));
    skip_ws();
    if (i >= obj.size() || obj[i] != ':') break;
    ++i;
    skip_ws();
    // The value: one token, or a balanced run for a container or string.
    const size_t vs = i;
    int depth = 0;
    bool in_str = false;
    esc = false;
    for (; i < obj.size(); ++i) {
      const char ch = obj[i];
      if (in_str) {
        if (esc) esc = false;
        else if (ch == '\\') esc = true;
        else if (ch == '"') in_str = false;
        continue;
      }
      if (ch == '"') { in_str = true; continue; }
      if (ch == '{' || ch == '[') { ++depth; continue; }
      if (ch == '}' || ch == ']') { if (depth == 0) break; --depth; continue; }
      if (ch == ',' && depth == 0) break;
    }
    std::string val = obj.substr(vs, i - vs);
    while (!val.empty() && (val.back()==' '||val.back()=='\t'||val.back()=='\n'||val.back()=='\r'))
      val.pop_back();
    out.emplace_back(key, val);
    if (i < obj.size() && obj[i] == ',') ++i;
  }
  return out;
}

// response_format_template from the reference: the schema, verbatim, under a fixed heading.
std::string response_format_block(const std::string& schema_json) {
  return "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" +
         json_compact(schema_json);
}

std::string apply_tool_namespace(const ToolDef& td) {
  std::string ns = td.ns;
  std::string ns_desc = td.ns_desc;
  auto members = json_members(td.function_json);
  if (members.empty()) return json_compact(td.function_json);
  for (const auto& kv : members) {
    if (kv.first == "namespace") {
      if (json_is_string(kv.second)) {
        if (ns.empty()) ns = json_unquote(kv.second);
      } else {
        for (const auto& nkv : json_members(kv.second)) {
          if (nkv.first == "name" && ns.empty()) ns = json_unquote(nkv.second);
          if (nkv.first == "description" && ns_desc.empty()) ns_desc = json_unquote(nkv.second);
        }
      }
    }
  }
  if (ns.empty() && ns_desc.empty()) return json_compact(td.function_json);
  std::string out = "{";
  bool first = true;
  bool had_desc = false;
  for (const auto& kv : members) {
    if (kv.first == "namespace") continue;
    if (!first) out += ", ";
    first = false;
    if (kv.first == "name" && !ns.empty() && json_is_string(kv.second)) {
      std::string bare = json_unquote(kv.second);
      if (bare.find("::") == std::string::npos) bare = ns + "::" + bare;
      out += json_quote("name") + ": " + json_quote(bare);
    } else if (kv.first == "description" && !ns_desc.empty() && json_is_string(kv.second)) {
      had_desc = true;
      std::string d = json_unquote(kv.second);
      out += json_quote("description") + ": " + json_quote(ns_desc + "\n" + d);
    } else {
      out += json_quote(kv.first) + ": " + json_compact(kv.second);
    }
  }
  if (!ns_desc.empty() && !had_desc) {
    if (!first) out += ", ";
    out += json_quote("description") + ": " + json_quote(ns_desc + "\n");
  }
  out += "}";
  return out;
}

std::string tools_block(const std::vector<ToolDef>& tools, bool v41) {
  std::string schemas;
  for (size_t i = 0; i < tools.size(); ++i) {
    if (i) schemas += "\n";
    schemas += apply_tool_namespace(tools[i]);
  }
  const std::string tag_calls = v41 ? " calls" : "tool_calls";
  const std::string tag_inv   = v41 ? " invoke" : "invoke";
  const std::string tag_param = v41 ? " parameter" : "parameter";

  std::string s = "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
                  "You can invoke tools by writing a \"<";
  s += kDsml; s += tag_calls; s += ">\" block like the following:\n\n<";
  s += kDsml; s += tag_calls; s += ">\n<";
  s += kDsml; s += tag_inv; s += " name=\"$TOOL_NAME\">\n<";
  s += kDsml; s += tag_param; s += " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</";
  s += kDsml; s += tag_param; s += ">\n...\n</";
  s += kDsml; s += tag_inv; s += ">\n<";
  s += kDsml; s += tag_inv; s += " name=\"$TOOL_NAME2\">\n...\n</";
  s += kDsml; s += tag_inv; s += ">\n</";
  s += kDsml; s += tag_calls; s += ">\n\nString parameters should be specified as is and set "
                  "`string=\"true\"`. For all other types (numbers, booleans, arrays, objects), "
                  "pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is "
                  "enabled (triggered by ";
  s += kThinkStart; s += "), you MUST output your complete reasoning inside ";
  s += kThinkStart; s += "..."; s += kThinkEnd;
  s += " BEFORE any tool calls or final response.\n\nOtherwise, output directly after ";
  s += kThinkEnd; s += " with tool calls or final response.\n\n### Available Tool Schemas\n\n";
  s += schemas;
  s += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke "
       "tool calls.\n";
  return s;
}

// One assistant turn's DSML block, from `encode_arguments_to_dsml` and `tool_call_template`.
std::string render_tool_calls(const std::vector<ToolCall>& tcs, bool v41 = false) {
  const std::string tag_calls = v41 ? " calls" : "tool_calls";
  const std::string tag_inv   = v41 ? " invoke" : "invoke";
  const std::string tag_param = v41 ? " parameter" : "parameter";

  std::string out = "\n\n<";
  out += kDsml; out += tag_calls; out += ">\n";
  for (size_t i = 0; i < tcs.size(); ++i) {
    if (i) out += "\n";
    std::string qname = tcs[i].name;
    if (!tcs[i].ns.empty() && qname.find("::") == std::string::npos)
      qname = tcs[i].ns + "::" + qname;
    out += "<"; out += kDsml; out += tag_inv; out += " name=\"" + qname + "\">\n";
    // Arguments that are not a JSON object are wrapped under an "arguments" key, exactly as the
    // reference's except-branch does, rather than dropped.
    auto members = json_members(tcs[i].arguments);
    std::vector<std::pair<std::string, std::string>> fallback;
    if (members.empty() && !tcs[i].arguments.empty() &&
        tcs[i].arguments.find('{') == std::string::npos) {
      fallback.emplace_back("arguments", json_quote(tcs[i].arguments));
      members = fallback;
    }
    for (size_t m = 0; m < members.size(); ++m) {
      if (m) out += "\n";
      const bool is_str = json_is_string(members[m].second);
      out += "<"; out += kDsml; out += tag_param;
      out += " name=\"" + members[m].first + "\" string=\"";
      out += is_str ? "true" : "false";
      out += "\">";
      out += is_str ? json_unquote(members[m].second) : json_compact(members[m].second);
      out += "</"; out += kDsml; out += tag_param; out += ">";
    }
    out += "\n</"; out += kDsml; out += tag_inv; out += ">";
  }
  out += "\n</"; out += kDsml; out += tag_calls; out += ">";
  return out;
}

int last_user_index(const std::vector<ChatMsg>& m) {
  for (int i = (int)m.size() - 1; i >= 0; --i)
    if (m[(size_t)i].role == "user" || m[(size_t)i].role == "developer") return i;
  return -1;
}

const char* task_token(const std::string& t) {
  if (t == "action")    return "<｜action｜>";
  if (t == "query")     return "<｜query｜>";
  if (t == "authority") return "<｜authority｜>";
  if (t == "domain")    return "<｜domain｜>";
  if (t == "title")     return "<｜title｜>";
  if (t == "read_url")  return "<｜read_url｜>";
  return nullptr;
}

// merge_tool_messages: the format has no standalone tool role, so a run of tool results becomes one
// user message whose content is the results joined by "\n\n", each in a <tool_result> wrapper. A
// user message following them joins the same block.
std::vector<ChatMsg> merge_tool_messages(const std::vector<ChatMsg>& in) {
  std::vector<ChatMsg> out;
  bool last_was_merged = false;
  for (const ChatMsg& m : in) {
    if (m.role == "tool") {
      std::string block = "<tool_result>" + m.content + "</tool_result>";
      if (last_was_merged && !out.empty()) out.back().content += "\n\n" + block;
      else {
        ChatMsg u; u.role = "user"; u.content = block;
        out.push_back(u);
        last_was_merged = true;
      }
      continue;
    }
    if (m.role == "user" && last_was_merged && !out.empty() && out.back().task.empty()) {
      out.back().content += "\n\n" + m.content;
      out.back().task = m.task;
      continue;
    }
    out.push_back(m);
    last_was_merged = false;
  }
  return out;
}

}  // namespace

const char* const kToolCallsOpen    = "\n\n<｜DSML｜tool_calls";
const char* const kToolCallsOpenV41 = "\n\n<｜DSML｜ calls";
const char* const kThinkEnd         = "</think>";
const char* const kEosStr           = "<｜end▁of▁sentence｜>";

std::string encode_messages(const std::vector<ChatMsg>& in, const EncodeOpts& opts) {
  std::vector<ChatMsg> msgs = merge_tool_messages(in);

  // Tools anywhere force the whole conversation to keep its reasoning.
  bool has_tools = false;
  for (const ChatMsg& m : msgs) has_tools = has_tools || !m.tools.empty();
  const bool thinking = opts.thinking_mode == ThinkingMode::Thinking;
  const bool drop = opts.drop_thinking && !has_tools;

  const int last_user = last_user_index(msgs);

  std::string out = opts.add_bos ? kBos : "";
  for (size_t idx = 0; idx < msgs.size(); ++idx) {
    const ChatMsg& m = msgs[idx];

    if (opts.v41) {
      // render_reasoning_effort + the SYSTEM_SP_TOKEN rule in V4.1's `_render_message`: the token
      // leads message 0 when there is an effort prefix or message 0 is a system message, and a
      // system message anywhere later carries its own.
      const bool effort = idx == 0 && thinking;
      if (idx == 0 && (effort || m.role == "system")) out += kSystem;
      if (effort)
        out += "Reasoning Effort: " + std::to_string(opts.effort_budget) +
               " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
      if (idx > 0 && m.role == "system") out += kSystem;
    } else if (idx == 0 && thinking) {
      if (opts.reasoning_effort == ReasoningEffort::High) out += kEffortHigh;
      else if (opts.reasoning_effort == ReasoningEffort::Max) out += kEffortMax;
    }

    if (m.role == "system") {
      out += m.content;
      if (!m.tools.empty()) out += "\n\n" + tools_block(m.tools, opts.v41);
      if (!m.response_format.empty()) out += "\n\n" + response_format_block(m.response_format);
    } else if (m.role == "developer") {
      out += kUser;
      out += m.content;
      if (!m.tools.empty()) out += "\n\n" + tools_block(m.tools, opts.v41);
      if (!m.response_format.empty()) out += "\n\n" + response_format_block(m.response_format);
    } else if (m.role == "user") {
      out += kUser;
      out += m.content;
    } else if (m.role == "latest_reminder") {
      out += kReminder;
      out += m.content;
    } else if (m.role == "assistant") {
      // A turn that answers a quick-instruction task carries no thinking at all.
      const bool prev_task = idx > 0 && !msgs[idx - 1].task.empty();
      if (thinking && !prev_task && (!drop || (int)idx > last_user))
        out += m.reasoning_content + kThinkEnd;
      out += m.content;
      if (!m.tool_calls.empty()) out += render_tool_calls(m.tool_calls, opts.v41);
      out += kEosStr;
    }

    // The transition, which only fires when an assistant turn (or the end) follows.
    if (idx + 1 < msgs.size() && msgs[idx + 1].role != "assistant" &&
        msgs[idx + 1].role != "latest_reminder")
      continue;

    if (const char* tt = task_token(m.task)) {
      if (m.task != "action") {
        out += tt;
      } else {
        out += kAssistant;
        const bool is_final = (idx + 1 == msgs.size());
        out += (thinking && !(opts.suppress_final_think && is_final)) ? kThinkStart : kThinkEnd;
        out += tt;
      }
    } else if (m.role == "user" || m.role == "developer") {
      out += kAssistant;
      const bool is_final = (idx + 1 == msgs.size());
      if (thinking && (!drop || (int)idx >= last_user) && !(opts.suppress_final_think && is_final))
        out += kThinkStart;
      else
        out += kThinkEnd;
    }
  }
  return out;
}

// ---- parsing ------------------------------------------------------------------------------------

namespace {

// The reference's _read_until_stop: the EARLIEST of the stops wins, not the first listed.
[[maybe_unused]] size_t read_until(size_t i, const std::string& t, const std::vector<std::string>& stops,
                                   std::string* content, std::string* matched) {
  size_t best = t.size();
  const std::string* hit = nullptr;
  for (const std::string& s : stops) {
    const size_t p = t.find(s, i);
    if (p != std::string::npos && p < best) { best = p; hit = &s; }
  }
  *content = t.substr(i, best - i);
  if (hit) { *matched = *hit; return best + hit->size(); }
  matched->clear();
  return t.size();
}

}  // namespace

// ---- DSML repair (bare tags, truncated V4.1 wrappers, doubled bars, aborted drafts) ----------
//
// DeepSeek-V4.1 occasionally glitches when opening or closing a DSML tool-call block:
//   1. Truncated opening wrapper: `<｜DSML｜>\n<｜DSML｜ invoke…`, `<｜DSML｜\n\n<｜DSML｜ invoke…`,
//      `<｜DSML｜<｜DSML｜ invoke…`, or bare `<｜DSML｜ invoke…` with the `<｜DSML｜ calls>` wrapper line
//      dropped entirely.
//   2. Doubled DSML bar token: `</｜DSML｜｜DSML｜ parameter>`.
//   3. Marker-less `<invoke name="X">…</invoke>` with the DSML bar dropped completely.
//   4. Aborted tool-call draft mid-parameter followed by deliberation, a second `</think>`, and a
//      fresh `<｜DSML｜ calls>` block.
//
// Repairs operate only on the post-`</think>` span in Thinking mode so prose quoting `<invoke` or
// `｜DSML｜` inside `<think>…</think>` is never mutated into a bogus tool call.
std::string repair_bare_dsml(const std::string& s, ThinkingMode mode) {
  const std::string ds(kDsml);
  std::string t = s;

  // Case 4: in Thinking mode, if an unclosed tool-call draft after `</think>` was abandoned and the
  // model emitted another `</think>` followed immediately by a new `<｜DSML｜` block, drop the
  // abandoned draft between the two `</think>` markers so the final block parses cleanly.
  if (mode == ThinkingMode::Thinking) {
    const std::string te_str(kThinkEnd);
    const size_t te1 = t.find(te_str);
    if (te1 != std::string::npos) {
      size_t best_te2 = std::string::npos;
      size_t scan = te1 + te_str.size();
      while (scan < t.size()) {
        const size_t te2 = t.find(te_str, scan);
        if (te2 == std::string::npos) break;
        const size_t c1 = t.find("</" + ds + " calls>", te1);
        const size_t c2 = t.find("</" + ds + "tool_calls>", te1);
        const bool closed_before = (c1 != std::string::npos && c1 < te2) ||
                                   (c2 != std::string::npos && c2 < te2);
        size_t after = te2 + te_str.size();
        while (after < t.size() &&
               (t[after] == ' ' || t[after] == '\t' || t[after] == '\n' || t[after] == '\r'))
          ++after;
        if (!closed_before && t.compare(after, 1 + ds.size(), "<" + ds) == 0) {
          best_te2 = te2;
        }
        scan = te2 + te_str.size();
      }
      if (best_te2 != std::string::npos) {
        t.erase(te1, best_te2 - te1);
      }
    }
  }

  size_t body_start = 0;
  if (mode == ThinkingMode::Thinking) {
    const size_t te = t.find(kThinkEnd);
    if (te == std::string::npos) return t;
    body_start = te + std::string(kThinkEnd).size();
  }

  std::string body = t.substr(body_start);
  const std::string eos(kEosStr);
  const bool had_eos = body.size() >= eos.size() &&
                       body.compare(body.size() - eos.size(), eos.size(), eos) == 0;

  // Case 2: collapse doubled `｜DSML｜｜DSML｜` tokens in the post-think body.
  {
    const std::string dup = ds + ds;
    size_t p = 0;
    while ((p = body.find(dup, p)) != std::string::npos) {
      body.replace(p, dup.size(), ds);
    }
  }

  auto rep_body = [&](const std::string& from, const std::string& to) {
    size_t p = 0;
    while ((p = body.find(from, p)) != std::string::npos) {
      body.replace(p, from.size(), to);
      p += to.size();
    }
  };

  // Case 3: marker-less `<invoke name="X">` when no marked DSML tags exist in `body`.
  const bool has_marked_calls = body.find("<" + ds + "tool_calls") != std::string::npos ||
                                body.find("<" + ds + " calls") != std::string::npos;
  const bool has_marked_inv   = body.find("<" + ds + "invoke") != std::string::npos ||
                                body.find("<" + ds + " invoke") != std::string::npos;
  if (!has_marked_calls && !has_marked_inv && body.find("<invoke name=") != std::string::npos) {
    rep_body("<invoke name=", "<" + ds + "invoke name=");
    rep_body("</invoke>", "</" + ds + "invoke>");
    rep_body("<parameter name=", "<" + ds + "parameter name=");
    rep_body("</parameter>", "</" + ds + "parameter>");
    const std::string wrapper = "<" + ds + "tool_calls";
    const std::string io = "<" + ds + "invoke";
    const std::string ic = "</" + ds + "invoke>";
    const size_t o = body.find(io);
    const size_t c = body.rfind(ic);
    if (o != std::string::npos && c != std::string::npos && c >= o) {
      body.resize(c + ic.size());
      body.insert(o, "\n\n" + wrapper + ">\n");
      body += "\n</" + ds + "tool_calls>";
      if (had_eos) body += eos;
    }
    return t.substr(0, body_start) + body;
  }

  // Case 1: marked `<｜DSML｜ invoke` / `<｜DSML｜invoke` present, but the opening
  // `<｜DSML｜ calls>` / `<｜DSML｜tool_calls>` wrapper was truncated to `<｜DSML｜>`, `<｜DSML｜\n\n`,
  // `<｜DSML｜<｜DSML｜ invoke`, `<｜DSML｜calls>`, or omitted completely.
  if (!has_marked_calls && has_marked_inv) {
    const size_t p_v41 = body.find("<" + ds + " invoke");
    const size_t p_v4  = body.find("<" + ds + "invoke");
    const bool is_v41  = p_v41 != std::string::npos && (p_v4 == std::string::npos || p_v41 <= p_v4);
    const size_t inv_pos = is_v41 ? p_v41 : p_v4;

    size_t open_pos = inv_pos;
    const std::string ds_open = "<" + ds;
    if (inv_pos > 0) {
      const size_t cand = body.rfind(ds_open, inv_pos - 1);
      if (cand != std::string::npos) {
        const size_t mid = cand + ds_open.size();
        const std::string between = body.substr(mid, inv_pos - mid);
        if (between.find_first_not_of(" >\n\r\tcalls") == std::string::npos) {
          open_pos = cand;
        }
      }
    }
    const std::string fixed_open = "<" + ds + (is_v41 ? " calls>\n" : "tool_calls>\n");
    body.replace(open_pos, inv_pos - open_pos, fixed_open);
  }

  // Also normalize truncated closing wrappers (`</｜DSML｜>` or `</｜DSML｜calls>`) if a calls block
  // is open.
  if (body.find("<" + ds + " calls") != std::string::npos ||
      body.find("<" + ds + "tool_calls") != std::string::npos) {
    const bool is_v41 = body.find("<" + ds + " calls") != std::string::npos;
    const std::string want_close = "</" + ds + (is_v41 ? " calls>" : "tool_calls>");
    rep_body("</" + ds + ">", want_close);
    rep_body("</" + ds + "calls>", want_close);
    if (body.find("</" + ds + " calls>") == std::string::npos &&
        body.find("</" + ds + "tool_calls>") == std::string::npos) {
      const std::string ic_v41 = "</" + ds + " invoke>";
      const std::string ic_v4  = "</" + ds + "invoke>";
      const size_t c1 = body.rfind(ic_v41);
      const size_t c2 = body.rfind(ic_v4);
      size_t end_inv = std::string::npos;
      if (c1 != std::string::npos) end_inv = c1 + ic_v41.size();
      if (c2 != std::string::npos && (end_inv == std::string::npos || c2 + ic_v4.size() > end_inv))
        end_inv = c2 + ic_v4.size();
      if (end_inv != std::string::npos) {
        body.resize(end_inv);
        body += "\n" + want_close;
        if (had_eos) body += eos;
      }
    }
  }

  return t.substr(0, body_start) + body;
}

namespace {

// Counts open and close tags from `from` forward to check whether a candidate tag sits inside
// an unclosed tool call or invoke.
class QuotedCall {
 public:
  QuotedCall(const std::string& text, size_t from, std::string open, std::string close)
      : text_(text), open_(std::move(open)), close_(std::move(close)),
        next_open_(text.find(open_, from)), next_close_(text.find(close_, from)) {}

  bool open_before(size_t to) {
    for (; next_open_ != std::string::npos && next_open_ + open_.size() <= to;
         next_open_ = text_.find(open_, next_open_ + open_.size())) {
      ++depth_;
    }
    for (; next_close_ != std::string::npos && next_close_ + close_.size() <= to;
         next_close_ = text_.find(close_, next_close_ + close_.size())) {
      --depth_;
    }
    return depth_ > 0;
  }

 private:
  const std::string& text_;
  std::string open_, close_;
  size_t next_open_, next_close_;
  long depth_ = 0;
};

size_t skip_ws_local(const std::string& s, size_t pos) {
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

bool is_valid_json_non_string(const std::string& raw) {
  size_t p = skip_ws_local(raw, 0);
  if (p >= raw.size()) return false;
  if (raw.compare(p, 4, "true") == 0) return skip_ws_local(raw, p + 4) == raw.size();
  if (raw.compare(p, 5, "false") == 0) return skip_ws_local(raw, p + 5) == raw.size();
  if (raw.compare(p, 4, "null") == 0) return skip_ws_local(raw, p + 4) == raw.size();
  if (raw[p] == '-' || (raw[p] >= '0' && raw[p] <= '9')) {
    char* endptr = nullptr;
    (void)std::strtod(raw.c_str() + p, &endptr);
    if (!endptr || endptr == raw.c_str() + p) return false;
    return skip_ws_local(raw, (size_t)(endptr - raw.c_str())) == raw.size();
  }
  if (raw[p] == '[' || raw[p] == '{') {
    std::vector<char> stack;
    bool in_str = false, esc = false;
    for (size_t i = p; i < raw.size(); ++i) {
      char c = raw[i];
      if (in_str) {
        if (esc) esc = false;
        else if (c == '\\') esc = true;
        else if (c == '"') in_str = false;
        continue;
      }
      if (c == '"') in_str = true;
      else if (c == '[' || c == '{') stack.push_back(c);
      else if (c == ']') {
        if (stack.empty() || stack.back() != '[') return false;
        stack.pop_back();
        if (stack.empty()) return skip_ws_local(raw, i + 1) == raw.size();
      } else if (c == '}') {
        if (stack.empty() || stack.back() != '{') return false;
        stack.pop_back();
        if (stack.empty()) return skip_ws_local(raw, i + 1) == raw.size();
      }
    }
    return stack.empty() && !in_str;
  }
  return false;
}

}  // namespace

ParsedMessage parse_completion(const std::string& text_in, ThinkingMode mode) {
  ParsedMessage out;
  const std::string text = repair_bare_dsml(text_in, mode);
  const std::string dsml = kDsml;

  auto fail = [&](const char* why) {
    out.ok = false;
    out.error = why;
    out.content = text;              // the raw text, which is all a caller can honestly be given
    out.reasoning_content.clear();
    out.tool_calls.clear();
    return out;
  };

  size_t i = 0;
  if (mode == ThinkingMode::Thinking) {
    const std::string te_str(kThinkEnd);
    size_t te = text.find(te_str);
    if (te == std::string::npos) return fail("missing </think>");
    out.reasoning_content = text.substr(0, te);
    i = te + te_str.size();
  }

  const std::string tc_open_v41 = "<" + dsml + " calls";
  const std::string tc_open_v4  = "<" + dsml + "tool_calls";
  const std::string tc_open_xml = "<tool_call>";
  const std::string inv_bare    = "<invoke name=\"";

  const std::vector<std::string> open_markers = {
    "\n\n" + tc_open_v41, "\n\n" + tc_open_v4,
    tc_open_v41, tc_open_v4,
    "\n\n" + tc_open_xml, tc_open_xml
  };

  std::string accumulated_content;

  while (i < text.size()) {
    size_t next_block = std::string::npos;
    size_t opener_len = 0;
    bool is_bare_inv = false;

    for (const auto& marker : open_markers) {
      size_t pos = text.find(marker, i);
      if (pos != std::string::npos && pos < next_block) {
        next_block = pos;
        opener_len = marker.size();
        is_bare_inv = false;
      }
    }

    if (next_block == std::string::npos) {
      size_t pos = text.find(inv_bare, i);
      if (pos != std::string::npos && pos < next_block) {
        next_block = pos;
        opener_len = 0;
        is_bare_inv = true;
      }
    }

    if (next_block == std::string::npos) {
      const std::string eos(kEosStr);
      size_t eos_pos = text.find(eos, i);
      size_t end_pos = (eos_pos != std::string::npos) ? eos_pos : text.size();
      std::string tail = text.substr(i, end_pos - i);
      if (!tail.empty()) {
        if (!accumulated_content.empty() && accumulated_content.back() != '\n' && tail.front() != '\n')
          accumulated_content += "\n";
        accumulated_content += tail;
      }
      break;
    }

    std::string leading = text.substr(i, next_block - i);
    if (!leading.empty()) {
      if (!accumulated_content.empty() && accumulated_content.back() != '\n' && leading.front() != '\n')
        accumulated_content += "\n";
      accumulated_content += leading;
    }

    size_t p = next_block + opener_len;
    if (!is_bare_inv) {
      size_t gt = text.find('>', p);
      if (gt == std::string::npos) return fail("unterminated tool_calls open tag");
      p = gt + 1;
    }

    const std::string cl_v41 = "</" + dsml + " calls>";
    const std::string cl_v4  = "</" + dsml + "tool_calls>";
    const std::string cl_xml = "</tool_call>";

    bool in_block = true;
    while (in_block && p < text.size()) {
      p = skip_ws_local(text, p);
      if (p >= text.size()) break;

      if (text.compare(p, cl_v41.size(), cl_v41) == 0) { p += cl_v41.size(); break; }
      if (text.compare(p, cl_v4.size(), cl_v4) == 0) { p += cl_v4.size(); break; }
      if (text.compare(p, cl_xml.size(), cl_xml) == 0) { p += cl_xml.size(); break; }

      std::string inv_open_v41 = "<" + dsml + " invoke";
      std::string inv_open_v4  = "<" + dsml + "invoke";
      std::string inv_open_bare = "<invoke";
      std::string inv_open_func = "<function=";

      size_t inv_len = 0;
      bool is_func_syntax = false;
      if (text.compare(p, inv_open_v41.size(), inv_open_v41) == 0) inv_len = inv_open_v41.size();
      else if (text.compare(p, inv_open_v4.size(), inv_open_v4) == 0) inv_len = inv_open_v4.size();
      else if (text.compare(p, inv_open_bare.size(), inv_open_bare) == 0) inv_len = inv_open_bare.size();
      else if (text.compare(p, inv_open_func.size(), inv_open_func) == 0) {
        inv_len = inv_open_func.size();
        is_func_syntax = true;
      }

      if (inv_len == 0) {
        if (is_bare_inv) break;
        return fail("tool call format: expected invoke or block close");
      }

      p += inv_len;
      ToolCall tc;
      if (is_func_syntax) {
        size_t gt = text.find('>', p);
        if (gt == std::string::npos) return fail("tool name format");
        tc.name = text.substr(p, gt - p);
        p = gt + 1;
      } else {
        size_t gt = text.find('>', p);
        if (gt == std::string::npos) return fail("tool name format");
        size_t nq = text.find("name=\"", p);
        if (nq == std::string::npos || nq >= gt) return fail("tool name format");
        size_t ne = text.find('"', nq + 6);
        if (ne == std::string::npos || ne >= gt) return fail("tool name format");
        tc.name = text.substr(nq + 6, ne - (nq + 6));
        p = gt + 1;
      }

      const size_t sep = tc.name.find("::");
      if (sep != std::string::npos) {
        tc.ns = tc.name.substr(0, sep);
        tc.name = tc.name.substr(sep + 2);
      }

      std::string args = "{";
      std::set<std::string> seen;
      bool first_param = true;

      const std::string pc_v41 = "</" + dsml + " parameter>";
      const std::string pc_v4  = "</" + dsml + "parameter>";
      const std::string pc_bare = "</parameter>";

      const std::string po_v41 = "<" + dsml + " parameter";
      const std::string po_v4  = "<" + dsml + "parameter";
      const std::string po_bare = "<parameter";

      const std::string ic_v41 = "</" + dsml + " invoke>";
      const std::string ic_v4  = "</" + dsml + "invoke>";
      const std::string ic_bare = "</invoke>";
      const std::string ic_func = "</function>";

      while (p < text.size()) {
        p = skip_ws_local(text, p);
        if (p >= text.size()) break;

        if (text.compare(p, ic_v41.size(), ic_v41) == 0) { p += ic_v41.size(); break; }
        if (text.compare(p, ic_v4.size(), ic_v4) == 0) { p += ic_v4.size(); break; }
        if (text.compare(p, ic_bare.size(), ic_bare) == 0) { p += ic_bare.size(); break; }
        if (text.compare(p, ic_func.size(), ic_func) == 0) { p += ic_func.size(); break; }

        size_t p_len = 0;
        bool is_func_param = false;
        if (text.compare(p, po_v41.size(), po_v41) == 0) p_len = po_v41.size();
        else if (text.compare(p, po_v4.size(), po_v4) == 0) p_len = po_v4.size();
        else if (text.compare(p, po_bare.size(), po_bare) == 0) {
          if (p + 11 <= text.size() && text.compare(p, 11, "<parameter=") == 0) {
            p_len = 11;
            is_func_param = true;
          } else {
            p_len = po_bare.size();
          }
        }

        if (p_len == 0) {
          if (text.compare(p, cl_v41.size(), cl_v41) == 0 ||
              text.compare(p, cl_v4.size(), cl_v4) == 0 ||
              text.compare(p, cl_xml.size(), cl_xml) == 0) {
            break;
          }
          return fail("parameter format: expected parameter opener or invoke close");
        }

        p += p_len;
        std::string key, is_str = "true";
        if (is_func_param) {
          size_t gt = text.find('>', p);
          if (gt == std::string::npos) return fail("parameter format");
          key = text.substr(p, gt - p);
          p = gt + 1;
        } else {
          size_t gt = text.find('>', p);
          if (gt == std::string::npos) return fail("parameter format: missing '>'");
          size_t kq = text.find("name=\"", p);
          if (kq == std::string::npos || kq >= gt) return fail("parameter format: missing name=\"");
          size_t ke = text.find('"', kq + 6);
          if (ke == std::string::npos || ke >= gt) return fail("parameter format");
          key = text.substr(kq + 6, ke - (kq + 6));
          size_t sq = text.find("string=\"", p);
          if (sq != std::string::npos && sq >= kq && sq <= ke) {
            sq = text.find("string=\"", ke + 1);
          }
          if (sq != std::string::npos && sq < gt) {
            size_t se = text.find('"', sq + 8);
            if (se == std::string::npos || se >= gt) return fail("parameter format");
            is_str = text.substr(sq + 8, se - (sq + 8));
          }
          if (is_str != "true" && is_str != "false")
            return fail("parameter string= must be true|false");
          p = gt + 1;
        }

        if (seen.count(key)) return fail("duplicate parameter name");
        seen.insert(key);

        const size_t val_start = p;
        QuotedCall q_inv1(text, val_start, inv_open_v41, ic_v41);
        QuotedCall q_inv2(text, val_start, inv_open_v4, ic_v4);
        QuotedCall q_inv3(text, val_start, inv_open_bare, ic_bare);
        QuotedCall q_tc(text, val_start, "<tool_call>", "</tool_call>");
        QuotedCall q_fn(text, val_start, "<function=", "</function>");

        auto is_in_quoted_call = [&](size_t to) {
          return q_inv1.open_before(to) || q_inv2.open_before(to) ||
                 q_inv3.open_before(to) || q_tc.open_before(to) ||
                 q_fn.open_before(to);
        };

        size_t best_close = std::string::npos;
        size_t best_close_len = 0;

        auto test_candidate = [&](size_t at, size_t clen, bool respect_quotes) -> bool {
          if (respect_quotes && is_in_quoted_call(at)) return false;
          const size_t after = skip_ws_local(text, at + clen);
          const bool next_ok =
              after == text.size() ||
              text.compare(after, po_v41.size(), po_v41) == 0 ||
              text.compare(after, po_v4.size(), po_v4) == 0 ||
              text.compare(after, po_bare.size(), po_bare) == 0 ||
              text.compare(after, ic_v41.size(), ic_v41) == 0 ||
              text.compare(after, ic_v4.size(), ic_v4) == 0 ||
              text.compare(after, ic_bare.size(), ic_bare) == 0 ||
              text.compare(after, ic_func.size(), ic_func) == 0 ||
              text.compare(after, cl_v41.size(), cl_v41) == 0 ||
              text.compare(after, cl_v4.size(), cl_v4) == 0 ||
              text.compare(after, cl_xml.size(), cl_xml) == 0;
          if (!next_ok) return false;
          if (is_str == "false") {
            std::string cand_val = text.substr(val_start, at - val_start);
            if (!is_valid_json_non_string(cand_val)) return false;
          }
          return true;
        };

        for (const bool respect_quotes : {true, false}) {
          size_t scan = val_start;
          while (scan < text.size()) {
            size_t next_c = std::string::npos;
            size_t next_clen = 0;
            for (const auto& tag : {pc_v41, pc_v4, pc_bare}) {
              size_t cand = text.find(tag, scan);
              if (cand != std::string::npos && (next_c == std::string::npos || cand < next_c)) {
                next_c = cand;
                next_clen = tag.size();
              }
            }
            if (next_c == std::string::npos) break;
            if (test_candidate(next_c, next_clen, respect_quotes)) {
              best_close = next_c;
              best_close_len = next_clen;
              break;
            }
            scan = next_c + 1;
          }
          if (best_close != std::string::npos) break;
        }

        if (best_close == std::string::npos) {
          return fail("unclosed parameter");
        }

        std::string val = text.substr(val_start, best_close - val_start);
        p = best_close + best_close_len;

        if (!first_param) args += ", ";
        first_param = false;
        args += json_quote(key) + ": " + (is_str == "true" ? json_quote(val) : val);
      }

      args += "}";
      tc.arguments = args;
      out.tool_calls.push_back(std::move(tc));
    }

    i = p;
  }

  out.content = accumulated_content;

  const std::string eos(kEosStr);
  size_t ep = out.content.rfind(eos);
  if (ep != std::string::npos && ep + eos.size() == out.content.size()) {
    out.content.resize(ep);
  }

  // A special token loose in the text means the parse mis-framed something.
  for (const char* sp : {kBos, kEosStr, kThinkStart, kThinkEnd}) {
    if (out.content.find(sp) != std::string::npos) return fail("special token in content");
    if (out.tool_calls.empty() && out.reasoning_content.find(sp) != std::string::npos)
      return fail("special token in content");
  }

  return out;
}

size_t ToolBlockScanState::scan(const std::string& s, size_t at) {
  if (!initialized) {
    pos = at;
    initialized = true;
  }
  const std::string ds(kDsml);
  const std::string calls_o1 = "<" + ds + " calls";
  const std::string calls_o2 = "<" + ds + "tool_calls";
  const std::string calls_o3 = "<" + ds + "calls>";
  const std::string calls_o4 = "<" + ds + ">";

  const std::string inv_o1 = "<" + ds + " invoke";
  const std::string inv_o2 = "<" + ds + "invoke";
  const std::string inv_o3 = "<invoke";

  const std::string par_o1 = "<" + ds + " parameter";
  const std::string par_o2 = "<" + ds + "parameter";
  const std::string par_o3 = "<parameter";

  const std::string par_c1 = "</" + ds + " parameter>";
  const std::string par_c2 = "</" + ds + "parameter>";
  const std::string par_c3 = "</parameter>";

  const std::string inv_c1 = "</" + ds + " invoke>";
  const std::string inv_c2 = "</" + ds + "invoke>";
  const std::string inv_c3 = "</invoke>";

  const std::string calls_c1 = "</" + ds + " calls>";
  const std::string calls_c2 = "</" + ds + "tool_calls>";
  const std::string calls_c3 = "</" + ds + "calls>";
  const std::string calls_c4 = "</" + ds + ">";

  const std::string* all_tags[] = {
      &calls_o1, &calls_o2, &calls_o3, &calls_o4,
      &inv_o1,   &inv_o2,   &inv_o3,
      &par_o1,   &par_o2,   &par_o3,
      &par_c1,   &par_c2,   &par_c3,
      &inv_c1,   &inv_c2,   &inv_c3,
      &calls_c1, &calls_c2, &calls_c3, &calls_c4,
  };

  auto is_ws_or_gt = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>';
  };

  while (pos < s.size()) {
    const size_t lt = s.find('<', pos);
    if (lt == std::string::npos) {
      pos = s.size();
      return std::string::npos;
    }
    pos = lt;
    const size_t rem = s.size() - lt;

    // Hold at `lt` if the tail of `s` is a strict prefix of any tracked tag so a tag split
    // across tokens is never skipped.
    for (const std::string* tag : all_tags) {
      if (rem < tag->size() && tag->compare(0, rem, s, lt, rem) == 0) {
        return std::string::npos;
      }
    }

    auto match_open_delim = [&](const std::string& tag, bool allow_eq, bool* need_more) -> bool {
      if (rem < tag.size() || s.compare(lt, tag.size(), tag) != 0) return false;
      if (rem == tag.size()) { *need_more = true; return false; }
      const char nxt = s[lt + tag.size()];
      return is_ws_or_gt(nxt) || (allow_eq && nxt == '=');
    };
    auto match_exact = [&](const std::string& tag) -> bool {
      return rem >= tag.size() && s.compare(lt, tag.size(), tag) == 0;
    };

    bool need_more = false;
    // 1. Calls openers
    if (match_open_delim(calls_o1, false, &need_more)) {
      ++calls_depth; saw_calls = true; pos = lt + calls_o1.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_open_delim(calls_o2, false, &need_more)) {
      ++calls_depth; saw_calls = true; pos = lt + calls_o2.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_exact(calls_o3)) {
      ++calls_depth; saw_calls = true; pos = lt + calls_o3.size(); continue;
    }
    if (match_exact(calls_o4) && param_depth == 0 && inv_depth == 0) {
      ++calls_depth; saw_calls = true; pos = lt + calls_o4.size(); continue;
    }

    // 2. Invoke openers
    if (match_open_delim(inv_o1, false, &need_more)) {
      ++inv_depth; saw_inv = true; pos = lt + inv_o1.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_open_delim(inv_o2, false, &need_more)) {
      ++inv_depth; saw_inv = true; pos = lt + inv_o2.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_open_delim(inv_o3, false, &need_more)) {
      ++inv_depth; saw_inv = true; pos = lt + inv_o3.size(); continue;
    }
    if (need_more) return std::string::npos;

    // 3. Parameter openers
    if (match_open_delim(par_o1, false, &need_more)) {
      ++param_depth; pos = lt + par_o1.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_open_delim(par_o2, false, &need_more)) {
      ++param_depth; pos = lt + par_o2.size(); continue;
    }
    if (need_more) return std::string::npos;
    if (match_open_delim(par_o3, true, &need_more)) {
      ++param_depth; pos = lt + par_o3.size(); continue;
    }
    if (need_more) return std::string::npos;

    // 4. Parameter closers
    size_t par_clen = 0;
    if (match_exact(par_c1)) par_clen = par_c1.size();
    else if (match_exact(par_c2)) par_clen = par_c2.size();
    else if (match_exact(par_c3)) par_clen = par_c3.size();
    if (par_clen) {
      if (param_depth > 0) {
        const size_t after = skip_ws_local(s, lt + par_clen);
        if (after == s.size()) return std::string::npos;
        const size_t arem = s.size() - after;
        const std::string ds_open = "<" + ds;
        const std::string ds_close = "</" + ds;
        if (arem < ds_open.size() && ds_open.compare(0, arem, s, after, arem) == 0)
          return std::string::npos;
        if (arem < ds_close.size() && ds_close.compare(0, arem, s, after, arem) == 0)
          return std::string::npos;
        if (s.compare(after, ds_open.size(), ds_open) == 0 ||
            s.compare(after, ds_close.size(), ds_close) == 0) {
          --param_depth;
        } else {
          const char* next_tags[] = {"<parameter", "</parameter>", "</invoke>",
                                     "</tool_call>", "</function>"};
          for (const char* nt : next_tags) {
            const size_t nlen = std::strlen(nt);
            if (arem < nlen && std::strncmp(s.data() + after, nt, arem) == 0)
              return std::string::npos;
            if (arem >= nlen && s.compare(after, nlen, nt) == 0) {
              --param_depth;
              break;
            }
          }
        }
      }
      pos = lt + par_clen;
      continue;
    }

    // 5. Invoke closers
    size_t inv_clen = 0;
    bool bare_inv_close = false;
    if (match_exact(inv_c1)) inv_clen = inv_c1.size();
    else if (match_exact(inv_c2)) inv_clen = inv_c2.size();
    else if (match_exact(inv_c3)) { inv_clen = inv_c3.size(); bare_inv_close = true; }
    if (inv_clen) {
      if (param_depth > 0) {
        if (inv_depth > 1) --inv_depth;
        pos = lt + inv_clen;
        continue;
      }
      if (inv_depth > 0) --inv_depth;
      pos = lt + inv_clen;
      if (bare_inv_close && !saw_calls && calls_depth == 0 && inv_depth == 0 && param_depth == 0) {
        return pos;
      }
      continue;
    }

    // 6. Calls closers
    size_t calls_clen = 0;
    if (match_exact(calls_c1)) calls_clen = calls_c1.size();
    else if (match_exact(calls_c2)) calls_clen = calls_c2.size();
    else if (match_exact(calls_c3)) calls_clen = calls_c3.size();
    else if (match_exact(calls_c4)) calls_clen = calls_c4.size();
    if (calls_clen) {
      if (param_depth > 0 || inv_depth > 0) {
        if (calls_depth > 1) --calls_depth;
        pos = lt + calls_clen;
        continue;
      }
      if (calls_depth > 1) {
        --calls_depth;
        pos = lt + calls_clen;
        continue;
      }
      calls_depth = 0;
      pos = lt + calls_clen;
      return pos;
    }

    pos = lt + 1;
  }
  return std::string::npos;
}

size_t tool_block_close_end(const std::string& s, size_t at) {
  ToolBlockScanState st;
  return st.scan(s, at);
}

}  // namespace aff
