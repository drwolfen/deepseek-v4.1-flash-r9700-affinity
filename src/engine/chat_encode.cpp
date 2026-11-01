#include "engine/chat_encode.h"

#include <algorithm>
#include <cstddef>
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
        out += thinking ? kThinkStart : kThinkEnd;
        out += tt;
      }
    } else if (m.role == "user" || m.role == "developer") {
      out += kAssistant;
      if (thinking && (!drop || (int)idx >= last_user)) out += kThinkStart;
      else                                              out += kThinkEnd;
    }
  }
  return out;
}

// ---- parsing ------------------------------------------------------------------------------------

namespace {

// The reference's _read_until_stop: the EARLIEST of the stops wins, not the first listed.
size_t read_until(size_t i, const std::string& t, const std::vector<std::string>& stops,
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
    if (te != std::string::npos) body_start = te + std::string(kThinkEnd).size();
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

ParsedMessage parse_completion(const std::string& text_in, ThinkingMode mode) {
  ParsedMessage out;
  const std::string text = repair_bare_dsml(text_in, mode);
  const std::string dsml = kDsml;
  const std::string tc_open_v4  = std::string("\n\n<") + dsml + "tool_calls";
  const std::string tc_open_v41 = std::string("\n\n<") + dsml + " calls";
  const std::string tc_open_v41_raw = std::string("<") + dsml + " calls";
  const std::string tc_open_v4_raw  = std::string("<") + dsml + "tool_calls";

  auto fail = [&](const char* why) {
    out.ok = false;
    out.error = why;
    out.content = text;              // the raw text, which is all a caller can honestly be given
    out.reasoning_content.clear();
    out.tool_calls.clear();
    return out;
  };

  size_t i = 0;
  std::string chunk, stop;

  const std::vector<std::string> open_markers = {tc_open_v4, tc_open_v41, tc_open_v4_raw, tc_open_v41_raw};

  if (mode == ThinkingMode::Thinking) {
    std::vector<std::string> think_stops = {kThinkEnd};
    think_stops.insert(think_stops.end(), open_markers.begin(), open_markers.end());
    i = read_until(i, text, think_stops, &chunk, &stop);
    if (stop != kThinkEnd) return fail("missing </think>");
    out.reasoning_content = chunk;
  }

  std::vector<std::string> content_stops = {kEosStr};
  content_stops.insert(content_stops.end(), open_markers.begin(), open_markers.end());
  i = read_until(i, text, content_stops, &chunk, &stop);
  out.content = chunk;

  const bool is_tool_open = std::find(open_markers.begin(), open_markers.end(), stop) != open_markers.end();

  if (is_tool_open) {
    const std::string tc_close_v4   = std::string("</") + dsml + "tool_calls>";
    const std::string tc_close_v41  = std::string("</") + dsml + " calls>";
    const std::string inv_open_v4   = std::string("<") + dsml + "invoke";
    const std::string inv_open_v41  = std::string("<") + dsml + " invoke";
    const std::string inv_close_v4  = std::string("</") + dsml + "invoke";
    const std::string inv_close_v41 = std::string("</") + dsml + " invoke";
    const std::string par_open_v4   = std::string("<") + dsml + "parameter";
    const std::string par_open_v41  = std::string("<") + dsml + " parameter";
    const std::string par_close_v4  = std::string("/") + dsml + "parameter";
    const std::string par_close_v41 = std::string("/") + dsml + " parameter";

    // Inside the block: alternating `invoke` headers and `parameter` bodies until the close.
    // Both V4 (no leading space) and V4.1 (leading space) tag spellings are accepted at each step
    // so a single space-drop inside a V4.1 block does not abort the parse.
    while (i < text.size()) {
      i = read_until(i, text, {inv_open_v4, inv_open_v41, tc_close_v4, tc_close_v41}, &chunk, &stop);
      if (chunk != ">\n") return fail("tool call format: expected '>\\n'");
      if (stop == tc_close_v4 || stop == tc_close_v41) break;
      if (stop.empty()) return fail("unterminated tool_calls block");

      i = read_until(i, text, {par_open_v4, par_open_v41, inv_close_v4, inv_close_v41}, &chunk, &stop);
      // ` name="X">\n`
      const size_t nq = chunk.find("name=\"");
      const size_t ne = nq == std::string::npos ? std::string::npos : chunk.find("\">\n", nq + 6);
      if (nq == std::string::npos || ne == std::string::npos) return fail("tool name format");
      ToolCall tc;
      tc.name = chunk.substr(nq + 6, ne - nq - 6);
      const size_t sep = tc.name.find("::");
      if (sep != std::string::npos) {
        tc.ns = tc.name.substr(0, sep);
        tc.name = tc.name.substr(sep + 2);
      }

      std::string args = "{";
      std::set<std::string> seen;
      bool first = true;
      while (stop == par_open_v4 || stop == par_open_v41) {
        i = read_until(i, text, {par_close_v4, par_close_v41}, &chunk, &stop);
        // ` name="k" string="true|false">value<`
        const size_t kq = chunk.find("name=\"");
        const size_t k_end = kq == std::string::npos ? std::string::npos : chunk.find("\" string=\"", kq + 6);
        if (kq == std::string::npos || k_end == std::string::npos) return fail("parameter format");
        const std::string key = chunk.substr(kq + 6, k_end - kq - 6);
        const size_t sv = k_end + 10;
        const size_t se = chunk.find("\">", sv);
        if (se == std::string::npos) return fail("parameter format");
        const std::string is_str = chunk.substr(sv, se - sv);
        if (is_str != "true" && is_str != "false") return fail("parameter string= must be true|false");
        std::string val = chunk.substr(se + 2);
        if (!val.empty() && val.back() == '<') val.pop_back();

        // The reference rejects a repeated parameter; a silently duplicated JSON key would otherwise
        // reach the caller and be resolved differently by whichever parser reads it.
        if (seen.count(key)) return fail("duplicate parameter name");
        seen.insert(key);
        if (!first) args += ", ";
        first = false;
        args += json_quote(key) + ": " + (is_str == "true" ? json_quote(val) : val);

        i = read_until(i, text, {par_open_v4, par_open_v41, inv_close_v4, inv_close_v41}, &chunk, &stop);
        if (chunk != ">\n") return fail("parameter format: expected '>\\n'");
      }
      args += "}";
      tc.arguments = args;
      out.tool_calls.push_back(tc);
    }
    i = read_until(i, text, {kEosStr}, &chunk, &stop);
    if (!chunk.empty()) return fail("content after tool calls");
  } else if (stop != kEosStr && !stop.empty()) {
    return fail("unexpected stop token");
  }

  // A special token loose in the text means the parse mis-framed something. When tool_calls have
  // already been framed after </think>, allow the reasoning block to have quoted `<think>` or
  // `｜DSML｜` in prose without discarding the valid tool call.
  for (const char* sp : {kBos, kEosStr, kThinkStart, kThinkEnd, kDsml}) {
    if (out.content.find(sp) != std::string::npos) return fail("special token in content");
    if (out.tool_calls.empty() && out.reasoning_content.find(sp) != std::string::npos)
      return fail("special token in content");
  }

  return out;
}

}  // namespace aff
