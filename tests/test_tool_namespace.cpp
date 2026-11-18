#include "engine/chat_encode.h"
#include "server/server.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

int main() {
  using namespace aff;
  EncodeOpts opts;
  opts.v41 = true;
  opts.thinking_mode = ThinkingMode::Thinking;
  opts.effort_budget = 75;

  ChatMsg sys;
  sys.role = "system";
  sys.content = "You are a helpful assistant.";
  sys.tools.push_back(ToolDef{
      R"({"name":"search","description":"Search files","parameters":{"type":"object"},"namespace":{"name":"mcp","description":"MCP server"}})"});
  ChatMsg usr;
  usr.role = "user";
  usr.content = "Find main.cpp";

  std::string prompt = encode_messages({sys, usr}, opts);
  assert(prompt.find(R"("name": "mcp::search")") != std::string::npos);
  assert(prompt.find(R"("description": "MCP server\nSearch files")") != std::string::npos);
  assert(prompt.find(R"("namespace")") == std::string::npos);

  // 1. Single invoke with string="true"
  std::string raw =
      "Thinking...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"mcp::search\">\n"
      "<｜DSML｜ parameter name=\"query\" string=\"true\">main.cpp</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm = parse_completion(raw, ThinkingMode::Thinking);
  assert(pm.ok);
  assert(pm.tool_calls.size() == 1);
  assert(pm.tool_calls[0].ns == "mcp");
  assert(pm.tool_calls[0].name == "search");
  assert(pm.tool_calls[0].arguments == R"({"query": "main.cpp"})");

  // Re-encode assistant turn with split ns + name and verify it renders mcp::search
  ChatMsg asst;
  asst.role = "assistant";
  asst.reasoning_content = pm.reasoning_content;
  asst.tool_calls = pm.tool_calls;
  ChatMsg tool_res;
  tool_res.role = "tool";
  tool_res.content = "src/main.cpp";
  std::string prompt2 = encode_messages({sys, usr, asst, tool_res}, opts);
  assert(prompt2.find("<｜DSML｜ invoke name=\"mcp::search\">") != std::string::npos);
  assert(prompt2.find("string=\"true\">main.cpp</｜DSML｜ parameter>") != std::string::npos);

  // 2. Radiance 9cf8d3d parity: typed argument tags with string="false" for int, bool, arrays, objects
  std::string v41_typed =
      "Thinking of queries...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"lookup\">\n"
      "<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"enabled\" string=\"false\">true</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"scores\" string=\"false\">[1.5, 2.0]</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm_typed = parse_completion(v41_typed, ThinkingMode::Thinking);
  assert(pm_typed.ok);
  assert(pm_typed.tool_calls.size() == 1);
  assert(pm_typed.tool_calls[0].name == "lookup");
  assert(pm_typed.tool_calls[0].arguments ==
         R"({"query": "value", "limit": 2, "enabled": true, "scores": [1.5, 2.0]})");

  // 3. Multi-call invoke in one block (golden case: get_weather and lookup)
  std::string v41_multi =
      "Planning tools...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"location\" string=\"true\">Beijing</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"unit\" string=\"true\">celsius</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "<｜DSML｜ invoke name=\"lookup\">\n"
      "<｜DSML｜ parameter name=\"query\" string=\"true\">北京 天气</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm_multi = parse_completion(v41_multi, ThinkingMode::Thinking);
  assert(pm_multi.ok);
  assert(pm_multi.tool_calls.size() == 2);
  assert(pm_multi.tool_calls[0].name == "get_weather");
  assert(pm_multi.tool_calls[0].arguments == R"({"location": "Beijing", "unit": "celsius"})");
  assert(pm_multi.tool_calls[1].name == "lookup");
  assert(pm_multi.tool_calls[1].arguments == R"({"query": "北京 天气"})");

  // Re-encode multi-call and verify typed argument render (string="true" vs string="false")
  ChatMsg asst_typed;
  asst_typed.role = "assistant";
  asst_typed.tool_calls = pm_typed.tool_calls;
  std::string re_encoded = encode_messages({asst_typed}, opts);
  assert(re_encoded.find("name=\"query\" string=\"true\">value</") != std::string::npos);
  assert(re_encoded.find("name=\"limit\" string=\"false\">2</") != std::string::npos);
  assert(re_encoded.find("name=\"enabled\" string=\"false\">true</") != std::string::npos);

  // 4. Invalid parameter string attribute (not true/false) must fail
  std::string v41_bad =
      "Thinking...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"test\">\n"
      "<｜DSML｜ parameter name=\"k\" string=\"invalid\">val</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm_bad = parse_completion(v41_bad, ThinkingMode::Thinking);
  assert(!pm_bad.ok);

  // 5. PR #1020 (a): Argument containing literal DSML parameter/invoke/calls tags
  std::string literal_dsml =
      "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"inner\">\n"
      "<｜DSML｜ parameter name=\"x\" string=\"true\">hello world</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  std::string v41_embedded =
      "Thinking of code...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"bash\">\n"
      "<｜DSML｜ parameter name=\"command\" string=\"true\">cat <<'EOF'\n" +
      literal_dsml +
      "\nEOF\n</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm_embedded = parse_completion(v41_embedded, ThinkingMode::Thinking);
  assert(pm_embedded.ok);
  assert(pm_embedded.tool_calls.size() == 1);
  assert(pm_embedded.tool_calls[0].name == "bash");
  assert(pm_embedded.tool_calls[0].arguments.find("hello world") != std::string::npos);
  assert(pm_embedded.tool_calls[0].arguments.find("inner") != std::string::npos);

  // 6. PR #1020 (b): Two tool calls separated by a sentence where call #1 holds embedded tags
  std::string v41_split =
      "I will run the command and inspect the file.</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"bash\">\n"
      "<｜DSML｜ parameter name=\"command\" string=\"true\">grep -n '</｜DSML｜ parameter>' test.cpp</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls>\n"
      "Now check the README:\n"
      "<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"read_file\">\n"
      "<｜DSML｜ parameter name=\"path\" string=\"true\">README.md</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls><｜end▁of▁sentence｜>";
  ParsedMessage pm_split = parse_completion(v41_split, ThinkingMode::Thinking);
  assert(pm_split.ok);
  assert(pm_split.tool_calls.size() == 2);
  assert(pm_split.tool_calls[0].name == "bash");
  assert(pm_split.tool_calls[0].arguments == R"({"command": "grep -n '</｜DSML｜ parameter>' test.cpp"})");
  assert(pm_split.tool_calls[1].name == "read_file");
  assert(pm_split.tool_calls[1].arguments == R"({"path": "README.md"})");
  assert(pm_split.content.find("Now check the README:") != std::string::npos);

  // 7. PR #1020 (c): Unclosed / malformed fallback safety
  // (i) unclosed parameter tag
  std::string v41_unclosed_param =
      "Thinking...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"bash\">\n"
      "<｜DSML｜ parameter name=\"command\" string=\"true\">ls -la\n"
      "</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls>";
  ParsedMessage pm_unclosed = parse_completion(v41_unclosed_param, ThinkingMode::Thinking);
  assert(!pm_unclosed.ok);
  assert(pm_unclosed.tool_calls.empty());

  // (ii) unclosed invoke tag
  std::string v41_unclosed_inv =
      "Thinking...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"bash\">\n"
      "<｜DSML｜ parameter name=\"command\" string=\"true\">ls</｜DSML｜ parameter>\n";
  ParsedMessage pm_unclosed_inv = parse_completion(v41_unclosed_inv, ThinkingMode::Thinking);
  assert(!pm_unclosed_inv.ok);
  assert(pm_unclosed_inv.tool_calls.empty());

  // (iii) malformed XML tag
  std::string v41_malformed =
      "Thinking...</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke invalid>\n"
      "</｜DSML｜ calls>";
  ParsedMessage pm_malformed = parse_completion(v41_malformed, ThinkingMode::Thinking);
  assert(!pm_malformed.ok);

  // 8. Strict UTF-8 streaming holdback, EOS sanitization, and validating json_escape
  {
    const std::string rep = "\xEF\xBF\xBD";

    // (a) Multi-byte token split: fullwidth bar '｜' (0xEF 0xBD 0x9C)
    const std::string bar = "\xEF\xBD\x9C";
    assert(utf8_safe_end(bar.substr(0, 1), 1) == 0);
    assert(utf8_safe_end(bar.substr(0, 2), 2) == 0);
    assert(utf8_safe_end(bar, 3) == 3);

    // (b) 4-byte emoji split: '🚀' (0xF0 0x9F 0x9A 0x80)
    const std::string rocket = "\xF0\x9F\x9A\x80";
    assert(utf8_safe_end("hi" + rocket.substr(0, 1), 3) == 2);
    assert(utf8_safe_end("hi" + rocket.substr(0, 2), 4) == 2);
    assert(utf8_safe_end("hi" + rocket.substr(0, 3), 5) == 2);
    assert(utf8_safe_end("hi" + rocket, 6) == 6);

    // (c) Early EOS inside <think> at position 173-174 with incomplete 2-byte prefix (0xE2 0x80 / 0xEF 0xBD)
    std::string think_prefix(173, 'a');
    std::string eos_e2 = think_prefix + "\xE2\x80";
    assert(utf8_safe_end(eos_e2, eos_e2.size()) == 173);
    std::string eos_flushed = sanitize_utf8(eos_e2.substr(173));
    assert(eos_flushed == rep);
    assert(json_escape(eos_flushed) == rep);

    std::string eos_ef = think_prefix + "\xEF\xBD";
    assert(utf8_safe_end(eos_ef, eos_ef.size()) == 173);
    assert(sanitize_utf8(eos_ef.substr(173)) == rep);
    assert(json_escape(eos_ef) == think_prefix + rep);

    // (d) Lone continuations & invalid lead bytes are NOT held back infinitely and sanitize to U+FFFD
    assert(utf8_safe_end("\x80", 1) == 1);
    assert(utf8_safe_end("\xFF", 1) == 1);
    assert(utf8_safe_end("\xED\xA0", 2) == 2); // surrogate prefix cannot become valid UTF-8
    assert(json_escape("\x80") == rep);
    assert(json_escape("\xFF") == rep);
    assert(json_escape("\xED\xA0\x80") == rep + rep + rep);
    assert(json_escape("\xE0\x80\xAF") == rep + rep + rep); // overlong 3-byte
    assert(json_escape("\xF0\x80\x80\xAF") == rep + rep + rep + rep); // overlong 4-byte

    // (e) Truncated UTF-8 followed by quote or ASCII + standard JSON escapes
    assert(json_escape("\xE2\x80\"abc") == rep + "\\\"abc");
    assert(json_escape("\xEF\xBDxyz") == rep + "xyz");
    assert(json_escape("\xF0\x9F\x9A!") == rep + "!");
    assert(json_escape("a\n\t\"\\b") == "a\\n\\t\\\"\\\\b");
    assert(json_escape("北京" + bar + rocket) == "北京" + bar + rocket);
  }

  // 9. tool_block_close_end & ToolBlockScanState incremental streaming with embedded DSML closers
  {
    const std::string ds = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    // (a) Open unclosed parameter containing literal </｜DSML｜ calls>, </｜DSML｜ invoke>, </｜DSML｜>
    const std::string prefix =
        "<" + ds + " calls>\n"
        "<" + ds + " invoke name=\"bash\">\n"
        "<" + ds + " parameter name=\"cmd\" string=\"true\">"
        "grep '</" + ds + " calls>' a.cpp && echo '</" + ds + " invoke>' && echo '</" + ds + ">'\n"
        "grep '</" + ds + " parameter>' b.cpp";
    CHECK(tool_block_close_end(prefix, 0) == std::string::npos);

    const std::string full =
        prefix + "\n</" + ds + " parameter>\n</" + ds + " invoke>\n</" + ds + " calls>";
    const std::string trailing = "\nsome trailing ramble";
    const std::string full_with_ramble = full + trailing;

    // Stream byte-by-byte (and 3-byte chunks) using ToolBlockScanState and stateless tool_block_close_end
    ToolBlockScanState scan_st;
    for (size_t len = 1; len < full.size(); ++len) {
      const std::string sub = full_with_ramble.substr(0, len);
      CHECK(tool_block_close_end(sub, 0) == std::string::npos);
      CHECK(scan_st.scan(sub, 0) == std::string::npos);
    }
    CHECK(tool_block_close_end(full, 0) == full.size());
    CHECK(scan_st.scan(full, 0) == full.size());
    CHECK(tool_block_close_end(full_with_ramble, 0) == full.size());

    // (b) Embedded complete DSML block inside parameter value
    const std::string nested_full =
        "<" + ds + " calls>\n"
        "<" + ds + " invoke name=\"bash\">\n"
        "<" + ds + " parameter name=\"cmd\" string=\"true\">cat <<'EOF'\n" +
        literal_dsml +
        "\nEOF\n</" + ds + " parameter>\n"
        "</" + ds + " invoke>\n"
        "</" + ds + " calls>";
    ToolBlockScanState nested_scan;
    for (size_t len = 1; len < nested_full.size(); ++len) {
      CHECK(nested_scan.scan(nested_full.substr(0, len), 0) == std::string::npos);
    }
    CHECK(nested_scan.scan(nested_full, 0) == nested_full.size());

    // (c) Bare <invoke> with embedded </invoke> inside open <parameter>
    const std::string bare_full =
        "<invoke name=\"write\">\n"
        "<parameter name=\"body\">echo '</invoke>'</parameter>\n"
        "</invoke>";
    ToolBlockScanState bare_scan;
    for (size_t len = 1; len < bare_full.size(); ++len) {
      CHECK(bare_scan.scan(bare_full.substr(0, len), 0) == std::string::npos);
    }
    CHECK(bare_scan.scan(bare_full, 0) == bare_full.size());
  }

  // 10. Attribute '< gt' bounds checks, reordered string="false" before name=,
  //     invalid JSON at EOF, and skipping repair_bare_dsml when </think> is missing.
  {
    const std::string ds = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
    // (a) Malformed invoke tag without name="..." followed by a second invoke tag with name="..."
    const std::string bad_inv_gt =
        "ok</think>\n\n<" + ds + " calls>\n"
        "<" + ds + " invoke invalid>\n"
        "<" + ds + " invoke name=\"foo\">\n"
        "</" + ds + " invoke>\n"
        "</" + ds + " calls>";
    CHECK(!parse_completion(bad_inv_gt, ThinkingMode::Thinking).ok);

    // (b) Malformed parameter tag without name="..." followed by a second parameter tag with name="..."
    const std::string bad_param_gt =
        "ok</think>\n\n<" + ds + " calls>\n"
        "<" + ds + " invoke name=\"foo\">\n"
        "<" + ds + " parameter invalid>\n"
        "<" + ds + " parameter name=\"x\" string=\"true\">1</" + ds + " parameter>\n"
        "</" + ds + " invoke>\n"
        "</" + ds + " calls>";
    CHECK(!parse_completion(bad_param_gt, ThinkingMode::Thinking).ok);

    // (c) string="false" appearing BEFORE name="limit"
    const std::string reordered =
        "ok</think>\n\n<" + ds + " calls>\n"
        "<" + ds + " invoke name=\"lookup\">\n"
        "<" + ds + " parameter string=\"false\" name=\"limit\">42</" + ds + " parameter>\n"
        "</" + ds + " invoke>\n"
        "</" + ds + " calls>";
    ParsedMessage pm_reord = parse_completion(reordered, ThinkingMode::Thinking);
    CHECK(pm_reord.ok);
    CHECK(pm_reord.tool_calls.size() == 1);
    CHECK(pm_reord.tool_calls[0].arguments == R"({"limit": 42})");

    // (d) string="false" with invalid JSON right before EOF (after == size) must fail
    const std::string bad_eof_json =
        "ok</think>\n\n<" + ds + " calls>\n"
        "<" + ds + " invoke name=\"lookup\">\n"
        "<" + ds + " parameter name=\"limit\" string=\"false\">not_json</" + ds + " parameter>";
    CHECK(!parse_completion(bad_eof_json, ThinkingMode::Thinking).ok);

    // (e) Thinking mode with NO </think> and bare <invoke name="x"> must skip repair_bare_dsml
    const std::string no_think_close =
        "reasoning mentions <invoke name=\"example\">foo</invoke> literally";
    ParsedMessage pm_no_te = parse_completion(no_think_close, ThinkingMode::Thinking);
    CHECK(!pm_no_te.ok);
    CHECK(pm_no_te.content == no_think_close);
  }

  std::printf("PASS: test_tool_namespace\n");
  return 0;
}
