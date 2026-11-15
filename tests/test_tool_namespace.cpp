#include "engine/chat_encode.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

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

  std::printf("PASS: test_tool_namespace\n");
  return 0;
}
