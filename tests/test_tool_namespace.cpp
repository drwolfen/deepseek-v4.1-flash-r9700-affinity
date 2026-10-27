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

  // Simulate model completion with namespaced DSML invoke
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

  std::printf("PASS: test_tool_namespace\n");
  return 0;
}
