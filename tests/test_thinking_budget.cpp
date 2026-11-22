#include "engine/chat_encode.h"
#include "engine/thinking_budget.h"
#include "engine/tokenizer.h"
#include "server/server.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace {

using namespace aff;

BudgetHook make_hook(int reserve = 4096, int loop_repeats = 3, bool freeze_enabled = true) {
  BudgetHook h;
  h.close_token_ids = {201, 128822, 271};
  h.marker_token_ids = {128822};
  h.hard_limit_remaining = reserve;
  h.loop_span = 64;
  h.loop_repeats = loop_repeats;
  h.freeze_enabled = freeze_enabled;
  h.freeze_min_tokens = 512;
  h.freeze_window = 256;
  h.freeze_nl_thresh = 64;
  h.freeze_distinct_max = 36;
  h.freeze_nl_token_ids = {201, 271};
  return h;
}

// Scenario 1: Natural 128822 settles without injection.
void test_natural_close() {
  BudgetHook h = make_hook(reply_reserve_within(4096, 16384));
  ThinkingBudget b(&h, 16384);
  for (int i = 0; i < 100; ++i) {
    auto d = b.apply(1000 + i);
    REQUIRE(!d.cut);
    REQUIRE(d.token == 1000 + i);
    REQUIRE(!b.settled());
  }
  auto d_end = b.apply(128822);
  REQUIRE(!d_end.cut);
  REQUIRE(d_end.token == 128822);
  REQUIRE(b.naturally_closed());
  REQUIRE(!b.forced_close());
  REQUIRE(!b.force_closed());
  REQUIRE(!b.injecting());
  REQUIRE(b.settled());
  REQUIRE(std::strcmp(b.close_reason(), "") == 0);
  // Post-settle tokens pass through untouched.
  auto d_ans = b.apply(5555);
  REQUIRE(!d_ans.cut);
  REQUIRE(d_ans.token == 5555);
  std::printf("PASS: test_natural_close\n");
}

// Scenario 2: Hard limit total_budget=16384, reserve=4096 cuts at 12288 + injects {201, 128822, 271};
// total_budget=64 scales via reply_reserve_within(4096, 64) == 32.
void test_hard_limit() {
  REQUIRE(reply_reserve_within(4096, 16384) == 4096);
  REQUIRE(reply_reserve_within(4096, 64) == 32);

  BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/0, /*freeze_enabled=*/false);
  ThinkingBudget b(&h, 16384);
  for (int i = 0; i < 12288; ++i) {
    auto d = b.apply(1000 + i);
    REQUIRE(!d.cut);
    REQUIRE(d.token == 1000 + i);
    REQUIRE(!b.forced_close());
  }
  // Token 12289 hits remaining == 4096 <= hard_limit_remaining: cut and inject 201.
  auto d0 = b.apply(99999);
  REQUIRE(d0.cut);
  REQUIRE(d0.token == 201);
  REQUIRE(b.forced_close());
  REQUIRE(b.force_closed());
  REQUIRE(b.injecting());
  REQUIRE(b.thought_tokens() == 12288);
  REQUIRE(std::strcmp(b.close_reason(), "hard_limit") == 0);
  REQUIRE(b.pending_close_tokens() == 2);
  REQUIRE(b.pending_close_token(0) == 128822);
  REQUIRE(b.pending_close_token(1) == 271);

  // Forced close tokens (matching drafts do not cut, mismatching do).
  auto d1 = b.apply(128822);
  REQUIRE(!d1.cut);
  REQUIRE(d1.token == 128822);
  auto d2 = b.apply(777);
  REQUIRE(d2.cut);
  REQUIRE(d2.token == 271);
  REQUIRE(!b.injecting());
  REQUIRE(b.settled());

  // Scaled budget=64 -> reserve=32 -> cuts after 32 thinking tokens.
  BudgetHook h64 = make_hook(reply_reserve_within(4096, 64), 0, false);
  ThinkingBudget b64(&h64, 64);
  for (int i = 0; i < 32; ++i) {
    auto d = b64.apply(2000 + i);
    REQUIRE(!d.cut);
  }
  auto dc = b64.apply(2999);
  REQUIRE(dc.cut);
  REQUIRE(dc.token == 201);
  REQUIRE(b64.thought_tokens() == 32);
  REQUIRE(std::strcmp(b64.close_reason(), "hard_limit") == 0);
  std::printf("PASS: test_hard_limit\n");
}

// Scenario 3: 64-tok repeat loop (repeats=3) force-closes.
void test_loop_repeat() {
  BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
  ThinkingBudget b(&h, 16384);
  for (int rep = 0; rep < 2; ++rep) {
    for (int i = 0; i < 64; ++i) {
      auto d = b.apply(5000 + i);
      REQUIRE(!d.cut);
      REQUIRE(!b.force_closed());
    }
  }
  bool fired = false;
  for (int i = 0; i < 64; ++i) {
    auto d = b.apply(5000 + i);
    if (i < 63) {
      REQUIRE(!d.cut);
    } else {
      REQUIRE(d.cut);
      REQUIRE(d.token == 201);
      fired = true;
    }
  }
  REQUIRE(fired);
  REQUIRE(b.force_closed());
  REQUIRE(std::strcmp(b.close_reason(), "loop_repeat") == 0);
  auto d1 = b.apply(128822);
  auto d2 = b.apply(271);
  REQUIRE(d1.token == 128822 && d2.token == 271);
  REQUIRE(b.settled());
  std::printf("PASS: test_loop_repeat\n");
}

// Scenario 4: Executive-freeze churn triggers "freeze_churn", while varied code with newlines (>60 distinct IDs) does NOT.
void test_executive_freeze_breaker() {
  // Positive case: 512 normal varied tokens + 256 low-diversity (Go/Now/Execute/Enough + 271 newline churn).
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/0, /*freeze_enabled=*/true);
    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 512; ++i) {
      auto d = b.apply(10000 + i);
      REQUIRE(!d.cut);
      REQUIRE(!b.force_closed());
    }
    const int32_t churn_words[4] = {41001, 41002, 41003, 41004};  // Go, Now, Execute, Enough
    bool fired = false;
    for (int i = 0; i < 256; ++i) {
      const int32_t tok = (i % 3 == 2) ? 271 : churn_words[i % 4];
      auto d = b.apply(tok);
      if (d.cut) {
        REQUIRE(d.token == 201);
        fired = true;
        break;
      }
    }
    REQUIRE(fired);
    REQUIRE(b.force_closed());
    REQUIRE(b.freeze_triggered());
    REQUIRE(std::strcmp(b.close_reason(), "freeze_churn") == 0);
    REQUIRE(b.apply(128822).token == 128822);
    REQUIRE(b.apply(271).token == 271);
    REQUIRE(b.settled());
  }

  // Negative case: 512 normal tokens + 256 varied Python/C++ code tokens with >=64 newlines and >60 distinct IDs.
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/0, /*freeze_enabled=*/true);
    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 512; ++i) {
      auto d = b.apply(10000 + i);
      REQUIRE(!d.cut);
    }
    for (int i = 0; i < 256; ++i) {
      const int32_t tok = (i % 3 == 2) ? 201 : (20000 + (i % 80));
      auto d = b.apply(tok);
      REQUIRE(!d.cut);
      REQUIRE(!b.force_closed());
      REQUIRE(!b.freeze_triggered());
    }
  }
  std::printf("PASS: test_executive_freeze_breaker\n");
}

// Scenario 5: Prefix-cache encode_chat invariance + server request parsing precedence.
void test_suppress_final_think_and_prefix_invariance() {
  // 5a. Server request parsing precedence.
  {
    const std::string body_ctk_none = R"({
      "model":"deepseek",
      "messages":[{"role":"user","content":"hi"}],
      "reasoning_effort":"none",
      "chat_template_kwargs":{"enable_thinking":true}
    })";
    CompletionRequest req1;
    REQUIRE(parse_completion_request(body_ctk_none, true, &req1));
    REQUIRE(req1.thinking_mode == "thinking");
    REQUIRE(req1.suppress_final_think == true);

    const std::string body_plain_none = R"({
      "model":"deepseek",
      "messages":[{"role":"user","content":"hi"}],
      "reasoning_effort":"off"
    })";
    CompletionRequest req2;
    REQUIRE(parse_completion_request(body_plain_none, true, &req2));
    REQUIRE(req2.thinking_mode == "chat");
    REQUIRE(req2.suppress_final_think == false);

    const std::string body_explicit = R"({
      "model":"deepseek",
      "messages":[{"role":"user","content":"hi"}],
      "thinking_mode":"thinking",
      "reasoning_effort":"0",
      "chat_template_kwargs":{"enable_thinking":true}
    })";
    CompletionRequest req3;
    REQUIRE(parse_completion_request(body_explicit, true, &req3));
    REQUIRE(req3.thinking_mode == "thinking");
    REQUIRE(req3.suppress_final_think == false);
  }

  // 5b. Build a minimal tokenizer.json in /tmp to verify exact token IDs end-to-end.
  char tmpl[] = "/tmp/aff_tb_tok_XXXXXX";
  int fd = ::mkstemp(tmpl);
  REQUIRE(fd >= 0);
  ::close(fd);
  const std::string tok_path(tmpl);
  {
    FILE* f = std::fopen(tok_path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::string j = R"({"model":{"vocab":{)";
    uint32_t next_id = 10;
    for (int ch = 33; ch <= 126; ++ch) {
      if (ch == '"' || ch == '\\') continue;
      char s[2] = {(char)ch, 0};
      j += "\"" + std::string(s) + "\":" + std::to_string(next_id++) + ",";
    }
    j += R"("\"":198,"\\":199,"\u0120":200,"\u010a":201,"\u010a\u010a":271})";
    j += R"(,"merges":["\u010a \u010a"]},)";
    j += R"("added_tokens":[)";
    j += R"({"id":0,"content":"<｜begin▁of▁sentence｜>"},)";
    j += R"({"id":1,"content":"<｜end▁of▁sentence｜>"},)";
    j += R"({"id":128803,"content":"<｜User｜>"},)";
    j += R"({"id":128804,"content":"<｜Assistant｜>"},)";
    j += R"({"id":128805,"content":"<｜System｜>"},)";
    j += R"({"id":128821,"content":"<think>"},)";
    j += R"({"id":128822,"content":"</think>"})";
    j += R"(]})";
    std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
  }

  Tokenizer tok;
  std::string err;
  REQUIRE(tok.load(tok_path, &err));
  std::remove(tok_path.c_str());

  REQUIRE((tok.encode("</think>", false) == std::vector<uint32_t>{128822}));
  REQUIRE((tok.encode("\n</think>\n\n", false) == std::vector<uint32_t>{201, 128822, 271}));

  std::vector<ChatMsg> msgs;
  {
    ChatMsg m;
    m.role = "system";
    m.content = "You are concise.";
    msgs.push_back(m);
  }
  {
    ChatMsg m;
    m.role = "user";
    m.content = "Compute 2+2.";
    msgs.push_back(m);
  }
  {
    ChatMsg m;
    m.role = "assistant";
    m.reasoning_content = "2 plus 2 equals 4.";
    m.content = "4";
    msgs.push_back(m);
  }
  {
    ChatMsg m;
    m.role = "user";
    m.content = "Now compute 3+3.";
    msgs.push_back(m);
  }

  EncodeOpts eo_think;
  eo_think.v41 = true;
  eo_think.thinking_mode = ThinkingMode::Thinking;
  eo_think.effort_budget = 75;
  eo_think.drop_thinking = false;
  eo_think.suppress_final_think = false;

  EncodeOpts eo_sup = eo_think;
  eo_sup.suppress_final_think = true;

  const std::string p_think = encode_messages(msgs, eo_think);
  const std::string p_sup = encode_messages(msgs, eo_sup);

  REQUIRE(p_sup.find("Reasoning Effort: 75") != std::string::npos);
  REQUIRE(p_sup.find("2 plus 2 equals 4.</think>4") != std::string::npos);

  const std::vector<uint32_t> ids_think = tok.encode(p_think, false);
  const std::vector<uint32_t> ids_sup = tok.encode(p_sup, false);

  REQUIRE(ids_think.size() == ids_sup.size());
  REQUIRE(ids_think.size() > 10);
  for (size_t i = 0; i + 1 < ids_think.size(); ++i) {
    REQUIRE(ids_think[i] == ids_sup[i]);
  }
  REQUIRE(ids_think.back() == 128821);
  REQUIRE(ids_sup.back() == 128822);

  std::vector<ChatMsg> non_gen = msgs;
  {
    ChatMsg m;
    m.role = "assistant";
    m.reasoning_content = "3 plus 3 is 6.";
    m.content = "6";
    non_gen.push_back(m);
  }
  REQUIRE(encode_messages(non_gen, eo_think) == encode_messages(non_gen, eo_sup));

  std::printf("PASS: test_suppress_final_think_and_prefix_invariance\n");
}

void test_full_vocab_freeze_short_loop_low_novelty_and_clean_trace() {
  // 1) Executive-freeze spiral using merged ".\n\n" token (990) & DRY-varied wording (~42 distinct tokens)
  {
    std::vector<int32_t> sorted_nl = {201, 271, 990, 1500};
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.freeze_nl_sorted = &sorted_nl;
    h.short_loop_span = 24;
    h.short_loop_repeats = 4;
    h.short_loop_window = 1024;
    h.novelty_window = 512;
    h.novelty_min_tokens = 1024;
    h.novelty_min_ratio = 0.28f;

    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 350; ++i) {
      auto d = b.apply(10000 + i);
      REQUIRE(!d.cut);
    }
    // 41 distinct word tokens + merged ".\n\n" token 990 = 42 distinct tokens (<= 56), >= 64 nl per 256.
    // Perturb cycle periodically so 24-tok short_loop does not trigger before freeze_churn.
    bool fired = false;
    for (int i = 0; i < 340; ++i) {
      const int32_t tok = (i % 3 == 2) ? 990 : (40000 + ((i * 7 + (i / 19)) % 41));
      auto d = b.apply(tok);
      if (d.cut) {
        fired = true;
        break;
      }
    }
    REQUIRE(fired);
    REQUIRE(b.force_closed());
    REQUIRE(b.thought_tokens() < 700);
    REQUIRE(std::strcmp(b.close_reason(), "freeze_churn") == 0);
  }

  // 2) Short-loop (span=24, repeats=4, window=1024) triggers early on 24-tok repeat.
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.short_loop_span = 24;
    h.short_loop_repeats = 4;
    h.short_loop_window = 1024;
    h.novelty_window = 512;
    h.novelty_min_tokens = 1024;
    h.novelty_min_ratio = 0.28f;

    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 50; ++i) REQUIRE(!b.apply(1000 + i).cut);
    bool fired = false;
    for (int rep = 0; rep < 4 && !fired; ++rep) {
      for (int j = 0; j < 24; ++j) {
        auto d = b.apply(30000 + j);
        if (d.cut) {
          fired = true;
          break;
        }
      }
    }
    REQUIRE(fired);
    REQUIRE(b.short_loop_triggered());
    REQUIRE(std::strcmp(b.close_reason(), "short_loop") == 0);
  }

  // 3) Bigram low-novelty spiral with DRY perturbations (no newlines, period > 1024 -> no short_loop).
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.short_loop_span = 24;
    h.short_loop_repeats = 4;
    h.short_loop_window = 1024;
    h.novelty_window = 512;
    h.novelty_min_tokens = 1024;
    h.novelty_min_ratio = 0.28f;

    ThinkingBudget b(&h, 16384);
    bool fired = false;
    for (int t = 0; t < 1500; ++t) {
      const int32_t tok = (t % 17 == 0) ? (7100 + ((t / 17) % 13)) : (7000 + (t % 12));
      auto d = b.apply(tok);
      if (d.cut) {
        fired = true;
        break;
      }
    }
    REQUIRE(fired);
    REQUIRE(b.novelty_triggered());
    REQUIRE(std::strcmp(b.close_reason(), "low_novelty") == 0);
  }

  // 4) Normal diverse math/code trace (2,000+ tokens) triggers ZERO false positives across all breakers.
  {
    std::vector<int32_t> sorted_nl = {201, 271, 990};
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.freeze_nl_sorted = &sorted_nl;
    h.short_loop_span = 24;
    h.short_loop_repeats = 4;
    h.short_loop_window = 1024;
    h.novelty_window = 512;
    h.novelty_min_tokens = 1024;
    h.novelty_min_ratio = 0.28f;

    ThinkingBudget b(&h, 16384);
    for (int t = 0; t < 2400; ++t) {
      const int32_t tok = (t % 20 == 19) ? 201 : (20000 + ((t * 131 + (t / 7) * 17) % 600));
      auto d = b.apply(tok);
      REQUIRE(!d.cut);
      REQUIRE(!b.force_closed());
    }
    auto end_dec = b.apply(128822);
    REQUIRE(!end_dec.cut);
    REQUIRE(b.naturally_closed());
    REQUIRE(!b.force_closed());
  }

  std::printf("PASS: test_full_vocab_freeze_short_loop_low_novelty_and_clean_trace\n");
}

// Scenario 7: Short mantra permutations (20-tok Let me execute/call/do/write/go) trigger
// short_loop (span=16, repeats=3) or short_low_novelty (window=128, min_tokens=160, min_ratio=0.20),
// while normal diverse reasoning over 2400 tokens stays clean.
void test_short_mantra_loop_and_short_novelty() {
  // 1) Exact 16-token span repeated 3x triggers short_loop.
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.short_loop_span = 16;
    h.short_loop_repeats = 3;
    h.short_loop_window = 1024;
    h.short_novelty_window = 128;
    h.short_novelty_min_tokens = 160;
    h.short_novelty_min_ratio = 0.20f;

    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 40; ++i) REQUIRE(!b.apply(1000 + i).cut);
    bool fired = false;
    for (int rep = 0; rep < 3 && !fired; ++rep) {
      for (int j = 0; j < 16; ++j) {
        if (b.apply(25000 + j).cut) { fired = true; break; }
      }
    }
    REQUIRE(fired);
    REQUIRE(b.short_loop_triggered());
    REQUIRE(std::strcmp(b.close_reason(), "short_loop") == 0);
  }

  // 2) Varying 20-token mantra permutations (~8-12 tokens, low bigram novelty) triggers short_low_novelty around ~160 tokens.
  {
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.short_loop_span = 16;
    h.short_loop_repeats = 3;
    h.short_loop_window = 1024;
    h.short_novelty_window = 128;
    h.short_novelty_min_tokens = 160;
    h.short_novelty_min_ratio = 0.20f;

    const int32_t verbs[5] = {8001, 8002, 8003, 8004, 8005}; // execute, call, do, write, go
    ThinkingBudget b(&h, 16384);
    for (int i = 0; i < 32; ++i) REQUIRE(!b.apply(12000 + i).cut);
    bool fired = false;
    int step = 0;
    while (!fired && step < 260) {
      const int j = step / 4;
      const int v = verbs[(j * 3 + (j / 5) + (j / 7) * 2 + (j * j) / 11) % 5];
      const int32_t phrase[4] = {9001, 9002, v, 9003}; // "\n\nLet me <verb>."
      for (int k = 0; k < 4; ++k) {
        if (b.apply(phrase[k]).cut) { fired = true; break; }
        ++step;
      }
    }
    REQUIRE(fired);
    REQUIRE(b.short_novelty_triggered());
    REQUIRE(b.thought_tokens() >= 160 && b.thought_tokens() < 240);
    REQUIRE(std::strcmp(b.close_reason(), "short_low_novelty") == 0);
  }

  // 3) Diverse reasoning over 2400 tokens triggers ZERO false positives.
  {
    std::vector<int32_t> sorted_nl = {201, 271, 990};
    BudgetHook h = make_hook(reply_reserve_within(4096, 16384), /*loop_repeats=*/3, /*freeze_enabled=*/true);
    h.freeze_distinct_max = 56;
    h.freeze_nl_sorted = &sorted_nl;
    h.short_loop_span = 16;
    h.short_loop_repeats = 3;
    h.short_loop_window = 1024;
    h.novelty_window = 512;
    h.novelty_min_tokens = 1024;
    h.novelty_min_ratio = 0.28f;
    h.short_novelty_window = 128;
    h.short_novelty_min_tokens = 160;
    h.short_novelty_min_ratio = 0.20f;

    ThinkingBudget b(&h, 16384);
    for (int t = 0; t < 2400; ++t) {
      const int32_t tok = (t % 20 == 19) ? 201 : (20000 + ((t * 131 + (t / 7) * 17) % 600));
      auto d = b.apply(tok);
      REQUIRE(!d.cut);
      REQUIRE(!b.force_closed());
    }
    REQUIRE(!b.apply(128822).cut);
    REQUIRE(b.naturally_closed());
  }

  std::printf("PASS: test_short_mantra_loop_and_short_novelty\n");
}

}  // namespace

int main() {
  test_natural_close();
  test_hard_limit();
  test_loop_repeat();
  test_executive_freeze_breaker();
  test_suppress_final_think_and_prefix_invariance();
  test_full_vocab_freeze_short_loop_low_novelty_and_clean_trace();
  test_short_mantra_loop_and_short_novelty();
  std::printf("ALL thinking_budget tests passed\n");
  return 0;
}
