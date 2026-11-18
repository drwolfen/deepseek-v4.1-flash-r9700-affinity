#include "engine/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

namespace {

void utf8_append(std::string& s, uint32_t cp) {
  if (cp < 0x80) s.push_back((char)cp);
  else if (cp < 0x800) {
    s.push_back((char)(0xC0 | (cp >> 6)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    s.push_back((char)(0xE0 | (cp >> 12)));
    s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

std::string json_esc(const std::string& in) {
  std::string out = "\"";
  for (unsigned char c : in) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out.push_back((char)c);
    }
  }
  out.push_back('"');
  return out;
}

std::string write_test_tokenizer_json() {
  std::vector<int> cps(256, -1);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    if (printable) cps[b] = b;
  }
  for (int b = 0; b < 256; ++b) {
    if (cps[b] < 0) cps[b] = 256 + n++;
  }
  std::vector<std::string> b2u(256);
  for (int b = 0; b < 256; ++b) utf8_append(b2u[b], (uint32_t)cps[b]);

  std::string j = "{\n  \"model\": {\n    \"vocab\": {\n";
  uint32_t next_id = 0;
  auto add_vocab = [&](const std::string& tok) {
    if (next_id > 0) j += ",\n";
    j += "      " + json_esc(tok) + ": " + std::to_string(next_id++);
  };
  for (int b = 0; b < 256; ++b) add_vocab(b2u[b]);

  std::vector<std::pair<std::string, std::string>> merges = {
      {"h", "e"}, {"he", "l"}, {"hel", "l"}, {"hell", "o"},
      {b2u[(uint8_t)' '], "w"}, {b2u[(uint8_t)' '] + "w", "o"},
      {b2u[(uint8_t)' '] + "wo", "r"}, {b2u[(uint8_t)' '] + "wor", "l"},
      {b2u[(uint8_t)' '] + "worl", "d"},
      {"t", "h"}, {"th", "e"},
      {b2u[(uint8_t)' '], "t"}, {b2u[(uint8_t)' '] + "t", "h"},
      {b2u[(uint8_t)' '] + "th", "e"},
      {"p", "a"}, {"pa", "r"}, {"par", "a"}, {"para", "m"},
      {"1", "2"}, {"12", "3"},
  };
  for (const auto& [a, b] : merges) add_vocab(a + b);

  j += "\n    },\n    \"merges\": [\n";
  for (size_t i = 0; i < merges.size(); ++i) {
    if (i > 0) j += ",\n";
    j += "      " + json_esc(merges[i].first + " " + merges[i].second);
  }
  j += "\n    ]\n  },\n  \"added_tokens\": [\n";

  const std::vector<std::pair<uint32_t, std::string>> added = {
      {1000, "<｜begin▁of▁sentence｜>"},
      {1001, "<｜end▁of▁sentence｜>"},
      {1002, "<｜User｜>"},
      {1003, "<｜Assistant｜>"},
      {1004, "<｜System｜>"},
      {1005, "<think>"},
      {1006, "</think>"},
      {1007, "<｜DSML｜ calls>"},
      {1008, "</｜DSML｜ calls>"},
      {1009, "<｜DSML｜"},
  };
  for (size_t i = 0; i < added.size(); ++i) {
    if (i > 0) j += ",\n";
    j += "    {\"id\": " + std::to_string(added[i].first) +
         ", \"content\": " + json_esc(added[i].second) + ", \"special\": true}";
  }
  j += "\n  ]\n}\n";

  char tmpl[] = "/tmp/aff_tok_test_XXXXXX";
  int fd = ::mkstemp(tmpl);
  CHECK(fd >= 0);
  size_t off = 0;
  while (off < j.size()) {
    ssize_t w = ::write(fd, j.data() + off, j.size() - off);
    CHECK(w > 0);
    off += (size_t)w;
  }
  ::close(fd);
  return std::string(tmpl);
}

std::vector<uint32_t> fresh_encode(const std::string& path, const std::string& text, bool add_bos) {
  aff::Tokenizer fresh;
  std::string err;
  CHECK(fresh.load(path, &err));
  return fresh.encode(text, add_bos);
}

} // namespace

int main() {
  const std::string tok_path = write_test_tokenizer_json();

  aff::Tokenizer tok;
  std::string err;
  CHECK(tok.load(tok_path, &err));
  CHECK(tok.bos() == 1000);
  CHECK(tok.eos() == 1001);
  CHECK(tok.vocab_size() == 1010);

  // 1. O(1) decode_one parity + out-of-range safety
  for (uint32_t id = 0; id < tok.vocab_size(); ++id) {
    CHECK(tok.decode_one(id) == tok.decode({id}));
  }
  CHECK(tok.decode_one(9999999u).empty());
  CHECK(tok.decode_one(1007) == "<｜DSML｜ calls>");

  // 2. Multi-turn DSML transcript + Unicode + NBSP (\xC2\xA0) + digits
  const std::string sys =
      "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75\n\nYou are helpful. NBSP:\xC2\xA0" "end.";
  const std::string u1 =
      "<｜User｜>hello world 1234567 北京 🚀 check the param please.";
  const std::string a1 =
      "<｜Assistant｜><think>the world is hello</think>\n\n"
      "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n"
      "<｜DSML｜ parameter name=\"cmd\" string=\"true\">echo hello\xC2\xA0" "world</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
  const std::string t1 =
      "<｜User｜>Tool output: hello\xC2\xA0" "world 北京 🚀<｜Assistant｜><think>\n";

  const std::string turn1 = sys + u1 + "<｜Assistant｜><think>\n";
  const std::string turn2 = sys + u1 + a1 + t1;
  const std::string turn3 = turn2 + "final answer: hello world 123.<｜end▁of▁sentence｜><｜User｜>next?<｜Assistant｜><think>\n";
  const std::string fork2b = sys + u1 + "<｜Assistant｜><think>diverged branch</think>done<｜end▁of▁sentence｜>";
  const std::string sess_b = "<｜begin▁of▁sentence｜><｜System｜>Different system prompt 999<｜User｜>hi 🚀<｜Assistant｜><think>\n";

  const std::vector<std::pair<std::string, bool>> cases = {
      {turn1, false},
      {turn1, false},  // exact cache hit
      {turn2, false},  // LCP hit through turn1 boundary
      {turn3, false},  // LCP hit through turn2 boundary
      {fork2b, false}, // forked branch off turn1 prefix
      {sess_b, false}, // separate session
      {turn3, false},  // multi-slot LCP hit back on session A
      {turn1, true},   // add_bos=true must not collide with add_bos=false
      {turn2, true},   // LCP hit with add_bos=true
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& [txt, add_bos] = cases[i];
    const auto got = tok.encode(txt, add_bos);
    const auto ref = fresh_encode(tok_path, txt, add_bos);
    CHECK(got == ref);
    // Verify round-trip (stripping added BOS if add_bos=true)
    std::string decoded = tok.decode(got);
    if (add_bos) {
      CHECK(decoded == std::string("<｜begin▁of▁sentence｜>") + txt);
    } else {
      CHECK(decoded == txt);
    }
  }

  // 3. Longest-match-first boundary near end of LCP (make sure <｜DSML｜ vs <｜DSML｜ calls> is exact)
  const std::string dsml_edge_a = sys + "<｜User｜><｜DSML｜ calls>abc";
  const std::string dsml_edge_b = sys + "<｜User｜><｜DSML｜ invoke>xyz";
  CHECK(tok.encode(dsml_edge_a, false) == fresh_encode(tok_path, dsml_edge_a, false));
  CHECK(tok.encode(dsml_edge_b, false) == fresh_encode(tok_path, dsml_edge_b, false));

  // 4. Concurrent multi-thread encode/decode_one parity
  const auto ref_turn3 = fresh_encode(tok_path, turn3, false);
  const auto ref_sessb = fresh_encode(tok_path, sess_b, true);
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int it = 0; it < 20; ++it) {
        if ((t + it) & 1) {
          auto ids = tok.encode(turn3, false);
          CHECK(ids == ref_turn3);
        } else {
          auto ids = tok.encode(sess_b, true);
          CHECK(ids == ref_sessb);
        }
        CHECK(tok.decode_one(1007) == "<｜DSML｜ calls>");
      }
    });
  }
  for (auto& th : threads) th.join();

  ::unlink(tok_path.c_str());
  std::printf("PASS: test_tokenizer\n");
  return 0;
}
