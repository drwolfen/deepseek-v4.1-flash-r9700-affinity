#include "engine/suffix_draft.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace {

using namespace aff;

void test_empty_and_short_seq() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(16, -1);

  // Empty sequence.
  {
    std::vector<uint32_t> seq;
    auto p = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/8, 0.5f, out.data());
    REQUIRE(p.n == 0);
    REQUIRE(p.match_len == 0);
    REQUIRE(p.score == 0.0f);
    REQUIRE(p.expect == 0.0f);
    REQUIRE(idx.owner() == 1);
  }

  // Short sequences (len <= min_match).
  for (uint32_t len = 1; len <= 4; ++len) {
    std::vector<uint32_t> seq(len, 42);
    auto p = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/8, 0.5f, out.data());
    REQUIRE(p.n == 0);
    REQUIRE(p.match_len == 0);
  }

  // Sequence of length min_match + 1 with no repeated suffix.
  {
    std::vector<uint32_t> seq = {10, 11, 12, 13, 14};
    auto p = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/8, 0.5f, out.data());
    REQUIRE(p.n == 0);
    REQUIRE(p.match_len == 0);
  }

  // Invalid parameters (min_match <= 0 or max_draft <= 0 or nullptr).
  {
    std::vector<uint32_t> seq = {1, 2, 3, 4, 5, 1, 2, 3, 4};
    auto p1 = idx.propose(seq, 1, 0, 8, 0.5f, out.data());
    REQUIRE(p1.n == 0);
    auto p2 = idx.propose(seq, 1, 4, 0, 0.5f, out.data());
    REQUIRE(p2.n == 0);
    auto p3 = idx.propose(seq, 1, 4, 8, 0.5f, nullptr);
    REQUIRE(p3.n == 0);
  }

  std::printf("PASS: test_empty_and_short_seq\n");
}

void test_exact_repeated_code_pattern() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(16, -1);

  // Code-like token pattern repeated, ending mid-pattern.
  const std::vector<uint32_t> pat = {
      100, 101, 102, 103, 104, 105, 106, 107,
      108, 109, 110, 111, 112, 113, 114, 115,
  };
  std::vector<uint32_t> seq;
  seq.insert(seq.end(), pat.begin(), pat.end());
  seq.insert(seq.end(), pat.begin(), pat.end());
  // Append first 8 tokens [100..107] so recurring continuation is [108..115].
  seq.insert(seq.end(), pat.begin(), pat.begin() + 8);

  auto p = idx.propose(seq, /*owner=*/7, /*min_match=*/4, /*max_draft=*/8, 0.5f, out.data());
  REQUIRE(p.n == 8);
  REQUIRE(p.match_len == 4 + SuffixDraftIndex::kExtra);
  REQUIRE(std::fabs(p.score - 1.0f) < 1e-6f);
  REQUIRE(std::fabs(p.expect - 8.0f) < 1e-5f);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(out[i] == static_cast<int32_t>(108 + i));
  }

  // Exact min_match=4 repeat only (prefix differs before last 4 tokens).
  std::vector<uint32_t> seq4 = {
      500, 501, 502, 10, 20, 30, 40, 41, 42, 43, 44, 45,
      600, 601, 602, 10, 20, 30, 40,
  };
  auto p4 = idx.propose(seq4, /*owner=*/8, /*min_match=*/4, /*max_draft=*/5, 0.5f, out.data());
  REQUIRE(p4.n == 5);
  REQUIRE(p4.match_len == 4);
  REQUIRE(std::fabs(p4.score - 1.0f) < 1e-6f);
  REQUIRE(std::fabs(p4.expect - 5.0f) < 1e-5f);
  for (int i = 0; i < 5; ++i) {
    REQUIRE(out[i] == 41 + i);
  }

  std::printf("PASS: test_exact_repeated_code_pattern\n");
}

void test_longest_suffix_wins() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(8, -1);

  // Earlier occurrence has full 7-token (min_match + 3) match -> [100, 101, 102].
  // Later occurrence only matches trailing 4 tokens [4, 5, 6, 7] -> [200, 201, 202].
  // Ending with [1, 2, 3, 4, 5, 6, 7] must select the 7-token match over the newer 4-token match.
  std::vector<uint32_t> seq = {
      1, 2, 3, 4, 5, 6, 7, 100, 101, 102,
      1, 2, 3, 4, 5, 6, 7, 100, 101, 102,
      91, 92, 93, 4, 5, 6, 7, 200, 201, 202,
      81, 82, 83, 1, 2, 3, 4, 5, 6, 7,
  };

  auto p = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/3, 0.5f, out.data());
  REQUIRE(p.match_len == 4 + SuffixDraftIndex::kExtra);
  REQUIRE(p.n == 3);
  REQUIRE(out[0] == 100);
  REQUIRE(out[1] == 101);
  REQUIRE(out[2] == 102);
  REQUIRE(std::fabs(p.score - (2.0f / 3.0f)) < 1e-5f);

  std::printf("PASS: test_longest_suffix_wins\n");
}

void test_branching_suffix_probability_and_cut() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(8, -1);

  // A=10, B=11, C=12, D=13; X=20 (twice), Y=21 (once).
  // Distinct pre/post tokens prevent matches longer than min_match=4.
  std::vector<uint32_t> seq = {
      1001, 1002, 1003, 10, 11, 12, 13, 21, 2001,
      1004, 1005, 1006, 10, 11, 12, 13, 20, 2002,
      1007, 1008, 1009, 10, 11, 12, 13, 20, 2003,
      1010, 1011, 1012, 10, 11, 12, 13,
  };

  // Strict threshold min_score=0.8 cuts because P(X | A B C D) = 2/3 < 0.8.
  auto p_hi = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/4, 0.8f, out.data());
  REQUIRE(p_hi.n == 0);
  REQUIRE(p_hi.score == 0.0f);
  REQUIRE(p_hi.expect == 0.0f);

  // Threshold min_score=0.5 accepts X (2/3 >= 0.5), then cuts on 2003 (D = 2/3 * 1/2 = 1/3 < 0.5).
  auto p_lo = idx.propose(seq, /*owner=*/1, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_lo.n == 1);
  REQUIRE(p_lo.match_len == 4);
  REQUIRE(out[0] == 20);
  REQUIRE(std::fabs(p_lo.score - (2.0f / 3.0f)) < 1e-4f);
  REQUIRE(std::fabs(p_lo.expect - (2.0f / 3.0f)) < 1e-4f);

  // Multi-step continuation after branch: if [B, C, D, X] always precedes Z=30,
  // both X and Z are drafted with D = 2/3 and expect = 4/3.
  std::vector<uint32_t> seq2 = {
      1001, 1002, 1003, 10, 11, 12, 13, 21, 2001,
      1004, 1005, 1006, 10, 11, 12, 13, 20, 30, 2002,
      1007, 1008, 1009, 10, 11, 12, 13, 20, 30, 2003,
      1010, 1011, 1012, 10, 11, 12, 13,
  };
  auto p_chain = idx.propose(seq2, /*owner=*/2, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_chain.n == 2);
  REQUIRE(out[0] == 20);
  REQUIRE(out[1] == 30);
  REQUIRE(std::fabs(p_chain.score - (2.0f / 3.0f)) < 1e-4f);
  REQUIRE(std::fabs(p_chain.expect - (4.0f / 3.0f)) < 1e-4f);

  std::printf("PASS: test_branching_suffix_probability_and_cut\n");
}

void test_incremental_matches_fresh_bitwise() {
  std::vector<uint32_t> full;
  full.reserve(320);
  for (uint32_t rep = 0; rep < 10; ++rep) {
    for (uint32_t t = 0; t < 24; ++t) {
      full.push_back(100 + t);
    }
    // Periodic branch perturbation.
    full.push_back((rep % 3 == 0) ? 900 : 901);
    for (uint32_t t = 0; t < 7; ++t) {
      full.push_back(200 + t);
    }
  }

  SuffixDraftIndex inc;
  std::vector<uint32_t> cur;
  std::vector<int32_t> out_inc(16, 0);
  std::vector<int32_t> out_fresh(16, 0);

  for (size_t i = 0; i < full.size(); ++i) {
    cur.push_back(full[i]);
    auto pi = inc.propose(cur, /*owner=*/55, /*min_match=*/4, /*max_draft=*/12, 0.35f, out_inc.data());

    SuffixDraftIndex fresh;
    auto pf = fresh.propose(cur, /*owner=*/55, /*min_match=*/4, /*max_draft=*/12, 0.35f, out_fresh.data());

    REQUIRE(pi.n == pf.n);
    REQUIRE(pi.match_len == pf.match_len);
    REQUIRE(pi.score == pf.score);
    REQUIRE(pi.expect == pf.expect);
    for (int j = 0; j < pi.n; ++j) {
      REQUIRE(out_inc[j] == out_fresh[j]);
    }
  }

  // Repeated call at same length produces identical output.
  std::vector<int32_t> out_repeat(16, 0);
  auto pr = inc.propose(cur, /*owner=*/55, /*min_match=*/4, /*max_draft=*/12, 0.35f, out_repeat.data());
  SuffixDraftIndex fresh_end;
  auto pf_end = fresh_end.propose(cur, /*owner=*/55, /*min_match=*/4, /*max_draft=*/12, 0.35f, out_fresh.data());
  REQUIRE(pr.n == pf_end.n);
  REQUIRE(pr.match_len == pf_end.match_len);
  REQUIRE(pr.score == pf_end.score);
  REQUIRE(pr.expect == pf_end.expect);
  for (int j = 0; j < pr.n; ++j) {
    REQUIRE(out_repeat[j] == out_fresh[j]);
  }

  std::printf("PASS: test_incremental_matches_fresh_bitwise\n");
}

void test_owner_change_shrink_and_rollback() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(16, -1);

  std::vector<uint32_t> seq_a = {
      10, 11, 12, 13, 77, 78, 79,
      20, 21, 22, 10, 11, 12, 13,
  };
  auto pa = idx.propose(seq_a, /*owner=*/100, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(pa.n == 4);
  REQUIRE(out[0] == 77 && out[1] == 78 && out[2] == 79 && out[3] == 20);
  REQUIRE(idx.owner() == 100);

  // Switch owner even with longer sequence: old request A data must be cleared.
  std::vector<uint32_t> seq_b = {
      30, 31, 32, 33, 88, 89,
      40, 41, 42, 43, 44, 45, 10, 11, 12, 13,
  };
  auto pb = idx.propose(seq_b, /*owner=*/200, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(idx.owner() == 200);
  REQUIRE(pb.n == 0);  // [10, 11, 12, 13] only existed in owner 100.

  // Add pattern inside owner 200 and verify it drafts properly.
  seq_b.insert(seq_b.end(), {30, 31, 32, 33});
  auto pb2 = idx.propose(seq_b, /*owner=*/200, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(pb2.n == 4);
  REQUIRE(out[0] == 88 && out[1] == 89 && out[2] == 40 && out[3] == 41);

  // Shrink sequence with same owner: triggers automatic reset and drops removed tail.
  seq_b.resize(6);  // [30, 31, 32, 33, 88, 89]
  auto p_shrink = idx.propose(seq_b, /*owner=*/200, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_shrink.n == 0);
  REQUIRE(idx.filled() == 6);

  // Appending [30, 31, 32, 33] after shrink recovers match cleanly.
  seq_b.insert(seq_b.end(), {30, 31, 32, 33});
  auto p_after = idx.propose(seq_b, /*owner=*/200, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_after.n == 4);
  REQUIRE(out[0] == 88 && out[1] == 89 && out[2] == 30 && out[3] == 31);

  // Explicit roll_back_to resets cleanly even if seq grows back past old filled_.
  idx.roll_back_to(4);
  REQUIRE(idx.filled() == 0);
  seq_b.assign({90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 30, 31, 32, 33});
  auto p_rb = idx.propose(seq_b, /*owner=*/200, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_rb.n == 0);

  // Changing min_match also resets automatically (and clamps before seq.size()).
  std::vector<uint32_t> seq_m = {1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5};
  auto pm5 = idx.propose(seq_m, /*owner=*/200, /*min_match=*/5, /*max_draft=*/8, 0.5f, out.data());
  REQUIRE(pm5.n == 6);
  REQUIRE(pm5.match_len == 5);
  REQUIRE(out[0] == 9 && out[1] == 1 && out[2] == 2 && out[3] == 3 && out[4] == 4 && out[5] == 5);

  std::printf("PASS: test_owner_change_shrink_and_rollback\n");
}

void test_high_load_resize_stability() {
  SuffixDraftIndex idx;
  std::vector<int32_t> out(16, -1);

  std::vector<uint32_t> seq;
  seq.reserve(15050);

  // Plant a unique 7-token pattern right at the start (before multiple table resizes).
  const std::vector<uint32_t> planted = {77001, 77002, 77003, 77004, 77005, 77006, 77007};
  const std::vector<uint32_t> cont = {88001, 88002, 88003, 88004};
  seq.insert(seq.end(), planted.begin(), planted.end());
  seq.insert(seq.end(), cont.begin(), cont.end());

  // Fill > 15,000 high-entropy unique tokens via LCG to force many open-table resizes.
  uint64_t state = 0x123456789abcdef0ULL;
  for (size_t i = 0; i < 15000; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t tok = 100000U + static_cast<uint32_t>((state >> 32) % 500000U);
    seq.push_back(tok);
    if ((i % 2500) == 2499) {
      // Periodically invoke propose during growth across resize boundaries.
      auto p_mid = idx.propose(seq, /*owner=*/999, /*min_match=*/4, /*max_draft=*/8, 0.4f, out.data());
      REQUIRE(p_mid.n >= 0 && p_mid.n <= 8);
    }
  }

  // Now append the planted suffix again and verify it survived all resizes intact.
  seq.insert(seq.end(), planted.begin(), planted.end());
  auto p_end = idx.propose(seq, /*owner=*/999, /*min_match=*/4, /*max_draft=*/4, 0.5f, out.data());
  REQUIRE(p_end.n == 4);
  REQUIRE(p_end.match_len == 7);
  REQUIRE(std::fabs(p_end.score - 1.0f) < 1e-6f);
  for (int i = 0; i < 4; ++i) {
    REQUIRE(out[i] == static_cast<int32_t>(cont[i]));
  }

  std::printf("PASS: test_high_load_resize_stability\n");
}

void test_suffix_gate_prose_backoff() {
  SuffixGate gate;
  // Prose DSpark runs strongly (~3.8 accepted in 120 ms -> utility 4.8/120 = 0.040 tok/ms).
  // Bare 4-token (min_match) and cold 5-token collisions are rejected immediately under DSpark
  // with 0 wasted probes.
  SuffixDraftIndex::Proposal bare_4{/*n=*/12, /*match_len=*/4, /*score=*/0.80f, /*expect=*/12.0f};
  SuffixDraftIndex::Proposal cold_5{/*n=*/12, /*match_len=*/5, /*score=*/0.80f, /*expect=*/12.0f};
  REQUIRE(!gate.should_preempt(bare_4, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  REQUIRE(!gate.should_preempt(cold_5, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));

  // A cold 6-token (min_match + 2) collision exceeds DSpark utility (11.8/204 = 0.0578 >= 0.0420),
  // probes once, gets only 1 accepted token (k=1 of 12), dropping acc_ema < 0.60 and arming cooldown=2.
  SuffixDraftIndex::Proposal chance_6{/*n=*/12, /*match_len=*/6, /*score=*/0.80f, /*expect=*/12.0f};
  REQUIRE(gate.should_preempt(chance_6, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  gate.observe(12, 1);
  REQUIRE(gate.acc_ema < 0.60f);
  REQUIRE(gate.cooldown == 2);

  // Next 2 calls (even with match_len=6 or 7) do NOT break cooldown because acc_ema < 0.65f,
  // decrementing cooldown down to 0.
  SuffixDraftIndex::Proposal chance_7{/*n=*/12, /*match_len=*/7, /*score=*/0.80f, /*expect=*/12.0f};
  REQUIRE(!gate.should_preempt(chance_6, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  REQUIRE(!gate.should_preempt(chance_7, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  REQUIRE(gate.cooldown == 0);

  // Even after cooldown clears, acc_ema ~ 0.53 -> sfx_u=(1+6.39)/204=0.0362 < 0.0420 -> stays DSpark.
  REQUIRE(!gate.should_preempt(chance_6, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));

  std::printf("PASS: test_suffix_gate_prose_backoff\n");
}

void test_suffix_gate_code_edit_burst() {
  SuffixGate gate;
  SuffixDraftIndex::Proposal code_long{/*n=*/12, /*match_len=*/7, /*score=*/1.0f, /*expect=*/12.0f};

  // Fresh gate enters immediately on a long exact match (match_len=7 >= min_match+2).
  REQUIRE(gate.should_preempt(code_long, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));

  // Cooldown breakout works when acc_ema >= 0.65f (simulate miss starting from high EMA).
  SuffixGate gate_cd;
  gate_cd.observe(6, 1);
  REQUIRE(gate_cd.cooldown == 2);
  gate_cd.acc_ema = 0.90f;
  REQUIRE(gate_cd.should_preempt(code_long, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  REQUIRE(gate_cd.cooldown == 0);

  // Full acceptance bursts maintain high acc_ema >= 0.95 and keep cooldown at 0.
  for (int i = 0; i < 4; ++i) {
    gate.observe(12, 12);
    REQUIRE(gate.cooldown == 0);
    REQUIRE(gate.miss_streak == 0);
    REQUIRE(gate.should_preempt(code_long, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  }
  REQUIRE(gate.acc_ema >= 0.95f);

  // In a proven streak (steps > 0, miss_streak == 0, acc_ema >= 0.80f), match_len=5 is accepted
  // while bare match_len=4 is still rejected under dspark_active=true.
  SuffixDraftIndex::Proposal streak_5{/*n=*/12, /*match_len=*/5, /*score=*/1.0f, /*expect=*/12.0f};
  SuffixDraftIndex::Proposal bare_4{/*n=*/12, /*match_len=*/4, /*score=*/1.0f, /*expect=*/12.0f};
  REQUIRE(gate.should_preempt(streak_5, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));
  REQUIRE(!gate.should_preempt(bare_4, 4, 2.0f, true, 3.8f, 120.0f, 204.0f, 1.05f));

  std::printf("PASS: test_suffix_gate_code_edit_burst\n");
}

void test_suffix_gate_no_dspark() {
  SuffixGate gate;
  SuffixDraftIndex::Proposal modest{/*n=*/4, /*match_len=*/4, /*score=*/0.7f, /*expect=*/3.0f};
  REQUIRE(gate.should_preempt(modest, 4, 2.0f, /*dspark_active=*/false, 3.8f, 120.0f, 108.0f,
                              1.05f));
  REQUIRE(!gate.should_preempt(modest, 4, 2.0f, /*dspark_active=*/true, 3.8f, 120.0f, 108.0f,
                               1.05f));

  std::printf("PASS: test_suffix_gate_no_dspark\n");
}

}  // namespace

int main() {
  test_empty_and_short_seq();
  test_exact_repeated_code_pattern();
  test_longest_suffix_wins();
  test_branching_suffix_probability_and_cut();
  test_incremental_matches_fresh_bitwise();
  test_owner_change_shrink_and_rollback();
  test_high_load_resize_stability();
  test_suffix_gate_prose_backoff();
  test_suffix_gate_code_edit_burst();
  test_suffix_gate_no_dspark();
  std::printf("ALL suffix_draft tests passed\n");
  return 0;
}
