#include "engine/suffix_draft.h"

#include <chrono>
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
  if (idx.rust_sam_active()) {
    REQUIRE(p.match_len == 24);
  } else {
    REQUIRE(p.match_len == 4 + SuffixDraftIndex::kExtra);
  }
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
  if (idx.rust_sam_active()) {
    REQUIRE(std::fabs(p.score - 1.0f) < 1e-5f);
  } else {
    REQUIRE(std::fabs(p.score - (2.0f / 3.0f)) < 1e-5f);
  }

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

void test_sam_parity_and_env_toggle() {
  std::vector<uint32_t> seq = {
      1001, 1002, 1003, 10, 11, 12, 13, 21, 2001,
      1004, 1005, 1006, 10, 11, 12, 13, 20, 30, 2002,
      1007, 1008, 1009, 10, 11, 12, 13, 20, 30, 2003,
      1010, 1011, 1012, 10, 11, 12, 13,
  };
  SuffixDraftIndex cpp_idx(false);
  REQUIRE(!cpp_idx.rust_sam_active());
  std::vector<int32_t> out_cpp(8, -1);
  auto pc = cpp_idx.propose(seq, 1, 4, 4, 0.5f, out_cpp.data());
  REQUIRE(pc.n == 2 && pc.match_len == 4);
  REQUIRE(out_cpp[0] == 20 && out_cpp[1] == 30);

#ifdef AFF_WITH_RUST
  // Default AFF_RUST_SAM=1 enables Rust SAM.
  ::unsetenv("AFF_RUST_SAM");
  SuffixDraftIndex rust_idx;
  REQUIRE(rust_idx.rust_sam_active());
  std::vector<int32_t> out_rust(8, -1);
  auto pr = rust_idx.propose(seq, 1, 4, 4, 0.5f, out_rust.data());
  REQUIRE(pr.n == pc.n);
  REQUIRE(pr.match_len == pc.match_len);
  REQUIRE(std::fabs(pr.score - pc.score) < 1e-5f);
  REQUIRE(std::fabs(pr.expect - pc.expect) < 1e-5f);
  for (int i = 0; i < pr.n; ++i) {
    REQUIRE(out_rust[i] == out_cpp[i]);
  }

  // AFF_RUST_SAM=0 cleanly falls back to C++ table.
  ::setenv("AFF_RUST_SAM", "0", 1);
  SuffixDraftIndex fallback_idx;
  REQUIRE(!fallback_idx.rust_sam_active());
  auto pf = fallback_idx.propose(seq, 1, 4, 4, 0.5f, out_rust.data());
  REQUIRE(pf.n == pc.n && pf.match_len == pc.match_len);
  ::unsetenv("AFF_RUST_SAM");
#endif

  std::printf("PASS: test_sam_parity_and_env_toggle\n");
}

void test_unbounded_match_len_24() {
  std::vector<uint32_t> seq;
  // Earlier distractor sharing only the trailing 7 tokens -> [9000..9011].
  seq.insert(seq.end(), {5001, 5002, 5003});
  for (uint32_t t = 1017; t < 1024; ++t) seq.push_back(t);
  for (uint32_t t = 9000; t < 9012; ++t) seq.push_back(t);

  // Exact 24-token pattern -> [2000..2011].
  for (uint32_t t = 1000; t < 1024; ++t) seq.push_back(t);
  for (uint32_t t = 2000; t < 2012; ++t) seq.push_back(t);
  seq.insert(seq.end(), {6001, 6002, 6003});
  for (uint32_t t = 1000; t < 1024; ++t) seq.push_back(t);

  SuffixDraftIndex idx;
  std::vector<int32_t> out(16, -1);
  auto p = idx.propose(seq, 42, 4, 12, 0.5f, out.data());
  REQUIRE(p.n == 12);
  if (idx.rust_sam_active()) {
    REQUIRE(p.match_len == 24);
    REQUIRE(std::fabs(p.score - 1.0f) < 1e-6f);
    REQUIRE(std::fabs(p.expect - 12.0f) < 1e-5f);
    for (int i = 0; i < 12; ++i) {
      REQUIRE(out[i] == static_cast<int32_t>(2000 + i));
    }
    SuffixGate gate;
    REQUIRE(gate.should_preempt(p, 4, 2.0f, true, 3.8f, 120.0f, 150.0f, 1.05f));
  } else {
    REQUIRE(p.match_len == 4 + SuffixDraftIndex::kExtra);
  }

  std::printf("PASS: test_unbounded_match_len_24\n");
}

void test_non_destructive_rollback_and_contains_ngram() {
  SuffixDraftIndex idx;
  const std::vector<uint32_t> prefix = {
      10, 11, 12, 13, 14, 15, 16, 17, 77, 78, 79, 80,
      90, 91, 92, 10, 11, 12, 13, 14, 15, 16, 17,
  };
  idx.reset(11);
  idx.append_delta(prefix.data(), prefix.size());
  REQUIRE(idx.filled() == prefix.size());

  const std::vector<uint32_t> p_ng = {10, 11, 12, 13, 14, 15};
  const std::vector<uint32_t> absent_ng = {10, 11, 12, 999};
  REQUIRE(idx.contains_ngram(p_ng.data(), p_ng.size()));
  REQUIRE(!idx.contains_ngram(absent_ng.data(), absent_ng.size()));
  REQUIRE(idx.contains_ngram_before(p_ng.data(), p_ng.size(), 8));
  REQUIRE(!idx.contains_ngram_before(p_ng.data(), p_ng.size(), 5));

  std::vector<uint32_t> extended = prefix;
  const std::vector<uint32_t> branch_b = {301, 302, 303, 304, 305};
  extended.insert(extended.end(), branch_b.begin(), branch_b.end());
  idx.append_delta(extended.data(), extended.size());
  REQUIRE(idx.filled() == extended.size());
  REQUIRE(idx.contains_ngram(branch_b.data(), branch_b.size()));
  REQUIRE(!idx.contains_ngram_before(branch_b.data(), branch_b.size(), prefix.size()));
  REQUIRE(idx.contains_ngram_before(branch_b.data(), branch_b.size(), extended.size()));

  // Non-destructive rollback_to preserves [0..prefix.size()) while dropping branch_b.
  idx.rollback_to(prefix.size());
  REQUIRE(idx.filled() == prefix.size());
  REQUIRE(idx.contains_ngram(p_ng.data(), p_ng.size()));
  REQUIRE(!idx.contains_ngram(branch_b.data(), branch_b.size()));

  std::vector<int32_t> out(8, -1);
  auto p = idx.propose(prefix, 11, 4, 4, 0.5f, out.data());
  REQUIRE(p.n == 4);
  REQUIRE(out[0] == 77 && out[1] == 78 && out[2] == 79 && out[3] == 80);

  // Append new branch C after rollback.
  std::vector<uint32_t> branch_c = prefix;
  const std::vector<uint32_t> c_tail = {401, 402, 403, 404};
  branch_c.insert(branch_c.end(), c_tail.begin(), c_tail.end());
  idx.append_delta(branch_c.data(), branch_c.size());
  REQUIRE(idx.contains_ngram(c_tail.data(), c_tail.size()));
  REQUIRE(!idx.contains_ngram(branch_b.data(), branch_b.size()));

  std::printf("PASS: test_non_destructive_rollback_and_contains_ngram\n");
}

void test_lru_lcp_pool_reuse() {
  SuffixSessionPool pool;

  // Build 64K base prompt.
  std::vector<uint32_t> turn1(65536);
  for (size_t i = 0; i < turn1.size(); ++i) {
    turn1[i] = static_cast<uint32_t>(1000u + (i % 50000u));
  }
  // Plant an exact 16-token pattern inside turn1.
  for (uint32_t j = 0; j < 16; ++j) turn1[1024 + j] = 800000u + j;
  for (uint32_t j = 0; j < 8; ++j) turn1[1040 + j] = 810000u + j;

  {
    auto lease1 = pool.acquire(turn1, 1ull);
    REQUIRE(lease1.lcp() == 0);
    // Simulate turn1 generation appending 128 tokens.
    for (uint32_t j = 0; j < 128; ++j) turn1.push_back(900000u + j);
    lease1.index().append_delta(turn1.data(), turn1.size());
    lease1.release(turn1);
  }
  REQUIRE(pool.size() == 1);
  REQUIRE(pool.stats().misses == 1);

  // Consecutive agent turn extending 64K -> +1024 delta tokens (fast LCP delta path).
  std::vector<uint32_t> turn2 = turn1;
  for (uint32_t j = 0; j < 1008; ++j) turn2.push_back(910000u + j);
  for (uint32_t j = 0; j < 16; ++j) turn2.push_back(800000u + j);

  {
    const auto t0 = std::chrono::steady_clock::now();
    auto lease2 = pool.acquire(turn2, 1ull);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(lease2.lcp() == turn1.size());
    std::vector<int32_t> out(8, -1);
    auto prop = lease2.index().propose(turn2, 1ull, 4, 8, 0.5f, out.data());
    REQUIRE(prop.n == 8);
    if (lease2.index().rust_sam_active()) {
      REQUIRE(prop.match_len == 16);
      REQUIRE(ms < 10.0);
    }
    for (int i = 0; i < 8; ++i) REQUIRE(out[i] == static_cast<int32_t>(810000 + i));
    lease2.release(turn2);
  }
  REQUIRE(pool.stats().hits == 1);

  // Partial LCP rollback (e.g. turn drop_thinking trims 128 tokens and extends to 140K).
  std::vector<uint32_t> turn3(turn1.begin(), turn1.begin() + 65536);
  turn3.resize(140000);
  for (size_t i = 65536; i < turn3.size(); ++i) {
    turn3[i] = static_cast<uint32_t>(70000u + (i % 40000u));
  }
  {
    auto lease3 = pool.acquire(turn3, 1ull);
    REQUIRE(lease3.lcp() == 65536);
    REQUIRE(lease3.index().filled() == 140000);
    const uint32_t needle[4] = {800000u, 800001u, 800002u, 800003u};
    REQUIRE(lease3.index().contains_ngram(needle, 4));
    lease3.release(turn3);
  }
  REQUIRE(pool.stats().hits == 2);

  // Fill remaining 7 pool entries and verify 8-entry LRU eviction + concurrent >8 fallback.
  std::vector<std::vector<uint32_t>> sessions(8);
  sessions[0] = turn3;
  for (size_t s = 1; s < 8; ++s) {
    sessions[s].assign(128, static_cast<uint32_t>(300000u + s * 1000u));
    auto ls = pool.acquire(sessions[s], 1ull);
    ls.release(sessions[s]);
  }
  REQUIRE(pool.size() == SuffixSessionPool::kCapacity);

  // Touch session 0 so session 1 becomes the LRU victim.
  {
    auto l0 = pool.acquire(sessions[0], 1ull);
    REQUIRE(l0.lcp() == sessions[0].size());
    l0.release(sessions[0]);
  }
  std::vector<uint32_t> session8(128, 450000u);
  {
    auto l8 = pool.acquire(session8, 1ull);
    REQUIRE(l8.lcp() == 0);
    l8.release(session8);
  }
  REQUIRE(pool.size() == SuffixSessionPool::kCapacity);
  // Session 0 survived; session 1 (LRU) was evicted.
  {
    auto l0_again = pool.acquire(sessions[0], 1ull);
    REQUIRE(l0_again.lcp() == sessions[0].size());
    l0_again.release(sessions[0]);
    auto l1_evicted = pool.acquire(sessions[1], 1ull);
    REQUIRE(l1_evicted.lcp() == 0);
    l1_evicted.release(sessions[1]);
  }

  // Holding 9 simultaneous leases exercises ephemeral fallback safely.
  std::vector<SuffixSessionPool::Lease> held;
  for (size_t s = 0; s < 9; ++s) {
    std::vector<uint32_t> tmp(64, static_cast<uint32_t>(500000u + s * 100u));
    held.push_back(pool.acquire(tmp, 1ull));
    REQUIRE(held.back().index().filled() == 64);
  }
  held.clear();

  std::printf("PASS: test_lru_lcp_pool_reuse\n");
}

void test_suffix_gate_deadlock_recovery() {
  aff::SuffixGate gate;
  gate.reset();
  const float dsp_acc = 4.68f;
  const float dsp_ms = 120.0f;

  // Simulate 10 proven suffix blocks (`12/12` accepted -> `good_steps == 10`).
  for (int b = 0; b < 10; ++b) {
    aff::SuffixDraftIndex::Proposal p{/*n=*/12, /*match_len=*/12, /*score=*/1.0f, /*expect=*/12.0f};
    const float sfx_ms = std::max(204.0f - 20.0f, 48.0f);
    REQUIRE(gate.should_preempt(p, 4, 2.0f, true, dsp_acc, dsp_ms, sfx_ms, 1.05f));
    gate.observe(12, 12);
  }
  REQUIRE(gate.good_steps == 10u);

  // Near-complete partial accept (11/12, only 1 rejected < 4) must NOT trigger edit-boundary guard,
  // allowing fast recovery on short m=7 matches.
  gate.observe(12, 11);
  REQUIRE(!gate.last_rejected);
  aff::SuffixDraftIndex::Proposal trap_7{/*n=*/12, /*match_len=*/7, /*score=*/1.0f, /*expect=*/12.0f};
  REQUIRE(gate.should_preempt(trap_7, 4, 2.0f, true, dsp_acc, dsp_ms, 184.0f, 1.05f));

  // Edit boundary rejecting >= 4 tokens with >= 50% accept (8/12) sets last_rejected and raises
  // req_match to min_match + 3 (7): m=6 is skipped cleanly while m=7 recovers.
  gate.observe(12, 8);
  REQUIRE(gate.last_rejected);
  aff::SuffixDraftIndex::Proposal trap_6{/*n=*/12, /*match_len=*/6, /*score=*/1.0f, /*expect=*/12.0f};
  REQUIRE(!gate.should_preempt(trap_6, 4, 2.0f, true, dsp_acc, dsp_ms, 184.0f, 1.05f));
  REQUIRE(gate.should_preempt(trap_7, 4, 2.0f, true, dsp_acc, dsp_ms, 184.0f, 1.05f));

  // Total 0/12 miss also sets last_rejected. Even after harsh 0/12 misses that drop acc_ema below
  // 0.65 and trigger cooldown, a verbatim streak (match_len >= 12, n >= 8, expect >= 8.0) breaks
  // the old deadlock immediately once m >= 12.
  gate.observe(12, 0);
  REQUIRE(gate.last_rejected);
  gate.observe(12, 0);
  REQUIRE(gate.miss_streak == 2u);
  REQUIRE(gate.cooldown == 4u);
  REQUIRE(gate.acc_ema < 0.65f);
  gate.observe_dspark();
  aff::SuffixDraftIndex::Proposal verbatim{
      /*n=*/12, /*match_len=*/14, /*score=*/1.0f, /*expect=*/12.0f};
  REQUIRE(gate.should_preempt(verbatim, 4, 2.0f, true, dsp_acc, dsp_ms, 184.0f, 1.05f));
  std::printf("PASS: test_suffix_gate_deadlock_recovery\n");
}

void test_suffix_gate_prose_unchanged() {
  aff::SuffixGate gate;
  gate.reset();
  const float dsp_acc = 2.6f;
  const float dsp_ms = 120.0f;
  // Prose with good_steps == 0 gets one 0/6 miss and stays backed off even on n=10, m=12.
  gate.observe(6, 0);
  REQUIRE(gate.good_steps == 0u);
  aff::SuffixDraftIndex::Proposal p{/*n=*/10, /*match_len=*/12, /*score=*/1.0f, /*expect=*/8.5f};
  REQUIRE(!gate.should_preempt(p, 4, 2.0f, true, dsp_acc, dsp_ms, 160.0f, 1.05f));
  std::printf("PASS: test_suffix_gate_prose_unchanged\n");
}

void test_suffix_burst_depth_15() {
  aff::SuffixDraftIndex idx(/*use_rust=*/true);
  std::vector<uint32_t> prompt;
  for (uint32_t i = 0; i < 64; ++i) prompt.push_back(1000u + (i % 32u));
  idx.append_delta(prompt.data(), prompt.size());
  // Last 12 tokens: 20..31; next 15 repeated tokens wrap 0..14 from index 32..46.
  std::vector<int32_t> out(16, -1);
  auto prop = idx.propose(prompt, 1ull, 4, 15, 0.35f, out.data());
  REQUIRE(prop.n == 15);
  REQUIRE(prop.match_len >= 12);
  for (int i = 0; i < 15; ++i) REQUIRE(out[i] == static_cast<int32_t>(1000u + i));

  // Verify burst_hi gate threshold: requires good_steps >= 3, acc_ema >= 0.94, match_len >= 16.
  aff::SuffixGate gate;
  gate.reset();
  gate.observe(12, 12);
  gate.observe(12, 12);
  REQUIRE(gate.good_steps == 2u);
  REQUIRE(gate.acc_ema >= 0.94f);
  REQUIRE(!(gate.good_steps >= 3u && gate.acc_ema >= 0.94f && idx.match_len() >= 16u));
  gate.observe(12, 12);
  REQUIRE(gate.good_steps == 3u);
  REQUIRE(gate.acc_ema >= 0.94f);
  if (idx.rust_sam_active()) {
    REQUIRE(idx.match_len() >= 16u);
  }
  std::printf("PASS: test_suffix_burst_depth_15\n");
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
  test_sam_parity_and_env_toggle();
  test_unbounded_match_len_24();
  test_non_destructive_rollback_and_contains_ngram();
  test_lru_lcp_pool_reuse();
  test_suffix_gate_deadlock_recovery();
  test_suffix_gate_prose_unchanged();
  test_suffix_burst_depth_15();
  std::printf("ALL suffix_draft tests passed\n");
  return 0;
}
