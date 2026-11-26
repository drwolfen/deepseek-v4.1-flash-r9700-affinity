#include "engine/cpu_dispatch.h"
#include "engine/engram.h"
#include "engine/expert_heat.h"
#include "engine/expert_kernel.h"
#include "engine/model.h"
#include "format/aff_format.h"
#include "quant/blockquant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

namespace {

using namespace aff;

void test_basic_tie_break_and_guards() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 0.98f;
  p.hysteresis = 1.25f;
  p.min_gain = 1.5f;
  eh.init(/*n_layer=*/2, /*n_expert=*/4, p);

  // Layer 0 holds 3 resident experts; layer 1 holds 1 resident expert.
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Vram);
  eh.set_tier(2, ExpertTier::Vram);
  eh.set_tier(3, ExpertTier::Pool);

  eh.set_tier(4, ExpertTier::Vram);
  eh.set_tier(5, ExpertTier::Pool);
  eh.set_tier(6, ExpertTier::Pool);
  eh.set_tier(7, ExpertTier::Pool);

  const uint32_t sel_e = 1; // expert 1 in layer 1 -> global idx 5
  const uint32_t cnt = 2;
  eh.observe(1, &sel_e, 1, &cnt);

  uint32_t prom = 0, dem = 0;
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 5u);
  // Victim at heat==0 comes from layer 0 (most resident experts = 3).
  CHECK(dem < 4u);

  std::printf("PASS: test_basic_tie_break_and_guards\n");
}

void test_max_block_credit() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 0.992f;
  p.hysteresis = 1.25f;
  p.min_gain = 3.0f;
  p.max_block_credit = 2;
  eh.init(/*n_layer=*/1, /*n_expert=*/4, p);

  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Pool);

  // Single-block sighting with count=6 must be capped at +2.0 (< min_gain 3.0).
  const uint32_t e = 1;
  const uint32_t count = 6;
  eh.observe(0, &e, 1, &count);
  CHECK(eh.heat(1) == 2.0f);

  uint32_t prom = 0, dem = 0;
  CHECK(!eh.pick(&prom, &dem));

  eh.note_missing(0, &e, 1);
  CHECK(!eh.pick_demand(&prom, &dem));

  // Second block sighting reaches 2.0 * 0.992 + 2.0 = 3.984 >= 3.0 -> passes.
  eh.observe(0, &e, 1, &count);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 1u && dem == 0u);

  eh.note_missing(0, &e, 1);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 1u && dem == 0u);

  std::printf("PASS: test_max_block_credit\n");
}

void test_cooldown_steps() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 1.0f;
  p.hysteresis = 1.0f;
  p.min_gain = 1.0f;
  p.cooldown_steps = 16;
  const uint32_t n_layer = 3;
  const uint32_t n_expert = 4;
  eh.init(n_layer, n_expert, p);

  // Initial tier setup (ticks == 0) must not arm cooldown.
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Vram);
  eh.set_tier(2, ExpertTier::Pool);
  eh.set_tier(3, ExpertTier::Pool);

  // Advance 1 tick so ticks_ > 0, and heat up pooled expert 2.
  const uint32_t e2 = 2;
  const uint32_t c5 = 5;
  eh.observe(0, &e2, 1, &c5);
  CHECK(eh.ticks() == 1);

  uint32_t prom = 0, dem = 0;
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 2u);

  // Publish swap: 2 -> Vram, 0 -> Pool (both enter cooldown for 16 * n_layer = 48 ticks).
  eh.set_tier(2, ExpertTier::Vram);
  eh.set_tier(0, ExpertTier::Pool);

  // Even if expert 0 becomes hotter than expert 1, cooldown blocks re-promotion/demotion
  // for 16 * n_layer - 1 ticks.
  const uint32_t e0 = 0;
  const uint32_t c10 = 10;
  for (uint32_t t = 0; t < p.cooldown_steps * n_layer - 1; ++t) {
    eh.observe(0, &e0, 1, &c10);
    // Expert 0 (on cooldown) cannot be promoted; expert 2 (on cooldown) cannot be demoted,
    // though expert 1 is not on cooldown so only 0 is blocked as candidate.
    CHECK(!eh.pick(&prom, &dem));
    eh.note_missing(0, &e0, 1);
    CHECK(!eh.pick_demand(&prom, &dem));
  }

  // On the 16 * n_layer-th tick, cooldown expires and expert 0 can be promoted again.
  eh.observe(0, &e0, 1, &c10);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 0u && dem == 1u);

  eh.note_missing(0, &e0, 1);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 0u && dem == 1u);

  // Verify cooldown also protects a recently promoted Vram victim from immediate demotion.
  // Publish the (0 -> Vram, 1 -> Pool) swap, then place idx 4 in Vram (all at ticks > 0 -> cooldown).
  // Expert 3 was placed in Pool at ticks==0 (not on cooldown).
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Pool);
  eh.set_tier(4, ExpertTier::Vram);
  const uint32_t e_l0_2 = 2;
  const uint32_t e_l0_3 = 3;
  eh.observe(0, &e_l0_2, 1, &c10);
  eh.observe(0, &e_l0_3, 1, &c10);
  // Now Vram experts: 0 (high heat, on cooldown), 2 (cooldown expired, heat=15), 4 (ON COOLDOWN, heat=0).
  // Pool expert 3 (not on cooldown, heat=10): fails against 2 (10 < 15+1) and must skip 4 due to cooldown.
  CHECK(!eh.pick(&prom, &dem));
  eh.note_missing(0, &e_l0_3, 1);
  CHECK(!eh.pick_demand(&prom, &dem));

  // Advance cooldown_steps * n_layer ticks on layer 2 (leaves layer 0 & 1 heats untouched).
  const uint32_t e_l2_0 = 0;
  for (uint32_t t = 0; t < p.cooldown_steps * n_layer; ++t) eh.observe(2, &e_l2_0, 1, &c5);
  CHECK(eh.pick(&prom, &dem));
  CHECK(prom == 3u && dem == 4u);

  std::printf("PASS: test_cooldown_steps\n");
}

void test_engram_barrett_and_push_parity() {
  // 1. Barrett / Lemire 128-bit fastmod parity across edge divisors and edge dividends.
  constexpr uint64_t kMaxU64 = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kMaxI64 = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const uint64_t primes[] = {
    1ULL, 2ULL, 3ULL, 5ULL, 7ULL, 17ULL, 65521ULL, 99091ULL, 16000001ULL,
    (1ULL << 31) - 1ULL, (1ULL << 32) + 15ULL, (1ULL << 62) + 1ULL, kMaxU64 - 58ULL, kMaxU64
  };
  for (uint64_t p : primes) {
    const EngramPrimeMul mul = make_barrett_mul(p);
    const uint64_t edges[] = {
      0ULL, 1ULL, 2ULL,
      p > 1ULL ? p - 1ULL : 0ULL, p, p < kMaxU64 ? p + 1ULL : kMaxU64,
      p <= (kMaxU64 / 2ULL) ? (2ULL * p - 1ULL) : kMaxU64,
      p <= (kMaxU64 / 2ULL) ? (2ULL * p) : kMaxU64,
      (1ULL << 32) - 1ULL, 1ULL << 32,
      kMaxI64 - 1ULL, kMaxI64, kMaxI64 + 1ULL,
      kMaxU64 - 1ULL, kMaxU64
    };
    for (uint64_t x : edges) CHECK(barrett_mod(x, p, mul) == x % p);
    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ p;
    for (int i = 0; i < 128; ++i) {
      rng ^= rng >> 12;
      rng ^= rng << 25;
      rng ^= rng >> 27;
      const uint64_t x = rng * 0x2545F4914F6CDD1DULL;
      CHECK(barrett_mod(x, p, mul) == x % p);
    }
  }

  // 2. Write a synthetic v2 .engram + shard file so EngramConsts::load + parallel push +
  //    gather_mxfp8 (P=16 prefetch + AVX2 __m256i copy) are all exercised end-to-end.
  const std::string tag =
#ifdef __linux__
      std::to_string(static_cast<unsigned long>(::getpid()));
#else
      "test";
#endif
  const std::string dir = "/tmp";
  const std::string shard_name = "aff_w23_test_shard_" + tag + ".bin";
  const std::string shard_path = dir + "/" + shard_name;
  const std::string eng_path = dir + "/aff_w23_test_" + tag + ".engram";

  const uint32_t n_layer = 2u, max_ngram = 4u, n_heads = 8u, head_dim = 256u;
  const uint32_t n_cols = (max_ngram - 1u) * n_heads;  // 24
  const uint32_t vocab = 256u, cvocab = 128u, pad = 2u;
  const uint64_t row_count = 8192u;
  const uint64_t rows[2] = {row_count, row_count};
  const int64_t mult[8] = {
    1000003LL, 2000003LL, 3000017LL, 4000037LL,
    5000011LL, 6000011LL, 7000003LL, 8000009LL
  };
  int64_t primes_tbl[48];
  int64_t offsets_tbl[48];
  for (uint32_t l = 0; l < n_layer; ++l) {
    int64_t acc = 0;
    for (uint32_t c = 0; c < n_cols; ++c) {
      const int64_t p = 101LL + static_cast<int64_t>((l * n_cols + c) * 2u);
      primes_tbl[l * n_cols + c] = p;
      offsets_tbl[l * n_cols + c] = acc;
      acc += p;
    }
    CHECK(static_cast<uint64_t>(acc) <= row_count);
  }
  int32_t token_map[256];
  for (uint32_t i = 0; i < vocab; ++i) token_map[i] = static_cast<int32_t>((i * 7u + 3u) % cvocab);

  const uint64_t w_bytes = row_count * head_dim;
  const uint64_t s_bytes = row_count * (head_dim / 32u);
  std::vector<uint8_t> shard_w(w_bytes);
  std::vector<uint8_t> shard_s(s_bytes);
  for (size_t i = 0; i < w_bytes; ++i) shard_w[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFFu);
  for (size_t i = 0; i < s_bytes; ++i) shard_s[i] = static_cast<uint8_t>(120u + (i & 7u));
  // Deliberately place an unaligned prefix before layer 0 weights to exercise unaligned AVX2 loadu.
  const uint64_t prefix_pad = 13u;
  const uint64_t w_off0 = prefix_pad;
  const uint64_t s_off0 = w_off0 + w_bytes + 7u;
  const uint64_t w_off1 = s_off0 + s_bytes + 11u;
  const uint64_t s_off1 = w_off1 + w_bytes + 5u;
  {
    std::FILE* sf = std::fopen(shard_path.c_str(), "wb");
    CHECK(sf != nullptr);
    const std::vector<uint8_t> zeros(32, 0);
    CHECK(std::fwrite(zeros.data(), 1, prefix_pad, sf) == prefix_pad);
    CHECK(std::fwrite(shard_w.data(), 1, w_bytes, sf) == w_bytes);
    CHECK(std::fwrite(zeros.data(), 1, 7u, sf) == 7u);
    CHECK(std::fwrite(shard_s.data(), 1, s_bytes, sf) == s_bytes);
    CHECK(std::fwrite(zeros.data(), 1, 11u, sf) == 11u);
    CHECK(std::fwrite(shard_w.data(), 1, w_bytes, sf) == w_bytes);
    CHECK(std::fwrite(zeros.data(), 1, 5u, sf) == 5u);
    CHECK(std::fwrite(shard_s.data(), 1, s_bytes, sf) == s_bytes);
    std::fclose(sf);
  }
  {
    std::FILE* ef = std::fopen(eng_path.c_str(), "wb");
    CHECK(ef != nullptr);
    const char magic[8] = {'A', 'F', 'F', 'E', 'N', 'G', 'R', 'M'};
    const uint32_t hdr[9] = {2u, n_layer, max_ngram, n_heads, head_dim, n_cols, vocab, cvocab, pad};
    CHECK(std::fwrite(magic, 1, 8, ef) == 8);
    CHECK(std::fwrite(hdr, sizeof(uint32_t), 9, ef) == 9);
    CHECK(std::fwrite(rows, sizeof(uint64_t), 2, ef) == 2);
    CHECK(std::fwrite(mult, sizeof(int64_t), 8, ef) == 8);
    CHECK(std::fwrite(primes_tbl, sizeof(int64_t), 48, ef) == 48);
    CHECK(std::fwrite(offsets_tbl, sizeof(int64_t), 48, ef) == 48);
    CHECK(std::fwrite(token_map, sizeof(int32_t), 256, ef) == 256);
    const uint32_t slen = static_cast<uint32_t>(shard_name.size());
    const uint64_t q0[4] = {w_off0, s_off0, w_bytes, s_bytes};
    const uint64_t q1[4] = {w_off1, s_off1, w_bytes, s_bytes};
    CHECK(std::fwrite(&slen, sizeof(uint32_t), 1, ef) == 1);
    CHECK(std::fwrite(shard_name.data(), 1, slen, ef) == slen);
    CHECK(std::fwrite(q0, sizeof(uint64_t), 4, ef) == 4);
    CHECK(std::fwrite(&slen, sizeof(uint32_t), 1, ef) == 1);
    CHECK(std::fwrite(shard_name.data(), 1, slen, ef) == slen);
    CHECK(std::fwrite(q1, sizeof(uint64_t), 4, ef) == 4);
    std::fclose(ef);
  }

  EngramConsts ec;
  std::string err;
  CHECK(ec.load(eng_path, &err));
  CHECK(ec.prime_muls.size() == ec.primes.size());

  // 3. Compare batch push (n = 1024 >= 256 parallel path) vs single-token push + exact '%' ref.
  const uint32_t N = 1024u;
  std::vector<uint32_t> ids(N);
  std::vector<uint8_t> alive(N, 1u);
  for (uint32_t t = 0; t < N; ++t) {
    ids[t] = (t * 37u + 11u) % (vocab + 5u);
    if (t == 17u || t == 255u || t == 256u || t == 512u) alive[t] = 0u;
  }

  EngramHash h_batch, h_serial;
  h_batch.init(&ec, 16);
  h_serial.init(&ec, 16);
  std::vector<int64_t> out_batch((size_t)N * n_layer * n_cols, -1);
  std::vector<int64_t> out_serial((size_t)N * n_layer * n_cols, -2);
  h_batch.push(ids.data(), N, 0, out_batch.data(), alive.data());
  for (uint32_t t = 0; t < N; ++t) {
    h_serial.push(&ids[t], 1u, t, out_serial.data() + (size_t)t * n_layer * n_cols, &alive[t]);
  }
  CHECK(out_batch == out_serial);

  // Verify against exact scalar '%' reference at key positions across the parallel threshold.
  std::vector<int32_t> ref_cache(N, EngramConsts::kDead);
  for (uint32_t t = 0; t < N; ++t) {
    const int32_t m = ids[t] < vocab ? token_map[ids[t]] : 0;
    ref_cache[t] = alive[t] ? m : EngramConsts::kDead;
  }
  for (uint32_t t = 0; t < N; ++t) {
    int64_t tok[4];
    bool blocked = false;
    for (uint32_t s = 0; s < max_ngram; ++s) {
      const int64_t src_pos = (int64_t)t - (int64_t)s;
      const int32_t src = src_pos >= 0 ? ref_cache[(size_t)src_pos] : EngramConsts::kDead;
      if (src_pos < 0 || src == EngramConsts::kDead) blocked = true;
      tok[s] = blocked ? (int64_t)pad : (int64_t)src;
    }
    for (uint32_t l = 0; l < n_layer; ++l) {
      int64_t rolling = tok[0] * mult[l * max_ngram];
      const int64_t* got = out_batch.data() + ((size_t)t * n_layer + l) * n_cols;
      for (uint32_t i = 1; i < max_ngram; ++i) {
        rolling ^= tok[i] * mult[l * max_ngram + i];
        for (uint32_t h = 0; h < n_heads; ++h) {
          const uint32_t col = (i - 1u) * n_heads + h;
          const int64_t want = rolling % primes_tbl[l * n_cols + col] + offsets_tbl[l * n_cols + col];
          CHECK(got[col] == want);
        }
      }
    }
  }

  // 4. Verify gather_mxfp8 prefetch + AVX2 copy + out-of-range sentinel handling.
  EngramTables tables;
  CHECK(tables.open(&ec, dir, &err));
  CHECK(tables.has_mxfp8(0));
  const uint32_t stride = tables.raw_mxfp8_stride();
  CHECK(stride == n_cols * head_dim + n_cols * (head_dim / 32u));
  std::vector<int64_t> test_ids(out_batch.begin(), out_batch.begin() + n_cols);
  test_ids[0] = -1;                              // out of range -> 0x00 / 0xFF
  test_ids[n_cols - 1u] = static_cast<int64_t>(row_count);  // out of range
  std::vector<uint8_t> raw(stride + 13u, 0x5Au);
  // Write to an unaligned destination pointer (+5) to verify unaligned AVX2 storeu safety.
  uint8_t* dst = raw.data() + 5u;
  tables.gather_mxfp8(0, test_ids.data(), dst);
  const uint32_t nsc = head_dim / 32u;
  for (uint32_t c = 0; c < n_cols; ++c) {
    const uint8_t* wd = dst + (size_t)c * head_dim;
    const uint8_t* sd = dst + (size_t)n_cols * head_dim + (size_t)c * nsc;
    if (test_ids[c] < 0 || (uint64_t)test_ids[c] >= row_count) {
      for (uint32_t k = 0; k < head_dim; ++k) CHECK(wd[k] == 0x00u);
      for (uint32_t k = 0; k < nsc; ++k) CHECK(sd[k] == 0xFFu);
    } else {
      const size_t woff = (size_t)test_ids[c] * head_dim;
      const size_t soff = (size_t)test_ids[c] * nsc;
      CHECK(std::memcmp(wd, shard_w.data() + woff, head_dim) == 0);
      CHECK(std::memcmp(sd, shard_s.data() + soff, nsc) == 0);
    }
  }

  std::remove(eng_path.c_str());
  std::remove(shard_path.c_str());
  std::printf("PASS: test_engram_barrett_and_push_parity\n");
}

void test_expert_kernel_v2_parity() {
  const int saved_var = cpu_kernel_variant();
  if (active_cpu_isa() == CpuIsa::Avx512) {
    // Skylake-SP (AVX512F without VBMI) must default to dual-accumulator v2 (2), not v4/v1 fallback.
    CHECK(saved_var == 2);
  }

  const QuantSpec spec = expert_quant_spec(AFF_Q2P875);
  Codebook cb;
  cb.pair = 2u;
  cb.variants = 4u;
  cb.k = 32u;
  cb.v.resize((size_t)cb.variants * cb.k * cb.pair);
  for (uint32_t v = 0; v < cb.variants; ++v) {
    for (uint32_t c = 0; c < cb.k; ++c) {
      cb.v[((size_t)v * cb.k + c) * 2u + 0u] = (static_cast<float>(c) - 15.5f) * (0.05f + 0.01f * v);
      cb.v[((size_t)v * cb.k + c) * 2u + 1u] = (15.5f - static_cast<float>(c)) * (0.04f + 0.015f * v);
    }
  }

  for (uint64_t cols : {64ULL, 96ULL}) {
    const uint64_t rows = 16u;
    std::vector<float> W(rows * cols, 0.0f);
    for (uint64_t r = 0; r < rows; ++r) {
      for (uint64_t c = 0; c < cols; ++c) {
        // Leave row 3 blocks 2..3 zero so zero-scale fast-skip path is also exercised.
        if (r == 3u && c >= 32u && c < 64u) continue;
        const uint64_t idx = r * cols + c;
        W[idx] = std::sin(static_cast<float>(idx) * 0.37f) * 1.3f;
      }
    }
    QuantizedMatrix q;
    std::string err;
    CHECK(quantize_matrix(W.data(), rows, cols, spec, cb, &q, &err));
    const ExpertMatrixView view = view_of(q);

    std::vector<float> x(cols);
    for (uint64_t c = 0; c < cols; ++c) x[c] = std::cos(static_cast<float>(c) * 0.23f);

    std::vector<float> y_ref(rows, 0.25f), y_v1(rows, 0.25f), y_v2(rows, 0.25f), y_v4(rows, 0.25f);
    {
      std::vector<float> tmp(rows, 0.0f);
      expert_gemv_reference(view, cb, x.data(), tmp.data());
      for (uint64_t r = 0; r < rows; ++r) y_ref[r] += 0.75f * tmp[r];
    }

    cpu_kernel_variant(1);
    expert_gemv_scaled(view, cb, x.data(), 0.75f, y_v1.data());

    cpu_kernel_variant(2);
    expert_gemv_scaled(view, cb, x.data(), 0.75f, y_v2.data());

    cpu_kernel_variant(4);
    expert_gemv_scaled(view, cb, x.data(), 0.75f, y_v4.data());

    cpu_kernel_variant(saved_var);

    for (uint64_t r = 0; r < rows; ++r) {
      CHECK(std::fabs(y_v2[r] - y_ref[r]) < 1e-3f);
      CHECK(std::fabs(y_v2[r] - y_v1[r]) < 1e-4f);
      CHECK(std::fabs(y_v4[r] - y_v2[r]) < 1e-4f);
    }
  }
  std::printf("PASS: test_expert_kernel_v2_parity\n");
}

void test_pin_numa0_behavior() {
#ifdef __linux__
  const char* orig_env = std::getenv("AFF_PIN_NUMA0");
  const std::string saved_env = orig_env ? orig_env : "";
  const bool had_env = orig_env != nullptr;

  // Test AFF_PIN_NUMA0=0 disables pinning cleanly on a dedicated thread.
  ::setenv("AFF_PIN_NUMA0", "0", 1);
  {
    bool res = true;
    std::thread([&] { res = pin_thread_numa0(); }).join();
    CHECK(!res);
  }

  // Test empty intersection (if CPU 24 is online & allowed in current cpuset) no-ops gracefully.
  ::setenv("AFF_PIN_NUMA0", "1", 1);
  cpu_set_t cur;
  CPU_ZERO(&cur);
  if (sched_getaffinity(0, sizeof(cur), &cur) == 0 && CPU_ISSET(24, &cur)) {
    std::thread([&] {
      cpu_set_t s1, before, after;
      CPU_ZERO(&s1);
      CPU_SET(24, &s1);
      if (sched_setaffinity(0, sizeof(s1), &s1) == 0) {
        CPU_ZERO(&before);
        CPU_ZERO(&after);
        CHECK(sched_getaffinity(0, sizeof(before), &before) == 0);
        const bool pinned = pin_thread_numa0();
        CHECK(!pinned);
        CHECK(sched_getaffinity(0, sizeof(after), &after) == 0);
        CHECK(CPU_EQUAL(&before, &after));
      }
    }).join();
  }

  if (had_env) ::setenv("AFF_PIN_NUMA0", saved_env.c_str(), 1);
  else ::unsetenv("AFF_PIN_NUMA0");
#endif
  std::printf("PASS: test_pin_numa0_behavior\n");
}

void test_demand_append_dedup_and_clear() {
  ExpertHeat eh;
  HeatParams p;
  p.decay = 1.0f;
  p.hysteresis = 1.25f;
  p.min_gain = 1.5f;
  // 3 layers, 4 experts each -> 12 global indices.
  eh.init(/*n_layer=*/3, /*n_expert=*/4, p);

  // Layer 0: 3 in Vram (0,1,2), 1 in Pool (3).
  eh.set_tier(0, ExpertTier::Vram);
  eh.set_tier(1, ExpertTier::Vram);
  eh.set_tier(2, ExpertTier::Vram);
  eh.set_tier(3, ExpertTier::Pool);
  // Layer 1: 1 in Vram (4), 3 in Pool (5,6,7).
  eh.set_tier(4, ExpertTier::Vram);
  eh.set_tier(5, ExpertTier::Pool);
  eh.set_tier(6, ExpertTier::Pool);
  eh.set_tier(7, ExpertTier::Pool);
  // Layer 2: 2 in Vram (8,9), 2 in Pool (10,11).
  eh.set_tier(8, ExpertTier::Vram);
  eh.set_tier(9, ExpertTier::Vram);
  eh.set_tier(10, ExpertTier::Pool);
  eh.set_tier(11, ExpertTier::Pool);

  // Warm up pooled candidates with distinct credits:
  // idx 3 (cold miss: heat 1.0 < min_gain 1.5, rejected),
  // idx 5 (heat 2.0 >= 1.5, accepted),
  // idx 7 (heat 3.0 >= 1.5, accepted),
  // idx 11 (heat 4.0 >= 1.5, accepted).
  const uint32_t e0[] = {3};
  const uint32_t c0[] = {1};
  eh.observe(0, e0, 1, c0);
  const uint32_t e1[] = {1, 3};
  const uint32_t c1[] = {2, 3};
  eh.observe(1, e1, 2, c1);
  const uint32_t e2[] = {3};
  const uint32_t c2[] = {4};
  eh.observe(2, e2, 1, c2);

  // Multi-layer append with duplicates both within a call and across calls.
  const uint32_t m0[] = {3, 3};
  const uint32_t m1[] = {1, 3, 1};
  const uint32_t m1_dup[] = {1};
  const uint32_t m2[] = {3, 3};
  eh.clear_demand();
  eh.note_missing(0, m0, 2, /*append=*/true);
  eh.note_missing(1, m1, 3, /*append=*/true);
  eh.note_missing(1, m1_dup, 1, /*append=*/true);
  eh.note_missing(2, m2, 2, /*append=*/true);

  uint32_t prom = 0, dem = 0;
  // First pick skips idx 3 (heat 1.0 < 1.5), admits idx 5 (layer 1 expert 1),
  // demotes coldest from layer 0 (res_count 3 > layer 2's 2 > layer 1's 1).
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 5u);
  CHECK(dem < 4u);
  eh.set_busy(prom, true);
  eh.set_busy(dem, true);

  // Second pick continues through remaining queue: idx 7 (deduped from [1, 3, 1] + [1]).
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 7u);
  CHECK(dem < 4u);
  eh.set_busy(prom, true);
  eh.set_busy(dem, true);

  // Third pick returns idx 11 (deduped from [3, 3] on layer 2).
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 11u);
  CHECK(dem < 4u);
  eh.set_busy(prom, true);
  eh.set_busy(dem, true);

  // Queue is now empty because duplicates were filtered.
  CHECK(!eh.pick_demand(&prom, &dem));

  // Verify clear_demand() resets demand_seen_ and drops queued misses.
  eh.set_busy(5, false);
  eh.set_busy(7, false);
  eh.set_busy(11, false);
  eh.set_busy(0, false);
  eh.set_busy(1, false);
  eh.set_busy(2, false);
  eh.note_missing(1, m1, 3, /*append=*/true);
  eh.clear_demand();
  CHECK(!eh.pick_demand(&prom, &dem));
  eh.note_missing(1, m1, 3, /*append=*/true);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 5u);

  // Verify default append=false clears prior queued layers.
  eh.note_missing(2, m2, 2);
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == 11u);
  CHECK(!eh.pick_demand(&prom, &dem));

  // Verify pick_demand() and pick() choose the exact same victim (tie-break included) when
  // the hottest pooled expert is also at the head of demand_.
  const uint32_t d_hot[] = {3};
  eh.note_missing(2, d_hot, 1);  // idx 11 (heat 4.0, hottest overall)
  uint32_t p_pick = 0, d_pick = 0;
  CHECK(eh.pick(&p_pick, &d_pick));
  CHECK(eh.pick_demand(&prom, &dem));
  CHECK(prom == p_pick && dem == d_pick);

  std::printf("PASS: test_demand_append_dedup_and_clear\n");
}

}  // namespace

int main() {
  test_basic_tie_break_and_guards();
  test_max_block_credit();
  test_cooldown_steps();
  test_engram_barrett_and_push_parity();
  test_expert_kernel_v2_parity();
  test_pin_numa0_behavior();
  test_demand_append_dedup_and_clear();
  std::printf("PASS: all expert_heat tests passed.\n");
  return 0;
}
