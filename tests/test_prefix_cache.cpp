#include "engine/prefix_cache.h"
#include "server/server.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond)                                                               \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

namespace {

void rm_rf(const std::string& path) {
  DIR* d = opendir(path.c_str());
  if (!d) return;
  struct dirent* p;
  while ((p = readdir(d))) {
    if (std::strcmp(p->d_name, ".") == 0 || std::strcmp(p->d_name, "..") == 0) continue;
    std::string sub = path + "/" + p->d_name;
    struct stat st{};
    if (lstat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) rm_rf(sub);
    else ::unlink(sub.c_str());
  }
  closedir(d);
  ::rmdir(path.c_str());
}

struct SimContext {
  std::vector<uint8_t> ring;
  std::vector<uint8_t> app;
  uint64_t app_row_bytes = 1024;
  uint64_t ring_row_bytes = 4096;
};

bool sim_save(void* ctx, uint8_t part, uint32_t /*layer*/, uint32_t /*rank*/,
              uint64_t row0, uint64_t nrows, void* dst) {
  auto* c = static_cast<SimContext*>(ctx);
  if (part == 0) {
    if (row0 + nrows > 1) return false;
    std::memcpy(dst, c->ring.data() + row0 * c->ring_row_bytes, (size_t)(nrows * c->ring_row_bytes));
  } else {
    if ((row0 + nrows) * c->app_row_bytes > c->app.size()) return false;
    std::memcpy(dst, c->app.data() + row0 * c->app_row_bytes, (size_t)(nrows * c->app_row_bytes));
  }
  return true;
}

bool sim_load(void* ctx, uint8_t part, uint32_t /*layer*/, uint32_t /*rank*/,
              uint64_t row0, uint64_t nrows, const void* src) {
  auto* c = static_cast<SimContext*>(ctx);
  if (part == 0) {
    if (row0 + nrows > 1) return false;
    std::memcpy(c->ring.data() + row0 * c->ring_row_bytes, src, (size_t)(nrows * c->ring_row_bytes));
  } else {
    if ((row0 + nrows) * c->app_row_bytes > c->app.size()) return false;
    std::memcpy(c->app.data() + row0 * c->app_row_bytes, src, (size_t)(nrows * c->app_row_bytes));
  }
  return true;
}

void test_ignore_eos_parsing() {
  aff::CompletionRequest req;

  // 1. Chat endpoint with ignore_eos = true
  std::string body1 = R"({"model":"affinity","messages":[{"role":"user","content":"hi"}],"ignore_eos":true})";
  assert(aff::parse_completion_request(body1, true, &req));
  assert(req.ignore_eos == true);

  // 2. Chat endpoint with ignore_eos = false
  std::string body2 = R"({"model":"affinity","messages":[{"role":"user","content":"hi"}],"ignore_eos":false})";
  assert(aff::parse_completion_request(body2, true, &req));
  assert(req.ignore_eos == false);

  // 3. Raw completions endpoint with ignore_eos = true
  std::string body3 = R"({"model":"affinity","prompt":"Hello world","ignore_eos":true})";
  assert(aff::parse_completion_request(body3, false, &req));
  assert(req.ignore_eos == true);

  // 4. Raw completions endpoint without ignore_eos (defaults to false)
  std::string body4 = R"({"model":"affinity","prompt":"Hello world"})";
  req.ignore_eos = true;
  assert(aff::parse_completion_request(body4, false, &req));
  assert(req.ignore_eos == false);

  std::printf("PASS: test_ignore_eos_parsing\n");
}

void test_prefix_cache_delta() {
  char temp_template[] = "/tmp/aff_pcache_test_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  assert(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  SimContext sim;
  sim.ring.assign(4096, 0x5A);
  sim.app.resize(1024 * 1024, 0);
  for (size_t i = 0; i < sim.app.size(); ++i) sim.app[i] = (uint8_t)((i * 31 + 7) & 0xFF);

  std::vector<aff::KvRegion> regions = {
      {/*part=*/0, /*layer=*/0, /*row_bytes=*/4096, /*rows=*/1,    /*ratio=*/0},
      {/*part=*/1, /*layer=*/0, /*row_bytes=*/1024, /*rows=*/1024, /*ratio=*/1},
  };
  aff::KvIo io{&sim_save, &sim_load, &sim};

  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = 64;
  opt.budget_bytes = 128ull << 20; // 128 MiB
  opt.ckpt_percent = 25;
  opt.queue_depth = 2;
  opt.ranks = 1;
  opt.delta = true;
  opt.persistent = true;

  const uint64_t fp = 0x123456789ABCDEF0ull;
  std::string err;

  aff::PrefixCache cache;
  assert(cache.open(opt, regions, io, fp, &err));
  assert(cache.ready());
  assert(cache.delta_enabled());

  // Generate 256 token sequence
  std::vector<uint32_t> ids(256);
  for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = 1000 + i * 3;

  // Step 1: Publish at n = 74 (block 0 of 64 tokens + 10 tail tokens).
  // Base block 1 does not exist yet, so checkpoint is written as full.
  assert(cache.publish(ids.data(), 74, &err));
  cache.drain();
  assert(cache.stats().blocks_written == 1);
  assert(cache.stats().ckpts_written == 1);
  assert(cache.stats().delta_ckpts_written == 0);

  // Clear simulated device KV cache
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);

  uint32_t matched = 0;
  uint32_t hit = cache.lookup(ids.data(), 74, &matched);
  assert(hit == 74);
  assert(matched == 1);
  assert(cache.restore(ids.data(), 74, &err));
  assert(cache.stats().delta_ckpts_restored == 0);

  // Verify bit-exact restore of ring and app rows 0..73
  for (size_t i = 0; i < 4096; ++i) assert(sim.ring[i] == 0x5A);
  for (size_t i = 0; i < 74 * 1024; ++i) {
    uint8_t expected = (uint8_t)((i * 31 + 7) & 0xFF);
    assert(sim.app[i] == expected);
  }

  // Step 2: Now publish full block 1 (tokens 0..128).
  // This covers tokens 64..128. The checkpoint at 74 (tail=10) is now superseded and
  // marked state_only = true.
  assert(cache.publish(ids.data(), 128, &err));
  cache.drain();
  assert(cache.stats().blocks_written == 2);

  // Clear memory and restore at 74 again.
  // It should restore via state-only delta path using base block 1!
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  assert(cache.restore(ids.data(), 74, &err));
  assert(cache.stats().delta_ckpts_restored == 1);

  // Verify bit-exact restore
  for (size_t i = 0; i < 4096; ++i) assert(sim.ring[i] == 0x5A);
  for (size_t i = 0; i < 74 * 1024; ++i) {
    uint8_t expected = (uint8_t)((i * 31 + 7) & 0xFF);
    assert(sim.app[i] == expected);
  }

  // Step 3: Publish new checkpoint at 84 (tail=20) hanging off block 0.
  // Base block 1 ALREADY exists, so this checkpoint must be emitted as a state-only delta!
  assert(cache.publish(ids.data(), 84, &err));
  cache.drain();
  assert(cache.stats().delta_ckpts_written == 1);

  // Clear memory and restore at 84.
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  assert(cache.restore(ids.data(), 84, &err));
  assert(cache.stats().delta_ckpts_restored == 2);

  // Verify bit-exact restore of ring and app rows 0..83
  for (size_t i = 0; i < 4096; ++i) assert(sim.ring[i] == 0x5A);
  for (size_t i = 0; i < 84 * 1024; ++i) {
    uint8_t expected = (uint8_t)((i * 31 + 7) & 0xFF);
    assert(sim.app[i] == expected);
  }

  // Step 4: Index persistence test (close() persists index.bin; open() reloads it)
  cache.close();

  aff::PrefixCache cache2;
  CHECK(cache2.open(opt, regions, io, fp, &err));

  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  assert(cache2.restore(ids.data(), 84, &err));
  for (size_t i = 0; i < 4096; ++i) assert(sim.ring[i] == 0x5A);
  for (size_t i = 0; i < 84 * 1024; ++i) {
    uint8_t expected = (uint8_t)((i * 31 + 7) & 0xFF);
    assert(sim.app[i] == expected);
  }
  cache2.close();

  rm_rf(cache_dir);
  std::printf("PASS: test_prefix_cache_delta\n");
}

struct MultiRegionContext {
  struct RegBuf {
    uint8_t part;
    uint32_t layer;
    uint64_t row_bytes;
    uint64_t rows;
    uint32_t ratio;
    std::vector<uint8_t> data[2]; // per rank
  };
  std::vector<RegBuf> regs;

  RegBuf* find(uint8_t part, uint32_t layer) {
    for (auto& r : regs) {
      if (r.part == part && r.layer == layer) return &r;
    }
    return nullptr;
  }
};

bool multi_save(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                uint64_t row0, uint64_t nrows, void* dst) {
  auto* c = static_cast<MultiRegionContext*>(ctx);
  if (rank >= 2) return false;
  auto* r = c->find(part, layer);
  if (!r || row0 + nrows > r->rows) return false;
  std::memcpy(dst, r->data[rank].data() + row0 * r->row_bytes,
              (size_t)(nrows * r->row_bytes));
  return true;
}

bool multi_load(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                uint64_t row0, uint64_t nrows, const void* src) {
  auto* c = static_cast<MultiRegionContext*>(ctx);
  if (rank >= 2) return false;
  auto* r = c->find(part, layer);
  if (!r || row0 + nrows > r->rows) return false;
  std::memcpy(r->data[rank].data() + row0 * r->row_bytes, src,
              (size_t)(nrows * r->row_bytes));
  return true;
}

void test_prefix_cache_delta_multiregion_tp2() {
  char temp_template[] = "/tmp/aff_pcache_tp2_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  assert(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  const uint32_t B = 512;
  const uint32_t total_tokens = 2560; // > 2048

  // 2 rolling rings + 6 heterogeneous append-only regions (Comp, Idx8, IdxScale across 2 layers)
  MultiRegionContext sim;
  std::vector<aff::KvRegion> regions;
  auto add_reg = [&](uint8_t part, uint32_t layer, uint64_t row_bytes, uint64_t rows,
                     uint32_t ratio) {
    aff::KvRegion kr{part, layer, row_bytes, rows, ratio};
    regions.push_back(kr);
    MultiRegionContext::RegBuf rb;
    rb.part = part;
    rb.layer = layer;
    rb.row_bytes = row_bytes;
    rb.rows = rows;
    rb.ratio = ratio;
    for (int rk = 0; rk < 2; ++rk) {
      rb.data[rk].resize((size_t)(rows * row_bytes));
      for (size_t i = 0; i < rb.data[rk].size(); ++i) {
        // Replicated bit-identical across ranks (as required by KvIo invariant)
        rb.data[rk][i] = (uint8_t)(((i * 131 + part * 17 + layer * 53 + 19) ^ (i >> 7)) & 0xFF);
      }
    }
    sim.regs.push_back(std::move(rb));
  };

  add_reg(/*part=*/0, /*layer=*/0, 2048, 8, 0);   // Raw ring L0
  add_reg(/*part=*/4, /*layer=*/0, 512,  4, 0);   // Hist ring L0
  add_reg(/*part=*/1, /*layer=*/0, 1024, total_tokens,     1); // Comp L0 (ratio 1)
  add_reg(/*part=*/2, /*layer=*/0, 512,  total_tokens / 8, 8); // Idx8 L0 (ratio 8)
  add_reg(/*part=*/3, /*layer=*/0, 64,   total_tokens / 8, 8); // IdxScale L0 (ratio 8)
  add_reg(/*part=*/1, /*layer=*/1, 1024, total_tokens,     1); // Comp L1 (ratio 1)
  add_reg(/*part=*/2, /*layer=*/1, 512,  total_tokens / 8, 8); // Idx8 L1 (ratio 8)
  add_reg(/*part=*/3, /*layer=*/1, 64,   total_tokens / 8, 8); // IdxScale L1 (ratio 8)

  // Save golden snapshot of all regions & ranks
  std::vector<std::vector<uint8_t>> golden(sim.regs.size());
  for (size_t ri = 0; ri < sim.regs.size(); ++ri) golden[ri] = sim.regs[ri].data[0];

  aff::KvIo io{&multi_save, &multi_load, &sim};
  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = B;
  opt.budget_bytes = 256ull << 20;
  opt.ckpt_percent = 25;
  opt.queue_depth = 2;
  opt.ranks = 2; // Dual R9700 TP=2
  opt.delta = true;

  const uint64_t fp = 0xCAFEBABE12345678ull;
  std::string err;
  aff::PrefixCache cache;
  assert(cache.open(opt, regions, io, fp, &err));
  assert(cache.ready());
  assert(cache.delta_enabled());

  std::vector<uint32_t> ids(total_tokens);
  for (uint32_t i = 0; i < total_tokens; ++i) ids[i] = 5000 + i * 7;

  // Step 1: Publish pos = 1805 (3 * 512 + 269, tail = 269). Base block 4 (pos=2048) does not exist yet.
  const uint32_t pos1 = 1805;
  assert(cache.publish(ids.data(), pos1, &err));
  cache.drain();
  assert(cache.stats().blocks_written == 3);
  assert(cache.stats().delta_ckpts_written == 0);

  // Step 2: Cross 2048 (4 * 512) -> block 3 (1536..2048) is published, marking pos=1805 state_only=true.
  assert(cache.publish(ids.data(), 2048, &err));
  cache.drain();
  assert(cache.stats().blocks_written == 4);

  // Zero out BOTH ranks across ALL regions before restoring pos = 1805 via delta path
  for (auto& r : sim.regs) {
    for (int rk = 0; rk < 2; ++rk) std::fill(r.data[rk].begin(), r.data[rk].end(), 0);
  }

  uint32_t matched = 0;
  assert(cache.lookup(ids.data(), pos1, &matched) == pos1);
  assert(matched == 3);
  assert(cache.restore(ids.data(), pos1, &err));
  assert(cache.stats().delta_ckpts_restored == 1);

  // Verify 100% byte-for-byte equality across ALL regions and ALL ranks!
  for (size_t ri = 0; ri < sim.regs.size(); ++ri) {
    const auto& r = sim.regs[ri];
    const uint64_t expected_rows = r.ratio == 0 ? r.rows : (pos1 / r.ratio);
    const size_t expected_bytes = (size_t)(expected_rows * r.row_bytes);
    for (int rk = 0; rk < 2; ++rk) {
      assert(std::memcmp(r.data[rk].data(), golden[ri].data(), expected_bytes) == 0);
    }
  }

  // Step 3: Publish pos = 1850 (tail = 314) when base block 2048 already exists -> emits delta directly
  const uint32_t pos2 = 1850;
  // Restore golden data to sim before publishing pos2
  for (size_t ri = 0; ri < sim.regs.size(); ++ri) {
    sim.regs[ri].data[0] = golden[ri];
    sim.regs[ri].data[1] = golden[ri];
  }
  assert(cache.publish(ids.data(), pos2, &err));
  cache.drain();
  assert(cache.stats().delta_ckpts_written == 1);

  for (auto& r : sim.regs) {
    for (int rk = 0; rk < 2; ++rk) std::fill(r.data[rk].begin(), r.data[rk].end(), 0);
  }
  assert(cache.restore(ids.data(), pos2, &err));
  assert(cache.stats().delta_ckpts_restored == 2);
  for (size_t ri = 0; ri < sim.regs.size(); ++ri) {
    const auto& r = sim.regs[ri];
    const uint64_t expected_rows = r.ratio == 0 ? r.rows : (pos2 / r.ratio);
    const size_t expected_bytes = (size_t)(expected_rows * r.row_bytes);
    for (int rk = 0; rk < 2; ++rk) {
      assert(std::memcmp(r.data[rk].data(), golden[ri].data(), expected_bytes) == 0);
    }
  }

  cache.close();
  rm_rf(cache_dir);
  std::printf("PASS: test_prefix_cache_delta_multiregion_tp2\n");
}

void test_prefix_cache_cross_branch_divergence() {
  char temp_template[] = "/tmp/aff_pcache_div_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  CHECK(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  SimContext sim;
  sim.ring_row_bytes = 4096;
  sim.app_row_bytes = 1024;
  sim.ring.assign(4096, 0);
  sim.app.assign(256 * 1024, 0);

  std::vector<aff::KvRegion> regions = {
      {/*part=*/0, /*layer=*/0, /*row_bytes=*/4096, /*rows=*/1,   /*ratio=*/0},
      {/*part=*/1, /*layer=*/0, /*row_bytes=*/1024, /*rows=*/512, /*ratio=*/1},
  };
  aff::KvIo io{&sim_save, &sim_load, &sim};

  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = 64;
  // Tight budget (~8 blocks) so we can also test LRU eviction fallback cleanly
  opt.budget_bytes = 2ull << 20;
  opt.ckpt_percent = 25;
  opt.queue_depth = 2;
  opt.ranks = 1;
  opt.delta = true;

  const uint64_t fp = 0x733de762fee253e6ull;
  std::string err;
  aff::PrefixCache cache;
  CHECK(cache.open(opt, regions, io, fp, &err));
  const uint32_t n_blk_slots = cache.block_slots();

  // Branch A and Branch B share block 0 (0..63), then diverge at token 64.
  std::vector<uint32_t> branch_a(128), branch_b(128);
  for (uint32_t i = 0; i < 64; ++i) branch_a[i] = branch_b[i] = 100 + i;
  for (uint32_t i = 64; i < 128; ++i) {
    branch_a[i] = 10000 + i;
    branch_b[i] = 20000 + i;
  }

  // Fill sim with Branch A's state and publish checkpoint at pos = 74 (tail = 10).
  std::fill(sim.ring.begin(), sim.ring.end(), 0xA1);
  for (size_t i = 0; i < 74 * 1024; ++i) sim.app[i] = (uint8_t)((i + 0x11) & 0xFF);
  CHECK(cache.publish(branch_a.data(), 74, &err));
  cache.drain();

  // Branch B publishes full block 1 (0..128) with DIFFERENT tokens after shared block 0!
  std::fill(sim.ring.begin(), sim.ring.end(), 0xB2);
  for (size_t i = 0; i < 128 * 1024; ++i) sim.app[i] = (uint8_t)((i + 0x77) & 0xFF);
  CHECK(cache.publish(branch_b.data(), 128, &err));
  cache.drain();

  // Branch A's checkpoint at 74 must NOT have been marked state_only=true by Branch B,
  // and MUST restore Branch A's exact bytes (0xA1 ring, +0x11 app), NOT Branch B's!
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  CHECK(cache.lookup(branch_a.data(), 74, nullptr) == 74);
  CHECK(cache.restore(branch_a.data(), 74, &err));
  CHECK(cache.stats().delta_ckpts_restored == 0);
  for (size_t i = 0; i < 4096; ++i) CHECK(sim.ring[i] == 0xA1);
  for (size_t i = 0; i < 74 * 1024; ++i) CHECK(sim.app[i] == (uint8_t)((i + 0x11) & 0xFF));

  // Now publish Branch A's own block 1 (0..128), marking Branch A's checkpoint 74 state_only=true.
  for (size_t i = 64 * 1024; i < 128 * 1024; ++i) sim.app[i] = (uint8_t)((i + 0x11) & 0xFF);
  CHECK(cache.publish(branch_a.data(), 128, &err));
  cache.drain();

  // Evict Branch A's block 1 by filling the block pool with unrelated blocks (keep block 0 hot?
  // Actually evicting > n_blk_slots blocks drops block 1; then re-publish block 0 so lookup reaches
  // block 0 and finds checkpoint 74 whose base block 1 is gone!).
  std::vector<uint32_t> junk(64);
  for (uint32_t round = 0; round < n_blk_slots + 2; ++round) {
    for (uint32_t i = 0; i < 64; ++i) junk[i] = 900000 + round * 100 + i;
    CHECK(cache.publish(junk.data(), 64, &err));
  }
  cache.drain();
  // Re-publish Branch A's block 0 (only 64 tokens, so block 1 is NOT present!)
  std::fill(sim.ring.begin(), sim.ring.end(), 0xA1);
  for (size_t i = 0; i < 64 * 1024; ++i) sim.app[i] = (uint8_t)((i + 0x11) & 0xFF);
  CHECK(cache.publish(branch_a.data(), 64, &err));
  cache.drain();

  // Now restore Branch A at 74: checkpoint had state_only=true, base block 1 is gone,
  // and flags==0 on disk -> must fall back to full-checkpoint restore seamlessly!
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  if (cache.lookup(branch_a.data(), 74, nullptr) == 74) {
    CHECK(cache.restore(branch_a.data(), 74, &err));
    for (size_t i = 0; i < 4096; ++i) CHECK(sim.ring[i] == 0xA1);
    for (size_t i = 0; i < 74 * 1024; ++i) CHECK(sim.app[i] == (uint8_t)((i + 0x11) & 0xFF));
  }

  cache.close();
  rm_rf(cache_dir);
  std::printf("PASS: test_prefix_cache_cross_branch_divergence\n");
}

void test_prefix_cache_large_block_and_atomic_index() {
  char temp_template[] = "/tmp/aff_pcache_big_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  CHECK(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  const uint32_t B = 1536; // > 1000 tokens -> header > 4096 bytes
  const uint32_t total = 3072;
  SimContext sim;
  sim.ring_row_bytes = 4096;
  sim.app_row_bytes = 512;
  sim.ring.assign(4096, 0x3C);
  sim.app.resize((size_t)total * 512);
  for (size_t i = 0; i < sim.app.size(); ++i) sim.app[i] = (uint8_t)((i * 17 + 3) & 0xFF);

  std::vector<aff::KvRegion> regions = {
      {/*part=*/0, /*layer=*/0, /*row_bytes=*/4096, /*rows=*/1,     /*ratio=*/0},
      {/*part=*/1, /*layer=*/0, /*row_bytes=*/512,  /*rows=*/total, /*ratio=*/1},
  };
  aff::KvIo io{&sim_save, &sim_load, &sim};

  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = B;
  opt.budget_bytes = 64ull << 20;
  opt.ckpt_percent = 25;
  opt.queue_depth = 2;
  opt.ranks = 1;
  opt.delta = true;
  opt.persistent = true;

  // Live default V4.1 fingerprint compatibility check
  const uint64_t fp = 0x733de762fee253e6ull;
  std::string err;
  aff::PrefixCache cache;
  CHECK(cache.open(opt, regions, io, fp, &err));

  std::vector<uint32_t> ids(total);
  for (uint32_t i = 0; i < total; ++i) ids[i] = 42000 + i;

  // Publish block 1 (0..3072) FIRST (do NOT drain manually: publish's find_base_block must handle it)
  CHECK(cache.publish(ids.data(), 3072, &err));
  // Immediately publish delta checkpoint at 1536 + 1200 (> 1000 tokens tail!)
  const uint32_t pos = B + 1200;
  CHECK(cache.publish(ids.data(), pos, &err));
  CHECK(cache.stats().delta_ckpts_written == 1);
  cache.drain();

  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  CHECK(cache.lookup(ids.data(), pos, nullptr) == pos);
  CHECK(cache.restore(ids.data(), pos, &err));
  CHECK(cache.stats().delta_ckpts_restored == 1);
  for (size_t i = 0; i < 4096; ++i) CHECK(sim.ring[i] == 0x3C);
  for (size_t i = 0; i < (size_t)pos * 512; ++i) CHECK(sim.app[i] == (uint8_t)((i * 17 + 3) & 0xFF));

  // Atomic save_index on close(): index.bin exists, index.bin.tmp does NOT exist
  cache.close();
  struct stat st{};
  CHECK(::stat((cache_dir + "/index.bin").c_str(), &st) == 0);
  CHECK(::stat((cache_dir + "/index.bin.tmp").c_str(), &st) != 0);

  // Reopen and verify persistent reload with same fp
  aff::PrefixCache cache2;
  CHECK(cache2.open(opt, regions, io, fp, &err));
  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  CHECK(cache2.lookup(ids.data(), pos, nullptr) == pos);
  CHECK(cache2.restore(ids.data(), pos, &err));
  for (size_t i = 0; i < 4096; ++i) CHECK(sim.ring[i] == 0x3C);
  for (size_t i = 0; i < (size_t)pos * 512; ++i) CHECK(sim.app[i] == (uint8_t)((i * 17 + 3) & 0xFF));
  cache2.close();

  rm_rf(cache_dir);
  std::printf("PASS: test_prefix_cache_large_block_and_atomic_index\n");
}

struct AsyncMultiSim {
  MultiRegionContext base;
  struct PendingOp {
    bool is_save = false;
    uint8_t part = 0;
    uint32_t layer = 0;
    uint32_t rank = 0;
    uint64_t row0 = 0;
    uint64_t nrows = 0;
    void* dst = nullptr;
    const void* src = nullptr;
  };
  std::vector<PendingOp> pending;
  std::set<void*> pinned;
  bool pin_should_fail = false;
  uint64_t pin_calls = 0;
  uint64_t unpin_calls = 0;
  uint64_t sync_all_calls = 0;
  uint64_t sync_io_calls = 0;
  uint64_t async_save_calls = 0;
  uint64_t async_load_calls = 0;
};

bool async_sim_save_sync(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                         uint64_t row0, uint64_t nrows, void* dst) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->sync_io_calls;
  return multi_save(&s->base, part, layer, rank, row0, nrows, dst);
}

bool async_sim_load_sync(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                         uint64_t row0, uint64_t nrows, const void* src) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->sync_io_calls;
  return multi_load(&s->base, part, layer, rank, row0, nrows, src);
}

bool async_sim_save_async(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                          uint64_t row0, uint64_t nrows, void* dst) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->async_save_calls;
  s->pending.push_back({true, part, layer, rank, row0, nrows, dst, nullptr});
  return true;
}

bool async_sim_load_async(void* ctx, uint8_t part, uint32_t layer, uint32_t rank,
                          uint64_t row0, uint64_t nrows, const void* src) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->async_load_calls;
  s->pending.push_back({false, part, layer, rank, row0, nrows, nullptr, src});
  return true;
}

bool async_sim_sync_all(void* ctx) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->sync_all_calls;
  // Executing deferred copies inside sync_all proves host staging slices do not alias before sync.
  for (const auto& op : s->pending) {
    if (op.is_save) {
      CHECK(multi_save(&s->base, op.part, op.layer, op.rank, op.row0, op.nrows, op.dst));
    } else {
      CHECK(multi_load(&s->base, op.part, op.layer, op.rank, op.row0, op.nrows, op.src));
    }
  }
  s->pending.clear();
  return true;
}

bool async_sim_pin(void* ctx, void* ptr, uint64_t bytes) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->pin_calls;
  if (s->pin_should_fail || !ptr || !bytes) return false;
  CHECK(s->pinned.insert(ptr).second);
  return true;
}

void async_sim_unpin(void* ctx, void* ptr) {
  auto* s = static_cast<AsyncMultiSim*>(ctx);
  ++s->unpin_calls;
  CHECK(s->pinned.erase(ptr) == 1);
}

void test_prefix_cache_batched_pinned_io() {
  char temp_template[] = "/tmp/aff_pcache_batch_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  CHECK(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  const uint32_t B = 256;
  const uint32_t total_tokens = 1280; // 5 blocks

  auto make_sim = [&](bool pin_fail) {
    AsyncMultiSim sim;
    sim.pin_should_fail = pin_fail;
    std::vector<aff::KvRegion> regions;
    auto add_reg = [&](uint8_t part, uint32_t layer, uint64_t row_bytes, uint64_t rows,
                       uint32_t ratio) {
      regions.push_back(aff::KvRegion{part, layer, row_bytes, rows, ratio});
      MultiRegionContext::RegBuf rb;
      rb.part = part; rb.layer = layer; rb.row_bytes = row_bytes; rb.rows = rows; rb.ratio = ratio;
      for (int rk = 0; rk < 2; ++rk) {
        rb.data[rk].resize((size_t)(rows * row_bytes));
        for (size_t i = 0; i < rb.data[rk].size(); ++i)
          rb.data[rk][i] = (uint8_t)(((i * 97 + part * 31 + layer * 11 + 5) ^ (i >> 5)) & 0xFF);
      }
      sim.base.regs.push_back(std::move(rb));
    };
    add_reg(0, 0, 2048, 8, 0);
    add_reg(4, 0, 512, 4, 0);
    add_reg(1, 0, 512, total_tokens, 1);
    add_reg(2, 0, 256, total_tokens / 8, 8);
    add_reg(1, 1, 512, total_tokens, 1);
    add_reg(3, 1, 64, total_tokens / 8, 8);
    return std::make_pair(std::move(sim), std::move(regions));
  };

  auto [sim, regions] = make_sim(false);
  std::vector<std::vector<uint8_t>> golden(sim.base.regs.size());
  for (size_t i = 0; i < sim.base.regs.size(); ++i) golden[i] = sim.base.regs[i].data[0];

  aff::KvIo io{&async_sim_save_sync, &async_sim_load_sync, &sim,
               &async_sim_save_async, &async_sim_load_async, &async_sim_sync_all,
               &async_sim_pin, &async_sim_unpin};

  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = B;
  opt.budget_bytes = 64ull << 20;
  opt.ckpt_percent = 25;
  opt.queue_depth = 3;
  opt.ranks = 2;
  opt.delta = true;

  std::string err;
  aff::PrefixCache cache;
  CHECK(cache.open(opt, regions, io, 0x733de762fee253e6ull, &err));
  // Expected pins: stage[0..2] (3) + gather[0..1] (2) + ckbuf (1) + rbuf[0] (1) = 7
  CHECK(sim.pin_calls == 7);
  CHECK(sim.pinned.size() == 7);

  std::vector<uint32_t> ids(total_tokens);
  for (uint32_t i = 0; i < total_tokens; ++i) ids[i] = 7000 + i * 5;

  // Publish 3 full blocks + 100 tail tokens (868)
  const uint32_t pos1 = 3 * B + 100;
  CHECK(cache.publish(ids.data(), pos1, &err));
  cache.drain();
  CHECK(sim.sync_io_calls == 0);
  CHECK(sim.async_save_calls > 0);
  // 3 blocks + 1 checkpoint = 4 sync_all calls
  CHECK(sim.sync_all_calls == 4);

  // Zero both ranks and restore all 3 blocks (g=3 > 1) + checkpoint in batched async mode.
  for (auto& r : sim.base.regs)
    for (int rk = 0; rk < 2; ++rk) std::fill(r.data[rk].begin(), r.data[rk].end(), 0);
  sim.sync_all_calls = 0;
  sim.async_load_calls = 0;

  CHECK(cache.lookup(ids.data(), pos1, nullptr) == pos1);
  CHECK(cache.restore(ids.data(), pos1, &err));
  CHECK(sim.sync_io_calls == 0);
  CHECK(sim.async_load_calls > 0);
  // 1 sync_all for the 3-block batch across all regions/ranks + 1 sync_all for checkpoint (roll+tail)
  CHECK(sim.sync_all_calls == 2);

  for (size_t ri = 0; ri < sim.base.regs.size(); ++ri) {
    const auto& r = sim.base.regs[ri];
    const uint64_t expected_rows = r.ratio == 0 ? r.rows : (pos1 / r.ratio);
    const size_t expected_bytes = (size_t)(expected_rows * r.row_bytes);
    for (int rk = 0; rk < 2; ++rk)
      CHECK(std::memcmp(r.data[rk].data(), golden[ri].data(), expected_bytes) == 0);
  }

  // Now restore golden state and publish block 3 (1024) so pos1 checkpoint becomes delta-backed.
  for (size_t ri = 0; ri < sim.base.regs.size(); ++ri)
    sim.base.regs[ri].data[0] = sim.base.regs[ri].data[1] = golden[ri];
  CHECK(cache.publish(ids.data(), 4 * B, &err));
  cache.drain();

  for (auto& r : sim.base.regs)
    for (int rk = 0; rk < 2; ++rk) std::fill(r.data[rk].begin(), r.data[rk].end(), 0);
  sim.sync_all_calls = 0;
  CHECK(cache.restore(ids.data(), pos1, &err));
  CHECK(cache.stats().delta_ckpts_restored == 1);
  CHECK(sim.sync_all_calls == 2);
  for (size_t ri = 0; ri < sim.base.regs.size(); ++ri) {
    const auto& r = sim.base.regs[ri];
    const uint64_t expected_rows = r.ratio == 0 ? r.rows : (pos1 / r.ratio);
    const size_t expected_bytes = (size_t)(expected_rows * r.row_bytes);
    for (int rk = 0; rk < 2; ++rk)
      CHECK(std::memcmp(r.data[rk].data(), golden[ri].data(), expected_bytes) == 0);
  }

  cache.close();
  CHECK(sim.unpin_calls == 7);
  CHECK(sim.pinned.empty());
  rm_rf(cache_dir);

  // Pinning failure fallback: pin_host returns false -> pageable fallback with zero unpin calls.
  char temp_fb[] = "/tmp/aff_pcache_pinfb_XXXXXX";
  char* dir_fb = ::mkdtemp(temp_fb);
  CHECK(dir_fb != nullptr);
  std::string fb_dir(dir_fb);
  auto [sim_fb, regs_fb] = make_sim(true);
  aff::KvIo io_fb{&async_sim_save_sync, &async_sim_load_sync, &sim_fb,
                  &async_sim_save_async, &async_sim_load_async, &async_sim_sync_all,
                  &async_sim_pin, &async_sim_unpin};
  opt.dir = fb_dir;
  aff::PrefixCache cache_fb;
  CHECK(cache_fb.open(opt, regs_fb, io_fb, 0x733de762fee253e6ull, &err));
  CHECK(sim_fb.pin_calls == 7);
  CHECK(sim_fb.pinned.empty());
  CHECK(cache_fb.publish(ids.data(), pos1, &err));
  cache_fb.drain();
  for (auto& r : sim_fb.base.regs)
    for (int rk = 0; rk < 2; ++rk) std::fill(r.data[rk].begin(), r.data[rk].end(), 0);
  CHECK(cache_fb.restore(ids.data(), pos1, &err));
  for (size_t ri = 0; ri < sim_fb.base.regs.size(); ++ri) {
    const auto& r = sim_fb.base.regs[ri];
    const uint64_t expected_rows = r.ratio == 0 ? r.rows : (pos1 / r.ratio);
    const size_t expected_bytes = (size_t)(expected_rows * r.row_bytes);
    for (int rk = 0; rk < 2; ++rk)
      CHECK(std::memcmp(r.data[rk].data(), golden[ri].data(), expected_bytes) == 0);
  }
  cache_fb.close();
  CHECK(sim_fb.unpin_calls == 0);
  rm_rf(fb_dir);

  std::printf("PASS: test_prefix_cache_batched_pinned_io\n");
}

void test_prefix_cache_think_ckpt_simulation() {
  char temp_template[] = "/tmp/aff_pcache_think_XXXXXX";
  char* temp_dir = ::mkdtemp(temp_template);
  CHECK(temp_dir != nullptr);
  std::string cache_dir(temp_dir);

  const uint32_t B = 512;
  const uint32_t pos_think = 2 * B + 170; // 1194 tokens (at </think>)
  const uint32_t turn_t_end = pos_think + 300;
  const uint32_t turn_t1_len = pos_think + 406;

  SimContext sim;
  sim.ring_row_bytes = 4096;
  sim.app_row_bytes = 256;
  sim.ring.assign(4096, 0x5E);
  sim.app.resize(2048 * 256);
  for (size_t i = 0; i < sim.app.size(); ++i) sim.app[i] = (uint8_t)((i * 29 + 11) & 0xFF);

  std::vector<aff::KvRegion> regions = {
      {0, 0, 4096, 1, 0},
      {1, 0, 256, 2048, 1},
  };
  aff::KvIo io{&sim_save, &sim_load, &sim};

  aff::PrefixCache::Options opt;
  opt.dir = cache_dir;
  opt.block_tokens = B;
  opt.budget_bytes = 64ull << 20;
  opt.ckpt_percent = 25;
  opt.queue_depth = 2;
  opt.ranks = 1;
  opt.delta = true;

  std::string err;
  aff::PrefixCache cache;
  CHECK(cache.open(opt, regions, io, 0x733de762fee253e6ull, &err));

  std::vector<uint32_t> ids_turn_t(turn_t_end);
  for (uint32_t i = 0; i < turn_t_end; ++i) ids_turn_t[i] = 10000 + i;

  // Turn T publishes think-close checkpoint at pos_think, then full turn end (after tool call).
  CHECK(cache.publish(ids_turn_t.data(), pos_think, &err));
  CHECK(cache.publish(ids_turn_t.data(), turn_t_end, &err));
  cache.drain();

  // Turn T+1 shares [0 .. pos_think) identically, then diverges right after pos_think (e.g.
  // alphabetical JSON tool-call key reordering by client).
  std::vector<uint32_t> ids_turn_t1(turn_t1_len);
  for (uint32_t i = 0; i < pos_think; ++i) ids_turn_t1[i] = ids_turn_t[i];
  for (uint32_t i = pos_think; i < turn_t1_len; ++i) ids_turn_t1[i] = 90000 + i;

  uint32_t matched = 0;
  const uint32_t hit = cache.lookup(ids_turn_t1.data(), turn_t1_len, &matched);
  CHECK(hit == pos_think);
  CHECK(matched == 2);

  std::fill(sim.ring.begin(), sim.ring.end(), 0);
  std::fill(sim.app.begin(), sim.app.end(), 0);
  CHECK(cache.restore(ids_turn_t1.data(), hit, &err));
  for (size_t i = 0; i < 4096; ++i) CHECK(sim.ring[i] == 0x5E);
  for (size_t i = 0; i < (size_t)pos_think * 256; ++i)
    CHECK(sim.app[i] == (uint8_t)((i * 29 + 11) & 0xFF));

  cache.close();
  rm_rf(cache_dir);
  std::printf("PASS: test_prefix_cache_think_ckpt_simulation\n");
}

}  // namespace

int main() {
  test_ignore_eos_parsing();
  test_prefix_cache_delta();
  test_prefix_cache_delta_multiregion_tp2();
  test_prefix_cache_cross_branch_divergence();
  test_prefix_cache_large_block_and_atomic_index();
  test_prefix_cache_batched_pinned_io();
  test_prefix_cache_think_ckpt_simulation();
  std::printf("PASS: all prefix cache and ignore_eos tests passed.\n");
  return 0;
}
