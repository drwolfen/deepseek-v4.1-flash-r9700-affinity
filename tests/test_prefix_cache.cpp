#include "engine/prefix_cache.h"
#include "server/server.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

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

  // Step 4: Index persistence test
  assert(cache.save_index());
  cache.close();

  // Reopen cache and load persisted index
  aff::PrefixCache cache2;
  opt.persistent = true;
  assert(cache2.open(opt, regions, io, fp, &err));
  assert(cache2.load_index());

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

}  // namespace

int main() {
  test_ignore_eos_parsing();
  test_prefix_cache_delta();
  std::printf("PASS: all prefix cache and ignore_eos tests passed.\n");
  return 0;
}
