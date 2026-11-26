#include "host_pool.h"

#include "ui/log.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>

#include <sys/mman.h>
#include <cstdlib>
#include <unistd.h>
#ifdef __linux__
#include <cerrno>
#include <dirent.h>
#include <sys/syscall.h>
#endif

namespace aff {
namespace {

#ifdef __linux__
bool read_sysfs_int(const char* path, int* out) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) return false;
  int v = 0;
  const bool ok = std::fscanf(f, "%d", &v) == 1;
  std::fclose(f);
  if (ok) *out = v;
  return ok;
}

bool read_sysfs_token(const char* path, char* buf, size_t n) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) return false;
  char fmt[16];
  std::snprintf(fmt, sizeof fmt, "%%%zus", n - 1u);
  const bool ok = std::fscanf(f, fmt, buf) == 1;
  std::fclose(f);
  return ok;
}

int parse_kfd_properties(const char* path, uint64_t* simd, uint64_t* cpu, int* render_minor) {
  std::FILE* f = std::fopen(path, "r");
  if (!f) return 0;
  char key[64];
  unsigned long long val = 0;
  int found = 0;
  *simd = 0;
  *cpu = 0;
  *render_minor = -1;
  while (std::fscanf(f, "%63s %llu", key, &val) == 2) {
    if (std::strcmp(key, "simd_count") == 0) {
      *simd = (uint64_t)val;
      found = 1;
    } else if (std::strcmp(key, "cpu_cores_count") == 0) {
      *cpu = (uint64_t)val;
      found = 1;
    } else if (std::strcmp(key, "drm_render_minor") == 0) {
      *render_minor = (int)val;
    }
  }
  std::fclose(f);
  return found;
}

// Auto-detect GPU NUMA node via KFD topology and DRM sysfs; clamps -1 -> 0.
int detect_gpu_numa_node(const char** source_out) {
  if (DIR* d = ::opendir("/sys/class/kfd/kfd/topology/nodes")) {
    int best_node = -1;
    while (struct dirent* ent = ::readdir(d)) {
      if (ent->d_name[0] == '.') continue;
      char prop_path[512];
      std::snprintf(prop_path, sizeof prop_path,
                    "/sys/class/kfd/kfd/topology/nodes/%s/properties", ent->d_name);
      uint64_t simd = 0, cpu = 0;
      int render_minor = -1;
      if (!parse_kfd_properties(prop_path, &simd, &cpu, &render_minor) || simd == 0) continue;
      if (render_minor > 0) {
        char drm_path[256];
        std::snprintf(drm_path, sizeof drm_path,
                      "/sys/class/drm/renderD%d/device/numa_node", render_minor);
        int numa = -1;
        if (read_sysfs_int(drm_path, &numa)) {
          ::closedir(d);
          if (source_out) *source_out = "sysfs/kfd";
          return numa < 0 ? 0 : numa;
        }
      }
      char links_dir[512];
      std::snprintf(links_dir, sizeof links_dir,
                    "/sys/class/kfd/kfd/topology/nodes/%s/io_links", ent->d_name);
      if (DIR* ld = ::opendir(links_dir)) {
        uint64_t best_w = UINT64_MAX;
        while (struct dirent* le = ::readdir(ld)) {
          if (le->d_name[0] == '.') continue;
          char lp[1024];
          std::snprintf(lp, sizeof lp, "%s/%s/properties", links_dir, le->d_name);
          std::FILE* lf = std::fopen(lp, "r");
          if (!lf) continue;
          char k[64];
          unsigned long long v = 0, node_to = 0, weight = UINT64_MAX;
          bool has_to = false;
          while (std::fscanf(lf, "%63s %llu", k, &v) == 2) {
            if (std::strcmp(k, "node_to") == 0) { node_to = v; has_to = true; }
            else if (std::strcmp(k, "weight") == 0) { weight = v; }
          }
          std::fclose(lf);
          if (!has_to || weight >= best_w) continue;
          char tp[256];
          std::snprintf(tp, sizeof tp, "/sys/class/kfd/kfd/topology/nodes/%llu/properties",
                        node_to);
          uint64_t t_simd = 0, t_cpu = 0;
          int t_rm = -1;
          if (parse_kfd_properties(tp, &t_simd, &t_cpu, &t_rm) && t_cpu > 0) {
            best_w = weight;
            best_node = (int)node_to;
          }
        }
        ::closedir(ld);
      }
      if (best_node >= 0) break;
    }
    ::closedir(d);
    if (best_node >= 0) {
      if (source_out) *source_out = "sysfs/kfd";
      return best_node;
    }
  }
  if (DIR* d = ::opendir("/sys/class/drm")) {
    int fallback_node = -1;
    while (struct dirent* ent = ::readdir(d)) {
      if (std::strncmp(ent->d_name, "card", 4) != 0) continue;
      if (std::strchr(ent->d_name, '-') != nullptr) continue;
      char vpath[512], npath[512], vendor[32] = {};
      std::snprintf(vpath, sizeof vpath, "/sys/class/drm/%s/device/vendor", ent->d_name);
      std::snprintf(npath, sizeof npath, "/sys/class/drm/%s/device/numa_node", ent->d_name);
      int numa = -1;
      if (!read_sysfs_int(npath, &numa)) continue;
      const int clamped = numa < 0 ? 0 : numa;
      if (read_sysfs_token(vpath, vendor, sizeof vendor) &&
          std::strcmp(vendor, "0x1002") == 0) {
        ::closedir(d);
        if (source_out) *source_out = "sysfs/drm";
        return clamped;
      }
      if (fallback_node < 0) fallback_node = clamped;
    }
    ::closedir(d);
    if (fallback_node >= 0) {
      if (source_out) *source_out = "sysfs/drm";
      return fallback_node;
    }
  }
  if (source_out) *source_out = "default";
  return 0;
}
#endif

// A named field of /proc/meminfo, in bytes. 0 when it cannot be read, which callers treat as
// "unknown, do not clamp" rather than "no memory".
uint64_t meminfo_bytes(const char* key) {
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  const size_t n = std::strlen(key);
  uint64_t kb = 0;
  while (std::fgets(line, sizeof line, f))
    if (!std::strncmp(line, key, n)) { kb = std::strtoull(line + n, nullptr, 10); break; }
  std::fclose(f);
  return kb * 1024ull;
}

// How much of the VMA starting at `base` the kernel actually gave us on 2 MiB pages. /proc/self/smaps
// is the only place that says: mincore reports residency, not page size, and the madvise return
// value is a hint's return value, not an outcome. Zero on any parse failure — this is a diagnostic,
// nothing depends on it.
uint64_t thp_bytes_of(const void* base) {
  if (!base) return 0;
  FILE* f = ::fopen("/proc/self/smaps", "r");
  if (!f) return 0;
  char line[512];
  bool in_vma = false;
  uint64_t kb = 0;
  while (::fgets(line, sizeof line, f)) {
    unsigned long long lo = 0, hi = 0;
    if (::sscanf(line, "%llx-%llx", &lo, &hi) == 2) {
      if (in_vma) break;                          // walked past ours without finding the field
      in_vma = (lo == (unsigned long long)(uintptr_t)base);
    } else if (in_vma && ::strncmp(line, "AnonHugePages:", 14) == 0) {
      ::sscanf(line + 14, "%llu", &lo);
      kb = lo;
      break;
    }
  }
  ::fclose(f);
  return kb << 10;
}

} // namespace

HostExpertPool::~HostExpertPool() {
  if (base_) ::munmap(base_, alloc_bytes_);
  base_ = nullptr;
}

bool HostExpertPool::init(const AffReader* aff, uint64_t budget_bytes, Wanted wanted,
                          void* wanted_ctx, uint32_t n_layer, uint32_t n_expert,
                          std::string* err, Transform tf, void* tf_ctx, uint64_t out_stride,
                          uint64_t n_stage, uint64_t reserve_bytes) {
  aff_ = aff;
  n_layer_ = n_layer;
  n_expert_ = n_expert;
  slot_.assign((size_t)n_layer * n_expert, -1);
  stride_ = aff->layer(0).stride;
  out_stride_ = tf ? (out_stride ? out_stride : stride_) : stride_;
  st_ = {};
  if (!stride_) { if (err) *err = "container reports a zero expert stride"; return false; }
  if (out_stride_ < 1) { if (err) *err = "transform declares a zero output stride"; return false; }

  // Which experts want a home here, hottest first. Ranking matters even though the pool normally
  // holds all of them: when it does not, the ones that spill to disk should be the ones a token is
  // least likely to ask for.
  std::vector<uint32_t> want;
  want.reserve((size_t)n_layer * n_expert);
  for (uint32_t l = 0; l < n_layer; ++l)
    for (uint32_t e = 0; e < n_expert; ++e)
      if (!wanted || wanted(wanted_ctx, l, e)) want.push_back(l * n_expert + e);
  if (want.empty()) { if (err) *err = "nothing to pool"; return false; }

  if (const float* prof = aff->profile())
    std::stable_sort(want.begin(), want.end(),
                     [&](uint32_t a, uint32_t b) { return prof[a] > prof[b]; });

  // ---- clamp to what the machine has, HOWEVER the budget was arrived at ------------------------
  //
  // `reserve_bytes` is what must stay free for everything the pool does not account for: the
  // container's residual mapping, the loader's staging, the driver's page tables for a registered
  // range of this size. The auto-sizer in the caller already subtracts it; this repeats the check
  // because an EXPLICIT --host-pool-mib does not go through the auto-sizer.
  //
  // The failure is not clean, which is why the check is worth duplicating. Past the available
  // memory the modes escalate rather than erroring: first a SIGSEGV on first touch, then
  // hipHostRegister refusing, then the machine ceasing to respond at all.
  //
  // Overcommit is the mechanism: the mmap succeeds and the shortfall arrives later as a fault on a
  // page nothing can back. MemAvailable is already net of this process's own RSS, so it is the only
  // term here — subtracting RSS as well double-counts and shrinks the pool for nothing.
  if (reserve_bytes) {
    const uint64_t avail = meminfo_bytes("MemAvailable:");
    const uint64_t ceiling = avail > reserve_bytes ? avail - reserve_bytes : 0;
    if (ceiling && budget_bytes > ceiling) {
      aff::ui::err("host pool: asked for %.1f GiB, capping at %.1f GiB — %.1f GiB available, "
                   "keeping %.1f GiB free. A pool with no headroom faults on first touch rather "
                   "than failing cleanly.\n",
                   (double)budget_bytes / (1 << 30), (double)ceiling / (1 << 30),
                   (double)avail / (1 << 30), (double)reserve_bytes / (1 << 30));
      budget_bytes = ceiling;
    }
  }

  // Staging slots come off the top of the budget and hold no expert permanently. They are what an
  // SSD-tier expert is read into on activation, and they are here rather than in a private buffer so
  // that the bytes land in registered memory a tile can point at — an SSD expert then costs the
  // dispatch exactly what a pooled one costs, plus the pread and the transform that filled it.
  //
  // Sized from the WORST LAYER's spill, which is the only quantity the ring has to cover: a slot is
  // released by the events of the layer that claimed it, so layers never contend, and what one
  // layer can need is bounded by how many of ITS experts are on disk.
  //
  // `n_stage` from the caller is the hard ceiling (every expert in a layer at once). Taking that
  // literally reserves a whole layer's worth of experts, which is itself pushed onto the tier the
  // ring exists to serve — most of the spill it was meant to be insurance against.
  //
  // Iterating "cover the whole spill" does not converge to anything useful either: each reserved
  // slot displaces a pooled expert and so creates spill, and the fixed point is always the ceiling.
  // Two passes over the per-layer histogram is what actually answers it.
  const uint64_t fit = budget_bytes / out_stride_;
  n_stage_ = 0;
  for (int pass = 0; pass < 2 && n_stage; ++pass) {
    const uint64_t pooled = std::min<uint64_t>(fit - n_stage_, want.size());
    std::vector<uint32_t> per_layer(n_layer, 0u);
    for (size_t i = pooled; i < want.size(); ++i) ++per_layer[want[i] / n_expert];
    uint32_t worst = 0;
    for (uint32_t v : per_layer) worst = std::max(worst, v);
    // A little headroom: the spill set shifts as the ring takes slots, and a layer that needs one
    // more than the ring holds declines the dispatch rather than running slowly.
    const uint64_t need = std::min<uint64_t>(n_stage, (uint64_t)worst + worst / 4 + 4);
    if (need <= n_stage_) break;
    n_stage_ = need;
  }
  if (n_stage_ > fit) n_stage_ = 0;
  const uint64_t n_pool = std::min<uint64_t>(fit - n_stage_, want.size());
  if (n_pool == 0) {
    if (err) *err = "budget holds no experts (" + std::to_string(budget_bytes >> 20) + " MiB, " +
                    std::to_string(out_stride_ >> 20) + " MiB each)";
    return false;
  }
  st_.ssd = want.size() - n_pool;
  // ---- SAY HOW MUCH SHORT, because "some experts are on disk" and "247 MiB would fix it" are the
  // difference between a warning and an action -----------------------------------------------------
  //
  // An addressless expert forces its whole LAYER back onto the host build, because no kernel can
  // read an expert off disk — and that build keeps a drain that pins the host back to the card. So
  // the tier's cost is not proportional to its size but to how many DISTINCT LAYERS it touches, and
  // a small spill spread thin is the worst shape it has. Hence the shortfall in bytes rather than
  // the expert count.
  //
  // `--host-pool-reserve-mib` and not `--host-pool-mib`, because the auto-sized case is the one that
  // lands here — a caller who named an explicit budget already knows what they asked for.
  if (st_.ssd) {
    const double short_mib = (double)(st_.ssd * out_stride_) / 1048576.0;
    // The suggestion only makes sense when the budget WAS auto-sized; an explicit --host-pool-mib
    // means the caller already named the number and the reserve is not in play.
    if (reserve_bytes)
      aff::ui::err("host pool: %llu of %zu experts did not fit and fall to the SSD tier — the pool "
                   "is %.0f MiB short. Auto-sized, so `--host-pool-reserve-mib %.0f` or lower would "
                   "hold them. Every layer holding one keeps the HOST dispatch and its drain, so "
                   "this costs lookahead as well as preads.\n",
                   (unsigned long long)st_.ssd, want.size(), short_mib,
                   std::max(0.0, (double)reserve_bytes / 1048576.0 - short_mib));
    else
      aff::ui::err("host pool: %llu of %zu experts did not fit and fall to the SSD tier — the pool "
                   "is %.0f MiB short of holding them. Every layer holding one keeps the HOST "
                   "dispatch and its drain, so this costs lookahead as well as preads.\n",
                   (unsigned long long)st_.ssd, want.size(), short_mib);
  }

  // Anonymous, private, and populated up front. MAP_POPULATE is what makes the load cost visible
  // here rather than as a fault storm on the first token; without it the first pass over the pool
  // pays the same per-page cost this class exists to remove.
  //
  // THE PAGE SIZE OF THIS POOL DOES NOT MATTER — see `thp_bytes` and the note on the madvise below.
  alloc_bytes_ = (n_pool + n_stage_) * out_stride_;
  stage0_ = n_pool;
  // amdnas-fixes thp-first: populate AFTER the hugepage hint so the pool is registered with the GPU on 2 MiB pages.
  // A 4 KiB-page mapping of tens of GiB costs the card ~8 B/page of VRAM page tables (taken after the slab was sized) and
  // khugepaged then collapses the pages under a live GPU mapping; both ended in stale/invalid GPU PTEs = garbage or a
  // permission fault from the expert GEMM. AFF_POOL_THP=0 restores the upstream order.
  const char* thp_env = std::getenv("AFF_POOL_THP");
  const bool thp_first = !(thp_env && thp_env[0] == '0');
  void* p = ::mmap(nullptr, alloc_bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | (thp_first ? 0 : MAP_POPULATE), -1, 0);
  if (p == MAP_FAILED) {
    // Not fatal: without a pool every non-resident expert simply falls to the SSD tier, which is
    // slower but correct. Saying so beats aborting a load that can still run.
    if (err) *err = "could not reserve " + std::to_string(alloc_bytes_ >> 20) + " MiB for the pool";
    alloc_bytes_ = 0;
    st_.ssd = want.size();
    return false;
  }
  base_ = (uint8_t*)p;
#ifdef __linux__
  {
    const char* numa_env = std::getenv("AFF_POOL_NUMA_NODE");
    const bool has_env = (numa_env && *numa_env);
    const char* numa_src = "AFF_POOL_NUMA_NODE";
    const int numa_node = has_env ? std::atoi(numa_env) : detect_gpu_numa_node(&numa_src);
    static bool numa_logged = false;
    if (numa_node >= 0 && numa_node < 63) {
#ifdef SYS_mbind
      unsigned long nodemask = 1ul << numa_node;
      const long rc = ::syscall(SYS_mbind, base_, alloc_bytes_, 1 /*MPOL_PREFERRED*/,
                                &nodemask, 64, 0);
      if (!numa_logged) {
        numa_logged = true;
        if (rc == 0) {
          aff::ui::err("host pool: NUMA policy MPOL_PREFERRED -> node %d (%s)\n",
                       numa_node, numa_src);
        } else {
          aff::ui::err("host pool: NUMA mbind(node %d, %s) skipped (%s)\n",
                       numa_node, numa_src, std::strerror(errno));
        }
      }
#endif
    } else if (!numa_logged) {
      numa_logged = true;
      aff::ui::err("host pool: NUMA binding disabled (%s=%d)\n", numa_src, numa_node);
    }
  }
#endif
  // Tens of GiB is a lot of 4 KiB pages to walk, so ask for 2 MiB ones. The hint lands AFTER
  // MAP_POPULATE, so coverage is opportunistic — most of the pool, with khugepaged collapsing some
  // of the rest later.
  //
  // Doing it the "correct" way round (madvise, then populate via the pread pass) reaches nearly full
  // coverage, moves decode by nothing, and multiplies load time, because with defrag=madvise every
  // fault in that VMA may enter direct compaction. The CPU kernel says the same from the other side:
  // the expert path is COMPUTE-bound, far from memory per core, so a TLB miss it never waits on
  // cannot cost anything. The hint stays only because it is free where it is.
  (void)::madvise(base_, alloc_bytes_, MADV_HUGEPAGE);
#ifdef __linux__
  const char* const mlock_env = std::getenv("AFF_POOL_MLOCK");
  const bool pool_mlock = mlock_env && *mlock_env && std::strcmp(mlock_env, "0") != 0;
  if (pool_mlock) {
#ifndef MLOCK_ONFAULT
#define MLOCK_ONFAULT 1u
#endif
    int ml_rc = -1;
#ifdef SYS_mlock2
    ml_rc = (int)::syscall(SYS_mlock2, base_, alloc_bytes_, MLOCK_ONFAULT);
#else
    ml_rc = ::mlock(base_, alloc_bytes_);
#endif
    if (ml_rc != 0) {
      aff::ui::err("host pool: mlock2(ONFAULT) failed for %.1f GiB (errno=%d: %s) — continuing unlocked\n",
                   (double)alloc_bytes_ / (1 << 30), errno, std::strerror(errno));
    }
  }
#endif
  if (thp_first) {
    const auto tp0 = std::chrono::steady_clock::now();
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    constexpr uint64_t kTouchStride = 2ull << 20;
    const uint64_t n_pages = (alloc_bytes_ + kTouchStride - 1ull) / kTouchStride;
    const uint32_t nt = (uint32_t)std::min<uint64_t>(4ull, std::max<uint64_t>(1ull, n_pages));
    auto worker = [&](uint32_t tid) {
      const uint64_t p0 = (n_pages * tid) / nt;
      const uint64_t p1 = (n_pages * (tid + 1u)) / nt;
      const uint64_t lo = p0 * kTouchStride;
      const uint64_t hi = std::min<uint64_t>(alloc_bytes_, p1 * kTouchStride);
      if (lo >= hi) return;
      if (::madvise(base_ + lo, hi - lo, MADV_POPULATE_WRITE) != 0) {
        volatile uint8_t* const vp = base_;
        for (uint64_t o = lo; o < hi; o += kTouchStride) vp[o] = 0;   // one touch a 2 MiB page
      }
    };
    if (nt <= 1u) {
      worker(0u);
    } else {
      std::thread thr[4];
      for (uint32_t t = 0; t < nt; ++t) thr[t] = std::thread(worker, t);
      for (uint32_t t = 0; t < nt; ++t) thr[t].join();
    }
    aff::ui::err("host pool: hugepage-first populate of %.1f GiB in %.1f s (amdnas-fixes thp-first; AFF_POOL_THP=0 to disable)\n",
                 (double)alloc_bytes_ / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count());
  }

  const auto t0 = std::chrono::steady_clock::now();
  {
    // A handful of readers is enough to saturate a sequential NVMe read; past that they only queue.
    // The copy is pread straight into the pool: no mapping, no fault chain, and the page cache never
    // has to hold a second copy of bytes the pool is about to own outright.
    const unsigned kReaders = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<uint64_t> cursor{0};
    std::vector<std::thread> th;
    for (unsigned t = 0; t < kReaders; ++t) {
      th.emplace_back([&] {
        // Only allocated when a transform is asked for. Without one the pread still lands straight
        // in the pool, which is what makes the untransformed path cost exactly what it used to.
        std::vector<uint8_t> stage(tf ? (size_t)stride_ : 0);
        for (;;) {
          const uint64_t k = cursor.fetch_add(1);
          if (k >= n_pool) break;
          const uint32_t idx = want[k];
          const uint64_t off = aff_->expert_file_offset(idx / n_expert_, idx % n_expert_);
          uint8_t* dst = base_ + k * out_stride_;
          uint8_t* land = tf ? stage.data() : dst;
          uint64_t got = 0;
          while (got < stride_) {
            const ssize_t r = ::pread(aff_->fd(), land + got, stride_ - got, (off_t)(off + got));
            if (r <= 0) break;
            got += (uint64_t)r;
          }
          if (got != stride_) continue;
          if (tf) tf(tf_ctx, idx / n_expert_, land, dst);
          slot_[idx] = (int64_t)k;
        }
      });
    }
    for (auto& x : th) x.join();
  }
  st_.load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  st_.load_gbps = st_.load_seconds > 0 ? (double)alloc_bytes_ / st_.load_seconds / 1e9 : 0.0;
  st_.pooled = 0;
  for (int64_t s : slot_) if (s >= 0) ++st_.pooled;
  st_.ssd = want.size() - st_.pooled;
  st_.bytes = alloc_bytes_;
  st_.thp_bytes = thp_bytes_of(base_);

  // The pool owns these bytes now. Drop the container's page-cache copy so the kernel is not holding
  // a duplicate of memory the process already has — that duplicate is precisely the pressure that
  // makes the mmap path collapse.
  for (uint32_t l = 0; l < n_layer_; ++l)
    for (uint32_t e = 0; e < n_expert_; ++e)
      if (slot_[(size_t)l * n_expert_ + e] >= 0) aff_->advise_expert(l, e, /*want=*/false);

  return true;
}

bool HostExpertPool::fetch(uint32_t layer, uint32_t expert, uint8_t* dst) const {
  if (!aff_ || !dst || !stride_) return false;
  const uint64_t off = aff_->expert_file_offset(layer, expert);
  uint64_t got = 0;
  while (got < stride_) {
    const ssize_t r = ::pread(aff_->fd(), dst + got, stride_ - got, (off_t)(off + got));
    if (r <= 0) return false;
    got += (uint64_t)r;
  }
  ssd_reads_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

} // namespace aff
