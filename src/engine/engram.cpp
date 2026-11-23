#include "engine/engram.h"

#include "engine/threadpool.h"
#include "quant/source_dtypes.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace aff {
namespace {

constexpr char kMagic[8] = {'A', 'F', 'F', 'E', 'N', 'G', 'R', 'M'};

template <typename T>
bool rd(FILE* f, T* dst, size_t n) {
  return std::fread(dst, sizeof(T), n, f) == n;
}

}  // namespace

bool EngramConsts::load(const std::string& path, std::string* err) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { if (err) *err = "cannot open " + path; return false; }
  auto fail = [&](const char* m) { if (err) *err = path + ": " + m; std::fclose(f); return false; };

  char magic[8];
  if (!rd(f, magic, 8) || std::memcmp(magic, kMagic, 8) != 0) return fail("not an engram file");
  uint32_t h[9];
  if (!rd(f, h, 9)) return fail("short header");
  if (h[0] != 1u && h[0] != 2u) return fail("unsupported version");
  n_layer = h[1]; max_ngram = h[2]; n_heads = h[3]; head_dim = h[4];
  n_cols = h[5]; vocab = h[6]; cvocab = h[7]; pad = h[8];
  // The shapes are load-bearing arithmetic below, not decoration: n_cols indexes the offsets and
  // max_ngram bounds the look-back, so a file that disagrees with itself would read past its own
  // tables rather than hash differently.
  if (!n_layer || max_ngram < 2u || !n_heads || !head_dim || !vocab) return fail("degenerate header");
  if (n_cols != (max_ngram - 1u) * n_heads) return fail("n_cols disagrees with max_ngram * n_heads");

  rows.resize(n_layer);
  mult.resize((size_t)n_layer * max_ngram);
  primes.resize((size_t)n_layer * (max_ngram - 1u) * n_heads);
  offsets.resize((size_t)n_layer * n_cols);
  token_map.resize(vocab);
  if (!rd(f, rows.data(), rows.size())) return fail("short rows");
  if (!rd(f, mult.data(), mult.size())) return fail("short multipliers");
  if (!rd(f, primes.data(), primes.size())) return fail("short primes");
  if (!rd(f, offsets.data(), offsets.size())) return fail("short offsets");
  if (!rd(f, token_map.data(), token_map.size())) return fail("short token map");
  // Version 2 and later carry where each layer's table lives. Version 1 files hash correctly and
  // simply cannot look anything up, which is what `tables.empty()` means to the caller.
  if (h[0] >= 2u) {
    tables.resize(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
      uint32_t len = 0;
      if (!rd(f, &len, 1) || !len || len > 4096u) return fail("bad shard name");
      std::string name(len, '\0');
      if (!rd(f, &name[0], len)) return fail("short shard name");
      uint64_t q[4];
      if (!rd(f, q, 4)) return fail("short table offsets");
      tables[l] = {name, q[0], q[1], q[2], q[3]};
      // The ranges have to describe the table this file already declared, or a lookup indexes one
      // geometry through another's stride.
      if (q[2] != rows[l] * head_dim) return fail("table weight range disagrees with rows x dim");
      if (q[3] != rows[l] * (head_dim / 32u)) return fail("table scale range disagrees");
    }
  }
  std::fclose(f);

  if (pad >= cvocab) return fail("pad id outside the compressed vocab");
  for (uint32_t i = 0; i < vocab; ++i)
    if (token_map[i] < 0 || (uint32_t)token_map[i] >= cvocab) return fail("token map out of range");
  // Every bucket range must fit inside its layer's table, or a lookup walks off the mapping. The
  // last range ends at its own offset plus its own modulus.
  for (uint32_t l = 0; l < n_layer; ++l) {
    const int64_t end = offsets[(size_t)l * n_cols + n_cols - 1u] +
                        prime(l, max_ngram - 2u, n_heads - 1u);
    if ((uint64_t)end > rows[l]) return fail("bucket ranges overflow the table");
  }
  std::string mul_err;
  if (!build_prime_muls(&mul_err)) return fail(mul_err.c_str());
  return true;
}

bool EngramConsts::build_prime_muls(std::string* err) {
  prime_muls.resize(primes.size());
  for (size_t k = 0; k < primes.size(); ++k) {
    if (primes[k] <= 0) {
      if (err) *err = "non-positive prime";
      return false;
    }
    const uint64_t p = static_cast<uint64_t>(primes[k]);
    const EngramPrimeMul mul = make_barrett_mul(p);
    prime_muls[k] = mul;

    constexpr uint64_t kMaxU64 = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t kMaxI64 = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t edges[] = {
      0ULL,
      1ULL,
      p > 1ULL ? p - 1ULL : 0ULL,
      p,
      p < kMaxU64 ? p + 1ULL : kMaxU64,
      p <= (kMaxU64 / 2ULL) ? (2ULL * p - 1ULL) : kMaxU64,
      p <= (kMaxU64 / 2ULL) ? (2ULL * p) : kMaxU64,
      kMaxI64 - 1ULL,
      kMaxI64,
      kMaxI64 + 1ULL,
      kMaxU64 - 1ULL,
      kMaxU64,
    };
    for (uint64_t x : edges) {
      const uint64_t got = barrett_mod(x, p, mul);
      const uint64_t want = x % p;
      assert(got == want);
      if (got != want) {
        if (err) *err = "barrett_mod parity mismatch";
        return false;
      }
    }
  }
  return true;
}

void EngramHash::init(const EngramConsts* c, uint64_t hint) {
  c_ = c;
  // A hint, not a bound: the cache grows in `push`. Sizing it from --kv-size would take 4 MB of
  // first-touched zeroes at a 1M context for a buffer most runs never reach the end of.
  cache_.assign((size_t)(hint ? hint : 4096u) + 1u, EngramConsts::kDead);
  fallback_muls_.clear();
  if (c_ && c_->prime_muls.size() != c_->primes.size()) {
    fallback_muls_.resize(c_->primes.size());
    for (size_t k = 0; k < c_->primes.size(); ++k) {
      if (c_->primes[k] > 0)
        fallback_muls_[k] = make_barrett_mul(static_cast<uint64_t>(c_->primes[k]));
    }
  }
  filled_ = 0;
}

void EngramHash::grow(uint64_t end) {
  if ((size_t)end <= cache_.size()) return;
  // Grow geometrically: a sequence that reaches 1M positions must not reallocate a million times,
  // and the look-back only ever reads BEHIND the write, so nothing is lost by moving the buffer.
  size_t want = cache_.size() ? cache_.size() : 4096u;
  while (want < (size_t)end) want *= 2u;
  cache_.resize(want, EngramConsts::kDead);
}

void EngramHash::seed(const uint32_t* ids, uint32_t n, uint64_t pos0) {
  if (!c_ || !n) return;
  grow(pos0 + n);
  for (uint32_t t = 0; t < n; ++t) {
    const uint32_t id = ids[t];
    cache_[(size_t)(pos0 + t)] = id < c_->vocab ? c_->token_map[id] : 0;
  }
  if (pos0 + n > filled_) filled_ = pos0 + n;
}

void EngramHash::push(const uint32_t* ids, uint32_t n, uint64_t pos0, int64_t* out,
                      const uint8_t* alive) {
  if (!c_ || !n) return;
  const EngramConsts& c = *c_;
  const uint32_t ng = c.max_ngram;

  // The compressed ids first, so a later position in this same batch can look back at an earlier
  // one. The reference writes the whole chunk into its cache before hashing any of it; completing
  // this pass before the parallel hash phase preserves the exact causal sliding-window invariant.
  grow(pos0 + n);
  for (uint32_t t = 0; t < n; ++t) {
    const size_t p = (size_t)(pos0 + t);
    const uint32_t id = ids[t];
    const int32_t m = id < c.vocab ? c.token_map[id] : 0;
    cache_[p] = (alive && !alive[t]) ? EngramConsts::kDead : m;
  }
  filled_ = pos0 + n;

  const EngramPrimeMul* muls = nullptr;
  if (c.prime_muls.size() == c.primes.size()) {
    muls = c.prime_muls.data();
  } else if (fallback_muls_.size() == c.primes.size()) {
    muls = fallback_muls_.data();
  }

  // Once cache_ holds all compressed ids for [pos0, pos0 + n), each position t depends only on
  // read-only cache_[pos - (ng - 1) .. pos] and writes disjointly to out[t], so parallel execution
  // for n >= 256 is 100% bit-identical to serial execution.
  parallel_for((uint64_t)n, 256, [&](uint64_t lo, uint64_t hi) {
    int64_t tok_stack[16];
    std::vector<int64_t> tok_heap;
    int64_t* tok = ng <= 16 ? tok_stack : (tok_heap.resize(ng), tok_heap.data());
    const size_t prime_layer_stride = (size_t)(ng - 1u) * c.n_heads;
    for (uint64_t t = lo; t < hi; ++t) {
      const int64_t pos = (int64_t)(pos0 + t);
      // Look back `ng` slots. `blocked` latches: past the start of the sequence or past a dead
      // position, this and every LONGER n-gram take the pad id.
      bool blocked = false;
      for (uint32_t s = 0; s < ng; ++s) {
        const int64_t src_pos = pos - (int64_t)s > 0 ? pos - (int64_t)s : 0;
        const int32_t src = cache_[(size_t)src_pos];
        if (pos < (int64_t)s || src == EngramConsts::kDead) blocked = true;
        tok[s] = blocked ? (int64_t)c.pad : (int64_t)src;
      }
      for (uint32_t l = 0; l < c.n_layer; ++l) {
        const int64_t* ml = &c.mult[(size_t)l * ng];
        const int64_t* pl = &c.primes[(size_t)l * prime_layer_stride];
        const EngramPrimeMul* pm = muls ? (muls + (size_t)l * prime_layer_stride) : nullptr;
        const int64_t* ol = &c.offsets[(size_t)l * c.n_cols];
        // XOR the multiplied ids together one look-back at a time, so the running value after step i
        // is the hash of the (i+1)-gram. Products stay non-negative — the multiplier bound in the
        // reference exists so `id * multiplier` cannot overflow int64 — which is what lets the
        // modulus below match Python's, whose result takes the sign of the divisor.
        int64_t rolling = tok[0] * ml[0];
        int64_t* orow = out + ((size_t)t * c.n_layer + l) * c.n_cols;
        for (uint32_t i = 1; i < ng; ++i) {
          rolling ^= tok[i] * ml[i];
          const uint32_t base_col = (i - 1u) * c.n_heads;
          if (__builtin_expect(rolling >= 0 && pm != nullptr, 1)) {
            const uint64_t ux = static_cast<uint64_t>(rolling);
            for (uint32_t h = 0; h < c.n_heads; ++h) {
              const uint32_t col = base_col + h;
              const uint64_t p = static_cast<uint64_t>(pl[col]);
              orow[col] = static_cast<int64_t>(barrett_mod(ux, p, pm[col])) + ol[col];
            }
          } else {
            for (uint32_t h = 0; h < c.n_heads; ++h) {
              const uint32_t col = base_col + h;
              orow[col] = rolling % pl[col] + ol[col];
            }
          }
        }
      }
    }
  });
}

// ---- the tables ---------------------------------------------------------------------------------

EngramTables::~EngramTables() {
  for (Map& m : m_) {
    if (m.wbase) ::munmap(m.wbase, m.wlen);
    if (m.sbase) ::munmap(m.sbase, m.slen);
  }
}

bool EngramTables::open(const EngramConsts* c, const std::string& dir, std::string* err) {
  if (!c || c->tables.empty()) {
    if (err) *err = "this engram file carries no table locations (re-run engram_prep.py)";
    return false;
  }
  const long page = ::sysconf(_SC_PAGESIZE);
  m_.assign(c->n_layer, Map{});
  for (uint32_t l = 0; l < c->n_layer; ++l) {
    const EngramConsts::Table& t = c->tables[l];
    const std::string path = dir + "/" + t.shard;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { if (err) *err = "cannot open " + path; return false; }
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); if (err) *err = "cannot stat " + path; return false; }
    if ((uint64_t)st.st_size < t.w_off + t.w_bytes || (uint64_t)st.st_size < t.s_off + t.s_bytes) {
      ::close(fd);
      if (err) *err = path + " is shorter than the table it is supposed to hold";
      return false;
    }
    // mmap wants a page-aligned offset, and neither range is aligned: map from the page below and
    // carry the remainder as a pointer bias.
    auto map_one = [&](uint64_t off, uint64_t len, void** base, size_t* blen,
                       const uint8_t** ptr) -> bool {
      const uint64_t lo = off & ~(uint64_t)(page - 1);
      *blen = (size_t)(off - lo + len);
      // MAP_NORESERVE: this is 91.6 GiB of address space against a demand-paged read of a few
      // hundred rows a token, and reserving swap for it would fail outright.
      *base = ::mmap(nullptr, *blen, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, (off_t)lo);
      if (*base == MAP_FAILED) { *base = nullptr; return false; }
      // Random is the truth: n-gram ids are a hash, so consecutive lookups are nowhere near each
      // other and read-ahead would fetch pages nothing asks for.
      (void)::madvise(*base, *blen, MADV_RANDOM);
      *ptr = (const uint8_t*)*base + (off - lo);
      return true;
    };
    if (!map_one(t.w_off, t.w_bytes, &m_[l].wbase, &m_[l].wlen, &m_[l].w) ||
        !map_one(t.s_off, t.s_bytes, &m_[l].sbase, &m_[l].slen, &m_[l].s)) {
      ::close(fd);
      if (err) *err = "cannot map the engram table in " + path;
      return false;
    }
    m_[l].rows = c->rows[l];
    bytes_ += t.w_bytes + t.s_bytes;
    // Opt-in page-cache warm. The mapping is MADV_RANDOM, so without this each first-touch row is
    // its own 4 KiB demand read from NVMe; a decode step gathers 24 rows a layer and pays that
    // latency on the host. RAM is sized for it (2 x 91.6 GiB against a box already holding a
    // 150 GiB expert pool), at the cost of reading the tables once at startup.
    if (std::getenv("AFF_ENGRAM_WARM")) {
      ::madvise(m_[l].wbase, m_[l].wlen, MADV_WILLNEED);
      ::madvise(m_[l].sbase, m_[l].slen, MADV_WILLNEED);
    }
    // The mapping keeps the file alive; the descriptor does not have to.
    ::close(fd);
  }
  c_ = c;
  return true;
}

void EngramTables::gather(uint32_t layer, const int64_t* ids, float* out) const {
  if (!c_ || layer >= m_.size()) return;
  const EngramConsts& c = *c_;
  const Map& m = m_[layer];
  const uint32_t dim = c.head_dim, nsc = dim / 32u;      // one E8M0 scale per 32 values
  for (uint32_t col = 0; col < c.n_cols; ++col) {
    float* dst = out + (size_t)col * dim;
    const int64_t id = ids[col];
    // Out of range comes back as zeros, which is what the reference's own bounds mask produces —
    // it is the shard-parallel path's "not mine", and at world_size 1 it can only mean a corrupt id.
    if (id < 0 || (uint64_t)id >= m.rows) { std::memset(dst, 0, (size_t)dim * sizeof(float)); continue; }
    const uint8_t* w = m.w + (size_t)id * dim;
    const uint8_t* sc = m.s + (size_t)id * nsc;
    for (uint32_t b = 0; b < nsc; ++b) {
      // E8M0: the byte IS the exponent, biased by 127. 0xFF is the NaN encoding and never a scale.
      const float s = sc[b] == 0xFFu ? 0.0f : std::ldexp(1.0f, (int)sc[b] - 127);
      for (uint32_t i = 0; i < 32u; ++i)
        dst[b * 32u + i] = fp8_e4m3_to_float(w[b * 32u + i]) * s;
    }
  }
}

void EngramTables::gather_mxfp8(uint32_t layer, const int64_t* ids, uint8_t* dst) const {
  if (!c_ || layer >= m_.size() || !dst || c_->head_dim % 32u != 0u) return;
  const EngramConsts& c = *c_;
  const Map& m = m_[layer];
  const uint32_t dim = c.head_dim, nsc = dim / 32u;
  const uint32_t in_dim = c.n_cols * dim;
  uint8_t* const w_base = dst;
  uint8_t* const s_base = dst + in_dim;
  if (!m.w || !m.s || !ids) {
    std::memset(w_base, 0x00, in_dim);
    std::memset(s_base, 0xFF, (size_t)c.n_cols * nsc);
    return;
  }
#if defined(__x86_64__) || defined(_M_X64)
  constexpr uint32_t kPrefetchAhead = 16u;
  auto prefetch_col = [&](uint32_t k) {
    const int64_t pid = ids[k];
    if (pid >= 0 && (uint64_t)pid < m.rows) {
      const char* pw = reinterpret_cast<const char*>(m.w + (size_t)pid * dim);
      for (uint32_t off = 0; off < dim; off += 64u) _mm_prefetch(pw + off, _MM_HINT_T0);
      const char* ps = reinterpret_cast<const char*>(m.s + (size_t)pid * nsc);
      for (uint32_t off = 0; off < nsc; off += 64u) _mm_prefetch(ps + off, _MM_HINT_T0);
    }
  };
  const uint32_t prime_n = std::min(kPrefetchAhead, c.n_cols);
  for (uint32_t k = 0; k < prime_n; ++k) prefetch_col(k);
#endif
#ifdef __AVX2__
  const __m256i vzero = _mm256_setzero_si256();
#endif
  for (uint32_t col = 0; col < c.n_cols; ++col) {
#if defined(__x86_64__) || defined(_M_X64)
    if (col + kPrefetchAhead < c.n_cols) prefetch_col(col + kPrefetchAhead);
#endif
    uint8_t* const w_dst = w_base + (size_t)col * dim;
    uint8_t* const s_dst = s_base + (size_t)col * nsc;
    const int64_t id = ids[col];
    if (id < 0 || (uint64_t)id >= m.rows) {
#ifdef __AVX2__
      for (uint32_t off = 0; off < dim; off += 32u)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(w_dst + off), vzero);
#else
      std::memset(w_dst, 0x00, dim);
#endif
      std::memset(s_dst, 0xFF, nsc);
      continue;
    }
    const uint8_t* const w_src = m.w + (size_t)id * dim;
    const uint8_t* const s_src = m.s + (size_t)id * nsc;
#ifdef __AVX2__
    // 256-bit AVX2 copy only (no 512-bit ZMM instructions) to avoid Skylake-SP License L2 downclock.
    for (uint32_t off = 0; off < dim; off += 32u) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_src + off));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(w_dst + off), v);
    }
#else
    std::memcpy(w_dst, w_src, dim);
#endif
    if (nsc == 8u) {
      uint64_t sv;
      std::memcpy(&sv, s_src, 8);
      std::memcpy(s_dst, &sv, 8);
    } else {
      std::memcpy(s_dst, s_src, nsc);
    }
  }
}

}  // namespace aff
