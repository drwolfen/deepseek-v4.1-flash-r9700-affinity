// N-rank all-reduce for gfx12 cards, adapted from vllm-radiance's radiance_ar_ext.
//
// The original is a PyTorch extension for TP=2 across two PROCESSES, so it trades IPC handles
// (hipIpcGetMemHandle / open_shared) to give each rank a pointer into the other's scratch. This
// engine drives every card from ONE process, so that whole layer disappears: with
// hipDeviceEnablePeerAccess, a kernel on device 0 dereferences a device-1 allocation directly.
//
// What carries over unchanged, because it is what makes the crossing fast on this hardware:
//
//   * ONE-SHOT PUSH, not a ring. Each rank writes its input into every peer's scratch, then reduces
//     (my input + the peers' data now in my scratch) into out. One crossing per peer, not two.
//   * FINE-GRAINED allocation (hipExtMallocWithFlags + hipDeviceMallocFinegrained). Uncached, so
//     stores post straight to the fabric instead of sitting in L2 waiting for a writeback.
//   * `s_wait_storecnt 0x0` as the drain, NOT __threadfence_system. A threadfence emits an L2
//     writeback-invalidate per wavefront, which is pure waste on memory that was never cached and
//     floods the fabric at high occupancy. The wait drains posted writes with no invalidate, and
//     PCIe keeps posted writes to the same peer ordered, so the release flag lands after the data.
//   * PER-BLOCK flags and no grid barrier. Block b owns a contiguous chunk, pushes it, sets each
//     peer's flag for b, spins on its own flags for b, reduces its own chunk. Blocks are
//     independent.
//   * Scratch DOUBLE-BUFFERED by sequence parity, so a slot cannot be reused while a peer is
//     still draining the previous one.
//
// The sequence counter is device-resident and incremented by the kernel. In vllm that was for
// cudagraph replay safety; here it keeps the ranks in lockstep across calls without the host having
// to pass a matching number to all of them.

#include "gpu/tp_allreduce.h"
#include "gpu/dense_gpu.h"        // kMaxRanks, asserted equal to kTpMaxRanks in tp_init
#include "gpu/device_math.h"
#include "gpu/hip_common.h"
#include "gpu/wmma.h"

#include "ui/log.h"
#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aff {

namespace {

// System scope: the reader is the CPU, and a device-scope store would leave it in a cache the host
// cannot see. `atomicMax` rather than a plain store so the FIRST failing sequence survives a later
// one — the interesting number is where it started, not where it was noticed.
__device__ inline void note_timeout(unsigned int* t, unsigned int s) {
  if (!t) return;
  __hip_atomic_fetch_max(t, s ? s : 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Two hipSetDevice calls a reduce, unconditionally. Caching what this translation unit last set is
// wrong: the engine changes the current device between entry points, so the fix if this ever shows
// up in a profile is for the caller to state which device is current, not to guess again.
inline void set_device(int d) { (void)hipSetDevice(d); }

constexpr size_t kTpWakeBytes = 64;
constexpr int kTpFp8RegCacheGroups = 8;
constexpr int kTpCodedRegCacheGroups = 4;
constexpr int kTpCodedStageIters = 2;
constexpr size_t kTpStagedMinBytes = 4u << 20;  // 4 MiB floor for staged TP=2 FP8/Coded overlap
constexpr int kTpFp8DefaultStages = 8;
constexpr int kTpFp8MaxStages = 8;
constexpr uint32_t kTpArFlagFastRel   = 1u << 0;
constexpr uint32_t kTpArFlagWakeEarly = 1u << 1;
// Per-peer flag stride: [0, kTpMaxBlocks) holds unstaged block flags;
// [kTpMaxBlocks, (1 + kTpFp8MaxStages) * kTpMaxBlocks) holds per-(stage, block) flags.
constexpr size_t kTpFlagStride = (size_t)kTpMaxBlocks * (1u + (size_t)kTpFp8MaxStages);

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void store_sys_relaxed(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void publish_flag(unsigned int* p, unsigned int v, bool fast_rel) {
  if (fast_rel) store_sys_relaxed(p, v);
  else store_sys_rel(p, v);
}
__device__ __forceinline__ unsigned int load_sys_relaxed(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void fence_sys_acq() {
#ifdef __HIP_DEVICE_COMPILE__
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#else
  asm volatile("" ::: "memory");
#endif
}

// The peers a rank pushes to, resolved on the host so the kernel does no index arithmetic over rank
// ids. Both arrays are already offset to THIS rank's sub-slot in that peer's memory — see
// tp_peer_slot — so the kernel only ever adds the double-buffer slot.
template <int NPEER>
struct TpPush {
  float4*       scratch[NPEER];
  unsigned int* flags[NPEER];
  unsigned int* wake[NPEER];
};

__device__ __forceinline__ float4 f4add(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// NPEER is a template argument, not a runtime one, so that at two ranks — the configuration this
// engine is tuned on — the push and the reduce collapse to exactly the single-peer straight line
// they were before the generalisation, with no loop and no indirection through the peer table.
// NRANK_SYNC ports N-rank flag sync from StillDeadcode/libr4d PR #2 (9e0728e).
template <int NPEER, bool NRANK_SYNC = false, bool REG_CACHE = false>
__global__ __launch_bounds__(256) void tp_ar_kernel(const float4* __restrict__ in,
                             TpPush<NPEER> push, const float4* my_scratch,
                             float4* __restrict__ out,
                             unsigned int* my_flags,
                             unsigned int* seq_ctrs, unsigned int* timeout,
                             int n4, int slot_stride4, int peer_stride4, int own,
                             uint32_t ar_flags) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) {
#pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_relaxed(push.wake[j], 1u);
  }

  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) {
#pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_relaxed(push.wake[j], s);
  }

  float4* ps[NPEER];
  const float4* ms[NPEER];
#pragma unroll
  for (int j = 0; j < NPEER; ++j) {
    ps[j] = push.scratch[j] + (size_t)slot * slot_stride4;
    ms[j] = my_scratch + (size_t)slot * slot_stride4 + (size_t)j * peer_stride4;
  }

  const int w = (n4 + nb - 1) / nb;
  const int start = b * w;
  const int end = (start + w < n4) ? (start + w) : n4;

  // Push my chunk into every peer's scratch, unrolled for memory-level parallelism: PCIe wants many
  // writes in flight, and one per thread per iteration does not saturate it. `in` is read once and
  // fanned out from registers — the crossings are what this costs, not the local reads.
  constexpr int kExactRegCap = 8;
  float4 r_in[(REG_CACHE && NPEER == 1) ? kExactRegCap : 1];
  if constexpr (REG_CACHE && NPEER == 1) {
    const int k0 = start + tid;
#pragma unroll
    for (int i = 0; i < kExactRegCap; ++i) {
      const int k = k0 + i * nt;
      if (k < end) r_in[i] = in[k];
    }
#pragma unroll
    for (int i = 0; i < kExactRegCap; ++i) {
      const int k = k0 + i * nt;
      if (k < end) ps[0][k] = r_in[i];
    }
    for (int k = k0 + kExactRegCap * nt; k < end; k += nt) ps[0][k] = in[k];
  } else {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const float4 v0 = in[k], v1 = in[k + nt], v2 = in[k + 2 * nt], v3 = in[k + 3 * nt];
#pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        ps[j][k] = v0; ps[j][k + nt] = v1; ps[j][k + 2 * nt] = v2; ps[j][k + 3 * nt] = v3;
      }
    }
    for (; k < end; k += nt) {
      const float4 v = in[k];
#pragma unroll
      for (int j = 0; j < NPEER; ++j) ps[j][k] = v;
    }
  }

  // Drain posted writes without an L2 invalidate — see the header comment.
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  aff::lds_barrier();
  const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;

  if constexpr (NRANK_SYNC && NPEER > 1) {
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < NPEER; ++j) publish_flag(&push.flags[j][b], s, fast_rel);
    }
    if (tid < NPEER) {
      const unsigned int* f = &my_flags[tid * kTpFlagStride + b];
      unsigned long long z = 0;
      unsigned int v = load_sys_relaxed(f);
      if ((int)(v - s) < 0) {
        do {
          __builtin_amdgcn_s_sleep(1);
          v = load_sys_relaxed(f);
        } while ((int)(v - s) < 0 && ++z <= kTpSpinMax);
        if ((int)(v - s) < 0) note_timeout(timeout, s);
      }
    }
  } else {
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < NPEER; ++j) publish_flag(&push.flags[j][b], s, fast_rel);
      unsigned long long z = 0;
#pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        const unsigned int* f = &my_flags[j * kTpFlagStride + b];
        unsigned int v = load_sys_relaxed(f);
        if ((int)(v - s) < 0) {
          do {
            __builtin_amdgcn_s_sleep(1);
            v = load_sys_relaxed(f);
          } while ((int)(v - s) < 0 && ++z <= kTpSpinMax);
          if ((int)(v - s) < 0) note_timeout(timeout, s);
        }
      }
    }
  }
  aff::lds_barrier();
  fence_sys_acq();

  // out = the sum of every rank's input, mine from `in` and the rest from my scratch.
  //
  // THE ORDER IS BY RANK, NOT "MINE FIRST". Float addition is commutative but not associative, so
  // summing my own contribution first and the peers after it gives rank 1 `x1+x0+x2` where rank 0
  // gets `x0+x1+x2` — the same value to within an ulp, and a DIFFERENT one in bits. An all-reduce
  // leaves a replicated tensor; two cards holding numbers that differ in the last bit are two cards
  // running slightly different models, and everything downstream of it compounds that. At two ranks
  // the sum has two terms and commutativity alone settles it, which is why the ordering only starts
  // to matter at three.
  {
    int k = start + tid;
    if constexpr (NPEER == 1) {
      if constexpr (REG_CACHE) {
        const int k0 = start + tid;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          float4 r_peer[2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int kj = k0 + (g * 2 + j) * nt;
            if (kj < end) r_peer[j] = ms[0][kj];
          }
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int kj = k0 + (g * 2 + j) * nt;
            if (kj < end) out[kj] = f4add(r_in[g * 2 + j], r_peer[j]);
          }
        }
        for (int kt = k0 + kExactRegCap * nt; kt < end; kt += nt) out[kt] = f4add(in[kt], ms[0][kt]);
      } else {
        // The two-rank straight line, kept verbatim: one peer, no source table, no select.
        for (; k + 3 * nt < end; k += 4 * nt) {
          const float4 a0 = in[k], a1 = in[k + nt], a2 = in[k + 2 * nt], a3 = in[k + 3 * nt];
          const float4 b0 = ms[0][k], b1 = ms[0][k + nt];
          const float4 b2 = ms[0][k + 2 * nt], b3 = ms[0][k + 3 * nt];
          out[k]          = f4add(a0, b0); out[k + nt]     = f4add(a1, b1);
          out[k + 2 * nt] = f4add(a2, b2); out[k + 3 * nt] = f4add(a3, b3);
        }
        for (; k < end; k += nt) out[k] = f4add(in[k], ms[0][k]);
      }
    } else {
      // src[p] is where rank p's contribution lives as this rank sees it: `in` for my own, the
      // matching sub-slot for everyone else. Summing over p in order is then the same expression on
      // every rank by construction rather than by an argument about which slot holds what.
      const float4* src[NPEER + 1];
#pragma unroll
      for (int p = 0; p <= NPEER; ++p) src[p] = (p == own) ? in : ms[p < own ? p : p - 1];
      for (; k + 3 * nt < end; k += 4 * nt) {
        float4 a0 = src[0][k], a1 = src[0][k + nt];
        float4 a2 = src[0][k + 2 * nt], a3 = src[0][k + 3 * nt];
#pragma unroll
        for (int p = 1; p <= NPEER; ++p) {
          a0 = f4add(a0, src[p][k]);          a1 = f4add(a1, src[p][k + nt]);
          a2 = f4add(a2, src[p][k + 2 * nt]); a3 = f4add(a3, src[p][k + 3 * nt]);
        }
        out[k] = a0; out[k + nt] = a1; out[k + 2 * nt] = a2; out[k + 3 * nt] = a3;
      }
      for (; k < end; k += nt) {
        float4 a = src[0][k];
#pragma unroll
        for (int p = 1; p <= NPEER; ++p) a = f4add(a, src[p][k]);
        out[k] = a;
      }
    }
  }
}

// FP8 variant of the push. Only the CROSSING is quantised, and both ranks reduce against the
// quantised form of BOTH halves so the two cards end on the same bits — see the reduce loop.
//
// Scratch layout per slot: [ e4m3 bytes | ... | f32 scales ], scales at a fixed offset so both
// ranks agree without exchanging metadata.
__device__ __forceinline__ uint32_t pack4_e4m3(float a, float b, float c, float d) {
  const float c0 = fminf(448.0f, fmaxf(-448.0f, a));
  const float c1 = fminf(448.0f, fmaxf(-448.0f, b));
  const float c2 = fminf(448.0f, fmaxf(-448.0f, c));
  const float c3 = fminf(448.0f, fmaxf(-448.0f, d));
#if __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
  int pk = 0;
  pk = __builtin_amdgcn_cvt_pk_fp8_f32(c0, c1, pk, false);
  pk = __builtin_amdgcn_cvt_pk_fp8_f32(c2, c3, pk, true);
  return (uint32_t)pk;
#else
  return (uint32_t)aff::e4m3_enc_sw(c0) | ((uint32_t)aff::e4m3_enc_sw(c1) << 8) |
         ((uint32_t)aff::e4m3_enc_sw(c2) << 16) | ((uint32_t)aff::e4m3_enc_sw(c3) << 24);
#endif
}

template <int NPEER>
struct TpPush8 {
  unsigned char* base[NPEER];
  unsigned int*  flags[NPEER];
  unsigned int*  wake[NPEER];
};

template <int NPEER, bool NRANK_SYNC = false, bool CACHE_REGS = false>
__global__ __launch_bounds__(256) void tp_ar_fp8_kernel(const float* __restrict__ in,
                                 TpPush8<NPEER> push, const unsigned char* my_base,
                                 float* __restrict__ out,
                                 unsigned int* my_flags,
                                 unsigned int* seq_ctrs, unsigned int* timeout,
                                 int n, int slot_stride_bytes, int peer_stride_bytes,
                                 int scale_off_bytes, int own, uint32_t ar_flags) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) {
#pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_relaxed(push.wake[j], 1u);
  }
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) {
#pragma unroll
    for (int j = 0; j < NPEER; ++j) store_sys_relaxed(push.wake[j], s);
  }

  unsigned char* pb[NPEER];
  float* p_scale[NPEER];
  const unsigned char* mb[NPEER];
  const float* m_scale[NPEER];
#pragma unroll
  for (int j = 0; j < NPEER; ++j) {
    pb[j] = push.base[j] + (size_t)slot * slot_stride_bytes;
    p_scale[j] = (float*)(pb[j] + scale_off_bytes);
    mb[j] = my_base + (size_t)slot * slot_stride_bytes + (size_t)j * peer_stride_bytes;
    m_scale[j] = (const float*)(mb[j] + scale_off_bytes);
  }

  // ONE WAVE per group, not one thread. Each lane owns 4 contiguous elements, packs its 4
  // quantized E4M3 bytes into a uint32_t for coalesced 4-byte P2P BAR writes/reads, and when the
  // per-wave group count fits within kTpFp8RegCacheGroups keeps quantized codes + scales in
  // registers across the handshake to avoid a second pass over `in`.
  const int n_groups = (n + kTpFp8Group - 1) / kTpFp8Group;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int gw = (n_groups + nb - 1) / nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  constexpr int kPerLane = kTpFp8Group / kWave;      // 4

  uint32_t reg_pk[CACHE_REGS ? kTpFp8RegCacheGroups : 1];
  float reg_sc[CACHE_REGS ? kTpFp8RegCacheGroups : 1];

  {
    int gi = 0;
    for (int g = g0 + wave; g < g1; g += waves, ++gi) {
      const int lo = g * kTpFp8Group;
      const int base_idx = lo + lane * kPerLane;
      float v[kPerLane];
      float amax = 0.0f;
      #pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        const int idx = base_idx + i;
        v[i] = (idx < n) ? in[idx] : 0.0f;
        amax = fmaxf(amax, fabsf(v[i]));
      }
      amax = aff::wave_reduce_max_dpp(amax);

      const float scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
      const float inv = 1.0f / scale;
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) if (lane == 0) p_scale[j][g] = scale;

      const uint32_t pk = pack4_e4m3(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
      if (base_idx + (kPerLane - 1) < n) {
        #pragma unroll
        for (int j = 0; j < NPEER; ++j) *(uint32_t*)(pb[j] + base_idx) = pk;
      } else {
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) {
            const unsigned char q = (unsigned char)(pk >> (8 * i));
            #pragma unroll
            for (int j = 0; j < NPEER; ++j) pb[j][idx] = q;
          }
        }
      }
      if constexpr (CACHE_REGS) {
        if (gi < kTpFp8RegCacheGroups) {
          reg_pk[gi] = pk;
          reg_sc[gi] = scale;
        }
      }
    }
  }

  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  aff::lds_barrier();
  const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;
  if constexpr (NRANK_SYNC && NPEER > 1) {
    if (tid == 0) {
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) publish_flag(&push.flags[j][b], s, fast_rel);
    }
    if (tid < NPEER) {
      const unsigned int* f = &my_flags[tid * kTpFlagStride + b];
      unsigned long long z = 0;
      unsigned int fv = load_sys_relaxed(f);
      if ((int)(fv - s) < 0) {
        do {
          __builtin_amdgcn_s_sleep(1);
          fv = load_sys_relaxed(f);
        } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
        if ((int)(fv - s) < 0) note_timeout(timeout, s);
      }
    }
  } else {
    if (tid == 0) {
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) publish_flag(&push.flags[j][b], s, fast_rel);
      unsigned long long z = 0;
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        const unsigned int* f = &my_flags[j * kTpFlagStride + b];
        unsigned int fv = load_sys_relaxed(f);
        if ((int)(fv - s) < 0) {
          do {
            __builtin_amdgcn_s_sleep(1);
            fv = load_sys_relaxed(f);
          } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
          if ((int)(fv - s) < 0) note_timeout(timeout, s);
        }
      }
    }
  }
  aff::lds_barrier();
  fence_sys_acq();

  // Reduce against MY OWN QUANTISED value, not my exact one — see the replication comment above.
  {
    int gi = 0;
    for (int g = g0 + wave; g < g1; g += waves, ++gi) {
      const int lo = g * kTpFp8Group;
      const int base_idx = lo + lane * kPerLane;
      float sc[NPEER];
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) sc[j] = m_scale[j][g];

      float mine_scale;
      uint32_t pk = 0u;
      auto quant_local = [&]() {
        float v[kPerLane];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          v[i] = (idx < n) ? in[idx] : 0.0f;
          amax = fmaxf(amax, fabsf(v[i]));
        }
        amax = aff::wave_reduce_max_dpp(amax);
        mine_scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
        const float mine_inv = 1.0f / mine_scale;
        pk = pack4_e4m3(v[0] * mine_inv, v[1] * mine_inv, v[2] * mine_inv, v[3] * mine_inv);
      };
      if constexpr (CACHE_REGS) {
        if (gi < kTpFp8RegCacheGroups) {
          mine_scale = reg_sc[gi];
          pk = reg_pk[gi];
        } else {
          quant_local();
        }
      } else {
        quant_local();
      }

      if (base_idx + (kPerLane - 1) < n) {
        uint32_t peer_pk[NPEER];
        #pragma unroll
        for (int j = 0; j < NPEER; ++j) peer_pk[j] = *(const uint32_t*)(mb[j] + base_idx);
        if constexpr (NPEER == 1) {
          const aff::aff_f32x2 m0 = aff::e4m3_decode_pk<false>(pk);
          const aff::aff_f32x2 m1 = aff::e4m3_decode_pk<true>(pk);
          const aff::aff_f32x2 p0 = aff::e4m3_decode_pk<false>(peer_pk[0]);
          const aff::aff_f32x2 p1 = aff::e4m3_decode_pk<true>(peer_pk[0]);
          out[base_idx + 0] = m0.x * mine_scale + p0.x * sc[0];
          out[base_idx + 1] = m0.y * mine_scale + p0.y * sc[0];
          out[base_idx + 2] = m1.x * mine_scale + p1.x * sc[0];
          out[base_idx + 3] = m1.y * mine_scale + p1.y * sc[0];
        } else {
          float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
          #pragma unroll
          for (int p = 0; p <= NPEER; ++p) {
            uint32_t q;
            float s_val;
            if (p == own) {
              q = pk;
              s_val = mine_scale;
            } else {
              const int j = p < own ? p : p - 1;
              q = peer_pk[j];
              s_val = sc[j];
            }
            const aff::aff_f32x2 d0 = aff::e4m3_decode_pk<false>(q);
            const aff::aff_f32x2 d1 = aff::e4m3_decode_pk<true>(q);
            acc0 += d0.x * s_val;
            acc1 += d0.y * s_val;
            acc2 += d1.x * s_val;
            acc3 += d1.y * s_val;
          }
          out[base_idx + 0] = acc0;
          out[base_idx + 1] = acc1;
          out[base_idx + 2] = acc2;
          out[base_idx + 3] = acc3;
        }
      } else {
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) {
            const float mine = aff::e4m3_decode((pk >> (8 * i)) & 0xFFu) * mine_scale;
            if constexpr (NPEER == 1) {
              out[idx] = mine + aff::e4m3_decode(mb[0][idx]) * sc[0];
            } else {
              float acc = 0.0f;
              #pragma unroll
              for (int p = 0; p <= NPEER; ++p) {
                if (p == own) { acc += mine; continue; }
                const int j = p < own ? p : p - 1;
                acc += aff::e4m3_decode(mb[j][idx]) * sc[j];
              }
              out[idx] = acc;
            }
          }
        }
      }
    }
  }
}

// Staged 2-rank FP8 all-reduce for large prefill payloads: splits each block's [g0, g1) range into
// S contiguous stages on 128-element group boundaries and overlaps PCIe BAR push of stage j with
// local wait + quant + HBM reduce of stage j - 1. Wire format and outputs are bit-identical to
// tp_ar_fp8_kernel.
template <bool CacheStage>
__global__ __launch_bounds__(256) void tp_ar_fp8_staged_kernel(const float* __restrict__ in,
                                        TpPush8<1> push, const unsigned char* my_base,
                                        float* __restrict__ out,
                                        unsigned int* my_flags,
                                        unsigned int* seq_ctrs, unsigned int* timeout,
                                        int n, int slot_stride_bytes, int scale_off_bytes,
                                        int stages, uint32_t ar_flags) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], 1u);
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], s);

  unsigned char* const pb0 = push.base[0] + (size_t)slot * slot_stride_bytes;
  float* const p_scale0 = (float*)(pb0 + scale_off_bytes);
  const unsigned char* const mb0 = my_base + (size_t)slot * slot_stride_bytes;
  const float* const m_scale0 = (const float*)(mb0 + scale_off_bytes);

  const int n_groups = (n + kTpFp8Group - 1) / kTpFp8Group;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int gw = (n_groups + nb - 1) / nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  const int gw_blk = (g1 > g0) ? (g1 - g0) : 0;
  constexpr int kPerLane = kTpFp8Group / kWave;  // 4
  // Two ping-pong banks of 30 groups take 2 * 30 * (32 * 4 + 4) = 7,920 B <= 8 KiB LDS.
  // Wave k owns identical groups g in both push_stage(st) and fold_stage(st), and the parity
  // bank (st & 1) separates in-flight push_stage(st) from fold_stage(st - 1).
  constexpr int kStageCacheGroups = 30;
  __shared__ uint32_t s_pk[CacheStage ? 2 : 1][CacheStage ? kStageCacheGroups : 1]
                          [CacheStage ? kWave : 1];
  __shared__ float s_sc[CacheStage ? 2 : 1][CacheStage ? kStageCacheGroups : 1];

  auto quant_group = [&](int g, float& mine_scale, uint32_t& pk, bool write_peer) {
    const int lo = g * kTpFp8Group;
    const int base_idx = lo + lane * kPerLane;
    float v[kPerLane];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int idx = base_idx + i;
      v[i] = (idx < n) ? in[idx] : 0.0f;
      amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = aff::wave_reduce_max_dpp(amax);
    mine_scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
    const float inv = 1.0f / mine_scale;
    if (write_peer && lane == 0) p_scale0[g] = mine_scale;
    pk = pack4_e4m3(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
    if (write_peer) {
      if (base_idx + (kPerLane - 1) < n) {
        *(uint32_t*)(pb0 + base_idx) = pk;
      } else {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) pb0[idx] = (unsigned char)(pk >> (8 * i));
        }
      }
    }
  };

  auto push_stage = [&](int st) {
    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    for (int g = sg0 + wave; g < sg1; g += waves) {
      float sc_tmp;
      uint32_t pk_tmp;
      quant_group(g, sc_tmp, pk_tmp, true);
      if constexpr (CacheStage) {
        const int lg = g - sg0;
        if (lg < kStageCacheGroups) {
          s_pk[bank][lg][lane] = pk_tmp;
          if (lane == 0) s_sc[bank][lg] = sc_tmp;
        }
      }
    }
  };

  const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;
  auto publish_stage = [&](int st) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    aff::lds_barrier();
    if (tid == 0) publish_flag(&push.flags[0][(size_t)(1 + st) * kTpMaxBlocks + b], s, fast_rel);
  };

  auto fold_stage = [&](int st) {
    if (tid == 0) {
      const unsigned int* const f = &my_flags[(size_t)(1 + st) * kTpMaxBlocks + b];
      unsigned long long z = 0;
      unsigned int fv = load_sys_relaxed(f);
      if ((int)(fv - s) < 0) {
        do {
          __builtin_amdgcn_s_sleep(1);
          fv = load_sys_relaxed(f);
        } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
        if ((int)(fv - s) < 0) note_timeout(timeout, s);
      }
    }
    aff::lds_barrier();
    fence_sys_acq();

    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    for (int g = sg0 + wave; g < sg1; g += waves) {
      const int lo = g * kTpFp8Group;
      const int base_idx = lo + lane * kPerLane;
      const float peer_scale = m_scale0[g];
      float mine_scale;
      uint32_t pk;
      if constexpr (CacheStage) {
        const int lg = g - sg0;
        if (lg < kStageCacheGroups) {
          pk = s_pk[bank][lg][lane];
          mine_scale = s_sc[bank][lg];
        } else {
          quant_group(g, mine_scale, pk, false);
        }
      } else {
        quant_group(g, mine_scale, pk, false);
      }

      if (base_idx + (kPerLane - 1) < n) {
        const uint32_t peer_pk = *(const uint32_t*)(mb0 + base_idx);
        const aff::aff_f32x2 m0 = aff::e4m3_decode_pk<false>(pk);
        const aff::aff_f32x2 m1 = aff::e4m3_decode_pk<true>(pk);
        const aff::aff_f32x2 p0 = aff::e4m3_decode_pk<false>(peer_pk);
        const aff::aff_f32x2 p1 = aff::e4m3_decode_pk<true>(peer_pk);
        out[base_idx + 0] = m0.x * mine_scale + p0.x * peer_scale;
        out[base_idx + 1] = m0.y * mine_scale + p0.y * peer_scale;
        out[base_idx + 2] = m1.x * mine_scale + p1.x * peer_scale;
        out[base_idx + 3] = m1.y * mine_scale + p1.y * peer_scale;
      } else {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) {
            const float mine = aff::e4m3_decode((pk >> (8 * i)) & 0xFFu) * mine_scale;
            out[idx] = mine + aff::e4m3_decode(mb0[idx]) * peer_scale;
          }
        }
      }
    }
  };

  push_stage(0);
  publish_stage(0);
  for (int st = 1; st < stages; ++st) {
    push_stage(st);
    fold_stage(st - 1);
    publish_stage(st);
  }
  fold_stage(stages - 1);
}

// Staged 2-rank FP8-wire all-reduce fused with prefill HC post/expand over [n_embd][nb].
template <uint32_t NH, bool HasControl, bool CacheStage>
__global__ __launch_bounds__(256) void tp_ar_fp8_hc_expand_kernel(
    uint16_t* out_hc, float* __restrict__ block, const uint16_t* hc,
    const float* __restrict__ post, const float* __restrict__ comb, TpPush8<1> push,
    const unsigned char* my_base, unsigned int* my_flags, unsigned int* seq_ctrs,
    unsigned int* timeout, uint32_t n_embd, uint32_t nb, int n, int slot_stride_bytes,
    int scale_off_bytes, int stages, int zb, uint32_t ar_flags) {
  const int b = blockIdx.x, grid_nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], 1u);
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], s);

  unsigned char* const pb0 = push.base[0] + (size_t)slot * slot_stride_bytes;
  float* const p_scale0 = (float*)(pb0 + scale_off_bytes);
  const unsigned char* const mb0 = my_base + (size_t)slot * slot_stride_bytes;
  const float* const m_scale0 = (const float*)(mb0 + scale_off_bytes);

  const int n_groups = (n + kTpFp8Group - 1) / kTpFp8Group;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int gw = (n_groups + grid_nb - 1) / grid_nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  const int gw_blk = (g1 > g0) ? (g1 - g0) : 0;
  constexpr int kPerLane = kTpFp8Group / kWave;  // 4
  constexpr int kStageCacheGroups = 30;
  __shared__ uint32_t s_pk[CacheStage ? 2 : 1][CacheStage ? kStageCacheGroups : 1]
                          [CacheStage ? kWave : 1];
  __shared__ float s_sc[CacheStage ? 2 : 1][CacheStage ? kStageCacheGroups : 1];
  const uint32_t nb4 = nb >> 2;

  auto quant_group = [&](int g, float& mine_scale, uint32_t& pk, bool write_peer) {
    const int lo = g * kTpFp8Group;
    const int base_idx = lo + lane * kPerLane;
    float v[kPerLane];
    if (base_idx + (kPerLane - 1) < n) {
      const float4 bv = *(const float4*)(block + base_idx);
      v[0] = bv.x; v[1] = bv.y; v[2] = bv.z; v[3] = bv.w;
    } else {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        const int idx = base_idx + i;
        v[i] = (idx < n) ? block[idx] : 0.0f;
      }
    }
    const float amax = aff::wave_reduce_max_dpp(
        fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    mine_scale = (amax > 0.0f) ? amax * (1.0f / 448.0f) : 1.0f;
    const float inv = 1.0f / mine_scale;
    if (write_peer && lane == 0) p_scale0[g] = mine_scale;
    pk = pack4_e4m3(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
    if (write_peer) {
      if (base_idx + (kPerLane - 1) < n) {
        *(uint32_t*)(pb0 + base_idx) = pk;
      } else {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) pb0[idx] = (unsigned char)(pk >> (8 * i));
        }
      }
    }
  };

  auto push_stage = [&](int st) {
    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    for (int g = sg0 + wave; g < sg1; g += waves) {
      float sc_tmp;
      uint32_t pk_tmp;
      quant_group(g, sc_tmp, pk_tmp, true);
      if constexpr (CacheStage) {
        const int lg = g - sg0;
        if (lg < kStageCacheGroups) {
          s_pk[bank][lg][lane] = pk_tmp;
          if (lane == 0) s_sc[bank][lg] = sc_tmp;
        }
      }
    }
  };

  const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;
  auto publish_stage = [&](int st) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    aff::lds_barrier();
    if (tid == 0) publish_flag(&push.flags[0][(size_t)(1 + st) * kTpMaxBlocks + b], s, fast_rel);
  };

  auto fold_stage = [&](int st) {
    if (tid == 0) {
      const unsigned int* const f = &my_flags[(size_t)(1 + st) * kTpMaxBlocks + b];
      unsigned long long z = 0;
      unsigned int fv = load_sys_relaxed(f);
      if ((int)(fv - s) < 0) {
        do {
          __builtin_amdgcn_s_sleep(1);
          fv = load_sys_relaxed(f);
        } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
        if ((int)(fv - s) < 0) note_timeout(timeout, s);
      }
    }
    aff::lds_barrier();
    fence_sys_acq();

    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    for (int g = sg0 + wave; g < sg1; g += waves) {
      const int lo = g * kTpFp8Group;
      const int base_idx = lo + lane * kPerLane;
      if (base_idx >= n) continue;
      const float peer_scale = m_scale0[g];
      float mine_scale;
      uint32_t pk;
      if constexpr (CacheStage) {
        const int lg = g - sg0;
        if (lg < kStageCacheGroups) {
          pk = s_pk[bank][lg][lane];
          mine_scale = s_sc[bank][lg];
        } else {
          quant_group(g, mine_scale, pk, false);
        }
      } else {
        quant_group(g, mine_scale, pk, false);
      }
      const uint32_t peer_pk = *(const uint32_t*)(mb0 + base_idx);
      const aff::aff_f32x2 m0 = aff::e4m3_decode_pk<false>(pk);
      const aff::aff_f32x2 m1 = aff::e4m3_decode_pk<true>(pk);
      const aff::aff_f32x2 p0 = aff::e4m3_decode_pk<false>(peer_pk);
      const aff::aff_f32x2 p1 = aff::e4m3_decode_pk<true>(peer_pk);
      const float bv[4] = {
          m0.x * mine_scale + p0.x * peer_scale,
          m0.y * mine_scale + p0.y * peer_scale,
          m1.x * mine_scale + p1.x * peer_scale,
          m1.y * mine_scale + p1.y * peer_scale,
      };
      if (zb) *(float4*)(block + base_idx) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if constexpr (HasControl) {
        const uint32_t idx4 = (uint32_t)base_idx >> 2;
        const uint32_t d = idx4 / nb4;
        const uint32_t b0 = (idx4 - d * nb4) << 2;
        float v[NH][4];
#pragma unroll
        for (uint32_t src = 0; src < NH; ++src) {
          const uint2 h4 = *(const uint2*)(hc + ((size_t)src * n_embd + d) * nb + b0);
          v[src][0] = wmma::bf16_to_f32((uint16_t)(h4.x & 0xFFFFu));
          v[src][1] = wmma::bf16_to_f32((uint16_t)(h4.x >> 16));
          v[src][2] = wmma::bf16_to_f32((uint16_t)(h4.y & 0xFFFFu));
          v[src][3] = wmma::bf16_to_f32((uint16_t)(h4.y >> 16));
        }
#pragma unroll
        for (uint32_t dst = 0; dst < NH; ++dst) {
          const float4 pv4 = *(const float4*)(post + (size_t)dst * nb + b0);
          float acc[4] = {bv[0] * pv4.x, bv[1] * pv4.y, bv[2] * pv4.z, bv[3] * pv4.w};
#pragma unroll
          for (uint32_t src = 0; src < NH; ++src) {
            const float4 cv4 = *(const float4*)(comb + (size_t)(dst + src * NH) * nb + b0);
            acc[0] += cv4.x * v[src][0];
            acc[1] += cv4.y * v[src][1];
            acc[2] += cv4.z * v[src][2];
            acc[3] += cv4.w * v[src][3];
          }
          const uint16_t h0 = wmma::f32_to_bf16(acc[0]);
          const uint16_t h1 = wmma::f32_to_bf16(acc[1]);
          const uint16_t h2 = wmma::f32_to_bf16(acc[2]);
          const uint16_t h3 = wmma::f32_to_bf16(acc[3]);
          *(uint2*)(out_hc + ((size_t)dst * n_embd + d) * nb + b0) =
              make_uint2((uint32_t)h0 | ((uint32_t)h1 << 16), (uint32_t)h2 | ((uint32_t)h3 << 16));
        }
      } else {
        const uint2 h4 = *(const uint2*)(out_hc + base_idx);
        const uint16_t h0 = wmma::f32_to_bf16(wmma::bf16_to_f32((uint16_t)(h4.x & 0xFFFFu)) + bv[0]);
        const uint16_t h1 = wmma::f32_to_bf16(wmma::bf16_to_f32((uint16_t)(h4.x >> 16)) + bv[1]);
        const uint16_t h2 = wmma::f32_to_bf16(wmma::bf16_to_f32((uint16_t)(h4.y & 0xFFFFu)) + bv[2]);
        const uint16_t h3 = wmma::f32_to_bf16(wmma::bf16_to_f32((uint16_t)(h4.y >> 16)) + bv[3]);
        *(uint2*)(out_hc + base_idx) =
            make_uint2((uint32_t)h0 | ((uint32_t)h1 << 16), (uint32_t)h2 | ((uint32_t)h3 << 16));
      }
    }
  };

  push_stage(0);
  publish_stage(0);
  for (int st = 1; st < stages; ++st) {
    push_stage(st);
    fold_stage(st - 1);
    publish_stage(st);
  }
  fold_stage(stages - 1);
}

// Lossless 12.125-bit exponent-delta BF16 codec (R2) for 64-element groups across 8 adjacent
// Wave32 lanes. Encodes normals within span <= 14 exponents + exact +-0.0 into 96 B/group + 1 B
// base_exp, with 128 B raw BF16 fallback (base_exp == 0xFF) for subnormals, Inf/NaN, or wide
// exponent span so all 65,536 BF16 values round-trip bit-exactly.
__device__ __forceinline__ unsigned r4d_arc_shfl(unsigned v, int src) {
  return (unsigned)__builtin_amdgcn_ds_bpermute(src << 2, (int)v);
}

struct R4dArcEnc { unsigned w[3]; unsigned base; };

__device__ __forceinline__ R4dArcEnc r4d_arc_enc(int4 v) {
  const unsigned d[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
  unsigned emin = 255u, emax = 0u, bad = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu;
    if (e) { emin = e < emin ? e : emin; emax = e > emax ? e : emax; }
    bad |= (unsigned)(e == 0u && m != 0u) | (unsigned)(e == 255u);
  }
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int s = 1; s < 8; s <<= 1) {
    const unsigned a = r4d_arc_shfl(emin, lane ^ s);
    const unsigned b = r4d_arc_shfl(emax, lane ^ s);
    const unsigned c = r4d_arc_shfl(bad, lane ^ s);
    emin = a < emin ? a : emin;
    emax = b > emax ? b : emax;
    bad |= c;
  }
  R4dArcEnc r;
  const unsigned base = (emin == 255u) ? 1u : emin;
  r.base = (bad || emax > base + 14u) ? 0xFFu : base;
  unsigned c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu, sg = u >> 15;
    c[i] = (sg << 11) | (m << 4) | (e ? (e - base + 1u) & 15u : 0u);
  }
  r.w[0] = c[0] | (c[1] << 12) | (c[2] << 24);
  r.w[1] = (c[2] >> 8) | (c[3] << 4) | (c[4] << 16) | (c[5] << 28);
  r.w[2] = (c[5] >> 4) | (c[6] << 8) | (c[7] << 20);
  return r;
}

__device__ __forceinline__ int4 r4d_arc_dec(const unsigned (&w)[3], unsigned base) {
  unsigned c[8];
  c[0] = w[0] & 0xfffu;
  c[1] = (w[0] >> 12) & 0xfffu;
  c[2] = (w[0] >> 24) | ((w[1] & 0xfu) << 8);
  c[3] = (w[1] >> 4) & 0xfffu;
  c[4] = (w[1] >> 16) & 0xfffu;
  c[5] = (w[1] >> 28) | ((w[2] & 0xffu) << 4);
  c[6] = (w[2] >> 8) & 0xfffu;
  c[7] = w[2] >> 20;
  unsigned u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned off = c[i] & 15u;
    u[i] = ((c[i] >> 11) << 15) | ((off ? base + off - 1u : 0u) << 7) | ((c[i] >> 4) & 0x7fu);
  }
  int4 v;
  v.x = (int)(u[0] | (u[1] << 16));
  v.y = (int)(u[2] | (u[3] << 16));
  v.z = (int)(u[4] | (u[5] << 16));
  v.w = (int)(u[6] | (u[7] << 16));
  return v;
}

__device__ __forceinline__ int4 coded_load_bf16_chunk(const float* __restrict__ in, int n,
                                                      int base_idx, bool vec_in) {
  uint16_t h[8];
  if (vec_in && base_idx + 7 < n) {
    const float4 v0 = *(const float4*)(in + base_idx);
    const float4 v1 = *(const float4*)(in + base_idx + 4);
    h[0] = wmma::f32_to_bf16(v0.x); h[1] = wmma::f32_to_bf16(v0.y);
    h[2] = wmma::f32_to_bf16(v0.z); h[3] = wmma::f32_to_bf16(v0.w);
    h[4] = wmma::f32_to_bf16(v1.x); h[5] = wmma::f32_to_bf16(v1.y);
    h[6] = wmma::f32_to_bf16(v1.z); h[7] = wmma::f32_to_bf16(v1.w);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = base_idx + i;
      h[i] = (idx < n) ? wmma::f32_to_bf16(in[idx]) : 0u;
    }
  }
  int4 r;
  r.x = (int)((uint32_t)h[0] | ((uint32_t)h[1] << 16));
  r.y = (int)((uint32_t)h[2] | ((uint32_t)h[3] << 16));
  r.z = (int)((uint32_t)h[4] | ((uint32_t)h[5] << 16));
  r.w = (int)((uint32_t)h[6] | ((uint32_t)h[7] << 16));
  return r;
}

__device__ __forceinline__ void coded_push_group(uint32_t* pb, uint8_t* p_bases, int g, int sub,
                                                 int4 raw) {
  const R4dArcEnc e = r4d_arc_enc(raw);
  uint32_t* const g_pb = pb + (size_t)g * 32u;
  if (e.base == 0xFFu) {
    *(int4*)(g_pb + sub * 4) = raw;
  } else {
    g_pb[sub]      = e.w[0];
    g_pb[8 + sub]  = e.w[1];
    g_pb[16 + sub] = e.w[2];
  }
  if (sub == 0) p_bases[g] = (uint8_t)e.base;
}

__device__ __forceinline__ void coded_fold_group(const uint32_t* mb, const uint8_t* m_bases,
                                                 float* __restrict__ out, int n, int g, int sub,
                                                 int4 raw, bool vec_out) {
  const uint32_t* const g_mb = mb + (size_t)g * 32u;
  const uint8_t pbase = m_bases[g];
  int4 peer_raw;
  if (pbase == 0xFFu) {
    peer_raw = *(const int4*)(g_mb + sub * 4);
  } else {
    unsigned w[3];
    w[0] = g_mb[sub];
    w[1] = g_mb[8 + sub];
    w[2] = g_mb[16 + sub];
    peer_raw = r4d_arc_dec(w, (unsigned)pbase);
  }
  const int base_idx = g * kTpCodedGroup + sub * 8;
  const unsigned md[4] = {(unsigned)raw.x, (unsigned)raw.y, (unsigned)raw.z, (unsigned)raw.w};
  const unsigned pd[4] = {(unsigned)peer_raw.x, (unsigned)peer_raw.y, (unsigned)peer_raw.z,
                          (unsigned)peer_raw.w};
  if (base_idx + 7 < n) {
    float o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t mh = (uint16_t)((md[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
      const uint16_t ph = (uint16_t)((pd[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
      o[i] = wmma::bf16_to_f32(mh) + wmma::bf16_to_f32(ph);
    }
    if (vec_out) {
      *(float4*)(out + base_idx)     = make_float4(o[0], o[1], o[2], o[3]);
      *(float4*)(out + base_idx + 4) = make_float4(o[4], o[5], o[6], o[7]);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[base_idx + i] = o[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = base_idx + i;
      if (idx < n) {
        const uint16_t mh = (uint16_t)((md[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
        const uint16_t ph = (uint16_t)((pd[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
        out[idx] = wmma::bf16_to_f32(mh) + wmma::bf16_to_f32(ph);
      }
    }
  }
}

template <bool CACHE_REGS = false>
__global__ __launch_bounds__(256) void tp_ar_coded_kernel(
    const float* __restrict__ in, TpPush8<1> push, const unsigned char* my_base,
    float* __restrict__ out, unsigned int* my_flags, unsigned int* seq_ctrs,
    unsigned int* timeout, int n, size_t slot_stride_bytes, size_t base_off_bytes,
    uint32_t ar_flags) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], 1u);
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);
  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], s);

  unsigned char* const pb_bytes = push.base[0] + (size_t)slot * slot_stride_bytes;
  uint32_t* const pb = (uint32_t*)pb_bytes;
  uint8_t* const p_bases = (uint8_t*)(pb_bytes + base_off_bytes);
  const unsigned char* const mb_bytes = my_base + (size_t)slot * slot_stride_bytes;
  const uint32_t* const mb = (const uint32_t*)mb_bytes;
  const uint8_t* const m_bases = (const uint8_t*)(mb_bytes + base_off_bytes);

  const int n_groups = (n + kTpCodedGroup - 1) / kTpCodedGroup;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int sub = lane & 7;
  const int gw = (n_groups + nb - 1) / nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  const bool vec_in = (((uintptr_t)in & 15u) == 0u);
  const bool vec_out = (((uintptr_t)out & 15u) == 0u);

  int4 reg_raw[CACHE_REGS ? kTpCodedRegCacheGroups : 1];
  if constexpr (CACHE_REGS) {
#pragma unroll
    for (int gi = 0; gi < kTpCodedRegCacheGroups; ++gi) {
      const int gw_i = g0 + wave * 4 + gi * (waves * 4);
      if (gw_i < g1) {
        const int g = gw_i + (lane >> 3);
        if (g < g1) {
          const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
          reg_raw[gi] = raw;
          coded_push_group(pb, p_bases, g, sub, raw);
        }
      }
    }
    for (int gw_i = g0 + wave * 4 + kTpCodedRegCacheGroups * (waves * 4); gw_i < g1;
         gw_i += waves * 4) {
      const int g = gw_i + (lane >> 3);
      if (g < g1) {
        const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        coded_push_group(pb, p_bases, g, sub, raw);
      }
    }
  } else {
    for (int gw_i = g0 + wave * 4; gw_i < g1; gw_i += waves * 4) {
      const int g = gw_i + (lane >> 3);
      if (g < g1) {
        const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        coded_push_group(pb, p_bases, g, sub, raw);
      }
    }
  }

  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  aff::lds_barrier();
  if (tid == 0) {
    const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;
    publish_flag(&push.flags[0][b], s, fast_rel);
    const unsigned int* const f = &my_flags[b];
    unsigned long long z = 0;
    unsigned int fv = load_sys_relaxed(f);
    if ((int)(fv - s) < 0) {
      do {
        __builtin_amdgcn_s_sleep(1);
        fv = load_sys_relaxed(f);
      } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
      if ((int)(fv - s) < 0) note_timeout(timeout, s);
    }
  }
  aff::lds_barrier();
  fence_sys_acq();

  if constexpr (CACHE_REGS) {
#pragma unroll
    for (int gi = 0; gi < kTpCodedRegCacheGroups; ++gi) {
      const int gw_i = g0 + wave * 4 + gi * (waves * 4);
      if (gw_i < g1) {
        const int g = gw_i + (lane >> 3);
        if (g < g1) {
          coded_fold_group(mb, m_bases, out, n, g, sub, reg_raw[gi], vec_out);
        }
      }
    }
    for (int gw_i = g0 + wave * 4 + kTpCodedRegCacheGroups * (waves * 4); gw_i < g1;
         gw_i += waves * 4) {
      const int g = gw_i + (lane >> 3);
      if (g < g1) {
        const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        coded_fold_group(mb, m_bases, out, n, g, sub, raw, vec_out);
      }
    }
  } else {
    for (int gw_i = g0 + wave * 4; gw_i < g1; gw_i += waves * 4) {
      const int g = gw_i + (lane >> 3);
      if (g < g1) {
        const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        coded_fold_group(mb, m_bases, out, n, g, sub, raw, vec_out);
      }
    }
  }
}

template <bool CacheStage>
__global__ __launch_bounds__(256) void tp_ar_coded_staged_kernel(
    const float* __restrict__ in, TpPush8<1> push, const unsigned char* my_base,
    float* __restrict__ out, unsigned int* my_flags, unsigned int* seq_ctrs,
    unsigned int* timeout, int n, size_t slot_stride_bytes, size_t base_off_bytes,
    int stages, uint32_t ar_flags) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], 1u);
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);
  if (!(ar_flags & kTpArFlagWakeEarly) && b == 0 && tid == 0) store_sys_relaxed(push.wake[0], s);

  unsigned char* const pb_bytes = push.base[0] + (size_t)slot * slot_stride_bytes;
  uint32_t* const pb = (uint32_t*)pb_bytes;
  uint8_t* const p_bases = (uint8_t*)(pb_bytes + base_off_bytes);
  const unsigned char* const mb_bytes = my_base + (size_t)slot * slot_stride_bytes;
  const uint32_t* const mb = (const uint32_t*)mb_bytes;
  const uint8_t* const m_bases = (const uint8_t*)(mb_bytes + base_off_bytes);

  const int n_groups = (n + kTpCodedGroup - 1) / kTpCodedGroup;
  const int waves = nt / kWave;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int sub = lane & 7;
  const int gw = (n_groups + nb - 1) / nb;
  const int g0 = b * gw;
  const int g1 = (g0 + gw < n_groups) ? (g0 + gw) : n_groups;
  const int gw_blk = (g1 > g0) ? (g1 - g0) : 0;
  const bool vec_in = (((uintptr_t)in & 15u) == 0u);
  const bool vec_out = (((uintptr_t)out & 15u) == 0u);
  const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;

  __shared__ int4 s_raw[CacheStage ? 2 : 1][CacheStage ? kTpCodedStageIters : 1]
                       [CacheStage ? 256 : 1];

  auto push_stage = [&](int st) {
    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    int li = 0;
    for (int gw_i = sg0 + wave * 4; gw_i < sg1; gw_i += waves * 4, ++li) {
      const int g = gw_i + (lane >> 3);
      if (g < sg1) {
        const int4 raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        if constexpr (CacheStage) {
          if (li < kTpCodedStageIters) s_raw[bank][li][tid] = raw;
        }
        coded_push_group(pb, p_bases, g, sub, raw);
      }
    }
  };

  auto publish_stage = [&](int st) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    aff::lds_barrier();
    if (tid == 0) publish_flag(&push.flags[0][(size_t)(1 + st) * kTpMaxBlocks + b], s, fast_rel);
  };

  auto fold_stage = [&](int st) {
    if (tid == 0) {
      const unsigned int* const f = &my_flags[(size_t)(1 + st) * kTpMaxBlocks + b];
      unsigned long long z = 0;
      unsigned int fv = load_sys_relaxed(f);
      if ((int)(fv - s) < 0) {
        do {
          __builtin_amdgcn_s_sleep(1);
          fv = load_sys_relaxed(f);
        } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
        if ((int)(fv - s) < 0) note_timeout(timeout, s);
      }
    }
    aff::lds_barrier();
    fence_sys_acq();

    const int sg0 = g0 + (gw_blk * st) / stages;
    const int sg1 = g0 + (gw_blk * (st + 1)) / stages;
    const int bank = st & 1;
    int li = 0;
    for (int gw_i = sg0 + wave * 4; gw_i < sg1; gw_i += waves * 4, ++li) {
      const int g = gw_i + (lane >> 3);
      if (g < sg1) {
        int4 raw;
        if constexpr (CacheStage) {
          if (li < kTpCodedStageIters) raw = s_raw[bank][li][tid];
          else raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        } else {
          raw = coded_load_bf16_chunk(in, n, g * kTpCodedGroup + sub * 8, vec_in);
        }
        coded_fold_group(mb, m_bases, out, n, g, sub, raw, vec_out);
      }
    }
  };

  push_stage(0);
  publish_stage(0);
  for (int st = 1; st < stages; ++st) {
    push_stage(st);
    fold_stage(st - 1);
    publish_stage(st);
  }
  fold_stage(stages - 1);
}

// Fused 2-rank exact FP32 all-reduce + batch HC post/expand (+ optional DSpark tap, V4.1
// pre_in collapse+sq, and side-band Infinity Cache prefetch) over live tokens [0, live) using
// ctx.scratch (0 B extra VRAM) and zeroing `block` across all [0, nb).
template <uint32_t NH, bool HasTap, bool HasControl, bool LiveOne, bool HasPreCollapse,
          bool RegForm, bool Bf16Wire = false, bool HcDpp = false, bool FixedTok = false>
__global__ __launch_bounds__(256) void tp_ar_batch_hc_expand_kernel(
    uint16_t* out_hc, float* __restrict__ block, const uint16_t* hc,
    const float* __restrict__ post, const float* __restrict__ comb, uint32_t n_embd, uint32_t nb,
    uint32_t live, int zb, float* __restrict__ tap, uint32_t b_lo, uint32_t n_keep,
    uint32_t row0, uint32_t rows_cap, uint32_t row_stride, uint32_t slot_off,
    float* __restrict__ peer_scratch, const float* __restrict__ my_scratch,
    unsigned int* __restrict__ peer_flags, unsigned int* __restrict__ my_flags,
    unsigned int* __restrict__ peer_wake, unsigned int* __restrict__ seq_ctrs,
    unsigned int* __restrict__ timeout, size_t slot_floats, uint32_t ar_blk,
    uint32_t ar_flags, const float* __restrict__ pre_in, float* __restrict__ out_cur,
    float* __restrict__ out_part, const uint4* __restrict__ pf_p0, size_t pf_n16_0,
    const uint4* __restrict__ pf_p1, size_t pf_n16_1, uint32_t pf_blocks) {
  const uint32_t b_blk = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if ((ar_flags & kTpArFlagWakeEarly) && b_blk == 0u && tid == 0u) {
    store_sys_relaxed(peer_wake, 1u);
  }
  if (b_blk >= ar_blk) {
    const uint32_t pid = b_blk - ar_blk;
    const size_t p_stride = (size_t)pf_blocks * nt;
    unsigned int init_f = 0u;
    if ((tid & 31u) == 0u) init_f = load_sys_relaxed(&my_flags[0]);
    init_f = (unsigned int)__shfl((int)init_f, 0, kWave);
    auto peer_done = [&]() -> bool {
      unsigned int f = 0u;
      if ((tid & 31u) == 0u) f = load_sys_relaxed(&my_flags[0]);
      f = (unsigned int)__shfl((int)f, 0, kWave);
      return f != init_f;
    };
    unsigned int acc = 0u;
    uint32_t iter = 0u;
    bool stop = false;
    for (size_t i = (size_t)pid * nt + tid; i < pf_n16_0; i += p_stride) {
      const uint4 v = pf_p0[i];
      acc ^= (v.x ^ v.y ^ v.z ^ v.w);
      if ((++iter & 15u) == 0u && peer_done()) { stop = true; break; }
    }
    if (!stop) {
      for (size_t i = (size_t)pid * nt + tid; i < pf_n16_1; i += p_stride) {
        const uint4 v = pf_p1[i];
        acc ^= (v.x ^ v.y ^ v.z ^ v.w);
        if ((++iter & 15u) == 0u && peer_done()) break;
      }
    }
    asm volatile("" :: "v"(acc));
    return;
  }

  constexpr bool kPreDpp = HasPreCollapse && HcDpp;
  constexpr bool kPreLds = HasPreCollapse && !HcDpp;
  __shared__ unsigned int s_seq;
  __shared__ float s_pre[HasPreCollapse ? (NH * 16u) : 1u];
  __shared__ float s_red_cur[kPreLds ? (16u * 256u) : 1u];
  __shared__ float s_red_hc[kPreLds ? (16u * 256u) : 1u];
  // 1 KiB per-wave reduction buffer ([hc/cur][tok][wave]) replacing 32 KiB s_red_cur/hc when HcDpp.
  __shared__ float s_dpp[kPreDpp ? 2u : 1u][kPreDpp ? (LiveOne ? 1u : 16u) : 1u][kPreDpp ? 8u : 1u];
  const uint32_t eff_live = LiveOne ? 1u : live;
  if (tid == 0u) s_seq = atomicAdd(&seq_ctrs[b_blk], 1u) + 1u;
  if constexpr (HasPreCollapse) {
    if (tid < NH * 16u) {
      const uint32_t h = tid >> 4;
      const uint32_t b = tid & 15u;
      s_pre[tid] = (b < eff_live) ? pre_in[(size_t)h * nb + b] : 0.0f;
    }
    if constexpr (kPreLds) {
      for (uint32_t b = 0u; b < eff_live; ++b) {
        s_red_cur[b * 256u + tid] = 0.0f;
        s_red_hc[b * 256u + tid] = 0.0f;
      }
    }
  }
  aff::lds_barrier();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (!(ar_flags & kTpArFlagWakeEarly) && b_blk == 0u && tid == 0u) {
    store_sys_relaxed(peer_wake, s);
  }

  float* const ps = peer_scratch + (size_t)slot * slot_floats;
  const float* const ms = my_scratch + (size_t)slot * slot_floats;
  uint16_t* const ps16 = reinterpret_cast<uint16_t*>(ps);
  const uint16_t* const ms16 = reinterpret_cast<const uint16_t*>(ms);

  const uint32_t d_per_blk_raw = (n_embd + ar_blk - 1u) / ar_blk;
  const uint32_t d_per_blk = RegForm ? ((d_per_blk_raw + 3u) & ~3u) : d_per_blk_raw;
  const uint32_t d0 = (uint32_t)min((size_t)n_embd, (size_t)b_blk * d_per_blk);
  const uint32_t d1 = (uint32_t)min((size_t)n_embd, (size_t)d0 + d_per_blk);
  const size_t idx0 = LiveOne ? (size_t)d0 : (size_t)d0 * eff_live;
  const size_t idx1 = LiveOne ? (size_t)d1 : (size_t)d1 * eff_live;
  const size_t blk_elems = idx1 - idx0;

  constexpr uint32_t kRegChunks = 3u;
  constexpr uint32_t kRegCap = kRegChunks * 4u;
  float r_blk[RegForm ? kRegCap : 1u];

  const uint32_t d0_t = LiveOne ? (d0 + (RegForm ? (tid * 4u) : tid))
                                : (d0 + (RegForm ? (tid * 4u) : tid) / eff_live);
  const uint32_t tok0_t = LiveOne ? 0u : ((RegForm ? (tid * 4u) : tid) % eff_live);

  if constexpr (RegForm) {
    const uint32_t step_d = LiveOne ? (nt * 4u) : ((nt * 4u) / eff_live);
    const uint32_t step_tok = LiveOne ? 0u : ((nt * 4u) - step_d * eff_live);
    const bool vec_blk = (!LiveOne) && (eff_live == nb) && ((nb & 3u) == 0u) &&
                         (((uintptr_t)block & 15u) == 0u);
    uint32_t d_g = d0_t, tok_g = tok0_t;
#pragma unroll
    for (uint32_t g = 0u; g < kRegChunks; ++g) {
      const size_t off = (size_t)g * (nt * 4u) + (size_t)tid * 4u;
      if (off + 3u < blk_elems) {
        if (vec_blk) {
          const float4 b4 = *(const float4*)(block + (size_t)d_g * nb + tok_g);
          if constexpr (Bf16Wire) {
            const uint16_t h0 = wmma::f32_to_bf16(b4.x);
            const uint16_t h1 = wmma::f32_to_bf16(b4.y);
            const uint16_t h2 = wmma::f32_to_bf16(b4.z);
            const uint16_t h3 = wmma::f32_to_bf16(b4.w);
            r_blk[g * 4u + 0u] = wmma::bf16_to_f32(h0);
            r_blk[g * 4u + 1u] = wmma::bf16_to_f32(h1);
            r_blk[g * 4u + 2u] = wmma::bf16_to_f32(h2);
            r_blk[g * 4u + 3u] = wmma::bf16_to_f32(h3);
            *(uint2*)(ps16 + idx0 + off) =
                make_uint2((uint32_t)h0 | ((uint32_t)h1 << 16),
                           (uint32_t)h2 | ((uint32_t)h3 << 16));
          } else {
            r_blk[g * 4u + 0u] = b4.x;
            r_blk[g * 4u + 1u] = b4.y;
            r_blk[g * 4u + 2u] = b4.z;
            r_blk[g * 4u + 3u] = b4.w;
            *(float4*)(ps + idx0 + off) = b4;
          }
        } else {
          uint32_t d = d_g, tok = tok_g;
#pragma unroll
          for (uint32_t j = 0u; j < 4u; ++j) {
            const float val = block[(size_t)d * nb + tok];
            if constexpr (Bf16Wire) {
              r_blk[g * 4u + j] = wmma::bf16_to_f32(wmma::f32_to_bf16(val));
            } else {
              r_blk[g * 4u + j] = val;
            }
            if constexpr (!LiveOne) {
              ++tok;
              if (tok == live) { tok = 0u; ++d; }
            } else {
              ++d;
            }
          }
          if constexpr (Bf16Wire) {
            const uint16_t h0 = wmma::f32_to_bf16(r_blk[g * 4u + 0u]);
            const uint16_t h1 = wmma::f32_to_bf16(r_blk[g * 4u + 1u]);
            const uint16_t h2 = wmma::f32_to_bf16(r_blk[g * 4u + 2u]);
            const uint16_t h3 = wmma::f32_to_bf16(r_blk[g * 4u + 3u]);
            *(uint2*)(ps16 + idx0 + off) =
                make_uint2((uint32_t)h0 | ((uint32_t)h1 << 16),
                           (uint32_t)h2 | ((uint32_t)h3 << 16));
          } else {
            *(float4*)(ps + idx0 + off) =
                make_float4(r_blk[g * 4u + 0u], r_blk[g * 4u + 1u],
                            r_blk[g * 4u + 2u], r_blk[g * 4u + 3u]);
          }
        }
      } else if (off < blk_elems) {
        uint32_t d = d_g, tok = tok_g;
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
          if (off + j < blk_elems) {
            const float val = block[(size_t)d * nb + tok];
            if constexpr (Bf16Wire) {
              const uint16_t h = wmma::f32_to_bf16(val);
              r_blk[g * 4u + j] = wmma::bf16_to_f32(h);
              ps16[idx0 + off + j] = h;
            } else {
              r_blk[g * 4u + j] = val;
              ps[idx0 + off + j] = val;
            }
          }
          if constexpr (!LiveOne) {
            ++tok;
            if (tok == live) { tok = 0u; ++d; }
          } else {
            ++d;
          }
        }
      }
      if constexpr (!LiveOne) {
        tok_g += step_tok;
        d_g += step_d;
        if (tok_g >= live) { tok_g -= live; ++d_g; }
      } else {
        d_g += nt * 4u;
      }
    }
  } else {
    const uint32_t step_d = LiveOne ? nt : (nt / live);
    const uint32_t step_tok = LiveOne ? 0u : (nt - step_d * live);
    uint32_t d = d0_t, tok = tok0_t;
    for (size_t idx = idx0 + tid; idx < idx1; idx += nt) {
      const float val = block[(size_t)d * nb + tok];
      if constexpr (Bf16Wire) ps16[idx] = wmma::f32_to_bf16(val);
      else                    ps[idx] = val;
      if constexpr (!LiveOne) {
        tok += step_tok;
        d += step_d;
        if (tok >= live) { tok -= live; ++d; }
      } else {
        d += nt;
      }
    }
  }

  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  aff::lds_barrier();

  if (tid == 0u) {
    const bool fast_rel = (ar_flags & kTpArFlagFastRel) != 0u;
    publish_flag(&peer_flags[b_blk], s, fast_rel);
    const unsigned int* const f = &my_flags[b_blk];
    unsigned long long z = 0;
    unsigned int fv = load_sys_relaxed(f);
    if ((int)(fv - s) < 0) {
      do {
        __builtin_amdgcn_s_sleep(1);
        fv = load_sys_relaxed(f);
      } while ((int)(fv - s) < 0 && ++z <= kTpSpinMax);
      if ((int)(fv - s) < 0) note_timeout(timeout, s);
    }
  }
  aff::lds_barrier();
  fence_sys_acq();

  constexpr uint32_t kPreRegToks = (kPreDpp && !FixedTok) ? (LiveOne ? 1u : 16u) : 1u;
  float r_red_cur[kPreRegToks];
  float r_red_hc[kPreRegToks];
  if constexpr (kPreDpp) {
#pragma unroll
    for (uint32_t b = 0u; b < kPreRegToks; ++b) {
      r_red_cur[b] = 0.0f;
      r_red_hc[b] = 0.0f;
    }
  }

  auto process_elem = [&](uint32_t d, uint32_t tok, float bv) __attribute__((always_inline)) {
    const size_t blk_idx = (size_t)d * nb + tok;
    if (zb) block[blk_idx] = 0.0f;
    if constexpr (HasControl) {
      float v[NH];
#pragma unroll
      for (uint32_t src = 0u; src < NH; ++src)
        v[src] = wmma::bf16_to_f32(hc[((size_t)src * n_embd + d) * nb + tok]);
      float tap_acc = 0.0f;
      float cval = 0.0f;
      float hsq = 0.0f;
#pragma unroll
      for (uint32_t dst = 0u; dst < NH; ++dst) {
        float acc = bv * post[(size_t)dst * nb + tok];
#pragma unroll
        for (uint32_t src = 0u; src < NH; ++src)
          acc += comb[(size_t)(dst + src * NH) * nb + tok] * v[src];
        const uint16_t h16 = wmma::f32_to_bf16(acc);
        out_hc[((size_t)dst * n_embd + d) * nb + tok] = h16;
        if constexpr (HasTap || HasPreCollapse) {
          const float hf = wmma::bf16_to_f32(h16);
          if constexpr (HasTap) tap_acc += hf;
          if constexpr (HasPreCollapse) {
            cval += hf * s_pre[dst * 16u + tok];
            hsq += hf * hf;
          }
        }
      }
      if constexpr (HasTap) {
        if (tok >= b_lo && tok < b_lo + n_keep) {
          const uint32_t row = row0 + (tok - b_lo);
          if (row < rows_cap) tap[(size_t)row * row_stride + slot_off + d] = tap_acc / (float)NH;
        }
      }
      if constexpr (HasPreCollapse) {
        out_cur[blk_idx] = cval;
        if constexpr (kPreDpp) {
          if constexpr (LiveOne) {
            r_red_cur[0] += cval * cval;
            r_red_hc[0] += hsq;
          } else {
#pragma unroll
            for (uint32_t b = 0u; b < 16u; ++b) {
              if (b == tok) {
                r_red_cur[b] += cval * cval;
                r_red_hc[b] += hsq;
              }
            }
          }
        } else {
          s_red_cur[tok * 256u + tid] += cval * cval;
          s_red_hc[tok * 256u + tid] += hsq;
        }
      }
    } else {
      out_hc[blk_idx] = wmma::f32_to_bf16(wmma::bf16_to_f32(out_hc[blk_idx]) + bv);
    }
  };

  if constexpr (FixedTok) {
    // Fixed token-to-lane mapping (`my_tok = tid & 15`, 16 dims/step): hoists post/comb/pre_in
    // into registers once per thread and reduces with 2 scalar accumulators + one DPP xor<16>.
    const uint32_t my_tok = tid & 15u;
    const bool tok_live = (my_tok < eff_live);
    const bool tok_pad = (!tok_live) && (my_tok < nb);
    constexpr bool kPreloadComb = (NH <= 4u);
    float l_post[HasControl ? NH : 1u];
    float l_pre[HasControl ? NH : 1u];
    float l_comb[(HasControl && kPreloadComb) ? (NH * NH) : 1u];
    if constexpr (HasControl) {
      if (tok_live) {
#pragma unroll
        for (uint32_t dst = 0u; dst < NH; ++dst) {
          l_post[dst] = post[(size_t)dst * nb + my_tok];
          l_pre[dst] = s_pre[dst * 16u + my_tok];
          if constexpr (kPreloadComb) {
#pragma unroll
            for (uint32_t src = 0u; src < NH; ++src)
              l_comb[dst * NH + src] = comb[(size_t)(dst + src * NH) * nb + my_tok];
          }
        }
      }
    }
    for (uint32_t d = d0 + (tid >> 4); d < d1; d += 16u) {
      const size_t blk_idx = (size_t)d * nb + my_tok;
      if (tok_live) {
        const size_t ms_idx = idx0 + (size_t)(d - d0) * eff_live + my_tok;
        const float bv = block[blk_idx] + ms[ms_idx];
        if (zb) block[blk_idx] = 0.0f;
        float v[NH];
#pragma unroll
        for (uint32_t src = 0u; src < NH; ++src)
          v[src] = wmma::bf16_to_f32(hc[((size_t)src * n_embd + d) * nb + my_tok]);
        float tap_acc = 0.0f, cval = 0.0f, hsq = 0.0f;
#pragma unroll
        for (uint32_t dst = 0u; dst < NH; ++dst) {
          float acc = bv * l_post[dst];
#pragma unroll
          for (uint32_t src = 0u; src < NH; ++src) {
            const float cw = kPreloadComb
                                 ? l_comb[dst * NH + src]
                                 : comb[(size_t)(dst + src * NH) * nb + my_tok];
            acc += cw * v[src];
          }
          const uint16_t h16 = wmma::f32_to_bf16(acc);
          out_hc[((size_t)dst * n_embd + d) * nb + my_tok] = h16;
          const float hf = wmma::bf16_to_f32(h16);
          if constexpr (HasTap) tap_acc += hf;
          cval += hf * l_pre[dst];
          hsq += hf * hf;
        }
        if constexpr (HasTap) {
          if (my_tok >= b_lo && my_tok < b_lo + n_keep) {
            const uint32_t row = row0 + (my_tok - b_lo);
            if (row < rows_cap) tap[(size_t)row * row_stride + slot_off + d] = tap_acc / (float)NH;
          }
        }
        out_cur[blk_idx] = cval;
        r_red_cur[0] += cval * cval;
        r_red_hc[0] += hsq;
      } else if (tok_pad) {
        if (zb) block[blk_idx] = 0.0f;
#pragma unroll
        for (uint32_t dst = 0u; dst < NH; ++dst)
          out_hc[((size_t)dst * n_embd + d) * nb + my_tok] = 0u;
        out_cur[blk_idx] = 0.0f;
      }
    }
  } else if constexpr (RegForm) {
    const uint32_t step_d = LiveOne ? (nt * 4u) : ((nt * 4u) / eff_live);
    const uint32_t step_tok = LiveOne ? 0u : ((nt * 4u) - step_d * eff_live);
    uint32_t d_g = d0_t, tok_g = tok0_t;
#pragma unroll
    for (uint32_t g = 0u; g < kRegChunks; ++g) {
      const size_t off = (size_t)g * (nt * 4u) + (size_t)tid * 4u;
      if (off + 3u < blk_elems) {
        float m_arr[4];
        if constexpr (Bf16Wire) {
          const uint2 m2 = *(const uint2*)(ms16 + idx0 + off);
          m_arr[0] = wmma::bf16_to_f32((uint16_t)(m2.x & 0xFFFFu));
          m_arr[1] = wmma::bf16_to_f32((uint16_t)(m2.x >> 16));
          m_arr[2] = wmma::bf16_to_f32((uint16_t)(m2.y & 0xFFFFu));
          m_arr[3] = wmma::bf16_to_f32((uint16_t)(m2.y >> 16));
        } else {
          const float4 m4 = *(const float4*)(ms + idx0 + off);
          m_arr[0] = m4.x;
          m_arr[1] = m4.y;
          m_arr[2] = m4.z;
          m_arr[3] = m4.w;
        }
        uint32_t d = d_g, tok = tok_g;
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
          process_elem(d, tok, r_blk[g * 4u + j] + m_arr[j]);
          if constexpr (!LiveOne) {
            ++tok;
            if (tok == eff_live) { tok = 0u; ++d; }
          } else {
            ++d;
          }
        }
      } else if (off < blk_elems) {
        uint32_t d = d_g, tok = tok_g;
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
          if (off + j < blk_elems) {
            const float mv = Bf16Wire ? wmma::bf16_to_f32(ms16[idx0 + off + j]) : ms[idx0 + off + j];
            process_elem(d, tok, r_blk[g * 4u + j] + mv);
          }
          if constexpr (!LiveOne) {
            ++tok;
            if (tok == eff_live) { tok = 0u; ++d; }
          } else {
            ++d;
          }
        }
      }
      if constexpr (!LiveOne) {
        tok_g += step_tok;
        d_g += step_d;
        if (tok_g >= eff_live) { tok_g -= eff_live; ++d_g; }
      } else {
        d_g += nt * 4u;
      }
    }
  } else {
    const uint32_t step_d = LiveOne ? nt : (nt / eff_live);
    const uint32_t step_tok = LiveOne ? 0u : (nt - step_d * eff_live);
    uint32_t d = d0_t, tok = tok0_t;
    for (size_t idx = idx0 + tid; idx < idx1; idx += nt) {
      const float loc = block[(size_t)d * nb + tok];
      const float bv = Bf16Wire
          ? (wmma::bf16_to_f32(wmma::f32_to_bf16(loc)) + wmma::bf16_to_f32(ms16[idx]))
          : (loc + ms[idx]);
      process_elem(d, tok, bv);
      if constexpr (!LiveOne) {
        tok += step_tok;
        d += step_d;
        if (tok >= eff_live) { tok -= eff_live; ++d; }
      } else {
        d += nt;
      }
    }
  }

  if (!FixedTok && live < nb) {
    const uint32_t pad = nb - live;
    const size_t ptot = (size_t)(d1 - d0) * pad;
    const uint32_t pstep_d = nt / pad;
    const uint32_t pstep_tok = nt - pstep_d * pad;
    uint32_t pd = d0 + tid / pad;
    uint32_t ptok = tid % pad;
    for (size_t pidx = tid; pidx < ptot; pidx += nt) {
      const uint32_t tok = live + ptok;
      if (zb) block[(size_t)pd * nb + tok] = 0.0f;
      if constexpr (HasControl) {
#pragma unroll
        for (uint32_t dst = 0u; dst < NH; ++dst)
          out_hc[((size_t)dst * n_embd + pd) * nb + tok] = 0u;
        if constexpr (HasPreCollapse) out_cur[(size_t)pd * nb + tok] = 0.0f;
      } else {
        out_hc[(size_t)pd * nb + tok] = 0u;
      }
      ptok += pstep_tok;
      pd += pstep_d;
      if (ptok >= pad) { ptok -= pad; ++pd; }
    }
  }

  if constexpr (HasPreCollapse) {
    if constexpr (kPreDpp) {
      const uint32_t lane = tid & 31u, wave = tid >> 5;
      if constexpr (LiveOne) {
        const float w_hc  = aff::wave_reduce_add_dpp(r_red_hc[0]);
        const float w_cur = aff::wave_reduce_add_dpp(r_red_cur[0]);
        if (lane == 0u) {
          s_dpp[0][0][wave] = w_hc;
          s_dpp[1][0][wave] = w_cur;
        }
        aff::lds_barrier();
        if (wave == 0u) {
          const uint32_t kind = lane >> 3;
          const uint32_t w = lane & 7u;
          float v = (kind < 2u) ? s_dpp[kind][0][w] : 0.0f;
          // Exact 3-level binary tree across 8 waves (matching st = 128 -> 64 -> 32).
          v += aff::dpp_xor_lane<4u>(v);
          v += aff::dpp_xor_lane<2u>(v);
          v += aff::dpp_xor_lane<1u>(v);
          if (lane == 0u) out_part[(size_t)b_blk * 16u] = v;
          if (lane == 8u) out_part[(size_t)(32u + b_blk) * 16u] = v;
          if (lane >= 1u && lane < 16u) {
            out_part[(size_t)b_blk * 16u + lane] = 0.0f;
            out_part[(size_t)(32u + b_blk) * 16u + lane] = 0.0f;
          }
        }
      } else {
        if constexpr (FixedTok) {
          const float r_hc = r_red_hc[0] + aff::dpp_xor_lane<16u>(r_red_hc[0]);
          const float r_cur = r_red_cur[0] + aff::dpp_xor_lane<16u>(r_red_cur[0]);
          if (lane < eff_live) {
            s_dpp[0][lane][wave] = r_hc;
            s_dpp[1][lane][wave] = r_cur;
          }
        } else {
          // Constant-bound unroll keeps r_red_cur/r_red_hc in registers (zero scratch); eff_live is
          // uniform across the block so every wave enters wave_reduce_add_dpp together.
#pragma unroll
          for (uint32_t b = 0u; b < 16u; ++b) {
            if (b < eff_live) {
              const float w_hc  = aff::wave_reduce_add_dpp(r_red_hc[b]);
              const float w_cur = aff::wave_reduce_add_dpp(r_red_cur[b]);
              if (lane == 0u) {
                s_dpp[0][b][wave] = w_hc;
                s_dpp[1][b][wave] = w_cur;
              }
            }
          }
        }
        aff::lds_barrier();
        if (wave == 0u) {
          const uint32_t grp = lane >> 3;
          const uint32_t w   = lane & 7u;
          for (uint32_t b0 = 0u; b0 < eff_live; b0 += 4u) {
            const uint32_t b = b0 + grp;
            float v_hc  = (b < eff_live) ? s_dpp[0][b][w] : 0.0f;
            float v_cur = (b < eff_live) ? s_dpp[1][b][w] : 0.0f;
            v_hc  += aff::dpp_xor_lane<4u>(v_hc);
            v_hc  += aff::dpp_xor_lane<2u>(v_hc);
            v_hc  += aff::dpp_xor_lane<1u>(v_hc);
            v_cur += aff::dpp_xor_lane<4u>(v_cur);
            v_cur += aff::dpp_xor_lane<2u>(v_cur);
            v_cur += aff::dpp_xor_lane<1u>(v_cur);
            if (w == 0u && b < eff_live) {
              out_part[(size_t)b_blk * 16u + b] = v_hc;
              out_part[(size_t)(32u + b_blk) * 16u + b] = v_cur;
            }
          }
          if (lane >= eff_live && lane < 16u) {
            out_part[(size_t)b_blk * 16u + lane] = 0.0f;
            out_part[(size_t)(32u + b_blk) * 16u + lane] = 0.0f;
          }
        }
      }
    } else {
      aff::lds_barrier();
      for (uint32_t st = 128u; st > 0u; st >>= 1) {
        if (tid < st) {
          for (uint32_t b = 0u; b < eff_live; ++b) {
            s_red_cur[b * 256u + tid] += s_red_cur[b * 256u + tid + st];
            s_red_hc[b * 256u + tid]  += s_red_hc[b * 256u + tid + st];
          }
        }
        aff::lds_barrier();
      }
      if (tid < 16u) {
        const float v_hc  = (tid < eff_live) ? s_red_hc[tid * 256u]  : 0.0f;
        const float v_cur = (tid < eff_live) ? s_red_cur[tid * 256u] : 0.0f;
        out_part[(size_t)b_blk * 16u + tid] = v_hc;
        out_part[(size_t)(32u + b_blk) * 16u + tid] = v_cur;
      }
    }
  }
}

} // namespace

// Sub-slots a rank keeps: one per peer, none for itself. Zero at one rank, which is why every
// allocation below is guarded rather than assumed non-empty.
static inline size_t tp_npeer(const TpContext* ctx) { return ctx->ranks ? ctx->ranks - 1 : 0; }

static inline bool tp_scratch_fp8_enabled() {
  const char* v = std::getenv("AFF_TP_SCRATCH_FP8");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

// Floats one double-buffer slot spans on one rank: every peer's message, back to back.
static inline size_t tp_slot_floats(const TpContext* ctx) {
  return tp_npeer(ctx) * (ctx->scratch_peer_bytes / sizeof(float));
}

bool tp_grow(TpContext* ctx, size_t max_floats, std::string* err) {
  if (!ctx->ready) return false;
  const size_t want = (max_floats + 3) / 4 * 4;
  size_t want_bytes = want * sizeof(float);
  if (tp_scratch_fp8_enabled() && want_bytes >= kTpFp8MinBytes) {
    // Compact prefill scratch to FP8 code space + 128-elem block scales (16-byte aligned).
    // Exact/BF16 small-batch paths (decode/verify) check scratch_peer_bytes directly, and
    // oversized Exact/Coded fallbacks slice along scratch_peer_bytes.
    const size_t code_bytes = (want + 15u) & ~size_t(15u);
    const size_t n_groups = (want + kTpFp8Group - 1u) / kTpFp8Group;
    const size_t fp8_bytes = (code_bytes + n_groups * sizeof(float) + 15u) & ~size_t(15u);
    want_bytes = std::min(want_bytes, fp8_bytes);
  }
  want_bytes = std::max(ctx->scratch_peer_bytes, want_bytes);
  if (want <= ctx->max_floats && want_bytes <= ctx->scratch_peer_bytes) return true;
  const size_t np = tp_npeer(ctx);
  if (!np) {
    ctx->max_floats = std::max(ctx->max_floats, want);
    ctx->scratch_peer_bytes = want_bytes;
    return true;
  }
  if (want_bytes <= ctx->scratch_peer_bytes) {
    ctx->max_floats = std::max(ctx->max_floats, want);
    return true;
  }
  for (size_t r = 0; r < ctx->ranks; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    if (!hip_try(hipStreamSynchronize((hipStream_t)ctx->stream[r]), "drain before grow", err))
      return false;
    (void)hipFree(ctx->scratch[r]);
    ctx->scratch[r] = nullptr;
    void* sc = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&sc, TpContext::kTpSlots * np * want_bytes,
                                       hipDeviceMallocFinegrained),
                 "scratch regrow", err))
      return false;
    ctx->scratch[r] = (float*)sc;
  }
  ctx->max_floats = std::max(ctx->max_floats, want);
  ctx->scratch_peer_bytes = want_bytes;
  ctx->last_scratch_nb = 0;
  return true;
}

bool tp_fold_prepare(TpContext* ctx, size_t floats, std::string* err) {
  if (!ctx->ready) { if (err) *err = "tp not ready"; return false; }
  if (!floats) { if (err) *err = "zero-length fold"; return false; }
  // Two ranks only. The kernels that consume these buffers embed the single-peer protocol inside a
  // kernel doing other work, and there is no N-peer form of them — see the header. Refusing here is
  // how the callers learn to take the standalone collective instead, and it is not an error.
  if (ctx->ranks != 2) {
    if (err) *err = "the folded exchange is a two-rank path; this run has " +
                    std::to_string(ctx->ranks);
    return false;
  }
  const size_t want = (floats + 3) / 4 * 4;               // float4 alignment, as tp_init does
  if (want <= ctx->fold_floats) return true;
  for (int r = 0; r < 2; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    if (!hip_try(hipStreamSynchronize((hipStream_t)ctx->stream[r]), "drain before fold alloc", err))
      return false;
    (void)hipFree(ctx->fscratch[r]);
    ctx->fscratch[r] = nullptr;
    void* sc = nullptr;
    // Fine-grained for the same reason the main scratch is: the push has to POST to the fabric
    // rather than rest in this card's L2, where the peer would never see it.
    if (!hip_try(hipExtMallocWithFlags(&sc, TpContext::kTpSlots * want * sizeof(float),
                                       hipDeviceMallocFinegrained),
                 "fold scratch alloc", err))
      return false;
    ctx->fscratch[r] = (float*)sc;
    if (!ctx->fflags[r]) {
      void* fl = nullptr;
      if (!hip_try(hipExtMallocWithFlags(&fl, kTpMaxBlocks * sizeof(unsigned) + kTpWakeBytes,
                                         hipDeviceMallocFinegrained), "fold flags alloc", err))
        return false;
      ctx->fflags[r] = (unsigned*)fl;
      if (!hip_try(hipMemset(ctx->fflags[r], 0, kTpMaxBlocks * sizeof(unsigned) + kTpWakeBytes),
                   "memset fold flags", err))
        return false;
      if (!hip_try(hipMalloc(&ctx->fseq[r], kTpMaxBlocks * sizeof(unsigned)), "fold seq alloc", err))
        return false;
      if (!hip_try(hipMemset(ctx->fseq[r], 0, kTpMaxBlocks * sizeof(unsigned)), "memset fold seq",
                   err))
        return false;
    }
  }
  ctx->fold_floats = want;
  return true;
}

bool tp_init(TpContext* ctx, size_t max_floats, std::string* err, ihipStream_t* const* ext,
             size_t n_ranks, const int* devs_in) {
  static_assert(kTpMaxRanks == kMaxRanks, "the collective and the engine must size ranks alike");
  std::vector<int> owned;
  if (!devs_in || !n_ranks) {
    owned = gfx12_devices();
    if (owned.empty()) { if (err) *err = "no gfx12 device for TP"; return false; }
    n_ranks = owned.size();
    devs_in = owned.data();
  }
  if (n_ranks > kTpMaxRanks) {
    if (err) *err = std::to_string(n_ranks) + " ranks exceeds kTpMaxRanks";
    return false;
  }
  ctx->ranks = n_ranks;
  for (size_t r = 0; r < n_ranks; ++r) ctx->dev[r] = devs_in[r];
  ctx->max_floats = (max_floats + 3) / 4 * 4;          // float4 alignment
  ctx->scratch_peer_bytes = ctx->max_floats * sizeof(float);
  ctx->own_stream = (ext == nullptr);
  const size_t np = tp_npeer(ctx);

  for (size_t r = 0; r < n_ranks; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    // Peer access is what removes the IPC layer: a kernel here can write straight into another
    // card's allocation. Already-enabled is not an error, and neither is a peer that IS this device
    // — under AFF_LOGICAL_RANKS two ranks share a card, and a push between them is a local store
    // that needs no peer mapping at all.
    for (size_t q = 0; q < n_ranks; ++q) {
      if (q == r || ctx->dev[q] == ctx->dev[r]) continue;
      const hipError_t pe = hipDeviceEnablePeerAccess(ctx->dev[q], 0);
      if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
        // Not every pair of cards in a machine can reach each other. Say which pair, because the
        // fix is a topology change (a different slot, or HIP_VISIBLE_DEVICES) and not a flag.
        if (err) *err = "hipDeviceEnablePeerAccess from device " + std::to_string(ctx->dev[r]) +
                        " to " + std::to_string(ctx->dev[q]) + ": " + hipGetErrorString(pe);
        return false;
      }
      // Tolerating the error is not the same as clearing it. hipGetLastError is STICKY, so an
      // already-enabled peer link reported here surfaced at the next kernel launch check instead —
      // "peer access is already enabled" as the reason a batch_hc_seed_kernel launch failed, which
      // sends you looking at a kernel that is perfectly fine.
      (void)hipGetLastError();
    }
    if (ext) ctx->stream[r] = ext[r];
    else if (!hip_try(hipStreamCreate((hipStream_t*)&ctx->stream[r]), "hipStreamCreate", err))
      return false;

    // One rank has no peers, so no scratch and no flags: tp_all_reduce is the identity there and
    // never reaches a kernel.
    if (!np) continue;

    // Fine-grained => uncached => stores post to the fabric rather than resting in L2.
    void* sc = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&sc,
                                       TpContext::kTpSlots * np * ctx->max_floats * sizeof(float),
                                       hipDeviceMallocFinegrained), "scratch alloc", err))
      return false;
    ctx->scratch[r] = (float*)sc;

    void* fl = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&fl, np * kTpFlagStride * sizeof(unsigned) + kTpWakeBytes,
                                       hipDeviceMallocFinegrained), "flags alloc", err))
      return false;
    ctx->flags[r] = (unsigned*)fl;
    if (!hip_try(hipMemset(ctx->flags[r], 0, np * kTpFlagStride * sizeof(unsigned) + kTpWakeBytes),
                 "memset flags", err))
      return false;

    if (!hip_try(hipMalloc(&ctx->seq[r], kTpMaxBlocks * sizeof(unsigned)), "seq alloc", err))
      return false;
    if (!hip_try(hipMemset(ctx->seq[r], 0, kTpMaxBlocks * sizeof(unsigned)), "memset seq", err))
      return false;
  }
  // One word for every rank, in PINNED HOST memory so the host reads it with a load rather than a
  // copy — an instrument for a stall cannot itself need a synchronise. hipHostMalloc gives a pointer
  // every device can store to once peer access is up, which it is by here.
  if (!hip_try(hipHostMalloc((void**)&ctx->timeout, sizeof(unsigned), hipHostMallocMapped),
               "timeout alloc", err))
    return false;
  *ctx->timeout = 0u;
  if (!hip_try(hipHostGetDevicePointer((void**)&ctx->timeout_dev, ctx->timeout, 0),
               "timeout device pointer", err))
    return false;
  ctx->ready = true;
  return true;
}

void tp_reset_seq(TpContext& ctx) {
  for (size_t r = 0; r < ctx.ranks; ++r) {
    if (ctx.seq[r]) {
      set_device(ctx.dev[r]);
      // Zero the flags too: the protocol checks flag values against the sequence counter,
      // and a stale flag from a previous block could satisfy the wait prematurely.
      if (ctx.flags[r])
        AFF_HIP_CHECK(hipMemsetAsync(ctx.flags[r], 0,
                     (size_t)(ctx.ranks - 1) * kTpFlagStride * sizeof(unsigned),
                     (hipStream_t)ctx.stream[r]));
      set_device(ctx.dev[r]);
      AFF_HIP_CHECK(hipMemsetAsync(ctx.seq[r], 0, kTpMaxBlocks * sizeof(unsigned),
                                   (hipStream_t)ctx.stream[r]));
    }
    // fseq and fflags are 2-element arrays (fold path, 2 ranks only). Do not index past it.
    if (r < 2) {
      if (ctx.fflags[r]) {
        set_device(ctx.dev[r]);
        AFF_HIP_CHECK(hipMemsetAsync(ctx.fflags[r], 0, kTpMaxBlocks * sizeof(unsigned),
                                     (hipStream_t)ctx.stream[r]));
      }
      if (ctx.fseq[r]) {
        set_device(ctx.dev[r]);
        AFF_HIP_CHECK(hipMemsetAsync(ctx.fseq[r], 0, kTpMaxBlocks * sizeof(unsigned),
                                     (hipStream_t)ctx.stream[r]));
      }
    }
  }
  ctx.last_scratch_nb = 0;
}

void tp_check(const TpContext& ctx) {
  if (!ctx.ready || !ctx.timeout) return;
  const unsigned t = *ctx.timeout;
  if (!t) return;
  aff::ui::fatal("fatal: a rank gave up waiting for a peer in the all-reduce at collective %u and "
               "reduced against uninitialised scratch. Every token after that is wrong. This is a "
               "bug in the caller's ordering, not a slow peer: the spin bound is hours.\n", t);
  std::abort();
}

void tp_free(TpContext* ctx) {
  if (!ctx->ready) return;
  for (size_t r = 0; r < ctx->ranks; ++r) {
    (void)hipSetDevice(ctx->dev[r]);
    (void)hipFree(ctx->scratch[r]);
    (void)hipFree(ctx->flags[r]);
    (void)hipFree(ctx->seq[r]);
    ctx->scratch[r] = nullptr;
    ctx->flags[r] = nullptr;
    ctx->seq[r] = nullptr;
    if (r < 2) {
      (void)hipFree(ctx->fscratch[r]);
      (void)hipFree(ctx->fflags[r]);
      (void)hipFree(ctx->fseq[r]);
      ctx->fscratch[r] = nullptr;
      ctx->fflags[r] = nullptr;
      ctx->fseq[r] = nullptr;
    }
    if (ctx->own_stream) (void)hipStreamDestroy((hipStream_t)ctx->stream[r]);
    ctx->stream[r] = nullptr;
  }
  // Last: a rank that gave up is worth saying even on the way out, and after this the word is gone.
  tp_check(*ctx);
  (void)hipHostFree(ctx->timeout);
  ctx->timeout = nullptr;
  ctx->timeout_dev = nullptr;
  ctx->max_floats = 0;
  ctx->scratch_peer_bytes = 0;
  ctx->fold_floats = 0;
  ctx->last_scratch_nb = 0;
  ctx->ready = false;
}

namespace {

static inline bool tp_nrank_sync_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("AFF_TP_NRANK_SYNC");
    return v && *v && std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

static inline bool tp_ar_staged_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("AFF_TP_AR_STAGED");
    return !(v && *v && std::strcmp(v, "0") == 0);
  }();
  return enabled;
}

static inline bool tp_ar_staged_cache_enabled() {
  const char* v = std::getenv("AFF_TP_AR_STAGED_CACHE");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

static inline bool tp_ar_fast_rel_enabled() {
  const char* v = std::getenv("AFF_AR_FAST_REL");
  return v && std::strcmp(v, "1") == 0;
}

static inline bool tp_ar_wake_store_enabled() {
  const char* v = std::getenv("AFF_AR_WAKE_STORE");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

static inline bool tp_ar_reg_form_enabled() {
  const char* v = std::getenv("AFF_AR_REG_FORM");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

static inline bool tp_ar_hc_bf16_enabled() {
  const char* v = std::getenv("AFF_AR_HC_BF16");
  return v && *v && std::strcmp(v, "0") != 0;
}

static inline bool tp_hc_dpp_enabled() {
  const char* v = std::getenv("AFF_TP_HC_DPP");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

static inline bool tp_ar_coded_enabled() {
  const char* v = std::getenv("AFF_TP_AR_CODED");
  return v && *v && std::strcmp(v, "0") != 0;
}

static inline int tp_ar_fixed_blk_cfg() {
  const char* v = std::getenv("AFF_AR_FIXED_BLK");
  if (!v || !*v) return 32;
  const int b = std::atoi(v);
  if (b <= 0) return 0;
  return b > kTpMaxBlocks ? kTpMaxBlocks : b;
}

static inline bool tp_draft_proj_fixed_ar_enabled() {
  const char* v = std::getenv("AFF_DRAFT_PROJ_FIXED_AR");
  return !(v && *v && std::strcmp(v, "0") == 0);
}

static inline size_t tp_ar_prefetch_max_bytes() {
  const char* v = std::getenv("AFF_PREFETCH_MB");
  if (!v || !*v) return (size_t)8u << 20;
  const long mb = std::strtol(v, nullptr, 10);
  return mb > 0 ? ((size_t)mb << 20) : 0u;
}

static inline uint32_t tp_ar_flags_cfg() {
  uint32_t f = 0u;
  if (tp_ar_fast_rel_enabled()) f |= kTpArFlagFastRel;
  if (tp_ar_wake_store_enabled()) f |= kTpArFlagWakeEarly;
  return f;
}

static inline void tp_sync_on_grid_change(TpContext& ctx, int nb) {
  if (ctx.ranks > 1 && ctx.last_scratch_nb != 0 && ctx.last_scratch_nb != nb) {
    for (size_t r = 0; r < ctx.ranks; ++r) {
      set_device(ctx.dev[r]);
      AFF_HIP_CHECK(hipStreamSynchronize((hipStream_t)ctx.stream[r]));
    }
  }
  ctx.last_scratch_nb = nb;
}

static inline int tp_ar_fp8_stages_cfg() {
  static const int stages = [] {
    const char* v = std::getenv("AFF_TP_AR_STAGES");
    if (!v || !*v) return kTpFp8DefaultStages;
    const int s = std::atoi(v);
    if (s <= 1) return 1;
    return s > kTpFp8MaxStages ? kTpFp8MaxStages : s;
  }();
  return stages;
}

// One launch per rank, with that rank's peer table filled in. NPEER is a template argument so the
// two-rank path keeps the straight-line kernel it had; the switch below is the only place the live
// rank count meets it.
template <int NPEER>
void tp_launch_exact(TpContext& ctx, float* const* in, float* const* out, int n4, int nb, int nt,
                     int slot4, int peer4) {
  const bool tp_nrank_sync = tp_nrank_sync_enabled();
  const bool reg_cache = (NPEER == 1) && tp_ar_reg_form_enabled() &&
                         ((size_t)n4 <= (size_t)nb * (size_t)nt * 8u);
  const uint32_t ar_flags = tp_ar_flags_cfg();
  const size_t np = ctx.ranks - 1;
  for (size_t r = 0; r < ctx.ranks; ++r) {
    TpPush<NPEER> push{};
    for (size_t q = 0, j = 0; q < ctx.ranks; ++q) {
      if (q == r) continue;
      const size_t mine = tp_peer_slot(r, q);      // where rank r sits in rank q's tables
      push.scratch[j] = (float4*)((unsigned char*)ctx.scratch[q] + mine * ctx.scratch_peer_bytes);
      push.flags[j] = ctx.flags[q] + mine * kTpFlagStride;
      push.wake[j] = ctx.flags[q] + np * kTpFlagStride;
      ++j;
    }
    set_device(ctx.dev[r]);
    if (reg_cache) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER, false, true>), dim3(nb), dim3(nt), 0,
                         (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                         (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n4, slot4, peer4, (int)r, ar_flags);
    } else if (tp_nrank_sync && NPEER > 1) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER, true, false>), dim3(nb), dim3(nt), 0,
                         (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                         (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n4, slot4, peer4, (int)r, ar_flags);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER, false, false>), dim3(nb), dim3(nt), 0,
                         (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                         (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n4, slot4, peer4, (int)r, ar_flags);
    }
    AFF_HIP_CHECK(hipGetLastError());
  }
}

template <int NPEER>
void tp_launch_fp8(TpContext& ctx, float* const* in, float* const* out, int n, int nb,
                   int slot_bytes, int peer_bytes, int scale_off) {
  const bool tp_nrank_sync = tp_nrank_sync_enabled();
  const uint32_t ar_flags = tp_ar_flags_cfg();
  const size_t np = ctx.ranks - 1;
  const int n_groups = (n + kTpFp8Group - 1) / kTpFp8Group;
  const int gw = (n_groups + nb - 1) / nb;
  const bool cache_regs = (gw <= 8 * kTpFp8RegCacheGroups);
  if constexpr (NPEER == 1) {
    if (tp_ar_staged_enabled() && (size_t)n * sizeof(float) >= kTpStagedMinBytes) {
      const int stages = std::min(tp_ar_fp8_stages_cfg(), gw);
      if (stages >= 2) {
        const bool cache_stage = tp_ar_staged_cache_enabled();
        for (size_t r = 0; r < ctx.ranks; ++r) {
          TpPush8<1> push{};
          const size_t q = 1u - r;
          push.base[0] = (unsigned char*)ctx.scratch[q];
          push.flags[0] = ctx.flags[q];
          push.wake[0] = ctx.flags[q] + kTpFlagStride;
          set_device(ctx.dev[r]);
          if (cache_stage) {
            hipLaunchKernelGGL((tp_ar_fp8_staged_kernel<true>), dim3(nb), dim3(256), 0,
                               (hipStream_t)ctx.stream[r], in[r], push,
                               (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r],
                               ctx.seq[r], ctx.timeout_dev, n, slot_bytes, scale_off, stages,
                               ar_flags);
          } else {
            hipLaunchKernelGGL((tp_ar_fp8_staged_kernel<false>), dim3(nb), dim3(256), 0,
                               (hipStream_t)ctx.stream[r], in[r], push,
                               (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r],
                               ctx.seq[r], ctx.timeout_dev, n, slot_bytes, scale_off, stages,
                               ar_flags);
          }
          AFF_HIP_CHECK(hipGetLastError());
        }
        return;
      }
    }
  }
  for (size_t r = 0; r < ctx.ranks; ++r) {
    TpPush8<NPEER> push{};
    for (size_t q = 0, j = 0; q < ctx.ranks; ++q) {
      if (q == r) continue;
      const size_t mine = tp_peer_slot(r, q);
      push.base[j] = (unsigned char*)ctx.scratch[q] + mine * ctx.scratch_peer_bytes;
      push.flags[j] = ctx.flags[q] + mine * kTpFlagStride;
      push.wake[j] = ctx.flags[q] + np * kTpFlagStride;
      ++j;
    }
    set_device(ctx.dev[r]);
    if (tp_nrank_sync && NPEER > 1) {
      if (cache_regs)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, true, true>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r,
                           ar_flags);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, true, false>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r,
                           ar_flags);
    } else {
      if (cache_regs)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, false, true>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r,
                           ar_flags);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, false, false>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r,
                           ar_flags);
    }
    AFF_HIP_CHECK(hipGetLastError());
  }
}

void tp_launch_coded(TpContext& ctx, float* const* in, float* const* out, int n, int nb,
                     size_t slot_bytes, size_t base_off) {
  const uint32_t ar_flags = tp_ar_flags_cfg();
  const int n_groups = (n + kTpCodedGroup - 1) / kTpCodedGroup;
  const int gw = (n_groups + nb - 1) / nb;
  const bool cache_regs = (gw <= 8 * 4 * kTpCodedRegCacheGroups);
  if (tp_ar_staged_enabled() && (size_t)n * sizeof(float) >= kTpStagedMinBytes) {
    const int stages = std::min(tp_ar_fp8_stages_cfg(), gw);
    if (stages >= 2) {
      const bool cache_stage = tp_ar_staged_cache_enabled();
      for (size_t r = 0; r < 2; ++r) {
        TpPush8<1> push{};
        const size_t q = 1u - r;
        push.base[0] = (unsigned char*)ctx.scratch[q];
        push.flags[0] = ctx.flags[q];
        push.wake[0] = ctx.flags[q] + kTpFlagStride;
        set_device(ctx.dev[r]);
        if (cache_stage) {
          hipLaunchKernelGGL((tp_ar_coded_staged_kernel<true>), dim3(nb), dim3(256), 0,
                             (hipStream_t)ctx.stream[r], in[r], push,
                             (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r],
                             ctx.seq[r], ctx.timeout_dev, n, slot_bytes, base_off, stages,
                             ar_flags);
        } else {
          hipLaunchKernelGGL((tp_ar_coded_staged_kernel<false>), dim3(nb), dim3(256), 0,
                             (hipStream_t)ctx.stream[r], in[r], push,
                             (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r],
                             ctx.seq[r], ctx.timeout_dev, n, slot_bytes, base_off, stages,
                             ar_flags);
        }
        AFF_HIP_CHECK(hipGetLastError());
      }
      return;
    }
  }
  for (size_t r = 0; r < 2; ++r) {
    TpPush8<1> push{};
    const size_t q = 1u - r;
    push.base[0] = (unsigned char*)ctx.scratch[q];
    push.flags[0] = ctx.flags[q];
    push.wake[0] = ctx.flags[q] + kTpFlagStride;
    set_device(ctx.dev[r]);
    if (cache_regs) {
      hipLaunchKernelGGL((tp_ar_coded_kernel<true>), dim3(nb), dim3(256), 0,
                         (hipStream_t)ctx.stream[r], in[r], push,
                         (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n, slot_bytes, base_off, ar_flags);
    } else {
      hipLaunchKernelGGL((tp_ar_coded_kernel<false>), dim3(nb), dim3(256), 0,
                         (hipStream_t)ctx.stream[r], in[r], push,
                         (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n, slot_bytes, base_off, ar_flags);
    }
    AFF_HIP_CHECK(hipGetLastError());
  }
}

}  // namespace

void tp_all_reduce(TpContext& ctx, float* const* in, float* const* out, size_t n, TpReduce mode) {
  if (!ctx.ready || n == 0) return;
  const size_t np = tp_npeer(&ctx);
  if (!np) {
    // One rank: the sum over one term. A copy only when the caller did not alias, which the engine
    // always does — so in the shipping single-card configuration this costs nothing at all.
    if (in[0] != out[0]) {
      set_device(ctx.dev[0]);
      AFF_HIP_CHECK(hipMemcpyAsync(out[0], in[0], n * sizeof(float), hipMemcpyDeviceToDevice,
                                   (hipStream_t)ctx.stream[0]));
    }
    return;
  }
  if (mode == TpReduce::Auto) {
    const char* env_mode = std::getenv("AFF_TP_AR_MODE");
    if (env_mode && std::strcmp(env_mode, "exact") == 0) {
      mode = TpReduce::Exact;
    } else if (n * sizeof(float) >= kTpFp8MinBytes) {
      if (env_mode && std::strcmp(env_mode, "fp8") == 0) {
        mode = TpReduce::Fp8;
      } else if (env_mode && std::strcmp(env_mode, "coded") == 0) {
        mode = (ctx.ranks == 2) ? TpReduce::Coded : TpReduce::Fp8;
      } else {
        mode = (ctx.ranks == 2 && tp_ar_coded_enabled()) ? TpReduce::Coded : TpReduce::Fp8;
      }
    } else {
      mode = TpReduce::Exact;
    }
  }
  if (mode == TpReduce::Coded && ctx.ranks != 2) {
    mode = TpReduce::Fp8;
  }

  if (mode == TpReduce::Coded) {
    const size_t n_groups = (n + kTpCodedGroup - 1u) / kTpCodedGroup;
    const size_t base_off = n_groups * 128u;
    const size_t need_bytes = base_off + n_groups;
    const size_t slot_bytes = ctx.scratch_peer_bytes;
    if (need_bytes <= slot_bytes) {
      const int nn = (int)n;
      int nb = (int)((n_groups + 31u) / 32u);
      nb = std::max(1, std::min(nb, kTpMaxBlocks));
      tp_sync_on_grid_change(ctx, nb);
      tp_launch_coded(ctx, in, out, nn, nb, slot_bytes, base_off);
      return;
    }
    const size_t chunk_groups = slot_bytes / 129u;
    if (chunk_groups > 0u) {
      const size_t chunk_elems = chunk_groups * kTpCodedGroup;
      float* in_c[kTpMaxRanks];
      float* out_c[kTpMaxRanks];
      for (size_t off = 0; off < n; off += chunk_elems) {
        const size_t cur = std::min(chunk_elems, n - off);
        const size_t cur_groups = (cur + kTpCodedGroup - 1u) / kTpCodedGroup;
        const size_t cur_base_off = cur_groups * 128u;
        for (size_t r = 0; r < ctx.ranks; ++r) {
          in_c[r] = in[r] + off;
          out_c[r] = out[r] + off;
        }
        int nb = (int)((cur_groups + 31u) / 32u);
        nb = std::max(1, std::min(nb, kTpMaxBlocks));
        tp_sync_on_grid_change(ctx, nb);
        tp_launch_coded(ctx, in_c, out_c, (int)cur, nb, slot_bytes, cur_base_off);
      }
      return;
    }
    mode = TpReduce::Exact;
  }

  if (mode == TpReduce::Fp8) {
    const size_t peer_bytes = ctx.scratch_peer_bytes;
    const size_t slot_bytes = np * peer_bytes;
    const size_t n_groups_total = (n + kTpFp8Group - 1u) / kTpFp8Group;
    const size_t scale_off_total = (n + 15u) & ~size_t(15u);
    const size_t need_bytes = scale_off_total + n_groups_total * sizeof(float);
    auto launch_fp8_chunk = [&](float* const* in_p, float* const* out_p, size_t cur,
                                int scale_off) {
      const int nn = (int)cur;
      int nb = (int)(((cur + kTpFp8Group - 1u) / kTpFp8Group + 63u) / 64u);
      nb = std::max(1, std::min(nb, kTpMaxBlocks));
      tp_sync_on_grid_change(ctx, nb);
      switch (np) {
        case 1: tp_launch_fp8<1>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        case 2: tp_launch_fp8<2>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        case 3: tp_launch_fp8<3>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        case 4: tp_launch_fp8<4>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        case 5: tp_launch_fp8<5>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        case 6: tp_launch_fp8<6>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
        default: tp_launch_fp8<7>(ctx, in_p, out_p, nn, nb, (int)slot_bytes, (int)peer_bytes, scale_off); break;
      }
    };
    if (need_bytes <= peer_bytes) {
      launch_fp8_chunk(in, out, n, (int)scale_off_total);
      return;
    }
    const size_t max_chunk =
        (peer_bytes >= kTpFp8Group + sizeof(float))
            ? (peer_bytes / (kTpFp8Group + sizeof(float))) * kTpFp8Group
            : ((peer_bytes >= 20u) ? ((peer_bytes - sizeof(float)) & ~size_t(15u)) : 0u);
    if (max_chunk > 0u) {
      float* in_c[kTpMaxRanks];
      float* out_c[kTpMaxRanks];
      for (size_t off = 0; off < n; off += max_chunk) {
        const size_t cur = std::min(max_chunk, n - off);
        const int cur_scale_off = (int)((cur + 15u) & ~size_t(15u));
        for (size_t r = 0; r < ctx.ranks; ++r) {
          in_c[r] = in[r] + off;
          out_c[r] = out[r] + off;
        }
        launch_fp8_chunk(in_c, out_c, cur, cur_scale_off);
      }
      return;
    }
  }

  const size_t peer_bytes = ctx.scratch_peer_bytes;
  const size_t peer_floats4 = peer_bytes / sizeof(float4);
  if (!peer_floats4) return;
  const int peer4 = (int)peer_floats4;
  const int slot4 = (int)(np * peer_floats4);
  const int nt = 256;
  const size_t chunk_floats = peer_floats4 * 4u;
  auto launch_exact_chunk = [&](float* const* in_p, float* const* out_p, size_t cur) {
    const int n4 = (int)((cur + 3u) / 4u);
    // Block count from the payload. Small messages want ONE block: their cost is the handshake,
    // and extra blocks add flag traffic without adding transfer.
    int nb = (int)((n4 + kTpWordsPerBlock - 1) / kTpWordsPerBlock);
    const int fixed_blk = tp_ar_fixed_blk_cfg();
    if (np == 1u && fixed_blk > 0 && tp_draft_proj_fixed_ar_enabled() &&
        tp_ar_reg_form_enabled() && n4 >= kTpWordsPerBlock &&
        (size_t)n4 <= (size_t)fixed_blk * (size_t)nt * 8u) {
      // Pin draft-size crossings to AFF_AR_FIXED_BLK (32) so tp_sync_on_grid_change never stalls
      // both cards dropping 32 -> 15 -> 32 around DSpark draft_proj.
      nb = fixed_blk;
    } else {
      nb = std::max(1, std::min(nb, kTpMaxBlocks));
    }
    tp_sync_on_grid_change(ctx, nb);
    switch (np) {
      case 1: tp_launch_exact<1>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      case 2: tp_launch_exact<2>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      case 3: tp_launch_exact<3>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      case 4: tp_launch_exact<4>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      case 5: tp_launch_exact<5>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      case 6: tp_launch_exact<6>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
      default: tp_launch_exact<7>(ctx, in_p, out_p, n4, nb, nt, slot4, peer4); break;
    }
  };

  // Every rank's kernel must be IN FLIGHT together — each spins on the others' flags. Launching them
  // back to back on separate devices and streams achieves that; synchronising between the launches
  // would deadlock, and a blocking synchronise per all-reduce costs many times the crossing itself.
  if (n <= chunk_floats) {
    launch_exact_chunk(in, out, n);
  } else {
    float* in_c[kTpMaxRanks];
    float* out_c[kTpMaxRanks];
    for (size_t off = 0; off < n; off += chunk_floats) {
      const size_t cur = std::min(chunk_floats, n - off);
      for (size_t r = 0; r < ctx.ranks; ++r) {
        in_c[r] = in[r] + off;
        out_c[r] = out[r] + off;
      }
      launch_exact_chunk(in_c, out_c, cur);
    }
  }
}

inline bool tp_ar_fixed_tok_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("AFF_TP_AR_FIXED_TOK");
    return s && *s && !(s[0] == '0' && s[1] == '\0');
  }();
  return on;
}

bool tp_ar_batch_hc_expand(TpContext& ctx, uint16_t* const out_hc[kTpMaxRanks],
                           float* const block[kTpMaxRanks],
                           const uint16_t* const hc[kTpMaxRanks],
                           const float* const post[kTpMaxRanks],
                           const float* const comb[kTpMaxRanks], uint32_t n_embd,
                           uint32_t n_hc, uint32_t nb, uint32_t live, bool has_control,
                           int zb, float* const tap[kTpMaxRanks],
                           uint32_t b_lo, uint32_t n_keep, uint32_t row0,
                           uint32_t rows_cap, uint32_t row_stride, uint32_t slot_off,
                           const float* const pre_in[kTpMaxRanks],
                           float* const out_cur[kTpMaxRanks],
                           float* const out_part[kTpMaxRanks],
                           uint32_t* out_npart,
                           const void* const pf_ptr0[kTpMaxRanks],
                           size_t pf_bytes0,
                           const void* const pf_ptr1[kTpMaxRanks],
                           size_t pf_bytes1,
                           uint32_t pf_blocks) {
  if (!ctx.ready || ctx.ranks != 2 || !n_embd || !nb || !live || live > nb) return false;
  if (has_control && n_hc != 1u && n_hc != 2u && n_hc != 4u && n_hc != 8u) return false;
  const bool use_bf16_wire = tp_ar_hc_bf16_enabled() && (!has_control || n_hc == 4u);
  const size_t live_elems = (size_t)n_embd * live;
  const size_t wire_elem_bytes = use_bf16_wire ? sizeof(uint16_t) : sizeof(float);
  const size_t slot_floats = ctx.scratch_peer_bytes / sizeof(float);
  if (live_elems * wire_elem_bytes > ctx.scratch_peer_bytes || !slot_floats) return false;

  const bool has_pre_collapse =
      pre_in && pre_in[0] && pre_in[1] &&
      out_cur && out_cur[0] && out_cur[1] &&
      out_part && out_part[0] && out_part[1] &&
      has_control && nb == 16u;
  const bool hc_dpp = has_pre_collapse && tp_hc_dpp_enabled();
  const bool live_one = (live == 1u);
  const bool use_fixed_tok =
      tp_ar_fixed_tok_enabled() && hc_dpp && !live_one && nb == 16u && live <= 16u && !use_bf16_wire;
  const bool reg_form_env = tp_ar_reg_form_enabled();
  const int max_ar_blk = has_pre_collapse ? kTpHcnMaxBlocks : kTpMaxBlocks;
  const int fixed_blk = tp_ar_fixed_blk_cfg();
  int ar_blk;
  if (fixed_blk > 0 && reg_form_env && nb <= 16u && live <= 16u) {
    ar_blk = std::max(1, std::min(max_ar_blk, fixed_blk));
  } else if (reg_form_env && live <= 16u) {
    ar_blk = std::max(1, std::min(max_ar_blk,
                                  (int)((live_elems + 256u * 8u - 1u) / (256u * 8u))));
  } else {
    const size_t n4 = (live_elems + 3u) / 4u;
    ar_blk = std::max(1, std::min(max_ar_blk,
                                  (int)((n4 + kTpWordsPerBlock - 1u) / kTpWordsPerBlock)));
  }
  if (out_npart) *out_npart = has_pre_collapse ? (uint32_t)ar_blk : 0u;
  tp_sync_on_grid_change(ctx, ar_blk);

  constexpr uint32_t nt = 256u;
  constexpr uint32_t kRegCap = 12u;
  const uint32_t d_per_blk_raw = (n_embd + (uint32_t)ar_blk - 1u) / (uint32_t)ar_blk;
  const uint32_t d_per_blk_reg = (d_per_blk_raw + 3u) & ~3u;
  const bool use_reg_form =
      !use_fixed_tok && reg_form_env && ((size_t)d_per_blk_reg * live <= (size_t)nt * kRegCap);
  const bool has_tap = tap && tap[0] && tap[1] && n_keep > 0;
  const uint32_t ar_flags = tp_ar_flags_cfg();
  const size_t max_pf_bytes = (pf_blocks > 0u) ? tp_ar_prefetch_max_bytes() : 0u;

  auto clamp_pf = [](const void* p, size_t bytes, size_t budget,
                     const uint4*& out_p, size_t& out_n16) {
    out_p = nullptr;
    out_n16 = 0u;
    if (!p || bytes < 16u || budget < 16u) return;
    const size_t cap = std::min(bytes, budget);
    const uintptr_t addr = (uintptr_t)p;
    const size_t mis = (16u - (addr & 15u)) & 15u;
    if (cap <= mis) return;
    out_p = (const uint4*)(addr + mis);
    out_n16 = (cap - mis) >> 4;
  };

  for (size_t r = 0; r < 2; ++r) {
    const size_t q = 1u - r;
    float* const ps = ctx.scratch[q];
    const float* const ms = ctx.scratch[r];
    unsigned int* const pf = ctx.flags[q];
    unsigned int* const mf = ctx.flags[r];
    unsigned int* const pw = ctx.flags[q] + kTpFlagStride;
    float* const tap_r = has_tap ? tap[r] : nullptr;
    const float* const pre_r = has_pre_collapse ? pre_in[r] : nullptr;
    float* const cur_r = has_pre_collapse ? out_cur[r] : nullptr;
    float* const part_r = has_pre_collapse ? out_part[r] : nullptr;

    const uint4* pf_p0_r = nullptr;
    const uint4* pf_p1_r = nullptr;
    size_t pf_n16_0_r = 0u, pf_n16_1_r = 0u;
    if (max_pf_bytes >= 16u && (pf_bytes0 > 0u || pf_bytes1 > 0u)) {
      clamp_pf(pf_ptr0 ? pf_ptr0[r] : nullptr, pf_bytes0, max_pf_bytes, pf_p0_r, pf_n16_0_r);
      const size_t used = pf_n16_0_r << 4;
      const size_t rem_budget = (used < max_pf_bytes) ? (max_pf_bytes - used) : 0u;
      clamp_pf(pf_ptr1 ? pf_ptr1[r] : nullptr, pf_bytes1, rem_budget, pf_p1_r, pf_n16_1_r);
    }
    const uint32_t pf_blk_r = (pf_n16_0_r > 0u || pf_n16_1_r > 0u) ? pf_blocks : 0u;
    const uint32_t launch_blk = (uint32_t)ar_blk + pf_blk_r;

    set_device(ctx.dev[r]);
    const hipStream_t st = (hipStream_t)ctx.stream[r];
#define AFF_TP_BHC_EXP(NH_, TAP_, CTL_, ONE_, PRE_, REG_, BF16_, HCDPP_, FTOK_)                  \
  hipLaunchKernelGGL((tp_ar_batch_hc_expand_kernel<NH_, TAP_, CTL_, ONE_, PRE_, REG_, BF16_,      \
                                                   HCDPP_, FTOK_>),                               \
                     dim3(launch_blk), dim3(nt), 0, st, out_hc[r], block[r],                      \
                     hc ? hc[r] : nullptr, post ? post[r] : nullptr, comb ? comb[r] : nullptr,    \
                     n_embd, nb, live, zb, tap_r, b_lo, n_keep, row0, rows_cap, row_stride,       \
                     slot_off, ps, ms, pf, mf, pw, ctx.seq[r], ctx.timeout_dev, slot_floats,      \
                     (uint32_t)ar_blk, ar_flags, pre_r, cur_r, part_r, pf_p0_r, pf_n16_0_r,       \
                     pf_p1_r, pf_n16_1_r, pf_blk_r)
#define AFF_TP_BHC_REG(NH_, TAP_, CTL_, ONE_, PRE_, BF16_, HCDPP_)                                \
  do {                                                                                            \
    if (use_reg_form) AFF_TP_BHC_EXP(NH_, TAP_, CTL_, ONE_, PRE_, true, BF16_, HCDPP_, false);    \
    else              AFF_TP_BHC_EXP(NH_, TAP_, CTL_, ONE_, PRE_, false, BF16_, HCDPP_, false);   \
  } while (0)
#define AFF_TP_BHC_PRE(NH_, TAP_, ONE_, BF16_)                                                    \
  do {                                                                                            \
    if (has_pre_collapse) {                                                                       \
      if (hc_dpp) {                                                                               \
        if (!ONE_ && !BF16_ && use_fixed_tok) {                                                   \
          AFF_TP_BHC_EXP(NH_, TAP_, true, false, true, false, false, true, true);                 \
        } else {                                                                                  \
          AFF_TP_BHC_REG(NH_, TAP_, true, ONE_, true, BF16_, true);                               \
        }                                                                                         \
      } else {                                                                                    \
        AFF_TP_BHC_REG(NH_, TAP_, true, ONE_, true, BF16_, false);                                \
      }                                                                                           \
    } else {                                                                                      \
      AFF_TP_BHC_REG(NH_, TAP_, true, ONE_, false, BF16_, false);                                 \
    }                                                                                             \
  } while (0)
#define AFF_TP_BHC_TAP(NH_, ONE_, BF16_)                                                          \
  do {                                                                                            \
    if (has_tap) AFF_TP_BHC_PRE(NH_, true, ONE_, BF16_);                                          \
    else         AFF_TP_BHC_PRE(NH_, false, ONE_, BF16_);                                         \
  } while (0)
#define AFF_TP_BHC_ONE(NH_, BF16_)                                                                \
  do {                                                                                            \
    if (live_one) AFF_TP_BHC_TAP(NH_, true, BF16_);                                               \
    else          AFF_TP_BHC_TAP(NH_, false, BF16_);                                              \
  } while (0)
    if (!has_control) {
      if (use_bf16_wire) {
        if (live_one) AFF_TP_BHC_REG(4u, false, false, true, false, true, false);
        else          AFF_TP_BHC_REG(4u, false, false, false, false, true, false);
      } else {
        if (live_one) AFF_TP_BHC_REG(4u, false, false, true, false, false, false);
        else          AFF_TP_BHC_REG(4u, false, false, false, false, false, false);
      }
    } else if (n_hc == 4u) {
      if (use_bf16_wire) AFF_TP_BHC_ONE(4u, true);
      else               AFF_TP_BHC_ONE(4u, false);
    }
    else if (n_hc == 8u)   AFF_TP_BHC_ONE(8u, false);
    else if (n_hc == 2u)   AFF_TP_BHC_ONE(2u, false);
    else                   AFF_TP_BHC_ONE(1u, false);
#undef AFF_TP_BHC_ONE
#undef AFF_TP_BHC_TAP
#undef AFF_TP_BHC_PRE
#undef AFF_TP_BHC_REG
#undef AFF_TP_BHC_EXP
    AFF_HIP_CHECK(hipGetLastError());
  }
  return true;
}

inline bool prefill_fuse_ar_hc_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("AFF_PREFILL_FUSE_AR_HC");
    return !v || !*v || !(v[0] == '0' && v[1] == '\0');
  }();
  return on;
}

bool tp_ar_fp8_batch_hc_expand(TpContext& ctx, uint16_t* const out_hc[kTpMaxRanks],
                               float* const block[kTpMaxRanks],
                               const uint16_t* const hc[kTpMaxRanks],
                               const float* const post[kTpMaxRanks],
                               const float* const comb[kTpMaxRanks], uint32_t n_embd,
                               uint32_t n_hc, uint32_t nb, bool has_control, int zb) {
  if (!prefill_fuse_ar_hc_enabled()) return false;
  if (!ctx.ready || ctx.ranks != 2 || !n_embd || !nb || (nb & 3u) != 0u) return false;
  if (has_control && n_hc != 1u && n_hc != 2u && n_hc != 4u && n_hc != 8u) return false;

  const char* env_mode = std::getenv("AFF_TP_AR_MODE");
  if (env_mode && *env_mode && std::strcmp(env_mode, "fp8") != 0) return false;
  if ((!env_mode || !*env_mode) && tp_ar_coded_enabled()) return false;

  const size_t n_total = (size_t)n_embd * nb;
  if (n_total > (size_t)INT32_MAX || n_total * sizeof(float) < kTpFp8MinBytes) return false;

  if (!block || !block[0] || !block[1] || !out_hc || !out_hc[0] || !out_hc[1]) return false;
  if ((((uintptr_t)block[0] | (uintptr_t)block[1]) & 15u) != 0u) return false;
  if ((((uintptr_t)out_hc[0] | (uintptr_t)out_hc[1]) & 7u) != 0u) return false;
  if (has_control) {
    if (!hc || !hc[0] || !hc[1] || !post || !post[0] || !post[1] || !comb || !comb[0] || !comb[1])
      return false;
    if ((((uintptr_t)hc[0] | (uintptr_t)hc[1]) & 7u) != 0u) return false;
    if ((((uintptr_t)post[0] | (uintptr_t)post[1]) & 15u) != 0u) return false;
    if ((((uintptr_t)comb[0] | (uintptr_t)comb[1]) & 15u) != 0u) return false;
  }

  const size_t n_groups_total = (n_total + kTpFp8Group - 1u) / kTpFp8Group;
  const size_t scale_off_total = (n_total + 15u) & ~size_t(15u);
  const size_t need_bytes = scale_off_total + n_groups_total * sizeof(float);
  if (need_bytes > ctx.scratch_peer_bytes) return false;

  const int n = (int)n_total;
  const int n_groups = (int)n_groups_total;
  const int grid_nb = std::max(1, std::min((n_groups + 63) / 64, kTpMaxBlocks));
  const int gw = (n_groups + grid_nb - 1) / grid_nb;
  const int stages = (tp_ar_staged_enabled() && n_total * sizeof(float) >= kTpStagedMinBytes)
                         ? std::min(tp_ar_fp8_stages_cfg(), gw)
                         : 1;
  const bool cache_stage = tp_ar_staged_cache_enabled();
  const int slot_bytes = (int)ctx.scratch_peer_bytes;
  const int scale_off = (int)scale_off_total;
  const uint32_t ar_flags = tp_ar_flags_cfg();

  tp_sync_on_grid_change(ctx, grid_nb);

  for (size_t r = 0; r < 2; ++r) {
    TpPush8<1> push{};
    const size_t q = 1u - r;
    push.base[0] = (unsigned char*)ctx.scratch[q];
    push.flags[0] = ctx.flags[q];
    push.wake[0] = ctx.flags[q] + kTpFlagStride;
    set_device(ctx.dev[r]);
    const hipStream_t st = (hipStream_t)ctx.stream[r];
    const uint16_t* const hc_r = has_control ? hc[r] : nullptr;
    const float* const post_r = has_control ? post[r] : nullptr;
    const float* const comb_r = has_control ? comb[r] : nullptr;
#define AFF_TP_FP8_HC_LAUNCH(NH_, CTL_, CS_)                                                       \
  hipLaunchKernelGGL((tp_ar_fp8_hc_expand_kernel<NH_, CTL_, CS_>), dim3(grid_nb), dim3(256), 0,   \
                     st, out_hc[r], block[r], hc_r, post_r, comb_r, push,                          \
                     (const unsigned char*)ctx.scratch[r], ctx.flags[r], ctx.seq[r],               \
                     ctx.timeout_dev, n_embd, nb, n, slot_bytes, scale_off, stages, zb, ar_flags)
#define AFF_TP_FP8_HC_CS(NH_, CTL_)                                                                \
  do {                                                                                             \
    if (cache_stage) AFF_TP_FP8_HC_LAUNCH(NH_, CTL_, true);                                        \
    else             AFF_TP_FP8_HC_LAUNCH(NH_, CTL_, false);                                       \
  } while (0)
    if (!has_control) {
      AFF_TP_FP8_HC_CS(1u, false);
    } else if (n_hc == 4u) {
      AFF_TP_FP8_HC_CS(4u, true);
    } else if (n_hc == 8u) {
      AFF_TP_FP8_HC_CS(8u, true);
    } else if (n_hc == 2u) {
      AFF_TP_FP8_HC_CS(2u, true);
    } else {
      AFF_TP_FP8_HC_CS(1u, true);
    }
#undef AFF_TP_FP8_HC_CS
#undef AFF_TP_FP8_HC_LAUNCH
    AFF_HIP_CHECK(hipGetLastError());
  }
  return true;
}

void tp_sync(TpContext& ctx) {
  if (!ctx.ready) return;
  for (size_t r = 0; r < ctx.ranks; ++r) {
    set_device(ctx.dev[r]);
    AFF_HIP_CHECK(hipStreamSynchronize((hipStream_t)ctx.stream[r]));
  }
}

} // namespace aff

// amdnas-fixes launch-check: every hipLaunchKernelGGL in this file is followed by a launch-error check (2026-09-22).
