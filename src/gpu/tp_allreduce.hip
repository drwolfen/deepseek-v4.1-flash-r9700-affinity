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
#include "gpu/hip_common.h"
#include "gpu/wmma.h"

#include "ui/log.h"
#include <hip/hip_fp8.h>
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
constexpr size_t kTpStagedMinBytes = 4u << 20;  // 4 MiB floor for staged TP=2 FP8 overlap
constexpr int kTpFp8DefaultStages = 8;
constexpr int kTpFp8MaxStages = 8;
// Per-peer flag stride: [0, kTpMaxBlocks) holds unstaged block flags;
// [kTpMaxBlocks, (1 + kTpFp8MaxStages) * kTpMaxBlocks) holds per-(stage, block) flags.
constexpr size_t kTpFlagStride = (size_t)kTpMaxBlocks * (1u + (size_t)kTpFp8MaxStages);

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void store_sys_relaxed(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
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
template <int NPEER, bool NRANK_SYNC = false>
__global__ void tp_ar_kernel(const float4* __restrict__ in,
                             TpPush<NPEER> push, const float4* my_scratch,
                             float4* __restrict__ out,
                             unsigned int* my_flags,
                             unsigned int* seq_ctrs, unsigned int* timeout,
                             int n4, int slot_stride4, int peer_stride4, int own) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;

  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (b == 0 && tid == 0) {
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
  {
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
  __syncthreads();

  if constexpr (NRANK_SYNC && NPEER > 1) {
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
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
      for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
      unsigned long long z = 0;
#pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        const unsigned int* f = &my_flags[j * kTpFlagStride + b];
        unsigned int v = load_sys_relaxed(f);
        if (v < s) {
          do {
            __builtin_amdgcn_s_sleep(1);
            v = load_sys_relaxed(f);
          } while (v < s && ++z <= kTpSpinMax);
          if (v < s) note_timeout(timeout, s);
        }
      }
    }
  }
  __syncthreads();
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
      // The two-rank straight line, kept verbatim: one peer, no source table, no select.
      for (; k + 3 * nt < end; k += 4 * nt) {
        const float4 a0 = in[k], a1 = in[k + nt], a2 = in[k + 2 * nt], a3 = in[k + 3 * nt];
        const float4 b0 = ms[0][k], b1 = ms[0][k + nt];
        const float4 b2 = ms[0][k + 2 * nt], b3 = ms[0][k + 3 * nt];
        out[k]          = f4add(a0, b0); out[k + nt]     = f4add(a1, b1);
        out[k + 2 * nt] = f4add(a2, b2); out[k + 3 * nt] = f4add(a3, b3);
      }
      for (; k < end; k += nt) out[k] = f4add(in[k], ms[0][k]);
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
__device__ __forceinline__ unsigned char f2e4m3(float v) { __hip_fp8_e4m3 q(v); return q.__x; }
__device__ __forceinline__ float e4m3_2f(unsigned char b) {
  __hip_fp8_e4m3 q; q.__x = b; return (float)q;
}

template <int NPEER>
struct TpPush8 {
  unsigned char* base[NPEER];
  unsigned int*  flags[NPEER];
  unsigned int*  wake[NPEER];
};

template <int NPEER, bool NRANK_SYNC = false, bool CACHE_REGS = false>
__global__ void tp_ar_fp8_kernel(const float* __restrict__ in,
                                 TpPush8<NPEER> push, const unsigned char* my_base,
                                 float* __restrict__ out,
                                 unsigned int* my_flags,
                                 unsigned int* seq_ctrs, unsigned int* timeout,
                                 int n, int slot_stride_bytes, int peer_stride_bytes,
                                 int scale_off_bytes, int own) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (b == 0 && tid == 0) {
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

      uint32_t pk = 0u;
      #pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        const unsigned char q = (base_idx + i < n) ? f2e4m3(v[i] * inv) : 0u;
        pk |= (uint32_t)q << (8 * i);
      }
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
  __syncthreads();
  if constexpr (NRANK_SYNC && NPEER > 1) {
    if (tid == 0) {
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
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
      for (int j = 0; j < NPEER; ++j) store_sys_rel(&push.flags[j][b], s);
      unsigned long long z = 0;
      #pragma unroll
      for (int j = 0; j < NPEER; ++j) {
        const unsigned int* f = &my_flags[j * kTpFlagStride + b];
        unsigned int fv = load_sys_relaxed(f);
        if (fv < s) {
          do {
            __builtin_amdgcn_s_sleep(1);
            fv = load_sys_relaxed(f);
          } while (fv < s && ++z <= kTpSpinMax);
          if (fv < s) note_timeout(timeout, s);
        }
      }
    }
  }
  __syncthreads();
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
      if constexpr (CACHE_REGS) {
        if (gi < kTpFp8RegCacheGroups) {
          mine_scale = reg_sc[gi];
          pk = reg_pk[gi];
        } else {
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
          #pragma unroll
          for (int i = 0; i < kPerLane; ++i) {
            const unsigned char q = (base_idx + i < n) ? f2e4m3(v[i] * mine_inv) : 0u;
            pk |= (uint32_t)q << (8 * i);
          }
        }
      } else {
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
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const unsigned char q = (base_idx + i < n) ? f2e4m3(v[i] * mine_inv) : 0u;
          pk |= (uint32_t)q << (8 * i);
        }
      }

      if (base_idx + (kPerLane - 1) < n) {
        uint32_t peer_pk[NPEER];
        #pragma unroll
        for (int j = 0; j < NPEER; ++j) peer_pk[j] = *(const uint32_t*)(mb[j] + base_idx);
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const float mine = e4m3_2f((unsigned char)(pk >> (8 * i))) * mine_scale;
          if constexpr (NPEER == 1) {
            out[base_idx + i] = mine + e4m3_2f((unsigned char)(peer_pk[0] >> (8 * i))) * sc[0];
          } else {
            float acc = 0.0f;
            #pragma unroll
            for (int p = 0; p <= NPEER; ++p) {
              if (p == own) { acc += mine; continue; }
              const int j = p < own ? p : p - 1;
              acc += e4m3_2f((unsigned char)(peer_pk[j] >> (8 * i))) * sc[j];
            }
            out[base_idx + i] = acc;
          }
        }
      } else {
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) {
            const float mine = e4m3_2f((unsigned char)(pk >> (8 * i))) * mine_scale;
            if constexpr (NPEER == 1) {
              out[idx] = mine + e4m3_2f(mb[0][idx]) * sc[0];
            } else {
              float acc = 0.0f;
              #pragma unroll
              for (int p = 0; p <= NPEER; ++p) {
                if (p == own) { acc += mine; continue; }
                const int j = p < own ? p : p - 1;
                acc += e4m3_2f(mb[j][idx]) * sc[j];
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
__global__ void tp_ar_fp8_staged_kernel(const float* __restrict__ in,
                                        TpPush8<1> push, const unsigned char* my_base,
                                        float* __restrict__ out,
                                        unsigned int* my_flags,
                                        unsigned int* seq_ctrs, unsigned int* timeout,
                                        int n, int slot_stride_bytes, int scale_off_bytes,
                                        int stages) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (b == 0 && tid == 0) store_sys_relaxed(push.wake[0], s);

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
    pk = 0u;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const unsigned char q = (base_idx + i < n) ? f2e4m3(v[i] * inv) : 0u;
      pk |= (uint32_t)q << (8 * i);
    }
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

  auto publish_stage = [&](int st) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) store_sys_rel(&push.flags[0][(size_t)(1 + st) * kTpMaxBlocks + b], s);
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
    __syncthreads();
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
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const float mine = e4m3_2f((unsigned char)(pk >> (8 * i))) * mine_scale;
          const float peer = e4m3_2f((unsigned char)(peer_pk >> (8 * i))) * peer_scale;
          out[base_idx + i] = mine + peer;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          const int idx = base_idx + i;
          if (idx < n) {
            const float mine = e4m3_2f((unsigned char)(pk >> (8 * i))) * mine_scale;
            out[idx] = mine + e4m3_2f(mb0[idx]) * peer_scale;
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

// Fused 2-rank exact FP32 all-reduce + batch HC post/expand (+ optional DSpark tap) over live
// tokens [0, live) using ctx.scratch (0 B extra VRAM) and zeroing `block` across all [0, nb).
template <uint32_t NH, bool HasTap, bool HasControl, bool LiveOne>
__global__ void tp_ar_batch_hc_expand_kernel(
    uint16_t* __restrict__ out_hc, float* __restrict__ block, const uint16_t* __restrict__ hc,
    const float* __restrict__ post, const float* __restrict__ comb, uint32_t n_embd, uint32_t nb,
    uint32_t live, int zb, float* __restrict__ tap, uint32_t b_lo, uint32_t n_keep,
    uint32_t row0, uint32_t rows_cap, uint32_t row_stride, uint32_t slot_off,
    float* __restrict__ peer_scratch, const float* __restrict__ my_scratch,
    unsigned int* __restrict__ peer_flags, unsigned int* __restrict__ my_flags,
    unsigned int* __restrict__ peer_wake, unsigned int* __restrict__ seq_ctrs,
    unsigned int* __restrict__ timeout, size_t slot_floats) {
  const uint32_t b_blk = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq;
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b_blk], 1u) + 1u;
  __syncthreads();
  const unsigned int s = s_seq;
  const int slot = (int)(s % TpContext::kTpSlots);

  if (b_blk == 0 && tid == 0) store_sys_relaxed(peer_wake, s);

  float* const ps = peer_scratch + (size_t)slot * slot_floats;
  const float* const ms = my_scratch + (size_t)slot * slot_floats;

  const uint32_t d_per_blk = (n_embd + gridDim.x - 1u) / gridDim.x;
  const uint32_t d0 = (uint32_t)min((size_t)n_embd, (size_t)b_blk * d_per_blk);
  const uint32_t d1 = (uint32_t)min((size_t)n_embd, (size_t)d0 + d_per_blk);
  const size_t idx0 = LiveOne ? (size_t)d0 : (size_t)d0 * live;
  const size_t idx1 = LiveOne ? (size_t)d1 : (size_t)d1 * live;

  for (size_t idx = idx0 + tid; idx < idx1; idx += nt) {
    const uint32_t d = LiveOne ? (uint32_t)idx : (uint32_t)(idx / live);
    const uint32_t tok = LiveOne ? 0u : (uint32_t)(idx - (size_t)d * live);
    ps[idx] = block[(size_t)d * nb + tok];
  }

  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();

  if (tid == 0) {
    store_sys_rel(&peer_flags[b_blk], s);
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
  __syncthreads();
  fence_sys_acq();

  for (size_t idx = idx0 + tid; idx < idx1; idx += nt) {
    const uint32_t d = LiveOne ? (uint32_t)idx : (uint32_t)(idx / live);
    const uint32_t tok = LiveOne ? 0u : (uint32_t)(idx - (size_t)d * live);
    const size_t blk_idx = (size_t)d * nb + tok;
    const float bv = block[blk_idx] + ms[idx];
    if (zb) block[blk_idx] = 0.0f;
    if constexpr (HasControl) {
      float v[NH];
#pragma unroll
      for (uint32_t src = 0; src < NH; ++src)
        v[src] = wmma::bf16_to_f32(hc[((size_t)src * n_embd + d) * nb + tok]);
      float tap_acc = 0.0f;
#pragma unroll
      for (uint32_t dst = 0; dst < NH; ++dst) {
        float acc = bv * post[(size_t)dst * nb + tok];
#pragma unroll
        for (uint32_t src = 0; src < NH; ++src)
          acc += comb[(size_t)(dst + src * NH) * nb + tok] * v[src];
        const uint16_t h16 = wmma::f32_to_bf16(acc);
        out_hc[((size_t)dst * n_embd + d) * nb + tok] = h16;
        if constexpr (HasTap) tap_acc += wmma::bf16_to_f32(h16);
      }
      if constexpr (HasTap) {
        if (tok >= b_lo && tok < b_lo + n_keep) {
          const uint32_t row = row0 + (tok - b_lo);
          if (row < rows_cap) tap[(size_t)row * row_stride + slot_off + d] = tap_acc / (float)NH;
        }
      }
    } else {
      out_hc[blk_idx] = wmma::f32_to_bf16(wmma::bf16_to_f32(out_hc[blk_idx]) + bv);
    }
  }

  if (live < nb) {
    const uint32_t pad = nb - live;
    const size_t ptot = (size_t)(d1 - d0) * pad;
    for (size_t pidx = tid; pidx < ptot; pidx += nt) {
      const uint32_t d = d0 + (uint32_t)(pidx / pad);
      const uint32_t tok = live + (uint32_t)(pidx % pad);
      if (zb) block[(size_t)d * nb + tok] = 0.0f;
      if constexpr (HasControl) {
#pragma unroll
        for (uint32_t dst = 0; dst < NH; ++dst)
          out_hc[((size_t)dst * n_embd + d) * nb + tok] = 0u;
      } else {
        out_hc[(size_t)d * nb + tok] = 0u;
      }
    }
  }
}

} // namespace

// Sub-slots a rank keeps: one per peer, none for itself. Zero at one rank, which is why every
// allocation below is guarded rather than assumed non-empty.
static inline size_t tp_npeer(const TpContext* ctx) { return ctx->ranks ? ctx->ranks - 1 : 0; }

// Floats one double-buffer slot spans on one rank: every peer's message, back to back.
static inline size_t tp_slot_floats(const TpContext* ctx) {
  return tp_npeer(ctx) * ctx->max_floats;
}

bool tp_grow(TpContext* ctx, size_t max_floats, std::string* err) {
  if (!ctx->ready) return false;
  const size_t want = (max_floats + 3) / 4 * 4;
  if (want <= ctx->max_floats) return true;
  const size_t np = tp_npeer(ctx);
  if (!np) { ctx->max_floats = want; return true; }
  for (size_t r = 0; r < ctx->ranks; ++r) {
    if (!hip_try(hipSetDevice(ctx->dev[r]), "hipSetDevice", err)) return false;
    if (!hip_try(hipStreamSynchronize((hipStream_t)ctx->stream[r]), "drain before grow", err))
      return false;
    (void)hipFree(ctx->scratch[r]);
    ctx->scratch[r] = nullptr;
    void* sc = nullptr;
    if (!hip_try(hipExtMallocWithFlags(&sc, TpContext::kTpSlots * np * want * sizeof(float),
                                       hipDeviceMallocFinegrained),
                 "scratch regrow", err))
      return false;
    ctx->scratch[r] = (float*)sc;
  }
  ctx->max_floats = want;
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
    // fseq is a 2-element array (fold path, 2 ranks only). Do not index past it.
    if (r < 2 && ctx.fseq[r]) {
      set_device(ctx.dev[r]);
      AFF_HIP_CHECK(hipMemsetAsync(ctx.fseq[r], 0, kTpMaxBlocks * sizeof(unsigned),
                                   (hipStream_t)ctx.stream[r]));
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
  const size_t np = ctx.ranks - 1;
  for (size_t r = 0; r < ctx.ranks; ++r) {
    TpPush<NPEER> push{};
    for (size_t q = 0, j = 0; q < ctx.ranks; ++q) {
      if (q == r) continue;
      const size_t mine = tp_peer_slot(r, q);      // where rank r sits in rank q's tables
      push.scratch[j] = (float4*)(ctx.scratch[q] + mine * ctx.max_floats);
      push.flags[j] = ctx.flags[q] + mine * kTpFlagStride;
      push.wake[j] = ctx.flags[q] + np * kTpFlagStride;
      ++j;
    }
    set_device(ctx.dev[r]);
    if (tp_nrank_sync && NPEER > 1) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER, true>), dim3(nb), dim3(nt), 0,
                         (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                         (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n4, slot4, peer4, (int)r);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_kernel<NPEER, false>), dim3(nb), dim3(nt), 0,
                         (hipStream_t)ctx.stream[r], (const float4*)in[r], push,
                         (const float4*)ctx.scratch[r], (float4*)out[r], ctx.flags[r], ctx.seq[r],
                         ctx.timeout_dev, n4, slot4, peer4, (int)r);
    }
    AFF_HIP_CHECK(hipGetLastError());
  }
}

template <int NPEER>
void tp_launch_fp8(TpContext& ctx, float* const* in, float* const* out, int n, int nb,
                   int slot_bytes, int peer_bytes, int scale_off) {
  const bool tp_nrank_sync = tp_nrank_sync_enabled();
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
                               ctx.seq[r], ctx.timeout_dev, n, slot_bytes, scale_off, stages);
          } else {
            hipLaunchKernelGGL((tp_ar_fp8_staged_kernel<false>), dim3(nb), dim3(256), 0,
                               (hipStream_t)ctx.stream[r], in[r], push,
                               (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r],
                               ctx.seq[r], ctx.timeout_dev, n, slot_bytes, scale_off, stages);
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
      push.base[j] = (unsigned char*)(ctx.scratch[q] + mine * ctx.max_floats);
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
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, true, false>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);
    } else {
      if (cache_regs)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, false, true>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_ar_fp8_kernel<NPEER, false, false>), dim3(nb),
                           dim3(256), 0, (hipStream_t)ctx.stream[r], in[r], push,
                           (const unsigned char*)ctx.scratch[r], out[r], ctx.flags[r], ctx.seq[r],
                           ctx.timeout_dev, n, slot_bytes, peer_bytes, scale_off, (int)r);
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
  if (mode == TpReduce::Auto)
    mode = (n * sizeof(float) >= kTpFp8MinBytes) ? TpReduce::Fp8 : TpReduce::Exact;

  if (mode == TpReduce::Fp8) {
    const int nn = (int)n;
    int nb = (int)(((n + kTpFp8Group - 1) / kTpFp8Group + 63) / 64);
    nb = std::max(1, std::min(nb, kTpMaxBlocks));
    tp_sync_on_grid_change(ctx, nb);
    const int peer_bytes = (int)(ctx.max_floats * sizeof(float));
    const int slot_bytes = (int)(np * ctx.max_floats * sizeof(float));
    const int scale_off = (int)ctx.max_floats;      // codes occupy [0,n); scales start past them
    switch (np) {
      case 1: tp_launch_fp8<1>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 2: tp_launch_fp8<2>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 3: tp_launch_fp8<3>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 4: tp_launch_fp8<4>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 5: tp_launch_fp8<5>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      case 6: tp_launch_fp8<6>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
      default: tp_launch_fp8<7>(ctx, in, out, nn, nb, slot_bytes, peer_bytes, scale_off); break;
    }
    return;
  }

  const int n4 = (int)((n + 3) / 4);
  const int peer4 = (int)(ctx.max_floats / 4);
  const int slot4 = (int)(np * ctx.max_floats / 4);

  // Block count from the payload. Small messages want ONE block: their cost is the handshake, and
  // extra blocks add flag traffic without adding transfer.
  int nb = (int)((n4 + kTpWordsPerBlock - 1) / kTpWordsPerBlock);
  nb = std::max(1, std::min(nb, kTpMaxBlocks));
  tp_sync_on_grid_change(ctx, nb);
  const int nt = 256;

  // Every rank's kernel must be IN FLIGHT together — each spins on the others' flags. Launching them
  // back to back on separate devices and streams achieves that; synchronising between the launches
  // would deadlock, and a blocking synchronise per all-reduce costs many times the crossing itself.
  switch (np) {
    case 1: tp_launch_exact<1>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 2: tp_launch_exact<2>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 3: tp_launch_exact<3>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 4: tp_launch_exact<4>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 5: tp_launch_exact<5>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    case 6: tp_launch_exact<6>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
    default: tp_launch_exact<7>(ctx, in, out, n4, nb, nt, slot4, peer4); break;
  }
}

bool tp_ar_batch_hc_expand(TpContext& ctx, uint16_t* const out_hc[kTpMaxRanks],
                           float* const block[kTpMaxRanks],
                           const uint16_t* const hc[kTpMaxRanks],
                           const float* const post[kTpMaxRanks],
                           const float* const comb[kTpMaxRanks], uint32_t n_embd,
                           uint32_t n_hc, uint32_t nb, uint32_t live, bool has_control,
                           int zb, float* const tap[kTpMaxRanks],
                           uint32_t b_lo, uint32_t n_keep, uint32_t row0,
                           uint32_t rows_cap, uint32_t row_stride, uint32_t slot_off) {
  if (!ctx.ready || ctx.ranks != 2 || !n_embd || !nb || !live || live > nb) return false;
  if (has_control && n_hc != 1u && n_hc != 2u && n_hc != 4u && n_hc != 8u) return false;
  const size_t live_elems = (size_t)n_embd * live;
  if (live_elems > ctx.max_floats) return false;

  const size_t n4 = (live_elems + 3u) / 4u;
  const int blk = std::max(1, std::min(kTpMaxBlocks, (int)((n4 + kTpWordsPerBlock - 1u) / kTpWordsPerBlock)));
  tp_sync_on_grid_change(ctx, blk);
  constexpr uint32_t nt = 256u;
  const bool has_tap = tap && tap[0] && tap[1] && n_keep > 0;
  const bool live_one = (live == 1u);

  for (size_t r = 0; r < 2; ++r) {
    const size_t q = 1u - r;
    float* const ps = ctx.scratch[q];
    const float* const ms = ctx.scratch[r];
    unsigned int* const pf = ctx.flags[q];
    unsigned int* const mf = ctx.flags[r];
    unsigned int* const pw = ctx.flags[q] + kTpFlagStride;
    float* const tap_r = has_tap ? tap[r] : nullptr;
    set_device(ctx.dev[r]);
    const hipStream_t st = (hipStream_t)ctx.stream[r];
#define AFF_TP_BHC_EXP(NH_, TAP_, CTL_, ONE_)                                                     \
  hipLaunchKernelGGL((tp_ar_batch_hc_expand_kernel<NH_, TAP_, CTL_, ONE_>), dim3(blk), dim3(nt),  \
                     0, st, out_hc[r], block[r], hc ? hc[r] : nullptr,                            \
                     post ? post[r] : nullptr, comb ? comb[r] : nullptr, n_embd, nb, live, zb,   \
                     tap_r, b_lo, n_keep, row0, rows_cap, row_stride, slot_off, ps, ms, pf, mf,  \
                     pw, ctx.seq[r], ctx.timeout_dev, ctx.max_floats)
#define AFF_TP_BHC_CTL(NH_, TAP_, ONE_)                                                           \
  do {                                                                                            \
    if (has_control) AFF_TP_BHC_EXP(NH_, TAP_, true, ONE_);                                       \
    else             AFF_TP_BHC_EXP(NH_, TAP_, false, ONE_);                                      \
  } while (0)
#define AFF_TP_BHC_TAP(NH_, ONE_)                                                                 \
  do {                                                                                            \
    if (has_tap) AFF_TP_BHC_CTL(NH_, true, ONE_);                                                 \
    else         AFF_TP_BHC_CTL(NH_, false, ONE_);                                                \
  } while (0)
#define AFF_TP_BHC_ONE(NH_)                                                                       \
  do {                                                                                            \
    if (live_one) AFF_TP_BHC_TAP(NH_, true);                                                      \
    else          AFF_TP_BHC_TAP(NH_, false);                                                     \
  } while (0)
    if (!has_control || n_hc == 4u) AFF_TP_BHC_ONE(4u);
    else if (n_hc == 8u)            AFF_TP_BHC_ONE(8u);
    else if (n_hc == 2u)            AFF_TP_BHC_ONE(2u);
    else                            AFF_TP_BHC_ONE(1u);
#undef AFF_TP_BHC_ONE
#undef AFF_TP_BHC_TAP
#undef AFF_TP_BHC_CTL
#undef AFF_TP_BHC_EXP
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
