// HIP device plumbing for RDNA4 (gfx12). Deliberately thin: there is one target architecture, so
// there is no device abstraction and no backend dispatch.
//
// Three gfx12 facts that shape everything in src/gpu/:
//
//   * wave32 deliberately. RDNA can run wave64, and several HIP samples use width-less `__shfl_*`,
//     which silently means "whatever the wave is". Every cross-lane op here passes an explicit
//     width of 32 and every launch assumes 32 lanes.
//   * No async global->LDS copy. There is no `cp.async` equivalent; staging through LDS costs a
//     real round trip, so it only pays for data reused many times (the codebook, the activation).
//   * 256 B cachelines, not 128. Coalescing targets are twice as wide as CUDA intuition suggests.
//
// NEVER iterate 0..hipGetDeviceCount(); always go through gfx12_devices(). An integrated GPU is
// enumerated alongside the discrete cards and reports system RAM as its "VRAM", so a capacity check
// will not rule it out — and a launch that lands on a device this binary has no code objects for
// dies inside libamdhip64 with a SEGV rather than a clean error.
//
// One naming trap: hipDeviceProp_t::multiProcessorCount counts Work Group Processors on RDNA, and a
// WGP is two CUs. A part that rocminfo calls 64 CUs reports 32 here. See device_cus().

#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>

#include "ui/log.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace aff {

// Upper bound for per-device static tables. Sized above any configuration the engine supports so
// the tables are never resized; the live device count comes from gfx12_devices().
constexpr int kMaxGpuDevices = 8;

#define AFF_HIP_CHECK(expr)                                                              \
  do {                                                                                   \
    const hipError_t aff_err_ = (expr);                                                  \
    if (aff_err_ != hipSuccess) {                                                        \
      aff::ui::fatal("%s:%d: HIP error: %s (%s)\n", __FILE__, __LINE__,                  \
                     hipGetErrorString(aff_err_), #expr);                                \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)

// Non-fatal variant for probing: returns false and fills `err` instead of aborting.
inline bool hip_try(hipError_t e, const char* what, std::string* err) {
  if (e == hipSuccess) return true;
  if (err) *err = std::string(what) + ": " + hipGetErrorString(e);
  return false;
}

inline constexpr int kWave = 32;   // RDNA4 wave32; checked against the device, never inferred

// COMPUTE UNITS, WHICH IS NOT WHAT hipDeviceProp_t REPORTS. On RDNA `multiProcessorCount` counts
// Work Group Processors and a WGP is two CUs, so a 64-CU part reports 32. Taking that field as a
// CU count halves every budget derived from it and, worse, halves the width of a CU mask — a
// 64-CU part silently gets one covering only its first 32. The doubling is RDNA-specific: on CDNA
// the field already is the CU count.
inline uint32_t device_cus(int dev) {
  hipDeviceProp_t pr{};
  if (hipGetDeviceProperties(&pr, dev) != hipSuccess || pr.multiProcessorCount <= 0) {
    // No fallback. Guessing a CU count here silently mis-sizes every grid derived from it, and the
    // wrong answer is a slow run rather than an error anyone would notice.
    aff::ui::fatal("device %d: cannot read multiProcessorCount\n", dev);
    std::abort();
  }
  return (uint32_t)pr.multiProcessorCount * 2u;
}

// Total HIP devices, INCLUDING the iGPU. Rarely what you want — see gfx12_devices().
inline int hip_device_count() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  return n;
}

// The architecture this binary holds code objects for. CMake defines it; the fallback keeps the
// header usable in a translation unit compiled without it.
#ifndef AFF_GPU_ARCH
#define AFF_GPU_ARCH "gfx1201"
#endif

// Indices of the devices this build can run on. This is the only enumeration the engine should
// use: `hipSetDevice(0)` is right only by luck, and breaks the moment an integrated GPU sorts
// first.
//
// The match is against the compiled arch exactly, not the gfx12 family. A family prefix accepts
// cards this binary has no code objects for — gfx1200 is a shipping RDNA4 part — and the failure
// is a SEGV inside libamdhip64 on the first launch rather than a diagnosable error.
inline std::vector<int> gfx12_devices() {
  std::vector<int> out;
  const int n = hip_device_count();
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
    // gcnArchName carries feature suffixes (":sramecc+:xnack-"); compare only the base name.
    std::string arch(p.gcnArchName);
    arch = arch.substr(0, arch.find(':'));
    if (arch != AFF_GPU_ARCH) continue;
    // Every cross-lane op in src/gpu/ passes an explicit width of kWave and every launch assumes
    // that many lanes. A wave64 device would compute the wrong thing quietly, so refuse it here.
    if (p.warpSize != kWave) {
      aff::ui::err("device %d (%s) reports wave%d; this build assumes wave%d. Skipping.\n",
                   i, arch.c_str(), p.warpSize, kWave);
      continue;
    }
    out.push_back(i);
  }
  return out;
}

// Supported devices found, and every device that was rejected and why. Called once at startup so a
// user whose card is skipped is told which one and what the binary was built for, rather than
// meeting "no gfx12 device" with two cards installed.
inline void report_rejected_devices() {
  const int n = hip_device_count();
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
    std::string arch(p.gcnArchName);
    arch = arch.substr(0, arch.find(':'));
    if (arch != AFF_GPU_ARCH)
      aff::ui::err("device %d (%s) skipped: this build targets %s. Rebuild with "
                   "-DAFF_GPU_ARCH=%s to use it.\n", i, arch.c_str(), AFF_GPU_ARCH, arch.c_str());
  }
}

// Makes the first gfx12 device current. Returns false when there is none.
inline bool hip_select_device(int which = 0) {
  const std::vector<int> d = gfx12_devices();
  if ((size_t)which >= d.size()) return false;
  return hipSetDevice(d[which]) == hipSuccess;
}

struct DeviceInfo {
  int         index = 0;
  std::string arch;
  size_t      vram_total = 0;
  size_t      vram_free  = 0;
  int         cus = 0;
  int         lds_per_block = 0;
};

// Queries device `i`. Restores the previously-current device: reading free VRAM requires making a
// device current, and leaving that in place turned an informational summary into a trap that
// silently retargeted every subsequent allocation and launch at the iGPU.
inline bool hip_device_info(int i, DeviceInfo* out, std::string* err) {
  hipDeviceProp_t p{};
  if (!hip_try(hipGetDeviceProperties(&p, i), "hipGetDeviceProperties", err)) return false;
  out->index = i;
  out->arch = p.gcnArchName;
  out->cus = (int)device_cus(i);
  out->lds_per_block = (int)p.sharedMemPerBlock;
  int prev = 0;
  const bool have_prev = hipGetDevice(&prev) == hipSuccess;
  if (hipSetDevice(i) == hipSuccess) (void)hipMemGetInfo(&out->vram_free, &out->vram_total);
  if (have_prev) (void)hipSetDevice(prev);
  return true;
}

// ---- free VRAM, from the driver's counters ----------------------------------------------------
//
// `hipMemGetInfo` reports a LARGER free figure than the amdgpu driver's own sysfs counter on this
// machine: measured on 2x R9700 / ROCm 7.14, 1026 MiB by HIP against 677 MiB by sysfs, with the
// expert slab and the codebook already allocated and nothing else running on the cards.
//
// That difference is not noise, and it is not small: ~349 MiB, the same order as one full KV
// commit (0.66 GiB for 262144 positions). The KV grower pays for that commit out of the reserve the
// slab leaves, so sizing the slab from the optimistic number over-commits by exactly what the
// reserve exists to cover — a sequence that reaches the top of --kv-size then ends with tens of MiB
// of real headroom while the engine still believes it has hundreds.
//
// So take the SMALLER of the two. sysfs is the driver's view of the whole card, other processes
// included, which is the right question on a dedicated inference box; HIP is the fallback when the
// counter cannot be read, and also the floor when it disagrees downwards.
inline bool read_sysfs_u64(const std::string& path, uint64_t* out) {
  std::FILE* f = std::fopen(path.c_str(), "re");
  if (!f) return false;
  unsigned long long v = 0;
  const bool ok = std::fscanf(f, "%llu", &v) == 1;
  std::fclose(f);
  if (ok && out) *out = (uint64_t)v;
  return ok;
}

inline bool hip_free_vram_checked(int dev, uint64_t* free_bytes, uint64_t* total_bytes) {
  if (!free_bytes || !total_bytes) return false;
  int prev = 0;
  const bool have_prev = hipGetDevice(&prev) == hipSuccess;
  if (hipSetDevice(dev) != hipSuccess) return false;
  size_t fb = 0, tb = 0;
  const bool hip_ok = hipMemGetInfo(&fb, &tb) == hipSuccess;
  char bus[32] = {0};
  const bool named = hipDeviceGetPCIBusId(bus, (int)sizeof(bus), dev) == hipSuccess && bus[0];
  if (!hip_ok) {
    if (have_prev) (void)hipSetDevice(prev);
    return false;
  }
  uint64_t sys_total = 0, sys_used = 0, freeb = fb, totalb = tb;
  if (named) {
    const std::string base = std::string("/sys/bus/pci/devices/") + bus;
    if (read_sysfs_u64(base + "/mem_info_vram_total", &sys_total) &&
        read_sysfs_u64(base + "/mem_info_vram_used", &sys_used) && sys_used <= sys_total) {
      const uint64_t sys_free = sys_total - sys_used;
      if (sys_free < freeb) freeb = sys_free;
      if (totalb == 0 || sys_total == totalb) totalb = sys_total;
    }
  }
  if (have_prev) (void)hipSetDevice(prev);
  *free_bytes = freeb;
  *total_bytes = totalb;
  return true;
}

// RAII device buffer. No pooling: allocation happens at load time, not on the decode path.
template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { alloc(n); }
  ~DeviceBuffer() { free(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }

  // `pad_bytes` over-allocates so a kernel may read a few bytes past the logical end. The packed
  // formats are read through 64-bit windows at arbitrary bit offsets, so the last window of a
  // buffer legitimately straddles the end.
  void alloc(size_t n, size_t pad_bytes = 0) {
    free();
    AFF_HIP_CHECK(hipMalloc(&p_, n * sizeof(T) + pad_bytes));
    AFF_HIP_CHECK(hipMemset(p_, 0, n * sizeof(T) + pad_bytes));
    n_ = n;
  }
  void upload(const T* src, size_t n) {
    AFF_HIP_CHECK(hipMemcpy(p_, src, n * sizeof(T), hipMemcpyHostToDevice));
  }
  void download(T* dst, size_t n) const {
    AFF_HIP_CHECK(hipMemcpy(dst, p_, n * sizeof(T), hipMemcpyDeviceToHost));
  }
  void free() {
    if (p_) { (void)hipFree(p_); p_ = nullptr; n_ = 0; }
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }

private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

#ifdef __HIP_DEVICE_COMPILE__
#define AFF_WAVES(n) __attribute__((amdgpu_waves_per_eu((n), (n))))
#else
#define AFF_WAVES(n)
#endif

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
// Workgroup barrier for pure LDS producer/consumer synchronization: skips global memory
// invalidation/drain implied by `__syncthreads()`, ordering only local address space.
__device__ __forceinline__ void lds_barrier() {
#ifdef __HIP_DEVICE_COMPILE__
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  __syncthreads();
#endif
}

// Wave32 lane-XOR exchange via DPP16 `row_xmask` (offsets 1, 2, 4, 8) and `v_permlanex16_b32`
// (offset 16), avoiding the `ds_bpermute_b32` LDS crossbar queue. Bit-identical to
// `__shfl_xor(v, Lane, kWave)` when all 32 lanes participate.
template <uint32_t Lane>
__device__ __forceinline__ uint32_t dpp_xor_lane(uint32_t v) {
  static_assert(Lane == 1u || Lane == 2u || Lane == 4u || Lane == 8u || Lane == 16u,
                "Lane must be a Wave32 power-of-two xor offset in {1,2,4,8,16}");
#ifdef __HIP_DEVICE_COMPILE__
  if constexpr (Lane < 16u) {
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | (int)Lane, 0xf, 0xf, true);
  } else {
    return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u,
                                                  false, false);
  }
#else
  return (uint32_t)__shfl_xor((int)v, (int)Lane, kWave);
#endif
}

template <uint32_t Lane>
__device__ __forceinline__ int dpp_xor_lane(int v) {
  return (int)dpp_xor_lane<Lane>((uint32_t)v);
}

template <uint32_t Lane>
__device__ __forceinline__ float dpp_xor_lane(float v) {
  return __uint_as_float(dpp_xor_lane<Lane>(__float_as_uint(v)));
}

// Runtime-mask Wave32 xor shuffle using the same DPP16 lane mappings for {16, 8, 4, 2, 1}.
__device__ __forceinline__ uint32_t wave32_shuffle_xor(uint32_t v, uint32_t mask) {
  switch (mask) {
    case 1u:  return dpp_xor_lane<1u>(v);
    case 2u:  return dpp_xor_lane<2u>(v);
    case 4u:  return dpp_xor_lane<4u>(v);
    case 8u:  return dpp_xor_lane<8u>(v);
    case 16u: return dpp_xor_lane<16u>(v);
    default:  return (uint32_t)__shfl_xor((int)v, (int)mask, kWave);
  }
}

__device__ __forceinline__ int wave32_shuffle_xor(int v, uint32_t mask) {
  return (int)wave32_shuffle_xor((uint32_t)v, mask);
}

__device__ __forceinline__ float wave32_shuffle_xor(float v, uint32_t mask) {
  return __uint_as_float(wave32_shuffle_xor(__float_as_uint(v), mask));
}

// Wave32 butterfly reductions in exact 16 -> 8 -> 4 -> 2 -> 1 xor-mask order so floating-point
// accumulation order is bit-identical to `for (int o = 16; o > 0; o >>= 1) op(__shfl_xor(v, o, 32))`.
__device__ __forceinline__ float wave_reduce_add_dpp(float v) {
  v += dpp_xor_lane<16u>(v);
  v += dpp_xor_lane<8u>(v);
  v += dpp_xor_lane<4u>(v);
  v += dpp_xor_lane<2u>(v);
  v += dpp_xor_lane<1u>(v);
  return v;
}

__device__ __forceinline__ float wave_reduce_max_dpp(float v) {
  v = fmaxf(v, dpp_xor_lane<16u>(v));
  v = fmaxf(v, dpp_xor_lane<8u>(v));
  v = fmaxf(v, dpp_xor_lane<4u>(v));
  v = fmaxf(v, dpp_xor_lane<2u>(v));
  v = fmaxf(v, dpp_xor_lane<1u>(v));
  return v;
}

using aff_u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using aff_u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using aff_i32x4 = int __attribute__((ext_vector_type(4)));
using aff_i32x3 = int __attribute__((ext_vector_type(3)));
using aff_i32x2 = int __attribute__((ext_vector_type(2)));
using aff_u16_ua = uint16_t __attribute__((aligned(1)));

// gfx1201 (RDNA 4) buffer descriptor word 3 and temporal-hint fields (`CPol` / `TH`).
inline constexpr uint32_t kGfx12BufRsrc3 = 0x31004000u;
inline constexpr int kGfx12ThRt   = 0x00;
inline constexpr int kGfx12ThNt   = 0x01;
inline constexpr int kGfx12ThHt   = 0x02;
inline constexpr int kGfx12ThLu   = 0x03;
inline constexpr int kGfx12ThNtRt = 0x04;
inline constexpr int kGfx12ThRtNt = 0x05;
inline constexpr int kGfx12ThNtHt = 0x06;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
using gfx12_rsrc_t = __amdgpu_buffer_rsrc_t;

__device__ __forceinline__ gfx12_rsrc_t make_gfx12_rsrc(const void* base,
                                                        uint32_t bytes = 0xffffffffu) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, bytes, kGfx12BufRsrc3);
}

template <int TH>
__device__ __forceinline__ aff_u32x4 buf_load_u32x4(gfx12_rsrc_t r, uint32_t off) {
  const aff_i32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, (int)off, 0, TH);
  return aff_u32x4{(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3]};
}

template <int TH>
__device__ __forceinline__ uint3 buf_load_u32x3(gfx12_rsrc_t r, uint32_t off) {
  const aff_i32x3 v = __builtin_amdgcn_raw_buffer_load_b96(r, (int)off, 0, TH);
  return make_uint3((uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2]);
}

template <int TH>
__device__ __forceinline__ uint2 buf_load_u32x2(gfx12_rsrc_t r, uint32_t off) {
  const aff_i32x2 v = __builtin_amdgcn_raw_buffer_load_b64(r, (int)off, 0, TH);
  return make_uint2((uint32_t)v[0], (uint32_t)v[1]);
}

template <int TH>
__device__ __forceinline__ uint32_t buf_load_u32(gfx12_rsrc_t r, uint32_t off) {
  return (uint32_t)__builtin_amdgcn_raw_buffer_load_b32(r, (int)off, 0, TH);
}

template <int TH>
__device__ __forceinline__ uint16_t buf_load_u16(gfx12_rsrc_t r, uint32_t off) {
  return static_cast<uint16_t>(__builtin_amdgcn_raw_buffer_load_b16(r, (int)off, 0, TH));
}

template <int TH>
__device__ __forceinline__ uint8_t buf_load_u8(gfx12_rsrc_t r, uint32_t off) {
  return static_cast<uint8_t>(__builtin_amdgcn_raw_buffer_load_b8(r, (int)off, 0, TH));
}

__device__ __forceinline__ void prefetch_l2_uniform(const void* p, int lines = 1) {
  __builtin_amdgcn_s_prefetch_data(p, lines);
}
#else
struct gfx12_rsrc_t {
  const uint8_t* ptr;
};

__host__ __device__ __forceinline__ gfx12_rsrc_t make_gfx12_rsrc(const void* base,
                                                                 uint32_t = 0xffffffffu) {
  return gfx12_rsrc_t{static_cast<const uint8_t*>(base)};
}

template <int TH>
__device__ __forceinline__ aff_u32x4 buf_load_u32x4(gfx12_rsrc_t r, uint32_t off) {
  return *reinterpret_cast<const aff_u32x4*>(r.ptr + off);
}

template <int TH>
__device__ __forceinline__ uint3 buf_load_u32x3(gfx12_rsrc_t r, uint32_t off) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(r.ptr + off);
  return make_uint3(p[0], p[1], p[2]);
}

template <int TH>
__device__ __forceinline__ uint2 buf_load_u32x2(gfx12_rsrc_t r, uint32_t off) {
  return *reinterpret_cast<const uint2*>(r.ptr + off);
}

template <int TH>
__device__ __forceinline__ uint32_t buf_load_u32(gfx12_rsrc_t r, uint32_t off) {
  return *reinterpret_cast<const uint32_t*>(r.ptr + off);
}

template <int TH>
__device__ __forceinline__ uint16_t buf_load_u16(gfx12_rsrc_t r, uint32_t off) {
  return *reinterpret_cast<const uint16_t*>(r.ptr + off);
}

template <int TH>
__device__ __forceinline__ uint8_t buf_load_u8(gfx12_rsrc_t r, uint32_t off) {
  return *(r.ptr + off);
}

__device__ __forceinline__ void prefetch_l2_uniform(const void*, int = 1) {}
#endif

// Read-once global weight/scale load helpers (`TH_LOAD_NT` on gfx1201) to avoid evicting
// hot KV cache and activations from the 8 MiB L2 and 64 MiB MALL during narrow decode.
template <bool NT_LOAD, typename T>
__device__ __forceinline__ T load_nt(const T* p) {
#ifdef __HIP_DEVICE_COMPILE__
  if constexpr (NT_LOAD) return __builtin_nontemporal_load(p);
#endif
  return *p;
}

template <bool NT_LOAD>
__device__ __forceinline__ uint4 load_uint4_nt(const void* p) {
#ifdef __HIP_DEVICE_COMPILE__
  if constexpr (NT_LOAD) {
    const aff_u32x4 v = __builtin_nontemporal_load((const aff_u32x4*)p);
    return make_uint4(v[0], v[1], v[2], v[3]);
  }
#endif
  return *(const uint4*)p;
}

template <bool NT_LOAD>
__device__ __forceinline__ uint2 load_uint2_nt(const void* p) {
#ifdef __HIP_DEVICE_COMPILE__
  if constexpr (NT_LOAD) {
    const aff_u32x2 v = __builtin_nontemporal_load((const aff_u32x2*)p);
    return make_uint2(v[0], v[1]);
  }
#endif
  return *(const uint2*)p;
}

#ifdef __HIP_DEVICE_COMPILE__
template <bool NT_LOAD, typename T>
__device__ __forceinline__ T load_nt(const __attribute__((address_space(1))) T* p) {
  if constexpr (NT_LOAD) return __builtin_nontemporal_load(p);
  return *p;
}

template <bool NT_LOAD>
__device__ __forceinline__ uint4 load_uint4_nt(
    const __attribute__((address_space(1))) void* p) {
  if constexpr (NT_LOAD) {
    const aff_u32x4 v = __builtin_nontemporal_load(
        (const __attribute__((address_space(1))) aff_u32x4*)p);
    return make_uint4(v[0], v[1], v[2], v[3]);
  }
  const aff_u32x4 v = *(const __attribute__((address_space(1))) aff_u32x4*)p;
  return make_uint4(v[0], v[1], v[2], v[3]);
}

template <bool NT_LOAD>
__device__ __forceinline__ uint2 load_uint2_nt(
    const __attribute__((address_space(1))) void* p) {
  if constexpr (NT_LOAD) {
    const aff_u32x2 v = __builtin_nontemporal_load(
        (const __attribute__((address_space(1))) aff_u32x2*)p);
    return make_uint2(v[0], v[1]);
  }
  const aff_u32x2 v = *(const __attribute__((address_space(1))) aff_u32x2*)p;
  return make_uint2(v[0], v[1]);
}
#endif
#endif

// Host-cached gate for non-temporal loads on dense/GEMV decode weight/scale streams (default OFF
// to preserve gfx1201 64 MiB MALL + 8 MiB L2 reuse; opt-in with `AFF_DENSE_NT_LOAD=1`).
inline bool dense_nt_load_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("AFF_DENSE_NT_LOAD");
    return s && *s && !(s[0] == '0' && s[1] == '\0');
  }();
  return on;
}

// Host-cached gate for gfx1201 two-level V# buffer cache hints (`TH_LOAD_NT_RT`/`TH_LOAD_NT_HT`).
inline bool gfx12_cache_hints_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("AFF_GFX12_CACHE_HINTS");
    return !s || !*s || !(s[0] == '0' && s[1] == '\0');
  }();
  return on;
}

// Guards persistent / block-bounded launches against the gfx1201 full-card wave-slot cliff:
// wavefront counts in `[2016..2047] mod 2048` (and exact non-zero multiples of 2048) incur heavy
// tail scheduling stalls across the 64 CUs x 2 SIMDs x 16 slots = 2048 hardware wave slots.
// Only use on grid-stride kernels or kernels whose blockIdx.x excess exits cleanly.
inline constexpr uint32_t kGfx12CliffWaves  = 2048u;
inline constexpr uint32_t kGfx12CliffWindow = 32u;

__host__ __device__ __forceinline__ uint32_t safe_grid_blocks(uint32_t blocks,
                                                              uint32_t waves_per_block) {
  if (blocks == 0u || waves_per_block == 0u || (waves_per_block % kGfx12CliffWaves) == 0u) {
    return blocks;
  }
  uint32_t b = blocks;
  for (uint32_t i = 0u; i < 64u; ++i, ++b) {
    const uint32_t w = (uint32_t)(((uint64_t)b * waves_per_block) % kGfx12CliffWaves);
    if (w != 0u && w < kGfx12CliffWaves - kGfx12CliffWindow) return b;
  }
  return blocks;
}

} // namespace aff

// ---- amdnas-fixes malloc-trace --------------------------------------------------------------------------------------
// Every device allocation in src/gpu goes through here (this header is included by all of them, and the macro below
// rewrites the call sites). A failed hipMalloc is the one silent, VRAM-gated, permanent failure mode left after probes
// 8-15; it is logged with its site. AFF_MALLOC_TRACE=1 also logs successes >= 1 MiB.
inline hipError_t aff_hipMalloc_dbg(void** p, size_t n, const char* file, int line) {
  const hipError_t e = ::hipMalloc(p, n);
  static const bool trace = std::getenv("AFF_MALLOC_TRACE") != nullptr;
  int dev = -1; (void)hipGetDevice(&dev);
  if (e != hipSuccess) {
    size_t fb = 0, tb = 0; (void)hipMemGetInfo(&fb, &tb);
    aff::ui::err("HIPMALLOC-FAIL %s:%d dev %d bytes %zu (%.1f MiB) free %.1f MiB: %s\n", file, line, dev, n,
                 (double)n / 1048576.0, (double)fb / 1048576.0, hipGetErrorString(e));
  } else if (trace && n >= (1u << 20)) {
    aff::ui::err("hipmalloc %s:%d dev %d %.1f MiB -> %p\n", file, line, dev, (double)n / 1048576.0, *p);
  }
  return e;
}
#define hipMalloc(p, n) aff_hipMalloc_dbg((void**)(p), (n), __FILE__, __LINE__)
