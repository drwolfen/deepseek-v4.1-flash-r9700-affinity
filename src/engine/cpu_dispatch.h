#pragma once

#include <cstdint>

namespace aff {

enum class CpuIsa : uint8_t {
  Scalar = 0,
  Avx1 = 1,
  Avx2 = 2,
  Avx512 = 3,
  Avx512Vbmi = 4,
};

const char* cpu_isa_name(CpuIsa isa) noexcept;
CpuIsa hw_max_cpu_isa() noexcept;
CpuIsa active_cpu_isa() noexcept;
void set_cpu_isa_override(int isa_or_neg1) noexcept;

#if defined(__x86_64__) || defined(__i386__)
#define AFF_CPU_X86 1
#define AFF_TARGET_AVX1 __attribute__((noinline, target("avx")))
#define AFF_TARGET_AVX2 __attribute__((noinline, target("avx2,fma,f16c")))
#define AFF_TARGET_AVX512 __attribute__((noinline, target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
#define AFF_TARGET_AVX512VBMI \
  __attribute__((noinline, target("avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi,fma")))
#else
#define AFF_CPU_X86 0
#endif

}  // namespace aff
