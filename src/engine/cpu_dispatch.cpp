#include "engine/cpu_dispatch.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace aff {
namespace {

std::atomic<int> g_cpu_isa_override{-1};

CpuIsa detect_hw_max() noexcept {
#if AFF_CPU_X86
  __builtin_cpu_init();
  const bool has_avx = __builtin_cpu_supports("avx") != 0;
  if (!has_avx) return CpuIsa::Scalar;

  const bool has_avx2 =
      __builtin_cpu_supports("avx2") != 0 &&
      __builtin_cpu_supports("fma") != 0 &&
      __builtin_cpu_supports("f16c") != 0;
  if (!has_avx2) return CpuIsa::Avx1;

  const bool has_avx512 =
      __builtin_cpu_supports("avx512f") != 0 &&
      __builtin_cpu_supports("avx512bw") != 0 &&
      __builtin_cpu_supports("avx512dq") != 0 &&
      __builtin_cpu_supports("avx512vl") != 0;
  if (!has_avx512) return CpuIsa::Avx2;

  if (__builtin_cpu_supports("avx512vbmi") != 0) return CpuIsa::Avx512Vbmi;
  return CpuIsa::Avx512;
#else
  return CpuIsa::Scalar;
#endif
}

CpuIsa parse_isa_env(CpuIsa hw_max) noexcept {
  const char* raw = std::getenv("AFF_CPU_ISA");
  if (!raw || !*raw) return hw_max;

  char buf[32];
  size_t n = 0;
  for (; raw[n] && n + 1 < sizeof(buf); ++n) {
    buf[n] = static_cast<char>(std::tolower(static_cast<unsigned char>(raw[n])));
  }
  buf[n] = '\0';

  CpuIsa req = hw_max;
  if (std::strcmp(buf, "scalar") == 0 || std::strcmp(buf, "sse2") == 0) {
    req = CpuIsa::Scalar;
  } else if (std::strcmp(buf, "avx") == 0 || std::strcmp(buf, "avx1") == 0) {
    req = CpuIsa::Avx1;
  } else if (std::strcmp(buf, "avx2") == 0) {
    req = CpuIsa::Avx2;
  } else if (std::strcmp(buf, "avx512") == 0 || std::strcmp(buf, "avx-512") == 0) {
    req = CpuIsa::Avx512;
  } else if (std::strcmp(buf, "avx512vbmi") == 0 || std::strcmp(buf, "avx-512vbmi") == 0) {
    req = CpuIsa::Avx512Vbmi;
  } else {
    return hw_max;
  }
  return (static_cast<uint8_t>(req) <= static_cast<uint8_t>(hw_max)) ? req : hw_max;
}

}  // namespace

const char* cpu_isa_name(CpuIsa isa) noexcept {
  switch (isa) {
    case CpuIsa::Scalar: return "scalar";
    case CpuIsa::Avx1: return "avx1";
    case CpuIsa::Avx2: return "avx2";
    case CpuIsa::Avx512: return "avx512";
    case CpuIsa::Avx512Vbmi: return "avx512vbmi";
  }
  return "unknown";
}

CpuIsa hw_max_cpu_isa() noexcept {
  static const CpuIsa hw = detect_hw_max();
  return hw;
}

CpuIsa active_cpu_isa() noexcept {
  const CpuIsa hw = hw_max_cpu_isa();
  const int ov = g_cpu_isa_override.load(std::memory_order_relaxed);
  if (ov >= 0) {
    const uint8_t capped = (ov > static_cast<int>(hw))
                               ? static_cast<uint8_t>(hw)
                               : static_cast<uint8_t>(ov);
    return static_cast<CpuIsa>(capped);
  }
  static const CpuIsa env_isa = parse_isa_env(hw);
  return env_isa;
}

void set_cpu_isa_override(int isa_or_neg1) noexcept {
  if (isa_or_neg1 < 0) {
    g_cpu_isa_override.store(-1, std::memory_order_relaxed);
    return;
  }
  const int max_val = static_cast<int>(CpuIsa::Avx512Vbmi);
  g_cpu_isa_override.store(isa_or_neg1 > max_val ? max_val : isa_or_neg1,
                           std::memory_order_relaxed);
}

}  // namespace aff
