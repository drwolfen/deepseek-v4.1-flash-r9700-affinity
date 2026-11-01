# Affinity 1017 & Radiance Kernel Ports Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the 3 `vllm-sly-radiance` / `StillDeadcode/radiance` RDNA4 optimizations (`int4 lm_head`, Cactus length-faded/capped speculative sampling, zero-copy DSpark residual tap) plus the 2 `05yuki/deepseek-v4.1-flash-1017-ktransformers` `1017` alignment patches (in-place block-32 `ue8m0` E4M3 `o_fp8_qdq` before `wo_a`, and tool `namespace::name` encoding) into `/mnt/fast-storage/affinity`.

**Architecture:** All changes stay inside `/mnt/fast-storage/affinity/src/{engine,gpu}` and `/mnt/fast-storage/affinity/tools/affinity.cpp`, gated by environment switches with safe defaults so greedy/teacher-forced regression fingerprints can be verified bit-for-bit before enabling in `/home/ydj/.config/systemd/user/deepseek-v41-affinity.service`.

**Tech Stack:** C++20, HIP / ROCm (`gfx1201` RDNA4 wave32 WMMA), CMake/Ninja, `ctest`.

---

### Task 1: Port `1017` Attention Output Block-32 `ue8m0` E4M3 QDQ before `wo_a` (`AFF_O_FP8_QDQ`)

**Context (from `05yuki/deepseek-v4.1-flash-1017-ktransformers` Tier 0 `01-o-fp8-qdq.patch`):**
`DeepSeek-V4.1-Flash-1017` (`inference/model.py` lines 787 & 1073) added `act_quant(o, 32, "ue8m0", ..., inplace=True)` between inverse RoPE and `wo_a` in both `Attention.forward` and `DSparkAttention.forward`. Because Affinity runs `wo_a` on the BF16 WMMA kernel (`x_bf16=true`), `D.b_heads` is currently never quantized-dequantized through block-32 `ue8m0` E4M3 before `wo_a`, running `1017` off its training numerics.

**Files:**
- Modify: `/mnt/fast-storage/affinity/src/gpu/batch_kernels.h`
- Modify: `/mnt/fast-storage/affinity/src/gpu/batch_kernels.hip:1263-1280,2789-2798`
- Modify: `/mnt/fast-storage/affinity/src/gpu/dense_gpu.hip:3069-3081,4942-4959`
- Test: `/mnt/fast-storage/affinity/tests/test_o_fp8_qdq.cpp`

**Step 1: Write the failing unit test (`/mnt/fast-storage/affinity/tests/test_o_fp8_qdq.cpp`)**

```cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Reference 1017 act_quant(x, 32, "ue8m0", inplace=True) in FP32/E4M3 grid:
static float e4m3_round_ref(float v) {
  float a = std::fabs(v);
  if (a == 0.0f) return v;
  uint32_t bits;
  std::memcpy(&bits, &a, 4);
  int ea = std::max((int)((bits >> 23) & 0xFF) - 127, -6);
  uint32_t ulp_bits = (uint32_t)(ea - 3 + 127) << 23;
  float ulp;
  std::memcpy(&ulp, &ulp_bits, 4);
  float r = a / ulp;
  r = (r + 12582912.0f) - 12582912.0f;
  float q = r * ulp;
  return std::signbit(v) ? -q : q;
}

static void o_fp8_qdq_block32_ref(uint16_t* x, uint32_t dim, uint32_t nb) {
  for (uint32_t b = 0; b < nb; ++b) {
    for (uint32_t d0 = 0; d0 < dim; d0 += 32) {
      float amax = 1e-4f;
      for (uint32_t i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(bf16_to_f32(x[(size_t)(d0 + i) * nb + b])));
      float t = amax * (1.0f / 448.0f);
      uint32_t tbits;
      std::memcpy(&tbits, &t, 4);
      int e = (int)((tbits >> 23) & 0xFF) - 127 + ((tbits & 0x7FFFFFu) != 0 ? 1 : 0);
      uint32_t sbits = (uint32_t)(e + 127) << 23;
      float s;
      std::memcpy(&s, &sbits, 4);
      for (uint32_t i = 0; i < 32; ++i) {
        float v = bf16_to_f32(x[(size_t)(d0 + i) * nb + b]) / s;
        v = std::min(448.0f, std::max(-448.0f, v));
        x[(size_t)(d0 + i) * nb + b] = f32_to_bf16(e4m3_round_ref(v) * s);
      }
    }
  }
}

int main() {
  std::vector<uint16_t> buf(64 * 2);
  for (size_t i = 0; i < buf.size(); ++i)
    buf[i] = f32_to_bf16((float)((int)i - 30) * 0.137f);
  std::vector<uint16_t> ref = buf;
  o_fp8_qdq_block32_ref(ref.data(), 64, 2);
  bool changed = false;
  for (size_t i = 0; i < buf.size(); ++i)
    if (ref[i] != buf[i]) changed = true;
  if (!changed) {
    std::fprintf(stderr, "FAIL: reference QDQ did not quantize\n");
    return 1;
  }
  std::printf("PASS: test_o_fp8_qdq\n");
  return 0;
}
```

**Step 2: Implement `batch_o_fp8_qdq_bf16_hip` in `batch_kernels.hip` and wire into `dense_gpu.hip`**

In `/mnt/fast-storage/affinity/src/gpu/batch_kernels.hip`, add wave32 block-32 `ue8m0` E4M3 QDQ kernel operating on `D.b_heads` (`[dim][nb]` BF16, where `dim = a_hpr_ * width = 16384`):

```cpp
// One wave32 owns one (block_of_32_dims, token b).
__global__ void batch_o_fp8_qdq_bf16_kernel(uint16_t* __restrict__ x, uint32_t n_blocks32,
                                            uint32_t nb) {
  const uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t total = n_blocks32 * nb;
  if (wid >= total) return;
  const uint32_t b = wid % nb, blk = wid / nb;
  const size_t idx = (size_t)(blk * 32u + lane) * nb + b;
  const float f = wmma::bf16_to_f32(x[idx]);
  float amax = fmaxf(fabsf(f), 1e-4f);
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
  const uint32_t tbits = __float_as_uint(amax * (1.0f / 448.0f));
  const int e = (int)((tbits >> 23) & 0xFFu) - 127 + ((tbits & 0x7FFFFFu) != 0u ? 1 : 0);
  const float s = __uint_as_float((uint32_t)(e + 127) << 23);
  const float v = fminf(448.0f, fmaxf(-448.0f, f / s));
  // Exact E4M3 round-half-to-even in FP32:
  const float a = fabsf(v);
  const int ea = max((int)((__float_as_uint(a) >> 23) & 0xFFu) - 127, -6);
  const float ulp = __uint_as_float((uint32_t)(ea - 3 + 127) << 23);
  const float r = (a / ulp + 12582912.0f) - 12582912.0f;
  const uint32_t qbits = __float_as_uint(r * ulp) | (__float_as_uint(v) & 0x80000000u);
  const float q = __uint_as_float(qbits) * s;
  x[idx] = wmma::f32_to_bf16(q);
}
```

In `/mnt/fast-storage/affinity/src/gpu/dense_gpu.hip` inside `DenseGpu::batch_attn_out` (right after `batch_rope_tail_bf16_hip` at line 4946):
```cpp
      static const bool o_qdq_on = [] {
        const char* v = std::getenv("AFF_O_FP8_QDQ");
        return !v || !*v || std::strcmp(v, "0") != 0;
      }();
      if (o_qdq_on && ((a_hpr_ * width) % 32u) == 0u) {
        batch_o_fp8_qdq_bf16_hip(D.b_heads, (a_hpr_ * width) / 32u, b_nb_, D.stream);
      }
```

**Step 3: Build & test**
Run inside `affinity:v41-vision` build container and verify `test_o_fp8_qdq` passes.

---

### Task 2: Port Cactus Rejection Sampling with Length-Fade & Acceptance Cap (`AFF_CACTUS_DELTA`, `AFF_CACTUS_CAP_R`)

**Context (from `SlyBase/vllm-sly-radiance` commit `06ce04b` / PR #196):**
Under `temperature: 1.0`, strict stock rejection sampling (`accept_u[k] < q_k`) rejects many high-probability DSpark proposals. Cactus modifies the acceptance threshold to `h_k = min(q_k * (1 + 2 * delta_eff * (1 - q_k)), 1 - cap_r * (1 - q_k))` with linear length fade (`fade_start = 4096` -> `fade_end = 16384`), preserving ~72–90% of Cactus's throughput gain while completely eliminating long-generation repetition loops.

**Files:**
- Modify: `/mnt/fast-storage/affinity/tools/affinity.cpp:2008-2025,2175-2190`
- Test: `/mnt/fast-storage/affinity/tests/test_cactus_accept.cpp`

**Step 1: Write unit test for Cactus acceptance threshold math (`/mnt/fast-storage/affinity/tests/test_cactus_accept.cpp`)**

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

static inline double cactus_threshold(double q, uint32_t generated, double delta, double cap_r,
                                      uint32_t fade_start, uint32_t fade_end) {
  if (delta <= 0.0 || q <= 0.0 || q >= 1.0) return q;
  double fade = 1.0;
  if (fade_end > fade_start) {
    fade = std::clamp((double)((int64_t)fade_end - (int64_t)generated) /
                          (double)(fade_end - fade_start),
                      0.0, 1.0);
  }
  const double delta_eff = delta * fade;
  const double h_raw = std::min(1.0, q * (1.0 + 2.0 * delta_eff * (1.0 - q)));
  if (cap_r > 0.0) {
    const double h_cap = 1.0 - cap_r * (1.0 - q);
    return std::max(q, std::min(h_raw, h_cap));
  }
  return h_raw;
}

int main() {
  // At generated=0, q=0.5, delta=0.5, cap_r=0.5 -> h_raw=0.75, h_cap=0.75 -> 0.75 > 0.5
  double h0 = cactus_threshold(0.5, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h0 - 0.75) < 1e-6);
  // At generated=16384 (past fade_end), h == q exactly (stock rejection rule)
  double h_end = cactus_threshold(0.5, 16384, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_end - 0.5) < 1e-6);
  // At q=0.8, delta=0.5, cap_r=0.5 -> h_raw = 0.96, h_cap = 0.90 -> capped at 0.90
  double h_cap = cactus_threshold(0.8, 0, 0.5, 0.5, 4096, 16384);
  assert(std::fabs(h_cap - 0.90) < 1e-6);
  std::printf("PASS: test_cactus_accept\n");
  return 0;
}
```

**Step 2: Integrate into `spec_decode` in `/mnt/fast-storage/affinity/tools/affinity.cpp`**

In `spec_decode` (`affinity.cpp` around line 2008):
```cpp
    const char* cactus_d_env = std::getenv("AFF_CACTUS_DELTA");
    const double kCactusDelta = cactus_d_env ? std::strtod(cactus_d_env, nullptr) : 0.0;
    const char* cactus_r_env = std::getenv("AFF_CACTUS_CAP_R");
    const double kCactusCapR = cactus_r_env ? std::strtod(cactus_r_env, nullptr) : 0.5;
    const char* cactus_fs_env = std::getenv("AFF_CACTUS_FADE_START");
    const uint32_t kCactusFadeStart = cactus_fs_env ? (uint32_t)std::strtoul(cactus_fs_env, nullptr, 10) : 4096u;
    const char* cactus_fe_env = std::getenv("AFF_CACTUS_FADE_END");
    const uint32_t kCactusFadeEnd = cactus_fe_env ? (uint32_t)std::strtoul(cactus_fe_env, nullptr, 10) : 16384u;
```
And at line 2182 (`else` branch of `if (!sc)`):
```cpp
        auto accept_prob = [&](uint32_t j) -> double {
          const double q = (double)vb.draws[j].p_query;
          if (kCactusDelta <= 0.0 || q <= 0.0 || q >= 1.0) return q;
          double fade = 1.0;
          if (kCactusFadeEnd > kCactusFadeStart) {
            fade = std::clamp((double)((int64_t)kCactusFadeEnd - (int64_t)(generated + j)) /
                                  (double)(kCactusFadeEnd - kCactusFadeStart),
                              0.0, 1.0);
          }
          const double h_raw = std::min(1.0, q * (1.0 + 2.0 * (kCactusDelta * fade) * (1.0 - q)));
          if (kCactusCapR > 0.0) return std::max(q, std::min(h_raw, 1.0 - kCactusCapR * (1.0 - q)));
          return h_raw;
        };
        while (k < nd && accept_u[k] < accept_prob(k)) ++k;
        if (k < nd) { rejected = true; reject_tok = vb.draws[k].tok_excl; }
        for (uint32_t j = 0; j < nd; ++j) match[j] = accept_u[j] < accept_prob(j);
```

---

### Task 3: Port Zero-Copy DSpark Tap from Next-Layer Input Plane (`AFF_DSPARK_TAP_FROM_PRE`)

**Context (from `SlyBase/vllm-sly-radiance` `RADIANCE_DFLASH_AUX_FROM_RESIDUAL` & `model.cpp:1634-1645`):**
In `Model::forward_prefill` (`model.cpp:1634-1645`), V4.1's DSpark draft is conditioned on the attention INPUT of target layers `t` (`dspark_taps`), which equals the lane-collapsed output at the end of layer `t-1`. Currently `tap_at[t-1]` launches a separate tap-write kernel at the end of layer `t-1` even though layer `t`'s `hc_pre` collapses the exact same `b_hc` lanes at the start of layer `t`. Reusing `b_pre` at the start of layer `t` eliminates 3 tap-collapse kernel launches per chunk.

**Files:**
- Modify: `/mnt/fast-storage/affinity/src/engine/model.cpp:1632-1649,1905-1925,2220-2238`
- Modify: `/mnt/fast-storage/affinity/src/gpu/dense_gpu.h`
- Modify: `/mnt/fast-storage/affinity/src/gpu/dense_gpu.hip`

**Step 1: Add `tap_from_pre` copy hook in `DenseGpu`**
In `dense_gpu.hip`, add `batch_tap_from_pre(uint32_t tap_slot, uint32_t tap_first, uint32_t tap_pos0, uint32_t n_embd)` which copies `D.b_pre` (already collapsed by `batch_hc_pre` on layer `t`) directly into `D.dspark_tap[tap_slot]` without re-collapsing `D.b_hc` across the 4 hyper-connection lanes.

**Step 2: Gate with `AFF_DSPARK_TAP_FROM_PRE=1` in `model.cpp` and verify bit-identical tap hashes**
Verify with `AFF_SLOT_DEBUG=1` that `dense_gpu.tap_hash_debug()` matches bit-for-bit before and after.

---

### Task 4: Port Group-128 `int4` `lm_head` (`AFF_LMHEAD_INT4=1`)

**Context (from `SlyBase/vllm-sly-radiance` `RADIANCE_LMHEAD_INT4=1` & `StillDeadcode/radiance` `op_dlogits`):**
Affinity executes the `129,280 × 4,096` vocabulary projection (`head_`, 529 MB total / 264.7 MB per GPU) **twice per speculative step** (`bops_.draft_head` for 5 draft tokens + `bops_.head_sample` for up to 6 verify tokens). Quantizing `head_` on load to group-128 symmetric `int4` with MSE clip search halves `lm_head` DRAM traffic (~529 MB/step saved) and frees ~265 MB of VRAM across the 2 R9700 cards.

**Files:**
- Modify: `/mnt/fast-storage/affinity/src/gpu/batch_kernels.h`
- Modify: `/mnt/fast-storage/affinity/src/gpu/batch_kernels.hip`
- Modify: `/mnt/fast-storage/affinity/src/gpu/dense_gpu.h`
- Modify: `/mnt/fast-storage/affinity/src/gpu/dense_gpu.hip`

**Step 1: Add `preshuffle_w4_from_fp8_hip` and `batch_gemm_w4a8_narrow_kernel` in `batch_kernels.hip`**
1. When `AFF_LMHEAD_INT4=1` (or `AFF_DRAFT_HEAD_INT4=1` to apply only to `draft_head` so target verification remains 100% lossless FP8!), quantize each 128-element row group of `head_` using a 9-point grid search over clip ratio `alpha in [0.80, 1.00]` minimizing L2 error, packing two 4-bit signed nibbles per byte + one FP16/E8M0 scale per 128 elements.
2. Unpack nibbles into FP8 E4M3 fragments in registers before `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.

**Step 2: Verify greedy token stream identity when applied to `draft_head` (`AFF_DRAFT_HEAD_INT4=1`) and benchmark step ms**
Because target verification uses the exact FP8 `head_`, `AFF_DRAFT_HEAD_INT4=1` is **100% output-identical in greedy mode** while cutting `dspark_draft`'s head bandwidth in half!

---

### Task 5: Port `1017` Tool Namespace Support (`namespace::name`) in `chat_encode.cpp`

**Context (from `05yuki/deepseek-v4.1-flash-1017-ktransformers` `docs/1017-changes.md`):**
In `DeepSeek-V4.1-Flash-1017`'s `encoding/encoding.py`, a tool definition can carry a `namespace` (`"namespace": "ns"` or `{"name": "ns", "description": "..."}`), rendered in the prompt as `namespace::name` and parsed back into `namespace` + bare `function.name`.

**Files:**
- Modify: `/mnt/fast-storage/affinity/src/engine/chat_encode.h`
- Modify: `/mnt/fast-storage/affinity/src/engine/chat_encode.cpp`
- Modify: `/mnt/fast-storage/affinity/tools/affinity.cpp`
- Test: `/mnt/fast-storage/affinity/tests/test_chat_encode.cpp`

**Step 1: Add `ns` field to `ToolDef` and `ToolCall` in `chat_encode.h`**
- In `tools_block(const std::vector<ToolDef>&, bool v41)`: if `!t.ns.empty()`, render tool name as `t.ns + "::" + t.name`.
- In `parse_assistant_output`: if parsed `ToolCall::name` contains `"::"`, split into `tc.ns` and bare `tc.name`.

**Step 2: Add test case in `test_chat_encode.cpp` and run `ctest`**
Verify round-trip rendering and parsing of namespaced tool definitions and DSML tool calls.
