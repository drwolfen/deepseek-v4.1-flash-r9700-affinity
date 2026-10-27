# Changelog

All notable changes to the `deepseek-v4.1-flash-r9700-affinity` setup and inference engine are documented here.

## [v1.1.0] - 2026-10-27

### Summary & Measured Improvements
Measured in production on dual AMD Radeon AI PRO R9700 (`gfx1201`) across live agent sessions (Hermes, `session_id=20261025_122836_76edaab3`) comparing the pre-restart baseline (`145k–162k` tokens context, $n=47$ steps) against the new configuration (`163.5k–170.7k` tokens context, $n=9$ steps).

- **+28.6% Weighted Decode Throughput:** Increased from **17.08 tok/s** (13,010 tokens in 761.9s) to **21.96 tok/s** (3,907 tokens in 178.0s), with single-step peaks reaching **27.80 tok/s** (1,341 tokens in 48.15s at 166.3k context).
- **-20.4% Speculative Step Latency:** Median block latency dropped from **209.5 ms/block** to **166.7 ms/block** (-42.8 ms/block), with minimum block latency reaching **156.8 ms/block** (weighted mean latency dropped -15.6% from 198.9 ms/blk to 167.9 ms/blk).
- **+8.5% Speculative Yield:** Tokens accepted per block increased from **3.40 tok/block** to **3.69 tok/block** (acceptance rate up from 78.8% to 79.9%), eliminating low-acceptance tail collapse (<55% acceptance collapses completely resolved).
- **3.88x Memory Bandwidth Reduction on Draft LM Head:** On-GPU INT4 quantization (`int4_g128`) compressed the output head from 631.2 MiB/card (BF16) to **162.7 MiB/card** (-468.5 MiB/card per GPU per speculative step).

---

### Features & Ports Added

#### 1. On-GPU Group-128 Symmetric INT4 Output Head (`AFF_DRAFT_HEAD_INT4=1` / `AFF_LMHEAD_INT4=1`)
- **Origin:** Radiance `v1.12.0` draft-head compression and `05yuki/deepseek-v4.1-flash-1017-ktransformers`.
- **Implementation:** Added 9-point MSE-clipped `quantize_bf16_to_w4_g128_kernel` and `matvec_w4_multix_kernel` in `src/gpu/batch_kernels.hip`. Quantizes `output` tensor (`129280 × 5120`) dynamically at startup in <10 ms to group-128 INT4.
- **Verification:** Cosine similarity against BF16 reference measured at `0.999557`. Draft reads drop from 631.2 MiB to 162.7 MiB per card.

#### 2. Fused `hc_post` + DSpark Taps (`AFF_FUSE_HC_TAP=1`)
- **Origin:** Radiance `rad_hc_fuse` pipeline.
- **Implementation:** Extended `batch_hc_expand_reg_kernel` in `src/gpu/batch_kernels.hip` and added `DenseGpu::batch_hc_post_tap` in `src/gpu/dense_gpu.hip`. Fuses hidden-compression expansion with DSpark tap recording (layers 37, 38, 39) into a single kernel pass.
- **Verification:** Eliminates 3 kernel launches and 3 `b_hc` VRAM reads per step; `memcmp == 0` bit-identical match verified on `gfx1201`.

#### 3. Cactus Length-Faded Capped Rejection Sampling (`AFF_CACTUS_DELTA=0.35`, `AFF_CACTUS_CAP_R=0.5`)
- **Origin:** Radiance `v1.12.0` Cactus sampler.
- **Implementation:** Integrated length-dependent fade and acceptance capping in `tools/affinity.cpp`. Rejection probability fades smoothly between `AFF_CACTUS_FADE_START` (4,096 tokens) and `AFF_CACTUS_FADE_END` (16,384 tokens), preventing runaway reject cascades in long reasoning traces.

#### 4. Opt-in FP8 Output Activation QDQ (`AFF_O_FP8_QDQ=1`)
- **Origin:** DeepSeek-1017 KTransformers attention optimization.
- **Implementation:** Implemented `batch_o_fp8_qdq_bf16_kernel` and `o_fp8_qdq_f32_kernel` in `src/gpu/batch_kernels.hip` and wired into `DenseGpu::batch_attn_out`. Quantizes attention projection outputs directly to FP8. Opt-in to preserve exact bit-compatibility with existing on-disk persistent prefix cache (`pcache`) blocks.

#### 5. 1017 Tool Namespace Encoding & Parser (`namespace::name`)
- **Origin:** DeepSeek-1017 tool calling specification.
- **Implementation:** Added `apply_tool_namespace`, `render_tool_calls`, and `parse_completion` in `src/engine/chat_encode.cpp`, `src/engine/chat_encode.h`, `src/server/server.cpp`, `src/server/server.h`, and `tools/affinity.cpp`. Supports double-colon tool routing without corrupting DSML XML tool tags.

---

### Configuration & Service Updates
- **Systemd Service (`deploy/deepseek-v41-affinity.service`):**
  - Updated model targets to uncensored `model-uc-1017.aff`.
  - Added `-e AFF_FUSE_HC_TAP=1`.
  - Added `-e AFF_DRAFT_HEAD_INT4=1`.
  - Added `-e AFF_CACTUS_DELTA=0.35` and `-e AFF_CACTUS_CAP_R=0.5`.
  - Added `--persistent-prefix-cache` for instant prompt restoration across restarts.

---

## [v1.0.0] - 2026-10-17

- Initial release of R9700-tweaked affinity fork for DeepSeek-V4.1-Flash (uncensored 2.875-bpw).
- Dual R9700 `gfx1201` support, CPU ViT with AVX-512 GEMM, per-source compressed KV cache, DSpark draft head, engram sidecar, learned placement persistence (`--placement-save`), and bare DSML tool-call boundary recovery.
