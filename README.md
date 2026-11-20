# deepseek-v4.1-flash-r9700-affinity

A production setup for serving **DeepSeek-V4.1-Flash** (uncensored / abliterated `MoonMath-1017` build) on **dual AMD Radeon AI PRO R9700 GPUs** (`gfx1201`, RDNA 4, Wave32, TP=2), built on a tuned fork of the [affinity](https://codeberg.org/StillDeadcode/affinity) inference engine paired with a locally quantized 2.875-bpw `.aff` model container.

This repository captures the full stack: engine source (RDNA 4 HIP kernels + host runtime + server), `Dockerfile` (`ROCm 10.1.0-full` default), systemd/Docker deployment manifests, quantization recipes, unit/kernel regressions, and paired benchmarks measured on dual R9700 hardware.

> **Hardware scope:** Every benchmark and profile below was measured on a two-card **Radeon AI PRO R9700** (`2 × 32 GiB`, TP=2) system. Upstream supports single-card execution, which was not benchmarked here.

---

### Model Weights & Artifacts

- **MoonMath 1017 (`.aff` + `.dspark.aff` + `.engram`):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-MoonMath-aff](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-MoonMath-aff)
- **1017 revision:** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw)
- **Legacy 0925 revision:** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw)

**Documentation & Release Links:** [**`CHANGELOG.md`**](CHANGELOG.md) | **Latest Release:** [**`v5.3.0`**](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/releases/tag/v5.3.0) | **Benchmark Reports:** [**`v5.3.0 HIP Optimization`**](docs/v5.3.0-hip-kernel-optimization-benchmark.md) · [**`v5.2.0 Optimization`**](docs/benchmarks/2026-11-18-v5.2-deepseek-optimization.md) · [**`v5.1.0 & MoonMath-1017`**](docs/benchmarks/2026-11-16-v5.1-and-moonmath-1017-bakeoff.md) · [**`v5.0.0 & Radiance v2.1`**](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md)

---

## Table of Contents

1. [Release Highlights (Waves 0–6)](#release-highlights-waves-06)
2. [Architecture, Memory Topology & Hardware Platform](#architecture-memory-topology--hardware-platform)
3. [What's New in v5.3.0: RDNA 4 (`gfx1201`) HIP Kernel Deep Dive](#whats-new-in-v530-rdna-4-gfx1201-hip-kernel-deep-dive)
4. [v5.3.0 3-Arm Paired Benchmark & ROCm 10.1 vs 10.2 Evaluation](#v530-3-arm-paired-benchmark--rocm-101-vs-102-evaluation)
5. [Measured Performance Progression (Waves 0–6)](#measured-performance-progression-waves-06)
6. [Wave 4 Bake-Off: `affinity` vs `StillDeadcode/radiance v2.1.0`](#wave-4-bake-off-affinity-vs-stilldeadcoderadiance-v210)
7. [Rework and Patches (`v1.0.0` → `v5.3.0`)](#rework-and-patches-v100--v530)
8. [Production Safety Invariants & Environment Variable Reference](#production-safety-invariants--environment-variable-reference)
9. [Build, Test, Deploy & Rollback Guide](#build-test-deploy--rollback-guide)
10. [Credits](#credits)

---

## Release Highlights (Waves 0–6)

- **v5.3.0** (`dfb34cb`, `c20ed32`, `e1f6be3`, `d64eea9`): RDNA 4 (`gfx1201`, Wave32) HIP kernel optimizations across Wave A (`lds_barrier()`, DPP16 `dpp_xor_lane<Lane>()`, `AFF_WAVES(n)`, `safe_grid_blocks()`, vectorized HC all-reduce, async `MoePlanStatus` readback, packed token-major routing/logits) and Wave B (`AFF_FP8_PREFILL_4X2`, `AFF_FP8_SPLIT_INLINE`, `AFF_TP_AR_STAGED`, `AFF_FLASH_MIN_TILES`, `flash_kernel<2>` `occ=6` register/LDS tightening, `idx_quant_kernel` spill elimination, contiguous multi-GEMM output band + WGP `agent` acquire fence fix). Paired 3-arm benchmark on ROCm `10.1.0-full` (winning over ROCm `10.2.0-full` by **`+3.55%`** composite) reaches **41.47–42.23 tok/s** decode at 16K–64K (**`+24.9%` to `+43.8%` `tg`** over `v5.2.0`), **`+37.05%`** slot `blocks/s` (`5.586 -> 7.656`), **`21.33 tok/s`** 4-prompt DSpark mean (`+34.9%`), **`-34.0%`** `dense-gpu` device time, **`-23.0%` / `-23.7%`** `routed-expert` device time, and **`-85.2%`** `x_gu + x_down` at **`3,517`** resident experts (`[353, 355] MiB` idle free VRAM, `2.800 s` KFD release). Verified **13/13 filtered CPU `ctest` + 5/5 GPU binaries (ROCm 10.1 & 10.2) + 2/2 fallback-toggle runs PASS** (`0` `hipError`, `0` new ELF spills). See [`docs/v5.3.0-hip-kernel-optimization-benchmark.md`](docs/v5.3.0-hip-kernel-optimization-benchmark.md).
- **v5.2.0** (`44e81ee`): Reclaimed dead V4.1 Q-RMS/compressor VRAM and tiled Engram H2D/gather (`-616 MiB/card`), expanding GPU-resident MoE capacity by **+101 experts (`3,416 -> 3,517`)** at `[355, 354] MiB` free VRAM (`0` OOM/`0` spill). Fused `attn_out` prep (`AFF_FUSED_ATTN_OUT_PREP`), specialized `kFKNarrow=128` FP8 multi-GEMM (`AFF_FP8_NARROW128`), reused norm Q8 (`AFF_REUSE_NORM_Q8`), skipped decode Engram threadpool overhead (`live < 64`), and synchronized prefill before decode -> **41.00–41.72 tok/s** warm Pass-2 decode (`+30.2%` to `+56.1%` over `v5.1.2`), **`+6.8%` to `+42.0%`** cold Pass-1 decode (`21.33–38.74 tok/s`), **`+5.2%` to `+12.6%`** prefill, **`+14.4%` to `+17.3%`** 4-prompt `temp=0` DSpark decode, and **`-47.0%` / `-55.9%`** cached multi-turn TTFT (`64K -> 100K`). Hardened OpenAI SSE serving with `fd_alive` disconnect probing (`<=1.13 s` FIFO drop, `0` wasted tokens), 15 s empty-delta SSE keepalives, `PrefillAlive` abort hook, O(1) tokenizer decode + BPE/LCP caching, and hardened delta prefix-cache lookup. **4/4 GPU kernel suites + 9/9 CPU `ctest` suites PASS**. See [`docs/benchmarks/2026-11-18-v5.2-deepseek-optimization.md`](docs/benchmarks/2026-11-18-v5.2-deepseek-optimization.md).
- **v5.1.2** (`c52e95e`): Ported Lucebox `ThinkingBudget` (default `--thinking-reply-reserve 4096` / `AFF_THINK_REPLY_RESERVE`, `--thinking-loop-repeats 3` / `AFF_THINK_LOOP_REPEATS` over `loop_span=64`), added the Executive-Freeze Churn Breaker (`--thinking-freeze-breaker 1` / `AFF_THINK_FREEZE_BREAKER`, triggers `force_close` after 512 thinking tokens when trailing 256 tokens contain >=64 newline-group tokens and <=36 distinct IDs), synchronized `spec_decode` DSpark seeding/rollback across forced `</think>` injection (`\n</think>\n\n`), and added cache-preserving `reasoning_effort: "none"` (`"off"`/`"0"`) override (`suppress_final_think`) that preserves `idx == 0` effort prompts and historical `<think>...</think>` byte-identical across all `n_pre` tokens while closing only the final `<think>` (`128821` -> `128822`). 7/7 CPU unit tests PASS.
- **v5.1.1** (`63cd104`): Hotfix for multi-region/multi-rank (`TP=2`) delta prefix-cache restore offsets (`PrefixCache::restore()` indexes `impl_->rbuf[0]` via the full `B`-token `region_span` layout and per-rank stride across heterogeneous `app_` regions; on-disk `pcache` remains intact without wiping) and strict UTF-8 streaming holdback (`utf8_safe_end`), EOS/final-flush sanitization (`sanitize_utf8`), and validating `json_escape` (`U+FFFD` replacement for truncated/malformed sequences). 6/6 CPU unit tests PASS.
- **v5.1.0** (`839665f`): DSpark ring-attention tile skip (`AFF_DRAFT_SKIP_MASKED=1`, **+22.2%** decode speedup: 16.93 -> 20.69 tok/s), paired `temp=0` DSpark acceptance **`73.33%`** on `MoonMath-1017` (vs 67.52% on v5.0.0, **+5.81 pp**; **80.0%** on code, **+9.0%** paired mean gen), compact state-only delta prefix cache (`AFF_PCACHE_DELTA=1`, 1.42× multi-turn TTFT speedup), `ignore_eos=true` raw streaming, GPU `gridDim.y > 65535` overflow guard (#1006), DSML nested XML tool-argument protection and multi-call invoke splitting (#1020), and MoE stream RAW barrier validation (#983, #984). Retained `AFF_ATTN_LOOKAHEAD=0` default (-11.4% regression when enabled). 6/6 CPU unit tests PASS.
- **v5.0.0** (`264e8d5`): RDNA 4 `gfx1201` sparse indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`), `K_VECS_PADDED=136` + 8B Q staging, 4-pass 256-bucket radix top-512, Wave32 tie compaction, live-`seqused` ragged causal prefill, 36 MiB score scratch budget -> **3,416 resident experts (+49)**, **14.40× @64K** & **13.15× @128K** prefix-reused prefill, Needle **4/4 (100%)**, `refattn.py` **99.98%** parity, `quant_eval` **92.5%** (111/120), 9/9 tests PASS.
- **v4.5.0** (`9ec7c47`): Fused & chained hybrid compute (`AFF_HC_FUSED_V2=1`, `AFF_HC_CHAINED=1`, `AFF_HC_CHAINED_MIN_ROWS=1`) -> **`23.73 tok/s` decode median (`24.37` peak)**, **`22.4 tok/s` batch-8 aggregate**, `0.000000 nats` 8,184-tok KL (`PPL 2.6057`), `90.83%` `quant_eval`, 8/8 tests PASS.
- **v4.4.0** (`718b42e`): Upgrade to ROCm 10.1 (`20.41 vs 18.49 tok/s`, **+10.4%**), `AFF_CACTUS_ESCAPE=0.25` (**`0/20` repetition lock-in** @ `temp=0.6`), 64 GiB persistent prefix cache, Issue #12 `heat -> static` fallback, cross-shard `.scale` fix, DSML typed-arg parity.
- **v4.3.3** (`5275c64`): Unified 3-way branch history; `GET /health`, `--raw-completions`, `cached_tokens` reporting; 7/7 tests PASS.
- **Wave 4 Radiance v2.1.0 Bake-Off**: Offline ROCm 10.1 evaluation confirmed upstream open-source Radiance omits the `deepseek41` plugin, MLA/Engram kernels, and `F8_E8M0` scales.

---

## Architecture, Memory Topology & Hardware Platform

### Reference Machine

| Component | Specification |
|---|---|
| GPUs | **2 × AMD Radeon AI PRO R9700**, `gfx1201` (RDNA 4, Wave32, TP=2), 32 GiB VRAM each (`32,624 MiB` total reported/card) |
| CPU | **2 × Intel Xeon Platinum 8160** (Skylake-SP), 48 cores / 96 threads total (`--cpuset-cpus=0-23,48-71` pinned for serving) |
| Host RAM | **754 GiB DDR4** across 2 NUMA nodes |
| PCIe Interconnect | GPUs negotiate **Gen5 ×16**, switched host/peer uplink runs at **Gen3 ×16 (8.0 GT/s)** |
| Host OS | **Ubuntu 24.04.5 LTS** (Noble Numbat), kernel 7.0 |
| Container Base | **ROCm `10.1.0-full`** (`docker.io/rocm/dev-ubuntu-24.04:10.1.0-full`, `gfx1201`) |
| Image Tag | `deepseek-affinity:v5.3.0` |
| Serving Topology | OpenAI-compatible HTTP/SSE container on `:8000` (with optional local reverse proxy on `:8080`) |

### DeepSeek-V4.1-Flash Memory & Placement Architecture

DeepSeek-V4.1-Flash uses 40 layers, 384 routed experts (top-6 active per token), `hidden_size = 5120`, and `vocab_size = 129,280`. At 2.875 bpw the routed experts total ~93 GiB and the full quantized container is **191.32 GiB** — exceeding 64 GiB aggregate VRAM, so Affinity splits weights across GPU HBM/GDDR, 2 MiB transparent hugepages in host RAM, mmap'd NVMe tables, and GPU caches:

| Subsystem | Placement / Backing | Measured Footprint (`v5.3.0` Live Ledger) |
|---|---|---|
| **Dense Weights** | TP=2 sharded across both GPUs (`506` tensors; LM head BF16, DSpark head `int4_g128` `162.7 MiB/card`) | `10.01 GiB` total (`4,870 MiB/card` weights + `154 MiB` HIP ctx/staging) |
| **Routed MoE Experts (GPU)** | 2-way shards (`6.08 MiB/shard`), demand placement with `32` shadow slots; symmetric `3,493` + `24` card-1 asymmetric bonus slab | **`3,517` resident slots** (`21,978 / 22,124 MiB`, `41.4 GiB` VRAM, `22.7%` residency) |
| **Routed MoE Experts (Host Pool)** | 2 MiB THP-first pinned host pool (`AFF_POOL_THP=1`), PCIe DMA read-in-place (`CU-masked` low `16/64` CUs) | **`11,867` experts in `140.94 GiB` RAM** (`100%` 2 MiB pages, `0` on SSD tier) |
| **DSpark Speculative Drafter** | 3 blocks, block width 5, taps `[37, 38, 39]`, `384/384` draft experts resident | `5.22 GiB` (`2,446 MiB` draft expert slab + `480 MiB` dense + `4 MiB` arena per card) |
| **KV Cache & Prefill Arena** | Virtual span `327,680` positions reserved / `262,144` backed; `2816`-token batched prefill arena | `350 MiB/card` KV (`0.66 GiB` total) + `1,100 MiB/card` prefill + `440 MiB/card` compressor/indexer |
| **Engram Tables & ViT** | `32`-layer BF16 DeepSeek ViT on CPU; `188.8 GiB` Engram n-gram tables host-mmapped (`AFF_ENGRAM_WARM=1`) | `74 MiB/card` GPU + OS page cache mmap |
| **Persistent Prefix Cache** | Multi-slot NVMe state-only delta checkpoints (`AFF_PCACHE_DELTA=1`) | `64.0 GiB` configured (`512`-tok blocks, `60,505` block slots, `1,793` checkpoint slots) |
| **Total Card & Host Totals** | `free_vram_mib = [353, 355]` at load (`383 / 386 MiB` slab headroom) | **`32,008 / 32,154 MiB`** of `32,624 MiB` per card · **`141.94 GiB`** peak process RAM |

### Minimum RAM Floor

- **~160 GiB RAM** is the practical floor for this configuration (`512k` context reserved, DSpark on, Engram warm): the host routed-expert pool alone pins **`140.94–143.5 GiB`** alongside OS and model mmap buffers.
- Below that threshold, demand-paged experts spill to the SSD staging tier (4 slots / `48.6 MiB`) and generation collapses toward SSD read latency (~8 tok/s instead of ~21–42 tok/s).

---

## What's New in `v5.3.0`: RDNA 4 (`gfx1201`) HIP Kernel Deep Dive

`v5.3.0` focuses on `src/gpu/*` kernel efficiency, RDNA 4 Wave32 hardware intrinsics, VGPR/LDS occupancy boundaries, and PCIe TP=2 communication overlap without changing model numerics (`-ffp-contract=off`, BF16 target LM head, and `pcache` fingerprint `0x733de762fee253e6` preserved).

### Wave A: Zero-Functional-Change Primitives & Launch Overhead (`dfb34cb`)

`19 files changed, +3,161 / -878` across `src/gpu/*`, `Dockerfile`, and `tests/test_gpu_ports.hip`:

- **Local-only workgroup barriers (`lds_barrier()` in [`src/gpu/hip_common.h`](src/gpu/hip_common.h))**: Replaced global-draining `__syncthreads()` at pure-LDS producer/consumer boundaries across `attention_gpu.hip`, `batch_kernels.hip`, `dense_gpu.hip`, `expert_kernel.hip`, `indexer_gpu.hip`, and `sampler_gpu.hip` with `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local")` + `s_barrier` + `ACQUIRE`, eliminating unnecessary VMEM drain stalls.
- **Wave32 DPP16 lane exchanges (`dpp_xor_lane<Lane>()` in [`src/gpu/hip_common.h`](src/gpu/hip_common.h))**: Replaced `ds_bpermute_b32` LDS crossbar shuffles with `__builtin_amdgcn_update_dpp` (`row_xmask` for XOR offsets `{1, 2, 4, 8}`) and `__builtin_amdgcn_permlanex16` (offset `16`), powering `wave32_shuffle_xor`, `wave_reduce_sum_dpp`, `wave_reduce_max_dpp`, `batch_kernels.hip`, `expert_kernel.hip`, and `indexer_gpu.hip`.
- **`gfx1201` 2048-wave cliff guard (`safe_grid_blocks()` in [`src/gpu/hip_common.h`](src/gpu/hip_common.h))**: Guards persistent/grid-stride launches against wave-slot multiples near `mod 2048` (`[2016..2047]`) across the 64 CUs × 2 SIMDs × 16 wave slots = 2,048 hardware wave slots on `gfx1201`.
- **Occupancy hints (`AFF_WAVES(n)`) & vectorized HC all-reduce ([`src/gpu/dense_gpu.hip`](src/gpu/dense_gpu.hip))**: Added `hc_expand_ar_vec_kernel` and `hc_add_ar_vec_kernel` (`float4` vectorized global/peer scratch ops + `__builtin_amdgcn_s_sleep(1)` spin-wait backoff).
- **Zero-padding / packing & transfer cuts ([`src/gpu/dense_gpu.hip`](src/gpu/dense_gpu.hip))**:
  - `route_publish` + `pack_route_wt_kernel`: packs live `(sel, wt)` pairs tightly when `2 * b_live_ <= b_cap_` rather than pushing full-capacity padded route buffers over PCIe.
  - `batch_head_sample` (`pack_tm`, `head_pack_tm_kernel`, `head_unpack_tm_kernel`): packs `[nr, n]` logits token-major into `b_q` scratch on `n < b_nb_` before rank-to-rank `hipMemcpyPeerAsync` (shrinking decode cross-GPU logit copies when `n=1..5` inside `b_nb_=16`).
  - `draft_transpose_pad_kernel`: fuses zero-fill + transpose into `b_dx` for DSpark projections.
  - `quant_act_swiglu`: reuses `b_q` arena scratch when aligned (`dim % 128 == 0`) to cut temporary buffer traffic.
  - `batch_begin`: zeroes only the `[n_live, nb)` padding tail after `hipMemcpyAsync` rather than re-zeroing `[0, nb)` before host-to-device copy.
  - Flash-path indexer (`indexer_one` / `batch_indexer`): skips redundant byte-mask writes when attention consumes compact admission lists directly (`gather && flash_ok`).
- **Async placement check, keepalive & vmem ([`src/gpu/static_placement.hip`](src/gpu/static_placement.hip), [`src/gpu/keepalive.hip`](src/gpu/keepalive.hip), [`src/gpu/vmem.hip`](src/gpu/vmem.hip))**:
  - Replaced blocking per-512-layer `hipMemcpy(&S, plan_st_[dv], ...)` `MoePlanStatus` readback with non-blocking pinned `hipHostMalloc` + `hipEventQuery` (`plan_h_st`), skipped pooled expert kernel launch groups on cards/layers with zero pooled residents (`pooled_layer_cnt`), and added `lead_gate` idle bypass (`AFF_LEAD_GATE_IDLE=0` default).
  - Wired `keepalive_notify_active()` into dense entrypoints so the low-priority GPU keepalive thread skips redundant launches during active compute, and switched aligned `VirtualSpan::commit_to` zeroes to `hipMemsetAsync`.
- **Parameterized `Dockerfile`**: `ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:10.1.0-full` enables side-by-side ROCm 10.1 vs 10.2 container builds.

### Wave B: FP8 4×2 Prefill, Inline Split-K, Staged TP2 All-Reduce & VGPR/LDS Tuning (`c20ed32`)

`6 files changed, +801 / -136` across [`src/gpu/batch_kernels.hip`](src/gpu/batch_kernels.hip), [`src/gpu/tp_allreduce.hip`](src/gpu/tp_allreduce.hip), [`src/gpu/attn_flash.hip`](src/gpu/attn_flash.hip), [`src/gpu/indexer_gpu.hip`](src/gpu/indexer_gpu.hip), `tests/test_gpu_ports.hip`, and `tests/test_wave2_kernels.hip`:

- **FP8 `4×2` Prefill Wave Layout (`AFF_FP8_PREFILL_4X2=1`, [`src/gpu/batch_kernels.hip`](src/gpu/batch_kernels.hip))**: Reorganizes prefill FP8 GEMM (`NT=8, MW=8, FK=64, MTW=2`) from `8×1` waves to `4(M) × 2(N)` waves per workgroup (halving LDS `s_b` fragment reads per wave), adds `kGroupM=8` M-band tile swizzle across `gridDim.y` for MALL weight cache reuse, and uses clamped unconditional K-tile prefetches on `Full` shapes.
- **Inline Split-K Finisher (`AFF_FP8_SPLIT_INLINE=1`, [`src/gpu/batch_kernels.hip`](src/gpu/batch_kernels.hip))**: Adds a fixed 4 KiB (`kSplitCtrFloats = 1024`) zeroed arrival-counter header (`splitk_buf_with_ctr`) in split-K workspaces so the final arriving K-slice reduces partial `[z, rows, nb]` planes directly into `FinalY` (removing separate `splitk_reduce_kernel` launch overhead and allowing split-K even when `accum == true`).
- **Staged TP=2 FP8 All-Reduce (`AFF_TP_AR_STAGED=1`, `AFF_TP_AR_STAGES=8`, [`src/gpu/tp_allreduce.hip`](src/gpu/tp_allreduce.hip))**: Introduces `tp_ar_fp8_staged_kernel` with strided flag slots (`kTpFlagStride`) to pipeline `S=8` contiguous 128-element quantize/PCIe BAR push stages with prior-stage peer fold for TP=2 payloads `>= 4 MiB` (`kTpStagedMinBytes`).
- **Flash Attention Occupancy 5 -> 6 & Indexer Zero-Spill (`AFF_FLASH_MIN_TILES`, [`src/gpu/attn_flash.hip`](src/gpu/attn_flash.hip), [`src/gpu/indexer_gpu.hip`](src/gpu/indexer_gpu.hip))**: Aliases `s_af` onto `s_red`, computes softmax in-place inside `acc[NT][8]`, and removes per-element `pw[8]` staging in `flash_kernel<2>` to hold **235 VGPRs** (ROCm 10.1) / **240 VGPRs** (ROCm 10.2) and **21,392 B LDS** (**6 waves/SIMD**); removes `reg_v` array spilling in `idx_quant_kernel` (`144 B -> 0 B` scratch).

### Post-Wave B Multi-GEMM Band & WGP Acquire Fence Fix (`e1f6be3`)

`2 files changed, +177 / -2`:

- **Contiguous multi-GEMM output band fix ([`src/gpu/batch_kernels.hip`](src/gpu/batch_kernels.hip))**: `dense_gpu.hip` (`batch_attn_qkv` and `batch_shared_ffn`) passes `jobs[0].y = D.b_g` and `jobs[1].y = nullptr` expecting contiguous band offsets from `jobs[0].y`. Restored `J.y[i] = (ws ? ws : jobs[0].y) + (size_t)off * nb;` in both `batch_gemm_multi_hip` and `batch_gemm_fp8_multi_hip`, preventing a `0x0` null-pointer fault when `ksplit == 1` (`ws == nullptr`).
- **RDNA 4 WGP L0 acquire fence (`gemm_fp8_body`)**: Replaced `__threadfence()` after the inline split-K atomic counter reset with `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` under `__HIP_DEVICE_COMPILE__`, ensuring both CUs sharing a RDNA 4 WGP invalidate L0 vector caches before reading sibling K-slices.
- **Null-band regression suite ([`tests/test_wave2_kernels.hip`](tests/test_wave2_kernels.hip))**: Added `fp8_multi_null_band` (`1536+576` `ksplit==1` matching Layer 0 `attn_qkv` and `256+256` `ksplit>1` at `ps={0,1}`) and `bf16_multi_null_band` (`256+128`) with `jobs[1].y == nullptr` (`max_abs_err = 0.000000e+00`).

### Compiled Fatbin ELF Audit (`gfx1201`, 497 Kernels Across 14 Code Objects)

| Kernel / Metric | `v5.3.0-rc1` (ROCm `10.1.0-full`) | `v5.3.0-rc1` (ROCm `10.2.0-full`) |
|---|---|---|
| Total compiled `gfx1201` kernels | `497` (`14` code objects) | `497` (`14` code objects) |
| `flash_kernel<2u, true, true>` | `vgpr=235`, `sgpr=45`, `lds=21392`, `maxwg=128`, `scratch=0` (`occ=6`) | `vgpr=240`, `sgpr=45`, `lds=21392`, `maxwg=128`, `scratch=0` (`occ=6`) |
| `flash_kernel<2u, false, false>` | `vgpr=235`, `sgpr=40`, `lds=11344`, `maxwg=128`, `scratch=0` | `vgpr=240`, `sgpr=40`, `lds=11344`, `maxwg=128`, `scratch=0` |
| `idx_quant_kernel` | `vgpr=11`, `sgpr=18`, `lds=5280`, `maxwg=1024`, `scratch=0` | `vgpr=11`, `sgpr=18`, `lds=5280`, `maxwg=1024`, `scratch=0` |
| `tp_ar_fp8_staged_kernel` | `vgpr=21`, `sgpr=60`, `lds=4`, `maxwg=1024`, `scratch=0` | `vgpr=21`, `sgpr=60`, `lds=4`, `maxwg=1024`, `scratch=0` |
| `batch_gemm_multi_kernel` (all 8) | `vgpr=64..87`, `lds=18048..18944`, `scratch=0` | `vgpr=64..87`, `lds=18048..18944`, `scratch=0` |
| `batch_gemm_fp8_multi_kernel` (all 32) | `vgpr=68..107`, `lds=2248..38156`, `scratch=0` | `vgpr=68..107`, `lds=2248..38156`, `scratch=0` |
| Kernels with static `scratch > 0` | `1` (`hcn_v41_collapse_sq_kernel<16u, 8u>`, `vgpr=256`, `lds=576`, `scratch=292`) — **0 new spills** | `1` (`hcn_v41_collapse_sq_kernel<16u, 8u>`, `vgpr=256`, `lds=576`, `scratch=292`) — **0 new spills** |

---

## `v5.3.0` 3-Arm Paired Benchmark & ROCm 10.1 vs 10.2 Evaluation

All 3 arms ran back-to-back in an isolated single-tenant GPU window with identical `AFF_*` runtime flags, deterministic prompts, isolated scratch `pcache` / `placement.affprof`, and verified idle thermal cooldown (`NO_FFMPEG`). Full details are in [`docs/v5.3.0-hip-kernel-optimization-benchmark.md`](docs/v5.3.0-hip-kernel-optimization-benchmark.md).

### Startup Health & KFD Release Latency

| Metric | Arm 1 (`v5.2.0`, ROCm 10.1) | Arm 2 (`v5.3.0-rc1`, ROCm 10.1 — Winner) | Arm 3 (`v5.3.0-rc1`, ROCm 10.2) |
|---|---:|---:|---:|
| `/health` `resident_experts` | `3,517` | `3,517` | `3,517` |
| `/health` `free_vram_mib` | `[353, 356]` | `[353, 355]` | `[353, 356]` |
| Server load ready time | `84 s` | `84 s` | `84 s` |
| Clean shutdown KFD release | `2.750 s` | `2.800 s` | `2.812 s` |

### `bench_pp_tg.py` Paired Cold/Warm Prefill & Decode (`NGEN=192`)

| Depth (`prompt_tok`) | Metric | Arm 1 (`v5.2.0` 10.1) | Arm 2 (`v5.3.0-rc1` 10.1) | Arm 3 (`v5.3.0-rc1` 10.2) | Arm 2 vs Arm 1 |
|---|---|---:|---:|---:|---:|
| `l4k.txt` (`4,031`) | `pp1` (`ttft1`) | `565.6` (`7.1 s`) | **`586.0`** (`6.9 s`) | `592.7` (`6.8 s`) | `+3.6%` |
| `l4k.txt` (`4,031`) | `tg1` | `20.20` | **`25.70`** | `26.47` | **`+27.2%`** |
| `l4k.txt` (`4,031`) | `pp2` (`ttft2`) | `528.5` (`7.6 s`) | **`562.5`** (`7.2 s`) | `560.7` (`7.2 s`) | `+6.4%` |
| `l4k.txt` (`4,031`) | `tg2` | `33.79` | **`42.20`** | `46.03` | **`+24.9%`** |
| `l16k.txt` (`16,104`) | `pp1` (`ttft1`) | `1140.8` (`14.1 s`) | **`1210.7`** (`13.3 s`) | `1216.2` (`13.2 s`) | `+6.1%` |
| `l16k.txt` (`16,104`) | `tg1` | `29.13` | **`41.88`** | `39.30` | **`+43.8%`** |
| `l16k.txt` (`16,104`) | `pp2` (`ttft2`) | `1038.5` (`15.5 s`) | **`1104.4`** (`14.6 s`) | `1108.3` (`14.5 s`) | `+6.3%` |
| `l16k.txt` (`16,104`) | `tg2` | `31.47` | **`42.23`** | `42.11` | **`+34.2%`** |
| `l64k.txt` (`63,863`) | `pp1` (`ttft1`) | `1279.0` (`49.9 s`) | **`1348.7`** (`47.4 s`) | `1353.2` (`47.2 s`) | `+5.4%` |
| `l64k.txt` (`63,863`) | `tg1` | `29.38` | **`41.47`** | `39.95` | **`+41.1%`** |
| `l64k.txt` (`63,863`) | `pp2` (`ttft2`) | `1221.6` (`52.3 s`) | **`1291.8`** (`49.4 s`) | `1300.7` (`49.1 s`) | `+5.7%` |
| `l64k.txt` (`63,863`) | `tg2` | `30.33` | **`41.68`** | `39.17` | **`+37.4%`** |

- **Mean slot `blocks/s` (all 19 paired slot completions/arm)**: Arm 1 `5.5864 blocks/s` -> Arm 2 **`7.6562 blocks/s` (`+37.05%`)**, Arm 3 `7.3788 blocks/s`.

### 4-Prompt `temp=0.0` DSpark Streaming (`max_tokens=256`)

| Prompt (`ptok`) | Arm 1 `tg_stream` (`blk/s`) | Arm 2 `tg_stream` (`blk/s`) | Arm 3 `tg_stream` (`blk/s`) | Arm 2 DSpark (`acc%`, `tok/blk`) |
|---|---:|---:|---:|---|
| `math_proof` (`31`) | `15.26` (`4.83`) | **`19.92`** (`6.65`, **`+30.5%`**) | `18.11` (`5.87`) | `67.7%`, `2.98` |
| `cpp_bug` (`176`) | `15.91` (`4.69`) | **`21.86`** (`6.69`, **`+37.4%`**) | `19.34` (`5.85`) | `81.6%`, `3.24` |
| `json_schema` (`101`) | `18.07` (`4.09`) | **`24.95`** (`5.16`, **`+38.1%`**) | `21.63` (`4.81`) | `98.5%`, `4.74` |
| `logic_puzzle` (`119`) | `13.98` (`5.04`) | **`18.59`** (`6.71`, **`+33.0%`**) | `17.60` (`6.35`) | `66.8%`, `2.75` |
| **Mean** | `15.81` (`4.66`) | **`21.33`** (`6.30`, **`+34.9%`**) | `19.17` (`5.72`) | — |

### Multi-Turn `64K -> 100K` Cached TTFT & Decode

| Turn | Arm 1 `ttft` / `tg` | Arm 2 `ttft` / `tg` | Arm 3 `ttft` / `tg` | Arm 2 `tg` vs Arm 1 |
|---|---:|---:|---:|---:|
| `turn1_64k_cold` (`80,138`, `0` cached) | `82.733 s` / `13.68` | **`81.051 s` / `16.72`** | `82.172 s` / `16.84` | **`+22.2%`** |
| `turn2_64k_cached` (`80,412`, `80,393` cached) | `1.259 s` / `16.03` | **`1.056 s` / `23.89`** | `1.020 s` / `24.84` | **`+49.0%`** |
| `turn3_100k_extend` (`123,138`, `80,667` cached) | `40.519 s` / `19.77` | **`39.282 s` / `26.30`** | `38.298 s` / `25.11` | **`+33.0%`** |
| `turn4_100k_cached` (`123,409`, `123,393` cached) | `1.182 s` / `14.12` | **`1.002 s` / `20.08`** | `0.966 s` / `19.63` | **`+42.2%`** |

### ROCm `10.1.0-full` vs `10.2.0-full` Composite Gate

| Metric | Arm 2 (`ROCm 10.1.0-full`) | Arm 3 (`ROCm 10.2.0-full`) | Delta (Arm 3 vs Arm 2) |
|---|---:|---:|---:|
| `mean_pp` (tok/s) | `1017.35` | `1021.97` | `+0.45%` |
| `mean_tg` (tok/s) | **`39.19`** | `38.84` | `-0.91%` |
| `mean_bps` (blocks/s) | **`7.656`** | `7.379` | `-3.62%` |
| `mean_step_c` (4-prompt `tg_stream`) | **`21.33`** | `19.17` | `-10.13%` |
| **Composite Mean Relative Diff** | **Winner (`+3.55%` advantage)** | — | **`-3.551%`** |

### CLI `AFF_TRACE=dense,events,deep` Profile Diff (`l16k.txt`, `-n 128`)

| Trace Component | `v5.2.0` (`4ada6fa`) Device Est | `v5.3.0-rc1` (`e1f6be3`, ROCm 10.1) Device Est | Delta |
|---|---:|---:|---:|
| `dense-gpu` total | `9.943 s` | **`6.560 s`** | **`-34.0%`** |
| `routed-expert` card 0 | `28.257 s` | **`21.768 s`** | **`-23.0%`** |
| `routed-expert` card 1 | `28.461 s` | **`21.713 s`** | **`-23.7%`** |
| `x_gu + x_down` (card 0) | `8,619.40 ms` | **`1,278.99 ms`** | **`-85.2%`** |
| `b_reduce` | `1,190.81 ms` | **`637.08 ms`** | **`-46.5%`** |
| `reduce` | `869.61 ms` | **`592.86 ms`** | **`-31.8%`** |
| `b_attend` | `244.73 ms` | **`177.12 ms`** | **`-27.6%`** |
| `b_gate` | `840.58 ms` | **`741.35 ms`** | **`-11.8%`** |

---

## Measured Performance Progression (Waves 0–6)

Full historical methodology and charts are in [`docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md`](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md) and [`docs/benchmarks/2026-11-18-v5.2-deepseek-optimization.md`](docs/benchmarks/2026-11-18-v5.2-deepseek-optimization.md).

### Progression Summary Table

| Wave / Version | Git SHA | Decode Median / Warm `tg` | Batch-8 / 4-Prompt DSpark | Long Prefill (@64K) | Resident Experts | Repetition Rate (`temp=0.6`) | Test Suite |
|---|:---:|---:|---:|---:|---:|:---:|:---:|
| **v4.2.0 Baseline** | `14e91c4` | 16.33 tok/s | 10.9 tok/s (b8) | ~1,200 tok/s | 3,157 | frequent traps | 6/6 PASS |
| **v4.3.0 / v4.3.2** | `e2a4478` | 19.27 tok/s | 21.4 tok/s (b8) | 1,180 tok/s | 3,279 (+88 GPU1) | loop traps | 7/7 PASS |
| **Wave 0 (v4.3.3)** | `5275c64` | 18.31 tok/s | 16.9 tok/s (b8) | 1,206 tok/s | 3,367 | loop traps | 7/7 PASS |
| **Wave 1 (v4.4.0)** | `718b42e` | 20.41 tok/s | 20.9 tok/s (b8) | 1,208 tok/s | 3,367 | **0/20 (0%)** | 7/7 PASS |
| **Wave 2 (v4.5.0)** | `9ec7c47` | **23.73 tok/s** (24.37 peak) | **22.4 tok/s** (b8) | 1,215 tok/s | 3,367 | 0/20 (0%) | 8/8 PASS |
| **Wave 3 (v5.0.0)** | `264e8d5` | **23.73 tok/s** | **22.4 tok/s** (b8) | **21,448 tok/s (14.4×)** | **3,416 (+49)** | **0/20 (0%)** | **9/9 PASS** |
| **Wave 4A (v5.1.0)** | `839665f` | **20.69 tok/s** (22.63 peak) | — | **21,448 tok/s (14.4×)** | **3,416** | **0/20 (0%)** | **6/6 PASS** |
| **Wave 4B (v5.1.1 Hotfix)** | `63cd104` | **20.69 tok/s** (22.63 peak) | — | **21,448 tok/s (14.4×)** | **3,416** | **0/20 (0%)** | **6/6 PASS** |
| **Wave 4C (v5.1.2 ThinkingBudget)** | `c52e95e` | **20.69 tok/s** (22.63 peak) | — | **21,448 tok/s (14.4×)** | **3,416** | **0/20 (0%)** | **7/7 PASS** |
| **Wave 5 (v5.2.0 DeepSeek Opt)** | `44e81ee` | **41.00–41.72 tok/s** warm (`tg2`) | 15.81 tok/s (4p DSpark) | **1,325.8 tok/s** cold (`+5.2%`) | **3,517 (+101)** | **0/20 (0%)** | **9/9 CPU + 4/4 GPU PASS** |
| **Wave 6 (v5.3.0 HIP Opt)** | `e1f6be3` | **41.68–42.23 tok/s** (`+43.8%` `tg1`) | **21.33 tok/s** (4p DSpark, `+34.9%`) | **1,348.7 tok/s** cold (`+5.4%`) | **3,517** | **0/20 (0%)** | **13/13 CPU + 5/5 GPU + 2/2 FB PASS** |

### Wave 3 / Wave 4 Long-Context Sparse Indexer (`v5.0.0`)

Measured across deep sequence lengths with the RDNA 4 sparse indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`):

| Context Depth | Prompt Tokens | Baseline TTFT (`idx0`) | v5.0.0 TTFT (`idx1`) | Baseline Prefill | v5.0.0 Prefill | Speedup Factor |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **4K** (Pass 1) | 3,810 | 7.7 s | 3.2 s | 496.7 tok/s | 1,186.1 tok/s | **2.39×** |
| **16K** (Pass 1) | 15,163 | 12.9 s | 3.6 s | 1,173.0 tok/s | 4,220.3 tok/s | **3.60×** |
| **64K** (Pass 2) | 60,503 | 40.6 s | 2.8 s | 1,489.8 tok/s | 21,448.9 tok/s | **14.40×** |
| **128K** (Pass 2) | 120,976 | 56.6 s | 4.3 s | 2,137.1 tok/s | 28,092.6 tok/s | **13.15×** |

Cold uncached prefill (`cached_tokens == 0`) sustains **963.66 tok/s @32K** and **990.98 tok/s @64K**.

### Needle-in-a-Haystack & Numerical Parity (`v5.0.0`)

- **Needle-in-a-Haystack (`needle_ab`)**: **4/4 (100%)** retrieval accuracy across 4K, 16K, 64K, and 128K.
- **Reference Attention Parity (`refattn.py`)**: **99.98%** cosine parity against uncompressed dense attention.
- **KL Divergence**: **0.000000 nats** divergence (PPL 2.6057) across 8,184 tokens.
- **Quantization Eval (`quant_eval`)**: **92.50%** accuracy (111/120 items).

---

## Wave 4 Bake-Off: `affinity` vs `StillDeadcode/radiance v2.1.0`

Offline evaluation of upstream `StillDeadcode/radiance v2.1.0` (`2a2b756`) on ROCm 10.1 `gfx1201`:

| Feature / Architecture Component | `deepseek-affinity` | `StillDeadcode/radiance v2.1.0` |
|---|:---:|:---:|
| **Target Hardware Support** | AMD RDNA 4 `gfx1201` (Dual R9700) | AMD RDNA 4 `gfx1201` (Dual R9700) |
| **DeepSeek-V4.1 Architecture** | **Full Native Support** | **Blocked** (Plugin not in tree) |
| **Multi-Head Latent Attention (MLA)** | **Native Decoupled KV Kernel** | Missing in `libr4d 0.5.0` |
| **Sparse Indexer (Top-512)** | **Native `gfx1201` Radix Sort** | None |
| **Engram 188.8 GiB Demand Paging** | **Native Zero-Copy UVA** | None |
| **FP8 Scale Format (`F8_E8M0`)** | **Full Native Decoding** | **Unsupported** (`RAD_DT_INVALID`) |
| **Persistent Multi-Slot Prefix Cache** | 64 GiB NVMe multi-slot persistent | In-memory only |
| **Repetition Defense** | `AFF_CACTUS_ESCAPE=0.25` (`0/20` locked) | None |

---

## Rework and Patches (`v1.0.0` → `v5.3.0`)

> **Every commit below is traceable.** Click any short SHA to inspect its full diff on GitHub.

All historical branches are unified on `main`.

### Engine & Optimization History (`v1.0.0` → `v5.3.0`)

| Diff | Change | Measured Effect |
|---|---|---|
| [`91e78ba`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/91e78ba) | **DeepSeek ViT on the CPU**, wired into prefill and chat endpoint (2D-RoPE half rotation, channel-major aligner, aspect-preserving pad) | Embedding cosine vs reference `0.31 -> 0.85`; image grounding works |
| [`cbb3f07`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cbb3f07) | ViT **weight-stationary GEMM + AVX-512 attention** | `546×546`: `13.2 s -> 3.25 s`; `1024×1024`: `38 s` |
| [`e1a2025`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e1a2025) | ViT **query-blocked attention**; opt-in engram page-cache warm (`AFF_ENGRAM_WARM`) | `74×74`: `38.4 s -> 23.3 s`, numerically identical |
| [`cac0510`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cac0510) | Size compressed KV cache **per kv source, not per layer** | KV `1016 -> 100 MiB/card`; resident experts `2206 -> 2357` |
| [`997489c`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/997489c) | **Tile the indexer byte mask** over the prefill chunk | Prefill arena `2068 -> 1426 MiB/card`; resident experts `2357 -> 2462` |
| [`78244dc`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/78244dc) | **Persist learned placement heat plane** across restarts (`--placement-save`) | Residency survives restarts |
| [`a4c9f85`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a4c9f85) | Recover **marker-less ("bare") DSML tool calls** and stop at block close | Poisoned 117k replay dropped `4360 -> 195` tokens |
| [`bc42be2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bc42be2) | DSpark **confidence head** + opt-in gating (`AFF_SPEC_CONF_T`) | Wash on 2 cards — kept as calibration knob |
| [`27ec8fa`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/27ec8fa) | Opt-in DSpark backoff after zero-accept streak (`AFF_SPEC_BACKOFF=N`) | Kept as tunable knob |
| [`1ae7c09`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1ae7c09) | **SSE keep-alive** while tool call is withheld | Long tool calls avoid client idle timeouts |
| [`2a4478f`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2a4478f) | **On-GPU imatrix calibration**, **bounded replay** (`AFF_BOUNDED_REPLAY`), **soft resident bias** (`AFF_RESIDENT_BIAS`), pinned sampler staging, async peer-copy overlap, `AFF_BLOCK_AT` hooks | Host-device copies overlap; in-engine imatrix calibration |
| [`2779348`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2779348) · [`patches/AFF_NOBUDGET_MAX_TOKENS.patch`](patches/AFF_NOBUDGET_MAX_TOKENS.patch) | Bound default `max_tokens` when omitted in request | Prevents runaway 55k-token slots without cap |
| [`154eb2b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/154eb2b) | **`v1.1.0` Radiance & KTransformers ports**: `AFF_FUSE_HC_TAP`, `AFF_DRAFT_HEAD_INT4`, `AFF_O_FP8_QDQ`, Cactus capped rejection, `namespace::name` tool parsing | Decode `17.08 -> 21.96 tok/s` (`+28.6%`) @163k–171k ctx; `-42.8 ms/block` |
| [`23ec996`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/23ec996), [`6c77094`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/6c77094), [`308638e`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/308638e) | Coalesced TR `MT=1` sub-tile loads, `AFF_MOE_RECON_TILES=32`, opt-in `AFF_HADAMARD_ROT` & `AFF_PREFETCH_RING` | Removes ~480 empty GPU launches/verify step |
| [`7fd7a71`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/7fd7a71), [`9e5db70`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/9e5db70), [`1f0d668`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1f0d668) | **`v4.2.0` VRAM & placement guards**: `hip_free_vram_checked`, `--placement-save-secs`, hysteresis `1.6 / 3.0`, HTTP 400 guard | 479k free VRAM `66 -> 920 MiB`; H2D churn `113 -> 52 GiB` (`-54%`) |
| [`026a9b0`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/026a9b0), [`3a06ea1`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/3a06ea1) | **`v4.3.0`**: fused `AFF_V41_HC_FUSED=1`, LDS FP8 scale prefetch, zero-copy `b_pre` swap, `AFF_PREFILL_FREEZE_PLC=1`, `AFF_BONUS_SLAB=1` | Decode `16.33 -> 19.27 tok/s`; batch-8 `10.9 -> 21.4 tok/s`; `3,367` experts |
| [`21ba484`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/21ba484), [`fa1588b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/fa1588b), [`5275c64`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/5275c64) | **Wave 0 (`v4.3.3`)**: premature EOS `<think>` fix, `GET /health`, `--raw-completions`, `cached_tokens`, branch unification | Single linear `main`; 7/7 tests PASS |
| [`17682d7`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/17682d7), [`3551e14`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/3551e14), [`34610b4`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/34610b4), [`718b42e`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/718b42e) | **Wave 1 (`v4.4.0`) + ROCm 10.1**: multi-slot prefix cache, `heat -> static` fallback, `.scale` fix, `AFF_CACTUS_ESCAPE=0.25`, DSML typed tags | Decode `18.31 -> 20.41 tok/s` (+10.4%); repetition lock-in **`0/20 (0%)`** |
| [`f849997`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/f849997), [`bf2ae04`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bf2ae04), [`cbf07da`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cbf07da), [`14f149d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/14f149d), [`9ec7c47`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/9ec7c47) | **Wave 2 (`v4.5.0`)**: `AFF_HC_FUSED_V2=1`, `AFF_HC_CHAINED=1`, `AFF_HC_CHAINED_MIN_ROWS=1`, KL harness, gated `AFF_EMBED_HOST` & `AFF_MOE_SUMMED_DOWN` | Decode **23.73 tok/s** (24.37 peak); batch-8 **22.4 tok/s**; KL `0.000000` |
| [`570e787`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/570e787), [`a02f669`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a02f669), [`264e8d5`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/264e8d5) | **Wave 3 (`v5.0.0`)**: RDNA4 `gfx1201` sparse attention indexer (`AFF_SPARSE_INDEXER=1`), radix top-512, 36 MiB scratch budget | **14.40× @64K** & **13.15× @128K**; **3,416 experts (+49)**; 9/9 tests PASS |
| [`a041faf`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a041faf) | **Wave 4 bake-off**: offline ROCm 10.1 build & evaluation of `StillDeadcode/radiance v2.1.0` | Confirmed open-source Radiance lacks V4.1 plugin, MLA/Engram & `F8_E8M0` |
| [`68011d2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/68011d2), [`af2d390`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/af2d390), [`5dbad17`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/5dbad17), [`bd3bf03`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bd3bf03), [`39e48c6`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/39e48c6), [`1276495`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1276495), [`70b3aba`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/70b3aba), [`839665f`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/839665f) | **Wave 4A (`v5.1.0`)**: DSML nested XML & multi-invoke split (#1020), `AFF_PCACHE_DELTA=1` & `ignore_eos` (#989), `gridDim.y > 65535` guard (#1006), `AFF_DRAFT_SKIP_MASKED=1` (#958), stream join invariants (#983/#984), CLI tools | DSpark acceptance **73.33%** (+5.81 pp); `1.42×` cached TTFT; 6/6 CPU tests PASS |
| [`63cd104`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/63cd104) | **`v5.1.1` hotfix**: `TP=2` multi-region delta prefix-cache restore offset fix + strict UTF-8 streaming/JSON escaping | Fixes multi-region offset mismatch & early EOS UTF-8 chunks; 6/6 CPU tests PASS |
| [`c52e95e`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/c52e95e) | **`v5.1.2` ThinkingBudget**: `--thinking-reply-reserve 4096`, `--thinking-loop-repeats 3`, Executive-Freeze Churn Breaker, cache-preserving `reasoning_effort: "none"` | Prevents runaway `<think>` exhaustion with 100% prefix-cache preservation; 7/7 CPU tests PASS |
| [`45ea4eb`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/45ea4eb), [`f424cfd`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/f424cfd), [`54c5d78`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/54c5d78), [`d24e31d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/d24e31d), [`60676eb`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/60676eb) | **`v5.2.0` host/server/tokenizer/pcache**: `fd_alive` disconnect polling, FIFO queue, 15 s SSE keepalives, lock-free `/health`, skip prefill logits, O(1) `decode_one` + BPE/LCP cache | Disconnect drops in `1.12 s` (`0` wasted tokens); cached TTFT `-47.0% @64K` / `-55.9% @100K`; 9/9 CPU `ctest` PASS |
| [`79e8ac7`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/79e8ac7), [`edba510`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/edba510), [`b13cbf2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/b13cbf2), [`34deb71`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/34deb71), [`3f24d07`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/3f24d07), [`e626921`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e626921), [`44e81ee`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/44e81ee) | **`v5.2.0` HIP/VRAM/decode**: `-616 MiB/card` dead buffer & Engram tiling reclaim, `AFF_FUSED_ATTN_OUT_PREP`, `AFF_FP8_NARROW128`, `AFF_REUSE_NORM_Q8`, prefill-before-decode sync | **`+101` resident experts (`3,517`)**; warm `tg2` **41.00–41.72 tok/s** (`+30.2%..+56.1%`); 4/4 GPU PASS |
| [`dfb34cb`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/dfb34cb), [`c20ed32`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/c20ed32), [`e1f6be3`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e1f6be3) | **`v5.3.0` RDNA 4 Wave32 HIP optimization**: Wave A primitives (`lds_barrier`, `dpp_xor_lane`, `AFF_WAVES`), Wave B compute/comms (`AFF_FP8_PREFILL_4X2`, `AFF_FP8_SPLIT_INLINE`, `AFF_TP_AR_STAGED`, `flash_kernel<2>` `occ=6`, `idx_quant_kernel` `scratch=0`), multi-GEMM null-band & WGP L0 acquire fence fix | **`+24.9%..+43.8%` `tg`** (`41.47–42.23 tok/s` @16K–64K), **`+37.05%` slot `blocks/s`**, **`-34.0%` `dense-gpu`**, **`-85.2%` `x_gu+x_down`**, ROCm `10.1` `+3.55%` composite vs `10.2`, `0` new ELF spills, **13/13 CPU + 5/5 GPU + 2/2 fallback PASS** |

### V4.1 / Multimodal Port (branch `v41`, merged in history)

| Diff | Change | Improvement |
|---|---|---|
| [`e6070f5`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e6070f5) | V4.1 reasoning-effort prefix + `陪同下` token, byte-identical to reference `encoding.py` | Correct V4.1 chat encoding |
| [`e03a05b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e03a05b), [`774aa6a`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/774aa6a), [`4165fce`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4165fce), [`4bc938b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4bc938b) | Engram gate/stride, engram BF16 weights, V4.1 no per-head RMS on `q` | Fixed silent numerical mismatches |
| [`d91ae9d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/d91ae9d) | Engram n-gram layers 1 and 14 (`91.6 GiB` tables), indexer, compressor | V4.1 architecture support |
| [`2dbb093`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2dbb093) | Stop retrying KV growth the card cannot hold | Removes 64K decode cliff |

See [`CHANGELOG.md`](CHANGELOG.md) and [`THIRD_PARTY.md`](THIRD_PARTY.md) for full release notes and upstream attributions.

---

## Production Safety Invariants & Environment Variable Reference

### Hard Safety Invariants & Rejected Toggles

| Invariant / Flag | Setting | Reason |
|---|---|---|
| Target LM Head | **BF16** (`AFF_LMHEAD_INT4=0`) | Never enable `AFF_LMHEAD_INT4=1` on target model (degrades perplexity and draft acceptance). Draft head uses `AFF_DRAFT_HEAD_INT4=1`. |
| Micro-Batch Sizing | `AFF_EQUAL_UBATCH=0` | Rejected: adds kernel overhead without reducing chunk count. |
| Device Prefetch Ring | `AFF_PREFETCH_RING=0` | Rejected: consumes 8–12 resident expert slots without PCIe 3.0 throughput gain. |
| Attention Lookahead | `AFF_ATTN_LOOKAHEAD=0` | Rejected as default: regressed median decode by `-11.4%` (`20.69 -> 18.33 tok/s`). |
| Host Embedding Offload | `AFF_EMBED_HOST=0` | Rejected: `token_embd` is already host-mmap'd at `0 MiB` VRAM (`-16%` decode overhead when set). |
| Summed MoE Down | `AFF_MOE_SUMMED_DOWN=0` | Rejected: extra PCIe reduction overhead without speedup. |
| MoE Execution Tier | **100% GPU Tensor Cores** | Never offload routed MoE compute to CPU. |
| KV Commit Lifespan | **Monotonic Commit** | Never unmap KV pages live (`kv_shrink_to`) to avoid page-table deadlock during streaming. |

### Production Environment Knobs

| Variable | Value in Production | Purpose |
|---|---|---|
| `AFF_ENGRAM_WARM` | `1` | Pre-warm mmapped Engram page tables |
| `AFF_SPEC_BACKOFF` | `2` | Adaptive DSpark draft backoff after zero-accept streak |
| `AFF_ADAPTIVE_WIDTH` | `1` | Adaptive speculative verify block width |
| `AFF_BOUNDED_REPLAY` | `1` | Bounded context replay recovery |
| `AFF_FUSE_HC_TAP` | `1` | Fuse `hc_post` reduction with DSpark tap capture |
| `AFF_DRAFT_HEAD_INT4` | `1` | Group-128 INT4 DSpark draft LM head |
| `AFF_CACTUS_DELTA` | `0.35` | Cactus speculative divergence delta threshold |
| `AFF_CACTUS_CAP_R` | `0.5` | Cactus speculative rejection cap |
| `AFF_CACTUS_ESCAPE` | `0.25` | Convex escape mixture (`0/20` repetition lock-in @ `temp=0.6`) |
| `AFF_RESIDENT_BIAS` | `0.008` | Soft routing bias toward resident experts |
| `AFF_MOE_RECON_TILES` | `32` | Prefill MoE tile L2 reconciliation cap |
| `AFF_PREFETCH_RING` | `0` | Disabled expert prefetch ring |
| `AFF_NOBUDGET_MAX_TOKENS` | `24576` | Default token cap when request omits `max_tokens` |
| `AFF_THINK_REPLY_RESERVE` | `4096` | Reserved reply budget before `<think>` closure |
| `AFF_THINK_LOOP_REPEATS` | `3` | Loop detector threshold over `loop_span=64` |
| `AFF_THINK_FREEZE_BREAKER` | `1` | Executive-Freeze Churn Breaker inside `<think>` |
| `AFF_HC_FUSED_V2` | `1` | Fused `hc_read` + pipelined tiled up-mix |
| `AFF_HC_CHAINED` | `1` | Chained hybrid compute across layers |
| `AFF_HC_CHAINED_MIN_ROWS` | `1` | Enable chained HC down to decode `nb=1` |
| `AFF_SPARSE_INDEXER` | `1` | Native `gfx1201` radix top-512 sparse attention indexer |
| `AFF_INDEXER_SCORE_BUDGET_MB` | `36` | Bounded score scratch allocation per card (`<= 48 MiB`) |
| `AFF_DRAFT_SKIP_MASKED` | `1` | Skip causally masked tiles in DSpark verification |
| `AFF_ATTN_LOOKAHEAD` | `0` | Disabled decode KV lookahead |
| `AFF_PCACHE_DELTA` | `1` | State-only persistent prefix cache delta checkpoints |

### `v5.3.0` HIP Kernel Control & Diagnostic Toggles

| Variable | Default Behavior | Description |
|---|---|---|
| `AFF_FP8_PREFILL_4X2` | Enabled (`"0"` disables) | Enables `4×2` wave tiling + `kGroupM=8` MALL swizzle in prefill FP8 GEMMs |
| `AFF_FP8_SPLIT_INLINE` | Enabled (`"0"` disables) | Enables inline atomic split-K accumulation (`kSplitCtrFloats=1024`) in narrow decode FP8 GEMMs |
| `AFF_TP_AR_STAGED` | Enabled (`"0"` disables) | Enables staged `S=8` TP=2 FP8 BAR push/fold overlap for payloads `>= 4 MiB` |
| `AFF_TP_AR_STAGES` | `8` (clamped `1..8`) | Stage count for `tp_ar_fp8_staged_kernel` |
| `AFF_FLASH_MIN_TILES` | `0` (`1` if `n_tok<=16`, else `4`) | Minimum K-tiles per split slice in `mqa_attend_flash_hip` |
| `AFF_LEAD_GATE_IDLE` | Disabled (`"0"` / unset) | Set `"1"` to force `StaticPlacement::lead_gate` barriers even with no active host placement moves |

---

## Build, Test, Deploy & Rollback Guide

### Building the Docker Image

Default production toolchain is **`docker.io/rocm/dev-ubuntu-24.04:10.1.0-full`**:

```sh
SHORT_SHA=$(git rev-parse --short HEAD)
docker build -f Dockerfile \
  --build-arg AFF_VERSION=v5.3.0 \
  --build-arg AFF_GIT_SHA="$SHORT_SHA" \
  -t deepseek-affinity:v5.3.0 .
```

Optional ROCm `10.2.0-full` comparison build:

```sh
docker build -f Dockerfile \
  --build-arg BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:10.2.0-full \
  --build-arg AFF_VERSION=v5.3.0 \
  --build-arg AFF_GIT_SHA="$SHORT_SHA" \
  -t deepseek-affinity:v5.3.0-rocm10.2 .
```

### Verification Commands

CPU tests + ELF link audit (no GPU required):

```sh
docker run --rm --security-opt label=disable --entrypoint bash deepseek-affinity:v5.3.0 -c '
  ldd /usr/local/bin/affinity | grep -i "not found" && exit 1 || true
  ctest --test-dir /opt/affinity-src/build -E "^gpu_ports$" --output-on-failure
'
```

GPU kernel test suites (`5/5` binaries + `2/2` fallback-toggle runs, requires exclusive GPU access):

```sh
for t in test_gpu_ports test_moe_recon_tiles test_v41_indexer test_wave2_kernels test_prefetch_ring; do
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --security-opt label=disable --entrypoint "" -e HSA_ENABLE_COREDUMP=0 \
    deepseek-affinity:v5.3.0 "/opt/affinity-src/build/$t"
done

# Fallback-toggle regressions (AFF_FP8_PREFILL_4X2=0 AFF_FP8_SPLIT_INLINE=0, AFF_TP_AR_STAGED=0 AFF_FLASH_MIN_TILES=4)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --security-opt label=disable --entrypoint "" -e HSA_ENABLE_COREDUMP=0 \
  -e AFF_FP8_PREFILL_4X2=0 -e AFF_FP8_SPLIT_INLINE=0 \
  deepseek-affinity:v5.3.0 /opt/affinity-src/build/test_wave2_kernels

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --security-opt label=disable --entrypoint "" -e HSA_ENABLE_COREDUMP=0 \
  -e AFF_TP_AR_STAGED=0 -e AFF_FLASH_MIN_TILES=4 \
  deepseek-affinity:v5.3.0 /opt/affinity-src/build/test_gpu_ports
```

### Serving DeepSeek-V4.1-Flash

Run via [`deploy/run.sh`](deploy/run.sh) or systemd unit [`deploy/deepseek-v41-affinity.service`](deploy/deepseek-v41-affinity.service):

```sh
docker run --name deepseek-affinity \
  --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --cpuset-cpus=0-23,48-71 \
  -e AFF_ENGRAM_WARM=1 -e AFF_SPEC_BACKOFF=2 -e AFF_ADAPTIVE_WIDTH=1 \
  -e AFF_BOUNDED_REPLAY=1 -e AFF_FUSE_HC_TAP=1 -e AFF_DRAFT_HEAD_INT4=1 \
  -e AFF_CACTUS_DELTA=0.35 -e AFF_CACTUS_CAP_R=0.5 -e AFF_CACTUS_ESCAPE=0.25 \
  -e AFF_RESIDENT_BIAS=0.008 -e AFF_MOE_RECON_TILES=32 -e AFF_PREFETCH_RING=0 \
  -e AFF_NOBUDGET_MAX_TOKENS=24576 -e AFF_THINK_REPLY_RESERVE=4096 \
  -e AFF_THINK_LOOP_REPEATS=3 -e AFF_THINK_FREEZE_BREAKER=1 \
  -e AFF_HC_FUSED_V2=1 -e AFF_HC_CHAINED=1 \
  -e AFF_HC_CHAINED_MIN_ROWS=1 -e AFF_SPARSE_INDEXER=1 -e AFF_INDEXER_SCORE_BUDGET_MB=36 \
  -e AFF_DRAFT_SKIP_MASKED=1 -e AFF_ATTN_LOOKAHEAD=0 -e AFF_PCACHE_DELTA=1 \
  -p 8000:8000 \
  -v /models:/models \
  deepseek-affinity:v5.3.0 \
    -m /models/DeepSeek-V4.1-Flash-1017-MoonMath-aff/model-moonmath-1017.aff \
    --tokenizer /models/LLM-Models/DeepSeek-V4.1-Flash-1017-MoonMath-FP8/tokenizer.json \
    --host 0.0.0.0 --port 8000 \
    --raw-completions \
    --gpu-headroom-mib 384 \
    --dspark on \
    --dspark-model /models/DeepSeek-V4.1-Flash-1017-MoonMath-aff/model-moonmath-1017.dspark.aff \
    --engram /models/DeepSeek-V4.1-Flash-1017-MoonMath-aff/model-moonmath-1017.engram \
    --engram-tables /models/LLM-Models/DeepSeek-V4.1-Flash-1017-MoonMath-FP8 \
    --prefix-cache-dir /models/DeepSeek-V4.1-Flash-1017-MoonMath-aff/pcache \
    --prefix-cache-gib 64 \
    --persistent-prefix-cache \
    --kv-size 327680 \
    --kv-commit 262144 \
    --host-pool-mib 200000 --host-pool-reserve-mib 1024 \
    --placement-strategy demand \
    --placement-hysteresis 1.6 --placement-min-gain 3.0 \
    --placement-save /models/DeepSeek-V4.1-Flash-1017-MoonMath-aff/placement.affprof \
    --prefill-chunk-size 2816 \
    --default-effort 60 \
    --repeat-penalty 1.1 --repeat-last-n 96 --dry-delta 40 \
    --max-tokens-floor 0
```

Health check:

```sh
curl -sS http://127.0.0.1:8000/health
# {"status":"ok","version":"v5.3.0","git_sha":"...","resident_experts":3517,"free_vram_mib":[353,355]}
```

### Rollback to `v5.2.0`

```sh
systemctl --user stop deepseek-v41-affinity.service && \
cp ~/.config/systemd/user/deepseek-v41-affinity.service.bak-v5.2.0 ~/.config/systemd/user/deepseek-v41-affinity.service && \
systemctl --user daemon-reload && \
systemctl --user start deepseek-v41-affinity.service
```

---

## Credits

- DeepSeek for the base model architecture and reference code.
- Yoshi Exeler / StillDeadcode for affinity and Radiance algorithmic foundations.
- Lucebox AI ([Luce-Org/lucebox](https://github.com/Luce-Org/lucebox)) for bounded replay, adaptive verify width, and evaluation corpus.
- 05yuki ([05yuki/deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)) for INT4 LM head and tool namespace specs.
- `dealignai` and `pqhaz` for the uncensored and abliterated weights.
- Transformation of affinity into this R9700-tweaked engine was executed with DeepSeek-V4.1-Flash, Gemini 4 Carbon, Gemini 4 Argon, and Gemini 4 Hydrogen.
