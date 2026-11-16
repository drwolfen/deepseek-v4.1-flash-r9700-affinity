# deepseek-v4.1-flash-r9700-affinity

A working setup for serving **DeepSeek-V4.1-Flash** (the uncensored / abliterated build) on **two AMD
Radeon AI PRO R9700s**, using a fork of the [affinity](https://codeberg.org/StillDeadcode/affinity)
inference engine, plus a locally quantised 2.875-bpw container of the model.

This repository is the *setup*: the engine source (fork + our patches), the deployment unit, the
quantisation recipe, and the measurements, all captured on one machine. It exists so the exact
combination that produced the numbers below can be rebuilt.

The affinity engine was transformed into this **R9700-tweaked version with the help of
DeepSeek-V4.1-Flash, Gemini 4 Carbon, Gemini 4 Argon, Gemini 4 Hydrogen, and Gemini 3.8 Flash**
(used to port, patch and tune the engine — see *Rework and patches*).

> **Tested on exactly two R9700s.** Every number here was measured on a two-card box. The engine
> upstream supports one card, but that configuration was not benchmarked for this setup.

**Model files (MoonMath 1017):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-MoonMath-aff](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-MoonMath-aff)  
**Model files (1017 revision):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw)  
**Model files (legacy 0925):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw)  

**Changelog & Benchmarks:** [**`CHANGELOG.md`**](CHANGELOG.md) | **Latest Release:** [**`v5.1.0`**](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/releases/tag/v5.1.0) | **Bake-Off Reports:** [**`v5.1.0 & MoonMath-1017`**](docs/benchmarks/2026-11-16-v5.1-and-moonmath-1017-bakeoff.md) · [**`v5.0.0 & Radiance v2.1`**](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md)

**Release Highlights (Waves 0–4 Progression):**
- **v5.1.0** (`839665f`): DSpark ring-attention tile skip (`AFF_DRAFT_SKIP_MASKED=1`, **+22.2%** decode speedup: 16.93 -> 20.69 tok/s), paired `temp=0` DSpark acceptance **`73.33%`** on `MoonMath-1017` (vs 67.52% on v5.0.0, **+5.81 pp**; **80.0%** on code, **+9.0%** paired mean gen), compact state-only delta prefix cache (`AFF_PCACHE_DELTA=1`, 1.42× multi-turn TTFT speedup), `ignore_eos=true` raw streaming, GPU `gridDim.y > 65535` overflow guard (#1006) for deep contexts, DSML nested XML tool-argument protection and multi-call invoke splitting (#1020), MoE stream RAW barrier validation (#983, #984). Retained `AFF_ATTN_LOOKAHEAD=0` default (regressed median decode by -11.4%). Verified 3,416 resident experts, [314, 302] MiB active free VRAM, 0 OOM / 0 spill. 6/6 CPU unit tests PASS.
- **v5.0.0** (`264e8d5`): RDNA 4 gfx1201 sparse indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`), `K_VECS_PADDED=136` + 8B Q staging, 4-pass 256-bucket radix top-512, Wave32 tie compaction, live-`seqused` ragged causal prefill. Score budget scratch bounded to 36 MiB (`<= 48 MiB/card`), expanding GPU resident capacity to **3,416 MoE experts (+49)** with `[356, 354] MiB` free VRAM (`0 OOM / 0 spill`). Long-context: **14.40× @64K** & **13.15× @128K** prefix-reused prefill (128K TTFT `56.6s -> 4.3s`, `22.55 tok/s` decode). Cold uncached prefill `963.66 tok/s` @32K & `990.98 tok/s` @64K (`25.18 tok/s` decode). Needle retrieval **4/4 (100%)** across 4K–128K, `refattn.py` **99.98%** parity, `quant_eval` **92.5%** (111/120), 9/9 tests PASS.
- **v4.5.0** (`9ec7c47`): Fused & chained hybrid compute (`AFF_HC_FUSED_V2=1`, `AFF_HC_CHAINED=1`, `AFF_HC_CHAINED_MIN_ROWS=1`) -> **`23.73 tok/s` decode median (`24.37` peak)**, **`22.4 tok/s` batch-8 aggregate**, `0.000000 nats` 8,184-tok KL (`PPL 2.6057`), `90.83%` `quant_eval`, 8/8 tests PASS. Documented rejected toggles (`AFF_EMBED_HOST=0` as `token_embd` already host-mmap'd at 0 MiB VRAM; `AFF_MOE_SUMMED_DOWN=0`).
- **v4.4.0** (`718b42e`): Toolchain upgrade to ROCm 10.1 (`20.41 tok/s` vs ROCm 7.14 `18.49 tok/s`, **+10.4%**), `AFF_CACTUS_ESCAPE=0.25` convex mixture (**`0/20` repetition lock-in** @ `temp=0.6`), 64 GiB multi-slot persistent prefix cache, Issue #12 `heat -> static` automatic fallback, cross-shard `.scale` lookup fix, deterministic greedy restore, DSML typed-arg tag parity.
- **v4.3.3** (`5275c64`): Unified 3-way divergent branch history into single master tree; added `GET /health` JSON endpoint, `--raw-completions` pass-through, and `prompt_tokens_details.cached_tokens` reporting; 7/7 tests PASS.
- **Wave 4 Radiance v2.1.0 Bake-Off**: Upstream `StillDeadcode/radiance v2.1.0` built and evaluated on ROCm 10.1 `gfx1201`. Confirmed that `deepseek41` model architecture plugin, MLA/Engram kernels, and `F8_E8M0` scales are omitted in open-source Radiance, keeping `affinity v5.0.0` as the sole working production engine.

---

## Hardware (the machine the numbers come from)

| | |
|---|---|
| GPU | **2 × AMD Radeon AI PRO R9700**, `gfx1201` (RDNA4), 32 GiB each |
| CPU | Intel Xeon Platinum 8160 (Skylake-SP), 2 sockets × 24 cores / 48 threads = 96 threads |
| RAM | 754 GiB DDR4 (2 NUMA nodes) |
| PCIe | GPUs link at **Gen5 ×16**, but the switch **uplink runs Gen3 ×16 (8.0 GT/s)** — the host Xeon is a PCIe-3.0 CPU. |
| Host OS | **Ubuntu 24.04.5 LTS** (Noble Numbat), kernel 7.0.0, ROCm 10.1.0 |
| Engine build | Docker image `deepseek-affinity:v5.1.0` (ROCm 10.1.0-full, `gfx1201`) |
| Serving ports | Dual active: `:8000` (Docker container) and `:8080` (local port-8080 TCP proxy for agent pipeline) |

### Minimum RAM

The routed experts do not fit in 64 GiB of VRAM. At 2.875 bpw the routed experts are ~93 GiB and the
whole container is **191.32 GiB**; the two cards hold ~42 GiB of expert slab, so the rest lives in host
RAM.

- **~160 GiB RAM** is the practical floor for this configuration (512k context, DSpark, engram): the
  host expert pool alone is **143.5 GiB** plus the OS and page cache.
- Below that, experts spill to the SSD tier and decode collapses (the engine's own note: a 32 GiB card
  with 64 GB RAM runs ~8 tok/s instead of ~23+ because most experts end up on disk).
- At 512k context the KV cache is committed at 262,144 backed positions (0.66 GiB VRAM); the compressed
  indexer/prefix cache is on disk (`--prefix-cache-dir .../pcache`, 64 GiB).

---

## Measured Performance Progression (Waves 0–4)

Full methodology, telemetry charts, and raw outputs are documented in [`docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md`](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md).

### Progression Summary Table

| Wave / Version | Git SHA | Decode Median | Batch-8 Aggregate | Long Prefill (@64K) | Resident Experts | Repetition Rate (`temp=0.6`) | Test Suite |
|---|:---:|---:|---:|---:|---:|:---:|:---:|
| **v4.2.0 Baseline** | `14e91c4` | 16.33 tok/s | 10.9 tok/s | ~1,200 tok/s | 3,157 | frequent traps | 6/6 PASS |
| **v4.3.0 / v4.3.2** | `e2a4478` | 19.27 tok/s | 21.4 tok/s | 1,180 tok/s | 3,279 (+88 GPU1) | loop traps | 7/7 PASS |
| **Wave 0 (v4.3.3)** | `5275c64` | 18.31 tok/s | 16.9 tok/s | 1,206 tok/s | 3,367 | loop traps | 7/7 PASS |
| **Wave 1 (v4.4.0)** | `718b42e` | 20.41 tok/s | 20.9 tok/s | 1,208 tok/s | 3,367 | **0/20 (0%)** | 7/7 PASS |
| **Wave 2 (v4.5.0)** | `9ec7c47` | **23.73 tok/s** (24.37 peak) | **22.4 tok/s** | 1,215 tok/s | 3,367 | 0/20 (0%) | 8/8 PASS |
| **Wave 3 (v5.0.0)** | `264e8d5` | **23.73 tok/s** | **22.4 tok/s** | **21,448 tok/s (14.4×)** | **3,416 (+49)** | **0/20 (0%)** | **9/9 PASS** |
| **Wave 4A (v5.1.0)** | `839665f` | **20.69 tok/s** (22.63 peak) | **22.4 tok/s** | **21,448 tok/s (14.4×)** | **3,416** | **0/20 (0%)** | **6/6 PASS** |

### Wave 3 / Wave 4 Long-Context Prefill Acceleration (`v5.0.0`)

Measured across deep sequence lengths with the hardware RDNA 4 sparse indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`):

| Context Depth | Prompt Tokens | Baseline TTFT (`idx0`) | v5.0.0 TTFT (`idx1`) | Baseline Prefill | v5.0.0 Prefill | Speedup Factor |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **4K** (Pass 1) | 3,810 | 7.7 s | 3.2 s | 496.7 tok/s | 1,186.1 tok/s | **2.39×** |
| **16K** (Pass 1) | 15,163 | 12.9 s | 3.6 s | 1,173.0 tok/s | 4,220.3 tok/s | **3.60×** |
| **64K** (Pass 2) | 60,503 | 40.6 s | 2.8 s | 1,489.8 tok/s | 21,448.9 tok/s | **14.40×** |
| **128K** (Pass 2) | 120,976 | 56.6 s | 4.3 s | 2,137.1 tok/s | 28,092.6 tok/s | **13.15×** |

Cold uncached prefill (`cached_tokens == 0` via UUID4 nonces on live `:8000`) sustains **963.66 tok/s @32K** and **990.98 tok/s @64K** (with post-64K decode running at **25.18 tok/s**).

### Needle-in-a-Haystack & Numerical Parity

- **Needle-in-a-Haystack (`needle_ab`)**: **4/4 (100%)** retrieval accuracy across 4K, 16K, 64K, and 128K depth scales.
- **Reference Attention Parity (`refattn.py`)**: **99.98%** cosine parity against uncompressed dense attention.
- **KL Divergence**: **0.000000 nats** divergence (PPL 2.6057) across 8,184 tokens.
- **Quantization Eval (`quant_eval`)**: **92.50%** accuracy (111/120 items).

---

## Wave 4 Bake-Off: `affinity v5.0.0` vs `StillDeadcode/radiance v2.1.0`

In November 2026, upstream Radiance published version `v2.1.0` (`2a2b756`). We constructed an offline evaluation image (`radiance-eval:v2.1.0` on ROCm 10.1 `gfx1201`) to assess upstream compatibility.

### Architectural Feature Comparison

| Feature / Architecture Component | `deepseek-affinity:v5.0.0` | `StillDeadcode/radiance v2.1.0` |
|---|:---:|:---:|
| **Target Hardware Support** | AMD RDNA 4 `gfx1201` (Dual R9700) | AMD RDNA 4 `gfx1201` (Dual R9700) |
| **DeepSeek-V4.1 Architecture** | **Full Native Support** | **Blocked** (Plugin not in tree) |
| **Multi-Head Latent Attention (MLA)** | **Native Decoupled KV Kernel** | Missing in `libr4d 0.5.0` |
| **Sparse Indexer (Top-512)** | **Native gfx1201 Radix Sort** | None |
| **Engram 188.8 GiB Demand Paging** | **Native Zero-Copy UVA** | None |
| **FP8 Scale Format (`F8_E8M0`)** | **Full Native Decoding** | **Unsupported** (`RAD_DT_INVALID`) |
| **Persistent Multi-Slot Prefix Cache** | 64 GiB NVMe multi-slot persistent | In-memory only |
| **Repetition Defense** | `AFF_CACTUS_ESCAPE=0.25` (0/20 locked) | None |
| **MoE Resident Capacity** | **3,416 shards** (Dual GPU) | N/A (cannot load model) |
| **Decode Throughput (Single Stream)** | **23.73 tok/s** (24.37 peak) | N/A |
| **Batch-8 Throughput** | **22.4 tok/s** | N/A |
| **64K Prefill Throughput** | **21,448.9 tok/s** (2.8s TTFT) | N/A |

**Verdict**: Radiance v2.1.0 cannot serve DeepSeek-V4.1-Flash because the architecture plugin and MLA/Engram kernels are omitted from the open-source release, and its safetensors reader explicitly rejects `F8_E8M0` scales. `deepseek-affinity:v5.0.0` remains the sole working high-throughput production engine.

---

## Rework and patches (with measured effect)

> **Every commit below is traceable.** Click the short SHA to open the full diff on GitHub. All of the
> local work lives in this repository's history, so you can read the actual code change behind each
> claim — the vision port, the KV sizing, the tool-call recovery, the imatrix calibration, the
> `AFF_NOBUDGET` patch, the hyper-connection kernels, and the RDNA4 `gfx1201` sparse attention indexer.

All historical lines (`v1.0.0`, `v1.1.0`, `v41-tools-fix`, `merge/v41-tools-fix-v4.3.0`, and Waves 0–4 through `v5.0.0`) are unified in `main`.

### ⭐ Local work & engine updates (`v1.0.0` → `v5.0.0`)

These are the local changes and backports published in this repository's history, ordered chronologically from initial bring-up through `v5.0.0`:

| diff | change | measured effect |
|---|---|---|
| [`91e78ba`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/91e78ba) | **DeepSeek ViT on the CPU**, wired into prefill and the chat endpoint. Three reference discrepancies fixed (2D-RoPE half rotation, channel-major aligner, aspect-preserving pad) | embedding cosine vs reference 0.31 → 0.85; end-to-end image grounding works |
| [`cbb3f07`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cbb3f07) | ViT **weight-stationary GEMM + AVX-512 attention** | 546×546: 13.2 s → 3.25 s; 1024×1024: minutes → 38 s |
| [`e1a2025`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e1a2025) | ViT **query-blocked attention**; opt-in engram page-cache warm (`AFF_ENGRAM_WARM`) | 74×74: 38.4 s → 23.3 s, numerically identical |
| [`cac0510`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cac0510) | size the compressed KV cache **per kv source, not per layer** | KV 1016 → 100 MiB/card; resident experts 2206 → 2357 |
| [`997489c`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/997489c) | **tile the indexer byte mask** over the prefill chunk | prefill arena 2068 → 1426 MiB/card; resident experts 2357 → 2462 |
| [`78244dc`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/78244dc) | **persist the learned placement heat plane** across restarts (`--placement-save`) | residency survives restarts (no re-learn from zero) |
| [`a4c9f85`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a4c9f85) | recover **marker-less ("bare") DSML tool calls** and stop at the block close | a poisoned 117k-token replay dropped 4360 → 195 completion tokens; stops agent runaway/confabulation |
| [`bc42be2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bc42be2) | DSpark **confidence head** + opt-in gating (`AFF_SPEC_CONF_T`) | wash on 2 cards — kept as a calibration instrument |
| [`27ec8fa`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/27ec8fa) | opt-in DSpark backoff after a zero-accept streak (`AFF_SPEC_BACKOFF=N`) | wash on 2 cards (verify dominates, not the draft) — kept as a knob |
| [`1ae7c09`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1ae7c09) | **SSE keep-alive** while a tool call is withheld | long tool calls no longer trip client body-idle timeouts (300 s) |
| [`2a4478f`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2a4478f) | **on-GPU imatrix calibration** (`--calibrate-tokens/-chunk-size/-max-chunks`, `--imatrix-in/out`, `imatrix_accum_*` kernels), **bounded replay** (`AFF_BOUNDED_REPLAY`), **soft resident bias** (`AFF_RESIDENT_BIAS`), pinned sampler staging, async peer-copy overlap, `AFF_BLOCK_AT` hooks | engine emits its own imatrix from the model's activations; host↔device copies overlap |
| [`2779348`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2779348) · [`patches/AFF_NOBUDGET_MAX_TOKENS.patch`](patches/AFF_NOBUDGET_MAX_TOKENS.patch) | bound the default when a request sends **no** `max_tokens` (was: the whole remaining context) | runaway reasoning request can no longer hold a slot for 37 min / 55k tokens; explicit `max_tokens` untouched |
| [`154eb2b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/154eb2b) | **`v1.1.0` Radiance & KTransformers ports**: fused `hc_post` + DSpark tap (`AFF_FUSE_HC_TAP`), on-GPU group-128 INT4 draft head (`AFF_DRAFT_HEAD_INT4`), FP8 attention output QDQ (`AFF_O_FP8_QDQ`), Cactus capped rejection (`AFF_CACTUS_DELTA`, `AFF_CACTUS_CAP_R`), 1017 `namespace::name` tool parsing | decode 17.08 → 21.96 tok/s (+28.6%) @163k–171k ctx; step latency -42.8 ms/block |
| [`23ec996`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/23ec996), [`6c77094`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/6c77094), [`308638e`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/308638e) | coalesced TR `MT=1` sub-tile loads + decode-gated **`AFF_MOE_RECON_TILES=32`** L2 slicing; opt-in EXL3 block-32 Walsh–Hadamard rotation (`AFF_HADAMARD_ROT`) and device-planned prefetch ring (`AFF_PREFETCH_RING`) | removes ~480 empty GPU launches per verify step while keeping `rcap=32` on prefill |
| [`7fd7a71`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/7fd7a71), [`9e5db70`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/9e5db70), [`1f0d668`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1f0d668) | **`v4.2.0` VRAM accounting & placement guards**: sysfs-checked `hip_free_vram_checked`, periodic `--placement-save-secs` telemetry (moves/1000 ticks), hysteresis/min-gain `1.6 / 3.0`, HTTP 400 guard on over-long prompts | 479k-token free VRAM 66 → 920 MiB; H2D churn 113 → 52 GiB (-54%); promotions 9,533 → 4,422 |
| [`026a9b0`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/026a9b0), [`3a06ea1`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/3a06ea1) | **`v4.3.0` GPU line + `v1.1.0` merge**: fused V4.1 narrow `hc_pre` (`AFF_V41_HC_FUSED=1`), LDS FP8 scale prefetch, zero-copy `b_pre` swap, prefill placement freeze (`AFF_PREFILL_FREEZE_PLC=1`), asymmetric GPU1 bonus slab (`AFF_BONUS_SLAB=1`) | decode median 16.33 → 19.27 tok/s (+18.0%); batch-8 10.9 → 21.4 tok/s (+96.3%); resident experts 3,157 → 3,367 |
| [`21ba484`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/21ba484), [`fa1588b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/fa1588b), [`5275c64`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/5275c64) | **Wave 0 (`v4.3.3`)**: return `finish_reason: "stop"` on premature EOS inside `<think>`; add `GET /health` JSON endpoint, `--raw-completions`, and `cached_tokens` reporting; active release asserts in `test_gpu_ports`; unify 3-way branch divergence | single linearised `main`; clean `<think>` EOS termination; 7/7 tests PASS |
| [`17682d7`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/17682d7), [`3551e14`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/3551e14), [`34610b4`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/34610b4), [`718b42e`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/718b42e) | **Wave 1 (`v4.4.0`) + ROCm 10.1**: thread-safe multi-slot persistent prefix cache, `heat → static` placement auto-fallback, cross-shard `.scale` quantizer lookup, **Cactus convex escape mixture** (`AFF_CACTUS_ESCAPE=0.25`), V4.1 DSML typed-argument tag parity, bonus-slab `resident_experts` accounting | decode median 18.31 → 20.41 tok/s (+10.4% w/ ROCm 10.1); `temp=0.6` reasoning loop lock-in **0/20 (0%)** |
| [`f849997`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/f849997), [`bf2ae04`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bf2ae04), [`cbf07da`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cbf07da), [`14f149d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/14f149d), [`9ec7c47`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/9ec7c47) | **Wave 2 (`v4.5.0`)**: fused `hc_read` + pipelined tiled up-mix (`AFF_HC_FUSED_V2=1`), chained multi-row hyper-connection (`AFF_HC_CHAINED=1`, `AFF_HC_CHAINED_MIN_ROWS=1`), scheduler/drafter/KV/TP ports, 8,184-token KL harness (`forward_prefill`); gated off net-negative `AFF_EMBED_HOST` & `AFF_MOE_SUMMED_DOWN` | decode median **20.41 → 23.73 tok/s** (24.37 peak, **+23.1% vs v4.3.2**); batch-8 **22.4 tok/s**; KL **0.000000 nats** (PPL 2.6057); 8/8 tests PASS |
| [`570e787`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/570e787), [`a02f669`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a02f669), [`264e8d5`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/264e8d5) | **Wave 3 (`v5.0.0`)**: native RDNA4 `gfx1201` sparse attention indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`), `K_VECS_PADDED=136` + 8B Q staging, multi-CTA 4-pass 256-bucket radix top-512, Wave32 tie compaction, live-`seqused` ragged causal prefill, short-seq skip + chunk metadata reuse, 36 MiB score scratch budget | **14.40× @64K** (21,448.9 tok/s, TTFT 40.6 s → 2.8 s) & **13.15× @128K** (28,092.6 tok/s, TTFT 56.6 s → 4.3 s); resident experts **3,367 → 3,416 (+49)**; needle **4/4**, `refattn` **99.98%**, `quant_eval` **92.5%**, 9/9 tests PASS |
| [`a041faf`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a041faf) | **Wave 4 bake-off**: offline ROCm 10.1 `gfx1201` build & evaluation of `StillDeadcode/radiance v2.1.0` against `deepseek-affinity:v5.0.0` | verified open-source Radiance omits `deepseek41` plugin, MLA/Engram kernels, and `F8_E8M0` scales; `v5.0.0` retained in production |
| [`68011d2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/68011d2) | **#1020 DSML nested XML parser & multi-call invoke split**: protect literal `<tool_call>` / `<antml:invoke>` tags in argument strings; split multi-call blocks into distinct OpenAI function-call array items | nested XML literal tools and multi-call functions parse without truncation |
| [`af2d390`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/af2d390) | **#989 state-only delta prefix cache & ignore_eos**: save compact delta checkpoints (`AFF_PCACHE_DELTA=1`); support `ignore_eos=true` for raw streaming completions | 1.42× TTFT speedup on multi-turn prefix hit (1.22 s → 0.86 s); exact requested token streaming |
| [`5dbad17`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/5dbad17) | **#1006 GPU gridDim.y hardware overflow guard**: 2D tile batch token indexing to prevent hardware grid dimension overflow on prefill >65,535 rows | eliminates launch failures on deep sequences (128K context verified) |
| [`bd3bf03`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bd3bf03) | **#958 AFF_DRAFT_SKIP_MASKED ring tile skip**: skip masked causal tiles in DSpark ring-attention verification kernel | decode median 16.93 → 20.69 tok/s (**+22.2%**); verified bit-exact parity |
| [`39e48c6`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/39e48c6) | **#980, #992 AFF_ATTN_LOOKAHEAD & Wave32 DPP**: RDNA 4 decode KV prefetch and Wave32 cross-lane DPP reductions | evaluated across workloads; regressed median decode -11.4% (20.69 → 18.33 tok/s), retained as opt-in knob (default 0) |
| [`1276495`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1276495) | **#983, #984 MoE stream join invariants & multi-row dedup**: document stream join invariants and test multi-row unique-read deduplication across concurrent streams | proves memory barrier correctness across asynchronous stream execution |
| [`70b3aba`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/70b3aba), [`839665f`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/839665f) | **Wave 4A (`v5.1.0`) release**: tools (`aff-cfgcheck`, `aff-fp8check`, `aff-exphash`, `aff-bench`), defaults (`AFF_ATTN_LOOKAHEAD=0`, `AFF_DRAFT_SKIP_MASKED=1`, `AFF_PCACHE_DELTA=1`), version bump `v5.1.0` | 6/6 CPU unit tests PASS; MoonMath-1017 paired DSpark acceptance **73.33%** (+5.81 pp over v5.0.0; +9.0% paired mean gen); 3,416 resident experts, [314, 302] MiB free VRAM |

### V4.1 / multimodal port (fork `v41` branch, also in this history)

| diff | change | improvement |
|---|---|---|
| [`e6070f5`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e6070f5) | V4.1 reasoning-effort prefix + `陪同下` token, byte-identical to the reference `encoding.py` | correct V4.1 chat encoding |
| [`e03a05b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e03a05b), [`774aa6a`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/774aa6a), [`4165fce`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4165fce), [`4bc938b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4bc938b) | engram gate/stride, engram bf16 weights, V4.1 has no per-head RMS on q | four bugs that produced fluent-but-wrong text; fixes make the port correct |
| [`d91ae9d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/d91ae9d) | engram n-gram layers 1 and 14 (91.6 GiB tables), indexer, compressor | V4.1 architecture support |
| [`2dbb093`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2dbb093) | stop retrying a KV growth the card cannot hold | removes the 64k decode cliff |

See [`CHANGELOG.md`](CHANGELOG.md) and [`docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md`](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md) for full version history, benchmarks, and third-party attributions ([`THIRD_PARTY.md`](THIRD_PARTY.md)).

---

## Production Safety Invariants & Hard Knobs

The production engine enforces strict operational invariants:
- **Target LM Head BF16**: Never enable `AFF_LMHEAD_INT4=1` on the primary model (INT4 head incurs significant perplexity and acceptance loss). Draft model head INT4 is maintained (`AFF_DRAFT_HEAD_INT4=1`).
- **Equal Micro-Batching**: `AFF_EQUAL_UBATCH=0` (kept disabled; adds kernel overhead without reducing chunk count).
- **Prefetch Ring**: `AFF_PREFETCH_RING=0` (kept disabled; ring prefetch consumes 8–12 resident expert slots without throughput gain on PCIe 3.0 x16).
- **GPU Resident MoE**: MoE expert computations run 100% on GPU tensor cores; zero MoE compute is offloaded to CPU.
- **No Live KV Unmapping**: Live `kv_shrink_to` page unmapping is forbidden to prevent page table fragmentation and deadlock during multi-turn streaming.
- **Host Embedding Offload Rejected**: `AFF_EMBED_HOST=0` (investigation proved `token_embd` is already host-mmap'd at 0 MiB VRAM; setting the flag added host synchronization overhead with 0 MiB VRAM saved).

### Environment Knobs in the Served Container

```sh
AFF_ENGRAM_WARM=1
AFF_SPEC_BACKOFF=2
AFF_ADAPTIVE_WIDTH=1
AFF_BOUNDED_REPLAY=1
AFF_FUSE_HC_TAP=1
AFF_DRAFT_HEAD_INT4=1
AFF_CACTUS_DELTA=0.35
AFF_CACTUS_CAP_R=0.5
AFF_CACTUS_ESCAPE=0.25
AFF_RESIDENT_BIAS=0.008
AFF_MOE_RECON_TILES=32
AFF_PREFETCH_RING=0
AFF_NOBUDGET_MAX_TOKENS=16384
AFF_HC_FUSED_V2=1
AFF_HC_CHAINED=1
AFF_HC_CHAINED_MIN_ROWS=1
AFF_SPARSE_INDEXER=1
AFF_INDEXER_SCORE_BUDGET_MB=36
AFF_DRAFT_SKIP_MASKED=1
AFF_ATTN_LOOKAHEAD=0
AFF_PCACHE_DELTA=1
```

---

## Build and Run

### Building the Docker Container (ROCm 10.1)

```sh
docker build -f Dockerfile -t deepseek-affinity:v5.1.0 .
```

### Serving DeepSeek-V4.1-Flash (Production Topology)

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
  -e AFF_NOBUDGET_MAX_TOKENS=16384 -e AFF_HC_FUSED_V2=1 -e AFF_HC_CHAINED=1 \
  -e AFF_HC_CHAINED_MIN_ROWS=1 -e AFF_SPARSE_INDEXER=1 -e AFF_INDEXER_SCORE_BUDGET_MB=36 \
  -e AFF_DRAFT_SKIP_MASKED=1 -e AFF_ATTN_LOOKAHEAD=0 -e AFF_PCACHE_DELTA=1 \
  -p 8000:8000 \
  -v /models:/models \
  deepseek-affinity:v5.1.0 \
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
    --default-effort 75 \
    --repeat-penalty 1.1 --repeat-last-n 96 --dry-delta 40 \
    --max-tokens-floor 0
```

Health endpoint check:
```sh
curl -sS http://127.0.0.1:8000/health
# {"status":"ok","version":"v5.1.0","git_sha":"...","resident_experts":3416,"free_vram_mib":[204,300]}
```

---

## Credits

- DeepSeek for the base model architecture and reference code.
- Yoshi Exeler / StillDeadcode for affinity and Radiance algorithmic foundations.
- Lucebox AI ([Luce-Org/lucebox](https://github.com/Luce-Org/lucebox)) for bounded replay, adaptive verify width, and evaluation corpus.
- 05yuki ([05yuki/deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)) for INT4 LM head and tool namespace specs.
- `dealignai` and `pqhaz` for the uncensored and abliterated weights.
- Transformation of affinity into this R9700-tweaked engine was executed with DeepSeek-V4.1-Flash, Gemini 4 Carbon, Gemini 4 Argon, Gemini 4 Hydrogen, and Gemini 3.8 Flash.
