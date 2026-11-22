# Changelog

All notable changes to this fork. Format follows [Keep a Changelog](https://keepachangelog.com/),
versions follow [Semantic Versioning](https://semver.org/).

The engine serves DeepSeek-V4.1-Flash-1017 (`.aff`, 2.875 bpw FP4 experts, DSpark draft, Engram)
on 2× Radeon AI PRO R9700 (`gfx1201`). Numbers below are from that box unless stated.

## [5.6.2] — 2026-11-22 — "DSpark bug fixes, shared FP8 LM head (+55 experts to 3,699) & Wave 10A quick wins"

Wave 10A maintenance and performance release (`1676e05`). Fixes multi-`n_expert` router bias overwrite (`BUG-1`), draft router `mmap` over-read and per-call banding (`BUG-2`), `~DenseGpu()` interior-pointer / pinned-memory cleanup, dynamic per-block repeat penalty (`BUG-5`), think-scoped DRY prompt protection (`BUG-6`), and short-scale mantra/novelty `ThinkingBudget` breakers (`BUG-7`), while shipping five Wave 10A quick wins (`C4` shared FP8 LM head, `C1` Rank-0 router, `A1` fixed narrow `ar_blk=32`, `B4` `e4m3_decode_pk` + `AFF_DRAFT_ATTN_NHB=2`, `D1` `AFF_ADAPTIVE_COST_RESCALE=1`) that grow live GPU-resident experts by **`+55` (`3,644 -> 3,699`)** and push `l4k`/`l64k` warm decode to **`50.04 / 51.52 tok/s`**. Preserves `-ffp-contract=off` (`33/33`), `0` scratch / `0` VGPR spill (`673` `gfx1201` kernels), and `pcache` fingerprint `0x733de762fee543e6`.

### Added & Changed
- **Router & Teardown Fixes (`BUG-1`, `BUG-2`, `~DenseGpu()` in `src/engine/model.cpp`, `src/gpu/dense_gpu.h`, `src/gpu/dense_gpu.hip`)**:
  - `router_scratch()` now sizes `D.b_rbias` with uniform `r_bias_stride = std::max(r_max_expert_, n_expert)` and `batch_router_topk()` checks `n_expert <= D.r_bias_stride`, stopping draft stages (`n_expert=128`) from overwriting target layers 13/14 biases (`n_expert=384`).
  - `bind_layer_common()` uses `ffn_gate_inp` row count (`ne`) instead of `cfg.n_expert` (fixing `2.50 MiB` draft `mmap` over-read), and `reg_router`/`router`/`batch_router` compute `router_band()` per call with dynamically grown `D.d_rlog` (`d_rlog_cap`).
  - `~DenseGpu()` and `idx_grow()` avoid freeing interior aliases (`b_rwt`/`b_rn` in `b_rsel`, `h_rwt`/`h_rn` in `h_rsel`), add `b_conf`/`b_hcp` frees, and use `hipHostFree` on `hi_w`/`hi_nc`/`hi_ncb`.
- **Repetition, DRY & Mantra Loop Guards (`BUG-5`, `BUG-6`, `BUG-7` in `src/engine/thinking_budget.h`, `tools/affinity.cpp`)**:
  - Added per-block `refresh_penalty_ids()` (`AFF_REPEAT_PENALTY_DYNAMIC=1`) exempting `eos`, `think_marker_ids`, and DSML `dry_exempt_ids`.
  - Restricted DRY's `2x` prompt-verbatim threshold to post-`</think>` (`!in_think`).
  - Added `short_low_novelty` bigram diversity breaking (`AFF_THINK_SHORT_NOVELTY_WINDOW=128`, `MIN_TOKENS=160`, `MIN=0.20`), tuned `AFF_THINK_SHORT_LOOP_SPAN=16`, `REPEATS=3`, `WINDOW=1024`, `AFF_THINK_NOVELTY_WINDOW=512`, `MIN_TOKENS=1024`, `MIN=0.28`, and set `AFF_THINK_FREEZE_DISTINCT=56`.
- **Shared FP8 LM Head (`C4`) & Wave 10A Quick Wins (`C1`, `A1`, `B4`, `D1`)**:
  - Removed `-e AFF_DRAFT_HEAD_INT4=1` from `deploy/{deepseek-v41-affinity.service,run.sh}` so DSpark `draft_head` shares `AFF_LMHEAD_FP8=1` (`320.6 MiB/shard`), and enabled `AFF_ROUTER_RANK0_ONLY=1` (`C1`, removing Rank-1 `D.router` and unused `D.rbias` allocations) -> **`3,699` live resident experts (`+55` vs `v5.6.1`)** at `[354, 354]` MiB free VRAM.
  - Enabled `AFF_AR_FIXED_BLK=32` (`A1` in `src/gpu/tp_allreduce.hip`) for narrow `RegForm` batches (`nb <= 16 && live <= 16`) to avoid `tp_sync_on_grid_change` host stream syncs across DSpark↔Verify width changes.
  - Switched `attend_body<DT==4>` (`src/gpu/attention_gpu.hip`) to `2x e4m3_decode_pk` (`aff_f32x2`) and defaulted `AFF_DRAFT_ATTN_NHB=2` (`B4`) for `flat_n != 0 && n <= 8 && n_head >= 16`.
  - Enabled `AFF_ADAPTIVE_COST_RESCALE=1` (`D1` in `tools/affinity.cpp`) to proportionally rescale adjacent `cost_ema` buckets.
  - Empirically withheld `PCIe #5` (`D.ncomp` skip / `D.inc` ring) and `StageRing::fit` 4x shrink hysteresis after server benchmarking showed `-2%..-4.6%` prefill & decode regressions under `GPU_MAX_HW_QUEUES=1`.
  - Bumped `VERSION`, `CMakeLists.txt`, `Dockerfile`, and `deploy/` image tag to `5.6.2`.

### Performance & Verification
- **Clean Cold-Start NUMA-Local Suite (`v5.6.2` `1676e05`, `--default-effort 60`, `N0=100%`)**:
  - `l64k` (`63,863`): **`51.52 tok/s`** warm `tg2` (**`+10.1%`** vs `v5.6.1` `46.81`), `47.79` cold `tg1`, `1,405.8` `pp1` (`45.4 s`), `2,585.5` `pp2` (`24.7 s`).
  - `l4k` (`4,031`): **`50.04 tok/s`** `tg2` (**`+7.7%`** vs `v5.6.1` `46.48`), `28.52` `tg1`, `634.2` `pp1`; `l16k` (`16,104`): **`49.75 tok/s`** `tg2` (`+7.0%` vs `v5.6.1` `46.48`), `45.16` `tg1`, `1,298.5` `pp1`.
  - `p4k` (`NGEN=16`): **`21.24 tok/s`** `tg2` (**`+16.5%`** vs `v5.6.1` `18.23`); `code_edit` (`NGEN=256`): **`43.75 tok/s`** `tg2` (`8.00 tok/blk`); `step_c` 4-prompt DSpark: `19.90 tok/s` clean (`23.48 tok/s` / `75.13%` acc / `3.200 tok/blk` in `arm5a_nhb2` ablation).
- **Audits & Tests**: **`11/11` CPU `ctest`** (incl. `thinking_budget` Scenario 7), **`5/5` GPU binaries** (incl. `test_gpu_ports` Tests 26–28), **`33/33` `-ffp-contract=off`**, **`673` `affinity` / `672` `test_gpu_ports` kernels (`0` scratch / `0` VGPR spill)**, **`9/9` fast `DT==4` `mqa_attend*` ISA `>=2x v_cvt_pk_f32_fp8`**, `N0=47,804,160 (100.0%)`, `resident_experts=3,699`, `DRY 3/3`, `test_a_stride` (`33,792/35,144`), `test_b_think_ckpt` (`1,480/1,596`), `probe_bug567` `all_pass=true`, cross-version `pcache` (`3,584/4,030`, `0x733de762fee543e6`, `0` warnings), `0` `hipError`, `kfd_release_s=3.387 s`.

## [5.6.1] — 2026-11-22 — "Dynamic DSpark-aware SuffixGate: pure-DSpark reasoning speed + 8 tok/blk code edits"

Wave 9.1 follow-up release (`3f29dbf`, `cfcbb18`). Fixes false-positive 4-token prose preemption in `SuffixDraftIndex` (`R6`) via a live utility-, anchor-length-, and acceptance-aware `SuffixGate` (`AFF_DRAFT_SUFFIX_DYNAMIC=1`, `AFF_DRAFT_SUFFIX_GAIN=1.05`) without touching GPU kernels (`673` kernels, `0` scratch, `0` VGPR spill) or `pcache` fingerprint `0x733de762fee543e6`.

### Added & Changed
- **Dynamic DSpark-Aware `SuffixGate` (`3f29dbf`, `cfcbb18` in `src/engine/suffix_draft.h`, `tools/affinity.cpp`)**:
  - Added `AFF_DRAFT_SUFFIX_DYNAMIC=1` (default ON; `"0"` restores static `R6` gating) and `AFF_DRAFT_SUFFIX_GAIN=1.05` (minimum utility gain over GPU DSpark).
  - Under active GPU DSpark, compares effective suffix utility `(1 + expect * acc_ema) / cost_ema[prop_nd + 1]` against live DSpark utility `(1 + dspark_exp_acc) / cost_ema[B + 1] * sfx_gain`, enforces an anchor-length guard (`>= min_match + 2` cold / `>= min_match + 1` once `steps > 0 && miss_streak == 0 && acc_ema >= 0.80f`) so 4-token and cold 5-token prose collisions never waste verify blocks, tracks empirical suffix `acc_ema` (`0.55 / 0.45` blend), and arms exponential miss cooldown (`2, 4, 8, 16`) with `acc_ema >= 0.65f` breakout protection.
  - Added 3 unit tests in `tests/test_suffix_draft.cpp` (`test_suffix_gate_prose_backoff`, `test_suffix_gate_code_edit_burst`, `test_suffix_gate_no_dspark`; `10/10` subtests PASS).
  - Bumped `VERSION`, `CMakeLists.txt`, `Dockerfile`, and `deploy/{deepseek-v41-affinity.service,run.sh}` image tag to `5.6.1`.

### Performance & Verification
- **4-Arm Cold-Start NUMA-Local Benchmark (`v5.6.1` `cfcbb18` `Arm 4` vs `v5.6.0` / `v5.4.0` FAST, `--default-effort 60`, `N0=100%`)**:
  - `step_c` 4-prompt DSpark (`temp=0.0`): **`23.59 tok/s`** mean (`79.35%` acc, **`3.605 tok/blk`** — matches pure DSpark `23.63`, **`+19.7%`** vs `v5.6.0` `19.70` and **`+3.9%`** vs `v5.4.0` FAST `22.70`; `logic_puzzle` **`22.93 tok/s`** / `77.5%`, `cpp_bug` **`23.36 tok/s`**).
  - `code_edit.txt` (`NGEN=256`): **`43.51 tok/s`** warm `tg2` (`99.6%` acc, **`8.00 tok/blk`** — **`+9.8%`** vs pure DSpark `39.62` and **`+4.8%`** vs `v5.6.0` `41.50`).
  - `p4k` `tg2`: **`18.23 tok/s`** (`+5.9%` vs `v5.6.0`); `l4k`: `pp1 647.8 / tg2 46.48` (`102.4 ms/blk`); `l16k`: `pp1 1,349.5 / tg2 46.48` (`99.8 ms/blk`, `+6.1%` vs `v5.6.0`); `l64k`: `pp1 1,452.5 / tg1 45.85 / tg2 46.81` (`99.8 ms/blk`, beats `v5.4.0` `46.73` and `v5.6.0` `44.47`).
- **Diagnostics & Audits**: Documented `sar -B` / `mincore` Engram page-cache confounder (`1.7%` residency under advisory `MADV_WILLNEED` causing `+15–22 ms/blk` during `kswapd`-evicted cold boots); verified `--kv-size 327680 --kv-commit 262144` remains optimal under `--default-effort 75`; **`11/11` CPU `ctest`**, **`15/15` `aff_gpu` & `18/18` `aff_format+engine` `-ffp-contract=off`**, **`673` `gfx1201` kernels (`0` scratch, `0` VGPR spill)**, `N0=47,804,160 (100.0%)`, `resident_experts=3,644`, `0` `hipError`, `kfd_release_s=3.314 s`, and `pcache` `0x733de762fee543e6`.

## [5.6.0] — 2026-11-22 — "Wave 9 FP8 E4M3 native conversions, +27 resident experts, NUMA SYS_NICE & effort 75"

Wave 9 production release (`6383b19..54692df`). Converts remaining hot dense, Engram, attention-output, all-reduce, and target LM head stages onto native `gfx1201` hardware FP8 E4M3 (`F1–F8`), unlocks NUMA Node 0 `host_pool` page pinning via `--cap-add SYS_NICE`, adopts `--default-effort 75` in deploy manifests, and expands live GPU-resident MoE capacity to **`3,644`** (`+27` vs `v5.5.0`, `+51` vs `v5.4.0`) while preserving `-ffp-contract=off` (`15/15` `aff_gpu` & `27/27` `aff_format`) and prefix-cache fingerprint `0x733de762fee543e6`.

### Added & Changed
- **Sparse Indexer `idx_wq_b` FP8 (`F1`, `baf61cd`)**: Enabled `AFF_IDX_WQB_FP8=1` (default ON) so `DenseGpu::ps_for` preserves preshuffled FP8 weights on `idx_wq_b` (`(cols % 128) == 0`) and dispatches `batch_quant_tm_fp8_hip` + `batch_gemm_fp8_hip` in `indexer_one` and `batch_indexer_q`.
- **Grouped Block-32 `attn_out_a` Direct FP8 Path (`F2`, `b1e65c4`, `baf61cd`)**: Added `batch_attn_out_prep_fp8_hip` (LDS F32->BF16 + inverse RoPE tail + per-32 UE8M0 quantization) and `batch_gemm_fp8_grouped_b32_hip` (`AFF_ATTN_OUT_A_FP8=1` default ON when `AFF_O_FP8_QDQ=1`), keeping `attn_out_a` preshuffled FP8 across `attn_out` and `batch_attn_out`.
- **DSpark `mtp_main_proj` FP8 Quantized Tap (`F3`, `b1e65c4`, `baf61cd`)**: Added `n_live` tail zeroing (`0x00` E4M3, `1.0f` scale) to `batch_quant_tm_fp8_hip(..., n_live)` and enabled `AFF_MTP_PROJ_FP8=1` (default ON) on `mtp_main_proj`, skipping `draft_transpose_pad_kernel` in `draft_mtp_proj`.
- **Packed Hardware E4M3 TP=2 All-Reduce (`F4`, `6383b19`)**: Replaced scalar FP8 loops in `tp_allreduce.hip` (`tp_ar_fp8_kernel`, `tp_ar_fp8_staged_kernel`) with clamped 4-byte `pack4_e4m3` (`v_cvt_pk_fp8_f32`) and packed `e4m3_decode_pk` while preserving ascending rank accumulation order.
- **Raw MXFP8 Engram H2D + On-GPU Requant (`F5`, `b1e65c4`, `baf61cd`)**: Added `EngramTables::gather_mxfp8` and `engram_mxfp8_requant_fp8_hip` (`11` VGPRs, `16` occupancy), wired via `AFF_ENGRAM_RAW_H2D=1` (default ON with `AFF_ENGRAM_FP8=1`) to eliminate CPU F32 dequantization before H2D.
- **Group-128 FP8 E4M3 Target LM Head (`F6`, `b1e65c4`, `baf61cd`, `54692df`)**: Added chunked `quantize_bf16_to_fp8_g128_hip` and `matvec_fp8_g128_multix_hip` (`320.6 MiB/shard` vs `483.6 MiB/shard` `bf16x12`), promoted `AFF_LMHEAD_FP8=1` (`lmhead_fp8_enabled()`) to default ON in `54692df` after passing all F6 gates (`+27` resident experts: `3,617 -> 3,644` live / `3,586 -> 3,613` static), and quieted `supersedes AFF_LMHEAD_BF16X12` log output unless `AFF_LMHEAD_BF16X12` is explicitly set.
- **Hardware `e4m3_enc` & Subnormal Carry Fix (`F7`, `6383b19`)**: Switched `device_math.h` `e4m3_enc` to hardware `v_cvt_pk_fp8_f32` (with clamped `e4m3_enc_sw` fallback) and fixed the `[7.5*2^-9, 8*2^-9)` subnormal carry `0x08` across `device_math.h`, `attention_gpu.hip`, and `attn_flash.hip`.
- **2-Wide Norm-Quant & Uniform `AFF_O_FP8_QDQ=1` (`F8`, `b1e65c4`, `baf61cd`)**: Paired 2-wide `cvt_pk_fp8_f32` emission in `batch_rms_norm_q8_kernel`, `hcn_norm_kernel`, and `hcn_norm_v41_kernel`, and unified `o_fp8_qdq_enabled()` (`AFF_O_FP8_QDQ=1` default ON) across both single-token `attn_out` and batched `batch_attn_out`.
- **NUMA Node 0 `SYS_NICE` & Effort 75 (`2abc33f`, `54692df`)**: Added `--cap-add SYS_NICE` in `deploy/{deepseek-v41-affinity.service,run.sh}` to unlock `SYS_mbind(MPOL_PREFERRED -> node 0)` (`rc=0`, pinning `47,804,160 / 47,804,160` = `100.0%` of `host_pool` on Node 0 while letting `Engram` spill safely to Node 1 at `0.00 pgscank/s`), bumped `VERSION=5.6.0` / `CMakeLists.txt` / `Dockerfile`, and defaulted `--default-effort 75` (DeepSeek V4.1 native default high reasoning).

### Performance & Verification
- **Kernel-Level `AFF_TRACE=dense,events` (`l4k`, 3-Boot Cold-Start FAST)**: `b_attn_out` (`F2`) drops `211.71 -> 158.40–159.55 us/call` (**`-25.2%`**), `b_mv_host` (`F6`) drops `77.29 -> 48.74–48.99 us/call` (**`-36.9%`**, host wait `119.89 -> 17.22 ms`, **`-86%`**), and `dense-gpu` total improves from `6.426 s` to **`6.097–6.135 s`**.
- **Paired Cold-Start NUMA-Local Benchmark (`v5.5.0` NUMA vs `v5.6.0-rc2` NUMA, `--default-effort 60`, ZERO `AFF_LMHEAD_FP8` override)**:
  - `p4k.txt` (`NGEN=16`): `pp2` `441.6 -> 447.1`, `tg2` `16.38 -> 17.21 tok/s` (**`+5.1%`**).
  - `l4k.txt` (`4,031` tok): `pp1` `622.1 -> 647.9` (**`+4.1%`**), `tg1` `24.51 -> 26.18` (**`+6.8%`**), `pp2` `597.1 -> 607.0`, `tg2` `44.17 -> 45.65 tok/s` (**`+3.4%`**).
  - `l16k.txt` (`16,104` tok): `pp1` `1,306.9 -> 1,346.3` (**`+3.0%`**), `tg1` `38.66 -> 41.78` (**`+8.1%`**), `pp2` `1,183.7 -> 1,210.1`, `tg2` `40.40 -> 43.81 tok/s` (**`+8.4%`**).
  - `l64k.txt` (`63,863` tok): `pp1` `1,404.9 -> 1,454.9` (**`+3.6%`**), `pp2` `2,608.1 -> 2,678.7` (**`+2.7%`**), `tg2` `39.67 -> 44.47 tok/s` (**`+12.1%`**).
  - `code_edit.txt` (`6,819` tok, `NGEN=256`): `tg2` `40.02 -> 41.50 tok/s` (**`+3.7%`**).
  - `step_c` 4-prompt DSpark (`tg_stream`): `19.31 -> 19.70 tok/s` mean (**`+2.0%`**).
  - `step_d` multi-turn (`80K -> 123K`): `turn4_100k_cached` `19.48 -> 22.74 tok/s` (**`+16.7%`**, TTFT `1.069 s -> 0.914 s`).
- **`--default-effort 75` Smoke (`eff75_smoke`, `N0=100%`)**: `req00=0.43 s`, math `\boxed{12:50}` PASS (`ttft 2.38 s`, `tg 25.43 tok/s`, `rtok=337`, `ctok=716`), `create_incident` JSON `tool_calls` PASS (`ttft 3.575 s`, `tg 16.40 tok/s`, `rtok=34`), `0` freeze/loop aborts, `0` `hipError`, `kfd_release_s=3.105 s`.
- **ELF, Tests & Gates (`v5.6.0-rc1` & `v5.6.0-rc2` `54692df`)**: **`673`** `gfx1201` kernels with **`0` static scratch** and **`0` VGPR spill** (`hcn_norm_v41_kernel<16u,{1,2,4,8}>` = `{41, 41, 92, 91}` VGPRs, `engram_mxfp8_requant_fp8_kernel` = `11` VGPRs / `16` occ; `8` unused `NH=8` SGPR->VGPR only), **`16/16` full `ctest`** (`11/11` CPU + `5/5` GPU including `test_gpu_ports` Tests 23–25 & `test_wave2_kernels` Tests 11–14), F3 `64/64` exact parity (`4.27 vs 4.27`), F5 `64/64` parity + `--max-layers 15` `129,280/129,280` `rmse=0.0`, F6 logits gate (`top1_match=True`, `top10=9/10`, `top64=61/64`, `cos_sim=0.999905`, `kl_ref_cmp=0.001601 <= 0.005`, `f6_server` `48.73 / 44.53 / 43.94`, `DRY 3/3`), F7 cross-version `pcache` restore (`3,584/4,026`, `7/7` blocks, `0` warnings, `fp=0x733de762fee543e6`), `test_a_stride` (`33,792/35,144`), `test_b_think_ckpt` (`743/743`), and `kfd_release_s=3.400 s`.

## [5.5.0] — 2026-11-22 — "Radiance Wave 8 ports: pure-LDS barriers, bf16x12 LM head, pre-collapsed AR & SuffixDecoding"

Wave 8 production release (`e9279b1..55de363`). Ports seven RDNA 4 (`gfx1201`) and speculative-decoding optimizations (`R1–R7`) from Radiance while keeping target model numerics bit-exact (`-ffp-contract=off`, prefix-cache fingerprint `0x733de762fee543e6` preserved) and expanding GPU-resident MoE slots to **`3,617`** (`+24` vs `v5.4.0`).

### Added & Changed
- **Pure-LDS Barriers & Store-Wait/Agent Fences (`R1`, `e9279b1`)**: Replaced global-draining `__syncthreads()` and heavy `__threadfence()` calls with `aff::lds_barrier()`, `s_wait_storecnt 0x0`, and agent-scoped acquire/release fences across `indexer_gpu.hip`, `batch_kernels.hip`, `digest.hip`, `dense_gpu.hip`, and `tp_allreduce.hip`.
- **Register-Form AR+HC Expand & Early Wake Store (`R3`, `e9279b1`)**: Enabled `AFF_AR_REG_FORM=1` (register buffering up to `12` `block` elements across peer sync) and `AFF_AR_WAKE_STORE=1` (early peer wake store at kernel entry), with `AFF_AR_FAST_REL=0` default.
- **Pre-Collapsed AR + V4.1 RMSNorm Fusion (`R4`, `e9279b1`, `2df7638`)**: Enabled `AFF_FUSE_AR_RMSNORM=1` (`HasPreCollapse` in `tp_ar_batch_hc_expand_kernel` for `nb=16` narrow batches + `batch_hc_norm_v41_only_hip` / `batch_hc_head_norm_only_hip`), skipping redundant `hcn_v41_collapse_sq_kernel` launches.
- **Lossless 12.25-bit `bf16x12` Target LM Head (`R5`, `e9279b1`, `2df7638`)**: Enabled `AFF_LMHEAD_BF16X12=1` (32-col K-block SoA `b12_sm`/`b12_exp4`/`b12_mexp` + host CSR escape table for `±0.0`/subnormals/`±Inf`/wide-span exponents) with bit-identical `matvec_bf16x12_multix_hip` and `batch_gemm_bf16x12_hip` kernels, growing resident MoE slots from `3,593` to **`3,617` (+24)**.
- **Incremental SuffixDecoding Drafter & Depth Stop (`R6`, `27d1402`)**: Added `src/engine/suffix_draft.h` (`AFF_DRAFT_SUFFIX=1`, `MIN_MATCH=4`, `MIN_SCORE=0.35`, `MAX=12`, `MIN_EXPECT=2.0`) ahead of `dspark_draft` for zero-GPU-cost speculative proposals (`<=15` tokens) plus `AFF_DRAFT_DEPTH_STOP=1`.
- **Opt-In Ablated Features Locked Off (`R2`, `R7`, `e9279b1`, `2df7638`)**: Kept `AFF_TP_AR_CODED=0` (`-54.8%` KL vs FP32 at `~2.5%` 64K prefill cost; opt-in via `AFF_TP_AR_CODED=1` or `AFF_TP_AR_MODE=coded`) and `AFF_PREFETCH_BESIDE=0` (`AFF_PREFETCH_MB=8`, `AFF_PREFETCH_BLOCKS=16`; `-15.5%..-33.2%` when enabled).

### Performance & Verification
- **Stage B2 Paired Benchmark (`v5.4.0` vs `v5.5.0-rc1`, `--default-effort 60`)**:
  - `l4k` warm `tg2`: `34.40 -> 43.16 tok/s` (**`+25.5%`**).
  - `l16k` warm `tg2`: `33.77 -> 44.55 tok/s` (**`+31.9%`**).
  - `l64k` warm `tg2`: `32.08 -> 43.70 tok/s` (**`+36.2%`**).
  - `code_edit` warm `tg2`: `28.15 -> 40.27 tok/s` (**`+43.1%`**).
  - **4-Prompt `temp=0.0` DSpark**: **`+0.4%` to `+15.3%`** across prompts.
  - **Multi-Turn `80K -> 123K`**: **`+45.9%`** on `turn3` and **`+43.6%`** on `turn4`.
- **ELF, Tests & Reliability**: **`599`** `gfx1201` kernels with **`0` scratch bytes** and **`0` VGPR spill** (`8` unused `NH=8, HasControl=true, LiveOne=true` `tp_ar_batch_hc_expand_kernel` instances spill SGPR->VGPR keeping `scratch=0`; V4.1 `NH=4` has `0` spill), **`11/11` CPU `ctest`** + **`5/5` GPU binaries** + fallback-OFF matrix + **`64/64` static-placement parity** PASS, **`DRY 3/3`**, **`</think>` checkpoint `653/653` restored**, **`64`-block (`33,792`-tok) stride** verified at `64K`/`80K`, KFD release **`3.298 s`**, prefix-cache fingerprint `0x733de762fee543e6` preserved.

## [5.4.0] — 2026-11-21 — "RDNA 4 gfx1201 queue/SE-mask stabilization, zero-D2H host pool & agentic serving optimizations"

Wave 7 production release (`9f44992..b2ab8d3`). Eliminates RDNA 4 (`gfx1201`) KFD multi-queue / multi-SE decode bimodal slowdown (`GPU_MAX_HW_QUEUES=1` + single-SE `0.25` contiguous CU mask), removes all placement D2H traffic (`AFF_POOL_FULL=1`), expands resident MoE slots to `3,563` clean (`3,593` live production with card-1 bonus slab), adds chunk-aligned stride & `</think>` prefix-cache checkpoints, hardens DRY/ThinkingBudget behavior, and speeds up MoE/dense/indexer/Engram kernels.

### Added & Changed
- **RDNA 4 (`gfx1201`) KFD Single-Queue + Single-SE Hybrid Stabilization (`dcac58a`, `765a5af`)**: Auto-sets `GPU_MAX_HW_QUEUES=1` on `gfx12xx` (`AFF_KEEP_HW_QUEUES=1` to override), fixes WGP-vs-CU bitmask sizing with `hipExtStreamGetCUMask` readback, and defaults the hybrid PCIe stream to single-SE 25% contiguous (`AFF_HYBRID_CU_FRAC=0.25`, `AFF_HYBRID_CU_STRIPE=0`, `0x000000ff` SE0-only, `10` KFD queues vs `16`). Eliminates the `~17–26 us` per-kernel ROCm/TheRock `#9297` wait penalty and multi-SE contention (`6/6` = `100%` FAST mode vs bimodal `v5.3.0`).
- **Zero-D2H Immutable Full Host Pool & +76 Resident Experts (`e202e76`, `521c4b5`)**: `AFF_POOL_FULL=1` pins all `15,360` experts across `182.4 GiB` of 2 MiB THP RAM, eliminating `136+ GiB` D2H writeback (`0.00 GiB` D2H) and freeing VRAM headroom to reach **`3,563` resident experts** cleanly (`+76` vs clean `v5.3.0`, `+46` vs `v5.3.0` prod; **`3,593`** live production).
- **MoE, Dense, Indexer, TP2 & Engram Kernel Upgrades (`9f44992`, `4760b1f`, `97e9f1d`, `da295eb`)**: Branchless `kHalfTR`, explicit `stage_a` prefetch, `s_tok` LDS hoist, wider scatter grid, pre-allocated split-K/compressor buffers, live-only `AR + hc_post` expand fusion (`out_hc` tail zeroing fix), LDS-padded `K_VECS_PADDED` indexer + narrow verify (`nb<=8`), flash DPP reduction, W8A8 Engram `wkv` (`AFF_ENGRAM_FP8=1`), and hardware FP8 QDQ (`AFF_HW_OQDQ=1`).
- **Agentic Prefix Cache, SSE & Single-Pass Tokenizer (`6fce4ab`, `dcac58a`, `521c4b5`, `765a5af`)**: Pinned async batched KV save/load (`AFF_PCACHE_BATCH_IO=1`), post-SSE prefix-cache publish (`AFF_PCACHE_POST_SSE=1`), `</think>` checkpoints (`AFF_PCACHE_THINK_CKPT=1`), chunk-aligned `64`-block (`33,792`-tok) stride checkpoints via `std::gcd(prefill_chunk, B)`, single-pass `preencoded_ids` chat encoding, and `AFF_PCACHE_PREFILL_CKPT=0` default.
- **DRY & ThinkingBudget Hardening (`ff50a23`, `521c4b5`, `b2ab8d3`)**: Gen-only DRY (`AFF_DRY_GEN_ONLY=1`), `eos`/`</think>`/`<｜*` exemptions, `vb.penalty_ids` exclusions, `</think>` DRY floor reset (`AFF_DRY_RESET_ON_THINK_CLOSE=1`), raised `dry_min_repeats=4` + `c.second >= req_c` with `2x` protection for prompt-verbatim `(ng+1)`-grams, `completion_tokens_details.reasoning_tokens` usage reporting, anti-thrash `ExpertHeat`, and multi-scale/novelty/full-vocab freeze breakers.

### Performance & Verification
- **Clean 2-Arm Paired Benchmark (`rc5` vs `rc5_arm1 v5.3.0` / FAST `rc4_arm1 v5.3.0`, `NGEN=192`)**:
  - `4K` (`4,031` tok): `pp1 = 624.0 tok/s` (`+3.28%` / `+2.94%`), `tg2 = 48.24 tok/s` (`+8.16%` / `+6.16%`).
  - `16K` (`16,104` tok): `pp1 = 1,300.5 tok/s` (`+7.37%` / `+7.26%`), `tg2 = 45.81 tok/s` (`+1.15%` / `+3.92%`).
  - `64K` (`63,863` tok): `pp1 = 1,409.0 tok/s` (`45.3 s`, `+4.83%` / `+4.27%`), `tg2 = 46.73 tok/s` (`+17.86%` / `+10.13%`), `pp2 = 2,593.2 tok/s` (`24.6 s`).
  - **4-Prompt `temp=0.0` DSpark**: **`22.70 tok/s`** mean stream (`+15.17%` / `+9.08%`).
  - **Multi-Turn `80K -> 123K`**: Cold `80K` TTFT `77.696 s` (`-4.38%` / `-4.79%`), `123K` extend TTFT `37.031 s` (`-4.59%` / `-2.89%`), cached turns resume in `0.949 s` and `0.901 s`.
- **`rc6` Verification (`b2ab8d3`)**: `l4k` `tg2 = 46.24 tok/s` (`pp1 = 618.9`), `14/14` CPU `ctest`, `5/5` GPU binaries (`0` `hipError`), `539` `gfx1201` ELF kernels with `0` scratch bytes and `0` VGPR spill (`-ffp-contract=off` preserved), **`3/3` functional tests PASS** (`test_a_stride`, `test_b_think_ckpt`, and **`3/3` consecutive `test_c_dry_reasoning`**), prefix-cache fingerprint `0x733de762fee543e6` preserved.

## [5.3.0] — 2026-11-19 — "RDNA 4 Wave32 HIP kernel optimizations"

Wave 6 production release. Optimizes `gfx1201` Wave32 HIP kernels across Wave A (`dfb34cb`), Wave B (`c20ed32`), and multi-GEMM band / WGP fence fixes (`e1f6be3`), retaining ROCm `10.1.0-full` (`+3.55%` composite over ROCm `10.2.0-full`).

### Added
- **Wave A RDNA4 Primitives (`src/gpu/hip_common.h`, `dfb34cb`)**: `lds_barrier()`, `dpp_xor_lane<Lane>()` (`DPP16 row_xmask` + `v_permlanex16_b32`), and `AFF_WAVES(n)` occupancy hints across GPU kernels.
- **Wave B Kernel Tuning (`c20ed32`)**: FP8 4×2 prefill (`AFF_FP8_PREFILL_4X2`), inline split-K (`AFF_FP8_SPLIT_INLINE`), staged TP2 all-reduce (`AFF_TP_AR_STAGED`), `AFF_FLASH_MIN_TILES`, and VGPR/LDS tightening (`flash_kernel<2>` `235` VGPR / `21,392` B LDS / `occ=6`, `idx_quant_kernel` `scratch=0`, `tp_ar_fp8_staged_kernel` `scratch=0`, `0` new spills across `497` `gfx1201` kernels).
- **Multi-GEMM & WGP Fence Fix (`e1f6be3`)**: Contiguous output bands relative to `jobs[0].y` when `jobs[i].y == nullptr` (`batch_gemm_multi_hip`, `batch_gemm_fp8_multi_hip`), and `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` in `gemm_fp8_body`.

### Performance & Verification
- **Paired 3-Arm Benchmark (`NGEN=192`)**: `+24.9%` to `+43.8%` `tg` (`41.47–42.23 tok/s` @16K–64K), `+37.05%` slot `blocks/s`, `+34.9%` 4-prompt DSpark (`21.33 tok/s`), `-34.0%` `dense-gpu`, `-85.2%` `x_gu+x_down`, `resident_experts=3,517`, `free_vram_mib=[353, 355]`, `kfd_release_s=2.800 s`.
- **Tests**: 13/13 filtered CPU `ctest` + 5/5 GPU binaries (ROCm 10.1 & 10.2) + 2/2 fallback-toggle GPU runs PASS (`0` `hipError`).

## [5.2.0] — 2026-11-19 — "VRAM reclaim, fused attn_out prep & host/SSE/tokenizer pipeline"

Wave 5 production release (`44e81ee`, `4ada6fa`). Reclaimed dead V4.1 Q-RMS/compressor allocations and tiled Engram H2D (`-616 MiB/card`) while accelerating dense/SSE/tokenizer paths.

### Added & Changed
- **VRAM Reclaim & Resident Capacity (`b13cbf2`)**: Expanded GPU-resident MoE slots by **`+101`** (`3,416 -> 3,517` experts, `[355, 354] MiB` free VRAM).
- **HIP Decode & Prefill Fusions (`3f24d07`, `44e81ee`)**: Added `AFF_FUSED_ATTN_OUT_PREP`, `kFKNarrow=128` (`AFF_FP8_NARROW128`), `AFF_REUSE_NORM_Q8`, single-drain draft confidence, threadpool skip on decode Engram (`live < 64`), and prefill-before-decode synchronization.
- **Host, SSE & Tokenizer Optimizations (`45ea4eb`, `f424cfd`, `54c5d78`, `d24e31d`, `60676eb`)**: Client `fd_alive` disconnect cancellation (`1.12 s`, `0` wasted tokens), 15 s empty-delta SSE keepalives, lock-free `/health`, skipped prefill logits, O(1) `decode_one` + BPE/LCP caching, hardened O(1) delta prefix-cache lookup, and 9/9 CPU `ctest` + 4/4 GPU tests PASS.

### Performance
- Warm Pass-2 decode (`tg2`) **41.00–41.72 tok/s** (`+30.2%` to `+56.1%` over `v5.1.2`); cached multi-turn TTFT `-47.0% @64K` (`2.049 s -> 1.086 s`) and `-55.9% @100K` (`2.386 s -> 1.052 s`).

## [5.1.2] — 2026-11-17 — "ThinkingBudget, Executive-Freeze breaker & cache-preserving effort=none"

Wave 4C production release (`c52e95e`, `8025372`).
- Ported Lucebox `ThinkingBudget` (`AFF_THINK_REPLY_RESERVE=4096`, `AFF_THINK_LOOP_REPEATS=3` over `loop_span=64`), added `AFF_THINK_FREEZE_BREAKER=1`, synchronized DSpark speculative rollback across forced `</think>` injection, and added cache-preserving `reasoning_effort: "none"` (`suppress_final_think`); 7/7 CPU unit tests PASS.

## [5.1.1] — 2026-11-17 — "TP=2 delta prefix-cache restore offset & UTF-8 streaming hotfix"

Wave 4B hotfix (`63cd104`, `be545d6`).
- Fixed `PrefixCache::restore()` multi-region/TP=2 rank offset indexing across `app_` regions (no `pcache` wipe required) and added strict UTF-8 streaming holdback/sanitization and validating `json_escape`; 6/6 CPU unit tests PASS.

## [5.1.0] — 2026-11-16 — "DSpark masked tile skip, state-only delta pcache & DSML parser hardening"

Wave 4A production release (`839665f`, `e559b1b`).
- Added `AFF_DRAFT_SKIP_MASKED=1` (`+22.2%` decode: `16.93 -> 20.69 tok/s`, **`73.33%`** MoonMath-1017 DSpark acceptance vs `67.52%`), `AFF_PCACHE_DELTA=1` (`1.42x` TTFT), `ignore_eos=true`, `gridDim.y > 65535` guard (`#1006`), DSML nested XML / multi-invoke splitting (`#1020`), and MoE RAW barrier validation (`#983`, `#984`); retained `AFF_ATTN_LOOKAHEAD=0`; 6/6 CPU tests PASS.

## [5.0.0] — 2026-11-16 — "RDNA 4 sparse indexer & 3,416 resident experts"

Wave 3 production release. Implements hardware-specialized sparse indexer on RDNA 4 (`gfx1201`),
dramatically accelerating long-context prefill while expanding GPU resident MoE capacity.

### Added
- **RDNA 4 Sparse Indexer (`AFF_SPARSE_INDEXER=1`, `index_topk=512`)**:
  - `K_VECS_PADDED=136` padded key vectors with 8-byte query staging.
  - 4-pass 256-bucket multi-CTA radix sort kernel for top-512 candidate pruning.
  - Wave32 tie compaction, live-`seqused` tracking, and ragged causal prefill handling.
  - Bounded 36 MiB score scratch buffer (`AFF_INDEXER_SCORE_BUDGET_MB=36`, `<= 48 MiB/card`).
- **Resident Capacity Expansion**: Scratch memory reductions free critical VRAM, expanding resident
  MoE expert slots from 3,367 to **3,416 experts (+49 shards)** with `[356, 354] MiB` free VRAM.
- **Long-Context Speedups**:
  - 4K: TTFT 7.7s -> 3.2s (1,186.1 tok/s, 2.39×).
  - 16K: TTFT 12.9s -> 3.6s (4,220.3 tok/s, 3.60×).
  - 64K: TTFT 40.6s -> 2.8s (21,448.9 tok/s, **14.40×**).
  - 128K: TTFT 56.6s -> 4.3s (28,092.6 tok/s, **13.15×**).
  - Cold uncached prefill: 963.66 tok/s @32K, 990.98 tok/s @64K (25.18 tok/s decode post-64K).
- **Verification**:
  - Needle-in-a-Haystack: 4/4 (100%) retrieval across 4K, 16K, 64K, 128K.
  - `refattn.py` cosine match: 99.98% parity against dense attention.
  - KL divergence: 0.000000 nats (PPL 2.6057).
  - `quant_eval`: 92.50% accuracy (111/120 items).
  - Test suite: 9/9 PASS.

## [4.5.0] — 2026-11-16 — "fused and chained hybrid compute"

Wave 2 production release. Sweeps 9 optimization arms and deploys fused narrow hyper-connection and
chained hybrid compute kernels.

### Added
- **Fused & Chained HC (`AFF_HC_FUSED_V2=1`, `AFF_HC_CHAINED=1`, `AFF_HC_CHAINED_MIN_ROWS=1`)**:
  - Decode throughput median **23.73 tok/s** (peak 24.37 tok/s).
  - Batch-8 aggregate throughput **22.4 tok/s**.
  - Zero numerical degradation: 8,184-tok KL divergence is 0.000000 nats (PPL 2.6057).
  - `quant_eval`: 90.83% accuracy (109/120 items).
  - Test suite: 8/8 PASS.

### Investigated and Rejected
- `AFF_EMBED_HOST=1`: Produced 19.94 tok/s decode (-16%). Physical VRAM ledger confirmed `token_embd`
  is already host-mmap'd at 0 MiB VRAM; setting the flag added host synchronization overhead with zero
  memory benefit. Flag remains 0.
- `AFF_MOE_SUMMED_DOWN=1`: Added reduction overhead on PCIe 3.0 x16 without speed gain; kept 0.

## [4.4.0] — 2026-11-15 — "ROCm 10.1 & Cactus repetition escape"

Wave 1 production release. Modernizes compiler toolchain and cures agentic reasoning loops.

### Added
- **ROCm 10.1 Toolchain**:
  - Decode median 20.41 tok/s (+10.4% over ROCm 7.14 18.49 tok/s).
  - Greedy decode 18.96 tok/s, Batch-8 20.9 tok/s (+16.1%).
- **Cactus Escape Mixture (`AFF_CACTUS_ESCAPE=0.25`)**:
  - Resolves repetition traps in reasoning loops: **0/20 repetition lock-in** at `temp=0.6`.
- **Cache & Sharding Fixes**:
  - 64 GiB multi-slot persistent NVMe prefix cache (`--prefix-cache-dir .../pcache`).
  - Automatic `heat -> static` placement allocation fallback (Issue #12).
  - `Jubiloso` cross-shard scale index lookup fix.
  - Bit-identical greedy reproducibility restored.
  - Radiance typed-argument DSML tag parity (`9cf8d3d`).
  - `quant_eval`: 88.33% accuracy (106/120 items); 7/7 tests PASS.

## [4.3.3] — 2026-11-15 — "branch unification & observability"

Wave 0 production release. Unifies 3 divergent branch histories into single clean master tree.

### Added
- Observability: `GET /health` JSON health endpoint returning version, git SHA, and free VRAM.
- Passthrough: `--raw-completions` pass-through flag.
- Accounting: `prompt_tokens_details.cached_tokens` reporting.
- Test suite: 7/7 PASS with explicit test assertions enabled.

## [4.3.2] — 2026-11-03 — "benchmarks in the README"

No engine changes. README gains the v4.3 Arm 1 A/B against the v4.2.0 baseline: decode median
19.27 vs 16.33 tok/s (+18.0%), greedy decode 14.48 vs 12.07 (+20.0%), prefill 2048 680.1 vs
558.4 tok/s (+21.8%), batch-8 21.4 vs 10.9 tok/s (+96.3%), live 74.8k agent workload 21.0 tok/s, plus the
VRAM reclamation numbers (3,279 symmetric experts/card, +88 GPU1 bonus shards). Documents the
decode-gated MoE recon tile cap, Radiance LDS scale prefetch, `AFF_V41_HC_FUSED`,
`AFF_PREFILL_FREEZE_PLC`, `AFF_BONUS_SLAB` and the rejected `AFF_LMHEAD_INT4` arm (acceptance
66.4% → 53.5%).

## [4.3.1] — 2026-11-03 — "docs match the release"

No engine changes. README now points at the v4.3.0/v4.3.1 releases and documents the 4.2.0/4.3.0
work; `deploy/run.sh` and `deploy/deepseek-v41-affinity.service` now match the served container
(`--kv-size 327680`, `--gpu-headroom-mib 384`, placement guards 1.6/3.0,
`AFF_MOE_RECON_TILES=32`, `AFF_PREFETCH_RING=0`, persistent prefix cache). Build and test results
are unchanged from 4.3.0.

## [4.3.0] — 2026-11-02 — "carry the upstream ports"

Merges `main` (v1.1.0 Radiance/KTransformers ports) into the 4.2.0 GPU line. The two branches
diverged on 2026-10-12. Thirteen conflict hunks across five files were resolved in favour of this
line's implementations where both sides solved the same problem:

- **Prefill chunking** keeps `AFF_EQUAL_UBATCH`, bounded replay and the `limit`-based vision cuts;
  upstream's simpler `chunk` loop is not used.
- **LM-head INT4** keeps `launch_w4_head` over the shared `matvec_w4_multix_hip`; upstream's
  `AFF_LMHEAD_INT4` / `AFF_DRAFT_HEAD_INT4` switches remain honoured. The duplicate upstream
  declaration in `dense_gpu.hip` was removed.
- **Sampler history** keeps the device `smp_hist_` / `smp_dry_` buffers used by this line's
  repetition-penalty and DRY paths.
- **Prefetch ring teardown** frees `d_pref_slab_` / `d_pref_src_` per device.
- **CHANGELOG** keeps this file, which already contained the v1.1.0 and v1.0.0 sections.

Upstream changes that did not conflict (protected-expert residency, FP8 grouped RoPE, fused HC,
stage rings, deployment docs, licences) are carried over unchanged.

Build: `cmake --build` clean with ROCm 7.x / `gfx1201`; `ctest` 7/7 passed.

## [4.2.0] — 2026-11-01 — "measure the reserve"

The theme of this release is that the engine's *accounting* of VRAM was wrong in one direction and
its *placement churn* was higher than it needed to be. Both were found by instrumenting a serving
process, which previously published nothing at all.

### Added

- **Driver-counter VRAM sizing** (`hip_free_vram_checked`, `src/gpu/hip_common.h`). Expert-slab
  sizing now reads `mem_info_vram_total` / `mem_info_vram_used` from
  `/sys/bus/pci/devices/<bdf>/` via `hipDeviceGetPCIBusId`, and sizes from
  `min(hipMemGetInfo, sysfs)`. HIP remains the fallback *and* the floor. The sizing banner prints
  both figures so the gap is visible on every start:
  `placement sizing: card 0 free 20245 MiB (HIP 20862 MiB) ... -> 3130 slots`.

  Why: on this box `hipMemGetInfo` reports ~350 MiB a card MORE free VRAM than the driver does, with
  nothing else running. The slab was sized from the optimistic number, so the reserve that the KV
  grower is supposed to spend was partly imaginary. Measured before the change: a 479,603-token
  request left **66 MiB** free on GPU0; after it, **920 MiB**.

- **Placement telemetry on a timer** (`maybe_save_placement`, `tools/affinity.cpp`). A serving
  process now logs, once per `--placement-save-secs`:
  `placement: N ticks, P promotions / D demotions, R readmits (X%), F refused, starved S slab / P pool; GiB h2d, GiB d2h`.
  `engine_report()` only ever fired on CLI exits, so HTTP serving runs left no counters at all and
  a throughput regression had no explanation attached to it.

- **`AFF_EQUAL_UBATCH`** — split each prefill segment into as few equal ubatches as the chunk cap
  allows, instead of cap-sized ones with a remainder. **Default off**; measured negative here (see
  Rejected) and kept only so the experiment is reproducible. The refactor underneath it (segment
  limits computed before the cap, so bounded replay and vision spans cannot be crossed) is
  behaviour-preserving and always active.

- **Repeat / DRY penalties in the sampler** (carried in this release, previously uncommitted):
  `--repeat-penalty`, `--repeat-last-n` (device-side `repeat_penalty_kernel`) and sequence-level DRY
  (`--dry-delta`, `--dry-ngram`, `--dry-window`, `--dry-min-repeats`, `dry_penalty_kernel`), with
  `AFF_DRY_DEBUG=1` gating a diagnostic log. Added after an agent session replayed its own
  `reasoning_content` and compounded 1,162 `"Hmm —"` tics across 190 messages.

- **Cactus acceptance governor** (carried): `AFF_CACTUS_DELTA`, `AFF_CACTUS_CAP_R`,
  `AFF_CACTUS_FADE_START`, `AFF_CACTUS_FADE_END` bound what the draft may accept when the target is
  drifting, instead of letting a confident-but-wrong draft set the block length.

- **INT4 LM head** (carried): `AFF_DRAFT_HEAD_INT4` / `AFF_LMHEAD_INT4` quantise the output plane to
  `int4_g128` at load, in place of a BF16 copy.

- Tests: `tests/test_cactus_accept.cpp`, `tests/test_tool_namespace.cpp`, `tests/test_o_fp8_qdq.cpp`,
  `tests/test_w4_head.cpp`, `tests/test_gpu_ports.hip`, `tests/test_prefetch_ring.hip`.

### Changed

- **Placement guards, 1.25 / 1.5 → 1.6 / 3.0** (`--placement-hysteresis`, `--placement-min-gain`).
  Operator setting, not a code default: it is what the reference service unit runs, and the numbers
  are in this changelog rather than baked into `expert_heat.h`.
  Effect at equal tick counts: promotions 9,533 → 4,422, readmits 71 % → 53 %, H2D churn
  113 → 52 GiB, starved slab/pool 88/105 → 28/44. Throughput unchanged inside noise. The link budget
  this frees is the one the hybrid miss tier streams on.

- **`--kv-size` 524288 → 327680** in the reference unit. KV commit is monotonic (`decommit_to` is
  called only by the destructor), so reaching 512k positions spends 662 MiB permanently. With the
  reserve now honest that is affordable, but the peak *is* the whole reserve: 327,680 caps the
  standing loss at ~0.17 GiB while still covering the 307k-token prompts this box actually sees.
  See `docs/plans/2026-11-01-kv-shrink-scope.md` for the shrink path and why it is deferred.

- `--prefill-chunk-size` stays at 2816 (the 1408 A/B costs 20–30 % prefill to save ~50 MiB).

### Fixed

- Expert slab over-committing VRAM by the amount the KV grower is meant to spend (see Added).
- A serving process having no placement counters (see Added).
- `Model::forward_prefill`'s chunk boundaries are now derived from segment limits, so a vision span
  or the bounded-replay split cannot be crossed by a chunk cap. Previously correct by ordering;
  now correct by construction.

### Investigated and rejected

Each of these has measurements behind it, recorded in `./benchmarks/ab/`.

| change | result | verdict |
|---|---|---|
| `--gpu-headroom-mib` 1024 → 512 | +84 experts (+2.6 % residency), **zero** throughput, 420 MiB of KV growth then lands 96 MiB from the edge | reverted |
| equal-size ubatches | −5 % / −3 % / +0 % prefill. `ceil(seg/chunk)` is unchanged by the split, so the chunk *count* is identical, and this engine streams expert **tiles** (bytes consumed), not whole experts per ubatch — a short tail chunk never paid a full sweep | default off |
| `AFF_PREFETCH_RING` 8 → 12 | −4 % / +0 % / +2.5 % prefill, and it costs 8 expert slots | back to 8 |
| Hosted MoE (CPU expert arm) | built, validated, removed in an earlier cycle: the CPU decodes an expert several times slower than the link reads it, and it reintroduced per-token non-determinism | not rebuilt |
| whole-expert `up\|gate\|down` slab packing | already the layout: `.aff` stores the three roles contiguous per expert, `pool_tf` packs one 6.08 MiB shard per card, and a promotion is already one `hipMemcpyAsync` per card | nothing to do |
| `decommit_to` on new sequence | scoped only; saves 662 MiB of slack but **cannot** buy residency (peak still bounds the reserve) and unmaps GPU memory under a live engine | deferred |

### Measurements (2026-11-01)

Harness: `aff_bench.py` (2 warm + 5 decode passes, median of the last 3; prefill at 3,072 / 12,204 /
49,013 real prompt tokens; batch-8). Noise floor across arms: decode ±13 %, prefill ±4 %.

| arm | decode | prefill 3k / 12k / 49k | residency | churn |
|---|--:|--|--:|--|
| baseline | 14.86 | 659 / 1131 / 1562 | 3231 | readmits 71 % |
| guards 1.6 / 3.0 | 16.3–19.4 | 669 / 1139 / 1546 | 3231 | readmits 53 %, H2D 52 GiB |
| + driver-counter sizing | 18.8–19.1 | 685–692 / 1144–1211 / 1564–1618 | 3130 | — |

KV profile (`vram_watch.py`, 0.5 s sampling):

| positions | free GPU0 / GPU1 after |
|---|--:|
| 49,102 | 606 / 860 (unchanged) |
| 196,611 | 606 / 860 (unchanged) |
| 307,195 | 259 / 782 |
| 479,603 | 66 / 560 |

KV is backed to `--kv-commit` at load (0.66 GiB for 262,144 positions, 350 MiB/card) and grows at
0.66 GiB per 262,144 positions, linear. Short and medium prompts move nothing.

### Inspiration

The VRAM-accounting and admission-guard work was prompted by **[Avalon](https://github.com/freriadoc/Avalon)**
(`freriadoc/Avalon`, a llama.cpp fork), which runs Qwen3.8-Flash-Next on one 20 GB card by caching
whole expert tensors in otherwise-unused VRAM. What transferred, and what did not:

- **Taken — size the cache from the driver, not from the runtime.**
  `ggml/src/ggml-cuda/moe-cache.cu` sizes its expert cache from
  `/sys/bus/pci/devices/<bdf>/mem_info_vram_{total,used}` with the comment that `hipMemGetInfo`
  "underestimates" on a 20 GiB card. Here the error goes the other way and is ~350 MiB/card, but the
  lesson is the same: the runtime's free count is not the card's free count, and the component that
  pays for the difference is the one that grows later. See `hip_free_vram_checked`.

- **Taken — treat admission/eviction guards as the whole policy.** Avalon's `admit_after`,
  `readmit_after`, `lfu_window` and its "is the churn progress or thrash?" framing match this
  engine's `expert_heat.h` note that a swap must repay ~1.4 future hits. Reading `readmits` as a
  first-class metric is why the 71 % was visible at all; the guards were then raised until it halved.

- **Rejected — hosted MoE, whole-expert slabs, equal ubatches, TAIL SKIP, PREGATE.** The first two
  are already solved differently here (per-shard residency, contiguous expert records); equal
  ubatches do not apply because the streaming cost is per tile consumed; TAIL SKIP drops experts
  (quality); PREGATE is a measured negative upstream. Avalon's own R9700 report (issue #10) also
  warns that RDNA3 tuning does not transfer, which matched what we saw with prefetch depth.

Secondary input: the [Unsloth DeepSeek-V4.1-Flash-1017 GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
tree confirmed the local DSpark drafter is the same upstream `mtp.0..2` weights (norms bit-identical),
so no draft upgrade was available from it — and it quantified the Engram table cost
(MXFP8 202.8 GB vs IQ4_NL 110.6 GB), which remains the largest single memory item on this box.

### Upgrade notes

- **Prefix cache is invalidated once.** `kMaxKvPositions` is part of the store fingerprint
  (`prefix_cache.cpp`), so changing `--kv-size` drops the index and rebuilds it. It is dropped, not
  misread.
- **Restart is required**; the binary is bind-mounted at
  `/opt/affinity/bin/affinity` and reload takes ~95–170 s for the 191 GiB container.
- Residency on this box moves 3130–3231 experts between boots because the free-VRAM figure at load
  depends on what else touched the cards; the banner now reports both numbers so the variance is
  attributable.
- Recommended service environment (the reference unit):
  `AFF_PREFETCH_RING=8`, `AFF_ADAPTIVE_WIDTH=1`, `AFF_BOUNDED_REPLAY=1`, `AFF_FUSE_HC_TAP=1`,
  `AFF_DRAFT_HEAD_INT4=1`, `AFF_CACTUS_DELTA=0.35`, `AFF_CACTUS_CAP_R=0.5`, `AFF_RESIDENT_BIAS=0.008`,
  `AFF_MOE_RECON_TILES=32`, `AFF_NOBUDGET_MAX_TOKENS=16384`, plus
  `--placement-strategy demand --placement-hysteresis 1.6 --placement-min-gain 3.0`,
  `--gpu-headroom-mib 1024`, `--kv-size 327680 --kv-commit 262144`, `--prefill-chunk-size 2816`.
