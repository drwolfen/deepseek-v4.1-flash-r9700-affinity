# Changelog

All notable changes to this fork. Format follows [Keep a Changelog](https://keepachangelog.com/),
versions follow [Semantic Versioning](https://semver.org/).

The engine serves DeepSeek-V4.1-Flash-1017 (`.aff`, 2.875 bpw FP4 experts, DSpark draft, Engram)
on 2× Radeon AI PRO R9700 (`gfx1201`). Numbers below are from that box unless stated.

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
  `AFF_DRY_DEBUG=1` gating a diagnostic log. Added after a Hermes session replayed its own
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

Each of these has measurements behind it, recorded in `/mnt/fast-storage/benchmarks/ab/`.

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
  `/home/ydj/affinity-patched/affinity` and reload takes ~95–170 s for the 191 GiB container.
- Residency on this box moves 3130–3231 experts between boots because the free-VRAM figure at load
  depends on what else touched the cards; the banner now reports both numbers so the variance is
  attributable.
- Recommended service environment (the reference unit):
  `AFF_PREFETCH_RING=8`, `AFF_ADAPTIVE_WIDTH=1`, `AFF_BOUNDED_REPLAY=1`, `AFF_FUSE_HC_TAP=1`,
  `AFF_DRAFT_HEAD_INT4=1`, `AFF_CACTUS_DELTA=0.35`, `AFF_CACTUS_CAP_R=0.5`, `AFF_RESIDENT_BIAS=0.008`,
  `AFF_MOE_RECON_TILES=32`, `AFF_NOBUDGET_MAX_TOKENS=16384`, plus
  `--placement-strategy demand --placement-hysteresis 1.6 --placement-min-gain 3.0`,
  `--gpu-headroom-mib 1024`, `--kv-size 327680 --kv-commit 262144`, `--prefill-chunk-size 2816`.
