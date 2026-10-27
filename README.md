# deepseek-v4.1-flash-r9700-affinity

A working setup for serving **DeepSeek-V4.1-Flash** (the uncensored / abliterated build) on **two AMD
Radeon AI PRO R9700s**, using a fork of the [affinity](https://codeberg.org/StillDeadcode/affinity)
inference engine, plus a locally quantised 2.875-bpw container of the model.

This repository is the *setup*: the engine source (fork + our patches), the deployment unit, the
quantisation recipe, and the measurements, all captured on one machine. It exists so the exact
combination that produced the numbers below can be rebuilt.

The affinity engine was transformed into this **R9700-tweaked version with the help of
DeepSeek-V4.1-Flash and Gemini 4 Argon** (used to port, patch and tune the engine — see *Rework and
patches*).

> **Tested on exactly two R9700s.** Every number here was measured on a two-card box. The engine
> upstream supports one card, but that configuration was not benchmarked for this setup.

**Model files (weights):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw)

**Changelog & Benchmarks:** [**`CHANGELOG.md`**](CHANGELOG.md) | **Latest Release:** [**`v1.1.0`**](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/releases/tag/v1.1.0)

---

## Hardware (the machine the numbers come from)

| | |
|---|---|
| GPU | **2 × AMD Radeon AI PRO R9700**, `gfx1201` (RDNA4), 32 GiB each |
| CPU | Intel Xeon Platinum 8160 (Skylake-SP), 2 sockets × 24 cores / 48 threads = 96 threads |
| RAM | 754 GiB DDR4 (2 NUMA nodes) |
| PCIe | GPUs link at **Gen5 ×16**, but the switch **uplink runs Gen3 ×16 (8.0 GT/s)** — the host Xeon is a PCIe-3.0 CPU. See *Performance notes*. |
| Host OS | **Ubuntu 24.04.5 LTS** (Noble Numbat), kernel 7.0.0, ROCm 7.2.4 |
| Engine build | AMD TheRock ROCm 7.14 image (no host ROCm needed) |

### Minimum RAM

The routed experts do not fit in 64 GiB of VRAM. At 2.875 bpw the routed experts are ~93 GiB and the
whole container is **191.32 GiB**; the two cards hold ~41 GiB of expert slab, so the rest lives in host
RAM.

- **~160 GiB RAM** is the practical floor for this configuration (512k context, DSpark, engram): the
  host expert pool alone is **143.5 GiB** plus the OS and page cache.
- Below that, experts spill to the SSD tier and decode collapses (the engine's own note: a 32 GiB card
  with 64 GB RAM runs ~8 tok/s instead of ~19+ because most experts end up on disk).
- At 512k context the KV cache is committed at 262144 backed positions (0.66 GiB VRAM); the compressed
  indexer/prefix cache is on disk.

---

## Current speed (measured, this machine - v1.1.0 update)

Single stream, 512k-context configuration, DSpark speculative decoding on, thinking on (measured on live Hermes agent sessions at **163.5k–170.7k context**):

| | |
|---|---|
| **Decode, weighted average (163k–171k ctx)** | **21.96 tok/s** (**+28.6%** over v1.0.0 baseline of 17.08 tok/s) |
| **Decode, peak sustained bursts** | **27.80 tok/s** (e.g. 1,341 tokens in 48.15s at 166.3k context) |
| **Speculative step latency** | **166.7 ms/block median** (-20.4%), down to **156.8 ms/block** min |
| **Draft acceptance** | **79.9%** (3.69 tokens/block avg, up to 4.37 tokens/block; tail collapse resolved) |
| **Prefill** | ~1000 tok/s (persistent prefix cache restores 330+ blocks from disk on restarts) |

Decode remains PCIe-streaming bound for non-resident experts (3,248 in VRAM / 12,112 streamed over PCIe), but **fused `hc_post` + DSpark taps (`AFF_FUSE_HC_TAP=1`)**, **on-GPU group-128 INT4 LM head (`AFF_DRAFT_HEAD_INT4=1`)**, and **Cactus capped rejection sampling (`AFF_CACTUS_DELTA=0.35`, `AFF_CACTUS_CAP_R=0.5`)** have cut per-block overhead by **-42.8 ms/block** and lifted decode throughput by **+28.6%**.

---

## What comes from where

Nothing here is the model author's or the engine author's endorsement; this is a personal fork.

| component | origin | license / note |
|---|---|---|
| Inference engine | [StillDeadcode/affinity](https://codeberg.org/StillDeadcode/affinity) by Yoshi Exeler | MIT. Upstream of everything. |
| Engine fork (this tree) | `neuromaniacMD/affinity` (our GitHub) | MIT. Adds the rework listed below. Kept as provenance; this repo is the deployable copy. |
| Optimization ports | [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) · [lucebox.com](https://www.lucebox.com/) — **Lucebox AI**, Apache-2.0 | Three of the local knobs are **ports from Lucebox's `ds41-gorgon` work**: bounded replay, adaptive verify width (`AdaptiveSpecWidth`) and the protected massive/outlier expert set; the port's quality gate also uses Lucebox's 8,184-token fineweb-edu eval corpus. |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | DeepSeek license. Source of the reference `inference/` and `encoding/` code. |
| Uncensored weights (`-uc`) | [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8) | FP8 checkpoint that we re-quantised to 2.875 bpw. |
| Abliterated weights (alt) | [pqhaz/DeepSeek-V4.1-Flash-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-heretic) | Directional ablation of refusal behaviour; lower KL than `-uc`. A `model-heretic.aff` was built but `-uc` is what is served. |
| GGUF / imatrix | [Adolphsson/DeepSeek-V4.1-Flash-UNCENSORED-GGUF](https://huggingface.co/Adolphsson/DeepSeek-V4.1-Flash-UNCENSORED-GGUF) | Import-matrix source for the 2.875-bpw encode. |
| Quantiser | affinity's own `aff-quantize` | Routed experts only, 2.875 bpw codebook quant, imatrix-weighted. |

The engine itself, its design, the kernels and the base README are upstream's work and remain MIT
licensed (`LICENSE`, `FORK.md`, `UPSTREAM-README.md` kept unchanged).

---

## Rework and patches (with measured effect)

> **Every commit below is traceable.** Click the short SHA to open the full diff on GitHub. All of the
> local work lives in this repository's history, so you can read the actual code change behind each
> claim — the vision port, the KV sizing, the tool-call recovery, the imatrix calibration, the
> `AFF_NOBUDGET` patch.

Branches: this repository's history is the fork's `v41-tool-fix` line (mirroring upstream `a48b72e`
plus the V4.1 port) with ten further commits of **unpublished local work** that produced the served
binary, plus one commit capturing the on-GPU imatrix/placement working tree and one patch commit.

### ⭐ Local work — the part that is ours

These are the changes that are not in the fork's published branches. This is where the speed and the
agentic-robustness come from.

| diff | change | measured effect |
|---|---|---|
| [`91e78ba`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/91e78ba) | **DeepSeek ViT on the CPU**, wired into prefill and the chat endpoint. Three reference discrepancies fixed (2D-RoPE half rotation, channel-major aligner, aspect-preserving pad) | embedding cosine vs reference 0.31 → 0.85; end-to-end image grounding works |
| [`cbb3f07`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cbb3f07) | ViT **weight-stationary GEMM + AVX-512 attention** | 546×546: 13.2 s → 3.25 s; 1024×1024: minutes → 38 s |
| [`e1a2025`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e1a2025) | ViT **query-blocked attention**; opt-in engram page-cache warm (`AFF_ENGRAM_WARM`) | 74×74: 38.4 s → 23.3 s, numerically identical |
| [`cac0510`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/cac0510) | size the compressed KV cache **per kv source, not per layer** | KV 1016 → 100 MiB/card; resident experts 2206 → 2357 |
| [`997489c`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/997489c) | **tile the indexer byte mask** over the prefill chunk | prefill arena 2068 → 1426 MiB/card; resident experts 2357 → 2462 |
| [`78244dc`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/78244dc) | **persist the learned placement heat plane** across restarts (`--placement-save`) | residency survives restarts (no re-learn from zero) |
| [`a4c9f85`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/a4c9f85) | recover **marker-less ("bare") DSML tool calls** and stop at the block close | a poisoned 117k-token replay dropped 4360 → 195 completion tokens; stops the agent runaway/confabulation |
| [`bc42be2`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/bc42be2) | DSpark **confidence head** + opt-in gating (`AFF_SPEC_CONF_T`) | wash on 2 cards — kept as a calibration instrument |
| [`27ec8fa`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/27ec8fa) | opt-in DSpark backoff after a zero-accept streak (`AFF_SPEC_BACKOFF=N`) | wash on 2 cards (verify dominates, not the draft) — kept as a knob |
| [`1ae7c09`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/1ae7c09) | **SSE keep-alive** while a tool call is withheld | long tool calls no longer trip client body-idle timeouts (300 s) |
| [`2a4478f`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2a4478f) | working tree captured: **on-GPU imatrix calibration** (`--calibrate-tokens/-chunk-size/-max-chunks`, `--imatrix-in/out`, `imatrix_accum_*` kernels), **bounded replay** (`AFF_BOUNDED_REPLAY`), **soft resident bias** (`AFF_RESIDENT_BIAS`), pinned sampler staging, async peer-copy overlap, `AFF_BLOCK_AT` hooks | the engine can emit its own imatrix from the model's own activations; host↔device copies overlap |
| [`2779348`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2779348) · [`patches/AFF_NOBUDGET_MAX_TOKENS.patch`](patches/AFF_NOBUDGET_MAX_TOKENS.patch) | bound the default when a request sends **no** `max_tokens` (was: the whole remaining context) | a gateway reasoning loop can no longer hold the single slot for 37 min / 55k tokens; explicit `max_tokens` untouched |

Every commit message carries the same measurement detail as the table, with the exact before/after.

**Ported from [Lucebox](https://github.com/Luce-Org/lucebox)** (Apache-2.0, by **Lucebox AI** / `Luce-Org`,
in `2a4478f` and its neighbours): the V4.1 decoder SWA bounded replay (`AFF_BOUNDED_REPLAY`, from
Lucebox's `ds41-gorgon` profile — see `src/engine/model.cpp`), the cost-aware adaptive verify width
(`AFF_ADAPTIVE_WIDTH`, from Lucebox's `AdaptiveSpecWidth` — `tools/affinity.cpp`), and the protected
massive/outlier expert set (from Lucebox's `share/deepseek41/massive_experts.json` —
`src/gpu/static_placement.hip`). The port's quality gate also reuses Lucebox's 8,184-token
fineweb-edu eval corpus. Credit and thanks to Lucebox for these; the code comments mark each site.
(`ds41-gorgon` is Lucebox's launch profile — DeepSeek-V4.1 on an R9700 beside a 192 GB Ryzen AI Max+.)

### V4.1 / multimodal port (fork `v41` branch, also in this history)

| diff | change | improvement |
|---|---|---|
| [`e6070f5`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e6070f5) | V4.1 reasoning-effort prefix + `陪同下` token, byte-identical to the reference `encoding.py` | correct V4.1 chat encoding |
| [`e03a05b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/e03a05b), [`774aa6a`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/774aa6a), [`4165fce`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4165fce), [`4bc938b`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/4bc938b) | engram gate/stride, engram bf16 weights, V4.1 has no per-head RMS on q | four bugs that produced fluent-but-wrong text; fixes make the port correct |
| [`d91ae9d`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/d91ae9d) | engram n-gram layers 1 and 14 (91.6 GiB tables), indexer, compressor | V4.1 architecture support |
| [`2dbb093`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/commit/2dbb093) | stop retrying a KV growth the card cannot hold | removes the 64k decode cliff |

### Environment knobs used in the served unit

`AFF_ENGRAM_WARM=1` `AFF_SPEC_BACKOFF=2` `AFF_ADAPTIVE_WIDTH=1` `AFF_BOUNDED_REPLAY=1`
`AFF_FUSE_HC_TAP=1` `AFF_DRAFT_HEAD_INT4=1` `AFF_CACTUS_DELTA=0.35` `AFF_CACTUS_CAP_R=0.5`
`AFF_RESIDENT_BIAS=0.008` `AFF_NOBUDGET_MAX_TOKENS=16384`.

See [`CHANGELOG.md`](CHANGELOG.md) for full version history, benchmarks, and commit attribution.

---

## Model files

Built from the uncensored FP8 checkpoint with affinity's own `aff-quantize`. The weights are hosted on
the Hugging Face Hub:

**[`Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw`](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw)**

| file (HF repo) | size | role |
|---|---|---|
| [`model-uc.aff.part-00 … part-04`](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw/tree/main) | **191.32 GiB** total | the container (routed experts 2.875 bpw, dense FP8, embedding/head bf16). Split into 5 parts for the Hub's per-file limit. |
| [`model-uc.dspark.aff`](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw/blob/main/model-uc.dspark.aff) | 5.22 GiB | DSpark speculative-decoding draft head |
| [`model-uc.engram`](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw/blob/main/model-uc.engram) | 506 KiB | engram n-gram sidecar |
| [`uc-v41cb.raw`](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw/blob/main/uc-v41cb.raw) | 32 KiB | the 2×1024-entry codebook the experts were encoded against |

Reassemble the main container with `cat model-uc.aff.part-* > model-uc.aff`; the parts' hashes are in
`SHA256SUMS.parts`.

The quantisation recipe is in [`model/QUANT.md`](model/QUANT.md); checks that must pass: `aff-info`,
`aff-cfgcheck`, `aff-fp8check`, `aff-exphash` (15360 distinct experts), `aff-gpucheck --dq`.

---

## Build and run

**Prebuilt package:** [release `v1.0.0`](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/releases/tag/v1.0.0)
carries `affinity-r9700-1.0.0.tar.gz` — the patched engine binary for `gfx1201` plus `deploy/`,
`model/` and `patches/`.

Build the engine image (needs Docker; the host needs no ROCm):

```sh
docker build -f docker/Dockerfile -t affinity:v41-vision .
```

Serve (see [`deploy/deepseek-v41-affinity.service`](deploy/deepseek-v41-affinity.service) for the full
unit, and [`deploy/run.sh`](deploy/run.sh)):

```sh
docker run --name deepseek-affinity \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e AFF_ENGRAM_WARM=1 -e AFF_SPEC_BACKOFF=2 -e AFF_ADAPTIVE_WIDTH=1 \
  -e AFF_BOUNDED_REPLAY=1 -e AFF_RESIDENT_BIAS=0.008 -e AFF_NOBUDGET_MAX_TOKENS=16384 \
  -v /models:/models affinity:v41-vision \
    -m /models/model-uc.aff --tokenizer /models/tokenizer.json \
    --dspark on --dspark-model /models/model-uc.dspark.aff \
    --engram /models/model-uc.engram --engram-tables /models \
    --kv-size 524288 --kv-commit 262144 \
    --host-pool-mib 200000 --placement-strategy demand \
    --placement-save /models/placement.affprof \
    --prefill-chunk-size 2816 --default-effort 75 --max-tokens-floor 0
```

---

## Performance notes

- Decode is **PCIe-bandwidth bound**, not context bound: step time is flat from 32k to 300k
  (`corr(prompt, s/block) = 0.06`), and tracks drafted tokens (`corr = 0.72`).
- The GPUs negotiate Gen5, but the switch uplink is Gen3 because the Xeon 8160 is a PCIe-3.0 part.
  Moving the cards to a PCIe-4.0 host (`x16` uplink) should roughly halve the expert-streaming term →
  ~1.6× decode, ≈ 30 tok/s. **Keep the uplink at ×16**: Gen4 ×8 equals Gen3 ×16 and would gain nothing.
- AVX-512 VNNI on the host would not help this build: the MoE matmuls run on the GPU (experts streamed
  over PCIe), the host does routing, sampling and engram lookups only.

## Credits

DeepSeek for the model; Yoshi Exeler / StillDeadcode for affinity; **Lucebox AI
([Luce-Org/lucebox](https://github.com/Luce-Org/lucebox), Apache-2.0) for the bounded-replay,
adaptive-width and protected-expert optimizations we ported (and the eval corpus)**; `dealignai` and
`pqhaz` for the uncensored builds; the friends whose fixes landed in the fork's `v4-fixes`. **The
transformation of affinity into this R9700-tweaked engine was done with the help of
DeepSeek-V4.1-Flash and Gemini 4 Argon.** Engine code remains MIT; see [`LICENSE`](LICENSE) and
[`FORK.md`](FORK.md).

Third-party components and their licences (affinity MIT; Lucebox Apache-2.0 with the exact ported
sites; stb_image dual MIT/public-domain) are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
