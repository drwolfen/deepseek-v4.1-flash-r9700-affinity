# deepseek-v4.1-flash-r9700-affinity

A working setup for serving **DeepSeek-V4.1-Flash** (the uncensored / abliterated build) on **two AMD
Radeon AI PRO R9700s**, using a fork of the [affinity](https://codeberg.org/StillDeadcode/affinity)
inference engine, plus a locally quantised 2.875-bpw container of the model.

This repository is the *setup*: the engine source (fork + our patches), the deployment unit, the
quantisation recipe, and the measurements, all captured on one machine. It exists so the exact
combination that produced the numbers below can be rebuilt.

> **Tested on exactly two R9700s.** Every number here was measured on a two-card box. The engine
> upstream supports one card, but that configuration was not benchmarked for this setup.

---

## Hardware (the machine the numbers come from)

| | |
|---|---|
| GPU | **2 × AMD Radeon AI PRO R9700**, `gfx1201` (RDNA4), 32 GiB each |
| CPU | Intel Xeon Platinum 8160 (Skylake-SP), 2 sockets × 24 cores / 48 threads = 96 threads |
| RAM | 754 GiB DDR4 (2 NUMA nodes) |
| Storage | ~5.9 TB RAID for the model + prefix cache |
| PCIe | GPUs link at **Gen5 ×16**, but the switch **uplink runs Gen3 ×16 (8.0 GT/s)** — the host Xeon is a PCIe-3.0 CPU. See *Performance notes*. |
| Host OS | Ubuntu, kernel 7.0.0, ROCm 7.2.4 |
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

## Current speed (measured, this machine)

Single stream, 512k-context configuration, DSpark speculative decoding on, thinking on:

| | |
|---|---|
| **Decode, long-context reasoning** | **18–19.6 tok/s** (median ≈ 18.9), stable across 40+ hours |
| Decode, short / high-acceptance bursts | 24–25 tok/s |
| Draft acceptance | 54–93 % (2.3–4.9 tokens per verified block) |
| Prefill | ~1000 tok/s at 16k–48k (prefix cache hit 98.7 %, so prefill is rarely exercised) |

Decode is bound by moving the union of selected experts across PCIe: 3274/15360 experts (21.3 %) are
resident in the two cards, 12086 are read from the 143.5 GiB host pool over the bus. Per decode block
the engine streams ≈ 2.35 GiB/card, which is ≈ 95 % of the Gen3 ×16 ceiling — the link is saturated.
That is why the numbers are flat versus context length and why a Gen4 ×16 host would be the single
biggest speedup (~1.6×, to ~30 tok/s).

---

## What comes from where

Nothing here is the model author's or the engine author's endorsement; this is a personal fork.

| component | origin | license / note |
|---|---|---|
| Inference engine | [StillDeadcode/affinity](https://codeberg.org/StillDeadcode/affinity) by Yoshi Exeler | MIT. Upstream of everything. |
| Engine fork (this tree) | `neuromaniacMD/affinity` (our GitHub) | MIT. Adds the rework listed below. Kept as provenance; this repo is the deployable copy. |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | DeepSeek license. Source of the reference `inference/` and `encoding/` code. |
| Uncensored weights (`-uc`) | [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8) | FP8 checkpoint that we re-quantised to 2.875 bpw. |
| Abliterated weights (alt) | [pqhaz/DeepSeek-V4.1-Flash-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-heretic) | Directional ablation of refusal behaviour; lower KL than `-uc`. A `model-heretic.aff` was built but `-uc` is what is served. |
| GGUF / imatrix | [Adolphsson/DeepSeek-V4.1-Flash-UNCENSORED-GGUF](https://huggingface.co/Adolphsson/DeepSeek-V4.1-Flash-UNCENSORED-GGUF) | Import-matrix source for the 2.875-bpw encode. |
| Quantiser | affinity's own `aff-quantize` | Routed experts only, 2.875 bpw codebook quant, imatrix-weighted. |

The engine itself, its design, the kernels and the base README are upstream's work and remain MIT
licensed (`LICENSE`, `FORK.md`, `UPSTREAM-README.md` kept unchanged).

---

## Rework and patches (with measured effect)

Branches: `v4-fixes` and `v41` are the fork's published branches (mirroring upstream `a48b72e` plus the
fork's robustness fixes and the V4.1 port). This repository additionally carries the **unpublished
local work** that produced the served binary: ten commits ahead of `v41` plus one dirty-tree commit and
one patch.

### V4.1 / multimodal port (fork `v41` branch)

| commit | change | improvement |
|---|---|---|
| `e6070f5` | V4.1 reasoning-effort prefix + `<｜tool▁outputs▁begin｜>` token, byte-identical to the reference `encoding.py` | correct V4.1 chat encoding |
| `774aa6a`, `4165fce`, `4bc938b`, `e03a05b` | engram gate/stride, engram bf16 weights, V4.1 has no per-head RMS on q | four bugs that produced fluent-but-wrong text; fixes make the port correct |
| `d91ae9d` … `4bc938b` | engram n-gram layers 1 and 14 (91.6 GiB tables), indexer, compressor | V4.1 architecture support |
| `2dbb093` | stop retrying a KV growth the card cannot hold | removes the 64k decode cliff |
| `91e78ba` … | DeepSeek ViT on the CPU (see vision, below) | image input |

### Local work (this repo's tip, ahead of the fork's published branches)

| commit | change | measured effect |
|---|---|---|
| `91e78ba` | **DeepSeek ViT on the CPU**, wired into prefill and the chat endpoint. 3 discrepancies fixed (2D-RoPE half rotation, channel-major aligner, aspect-preserving pad) | embedding cosine vs reference 0.31 → 0.85; end-to-end image grounding works |
| `cbb3f07` | ViT **weight-stationary GEMM + AVX-512 attention** | 546×546: 13.2 s → 3.25 s; 1024×1024: minutes → 38 s |
| `e1a2025` | ViT query-blocked attention; opt-in engram page-cache warm (`AFF_ENGRAM_WARM`) | 74×74: 38.4 s → 23.3 s, numerically identical |
| `cac0510` | size the compressed KV cache **per kv source**, not per layer | KV 1016 → 100 MiB/card; resident experts 2206 → 2357 |
| `997489c` | tile the indexer byte mask over the prefill chunk | prefill arena 2068 → 1426 MiB/card; resident experts 2357 → 2462 |
| `27ec8fa` | opt-in DSpark backoff after a zero-accept streak (`AFF_SPEC_BACKOFF=N`) | wash on 2 cards (verify dominates, not the draft) — kept as a knob |
| `1ae7c09` | SSE keep-alive while a tool call is withheld | long tool calls no longer trip client body-idle timeouts (300 s) |
| `78244dc` | persist the learned placement heat plane (`--placement-save`) | residency survives restarts (no re-learn from zero) |
| `bc42be2` | DSpark **confidence head** + opt-in gating (`AFF_SPEC_CONF_T`) | wash on 2 cards — kept as a calibration instrument |
| `a4c9f85` | recover marker-less ("bare") DSML tool calls and stop at the block close | a poisoned 117k-token replay dropped 4360 → 195 completion tokens; stops the agent runaway/confabulation |
| *(dirty)* | on-GPU **imatrix calibration** (`--calibrate-tokens/-chunk-size/-max-chunks`, `--imatrix-in/out`), bounded replay (`AFF_BOUNDED_REPLAY`), soft resident bias (`AFF_RESIDENT_BIAS`), pinned sampler staging, async peer-copy overlap, `AFF_BLOCK_AT` hooks | engine can emit its own imatrix from the model's own activations; overlap of host↔device copies |
| `AFF_NOBUDGET_MAX_TOKENS` (`patches/`) | bound the default when a request sends **no** `max_tokens` (was: the whole remaining context) | a gateway reasoning loop can no longer hold the single slot for 37 min / 55k tokens; explicit `max_tokens` untouched |

### Environment knobs used in the served unit

`AFF_ENGRAM_WARM=1` `AFF_SPEC_BACKOFF=2` `AFF_ADAPTIVE_WIDTH=1` `AFF_BOUNDED_REPLAY=1`
`AFF_RESIDENT_BIAS=0.008` `AFF_NOBUDGET_MAX_TOKENS=16384`.

---

## Model files

Built from the uncensored FP8 checkpoint with affinity's own `aff-quantize`. The weights are hosted on
the Hugging Face Hub (see the link at the top / the model repo README):

| file | size | role |
|---|---|---|
| `model-uc.aff` | **191.32 GiB** | the container (routed experts 2.875 bpw, dense FP8, embedding/head bf16). Split into 5 parts for the Hub's per-file limit. |
| `model-uc.dspark.aff` | 5.22 GiB | DSpark speculative-decoding draft head |
| `model-uc.engram` | 506 KiB | engram n-gram sidecar |
| `uc-v41cb.raw` | 32 KiB | the 2×1024-entry codebook the experts were encoded against |

Reassemble the main container with `cat model-uc.aff.part-* > model-uc.aff` and verify with
`sha256sum -c SHA256SUMS`.

The quantisation recipe is in [`model/QUANT.md`](model/QUANT.md); checks that must pass: `aff-info`,
`aff-cfgcheck`, `aff-fp8check`, `aff-exphash` (15360 distinct experts), `aff-gpucheck --dq`.

---

## Build and run

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

DeepSeek for the model; Yoshi Exeler / StillDeadcode for affinity; `dealignai` and `pqhaz` for the
uncensored builds; the friends whose fixes landed in the fork's `v4-fixes`. Engine code remains MIT;
see [`LICENSE`](LICENSE) and [`FORK.md`](FORK.md).
