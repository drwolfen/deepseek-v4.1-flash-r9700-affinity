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

**Model files (1017 revision):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-1017-UNCENSORED-affinity-2.875bpw)  
**Model files (legacy 0925):** [huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw](https://huggingface.co/Muoniolalusta/DeepSeek-V4.1-Flash-UNCENSORED-affinity-2.875bpw)  

**Changelog & Benchmarks:** [**`CHANGELOG.md`**](CHANGELOG.md) | **Latest Release:** [**`v5.0.0`**](https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity/releases/tag/v5.0.0) | **Full Progression & Bake-Off:** [**`docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md`**](docs/benchmarks/2026-11-15-v5.0-and-radiance-v2.1-bakeoff.md)

**Release Highlights (Waves 0–4 Progression):**
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
| Engine build | Docker image `deepseek-affinity:v5.0.0` (ROCm 10.1.0-full, `gfx1201`) |
| Serving ports | Dual active: `:8000` (Docker container) and `:8080` (`r9v-proxy-8080.py` for Hermes `local-heavy`) |

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
| **Dual Serving (`:8000` & `:8080`)** | **Active & Verified Healthy** | N/A |

**Verdict**: Radiance v2.1.0 cannot serve DeepSeek-V4.1-Flash because the architecture plugin and MLA/Engram kernels are omitted from the open-source release, and its safetensors reader explicitly rejects `F8_E8M0` scales. `deepseek-affinity:v5.0.0` remains the sole working high-throughput production engine.

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
```

---

## Build and Run

### Building the Docker Container (ROCm 10.1)

```sh
docker build -f docker/Dockerfile -t deepseek-affinity:v5.0.0 .
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
  -p 8000:8000 \
  -v /mnt/fast-storage:/mnt/fast-storage \
  deepseek-affinity:v5.0.0 \
    -m /mnt/fast-storage/DeepSeek-V4.1-Flash-1017-UNCENSORED-aff/model-uc-1017.aff \
    --tokenizer /mnt/fast-storage/LLM-Models/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8/tokenizer.json \
    --host 0.0.0.0 --port 8000 \
    --raw-completions \
    --gpu-headroom-mib 384 \
    --dspark on \
    --dspark-model /mnt/fast-storage/DeepSeek-V4.1-Flash-1017-UNCENSORED-aff/model-uc-1017.dspark.aff \
    --engram /mnt/fast-storage/DeepSeek-V4.1-Flash-1017-UNCENSORED-aff/model-uc-1017.engram \
    --engram-tables /mnt/fast-storage/LLM-Models/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 \
    --prefix-cache-dir /mnt/fast-storage/DeepSeek-V4.1-Flash-1017-UNCENSORED-aff/pcache \
    --prefix-cache-gib 64 \
    --persistent-prefix-cache \
    --kv-size 327680 \
    --kv-commit 262144 \
    --host-pool-mib 200000 --host-pool-reserve-mib 1024 \
    --placement-strategy demand \
    --placement-hysteresis 1.6 --placement-min-gain 3.0 \
    --placement-save /mnt/fast-storage/DeepSeek-V4.1-Flash-1017-UNCENSORED-aff/placement.affprof \
    --prefill-chunk-size 2816 \
    --default-effort 75 \
    --repeat-penalty 1.1 --repeat-last-n 96 --dry-delta 40 \
    --max-tokens-floor 0
```

Health endpoint check:
```sh
curl -sS http://127.0.0.1:8000/health
# {"status":"ok","version":"v5.0.0","git_sha":"264e8d5","build_stamp":"2026-11-16T04:49:47Z","resident_experts":3416,"free_vram_mib":[204,300]}
```

---

## Credits

- DeepSeek for the base model architecture and reference code.
- Yoshi Exeler / StillDeadcode for affinity and Radiance algorithmic foundations.
- Lucebox AI ([Luce-Org/lucebox](https://github.com/Luce-Org/lucebox)) for bounded replay, adaptive verify width, and evaluation corpus.
- 05yuki ([05yuki/deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)) for INT4 LM head and tool namespace specs.
- `dealignai` and `pqhaz` for the uncensored and abliterated weights.
- Transformation of affinity into this R9700-tweaked engine was executed with DeepSeek-V4.1-Flash, Gemini 4 Carbon, Gemini 4 Argon, Gemini 4 Hydrogen, and Gemini 3.8 Flash.
