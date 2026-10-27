# Third-party components and attribution

This repository is a derivative work. It builds on, and incorporates code and weights from, several
other projects. Their licences and the parts used here are listed below.

## affinity — MIT

The inference engine this repository forks.
Copyright (c) 2026 Yoshi Exeler — <https://codeberg.org/StillDeadcode/affinity>.
Full text: [`LICENSE`](LICENSE). The fork's own changes are described in [`FORK.md`](FORK.md) and the
main [`README.md`](README.md).

## Lucebox — Apache License 2.0

**Lucebox AI** (`Luce-Org`) — <https://github.com/Luce-Org/lucebox> · <https://www.lucebox.com/>.
Licensed under the Apache License, Version 2.0. Full text:
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

These optimizations in the fork are **ports from Lucebox's DeepSeek-V4.1 `ds41-gorgon` work**:

| ported feature | location in this repo | origin |
|---|---|---|
| V4.1 decoder SWA **bounded replay** (`AFF_BOUNDED_REPLAY`) | `src/engine/model.cpp` | Lucebox `ds41-gorgon` profile |
| Cost-aware **adaptive verify width** (`AFF_ADAPTIVE_WIDTH`) | `tools/affinity.cpp` | Lucebox `AdaptiveSpecWidth` |
| **Protected massive/outlier experts** (never demoted) | `src/gpu/static_placement.hip` | Lucebox `share/deepseek41/massive_experts.json` |
| Quality-gate eval corpus (8,184 fineweb-edu tokens) | `docs/V41-*.md`, `model/` | Lucebox `eval/` |

Each ported site carries a code comment naming Lucebox as its source.

**On licence mixing.** The ported fragments originate under Apache-2.0; the repository as a whole is
distributed under the MIT licence in [`LICENSE`](LICENSE) (affinity's). The Apache-2.0 terms continue to
apply to the ported fragments; the required attribution and a copy of the Apache-2.0 licence are
retained here. No file in this repository is claimed to be relicensed away from Apache-2.0.

## Radiance — Yoshi Exeler

**Radiance engine** — <https://codeberg.org/StillDeadcode/radiance> (v1.12.0).
Optimizations ported to the fork in v1.1.0:
- Cactus length-faded and capped rejection sampling (`AFF_CACTUS_DELTA`, `AFF_CACTUS_CAP_R` in `tools/affinity.cpp`).
- Fused `batch_hc_expand_reg_kernel` (`AFF_FUSE_HC_TAP` in `src/gpu/batch_kernels.hip` and `src/gpu/dense_gpu.hip`).

## KTransformers (05yuki DeepSeek-1017 fork)

**05yuki** — <https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers>.
Optimizations and specifications ported in v1.1.0:
- Group-128 MSE-clipped symmetric INT4 output head quantization (`AFF_DRAFT_HEAD_INT4` in `src/gpu/batch_kernels.hip` and `src/gpu/dense_gpu.hip`).
- Opt-in FP8 output activation QDQ (`AFF_O_FP8_QDQ` in `src/gpu/batch_kernels.hip`).
- 1017 tool namespace qualification and completion parser (`src/engine/chat_encode.cpp`, `src/server/server.cpp`).

## stb_image — MIT / public domain (dual)

Bundled at `src/vision/stb_image.h` (v2.30) for image decode in the vision path.
Copyright (c) 2017 Sean Barrett — <http://nothings.org/stb>. Dual-licensed; its own licence text
(MIT or Unlicense / public domain) is embedded at the end of the file. Used here under either option.

## Other build inputs (not vendored code)

- **AMD TheRock ROCm** toolchain and the base container image — see `docker/Dockerfile` and its base
  image's own licences.
- **DeepSeek-V4.1-Flash** — model weights under the DeepSeek licence
  (<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>).

## Model weights distributed on the Hub (not in this repo)

The container this repo serves is a derivative of community checkpoints, each under its own terms:

- uncensored FP8 source: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`
- abliterated alternative: `pqhaz/DeepSeek-V4.1-Flash-heretic`
- GGUF import-matrix: `Adolphsson/DeepSeek-V4.1-Flash-UNCENSORED-GGUF`

The DeepSeek licence governs the weights. See the model card on the Hub.

## Development assistance

The transformation of the affinity engine into this R9700-tweaked build, and the patches in
[`patches/`](patches/), were developed with the help of **DeepSeek-V4.1-Flash** and **Gemini 4 Argon**.
