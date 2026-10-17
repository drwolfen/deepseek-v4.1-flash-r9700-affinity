# Quantisation recipe — `model-uc.aff`

How the 191.32 GiB (2.875 bpw) container served by this setup was produced, from the uncensored FP8
checkpoint. This follows affinity's own `aff-quantize` (the engine's upstream tool); only the routed
experts are quantised. Dense weights stay FP8, embedding/head bf16.

Inputs:

- **source checkpoint**: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (HF), 40 layers, 384 experts,
  hidden 5120, inter 2304.
- **imatrix**: `imatrix-dsv41-262144.gguf` (also a domain imatrix was generated on-GPU by the engine;
  see below).

## Steps

```sh
D=/models
CK=$D/DeepSeek-V4.1-Flash-UNCENSORED-FP8

# 1. codebook, once (~20 min)
aff-quantize --src "$CK" --cb-spread --codebook-out v41cb.raw

# 2. routed experts, sharded by layer (splittable across machines with --layer-range)
#    imatrix-weighted, 2.875 bpw with the engine's codebook quant (2 variants x 1024 entries)
aff-quantize --src "$CK" --codebook-raw v41cb.raw --imatrix imatrix-dsv41-262144.gguf \
             --experts-dir experts --layer-range 0:40 --threads N

# 3. DSpark draft head, quantised against the main container's codebook
aff-quantize --src "$CK" --dspark --codebook-raw v41cb.raw --experts-dir experts-dspark

# 4. dense pass, assembling the experts into the final container
aff-quantize --src "$CK" --codebook-raw v41cb.raw --imatrix imatrix-dsv41-262144.gguf \
             --experts-from experts --out model-uc.aff
aff-quantize --src "$CK" --dspark --codebook-raw v41cb.raw --experts-from experts-dspark \
             --out model-uc.dspark.aff

# 5. engram sidecar (needs tokenizers)
python3 tools/v41/engram_prep.py "$CK" model-uc.engram
```

`model/finish-stage-a.sh` in this repo drives steps 4–5 and runs the gates. Every worker must use the
same binary, codebook and imatrix.

## On-GPU imatrix (engine feature)

This fork adds a **calibration mode to the engine**: it accumulates the routed experts' input and
SwiGLU activation statistics on the GPU while serving a calibration corpus, and can save them as a
GGUF imatrix, so the quantisation can be weighted by the *model's own* activation distribution instead
of a generic corpus:

```
--calibrate-tokens FILE     tokens to feed
--calibrate-chunk-size N    (default 4096)
--calibrate-max-chunks N
--imatrix-in FILE           load an existing imatrix
--imatrix-out FILE          save the accumulated imatrix
--imatrix-save-chunks N     checkpoint every N chunks (default 100)
```

The accumulation kernels are `imatrix_accum_x_kernel` and `imatrix_accum_swiglu_kernel`.

## Gates (all must pass)

```sh
aff-info model-uc.aff                                   # layout / sizes
aff-cfgcheck model-uc.aff                               # config vs tensors
aff-fp8check model-uc.aff "$CK"                         # FP8 tensors bit-faithful
aff-exphash model-uc.aff | sort -u -k3 | wc -l          # want 15360 distinct experts
aff-gpucheck --dq model-uc.aff                          # GPU vs CPU on the real bytes
```

Measured build: 15360/15360 experts in ~622 s for the assembly stage; final container 191.32 GiB.
The codebook hash recorded at assembly was `539439d8cc1b65a3`.
