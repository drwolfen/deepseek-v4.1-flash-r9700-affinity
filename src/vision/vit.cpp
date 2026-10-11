#include "vit.h"
#include "engine/threadpool.h"
#include "quant/source_dtypes.h"

#include <immintrin.h>
#include "engine/matvec_bf16.inc"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace aff {

static inline float bf16_val(uint16_t v) {
  return bf16_to_float(v);
}

static inline float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

static inline float gelu(float x) {
  return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
}

static void matvec_dot_bf16(const uint16_t* W, const float* x, uint64_t cols, float* out_val) {
  // Single output row: the AVX-512 bf16 kernel already runs at the DRAM wall, and the scalar
  // bf16->f32 loop this replaces left the whole ViT memory-bound. See engine/matvec_bf16.inc.
  matvec_bf16_v2(W, x, 1, cols, out_val);
}

bool ViT::load(const std::string& model_dir, std::string* err) {
  ready_ = false;
  const std::string s1_path = model_dir + "/model-00001-of-00048.safetensors";
  const std::string s2_path = model_dir + "/model-00002-of-00048.safetensors";

  if (!shard1_.open(s1_path, err)) return false;
  if (!shard2_.open(s2_path, err)) return false;

  auto get_t1 = [&](const std::string& name) -> const uint16_t* {
    const StTensor* t = shard1_.find(name);
    if (!t) return nullptr;
    return reinterpret_cast<const uint16_t*>(t->data);
  };

  auto get_t2 = [&](const std::string& name) -> const uint16_t* {
    const StTensor* t = shard2_.find(name);
    if (!t) return nullptr;
    return reinterpret_cast<const uint16_t*>(t->data);
  };

  weights_.patch_proj_w = get_t1("vision.patch_embed.proj.weight");
  weights_.patch_proj_b = get_t1("vision.patch_embed.proj.bias");
  if (!weights_.patch_proj_w || !weights_.patch_proj_b) {
    if (err) *err = "missing patch embed weights in shard 1";
    return false;
  }

  weights_.blocks.resize(32);
  for (uint32_t i = 0; i < 32; ++i) {
    ViTBlockWeights& b = weights_.blocks[i];
    std::string p = "vision.blocks." + std::to_string(i) + ".";
    b.norm1_w     = get_t1(p + "norm1.weight");
    b.attn_wqkv_w = get_t1(p + "attn.wqkv.weight");
    b.attn_wqkv_b = get_t1(p + "attn.wqkv.bias");
    b.attn_wo_w   = get_t1(p + "attn.wo.weight");
    b.attn_wo_b   = get_t1(p + "attn.wo.bias");
    b.norm2_w     = get_t1(p + "norm2.weight");
    b.mlp_w1_w    = get_t1(p + "mlp.w1.weight");
    b.mlp_w2_w    = get_t1(p + "mlp.w2.weight");
    if (!b.norm1_w || !b.attn_wqkv_w || !b.attn_wqkv_b || !b.attn_wo_w || !b.attn_wo_b ||
        !b.norm2_w || !b.mlp_w1_w || !b.mlp_w2_w) {
      if (err) *err = "missing weights for vision block " + std::to_string(i);
      return false;
    }
  }

  weights_.final_norm_w = get_t1("vision.norm.weight");
  weights_.aligner_w1_w = get_t1("aligner.w1.weight");
  weights_.aligner_w1_b = get_t1("aligner.w1.bias");
  weights_.aligner_w2_w = get_t1("aligner.w2.weight");
  weights_.aligner_w2_b = get_t1("aligner.w2.bias");

  if (!weights_.final_norm_w || !weights_.aligner_w1_w || !weights_.aligner_w1_b ||
      !weights_.aligner_w2_w || !weights_.aligner_w2_b) {
    if (err) *err = "missing aligner or final norm weights in shard 1";
    return false;
  }

  weights_.image_start   = get_t2("image_start");
  weights_.image_end     = get_t2("image_end");
  weights_.image_newline = get_t2("image_newline");
  if (!weights_.image_start || !weights_.image_end || !weights_.image_newline) {
    if (err) *err = "missing image_start/end/newline special token embeddings in shard 2";
    return false;
  }

  ready_ = true;
  return true;
}

bool ViT::forward(const DecodedImage& img, std::vector<float>* out_embeddings,
                  uint32_t* n_tokens, std::string* err) const {
  if (!ready_) {
    if (err) *err = "ViT engine not initialized";
    return false;
  }
  if (img.width % 14 != 0 || img.height % 14 != 0 || img.data.empty()) {
    if (err) *err = "invalid image dimensions: must be multiples of 14 (patch size)";
    return false;
  }

  const int hp = img.h_patches;
  const int wp = img.w_patches;
  const uint32_t N = (uint32_t)(hp * wp);
  const uint32_t D = 1024;
  const uint32_t NH = 16;
  const uint32_t HD = 64;

  // 1. Patch Embedding: project 588 inputs -> 1024 dims
  std::vector<float> X((size_t)N * D);
  std::vector<float> patch_buf(588);

  parallel_for(N, 16, [&](uint64_t b0, uint64_t b1) {
    std::vector<float> pbuf(588);
    for (uint64_t p = b0; p < b1; ++p) {
      int pr = (int)p / wp;
      int pc = (int)p % wp;

      // Extract 14x14x3 patch in channel-first order: c * 196 + kh * 14 + kw
      for (int c = 0; c < 3; ++c) {
        for (int kh = 0; kh < 14; ++kh) {
          int y = pr * 14 + kh;
          for (int kw = 0; kw < 14; ++kw) {
            int x = pc * 14 + kw;
            pbuf[(size_t)c * 196 + kh * 14 + kw] = img.data[((size_t)y * img.width + x) * 3 + c];
          }
        }
      }

      float* x_out = X.data() + p * D;
      for (uint32_t d = 0; d < D; ++d) {
        float bias = bf16_val(weights_.patch_proj_b[d]);
        float dot = 0.0f;
        const uint16_t* w_row = weights_.patch_proj_w + (size_t)d * 588;
        matvec_dot_bf16(w_row, pbuf.data(), 588, &dot);
        x_out[d] = bias + dot;
      }
    }
  });

  // Precompute 2D RoPE cos/sin tables: 16 frequencies
  std::vector<float> theta(16);
  for (int j = 0; j < 16; ++j) {
    theta[j] = std::pow(10000.0f, -2.0f * (float)j / 32.0f);
  }

  std::vector<float> cos_r(hp * 16), sin_r(hp * 16);
  for (int r = 0; r < hp; ++r) {
    for (int j = 0; j < 16; ++j) {
      cos_r[r * 16 + j] = std::cos((float)r * theta[j]);
      sin_r[r * 16 + j] = std::sin((float)r * theta[j]);
    }
  }

  std::vector<float> cos_c(wp * 16), sin_c(wp * 16);
  for (int c = 0; c < wp; ++c) {
    for (int j = 0; j < 16; ++j) {
      cos_c[c * 16 + j] = std::cos((float)c * theta[j]);
      sin_c[c * 16 + j] = std::sin((float)c * theta[j]);
    }
  }

  // Intermediate buffers for transformer block
  std::vector<float> X_norm((size_t)N * D);
  std::vector<float> Q((size_t)N * D), K((size_t)N * D), V((size_t)N * D);
  std::vector<float> Attn_out((size_t)N * D);

  // 2. Iterate through 32 ViT blocks
  for (uint32_t blk = 0; blk < 32; ++blk) {
    const ViTBlockWeights& bw = weights_.blocks[blk];

    // a. Pre-norm 1 (RMSNorm)
    parallel_for(N, 64, [&](uint64_t b0, uint64_t b1) {
      for (uint64_t i = b0; i < b1; ++i) {
        const float* x_in = X.data() + i * D;
        float* x_out = X_norm.data() + i * D;
        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < D; ++d) sum_sq += x_in[d] * x_in[d];
        float inv = 1.0f / std::sqrt(sum_sq / (float)D + 1e-6f);
        for (uint32_t d = 0; d < D; ++d) {
          x_out[d] = x_in[d] * inv * bf16_val(bw.norm1_w[d]);
        }
      }
    });

    // b. QKV projection + bias + 2D RoPE
    parallel_for(N, 16, [&](uint64_t b0, uint64_t b1) {
      std::vector<float> qkv(3072);
      for (uint64_t i = b0; i < b1; ++i) {
        const float* x_in = X_norm.data() + i * D;
        int pr = (int)i / wp;
        int pc = (int)i % wp;

        for (uint32_t d = 0; d < 3072; ++d) {
          float bias = bf16_val(bw.attn_wqkv_b[d]);
          float dot = 0.0f;
          matvec_dot_bf16(bw.attn_wqkv_w + (size_t)d * D, x_in, D, &dot);
          qkv[d] = bias + dot;
        }

        float* qi = Q.data() + i * D;
        float* ki = K.data() + i * D;
        float* vi = V.data() + i * D;
        std::memcpy(qi, qkv.data(), D * sizeof(float));
        std::memcpy(ki, qkv.data() + D, D * sizeof(float));
        std::memcpy(vi, qkv.data() + 2 * D, D * sizeof(float));

        // Apply 2D RoPE to Q and K
        const float* cr = cos_r.data() + pr * 16;
        const float* sr = sin_r.data() + pr * 16;
        const float* cc = cos_c.data() + pc * 16;
        const float* sc = sin_c.data() + pc * 16;

        // Reference 2D RoPE (vision.py: get_vision_cos_sin / apply_rotary): a 32-long cos/sin, first
        // 16 entries the row position and next 16 the column, and the two 32-dim halves of each
        // 64-dim head rotate AGAINST each other. The interleaved-pair form this replaces was a
        // different convention and decorrelated the embeddings (cos ~0.37 against the reference).
        float cos32[32], sin32[32];
        for (int j = 0; j < 16; ++j) { cos32[j] = cr[j]; sin32[j] = sr[j]; }
        for (int j = 0; j < 16; ++j) { cos32[16 + j] = cc[j]; sin32[16 + j] = sc[j]; }
        for (uint32_t h = 0; h < NH; ++h) {
          float* qh = qi + h * HD;
          float* kh = ki + h * HD;
          for (int k = 0; k < 32; ++k) {
            float q1 = qh[k], q2 = qh[32 + k];
            qh[k]      = q1 * cos32[k] - q2 * sin32[k];
            qh[32 + k] = q2 * cos32[k] + q1 * sin32[k];
            float k1 = kh[k], k2 = kh[32 + k];
            kh[k]      = k1 * cos32[k] - k2 * sin32[k];
            kh[32 + k] = k2 * cos32[k] + k1 * sin32[k];
          }
        }
      }
    });

    // c. Scaled dot-product self-attention: softmax(Q @ K.T / sqrt(64)) @ V
    const float attn_scale = 1.0f / std::sqrt((float)HD);
    parallel_for(N, 16, [&](uint64_t b0, uint64_t b1) {
      std::vector<float> scores(N);
      for (uint64_t i = b0; i < b1; ++i) {
        float* out_tok = Attn_out.data() + i * D;
        for (uint32_t h = 0; h < NH; ++h) {
          const float* qi = Q.data() + i * D + h * HD;
          float max_sc = -1e30f;
          for (uint32_t j = 0; j < N; ++j) {
            const float* kj = K.data() + j * D + h * HD;
            float dot = 0.0f;
            for (uint32_t d = 0; d < HD; ++d) dot += qi[d] * kj[d];
            float sc = dot * attn_scale;
            scores[j] = sc;
            if (sc > max_sc) max_sc = sc;
          }
          float sum_exp = 0.0f;
          for (uint32_t j = 0; j < N; ++j) {
            scores[j] = std::exp(scores[j] - max_sc);
            sum_exp += scores[j];
          }
          float inv_sum = 1.0f / sum_exp;
          for (uint32_t j = 0; j < N; ++j) scores[j] *= inv_sum;

          float* out_head = out_tok + h * HD;
          for (uint32_t d = 0; d < HD; ++d) {
            float val = 0.0f;
            for (uint32_t j = 0; j < N; ++j) {
              val += scores[j] * (V.data() + j * D + h * HD)[d];
            }
            out_head[d] = val;
          }
        }
      }
    });

    // d. Attention Out projection + residual add
    parallel_for(N, 32, [&](uint64_t b0, uint64_t b1) {
      for (uint64_t i = b0; i < b1; ++i) {
        const float* attn_tok = Attn_out.data() + i * D;
        float* x_tok = X.data() + i * D;
        for (uint32_t d = 0; d < D; ++d) {
          float bias = bf16_val(bw.attn_wo_b[d]);
          float dot = 0.0f;
          matvec_dot_bf16(bw.attn_wo_w + (size_t)d * D, attn_tok, D, &dot);
          x_tok[d] += bias + dot;
        }
      }
    });

    // e. Pre-norm 2 (RMSNorm)
    parallel_for(N, 64, [&](uint64_t b0, uint64_t b1) {
      for (uint64_t i = b0; i < b1; ++i) {
        const float* x_in = X.data() + i * D;
        float* x_out = X_norm.data() + i * D;
        float sum_sq = 0.0f;
        for (uint32_t d = 0; d < D; ++d) sum_sq += x_in[d] * x_in[d];
        float inv = 1.0f / std::sqrt(sum_sq / (float)D + 1e-6f);
        for (uint32_t d = 0; d < D; ++d) {
          x_out[d] = x_in[d] * inv * bf16_val(bw.norm2_w[d]);
        }
      }
    });

    // f. SwiGLU MLP: w1 is [5632, 1024] (gate 2816, up 2816), w2 is [1024, 2816] + residual add
    parallel_for(N, 16, [&](uint64_t b0, uint64_t b1) {
      std::vector<float> gu(5632);
      std::vector<float> h(2816);
      for (uint64_t i = b0; i < b1; ++i) {
        const float* x_in = X_norm.data() + i * D;
        float* x_tok = X.data() + i * D;

        for (uint32_t d = 0; d < 5632; ++d) {
          float dot = 0.0f;
          matvec_dot_bf16(bw.mlp_w1_w + (size_t)d * D, x_in, D, &dot);
          gu[d] = dot;
        }

        for (uint32_t d = 0; d < 2816; ++d) {
          h[d] = silu(gu[d]) * gu[2816 + d];
        }

        for (uint32_t d = 0; d < D; ++d) {
          float dot = 0.0f;
          matvec_dot_bf16(bw.mlp_w2_w + (size_t)d * 2816, h.data(), 2816, &dot);
          x_tok[d] += dot;
        }
      }
    });
  }

  // 3. Final RMSNorm
  std::vector<float> X_final((size_t)N * D);
  parallel_for(N, 64, [&](uint64_t b0, uint64_t b1) {
    for (uint64_t i = b0; i < b1; ++i) {
      const float* x_in = X.data() + i * D;
      float* x_out = X_final.data() + i * D;
      float sum_sq = 0.0f;
      for (uint32_t d = 0; d < D; ++d) sum_sq += x_in[d] * x_in[d];
      float inv = 1.0f / std::sqrt(sum_sq / (float)D + 1e-6f);
      for (uint32_t d = 0; d < D; ++d) {
        x_out[d] = x_in[d] * inv * bf16_val(weights_.final_norm_w[d]);
      }
    }
  });

  // 4. Downsampling (3x3 pixel unshuffle). Reference Aligner: view(n_h,n_w,C) -> permute(2,0,1) ->
  //    F.pad zeros by (-n_w % 3, -n_h % 3) -> F.unfold(3, stride=3). The 9216-vector is therefore
  //    CHANNEL-major, index c*9 + dr*3 + dc, not the (dr,dc)-major form used before (which was
  //    transposed against the trained aligner, cos ~0.35 against the reference). hd/wd are ceil, so
  //    a partial feature-map row/col contributes zeros.
  const int hd = img.h_down;
  const int wd = img.w_down;
  const uint32_t M = (uint32_t)(hd * wd);
  const uint32_t IN_DIM = 9216;
  const uint32_t LLM_DIM = 5120;

  std::vector<float> X_down((size_t)M * IN_DIM, 0.0f);
  for (int r = 0; r < hd; ++r) {
    for (int c = 0; c < wd; ++c) {
      float* dst_tok = X_down.data() + (size_t)(r * wd + c) * IN_DIM;
      for (int dr = 0; dr < 3; ++dr) {
        for (int dc = 0; dc < 3; ++dc) {
          int pr = r * 3 + dr;
          int pc = c * 3 + dc;
          if (pr >= hp || pc >= wp) continue;   // padded feature-map cell -> stays zero
          const float* src_tok = X_final.data() + (size_t)(pr * wp + pc) * D;
          for (uint32_t ch = 0; ch < D; ++ch) dst_tok[ch * 9 + (dr * 3 + dc)] = src_tok[ch];
        }
      }
    }
  }

  // 5. Aligner MLP: w1 [5120, 9216] -> GELU -> w2 [5120, 5120] -> 5120 output
  std::vector<float> Aligned((size_t)M * LLM_DIM);
  parallel_for(M, 8, [&](uint64_t b0, uint64_t b1) {
    std::vector<float> a1(LLM_DIM);
    for (uint64_t i = b0; i < b1; ++i) {
      const float* x_in = X_down.data() + i * IN_DIM;
      float* out_tok = Aligned.data() + i * LLM_DIM;

      for (uint32_t d = 0; d < LLM_DIM; ++d) {
        float bias = bf16_val(weights_.aligner_w1_b[d]);
        float dot = 0.0f;
        matvec_dot_bf16(weights_.aligner_w1_w + (size_t)d * IN_DIM, x_in, IN_DIM, &dot);
        a1[d] = gelu(bias + dot);
      }

      for (uint32_t d = 0; d < LLM_DIM; ++d) {
        float bias = bf16_val(weights_.aligner_w2_b[d]);
        float dot = 0.0f;
        matvec_dot_bf16(weights_.aligner_w2_w + (size_t)d * LLM_DIM, a1.data(), LLM_DIM, &dot);
        out_tok[d] = bias + dot;
      }
    }
  });

  // 6. Token Packaging:
  // Layout: image_start, then for each row: (wd tokens, then image_newline), then image_end
  const uint32_t total_tokens = 1 + (uint32_t)hd * ((uint32_t)wd + 1) + 1;
  out_embeddings->resize((size_t)total_tokens * LLM_DIM);
  float* dst = out_embeddings->data();

  // Token 0: image_start
  bf16_dequant(weights_.image_start, LLM_DIM, dst);
  dst += LLM_DIM;

  for (int r = 0; r < hd; ++r) {
    for (int c = 0; c < wd; ++c) {
      const float* src = Aligned.data() + ((size_t)r * wd + c) * LLM_DIM;
      std::memcpy(dst, src, LLM_DIM * sizeof(float));
      dst += LLM_DIM;
    }
    // End of row: image_newline
    bf16_dequant(weights_.image_newline, LLM_DIM, dst);
    dst += LLM_DIM;
  }

  // Token Last: image_end
  bf16_dequant(weights_.image_end, LLM_DIM, dst);

  if (n_tokens) *n_tokens = total_tokens;
  return true;
}

bool ViT::process_base64_image(const std::string& b64_str,
                               std::vector<float>* out_embeddings,
                               uint32_t* n_tokens, std::string* err) const {
  std::vector<uint8_t> raw;
  if (!base64_decode(b64_str, &raw)) {
    if (err) *err = "failed to decode base64 image data";
    return false;
  }

  DecodedImage img;
  if (!decode_image_bytes(raw.data(), raw.size(), 576, &img, err)) {
    return false;
  }

  return forward(img, out_embeddings, n_tokens, err);
}

} // namespace aff
