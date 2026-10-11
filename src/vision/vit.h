#pragma once

#include "image_decode.h"
#include "model/safetensors.h"

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

struct ViTBlockWeights {
  const uint16_t* norm1_w = nullptr;     // [1024]
  const uint16_t* attn_wqkv_w = nullptr; // [3072, 1024]
  const uint16_t* attn_wqkv_b = nullptr; // [3072]
  const uint16_t* attn_wo_w = nullptr;   // [1024, 1024]
  const uint16_t* attn_wo_b = nullptr;   // [1024]
  const uint16_t* norm2_w = nullptr;     // [1024]
  const uint16_t* mlp_w1_w = nullptr;    // [5632, 1024] (SwiGLU gate+up)
  const uint16_t* mlp_w2_w = nullptr;    // [1024, 2816]
};

struct ViTWeights {
  const uint16_t* patch_proj_w = nullptr; // [1024, 588]
  const uint16_t* patch_proj_b = nullptr; // [1024]
  std::vector<ViTBlockWeights> blocks;    // 32 layers
  const uint16_t* final_norm_w = nullptr; // [1024]
  const uint16_t* aligner_w1_w = nullptr; // [5120, 9216]
  const uint16_t* aligner_w1_b = nullptr; // [5120]
  const uint16_t* aligner_w2_w = nullptr; // [5120, 5120]
  const uint16_t* aligner_w2_b = nullptr; // [5120]
  const uint16_t* image_start = nullptr;  // [5120]
  const uint16_t* image_end = nullptr;    // [5120]
  const uint16_t* image_newline = nullptr;// [5120]
};

class ViT {
public:
  ViT() = default;
  ~ViT() = default;

  bool load(const std::string& model_dir, std::string* err);
  bool ready() const { return ready_; }

  // Forward pass: takes decoded image, runs patch embedding, 32 ViT blocks,
  // 3x3 pixel unshuffle downsample, aligner MLP, and packages with image_start,
  // image_newline, image_end tokens.
  // Outputs `n_tokens` embeddings of dimension 5120 into `out_embeddings`.
  bool forward(const DecodedImage& img, std::vector<float>* out_embeddings,
               uint32_t* n_tokens, std::string* err) const;

  // Convenience helper: decodes base64 string and runs ViT forward pass.
  bool process_base64_image(const std::string& b64_str,
                            std::vector<float>* out_embeddings,
                            uint32_t* n_tokens, std::string* err) const;

private:
  SafeTensorsFile shard1_;
  SafeTensorsFile shard2_;
  ViTWeights weights_;
  bool ready_ = false;
};

} // namespace aff
