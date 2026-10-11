#include "vision/vit.h"
#include "vision/image_decode.h"
#include "ui/log.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  std::string model_dir = "/mnt/fast-storage/LLM-Models/DeepSeek-V4.1-Flash-UNCENSORED-FP8";
  if (argc > 1) model_dir = argv[1];

  aff::ui::out("aff-vitcheck: loading ViT from %s\n", model_dir.c_str());

  aff::ViT vit;
  std::string err;
  auto t0 = std::chrono::steady_clock::now();
  if (!vit.load(model_dir, &err)) {
    aff::ui::err("failed to load ViT: %s\n", err.c_str());
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  aff::ui::out("ViT loaded successfully in %.1f ms\n", load_ms);

  if (argc > 2) {
    // Image mode: argv[2] is an image file, argv[3] an optional raw-float32 dump path.
    std::ifstream f(argv[2], std::ios::binary);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (raw.empty()) { aff::ui::err("cannot read image %s\n", argv[2]); return 1; }
    aff::DecodedImage img;
    if (!aff::decode_image_bytes(raw.data(), raw.size(), 1024, &img, &err)) {
      aff::ui::err("image decode failed: %s\n", err.c_str());
      return 1;
    }
    aff::ui::out("image %dx%d -> %dx%d patches, %dx%d token grid\n",
                 img.width, img.height, img.h_patches, img.w_patches, img.h_down, img.w_down);
    std::vector<float> emb;
    uint32_t nt = 0;
    auto f0 = std::chrono::steady_clock::now();
    if (!vit.forward(img, &emb, &nt, &err)) {
      aff::ui::err("ViT forward failed: %s\n", err.c_str());
      return 1;
    }
    double fwd = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count();
    aff::ui::out("forward %.1f ms, %u tokens, %zu elements\n", fwd, nt, emb.size());
    if (argc > 3) {
      std::ofstream o(argv[3], std::ios::binary);
      o.write(reinterpret_cast<const char*>(emb.data()), (std::streamsize)(emb.size() * sizeof(float)));
      aff::ui::out("wrote %s\n", argv[3]);
    }
    return 0;
  }

  // Construct a synthetic 378x378 test image (9x9 downsampled tokens = 81 tokens)
  aff::DecodedImage img;
  img.width = 378;
  img.height = 378;
  img.h_patches = img.height / 14; // 27 patches
  img.w_patches = img.width / 14;  // 27 patches
  img.h_down = img.h_patches / 3;  // 9 macro rows
  img.w_down = img.w_patches / 3;  // 9 macro cols
  img.data.resize((size_t)img.width * img.height * 3, 0.0f);

  // Fill with a synthetic gradient pattern
  for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
      size_t idx = ((size_t)y * img.width + x) * 3;
      img.data[idx + 0] = (float)x / (float)img.width * 2.0f - 1.0f; // R
      img.data[idx + 1] = (float)y / (float)img.height * 2.0f - 1.0f; // G
      img.data[idx + 2] = 0.5f;                                       // B
    }
  }

  aff::ui::out("Running ViT forward pass for %dx%d image (%d patches, %dx%d downsampled tokens)...\n",
               img.width, img.height, img.h_patches * img.w_patches, img.h_down, img.w_down);

  std::vector<float> embeddings;
  uint32_t n_tokens = 0;

  auto f0 = std::chrono::steady_clock::now();
  if (!vit.forward(img, &embeddings, &n_tokens, &err)) {
    aff::ui::err("ViT forward failed: %s\n", err.c_str());
    return 1;
  }
  auto f1 = std::chrono::steady_clock::now();
  double fwd_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();

  aff::ui::out("ViT forward completed in %.1f ms!\n", fwd_ms);
  aff::ui::out("Produced %u visual tokens (dim %zu, total elements %zu)\n",
               n_tokens, (size_t)5120, embeddings.size());

  // Expected tokens: 1 (start) + 9 * (9 + 1) (rows + newline) + 1 (end) = 1 + 90 + 1 = 92 tokens
  uint32_t expected_tokens = 1 + img.h_down * (img.w_down + 1) + 1;
  if (n_tokens != expected_tokens) {
    aff::ui::err("token count mismatch: expected %u, got %u\n", expected_tokens, n_tokens);
    return 1;
  }

  // Check for NaN or Inf
  size_t nan_count = 0;
  float min_val = 1e30f, max_val = -1e30f, mean_val = 0.0f;
  for (float v : embeddings) {
    if (std::isnan(v) || std::isinf(v)) ++nan_count;
    if (v < min_val) min_val = v;
    if (v > max_val) max_val = v;
    mean_val += v;
  }
  mean_val /= (float)embeddings.size();

  if (nan_count > 0) {
    aff::ui::err("FATAL: %zu NaN or Inf values detected in embeddings!\n", nan_count);
    return 1;
  }

  aff::ui::out("Sanity check PASSED: 0 NaNs, min=%.4f, max=%.4f, mean=%.4f\n",
               min_val, max_val, mean_val);

  return 0;
}
