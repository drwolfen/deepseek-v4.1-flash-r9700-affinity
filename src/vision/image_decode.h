#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aff {

struct DecodedImage {
  int width = 0;       // multiple of 42
  int height = 0;      // multiple of 42
  int h_patches = 0;   // height / 14
  int w_patches = 0;   // width / 14
  int h_down = 0;      // h_patches / 3
  int w_down = 0;      // w_patches / 3
  std::vector<float> data; // [height * width * 3] normalized to [-1.0f, 1.0f]
};

// Decodes base64 string (handles data URI prefix 'data:image/...;base64,' if present).
bool base64_decode(const std::string& in, std::vector<uint8_t>* out);

// Decodes image bytes (JPEG, PNG, etc.) using stb_image and resizes to optimal macro multiples of 42.
bool decode_image_bytes(const uint8_t* raw_data, size_t raw_len,
                        int max_tokens,
                        DecodedImage* out, std::string* err);

} // namespace aff
