#include "image_decode.h"

#include <cmath>
#include <algorithm>
#include <cstring>

#define STB_IMAGE_STATIC
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace aff {

static const int8_t kB64Table[256] = {
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
  52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
  15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
  -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

bool base64_decode(const std::string& in, std::vector<uint8_t>* out) {
  if (!out) return false;
  out->clear();

  size_t start = 0;
  size_t comma = in.find(',');
  if (comma != std::string::npos && in.rfind("data:", 0) == 0) {
    start = comma + 1;
  }

  out->reserve((in.size() - start) * 3 / 4);
  uint32_t buf = 0;
  int bits = 0;

  for (size_t i = start; i < in.size(); ++i) {
    uint8_t c = (uint8_t)in[i];
    if (c == '=' || c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    int8_t val = kB64Table[c];
    if (val < 0) return false;
    buf = (buf << 6) | (uint32_t)val;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back((uint8_t)((buf >> bits) & 0xFF));
    }
  }
  return !out->empty();
}

// Reference image_processor.py geometry: patch 14, downsample 3, min_pixels 544*544, max 1024 tokens.
static constexpr int kPatch = 14;
static constexpr int kDown = 3;
static constexpr int64_t kMinPixels = 544 * 544;
static constexpr int kMaxTokens = 1024;

static inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

static int64_t llm_dim(int64_t patches) { return ceil_div(patches, kDown); }
static int64_t num_tokens(int64_t nh, int64_t nw) { return nh * (nw + 1) + 2; }

// image_processor.py:solve_resize_ratio -- largest aspect-preserving pixel size whose token grid fits.
static void solve_resize_ratio(double height, double width, int64_t* best_h, int64_t* best_w) {
  const double r = height / width;
  const double max_wf = std::sqrt((kMaxTokens - 2) / r + 0.25) - 0.5;
  const double max_hf = max_wf * r;
  const int64_t cell = (int64_t)kPatch * kDown;
  if (max_wf < 1.0) { *best_h = (kMaxTokens - 2) / 2 * cell; *best_w = cell; return; }
  if (max_hf < 1.0) { *best_h = cell; *best_w = (kMaxTokens - 3) * cell; return; }
  const double beta = std::min(std::floor(max_wf) * cell / width, std::floor(max_hf) * cell / height);
  *best_h = (int64_t)std::floor(height * beta / kPatch) * kPatch;
  *best_w = (int64_t)std::floor(width * beta / kPatch) * kPatch;
}

// image_processor.py:plan_image_grid -- upscale small images to min_pixels, then round to whole patches
// and shrink if the token grid would exceed max_n_token. Returns the padded canvas size.
static void plan_image_grid(int sw, int sh, int* out_w, int* out_h) {
  double w = sw, h = sh;
  if (w * h > 0 && w * h < (double)kMinPixels) {
    const double ratio = std::sqrt((double)kMinPixels / (w * h));
    w = (double)(int64_t)(w * ratio);
    h = (double)(int64_t)(h * ratio);
  }
  int64_t best_w = ceil_div((int64_t)std::ceil(w / kPatch), 1) * kPatch;
  int64_t best_h = ceil_div((int64_t)std::ceil(h / kPatch), 1) * kPatch;
  int64_t nh = llm_dim(best_h / kPatch), nw = llm_dim(best_w / kPatch);
  if (num_tokens(nh, nw) > kMaxTokens) solve_resize_ratio(h, w, &best_h, &best_w);
  *out_w = (int)best_w;
  *out_h = (int)best_h;
}

bool decode_image_bytes(const uint8_t* raw_data, size_t raw_len,
                        int max_tokens,
                        DecodedImage* out, std::string* err) {
  if (!raw_data || !raw_len || !out) {
    if (err) *err = "null or empty image buffer";
    return false;
  }
  (void)max_tokens;

  int sw = 0, sh = 0, channels = 0;
  uint8_t* pixels = stbi_load_from_memory(raw_data, (int)raw_len, &sw, &sh, &channels, 3);
  if (!pixels) {
    if (err) *err = std::string("image decode failed: ") + stbi_failure_reason();
    return false;
  }

  int cw = 0, chh = 0;
  plan_image_grid(sw, sh, &cw, &chh);

  // ImageOps.pad: aspect-preserving fit of the source into the canvas, centered on gray (127,127,127).
  int nw = cw, nh = chh;
  double ratio = (sw > sh) ? (double)cw / sw : (double)chh / sh;
  if ((double)sw > (double)sh) { nw = cw; nh = (int)std::llround(sh * ratio); }
  else                         { nh = chh; nw = (int)std::llround(sw * ratio); }
  nw = std::max(1, std::min(nw, cw));
  nh = std::max(1, std::min(nh, chh));
  const int off_x = (cw - nw) / 2, off_y = (chh - nh) / 2;

  out->width = cw;
  out->height = chh;
  out->h_patches = chh / kPatch;
  out->w_patches = cw / kPatch;
  out->h_down = (int)llm_dim(out->h_patches);
  out->w_down = (int)llm_dim(out->w_patches);

  // Canvas fill = 127 -> (127/255)*2 - 1.
  const float bg = (127.0f / 255.0f) * 2.0f - 1.0f;
  out->data.assign((size_t)cw * chh * 3, bg);

  // Bilinear resample of the source into the centered canvas region, normalized to [-1,1].
  for (int dy = 0; dy < nh; ++dy) {
    float sy = ((dy + 0.5f) * (float)sh / (float)nh) - 0.5f;
    int y0 = (int)std::floor(sy), y1 = y0 + 1;
    float fy = sy - y0;
    y0 = std::clamp(y0, 0, sh - 1);
    y1 = std::clamp(y1, 0, sh - 1);
    for (int dx = 0; dx < nw; ++dx) {
      float sx = ((dx + 0.5f) * (float)sw / (float)nw) - 0.5f;
      int x0 = (int)std::floor(sx), x1 = x0 + 1;
      float fx = sx - x0;
      x0 = std::clamp(x0, 0, sw - 1);
      x1 = std::clamp(x1, 0, sw - 1);
      const float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy);
      const float w10 = (1 - fx) * fy, w11 = fx * fy;
      const uint8_t* p00 = pixels + ((size_t)y0 * sw + x0) * 3;
      const uint8_t* p01 = pixels + ((size_t)y0 * sw + x1) * 3;
      const uint8_t* p10 = pixels + ((size_t)y1 * sw + x0) * 3;
      const uint8_t* p11 = pixels + ((size_t)y1 * sw + x1) * 3;
      float* dst = out->data.data() + ((size_t)(dy + off_y) * cw + (dx + off_x)) * 3;
      for (int c = 0; c < 3; ++c) {
        float v = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
        dst[c] = (v / 255.0f) * 2.0f - 1.0f;
      }
    }
  }

  stbi_image_free(pixels);
  return true;
}

} // namespace aff
