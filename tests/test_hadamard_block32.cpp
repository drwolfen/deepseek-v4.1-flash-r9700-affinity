#include "quant/hadamard.h"
#include "quant/blockquant.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  // 1. Verify (W * H_32^T) * (H_32 * x) == W * x for non-power-of-two cols = 2304
  const uint64_t rows = 16, cols = 2304;
  std::vector<float> W(rows * cols), x(cols);
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = std::sin((float)i * 0.13f) + ((i % 64 == 0) ? 12.5f : 0.0f); // outlier spikes
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::cos((float)i * 0.07f) - ((i % 96 == 0) ? 8.0f : 0.0f);

  std::vector<float> y_ref(rows, 0.0f);
  for (uint64_t r = 0; r < rows; ++r)
    for (uint64_t c = 0; c < cols; ++c)
      y_ref[r] += W[r * cols + c] * x[c];

  // Check incoherence improvement on outlier-heavy matrix
  const aff::IncoherenceStats st_before = aff::incoherence_stats(W.data(), W.size(), 32);
  std::vector<float> W_rot = W, x_rot = x;
  aff::randomized_hadamard_block32_rows(W_rot.data(), rows, cols);
  aff::randomized_hadamard_block32_rows(x_rot.data(), 1, cols);
  const aff::IncoherenceStats st_after = aff::incoherence_stats(W_rot.data(), W_rot.size(), 32);

  if (st_after.mean_block_max_over_rms >= st_before.mean_block_max_over_rms) {
    std::fprintf(stderr, "FAIL: block32 Hadamard did not reduce mean_block_max_over_rms (%.3f -> %.3f)\n",
                 st_before.mean_block_max_over_rms, st_after.mean_block_max_over_rms);
    return 1;
  }

  // Verify dot product preservation: (W_rot) . (x_rot) == W . x
  for (uint64_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint64_t c = 0; c < cols; ++c) acc += W_rot[r * cols + c] * x_rot[c];
    const float diff = std::fabs(acc - y_ref[r]);
    const float rel = diff / (std::fabs(y_ref[r]) + 1.0f);
    if (rel > 1e-4f) {
      std::fprintf(stderr, "FAIL: row %llu dot mismatch: got %f ref %f (diff %g, rel %g)\n",
                   (unsigned long long)r, acc, y_ref[r], diff, rel);
      return 1;
    }
  }

  std::printf("PASS: test_hadamard_block32 (block max/rms %.2f -> %.2f)\n",
              st_before.mean_block_max_over_rms, st_after.mean_block_max_over_rms);
  return 0;
}
