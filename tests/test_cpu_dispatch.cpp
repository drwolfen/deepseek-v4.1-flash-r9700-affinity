#include "engine/cpu_dispatch.h"
#include "engine/expert_kernel.h"
#include "engine/ops.h"
#include "quant/blockquant.h"
#include "quant/source_dtypes.h"
#include "server/server.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

namespace {

using namespace aff;

void check_matvec_bf16_all_tiers(int hw_max) {
  const uint64_t rows = 16;
  const uint64_t cols = 160;
  std::vector<uint16_t> W(rows * cols);
  std::vector<float> x(cols);
  for (uint64_t r = 0; r < rows; ++r) {
    for (uint64_t c = 0; c < cols; ++c) {
      const float v = std::sin(static_cast<float>(r * 17 + c) * 0.13f);
      W[r * cols + c] = float_to_bf16(v);
    }
  }
  for (uint64_t c = 0; c < cols; ++c) x[c] = std::cos(static_cast<float>(c) * 0.21f);

  const int saved_mv = matvec_bf16_variant();
  for (int v : {1, 2}) {
    matvec_bf16_variant(v);
    set_cpu_isa_override(0);
    std::vector<float> y_ref(rows, 0.0f);
    matvec_bf16(W.data(), x.data(), rows, cols, y_ref.data());

    for (int isa = 1; isa <= hw_max; ++isa) {
      set_cpu_isa_override(isa);
      std::vector<float> y_got(rows, 0.0f);
      matvec_bf16(W.data(), x.data(), rows, cols, y_got.data());
      for (uint64_t r = 0; r < rows; ++r) {
        CHECK(std::fabs(y_got[r] - y_ref[r]) < 1e-3f);
      }
    }
  }
  matvec_bf16_variant(saved_mv);
}

void check_matvec_fp8_all_tiers(int hw_max) {
  const uint64_t rows = 8;
  const uint64_t cols = 300;
  const uint64_t sc_cols = (cols + 127) / 128;
  std::vector<uint8_t> W(rows * cols);
  std::vector<uint8_t> S(((rows + 127) / 128) * sc_cols);
  std::vector<float> x(cols);
  for (uint64_t i = 0; i < W.size(); ++i) {
    const uint8_t code = static_cast<uint8_t>((i * 29u + 3u) & 0xFFu);
    W[i] = ((code & 0x7Fu) == 0x7Fu) ? 0x05u : code;  // avoid NaN encodings, exercise subnormals
  }
  for (uint64_t i = 0; i < S.size(); ++i) S[i] = static_cast<uint8_t>(125u + (i % 5u));
  for (uint64_t c = 0; c < cols; ++c) x[c] = std::sin(static_cast<float>(c) * 0.07f);

  DenseW dw;
  dw.data = W.data();
  dw.scales = S.data();
  dw.quant = AFF_FP8_E4M3;

  set_cpu_isa_override(0);
  std::vector<float> y_ref(rows, 0.0f);
  matvec_dense(dw, x.data(), rows, cols, y_ref.data());

  for (int isa = 1; isa <= hw_max; ++isa) {
    set_cpu_isa_override(isa);
    std::vector<float> y_got(rows, 0.0f);
    matvec_dense(dw, x.data(), rows, cols, y_got.data());
    for (uint64_t r = 0; r < rows; ++r) {
      CHECK(std::fabs(y_got[r] - y_ref[r]) < 1e-3f);
    }
  }
}

void check_expert_gemv_all_tiers(int hw_max) {
  const QuantSpec spec = expert_quant_spec(AFF_Q2P875);
  Codebook cb;
  cb.pair = 2u;
  cb.variants = 4u;
  cb.k = 32u;
  cb.v.resize((size_t)cb.variants * cb.k * cb.pair);
  for (uint32_t v = 0; v < cb.variants; ++v) {
    for (uint32_t c = 0; c < cb.k; ++c) {
      cb.v[((size_t)v * cb.k + c) * 2u + 0u] = (static_cast<float>(c) - 15.5f) * (0.05f + 0.01f * v);
      cb.v[((size_t)v * cb.k + c) * 2u + 1u] = (15.5f - static_cast<float>(c)) * (0.04f + 0.015f * v);
    }
  }

  const uint64_t rows = 16, cols = 96;
  std::vector<float> W(rows * cols, 0.0f);
  for (uint64_t r = 0; r < rows; ++r) {
    for (uint64_t c = 0; c < cols; ++c) {
      if (r == 2u && c >= 32u && c < 64u) continue;
      W[r * cols + c] = std::sin(static_cast<float>(r * cols + c) * 0.31f) * 1.1f;
    }
  }
  QuantizedMatrix q;
  std::string err;
  CHECK(quantize_matrix(W.data(), rows, cols, spec, cb, &q, &err));
  const ExpertMatrixView view = view_of(q);

  std::vector<float> x(cols);
  for (uint64_t c = 0; c < cols; ++c) x[c] = std::cos(static_cast<float>(c) * 0.19f);

  set_cpu_isa_override(0);
  const int saved_kv = cpu_kernel_variant();
  std::vector<float> y_ref(rows, 0.5f);
  cpu_kernel_variant(1);
  expert_gemv_scaled(view, cb, x.data(), 0.75f, y_ref.data());

  for (int isa = 0; isa <= hw_max; ++isa) {
    set_cpu_isa_override(isa);
    for (int kv : {1, 2, 4}) {
      cpu_kernel_variant(kv);
      std::vector<float> y_got(rows, 0.5f);
      expert_gemv_scaled(view, cb, x.data(), 0.75f, y_got.data());
      for (uint64_t r = 0; r < rows; ++r) {
        CHECK(std::fabs(y_got[r] - y_ref[r]) < 1e-4f);
      }
    }
  }
  cpu_kernel_variant(saved_kv);
}

void check_blockquant_all_tiers(int hw_max) {
  const QuantSpec spec = expert_quant_spec(AFF_Q2P875);
  Codebook cb;
  cb.pair = 2u;
  cb.variants = 4u;
  cb.k = 32u;
  cb.v.resize((size_t)cb.variants * cb.k * cb.pair);
  for (uint32_t v = 0; v < cb.variants; ++v) {
    for (uint32_t c = 0; c < cb.k; ++c) {
      cb.v[((size_t)v * cb.k + c) * 2u + 0u] = (static_cast<float>(c) - 15.5f) * (0.04f + 0.01f * v);
      cb.v[((size_t)v * cb.k + c) * 2u + 1u] = (15.5f - static_cast<float>(c)) * (0.03f + 0.01f * v);
    }
  }

  const uint64_t rows = 8, cols = 64;
  std::vector<float> W(rows * cols);
  for (uint64_t i = 0; i < W.size(); ++i) W[i] = std::cos(static_cast<float>(i) * 0.41f);

  const bool saved_simd = blockquant_use_simd();
  blockquant_use_simd(false);
  set_cpu_isa_override(0);
  QuantizedMatrix q_ref;
  std::string err;
  CHECK(quantize_matrix(W.data(), rows, cols, spec, cb, &q_ref, &err));

  blockquant_use_simd(true);
  for (int isa = 0; isa <= hw_max; ++isa) {
    set_cpu_isa_override(isa);
    QuantizedMatrix q_got;
    CHECK(quantize_matrix(W.data(), rows, cols, spec, cb, &q_got, &err));
    CHECK(q_got.data == q_ref.data);
    CHECK(q_got.scales == q_ref.scales);
    CHECK(q_got.rscales == q_ref.rscales);
  }
  blockquant_use_simd(saved_simd);
}

void check_json_escape_all_tiers(int hw_max) {
  const std::string clean(96, 'a');
  const std::string dirty = std::string(40, 'x') + "\"hello\"\n\\world\\" + std::string(45, 'z');
  set_cpu_isa_override(0);
  const std::string ref_clean = json_escape(clean);
  const std::string ref_dirty = json_escape(dirty);
  for (int isa = 1; isa <= hw_max; ++isa) {
    set_cpu_isa_override(isa);
    CHECK(json_escape(clean) == ref_clean);
    CHECK(json_escape(dirty) == ref_dirty);
  }
}

}  // namespace

int main() {
  const CpuIsa hw = hw_max_cpu_isa();
  const CpuIsa active = active_cpu_isa();
  std::printf("cpu_dispatch: hw_max=%s active=%s\n", cpu_isa_name(hw), cpu_isa_name(active));

  const int hw_i = static_cast<int>(hw);
  check_matvec_bf16_all_tiers(hw_i);
  check_matvec_fp8_all_tiers(hw_i);
  check_expert_gemv_all_tiers(hw_i);
  check_blockquant_all_tiers(hw_i);
  check_json_escape_all_tiers(hw_i);

  set_cpu_isa_override(-1);
  CHECK(active_cpu_isa() == active);
  std::printf("PASS: test_cpu_dispatch (tiers 0..%d checked)\n", hw_i);
  return 0;
}
