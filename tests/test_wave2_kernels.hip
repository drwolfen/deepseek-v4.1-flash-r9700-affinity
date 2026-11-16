#include "gpu/batch_kernels.h"
#include "gpu/expert_kernel_hip.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16_cpu(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32_cpu(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int main() {
  using namespace aff;

  // 1. Verify Tail Skip bounds across token counts (Avalon 9e4f26f)
  {
    const uint32_t test_nt[] = {1, 4, 16, 24, 64};
    for (uint32_t nt : test_nt) {
      bool ts = moe_tail_skip_enabled(nt);
      if (nt <= 16) {
        assert(ts == true);
      } else {
        assert(ts == false); // Enforce Avalon 9e4f26f: disable TAIL_SKIP whenever n_tokens > 16
      }
    }
    std::printf("PASS: moe_tail_skip_enabled bounds <=16 and >16\n");
  }

  // Check GPU device count
  int dev_count = 0;
  hipError_t err_cnt = hipGetDeviceCount(&dev_count);
  if (err_cnt != hipSuccess || dev_count <= 0) {
    std::printf("PASS: test_wave2_kernels offline checks (0 GPU devices available in ephemeral environment)\n");
    return 0;
  }

  std::printf("Found %d HIP device(s), proceeding with GPU kernel tests on RDNA4 gfx1201\n", dev_count);

  // 2. Dual-GPU portable mapped host embed_tokens gather vs device VRAM gather
  {
    const uint32_t vocab = 1024, n_embd = 512, nt = 16;
    const size_t nbytes = (size_t)vocab * n_embd * sizeof(uint16_t);

    uint16_t* h_pinned = nullptr;
    assert(hipHostMalloc((void**)&h_pinned, nbytes, hipHostMallocMapped | hipHostMallocPortable) == hipSuccess);
    for (size_t i = 0; i < (size_t)vocab * n_embd; ++i) {
      h_pinned[i] = f32_to_bf16_cpu(std::sin((float)i * 0.013f) * 2.0f);
    }

    std::vector<uint32_t> h_tok(nt);
    for (uint32_t t = 0; t < nt; ++t) h_tok[t] = (t * 61 + 17) % vocab;

    const int gpus_to_test = std::min(dev_count, 2);
    for (int dev = 0; dev < gpus_to_test; ++dev) {
      assert(hipSetDevice(dev) == hipSuccess);

      void* d_uva_ptr = nullptr;
      assert(hipHostGetDevicePointer(&d_uva_ptr, h_pinned, 0) == hipSuccess);
      assert(d_uva_ptr != nullptr);

      uint16_t* d_vram = nullptr;
      assert(hipMalloc(&d_vram, nbytes) == hipSuccess);
      assert(hipMemcpy(d_vram, h_pinned, nbytes, hipMemcpyHostToDevice) == hipSuccess);

      uint32_t* d_tok = nullptr;
      assert(hipMalloc(&d_tok, nt * sizeof(uint32_t)) == hipSuccess);
      assert(hipMemcpy(d_tok, h_tok.data(), nt * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);

      float *d_out_uva = nullptr, *d_out_vram = nullptr;
      assert(hipMalloc(&d_out_uva, (size_t)nt * n_embd * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out_vram, (size_t)nt * n_embd * sizeof(float)) == hipSuccess);

      embed_tokens_gather_hip((const uint16_t*)d_uva_ptr, d_tok, d_out_uva, nt, n_embd, nullptr);
      embed_tokens_gather_hip(d_vram, d_tok, d_out_vram, nt, n_embd, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<float> h_out_uva((size_t)nt * n_embd), h_out_vram((size_t)nt * n_embd);
      assert(hipMemcpy(h_out_uva.data(), d_out_uva, h_out_uva.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_out_vram.data(), d_out_vram, h_out_vram.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      float max_abs_err = 0.0f;
      for (size_t i = 0; i < h_out_uva.size(); ++i) {
        float err = std::fabs(h_out_uva[i] - h_out_vram[i]);
        if (err > max_abs_err) max_abs_err = err;
      }
      assert(max_abs_err == 0.0f); // Bit-exact parity required

      (void)hipFree(d_vram);
      (void)hipFree(d_tok);
      (void)hipFree(d_out_uva);
      (void)hipFree(d_out_vram);
    }
    (void)hipHostFree(h_pinned);
    std::printf("PASS: embed_tokens UVA gather vs VRAM gather on %d GPU(s) (max_abs_err == 0)\n", gpus_to_test);
  }

  // 3. Hyper-Connection refresh: fused v2 and chained decode vs CPU reference
  {
    assert(hipSetDevice(0) == hipSuccess);
    const long test_rows[] = {1, 4, 8, 16};
    const long n = 2560;
    const int hc = 4;

    for (long rows : test_rows) {
      const long total_h = rows * hc * n;
      std::vector<uint16_t> h_in(total_h);
      std::vector<float> w_in(n);
      for (long i = 0; i < total_h; ++i) h_in[i] = f32_to_bf16_cpu(std::sin((float)i * 0.003f) * 1.5f);
      for (long i = 0; i < n; ++i) w_in[i] = 1.0f + 0.1f * std::cos((float)i * 0.01f);

      std::vector<float> ref_norm(rows * n);
      for (long r = 0; r < rows; ++r) {
        for (long i = 0; i < n; ++i) {
          float sum = 0.0f;
          for (int c = 0; c < hc; ++c) {
            sum += bf16_to_f32_cpu(h_in[r * hc * n + c * n + i]);
          }
          sum *= (1.0f / (float)hc);
          sum *= w_in[i];
          ref_norm[r * n + i] = sum;
        }
      }

      uint16_t* d_h = nullptr;
      float* d_w = nullptr;
      float* d_out_norm_v2 = nullptr;
      float* d_out_norm_ch = nullptr;
      unsigned short* d_out_v2 = nullptr;
      unsigned short* d_out_ch = nullptr;
      uint32_t* d_sync = nullptr;

      assert(hipMalloc(&d_h, total_h * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_w, n * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out_norm_v2, rows * n * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out_norm_ch, rows * n * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_out_v2, rows * n * sizeof(unsigned short)) == hipSuccess);
      assert(hipMalloc(&d_out_ch, rows * n * sizeof(unsigned short)) == hipSuccess);
      assert(hipMalloc(&d_sync, 16 * sizeof(uint32_t)) == hipSuccess);
      assert(hipMemset(d_sync, 0, 16 * sizeof(uint32_t)) == hipSuccess);

      assert(hipMemcpy(d_h, h_in.data(), total_h * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_w, w_in.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

      batch_hc_read_fused_v2_hip(d_h, d_w, nullptr, 0, nullptr, d_out_v2, nullptr, nullptr, nullptr, 0,
                                 nullptr, d_out_norm_v2, rows, n, hc, 0, hc * n, 0.0f, nullptr, d_sync);
      assert(hipDeviceSynchronize() == hipSuccess);

      assert(hipMemset(d_sync, 0, 16 * sizeof(uint32_t)) == hipSuccess);
      batch_hc_chained_decode_hip(d_h, d_w, nullptr, 0, nullptr, 0, d_out_ch, d_out_norm_ch,
                                  rows, n, hc, 0, nullptr, d_sync, /*force_chained=*/true);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<float> got_v2(rows * n), got_ch(rows * n);
      assert(hipMemcpy(got_v2.data(), d_out_norm_v2, rows * n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(got_ch.data(), d_out_norm_ch, rows * n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      auto check_sim = [&](const std::vector<float>& got, const char* name) {
        double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
        float max_abs_err = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
          double a = ref_norm[i], b = got[i];
          dot_ab += a * b;
          norm_a += a * a;
          norm_b += b * b;
          float diff = std::fabs(got[i] - ref_norm[i]);
          if (diff > max_abs_err) max_abs_err = diff;
        }
        double cos_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
        std::printf("HC %s (rows=%ld): cos_sim=%f, max_abs_err=%e\n", name, rows, cos_sim, max_abs_err);
        assert(cos_sim > 0.99999);
        assert(max_abs_err < 1e-4f);
      };

      check_sim(got_v2, "Fused V2");
      check_sim(got_ch, "Chained");

      // Verify dispatched path as well
      assert(hipMemset(d_sync, 0, 16 * sizeof(uint32_t)) == hipSuccess);
      batch_hc_chained_decode_hip(d_h, d_w, nullptr, 0, nullptr, 0, d_out_ch, d_out_norm_ch,
                                  rows, n, hc, 0, nullptr, d_sync, /*force_chained=*/false);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(got_ch.data(), d_out_norm_ch, rows * n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      check_sim(got_ch, "Dispatched");

      (void)hipFree(d_h);
      (void)hipFree(d_w);
      (void)hipFree(d_out_norm_v2);
      (void)hipFree(d_out_norm_ch);
      (void)hipFree(d_out_v2);
      (void)hipFree(d_out_ch);
      (void)hipFree(d_sync);
    }
    std::printf("PASS: Hyper-Connection fused V2 & chained decode kernels across n_rows={1, 4, 8, 16}\n");
  }

  // 4. Summed MoE Down GEMM across n_tokens = {1, 4, 16, 24, 64}
  {
    assert(hipSetDevice(0) == hipSuccess);
    const uint32_t dim = 512;
    const uint32_t test_nt[] = {1, 4, 16, 24, 64};

    for (uint32_t nt : test_nt) {
      const uint32_t nb = nt;
      const uint32_t nb_stride = nb;
      const uint32_t total = nt * 4; // 4 slots per token

      std::vector<uint16_t> h_y((size_t)dim * total);
      for (size_t i = 0; i < h_y.size(); ++i) h_y[i] = f32_to_bf16_cpu(std::sin((float)i * 0.021f) * 1.2f);

      std::vector<uint32_t> h_ord(total);
      for (uint32_t i = 0; i < total; ++i) h_ord[i] = i;

      std::vector<uint32_t> h_run(nt + 1);
      for (uint32_t t = 0; t <= nt; ++t) h_run[t] = t * 4;

      std::vector<float> ref_block((size_t)dim * nb, 0.0f);
      for (uint32_t d = 0; d < dim; ++d) {
        for (uint32_t t = 0; t < nt; ++t) {
          float acc = 0.0f;
          for (uint32_t i = h_run[t]; i < h_run[t + 1]; ++i) {
            acc += bf16_to_f32_cpu(h_y[(size_t)d * total + h_ord[i]]);
          }
          ref_block[(size_t)d * nb_stride + t] = acc;
        }
      }

      uint16_t* d_y = nullptr;
      float* d_block = nullptr;
      uint32_t *d_ord = nullptr, *d_run = nullptr;

      assert(hipMalloc(&d_y, h_y.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_block, (size_t)dim * nb_stride * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_ord, total * sizeof(uint32_t)) == hipSuccess);
      assert(hipMalloc(&d_run, (nt + 1) * sizeof(uint32_t)) == hipSuccess);

      // Fill d_block with poison values to verify first_pass writes directly without memset
      std::vector<float> poison((size_t)dim * nb_stride, -9999.0f);
      assert(hipMemcpy(d_block, poison.data(), poison.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_y, h_y.data(), h_y.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_ord, h_ord.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_run, h_run.data(), (nt + 1) * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);

      expert_scatter_slots_summed_hip(d_y, d_block, d_ord, d_run, dim, nt, nb_stride, total,
                                      /*first_pass=*/true, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<float> got((size_t)dim * nb_stride);
      assert(hipMemcpy(got.data(), d_block, got.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      double dot_ab = 0.0, norm_a = 0.0, norm_b = 0.0;
      for (size_t i = 0; i < ref_block.size(); ++i) {
        double a = ref_block[i], b = got[i];
        dot_ab += a * b;
        norm_a += a * a;
        norm_b += b * b;
      }
      double cos_sim = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
      assert(cos_sim > 0.99999);

      (void)hipFree(d_y);
      (void)hipFree(d_block);
      (void)hipFree(d_ord);
      (void)hipFree(d_run);
    }
    std::printf("PASS: Summed MoE Down GEMM across n_tokens = {1, 4, 16, 24, 64}\n");
  }

  std::printf("ALL WAVE 2 PHASE 2B KERNEL TESTS PASSED.\n");
  return 0;
}
