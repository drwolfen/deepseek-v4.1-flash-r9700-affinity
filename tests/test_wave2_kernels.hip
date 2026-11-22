#include "gpu/batch_kernels.h"
#include "gpu/expert_kernel_hip.h"
#include "gpu/tp_allreduce.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static float e4m3_to_f32_cpu(uint8_t b) {
  const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
  const uint32_t ex = (b >> 3) & 0x0Fu;
  const uint32_t mn = b & 0x07u;
  if (ex == 15u && mn == 7u) return 0.0f;
  if (ex == 0u) {
    const float mag = (float)mn * (1.0f / 512.0f);
    return (b & 0x80u) ? -mag : mag;
  }
  const uint32_t u = sign | ((ex + 120u) << 23) | (mn << 20);
  float out = 0.0f;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

static float ue8m0_to_f32_cpu(uint8_t e) {
  if (e == 0u) return 0.0f;
  const uint32_t u = (uint32_t)e << 23;
  float out = 0.0f;
  std::memcpy(&out, &u, sizeof(out));
  return out;
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

  // 1b. Offline FP8 reference codec + narrow wave selector checks (no GPU required).
  {
    assert(e4m3_to_f32_cpu(0x38u) == 1.0f);
    assert(e4m3_to_f32_cpu(0xB8u) == -1.0f);
    assert(e4m3_to_f32_cpu(0x01u) == 1.0f / 512.0f);
    assert(e4m3_to_f32_cpu(0x7Eu) == 448.0f);
    assert(ue8m0_to_f32_cpu(127u) == 1.0f);
    assert(ue8m0_to_f32_cpu(128u) == 2.0f);
    assert(ue8m0_to_f32_cpu(126u) == 0.5f);
    assert(fp8_narrow_mw(5120u) >= 1u);
    assert(fp8_narrow_mw(1024u) >= 1u);
    std::printf("PASS: Offline FP8 E4M3/UE8M0 reference codec + fp8_narrow_mw check\n");
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

  // ---- Wave B1 + B2: FP8 prefill 4x2 + narrow inline split-K parity checks ------------------
  {
    assert(hipSetDevice(0) == hipSuccess);
      auto make_e4m3 = [](uint64_t idx) -> uint8_t {
        const uint32_t h = (uint32_t)(idx * 2654435761ull ^ (idx >> 13));
        const uint32_t ex = 4u + ((h >> 3) & 3u);
        const uint32_t mn = h & 7u;
        const uint8_t sign = (uint8_t)(h & 0x80u);
        return (uint8_t)(sign | (ex << 3) | mn);
      };
      auto make_ue8m0 = [](uint64_t idx) -> uint8_t {
        return (uint8_t)(126u + ((idx * 17u + 3u) % 3u));
      };
      auto run_fp8_case = [&](uint32_t rows, uint32_t cols, uint32_t nb, bool preshuffled,
                              bool accum, bool repeat_same_stream) {
        const uint32_t sc_rows = (rows + 127u) / 128u;
        const uint32_t sc_cols = cols / 128u;
        std::vector<uint8_t> h_w((size_t)rows * cols);
        std::vector<uint8_t> h_sw((size_t)sc_rows * sc_cols);
        std::vector<uint8_t> h_x((size_t)nb * cols);
        std::vector<float> h_sx((size_t)nb * sc_cols);
        std::vector<float> h_y_init((size_t)rows * nb, 0.0f);
        std::vector<float> h_y_gpu((size_t)rows * nb, 0.0f);

        for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = make_e4m3(i + 17u);
        for (size_t i = 0; i < h_sw.size(); ++i) h_sw[i] = make_ue8m0(i + 5u);
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = make_e4m3(i + 1000003u);
        for (uint32_t b = 0; b < nb; ++b) {
          for (uint32_t kb = 0; kb < sc_cols; ++kb) {
            h_sx[(size_t)b * sc_cols + kb] = 0.25f * (float)(1u + ((b * 7u + kb * 3u) % 5u));
          }
        }
        if (accum) {
          for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t b = 0; b < nb; ++b) {
              h_y_init[(size_t)r * nb + b] = 0.125f * (float)((int)((r * 13u + b * 5u) % 19u) - 9);
            }
          }
        }

        uint8_t* d_w = nullptr;
        uint8_t* d_w_ps = nullptr;
        uint8_t* d_sw = nullptr;
        uint8_t* d_x = nullptr;
        float* d_sx = nullptr;
        float* d_y = nullptr;
        assert(hipMalloc(&d_w, h_w.size()) == hipSuccess);
        assert(hipMalloc(&d_sw, h_sw.size()) == hipSuccess);
        assert(hipMalloc(&d_x, h_x.size()) == hipSuccess);
        assert(hipMalloc(&d_sx, h_sx.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y, h_y_init.size() * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_w, h_w.data(), h_w.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw, h_sw.data(), h_sw.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_x, h_x.data(), h_x.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sx, h_sx.data(), h_sx.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_y, h_y_init.data(), h_y_init.size() * sizeof(float),
                         hipMemcpyHostToDevice) == hipSuccess);

        const uint8_t* w_arg = d_w;
        if (preshuffled) {
          assert(hipMalloc(&d_w_ps, h_w.size()) == hipSuccess);
          aff::preshuffle_fp8_hip(d_w, d_w_ps, rows, cols, nullptr);
          w_arg = d_w_ps;
        }

        aff::batch_gemm_fp8_hip(w_arg, d_sw, d_x, d_sx, d_y, rows, cols, nb, 0u, 0u, nullptr,
                                preshuffled, 0u, 0u, accum, 0u);
        if (repeat_same_stream) {
          assert(hipMemcpy(d_y, h_y_init.data(), h_y_init.size() * sizeof(float),
                           hipMemcpyHostToDevice) == hipSuccess);
          aff::batch_gemm_fp8_hip(w_arg, d_sw, d_x, d_sx, d_y, rows, cols, nb, 0u, 0u, nullptr,
                                  preshuffled, 0u, 0u, accum, 0u);
        }
        assert(hipDeviceSynchronize() == hipSuccess);
        assert(hipMemcpy(h_y_gpu.data(), d_y, h_y_gpu.size() * sizeof(float),
                         hipMemcpyDeviceToHost) == hipSuccess);

        std::vector<uint32_t> check_rows;
        if (rows <= 1536u || nb <= 16u) {
          check_rows.resize(rows);
          for (uint32_t r = 0; r < rows; ++r) check_rows[r] = r;
        } else {
          for (uint32_t r = 0; r < rows; r += 17u) check_rows.push_back(r);
          const uint32_t extra[] = {127u, 128u, 255u, 256u, rows - 1u};
          for (uint32_t er : extra) {
            if (er < rows) check_rows.push_back(er);
          }
        }

        double max_abs_err = 0.0;
        double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
        for (uint32_t r : check_rows) {
          const uint32_t sr = r >> 7;
          for (uint32_t b = 0; b < nb; ++b) {
            float ref = accum ? h_y_init[(size_t)r * nb + b] : 0.0f;
            for (uint32_t kb = 0; kb < sc_cols; ++kb) {
              float blk = 0.0f;
              for (uint32_t k = 0; k < 128u; ++k) {
                const float wf = e4m3_to_f32_cpu(h_w[(size_t)r * cols + kb * 128u + k]);
                const float xf = e4m3_to_f32_cpu(h_x[(size_t)b * cols + kb * 128u + k]);
                blk += wf * xf;
              }
              const float ws = ue8m0_to_f32_cpu(h_sw[(size_t)sr * sc_cols + kb]);
              const float xs = h_sx[(size_t)b * sc_cols + kb];
              ref += blk * (ws * xs);
            }
            const double g = (double)h_y_gpu[(size_t)r * nb + b];
            const double rf = (double)ref;
            const double err = std::fabs(g - rf);
            if (err > max_abs_err) max_abs_err = err;
            dot += g * rf;
            norm_ref += rf * rf;
            norm_gpu += g * g;
          }
        }
        const double cos_sim = dot / (std::sqrt(norm_ref) * std::sqrt(norm_gpu) + 1e-12);
        std::printf("  fp8(%u,%u,%u ps=%d acc=%d rep=%d): max_abs_err=%.6e cos=%.9f\n", rows,
                    cols, nb, (int)preshuffled, (int)accum, (int)repeat_same_stream, max_abs_err,
                    cos_sim);
        assert(max_abs_err <= 2e-3);
        assert(cos_sim > 0.999999);

        (void)hipFree(d_w);
        if (d_w_ps) (void)hipFree(d_w_ps);
        (void)hipFree(d_sw);
        (void)hipFree(d_x);
        (void)hipFree(d_sx);
        (void)hipFree(d_y);
      };

      // B1 prefill large-N checks (Full and tail N, row-major and preshuffled).
      run_fp8_case(5120u, 1536u, 256u, false, false, false);
      run_fp8_case(5120u, 1536u, 256u, true, false, false);
      run_fp8_case(5120u, 1536u, 512u, true, false, false);
      run_fp8_case(5120u, 1536u, 260u, false, false, false);
      run_fp8_case(5120u, 1536u, 260u, true, false, false);

      // B2 narrow decode + split-K (accum=false and accum=true + consecutive counter reset).
      run_fp8_case(5120u, 1152u, 1u, false, false, false);
      run_fp8_case(5120u, 1152u, 6u, false, true, false);
      run_fp8_case(1024u, 1152u, 1u, false, false, true);
      run_fp8_case(1024u, 1152u, 6u, false, true, true);
      run_fp8_case(1024u, 1152u, 6u, true, true, true);
      run_fp8_case(1536u, 1152u, 4u, false, true, true);
      run_fp8_case(1024u, 1536u, 5u, false, true, true);

      // B2 grouped multi-job narrow + split-K + counter reset check.
      for (int ps_mode = 0; ps_mode <= 1; ++ps_mode) {
        const bool preshuffled = (ps_mode != 0);
        const uint32_t jrows[2] = {512u, 512u};
        const uint32_t cols = 1152u, nb = 4u, total = jrows[0] + jrows[1];
        const uint32_t sc_cols = cols / 128u;
        std::vector<uint8_t> h_w0((size_t)jrows[0] * cols), h_w1((size_t)jrows[1] * cols);
        std::vector<uint8_t> h_sw0((size_t)(jrows[0] / 128u) * sc_cols),
            h_sw1((size_t)(jrows[1] / 128u) * sc_cols);
        std::vector<uint8_t> h_x((size_t)nb * cols);
        std::vector<float> h_sx((size_t)nb * sc_cols);
        std::vector<float> h_y_gpu((size_t)total * nb, 0.0f);
        for (size_t i = 0; i < h_w0.size(); ++i) h_w0[i] = make_e4m3(i + 31u);
        for (size_t i = 0; i < h_w1.size(); ++i) h_w1[i] = make_e4m3(i + 99991u);
        for (size_t i = 0; i < h_sw0.size(); ++i) h_sw0[i] = make_ue8m0(i + 1u);
        for (size_t i = 0; i < h_sw1.size(); ++i) h_sw1[i] = make_ue8m0(i + 7u);
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = make_e4m3(i + 500009u);
        for (uint32_t b = 0; b < nb; ++b)
          for (uint32_t kb = 0; kb < sc_cols; ++kb)
            h_sx[(size_t)b * sc_cols + kb] = 0.5f * (float)(1u + ((b + kb) % 3u));

        uint8_t *d_w0 = nullptr, *d_w1 = nullptr, *d_ps0 = nullptr, *d_ps1 = nullptr;
        uint8_t *d_sw0 = nullptr, *d_sw1 = nullptr, *d_x = nullptr;
        float *d_sx = nullptr, *d_y = nullptr;
        assert(hipMalloc(&d_w0, h_w0.size()) == hipSuccess);
        assert(hipMalloc(&d_w1, h_w1.size()) == hipSuccess);
        assert(hipMalloc(&d_sw0, h_sw0.size()) == hipSuccess);
        assert(hipMalloc(&d_sw1, h_sw1.size()) == hipSuccess);
        assert(hipMalloc(&d_x, h_x.size()) == hipSuccess);
        assert(hipMalloc(&d_sx, h_sx.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y, h_y_gpu.size() * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_w0, h_w0.data(), h_w0.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_w1, h_w1.data(), h_w1.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw0, h_sw0.data(), h_sw0.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw1, h_sw1.data(), h_sw1.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_x, h_x.data(), h_x.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sx, h_sx.data(), h_sx.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemset(d_y, 0, h_y_gpu.size() * sizeof(float)) == hipSuccess);
        if (preshuffled) {
          assert(hipMalloc(&d_ps0, h_w0.size()) == hipSuccess);
          assert(hipMalloc(&d_ps1, h_w1.size()) == hipSuccess);
          aff::preshuffle_fp8_hip(d_w0, d_ps0, jrows[0], cols, nullptr);
          aff::preshuffle_fp8_hip(d_w1, d_ps1, jrows[1], cols, nullptr);
        }

        aff::BatchGemmJob jobs[2] = {
            {preshuffled ? d_ps0 : d_w0, d_sw0, d_y, jrows[0], 0u, 0u},
            {preshuffled ? d_ps1 : d_w1, d_sw1, d_y + (size_t)jrows[0] * nb, jrows[1], 0u, 0u},
        };
        assert(aff::batch_gemm_fp8_multi_hip(jobs, 2u, d_x, d_sx, cols, nb, 0u, 0u, nullptr,
                                             preshuffled));
        assert(aff::batch_gemm_fp8_multi_hip(jobs, 2u, d_x, d_sx, cols, nb, 0u, 0u, nullptr,
                                             preshuffled));
        assert(hipDeviceSynchronize() == hipSuccess);
        assert(hipMemcpy(h_y_gpu.data(), d_y, h_y_gpu.size() * sizeof(float),
                         hipMemcpyDeviceToHost) == hipSuccess);

        double max_abs_err = 0.0;
        for (uint32_t ji = 0; ji < 2u; ++ji) {
          const auto& hw = (ji == 0u) ? h_w0 : h_w1;
          const auto& hsw = (ji == 0u) ? h_sw0 : h_sw1;
          const size_t y_off = (ji == 0u) ? 0u : (size_t)jrows[0] * nb;
          for (uint32_t r = 0; r < jrows[ji]; ++r) {
            const uint32_t sr = r >> 7;
            for (uint32_t b = 0; b < nb; ++b) {
              float ref = 0.0f;
              for (uint32_t kb = 0; kb < sc_cols; ++kb) {
                float blk = 0.0f;
                for (uint32_t k = 0; k < 128u; ++k) {
                  blk += e4m3_to_f32_cpu(hw[(size_t)r * cols + kb * 128u + k]) *
                         e4m3_to_f32_cpu(h_x[(size_t)b * cols + kb * 128u + k]);
                }
                ref += blk * (ue8m0_to_f32_cpu(hsw[(size_t)sr * sc_cols + kb]) *
                              h_sx[(size_t)b * sc_cols + kb]);
              }
              const double err = std::fabs((double)h_y_gpu[y_off + (size_t)r * nb + b] - (double)ref);
              if (err > max_abs_err) max_abs_err = err;
            }
          }
        }
        std::printf("  fp8_multi(512+512, 1152, 4, ps=%d): max_abs_err=%.6e\n", (int)preshuffled,
                    max_abs_err);
        assert(max_abs_err <= 2e-3);

        (void)hipFree(d_w0);
        (void)hipFree(d_w1);
        if (d_ps0) (void)hipFree(d_ps0);
        if (d_ps1) (void)hipFree(d_ps1);
        (void)hipFree(d_sw0);
        (void)hipFree(d_sw1);
        (void)hipFree(d_x);
        (void)hipFree(d_sx);
        (void)hipFree(d_y);
      }

      std::printf("PASS: FP8 prefill 4x2 (B1) + inline split-K parity (B2)\n");

      // Regression: contiguous-band multi-GEMM with jobs[1].y == nullptr (dense_gpu.hip pattern)
      // Covers (a) batch_gemm_fp8_multi_hip ksplit==1 (Layer 0 attn_qkv "hi" shape: 1536+576, 5120, 16)
      // and (b) batch_gemm_fp8_multi_hip ksplit>1 (256+256, 5120, 16) with inline split-K fence.
      auto run_fp8_multi_null_band = [&](uint32_t r0, uint32_t r1, uint32_t cols, uint32_t nb,
                                         bool preshuffled) {
        const uint32_t jrows[2] = {r0, r1};
        const uint32_t total = r0 + r1;
        const uint32_t sc_cols = cols / 128u;
        std::vector<uint8_t> h_w0((size_t)r0 * cols), h_w1((size_t)r1 * cols);
        std::vector<uint8_t> h_sw0((size_t)((r0 + 127u) / 128u) * sc_cols),
            h_sw1((size_t)((r1 + 127u) / 128u) * sc_cols);
        std::vector<uint8_t> h_x((size_t)nb * cols);
        std::vector<float> h_sx((size_t)nb * sc_cols);
        std::vector<float> h_y_gpu((size_t)total * nb, 0.0f);

        for (size_t i = 0; i < h_w0.size(); ++i) h_w0[i] = make_e4m3(i + 41u);
        for (size_t i = 0; i < h_w1.size(); ++i) h_w1[i] = make_e4m3(i + 131071u);
        for (size_t i = 0; i < h_sw0.size(); ++i) h_sw0[i] = make_ue8m0(i + 3u);
        for (size_t i = 0; i < h_sw1.size(); ++i) h_sw1[i] = make_ue8m0(i + 11u);
        for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = make_e4m3(i + 700001u);
        for (uint32_t b = 0; b < nb; ++b)
          for (uint32_t kb = 0; kb < sc_cols; ++kb)
            h_sx[(size_t)b * sc_cols + kb] = 0.5f * (float)(1u + ((b * 3u + kb) % 3u));

        uint8_t *d_w0 = nullptr, *d_w1 = nullptr, *d_ps0 = nullptr, *d_ps1 = nullptr;
        uint8_t *d_sw0 = nullptr, *d_sw1 = nullptr, *d_x = nullptr;
        float *d_sx = nullptr, *d_y = nullptr;
        assert(hipMalloc(&d_w0, h_w0.size()) == hipSuccess);
        assert(hipMalloc(&d_w1, h_w1.size()) == hipSuccess);
        assert(hipMalloc(&d_sw0, h_sw0.size()) == hipSuccess);
        assert(hipMalloc(&d_sw1, h_sw1.size()) == hipSuccess);
        assert(hipMalloc(&d_x, h_x.size()) == hipSuccess);
        assert(hipMalloc(&d_sx, h_sx.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y, h_y_gpu.size() * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_w0, h_w0.data(), h_w0.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_w1, h_w1.data(), h_w1.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw0, h_sw0.data(), h_sw0.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sw1, h_sw1.data(), h_sw1.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_x, h_x.data(), h_x.size(), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_sx, h_sx.data(), h_sx.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemset(d_y, 0, h_y_gpu.size() * sizeof(float)) == hipSuccess);
        if (preshuffled) {
          assert(hipMalloc(&d_ps0, h_w0.size()) == hipSuccess);
          assert(hipMalloc(&d_ps1, h_w1.size()) == hipSuccess);
          aff::preshuffle_fp8_hip(d_w0, d_ps0, r0, cols, nullptr);
          aff::preshuffle_fp8_hip(d_w1, d_ps1, r1, cols, nullptr);
        }

        aff::BatchGemmJob jobs[2] = {
            {preshuffled ? d_ps0 : d_w0, d_sw0, d_y, r0, 0u, 0u},
            {preshuffled ? d_ps1 : d_w1, d_sw1, nullptr, r1, 0u, 0u},
        };
        assert(aff::batch_gemm_fp8_multi_hip(jobs, 2u, d_x, d_sx, cols, nb, 0u, 0u, nullptr,
                                             preshuffled));
        assert(hipGetLastError() == hipSuccess);
        assert(hipDeviceSynchronize() == hipSuccess);
        assert(hipMemcpy(h_y_gpu.data(), d_y, h_y_gpu.size() * sizeof(float),
                         hipMemcpyDeviceToHost) == hipSuccess);

        double max_abs_err = 0.0;
        for (uint32_t ji = 0; ji < 2u; ++ji) {
          const auto& hw = (ji == 0u) ? h_w0 : h_w1;
          const auto& hsw = (ji == 0u) ? h_sw0 : h_sw1;
          const size_t y_off = (ji == 0u) ? 0u : (size_t)r0 * nb;
          for (uint32_t r = 0; r < jrows[ji]; ++r) {
            const uint32_t sr = r >> 7;
            for (uint32_t b = 0; b < nb; ++b) {
              float ref = 0.0f;
              for (uint32_t kb = 0; kb < sc_cols; ++kb) {
                float blk = 0.0f;
                for (uint32_t k = 0; k < 128u; ++k) {
                  blk += e4m3_to_f32_cpu(hw[(size_t)r * cols + kb * 128u + k]) *
                         e4m3_to_f32_cpu(h_x[(size_t)b * cols + kb * 128u + k]);
                }
                ref += blk * (ue8m0_to_f32_cpu(hsw[(size_t)sr * sc_cols + kb]) *
                              h_sx[(size_t)b * sc_cols + kb]);
              }
              const double err =
                  std::fabs((double)h_y_gpu[y_off + (size_t)r * nb + b] - (double)ref);
              if (err > max_abs_err) max_abs_err = err;
            }
          }
        }
        std::printf("  fp8_multi_null_band(%u+%u, %u, %u, ps=%d): max_abs_err=%.6e\n", r0, r1,
                    cols, nb, (int)preshuffled, max_abs_err);
        assert(max_abs_err <= 2e-3);

        (void)hipFree(d_w0);
        (void)hipFree(d_w1);
        if (d_ps0) (void)hipFree(d_ps0);
        if (d_ps1) (void)hipFree(d_ps1);
        (void)hipFree(d_sw0);
        (void)hipFree(d_sw1);
        (void)hipFree(d_x);
        (void)hipFree(d_sx);
        (void)hipFree(d_y);
      };

      for (int ps_mode = 0; ps_mode <= 1; ++ps_mode) {
        const bool ps = (ps_mode != 0);
        run_fp8_multi_null_band(1536u, 576u, 5120u, 16u, ps); // ksplit == 1
        run_fp8_multi_null_band(256u, 256u, 5120u, 16u, ps);  // ksplit > 1
      }

      // (c) batch_gemm_multi_hip (BF16, non-inline split-K, ksplit == 1, jobs[1].y == nullptr).
      {
        const uint32_t jrows[2] = {256u, 128u};
        const uint32_t cols = 256u, nb = 16u, total = jrows[0] + jrows[1];
        std::vector<uint16_t> h_w0((size_t)jrows[0] * cols), h_w1((size_t)jrows[1] * cols);
        std::vector<uint16_t> h_x((size_t)cols * nb); // dim-major [cols][nb]
        std::vector<float> h_y_gpu((size_t)total * nb, 0.0f);
        for (size_t i = 0; i < h_w0.size(); ++i)
          h_w0[i] = f32_to_bf16_cpu(0.125f * (float)((int)(i % 13u) - 6));
        for (size_t i = 0; i < h_w1.size(); ++i)
          h_w1[i] = f32_to_bf16_cpu(0.125f * (float)((int)((i * 3u) % 11u) - 5));
        for (size_t i = 0; i < h_x.size(); ++i)
          h_x[i] = f32_to_bf16_cpu(0.25f * (float)((int)((i * 7u) % 9u) - 4));

        uint16_t *d_w0 = nullptr, *d_w1 = nullptr, *d_x = nullptr;
        float* d_y = nullptr;
        assert(hipMalloc(&d_w0, h_w0.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_w1, h_w1.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_x, h_x.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_y, h_y_gpu.size() * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_w0, h_w0.data(), h_w0.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemset(d_y, 0, h_y_gpu.size() * sizeof(float)) == hipSuccess);

        aff::BatchGemmJob jobs[2] = {
            {d_w0, nullptr, d_y, jrows[0], 0u, 0u},
            {d_w1, nullptr, nullptr, jrows[1], 0u, 0u},
        };
        assert(aff::batch_gemm_multi_hip(AFF_BF16, jobs, 2u, d_x, cols, nb, true, nullptr));
        assert(hipGetLastError() == hipSuccess);
        assert(hipDeviceSynchronize() == hipSuccess);
        assert(hipMemcpy(h_y_gpu.data(), d_y, h_y_gpu.size() * sizeof(float),
                         hipMemcpyDeviceToHost) == hipSuccess);

        double max_abs_err = 0.0;
        for (uint32_t ji = 0; ji < 2u; ++ji) {
          const auto& hw = (ji == 0u) ? h_w0 : h_w1;
          const size_t y_off = (ji == 0u) ? 0u : (size_t)jrows[0] * nb;
          for (uint32_t r = 0; r < jrows[ji]; ++r) {
            for (uint32_t b = 0; b < nb; ++b) {
              float ref = 0.0f;
              for (uint32_t c = 0; c < cols; ++c) {
                ref += bf16_to_f32_cpu(hw[(size_t)r * cols + c]) *
                       bf16_to_f32_cpu(h_x[(size_t)c * nb + b]);
              }
              const double err =
                  std::fabs((double)h_y_gpu[y_off + (size_t)r * nb + b] - (double)ref);
              if (err > max_abs_err) max_abs_err = err;
            }
          }
        }
        std::printf("  bf16_multi_null_band(256+128, 256, 16): max_abs_err=%.6e\n", max_abs_err);
        assert(max_abs_err <= 2e-3);

        (void)hipFree(d_w0);
        (void)hipFree(d_w1);
        (void)hipFree(d_x);
        (void)hipFree(d_y);
      }

      std::printf("PASS: Multi-GEMM contiguous null-band regressions\n");
  }

  // Test 7: Live-only tp_ar_batch_hc_expand parity vs unfused tp_all_reduce + batch_hc_expand_hip
  {
    int ndev = 0, can01 = 0, can10 = 0;
    if (hipGetDeviceCount(&ndev) == hipSuccess && ndev >= 1) {
      const bool p2p = ndev >= 2 &&
                       hipDeviceCanAccessPeer(&can01, 0, 1) == hipSuccess && can01 &&
                       hipDeviceCanAccessPeer(&can10, 1, 0) == hipSuccess && can10;
      const int devs[2] = {0, p2p ? 1 : 0};
      const uint32_t E = 256u, nb = 16u, max_nh = 8u;
      const size_t max_blk = (size_t)E * nb;
      aff::TpContext ctx;
      std::string err;
      assert(aff::tp_init(&ctx, max_blk, &err, nullptr, 2, devs));

      // Also verify Split-K reserve within <= 4 MiB/card succeeds on rank 0 streams.
      assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
      assert(aff::batch_splitk_reserve_hip(524224u, ctx.stream[0]));

      struct FuseCase { uint32_t live; uint32_t nh; bool ctl; bool tap; uint32_t b_lo; uint32_t keep; };
      const FuseCase fcases[] = {
          {1u, 8u, true, false, 0u, 0u},
          {4u, 8u, true, true, 1u, 2u},
          {5u, 8u, true, true, 1u, 3u},
          {6u, 8u, true, true, 0u, 6u},
          {16u, 4u, true, true, 2u, 8u},
          {6u, 8u, false, false, 0u, 0u},
      };

      const size_t hc_elems = (size_t)max_nh * E * nb;
      const size_t comb_elems = (size_t)max_nh * max_nh * nb;
      const size_t post_elems = (size_t)max_nh * nb;
      const size_t tap_elems = (size_t)nb * E;

      uint16_t *d_hc[2] = {}, *d_out_ref[2] = {}, *d_out_fuse[2] = {};
      float *d_blk_ref[2] = {}, *d_blk_fuse[2] = {}, *d_post[2] = {}, *d_comb[2] = {},
            *d_tap_ref[2] = {}, *d_tap_fuse[2] = {};
      float *d_pre_in[2] = {}, *d_cur_ref[2] = {}, *d_cur_fuse[2] = {},
            *d_part_ref[2] = {}, *d_part_fuse[2] = {};
      float *d_mix[2] = {}, *d_scale[2] = {}, *d_base[2] = {}, *d_nw[2] = {};
      float *d_pre_ref[2] = {}, *d_pre_fuse[2] = {}, *d_post_ref[2] = {}, *d_post_fuse[2] = {},
            *d_comb_ref[2] = {}, *d_comb_fuse[2] = {}, *d_norm_ref[2] = {}, *d_norm_fuse[2] = {};
      float *d_qs_ref[2] = {}, *d_qs_fuse[2] = {};
      uint8_t *d_q8_ref[2] = {}, *d_q8_fuse[2] = {};
      float *d_hcur_ref[2] = {}, *d_hpart_ref[2] = {}, *d_hnorm_ref[2] = {}, *d_hnorm_fuse[2] = {},
            *d_hnorm_tm_ref[2] = {}, *d_hnorm_tm_fuse[2] = {};
      const size_t mix_elems = (size_t)(2u * max_nh + max_nh * max_nh) * nb;
      const size_t base_elems = (size_t)(2u * max_nh + max_nh * max_nh);
      const size_t part_elems = (size_t)2u * kHcnChunks * nb;
      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        assert(hipMalloc(&d_hc[r], hc_elems * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_out_ref[r], hc_elems * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_out_fuse[r], hc_elems * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_blk_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_blk_fuse[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_post[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_comb[r], comb_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_tap_ref[r], tap_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_tap_fuse[r], tap_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_pre_in[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_cur_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_cur_fuse[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_part_ref[r], part_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_part_fuse[r], part_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_mix[r], mix_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_scale[r], 3u * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_base[r], base_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_nw[r], E * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_pre_ref[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_pre_fuse[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_post_ref[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_post_fuse[r], post_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_comb_ref[r], comb_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_comb_fuse[r], comb_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_norm_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_norm_fuse[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_q8_ref[r], (size_t)nb * E) == hipSuccess);
        assert(hipMalloc(&d_q8_fuse[r], (size_t)nb * E) == hipSuccess);
        assert(hipMalloc(&d_qs_ref[r], (size_t)nb * (E / 128u) * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_qs_fuse[r], (size_t)nb * (E / 128u) * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hcur_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hpart_ref[r], part_elems * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hnorm_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hnorm_fuse[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hnorm_tm_ref[r], max_blk * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_hnorm_tm_fuse[r], max_blk * sizeof(float)) == hipSuccess);
      }

      const char* prev_v41 = std::getenv("AFF_V41_HC_FUSED");
      std::string prev_v41_val = prev_v41 ? prev_v41 : "";
      setenv("AFF_V41_HC_FUSED", "1", 1);

      for (const FuseCase& fc : fcases) {
        const size_t nh_hc = (size_t)fc.nh * E * nb;
        const size_t nh_mix = (size_t)(2u * fc.nh + fc.nh * fc.nh) * nb;
        std::vector<uint16_t> h_hc(nh_hc), h_poison(nh_hc, 0x7FC0u);
        std::vector<float> h_blk0(max_blk), h_blk1(max_blk), h_post((size_t)fc.nh * nb),
            h_comb((size_t)fc.nh * fc.nh * nb), h_pre_in((size_t)fc.nh * nb),
            h_mix(nh_mix), h_scale{0.7f, 0.8f, 0.9f}, h_base(2u * fc.nh + fc.nh * fc.nh), h_nw(E);
        for (size_t i = 0; i < nh_hc; ++i)
          h_hc[i] = f32_to_bf16_cpu(0.05f * (float)((int)(i % 37u) - 18));
        for (uint32_t d = 0; d < E; ++d) {
          for (uint32_t b = 0; b < nb; ++b) {
            const float p0 = std::sin((float)(d * nb + b) * 0.03f);
            const float p1 = std::cos((float)(d * nb + b) * 0.05f);
            h_blk0[(size_t)d * nb + b] = (b < fc.live) ? p0 : 123.0f;
            h_blk1[(size_t)d * nb + b] = (b < fc.live) ? p1 : -456.0f;
          }
          h_nw[d] = 0.85f + 0.3f * std::sin((float)d * 0.07f);
        }
        for (size_t i = 0; i < h_post.size(); ++i) {
          h_post[i] = 0.25f + 0.05f * (float)(i % 9u);
          h_pre_in[i] = 0.15f + 0.07f * (float)((i * 3u) % 11u);
        }
        for (size_t i = 0; i < h_comb.size(); ++i)
          h_comb[i] = 0.02f * (float)((int)(i % 13u) - 6);
        for (size_t i = 0; i < h_mix.size(); ++i)
          h_mix[i] = std::sin((float)i * 0.09f) * 0.5f;
        for (size_t i = 0; i < h_base.size(); ++i)
          h_base[i] = 0.1f * (float)((int)(i % 5u) - 2);

        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
          const auto& hb = (r == 0) ? h_blk0 : h_blk1;
          assert(hipMemcpy(d_hc[r], h_hc.data(), nh_hc * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_out_ref[r], h_hc.data(), nh_hc * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
          const uint16_t* const fuse_init = fc.ctl ? h_poison.data() : h_hc.data();
          assert(hipMemcpy(d_out_fuse[r], fuse_init, nh_hc * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_blk_ref[r], hb.data(), max_blk * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_blk_fuse[r], hb.data(), max_blk * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_post[r], h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_comb[r], h_comb.data(), h_comb.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemset(d_tap_ref[r], 0, tap_elems * sizeof(float)) == hipSuccess);
          assert(hipMemset(d_tap_fuse[r], 0, tap_elems * sizeof(float)) == hipSuccess);
          assert(hipMemcpy(d_pre_in[r], h_pre_in.data(), h_pre_in.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_mix[r], h_mix.data(), h_mix.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_scale[r], h_scale.data(), h_scale.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_base[r], h_base.data(), h_base.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemcpy(d_nw[r], h_nw.data(), h_nw.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
          assert(hipMemset(d_part_ref[r], 0, part_elems * sizeof(float)) == hipSuccess);
          assert(hipMemset(d_part_fuse[r], 0, part_elems * sizeof(float)) == hipSuccess);
          assert(hipMemset(d_hpart_ref[r], 0, part_elems * sizeof(float)) == hipSuccess);
        }

        // Reference: standalone exact tp_all_reduce + batch_hc_expand_hip / add
        aff::tp_all_reduce(ctx, d_blk_ref, d_blk_ref, max_blk, aff::TpReduce::Exact);
        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
          if (fc.ctl) {
            aff::batch_hc_expand_hip(d_out_ref[r], d_blk_ref[r], d_hc[r], d_post[r], d_comb[r], E,
                                     fc.nh, nb, /*zb=*/1, ctx.stream[r],
                                     fc.tap ? d_tap_ref[r] : nullptr, fc.b_lo, fc.keep, 0u, nb, E, 0u);
          } else {
            aff::batch_hc_add_bf16_hip(d_out_ref[r], d_blk_ref[r], max_blk, /*zb=*/1, ctx.stream[r]);
          }
        }
        aff::tp_sync(ctx);

        // Fused live-only AR + HC expand/add (+ R4 HasPreCollapse when fc.ctl)
        uint16_t* const out_f[aff::kTpMaxRanks] = {d_out_fuse[0], d_out_fuse[1]};
        float* const blk_f[aff::kTpMaxRanks] = {d_blk_fuse[0], d_blk_fuse[1]};
        const uint16_t* const hc_f[aff::kTpMaxRanks] = {d_hc[0], d_hc[1]};
        const float* const post_f[aff::kTpMaxRanks] = {d_post[0], d_post[1]};
        const float* const comb_f[aff::kTpMaxRanks] = {d_comb[0], d_comb[1]};
        float* const tap_f[aff::kTpMaxRanks] = {fc.tap ? d_tap_fuse[0] : nullptr,
                                                fc.tap ? d_tap_fuse[1] : nullptr};
        const float* const pre_f[aff::kTpMaxRanks] = {fc.ctl ? d_pre_in[0] : nullptr,
                                                      fc.ctl ? d_pre_in[1] : nullptr};
        float* const cur_f[aff::kTpMaxRanks] = {fc.ctl ? d_cur_fuse[0] : nullptr,
                                                fc.ctl ? d_cur_fuse[1] : nullptr};
        float* const part_f[aff::kTpMaxRanks] = {fc.ctl ? d_part_fuse[0] : nullptr,
                                                 fc.ctl ? d_part_fuse[1] : nullptr};
        uint32_t npart = 0u;
        assert(aff::tp_ar_batch_hc_expand(ctx, out_f, blk_f, hc_f, post_f, comb_f, E, fc.nh, nb,
                                          fc.live, fc.ctl, /*zb=*/1, fc.tap ? tap_f : nullptr,
                                          fc.b_lo, fc.keep, 0u, nb, E, 0u,
                                          fc.ctl ? pre_f : nullptr, fc.ctl ? cur_f : nullptr,
                                          fc.ctl ? part_f : nullptr, &npart));
        if (fc.ctl) {
          assert(npart > 0u && npart <= aff::kHcnChunks);
          for (int r = 0; r < 2; ++r) {
            assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
            // Reference V4.1 collapse+sq+norm (hcn_v41_collapse_sq_kernel + hcn_norm_v41_kernel)
            aff::batch_hc_fused_narrow_hip(d_out_fuse[r], d_mix[r], d_scale[r], d_base[r], d_nw[r],
                                           d_pre_ref[r], d_post_ref[r], d_comb_ref[r], d_cur_ref[r],
                                           d_norm_ref[r], d_part_ref[r], d_q8_ref[r], d_qs_ref[r],
                                           E, fc.nh, nb, 20u, 1e-6f, 1e-6f, 1, ctx.stream[r],
                                           d_pre_in[r]);
            aff::batch_hc_head_narrow_hip(d_out_fuse[r], nullptr, nullptr, nullptr, d_nw[r],
                                          d_hcur_ref[r], d_hnorm_ref[r], d_hnorm_tm_ref[r],
                                          d_hpart_ref[r], E, fc.nh, nb, 1e-6f, 1e-6f, ctx.stream[r],
                                          d_pre_in[r]);
            // Fused R4 norm-only consumers
            aff::batch_hc_norm_v41_only_hip(d_cur_fuse[r], d_mix[r], d_scale[r], d_base[r], d_nw[r],
                                            d_pre_fuse[r], d_post_fuse[r], d_comb_fuse[r],
                                            d_norm_fuse[r], d_part_fuse[r], d_q8_fuse[r],
                                            d_qs_fuse[r], E, fc.nh, nb, npart, 20u, 1e-6f, 1e-6f,
                                            ctx.stream[r]);
            aff::batch_hc_head_norm_only_hip(d_cur_fuse[r], d_nw[r], d_hnorm_fuse[r],
                                             d_hnorm_tm_fuse[r], d_part_fuse[r], E, nb, npart,
                                             1e-6f, ctx.stream[r]);
          }
        } else {
          assert(npart == 0u);
        }
        aff::tp_sync(ctx);
        aff::tp_check(ctx);

        for (int r = 0; r < 2; ++r) {
          assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
          std::vector<uint16_t> href(nh_hc), hfuse(nh_hc);
          std::vector<float> bzero(max_blk), tref(tap_elems), tfuse(tap_elems);
          assert(hipMemcpy(href.data(), d_out_ref[r], nh_hc * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(hfuse.data(), d_out_fuse[r], nh_hc * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(bzero.data(), d_blk_fuse[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
          for (float v : bzero) assert(v == 0.0f);
          if (fc.ctl) {
            for (uint32_t h = 0; h < fc.nh; ++h)
              for (uint32_t d = 0; d < E; ++d) {
                for (uint32_t b = 0; b < fc.live; ++b)
                  assert(href[((size_t)h * E + d) * nb + b] == hfuse[((size_t)h * E + d) * nb + b]);
                for (uint32_t b = fc.live; b < nb; ++b)
                  assert(hfuse[((size_t)h * E + d) * nb + b] == 0u);
              }
          } else {
            for (uint32_t d = 0; d < E; ++d) {
              for (uint32_t b = 0; b < fc.live; ++b)
                assert(href[(size_t)d * nb + b] == hfuse[(size_t)d * nb + b]);
              for (uint32_t b = fc.live; b < nb; ++b)
                assert(hfuse[(size_t)d * nb + b] == 0u);
            }
          }
          if (fc.tap) {
            assert(hipMemcpy(tref.data(), d_tap_ref[r], tap_elems * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(tfuse.data(), d_tap_fuse[r], tap_elems * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(std::memcmp(tref.data(), tfuse.data(), fc.keep * E * sizeof(float)) == 0);
          }
          if (fc.ctl) {
            std::vector<float> cr(max_blk), cf(max_blk), nr(max_blk), nf(max_blk),
                pr((size_t)fc.nh * nb), pf((size_t)fc.nh * nb), por((size_t)fc.nh * nb),
                pof((size_t)fc.nh * nb), cbr((size_t)fc.nh * fc.nh * nb),
                cbf((size_t)fc.nh * fc.nh * nb), qsr((size_t)nb * (E / 128u)),
                qsf((size_t)nb * (E / 128u)), hnr(max_blk), hnf(max_blk), htmr(max_blk),
                htmf(max_blk);
            std::vector<uint8_t> q8r((size_t)nb * E), q8f((size_t)nb * E);
            assert(hipMemcpy(cr.data(), d_cur_ref[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(cf.data(), d_cur_fuse[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(nr.data(), d_norm_ref[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(nf.data(), d_norm_fuse[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(pr.data(), d_pre_ref[r], pr.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(pf.data(), d_pre_fuse[r], pf.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(por.data(), d_post_ref[r], por.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(pof.data(), d_post_fuse[r], pof.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(cbr.data(), d_comb_ref[r], cbr.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(cbf.data(), d_comb_fuse[r], cbf.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(q8r.data(), d_q8_ref[r], q8r.size(), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(q8f.data(), d_q8_fuse[r], q8f.size(), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(qsr.data(), d_qs_ref[r], qsr.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(qsf.data(), d_qs_fuse[r], qsf.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(hnr.data(), d_hnorm_ref[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(hnf.data(), d_hnorm_fuse[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(htmr.data(), d_hnorm_tm_ref[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(htmf.data(), d_hnorm_tm_fuse[r], max_blk * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

            assert(std::memcmp(cr.data(), cf.data(), max_blk * sizeof(float)) == 0);
            for (size_t i = 0; i < max_blk; ++i) {
              assert(std::fabs(nr[i] - nf[i]) <= 1e-6f);
              assert(std::fabs(hnr[i] - hnf[i]) <= 1e-6f);
              assert(std::fabs(htmr[i] - htmf[i]) <= 1e-6f);
            }
            for (size_t i = 0; i < pr.size(); ++i) {
              assert(std::fabs(pr[i] - pf[i]) <= 1e-6f);
              assert(std::fabs(por[i] - pof[i]) <= 1e-6f);
            }
            for (size_t i = 0; i < cbr.size(); ++i)
              assert(std::fabs(cbr[i] - cbf[i]) <= 1e-6f);
            for (size_t i = 0; i < qsr.size(); ++i)
              assert(std::fabs(qsr[i] - qsf[i]) <= 1e-6f);
            for (size_t i = 0; i < q8r.size(); ++i)
              assert(std::abs((int)q8r[i] - (int)q8f[i]) <= 1);
          }
        }
        if (fc.ctl) {
          auto cmp_ranks = [&](const char* tag, float* const ptr[2], size_t nbytes) {
            std::vector<uint8_t> a(nbytes), b(nbytes);
            assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
            assert(hipMemcpy(a.data(), ptr[0], nbytes, hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
            assert(hipMemcpy(b.data(), ptr[1], nbytes, hipMemcpyDeviceToHost) == hipSuccess);
            if (std::memcmp(a.data(), b.data(), nbytes) != 0) {
              const float* fa = reinterpret_cast<const float*>(a.data());
              const float* fb = reinterpret_cast<const float*>(b.data());
              const size_t nf = nbytes / sizeof(float);
              for (size_t k = 0; k < nf; ++k) {
                uint32_t ua = 0, ub = 0;
                std::memcpy(&ua, fa + k, 4);
                std::memcpy(&ub, fb + k, 4);
                if (ua != ub) {
                  std::fprintf(stderr,
                               "Test7 cmp_ranks FAIL %s live=%u nh=%u npart=%u idx=%zu: r0=%g(0x%08x) r1=%g(0x%08x)\n",
                               tag, fc.live, fc.nh, npart, k, fa[k], ua, fb[k], ub);
                  break;
                }
              }
              assert(false && "cmp_ranks mismatch");
            }
          };
          cmp_ranks("cur_fuse", d_cur_fuse, max_blk * sizeof(float));
          cmp_ranks("norm_fuse", d_norm_fuse, max_blk * sizeof(float));
          cmp_ranks("pre_fuse", d_pre_fuse, (size_t)fc.nh * nb * sizeof(float));
          cmp_ranks("post_fuse", d_post_fuse, (size_t)fc.nh * nb * sizeof(float));
          cmp_ranks("comb_fuse", d_comb_fuse, (size_t)fc.nh * fc.nh * nb * sizeof(float));
          cmp_ranks("qs_fuse", d_qs_fuse, (size_t)nb * (E / 128u) * sizeof(float));
          cmp_ranks("hnorm_fuse", d_hnorm_fuse, max_blk * sizeof(float));
          cmp_ranks("hnorm_tm_fuse", d_hnorm_tm_fuse, max_blk * sizeof(float));
          std::vector<uint8_t> q0((size_t)nb * E), q1((size_t)nb * E);
          assert(hipSetDevice(ctx.dev[0]) == hipSuccess);
          assert(hipMemcpy(q0.data(), d_q8_fuse[0], q0.size(), hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipSetDevice(ctx.dev[1]) == hipSuccess);
          assert(hipMemcpy(q1.data(), d_q8_fuse[1], q1.size(), hipMemcpyDeviceToHost) == hipSuccess);
          assert(std::memcmp(q0.data(), q1.data(), q0.size()) == 0);
        }
      }

      if (prev_v41) setenv("AFF_V41_HC_FUSED", prev_v41_val.c_str(), 1);
      else unsetenv("AFF_V41_HC_FUSED");

      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        for (void* p : {(void*)d_hc[r], (void*)d_out_ref[r], (void*)d_out_fuse[r],
                        (void*)d_blk_ref[r], (void*)d_blk_fuse[r], (void*)d_post[r],
                        (void*)d_comb[r], (void*)d_tap_ref[r], (void*)d_tap_fuse[r],
                        (void*)d_pre_in[r], (void*)d_cur_ref[r], (void*)d_cur_fuse[r],
                        (void*)d_part_ref[r], (void*)d_part_fuse[r], (void*)d_mix[r],
                        (void*)d_scale[r], (void*)d_base[r], (void*)d_nw[r],
                        (void*)d_pre_ref[r], (void*)d_pre_fuse[r], (void*)d_post_ref[r],
                        (void*)d_post_fuse[r], (void*)d_comb_ref[r], (void*)d_comb_fuse[r],
                        (void*)d_norm_ref[r], (void*)d_norm_fuse[r], (void*)d_q8_ref[r],
                        (void*)d_q8_fuse[r], (void*)d_qs_ref[r], (void*)d_qs_fuse[r],
                        (void*)d_hcur_ref[r], (void*)d_hpart_ref[r], (void*)d_hnorm_ref[r],
                        (void*)d_hnorm_fuse[r], (void*)d_hnorm_tm_ref[r], (void*)d_hnorm_tm_fuse[r]})
          (void)hipFree(p);
      }
      aff::batch_splitk_release_hip(ctx.stream[0]);
      aff::tp_free(&ctx);
      assert(hipSetDevice(0) == hipSuccess);
      std::printf("PASS: tp_ar_batch_hc_expand live-only parity\n");
    }
  }

  // Test 8: Exhaustive 65,536-BF16-value parity for HW vs SW o_fp8_qdq_wave32 (AFF_HW_OQDQ=1 vs =0)
  {
    const char* prev_hw = std::getenv("AFF_HW_OQDQ");
    std::string prev_hw_val = prev_hw ? prev_hw : "";

    // Pass A: all 65,536 BF16 values packed in 2048 Wave32 blocks (nb=1).
    {
      const uint32_t nb32 = 2048u;
      std::vector<uint16_t> h_in(65536u), h_hw(65536u), h_sw(65536u);
      for (uint32_t i = 0; i < 65536u; ++i) h_in[i] = (uint16_t)i;
      uint16_t* d_buf = nullptr;
      assert(hipMalloc(&d_buf, h_in.size() * sizeof(uint16_t)) == hipSuccess);

      setenv("AFF_HW_OQDQ", "0", 1);
      assert(hipMemcpy(d_buf, h_in.data(), h_in.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      aff::batch_o_fp8_qdq_bf16_hip(d_buf, nb32, 1u, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_sw.data(), d_buf, h_sw.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);

      setenv("AFF_HW_OQDQ", "1", 1);
      assert(hipMemcpy(d_buf, h_in.data(), h_in.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      aff::batch_o_fp8_qdq_bf16_hip(d_buf, nb32, 1u, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_hw.data(), d_buf, h_hw.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);

      for (uint32_t i = 0; i < 65536u; ++i) assert(h_hw[i] == h_sw[i]);
      (void)hipFree(d_buf);
    }

    // Pass B: every one of all 65,536 BF16 values tested in fixed-scale Wave32 context (lanes 1..31 = 1.0f),
    // exercising normal, subnormal (<2^-6), signed-zero, clamping (+-448), Inf, and NaN paths.
    {
      const uint32_t nb32 = 65536u;
      const size_t total = (size_t)nb32 * 32u;
      std::vector<uint16_t> h_in(total, f32_to_bf16_cpu(1.0f)), h_hw(total), h_sw(total);
      for (uint32_t v = 0; v < 65536u; ++v) h_in[(size_t)v * 32u] = (uint16_t)v;
      uint16_t* d_buf = nullptr;
      assert(hipMalloc(&d_buf, total * sizeof(uint16_t)) == hipSuccess);

      setenv("AFF_HW_OQDQ", "0", 1);
      assert(hipMemcpy(d_buf, h_in.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      aff::batch_o_fp8_qdq_bf16_hip(d_buf, nb32, 1u, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_sw.data(), d_buf, total * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);

      setenv("AFF_HW_OQDQ", "1", 1);
      assert(hipMemcpy(d_buf, h_in.data(), total * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      aff::batch_o_fp8_qdq_bf16_hip(d_buf, nb32, 1u, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_hw.data(), d_buf, total * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);

      for (size_t i = 0; i < total; ++i) assert(h_hw[i] == h_sw[i]);
      (void)hipFree(d_buf);
    }

    if (prev_hw) setenv("AFF_HW_OQDQ", prev_hw_val.c_str(), 1);
    else unsetenv("AFF_HW_OQDQ");
    std::printf("PASS: Exhaustive 65,536-BF16 HW vs SW o_fp8_qdq_wave32 parity\n");
  }

  // Test 9: Direct token-major quant (batch_quant_tm_fp8_hip) + W8A8 GEMM & narrow RoPE parity
  {
    const uint32_t dim = 256u, rows = 128u;
    for (uint32_t nt : {1u, 6u, 32u}) {
      std::vector<float> h_src((size_t)nt * dim), h_y((size_t)rows * nt, 0.0f);
      std::vector<uint8_t> h_w((size_t)rows * dim, 0x38u), h_sw((rows / 128u) * (dim / 128u), 127u);
      for (size_t i = 0; i < h_src.size(); ++i)
        h_src[i] = std::sin((float)(i + nt) * 0.11f) * 2.0f;
      for (size_t i = 0; i < h_w.size(); ++i)
        h_w[i] = (uint8_t)((i % 7u) | ((i % 2u) ? 0x30u : 0xb0u));

      float *d_src = nullptr, *d_qs = nullptr, *d_y = nullptr;
      uint8_t *d_q8 = nullptr, *d_w = nullptr, *d_ps = nullptr, *d_sw = nullptr;
      assert(hipMalloc(&d_src, h_src.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_q8, h_src.size()) == hipSuccess);
      assert(hipMalloc(&d_qs, (size_t)nt * (dim / 128u) * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_w, h_w.size()) == hipSuccess);
      assert(hipMalloc(&d_ps, h_w.size()) == hipSuccess);
      assert(hipMalloc(&d_sw, h_sw.size()) == hipSuccess);
      assert(hipMalloc(&d_y, h_y.size() * sizeof(float)) == hipSuccess);

      assert(hipMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_w, h_w.data(), h_w.size(), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(d_sw, h_sw.data(), h_sw.size(), hipMemcpyHostToDevice) == hipSuccess);
      aff::preshuffle_fp8_hip(d_w, d_ps, rows, dim, nullptr);
      aff::batch_quant_tm_fp8_hip(d_src, d_q8, d_qs, dim, nt, nullptr);
      aff::batch_gemm_fp8_hip(d_ps, d_sw, d_q8, d_qs, d_y, rows, dim, nt, 0u, 0u, nullptr, true);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<uint8_t> g_q8(h_src.size());
      std::vector<float> g_qs((size_t)nt * (dim / 128u));
      assert(hipMemcpy(g_q8.data(), d_q8, g_q8.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(g_qs.data(), d_qs, g_qs.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      double max_err = 0.0;
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < nt; ++b) {
          float ref = 0.0f;
          for (uint32_t kb = 0; kb < dim / 128u; ++kb) {
            float blk = 0.0f;
            for (uint32_t k = 0; k < 128u; ++k) {
              blk += e4m3_to_f32_cpu(h_w[(size_t)r * dim + kb * 128u + k]) *
                     e4m3_to_f32_cpu(g_q8[(size_t)b * dim + kb * 128u + k]);
            }
            ref += blk * g_qs[(size_t)b * (dim / 128u) + kb];
          }
          max_err = std::max(max_err, std::fabs((double)h_y[(size_t)r * nt + b] - (double)ref));
        }
      }
      assert(max_err <= 2e-3);
      for (void* p : {(void*)d_src, (void*)d_q8, (void*)d_qs, (void*)d_w, (void*)d_ps, (void*)d_sw, (void*)d_y})
        (void)hipFree(p);
    }

    // Narrow RoPE specialization parity (nb = {1, 6, 16} vs wide nb=32 for leading columns)
    aff::RopeDerived rope{1.0f, 1.0f / std::pow(10000.0f, 2.0f / 64.0f), 0.0f, 0.0f, 0.0f};
    const uint32_t nh = 4u, hd = 64u, rot = 64u, n = nh * hd;
    std::vector<float> h_x((size_t)n * 32u), h_inv((size_t)nh * 32u, 1.0f), h_rn((size_t)512u * 32u);
    std::vector<uint16_t> h_ref((size_t)32u * n);
    for (size_t i = 0; i < h_x.size(); ++i) h_x[i] = std::sin((float)i * 0.07f);
    for (size_t i = 0; i < h_rn.size(); ++i) h_rn[i] = std::cos((float)i * 0.03f);
    {
      float *dx = nullptr, *dinv = nullptr, *drn = nullptr, *drno = nullptr;
      uint16_t* ddst = nullptr;
      assert(hipMalloc(&dx, h_x.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&dinv, h_inv.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&ddst, h_ref.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&drn, h_rn.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&drno, h_rn.size() * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(dx, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(dinv, h_inv.data(), h_inv.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(hipMemcpy(drn, h_rn.data(), h_rn.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
      assert(aff::batch_head_rms_rope_tr_hip(dx, ddst, dinv, nh, hd, rot, 17u, rope, 1e-6f, 32u, nullptr, false));
      aff::batch_rms_norm_rope_hip(drn, nullptr, drno, 512u, rot, 17u, rope, 1e-6f, 32u, nullptr);
      assert(hipDeviceSynchronize() == hipSuccess);
      assert(hipMemcpy(h_ref.data(), ddst, h_ref.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
      std::vector<float> h_rno_ref(h_rn.size());
      assert(hipMemcpy(h_rno_ref.data(), drno, h_rno_ref.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);

      for (uint32_t nb : {1u, 6u, 16u}) {
        std::vector<float> sub_x((size_t)n * nb), sub_rn((size_t)512u * nb), sub_rno((size_t)512u * nb);
        std::vector<uint16_t> sub_dst((size_t)nb * n);
        for (uint32_t d = 0; d < n; ++d)
          for (uint32_t b = 0; b < nb; ++b) sub_x[(size_t)d * nb + b] = h_x[(size_t)d * 32u + b];
        for (uint32_t d = 0; d < 512u; ++d)
          for (uint32_t b = 0; b < nb; ++b) sub_rn[(size_t)d * nb + b] = h_rn[(size_t)d * 32u + b];
        assert(hipMemcpy(dx, sub_x.data(), sub_x.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        assert(hipMemcpy(drn, sub_rn.data(), sub_rn.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        assert(aff::batch_head_rms_rope_tr_hip(dx, ddst, dinv, nh, hd, rot, 17u, rope, 1e-6f, nb, nullptr, false));
        aff::batch_rms_norm_rope_hip(drn, nullptr, drno, 512u, rot, 17u, rope, 1e-6f, nb, nullptr);
        assert(hipDeviceSynchronize() == hipSuccess);
        assert(hipMemcpy(sub_dst.data(), ddst, sub_dst.size() * sizeof(uint16_t), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(sub_rno.data(), drno, sub_rno.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(std::memcmp(sub_dst.data(), h_ref.data(), (size_t)nb * n * sizeof(uint16_t)) == 0);
        for (uint32_t d = 0; d < 512u; ++d)
          for (uint32_t b = 0; b < nb; ++b)
            assert(sub_rno[(size_t)d * nb + b] == h_rno_ref[(size_t)d * 32u + b]);
      }
      for (void* p : {(void*)dx, (void*)dinv, (void*)ddst, (void*)drn, (void*)drno}) (void)hipFree(p);
    }
    std::printf("PASS: batch_quant_tm_fp8_hip + narrow head/rms RoPE bit-identical parity\n");
  }

  // Test 10: Lossless bf16x12 pack + matvec_bf16x12_multix_hip + batch_gemm_bf16x12_hip bit-exactness
  {
    const uint32_t rows = 256u, cols = 5120u;
    std::vector<uint16_t> h_w((size_t)rows * cols);
    for (size_t i = 0; i < h_w.size(); ++i)
      h_w[i] = f32_to_bf16_cpu(std::sin((float)(i % 613u) * 0.037f) * 1.5f);

    // Inject +-0.0, subnormals, +-Inf, and wide-span escape columns in both row halves [0,128) and [128,256)
    for (uint32_t base_r : {0u, 128u}) {
      h_w[(size_t)(base_r + 0u) * cols + 3u] = 0x0000u;     // +0.0
      h_w[(size_t)(base_r + 0u) * cols + 7u] = 0x8000u;     // -0.0
      h_w[(size_t)(base_r + 1u) * cols + 5u] = 0x0001u;     // +subnormal
      h_w[(size_t)(base_r + 1u) * cols + 11u] = 0x8042u;    // -subnormal
      h_w[(size_t)(base_r + 2u) * cols + 13u] = 0x7F80u;    // +Inf
      h_w[(size_t)(base_r + 3u) * cols + 19u] = 0xFF80u;    // -Inf
      // Wide exponent span > 14 within same 32-column block
      h_w[(size_t)(base_r + 4u) * cols + 32u] = f32_to_bf16_cpu(1.0f);
      h_w[(size_t)(base_r + 4u) * cols + 33u] = f32_to_bf16_cpu(std::ldexp(1.0f, -20));
      h_w[(size_t)(base_r + 4u) * cols + 63u] = f32_to_bf16_cpu(std::ldexp(1.0f, -24));
    }

    std::vector<uint32_t> esc_off;
    std::vector<uint16_t> esc_col, esc_val;
    build_bf16x12_esc_host(h_w.data(), rows, cols, esc_off, esc_col, esc_val);
    assert(!esc_col.empty());

    uint16_t *d_w = nullptr, *d_esc_col = nullptr, *d_esc_val = nullptr;
    uint8_t *d_sm = nullptr, *d_e4 = nullptr, *d_me = nullptr;
    uint32_t *d_esc_off = nullptr;
    assert(hipMalloc(&d_w, h_w.size() * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_sm, (size_t)rows * cols) == hipSuccess);
    assert(hipMalloc(&d_e4, (size_t)rows * (cols / 2u)) == hipSuccess);
    assert(hipMalloc(&d_me, (size_t)rows * (cols / 32u)) == hipSuccess);
    assert(hipMalloc(&d_esc_off, esc_off.size() * sizeof(uint32_t)) == hipSuccess);
    assert(hipMalloc(&d_esc_col, esc_col.size() * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_esc_val, esc_val.size() * sizeof(uint16_t)) == hipSuccess);
    assert(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_esc_off, esc_off.data(), esc_off.size() * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_esc_col, esc_col.data(), esc_col.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_esc_val, esc_val.data(), esc_val.size() * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);

    pack_bf16x12_hip(d_w, d_sm, d_e4, d_me, rows, cols, nullptr);
    assert(hipDeviceSynchronize() == hipSuccess);

    const uint32_t offsets[2] = {0u, 128u};
    for (uint32_t w_row_off : offsets) {
      const uint32_t sub_rows = (w_row_off == 0u) ? rows : (rows - w_row_off);

      // (a) matvec_bf16x12_multix_hip vs matvec_multix_hip for nb = 1, 5
      for (uint32_t nb : {1u, 5u}) {
        std::vector<float> h_xtm((size_t)nb * cols);
        for (size_t i = 0; i < h_xtm.size(); ++i) {
          const float v = std::cos((float)(i % 401u) * 0.031f);
          h_xtm[i] = (std::fabs(v) < 0.1f) ? 0.5f : v;
        }
        float *d_x = nullptr, *d_y_ref = nullptr, *d_y_b12 = nullptr;
        assert(hipMalloc(&d_x, h_xtm.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_ref, (size_t)nb * sub_rows * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_b12, (size_t)nb * sub_rows * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_x, h_xtm.data(), h_xtm.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

        const MvJob job{d_w + (size_t)w_row_off * cols, nullptr, AFF_BF16, sub_rows, d_y_ref, 1.0f};
        assert(matvec_multix_hip(&job, 1u, d_x, cols, nb, cols, nullptr));
        assert(matvec_bf16x12_multix_hip(
            d_sm + (size_t)w_row_off * cols,
            d_e4 + (size_t)w_row_off * (cols / 2u),
            d_me + (size_t)w_row_off * (cols / 32u),
            d_esc_off + w_row_off, d_esc_col, d_esc_val,
            d_x, d_y_b12, sub_rows, cols, nb, sub_rows, /*out_token_major=*/1, nullptr));
        assert(hipDeviceSynchronize() == hipSuccess);

        std::vector<float> y_ref((size_t)nb * sub_rows), y_b12((size_t)nb * sub_rows);
        assert(hipMemcpy(y_ref.data(), d_y_ref, y_ref.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(y_b12.data(), d_y_b12, y_b12.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        float max_diff = 0.0f;
        for (size_t i = 0; i < y_ref.size(); ++i) {
          assert(std::memcmp(&y_ref[i], &y_b12[i], sizeof(float)) == 0);
          if (std::isfinite(y_ref[i])) max_diff = std::max(max_diff, std::fabs(y_ref[i] - y_b12[i]));
        }
        assert(max_diff == 0.0f);
        (void)hipFree(d_x); (void)hipFree(d_y_ref); (void)hipFree(d_y_b12);
      }

      // (b) batch_gemm_bf16x12_hip vs batch_gemm_hip<AFF_BF16> for nb = 6, 16
      for (uint32_t nb : {6u, 16u}) {
        std::vector<float> h_xdm((size_t)cols * nb);
        for (size_t i = 0; i < h_xdm.size(); ++i) {
          const float v = std::sin((float)(i % 457u) * 0.029f);
          h_xdm[i] = (std::fabs(v) < 0.1f) ? 0.5f : v;
        }
        float *d_x = nullptr, *d_y_ref = nullptr, *d_y_b12 = nullptr;
        assert(hipMalloc(&d_x, h_xdm.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_ref, (size_t)sub_rows * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_y_b12, (size_t)sub_rows * nb * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_x, h_xdm.data(), h_xdm.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);

        batch_gemm_hip(AFF_BF16, d_w + (size_t)w_row_off * cols, nullptr, d_x, d_y_ref,
                       sub_rows, cols, nb, nullptr);
        batch_gemm_bf16x12_hip(d_sm, d_e4, d_me, d_esc_off, d_esc_col, d_esc_val,
                               d_x, d_y_b12, sub_rows, cols, nb, w_row_off, nullptr);
        assert(hipDeviceSynchronize() == hipSuccess);

        std::vector<float> y_ref((size_t)sub_rows * nb), y_b12((size_t)sub_rows * nb);
        assert(hipMemcpy(y_ref.data(), d_y_ref, y_ref.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        assert(hipMemcpy(y_b12.data(), d_y_b12, y_b12.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        float max_diff = 0.0f;
        for (size_t i = 0; i < y_ref.size(); ++i) {
          assert(std::memcmp(&y_ref[i], &y_b12[i], sizeof(float)) == 0);
          if (std::isfinite(y_ref[i])) max_diff = std::max(max_diff, std::fabs(y_ref[i] - y_b12[i]));
        }
        assert(max_diff == 0.0f);
        (void)hipFree(d_x); (void)hipFree(d_y_ref); (void)hipFree(d_y_b12);
      }
    }

    for (void* p : {(void*)d_w, (void*)d_sm, (void*)d_e4, (void*)d_me,
                    (void*)d_esc_off, (void*)d_esc_col, (void*)d_esc_val})
      (void)hipFree(p);
    std::printf("PASS: bf16x12 pack + matvec_bf16x12_multix + batch_gemm_bf16x12 bit-exactness\n");
  }

  // Test 11 (F2): batch_attn_out_prep_fp8_hip + batch_gemm_fp8_grouped_b32_hip parity + canary check
  {
    const uint32_t ng = 4u, group_dim = 4096u, rank = 1024u, head_dim = 512u, rope_dim = 64u;
    const uint32_t n_head = (ng * group_dim) / head_dim;  // 32
    const uint32_t nq = ng * group_dim;                   // 16384
    const uint32_t w_rows = ng * rank;                    // 4096
    const uint32_t ws_cols = group_dim / 128u;
    const RopeDerived rope{1.0f, 0.93f, 0.0f, 0.0f, 1.0f};
    const uint32_t pos0 = 17u;

    std::vector<uint8_t> h_w((size_t)w_rows * group_dim);
    std::vector<uint8_t> h_ws((size_t)(w_rows / 128u) * ws_cols);
    for (size_t i = 0; i < h_w.size(); ++i) {
      uint8_t b = (uint8_t)((i * 73u + 19u) & 0xFFu);
      if ((b & 0x7Fu) == 0x7Fu) b = (uint8_t)(b ^ 0x01u);
      h_w[i] = b;
    }
    for (size_t i = 0; i < h_ws.size(); ++i) h_ws[i] = (uint8_t)(0x7Cu + (i % 5u));

    uint8_t *d_w_raw = nullptr, *d_w_ps = nullptr, *d_ws = nullptr;
    assert(hipMalloc(&d_w_raw, h_w.size()) == hipSuccess);
    assert(hipMalloc(&d_w_ps, h_w.size()) == hipSuccess);
    assert(hipMalloc(&d_ws, h_ws.size()) == hipSuccess);
    assert(hipMemcpy(d_w_raw, h_w.data(), h_w.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ws, h_ws.data(), h_ws.size(), hipMemcpyHostToDevice) == hipSuccess);
    preshuffle_fp8_hip(d_w_raw, d_w_ps, w_rows, group_dim, nullptr);

    for (uint32_t nb : {1u, 6u, 16u, 128u, 260u}) {
      std::vector<float> h_src((size_t)nb * nq);
      for (size_t i = 0; i < h_src.size(); ++i) {
        h_src[i] = std::sin((float)(i % 389u) * 0.037f) * 3.1f +
                   std::cos((float)(i % 173u) * 0.019f) * 1.4f;
      }
      float *d_src = nullptr, *d_y_ref = nullptr;
      uint16_t* d_bf16_ref = nullptr;
      assert(hipMalloc(&d_src, h_src.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_bf16_ref, h_src.size() * sizeof(uint16_t)) == hipSuccess);
      assert(hipMalloc(&d_y_ref, (size_t)w_rows * nb * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);

      assert(batch_attn_out_prep_hip(d_src, d_bf16_ref, n_head, head_dim, rope_dim, pos0, rope,
                                     -1.0f, nb, /*o_qdq=*/true, nullptr));
      batch_gemm_grouped_hip(AFF_FP8_E4M3, d_w_raw, d_ws, d_bf16_ref, d_y_ref, ng, group_dim, rank,
                             nb, nullptr, /*x_bf16=*/true);

      const size_t kGuard = 4096u;
      const size_t q8_bytes = (size_t)nq * nb;
      const size_t sc_off = (q8_bytes + 255u) & ~size_t(255);
      const size_t sc_bytes = (size_t)(nq / 32u) * nb;
      const size_t alias_total = kGuard + sc_off + sc_bytes + kGuard;
      uint8_t* d_alias = nullptr;
      assert(hipMalloc(&d_alias, alias_total) == hipSuccess);
      assert(hipMemset(d_alias, 0xA5, alias_total) == hipSuccess);
      uint8_t* const d_q8 = d_alias + kGuard;
      uint8_t* const d_sc = d_q8 + sc_off;

      const size_t y_guard_elems = 1024u;
      float* d_y_buf = nullptr;
      assert(hipMalloc(&d_y_buf, ((size_t)w_rows * nb + 2u * y_guard_elems) * sizeof(float)) ==
             hipSuccess);
      assert(hipMemset(d_y_buf, 0x5A, ((size_t)w_rows * nb + 2u * y_guard_elems) * sizeof(float)) ==
             hipSuccess);
      float* const d_y_got = d_y_buf + y_guard_elems;

      assert(batch_attn_out_prep_fp8_hip(d_src, d_q8, d_sc, ng, group_dim, head_dim, rope_dim,
                                         pos0, rope, -1.0f, nb, nullptr));

      std::vector<float> y_ref((size_t)w_rows * nb), y_got((size_t)w_rows * nb);
      assert(hipMemcpy(y_ref.data(), d_y_ref, y_ref.size() * sizeof(float), hipMemcpyDeviceToHost) ==
             hipSuccess);
      std::vector<uint8_t> alias_host(alias_total);
      assert(hipMemcpy(alias_host.data(), d_alias, alias_total, hipMemcpyDeviceToHost) ==
             hipSuccess);
      for (size_t i = 0; i < kGuard; ++i) assert(alias_host[i] == 0xA5u);
      for (size_t i = kGuard + q8_bytes; i < kGuard + sc_off; ++i) assert(alias_host[i] == 0xA5u);
      for (size_t i = kGuard + sc_off + sc_bytes; i < alias_total; ++i)
        assert(alias_host[i] == 0xA5u);

      for (bool ps : {false, true}) {
        batch_gemm_fp8_grouped_b32_hip(ps ? d_w_ps : d_w_raw, d_ws, d_q8, d_sc, d_y_got, ng, rank,
                                       group_dim, nb, nullptr, ps);
        assert(hipDeviceSynchronize() == hipSuccess);
        std::vector<uint8_t> y_chk(((size_t)w_rows * nb + 2u * y_guard_elems) * sizeof(float));
        assert(hipMemcpy(y_chk.data(), d_y_buf, y_chk.size(), hipMemcpyDeviceToHost) == hipSuccess);
        for (size_t i = 0; i < y_guard_elems * sizeof(float); ++i) assert(y_chk[i] == 0x5Au);
        const size_t y_tail0 = (y_guard_elems + (size_t)w_rows * nb) * sizeof(float);
        for (size_t i = y_tail0; i < y_chk.size(); ++i) assert(y_chk[i] == 0x5Au);

        std::memcpy(y_got.data(), y_chk.data() + y_guard_elems * sizeof(float),
                    y_got.size() * sizeof(float));
        double dot = 0.0, n_ref = 0.0, n_got = 0.0, diff2 = 0.0;
        for (size_t i = 0; i < y_ref.size(); ++i) {
          const double r = y_ref[i], g = y_got[i];
          const double d = g - r;
          dot += g * r;
          n_ref += r * r;
          n_got += g * g;
          diff2 += d * d;
        }
        const double rel_l2 = std::sqrt(diff2 / std::max(n_ref, 1e-12));
        const double cos = dot / std::sqrt(std::max(n_ref * n_got, 1e-12));
        assert(rel_l2 < 1e-4);
        assert(cos > 0.99999);
      }

      (void)hipFree(d_src);
      (void)hipFree(d_bf16_ref);
      (void)hipFree(d_y_ref);
      (void)hipFree(d_alias);
      (void)hipFree(d_y_buf);
    }

    (void)hipFree(d_w_raw);
    (void)hipFree(d_w_ps);
    (void)hipFree(d_ws);
    std::printf("PASS: F2 batch_attn_out_prep_fp8 + batch_gemm_fp8_grouped_b32 parity & canary\n");
  }

  // Test 12 (F3): batch_quant_tm_fp8_hip(..., n_live) + batch_gemm_fp8_hip(ps=true) vs transpose_pad + GEMM
  {
    const uint32_t rows = 512u, cols = 15360u, nb = 16u;
    const uint32_t ws_cols = cols / 128u;
    std::vector<uint8_t> h_w((size_t)rows * cols);
    std::vector<uint8_t> h_ws((size_t)(rows / 128u) * ws_cols);
    for (size_t i = 0; i < h_w.size(); ++i) {
      uint8_t b = (uint8_t)((i * 59u + 17u) & 0xFFu);
      if ((b & 0x7Fu) == 0x7Fu) b = (uint8_t)(b ^ 0x01u);
      h_w[i] = b;
    }
    for (size_t i = 0; i < h_ws.size(); ++i) h_ws[i] = (uint8_t)(0x7Du + (i % 4u));

    uint8_t *d_w_raw = nullptr, *d_w_ps = nullptr, *d_ws = nullptr;
    assert(hipMalloc(&d_w_raw, h_w.size()) == hipSuccess);
    assert(hipMalloc(&d_w_ps, h_w.size()) == hipSuccess);
    assert(hipMalloc(&d_ws, h_ws.size()) == hipSuccess);
    assert(hipMemcpy(d_w_raw, h_w.data(), h_w.size(), hipMemcpyHostToDevice) == hipSuccess);
    assert(hipMemcpy(d_ws, h_ws.data(), h_ws.size(), hipMemcpyHostToDevice) == hipSuccess);
    preshuffle_fp8_hip(d_w_raw, d_w_ps, rows, cols, nullptr);

    for (uint32_t n_live : {1u, 5u, 6u, 16u}) {
      std::vector<float> h_tap((size_t)n_live * cols);
      std::vector<float> h_dm((size_t)cols * nb, 0.0f);
      for (uint32_t t = 0; t < n_live; ++t) {
        for (uint32_t c = 0; c < cols; ++c) {
          const float v = std::sin((float)(t * 31u + c) * 0.023f) * 2.5f;
          h_tap[(size_t)t * cols + c] = v;
          h_dm[(size_t)c * nb + t] = v;
        }
      }
      float *d_tap = nullptr, *d_dx = nullptr, *d_y_ref = nullptr, *d_y_got = nullptr, *d_qs = nullptr;
      uint8_t* d_q8 = nullptr;
      assert(hipMalloc(&d_tap, h_tap.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_dx, h_dm.size() * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_y_ref, (size_t)rows * nb * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_y_got, (size_t)rows * nb * sizeof(float)) == hipSuccess);
      assert(hipMalloc(&d_q8, (size_t)nb * cols) == hipSuccess);
      assert(hipMalloc(&d_qs, (size_t)nb * ws_cols * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_tap, h_tap.data(), h_tap.size() * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);
      assert(hipMemcpy(d_dx, h_dm.data(), h_dm.size() * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);

      batch_gemm_hip(AFF_FP8_E4M3, d_w_raw, d_ws, d_dx, d_y_ref, rows, cols, nb, nullptr);
      batch_quant_tm_fp8_hip(d_tap, d_q8, d_qs, cols, nb, nullptr, n_live);
      batch_gemm_fp8_hip(d_w_ps, d_ws, d_q8, d_qs, d_y_got, rows, cols, nb, 0u, 0u, nullptr,
                         /*preshuffled=*/true);
      assert(hipDeviceSynchronize() == hipSuccess);

      std::vector<uint8_t> h_q8((size_t)nb * cols);
      std::vector<float> h_qs((size_t)nb * ws_cols), y_ref((size_t)rows * nb), y_got((size_t)rows * nb);
      assert(hipMemcpy(h_q8.data(), d_q8, h_q8.size(), hipMemcpyDeviceToHost) == hipSuccess);
      assert(hipMemcpy(h_qs.data(), d_qs, h_qs.size() * sizeof(float), hipMemcpyDeviceToHost) ==
             hipSuccess);
      assert(hipMemcpy(y_ref.data(), d_y_ref, y_ref.size() * sizeof(float), hipMemcpyDeviceToHost) ==
             hipSuccess);
      assert(hipMemcpy(y_got.data(), d_y_got, y_got.size() * sizeof(float), hipMemcpyDeviceToHost) ==
             hipSuccess);

      for (uint32_t t = n_live; t < nb; ++t) {
        for (uint32_t c = 0; c < cols; ++c) assert(h_q8[(size_t)t * cols + c] == 0x00u);
        for (uint32_t b = 0; b < ws_cols; ++b) assert(h_qs[(size_t)t * ws_cols + b] == 1.0f);
        for (uint32_t r = 0; r < rows; ++r) assert(y_got[(size_t)r * nb + t] == 0.0f);
      }
      double dot = 0.0, n_ref = 0.0, n_got = 0.0;
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t t = 0; t < n_live; ++t) {
          const double rv = y_ref[(size_t)r * nb + t], gv = y_got[(size_t)r * nb + t];
          dot += rv * gv;
          n_ref += rv * rv;
          n_got += gv * gv;
        }
      }
      const double cos = dot / std::sqrt(std::max(n_ref * n_got, 1e-12));
      assert(cos > 0.999);

      for (void* p : {(void*)d_tap, (void*)d_dx, (void*)d_y_ref, (void*)d_y_got, (void*)d_q8, (void*)d_qs})
        (void)hipFree(p);
    }

    (void)hipFree(d_w_raw);
    (void)hipFree(d_w_ps);
    (void)hipFree(d_ws);
    std::printf("PASS: F3 batch_quant_tm_fp8_hip(..., n_live) + batch_gemm_fp8_hip(ps=true)\n");
  }

  // Test 13 (F6): quantize_bf16_to_fp8_g128_hip + matvec_fp8_g128_multix_hip (T=1..8, out_tm={0,1})
  {
    const uint32_t rows = 1024u, cols = 5120u;
    const uint32_t sc_cols = cols / 128u;
    std::vector<uint16_t> h_w((size_t)rows * cols);
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        const float grp_scale = 0.25f + 0.5f * (float)((r + (c / 128u)) % 5u);
        const float v = std::sin((float)(r * 19u + c * 7u) * 0.017f) * grp_scale;
        h_w[(size_t)r * cols + c] = f32_to_bf16_cpu(v);
      }
    }
    uint16_t *d_w_chunk = nullptr, *d_fp8_sc = nullptr;
    uint8_t* d_fp8 = nullptr;
    assert(hipMalloc(&d_w_chunk, (size_t)512u * cols * sizeof(uint16_t)) == hipSuccess);
    assert(hipMalloc(&d_fp8, (size_t)rows * cols) == hipSuccess);
    assert(hipMalloc(&d_fp8_sc, (size_t)rows * sc_cols * sizeof(uint16_t)) == hipSuccess);
    for (uint32_t off = 0; off < rows; off += 512u) {
      assert(hipMemcpy(d_w_chunk, h_w.data() + (size_t)off * cols,
                       (size_t)512u * cols * sizeof(uint16_t), hipMemcpyHostToDevice) == hipSuccess);
      quantize_bf16_to_fp8_g128_hip(d_w_chunk, d_fp8, d_fp8_sc, 512u, cols, off, nullptr);
    }
    assert(hipDeviceSynchronize() == hipSuccess);

    for (uint32_t T = 1u; T <= 8u; ++T) {
      std::vector<float> h_x((size_t)T * cols);
      for (size_t i = 0; i < h_x.size(); ++i)
        h_x[i] = std::cos((float)(i % 317u) * 0.029f) * 1.7f;
      std::vector<double> ref_tm((size_t)T * rows, 0.0);
      for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < rows; ++r) {
          double acc = 0.0;
          const uint16_t* wr = h_w.data() + (size_t)r * cols;
          const float* xt = h_x.data() + (size_t)t * cols;
          for (uint32_t c = 0; c < cols; ++c) acc += (double)bf16_to_f32_cpu(wr[c]) * (double)xt[c];
          ref_tm[(size_t)t * rows + r] = acc;
        }
      }

      float* d_x = nullptr;
      assert(hipMalloc(&d_x, h_x.size() * sizeof(float)) == hipSuccess);
      assert(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice) ==
             hipSuccess);

      for (int out_tm : {0, 1}) {
        const uint32_t ld = out_tm ? rows : 16u;
        const size_t y_elems = out_tm ? (size_t)T * ld : (size_t)rows * ld;
        float* d_y = nullptr;
        assert(hipMalloc(&d_y, y_elems * sizeof(float)) == hipSuccess);
        assert(hipMemset(d_y, 0, y_elems * sizeof(float)) == hipSuccess);
        assert(matvec_fp8_g128_multix_hip(d_fp8, d_fp8_sc, d_x, d_y, rows, cols, T, ld, out_tm,
                                          nullptr));
        assert(hipDeviceSynchronize() == hipSuccess);
        std::vector<float> h_y(y_elems);
        assert(hipMemcpy(h_y.data(), d_y, y_elems * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess);
        for (uint32_t t = 0; t < T; ++t) {
          double dot = 0.0, n_ref = 0.0, n_got = 0.0;
          for (uint32_t r = 0; r < rows; ++r) {
            const double rv = ref_tm[(size_t)t * rows + r];
            const double gv = out_tm ? h_y[(size_t)t * ld + r] : h_y[(size_t)r * ld + t];
            dot += rv * gv;
            n_ref += rv * rv;
            n_got += gv * gv;
          }
          const double cos = dot / std::sqrt(std::max(n_ref * n_got, 1e-12));
          assert(cos > 0.9995);
        }
        if (!out_tm) {
          for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t t = T; t < ld; ++t) assert(h_y[(size_t)r * ld + t] == 0.0f);
        }
        (void)hipFree(d_y);
      }
      (void)hipFree(d_x);
    }
    (void)hipFree(d_w_chunk);
    (void)hipFree(d_fp8);
    (void)hipFree(d_fp8_sc);
    std::printf("PASS: F6 quantize_bf16_to_fp8_g128 + matvec_fp8_g128_multix (T=1..8, out_tm={0,1})\n");
  }

  // Test 14 (F8): 2-wide batch_rms_norm_q8_hip, hcn_norm_kernel, hcn_norm_v41_kernel parity
  {
    const uint32_t nb = 16u;
    const uint32_t n_hc = 4u;
    const float eps = 1e-6f;
    for (uint32_t n : {128u, 1024u, 1536u, 4096u, 5120u}) {
      // (a) batch_rms_norm_q8_hip vs batch_rms_norm_hip + batch_quant_fp8_hip
      {
        const uint32_t nkb = n >> 7;
        std::vector<float> h_x((size_t)n * nb), h_w(n);
        std::vector<uint16_t> h_wbf16(n);
        for (size_t i = 0; i < h_x.size(); ++i)
          h_x[i] = std::sin((float)(i % 419u) * 0.031f) * 2.4f;
        for (uint32_t d = 0; d < n; ++d) {
          const float w = 0.7f + 0.5f * std::cos((float)d * 0.07f);
          h_w[d] = w;
          h_wbf16[d] = f32_to_bf16_cpu(w);
        }
        float *d_x = nullptr, *d_w = nullptr, *d_out_ref = nullptr, *d_out_got = nullptr;
        float *d_qs_ref = nullptr, *d_qs_got = nullptr;
        uint16_t* d_wbf16 = nullptr;
        uint8_t *d_q8_ref = nullptr, *d_q8_got = nullptr;
        assert(hipMalloc(&d_x, h_x.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_w, (size_t)n * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_wbf16, (size_t)n * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_out_ref, h_x.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_out_got, h_x.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_q8_ref, (size_t)nb * n) == hipSuccess);
        assert(hipMalloc(&d_q8_got, (size_t)nb * n) == hipSuccess);
        assert(hipMalloc(&d_qs_ref, (size_t)nb * nkb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_qs_got, (size_t)nb * nkb * sizeof(float)) == hipSuccess);
        assert(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_w, h_w.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_wbf16, h_wbf16.data(), (size_t)n * sizeof(uint16_t),
                         hipMemcpyHostToDevice) == hipSuccess);

        for (bool use_bf16_w : {false, true}) {
          const float* w32 = use_bf16_w ? nullptr : d_w;
          const uint16_t* w16 = use_bf16_w ? d_wbf16 : nullptr;
          batch_rms_norm_hip(d_x, w32, w16, d_out_ref, n, nb, eps, nullptr);
          batch_quant_fp8_hip(d_out_ref, d_q8_ref, d_qs_ref, n, nb, 0u, nullptr);
          assert(batch_rms_norm_q8_hip(d_x, w32, w16, d_out_got, d_q8_got, d_qs_got, n, nb, eps,
                                       nullptr));
          assert(hipDeviceSynchronize() == hipSuccess);

          std::vector<float> out_ref(h_x.size()), out_got(h_x.size());
          std::vector<uint8_t> q8_ref((size_t)nb * n), q8_got((size_t)nb * n);
          std::vector<float> qs_ref((size_t)nb * nkb), qs_got((size_t)nb * nkb);
          assert(hipMemcpy(out_ref.data(), d_out_ref, out_ref.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(out_got.data(), d_out_got, out_got.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(q8_ref.data(), d_q8_ref, q8_ref.size(), hipMemcpyDeviceToHost) ==
                 hipSuccess);
          assert(hipMemcpy(q8_got.data(), d_q8_got, q8_got.size(), hipMemcpyDeviceToHost) ==
                 hipSuccess);
          assert(hipMemcpy(qs_ref.data(), d_qs_ref, qs_ref.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(qs_got.data(), d_qs_got, qs_got.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(std::memcmp(out_got.data(), out_ref.data(), out_ref.size() * sizeof(float)) == 0);
          assert(std::memcmp(q8_got.data(), q8_ref.data(), q8_ref.size()) == 0);
          assert(std::memcmp(qs_got.data(), qs_ref.data(), qs_ref.size() * sizeof(float)) == 0);
        }
        for (void* p : {(void*)d_x, (void*)d_w, (void*)d_wbf16, (void*)d_out_ref, (void*)d_out_got,
                        (void*)d_q8_ref, (void*)d_q8_got, (void*)d_qs_ref, (void*)d_qs_got})
          (void)hipFree(p);
      }

      // (b) hcn_norm_kernel (pre_in == nullptr) & (c) hcn_norm_v41_kernel (pre_in != nullptr)
      {
        const uint32_t m_rows = 2u * n_hc + n_hc * n_hc;  // 24
        std::vector<uint16_t> h_hc((size_t)n_hc * n * nb);
        std::vector<float> h_mix((size_t)m_rows * nb), h_sc(m_rows), h_bs(m_rows), h_nw(n),
            h_pre_in((size_t)n_hc * nb);
        for (size_t i = 0; i < h_hc.size(); ++i)
          h_hc[i] = f32_to_bf16_cpu(std::sin((float)(i % 353u) * 0.041f) * 1.8f);
        for (size_t i = 0; i < h_mix.size(); ++i)
          h_mix[i] = std::cos((float)(i % 197u) * 0.053f) * 0.5f;
        for (uint32_t i = 0; i < m_rows; ++i) {
          h_sc[i] = 0.8f + 0.02f * (float)i;
          h_bs[i] = 0.05f * std::sin((float)i);
        }
        for (uint32_t d = 0; d < n; ++d) h_nw[d] = 0.75f + 0.35f * std::sin((float)d * 0.09f);
        for (size_t i = 0; i < h_pre_in.size(); ++i)
          h_pre_in[i] = 0.25f + 0.1f * std::cos((float)i * 0.3f);

        uint16_t* d_hc = nullptr;
        float *d_mix = nullptr, *d_sc = nullptr, *d_bs = nullptr, *d_nw = nullptr, *d_pre_in = nullptr;
        float *d_pre = nullptr, *d_post = nullptr, *d_comb = nullptr, *d_cur = nullptr, *d_part = nullptr;
        float *d_norm_ref = nullptr, *d_norm_got = nullptr, *d_qs_ref = nullptr, *d_qs_got = nullptr;
        uint8_t *d_q8_ref = nullptr, *d_q8_got = nullptr;
        const uint32_t nkb = std::max<uint32_t>(1u, n >> 7);

        assert(hipMalloc(&d_hc, h_hc.size() * sizeof(uint16_t)) == hipSuccess);
        assert(hipMalloc(&d_mix, h_mix.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_sc, h_sc.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_bs, h_bs.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_nw, h_nw.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_pre_in, h_pre_in.size() * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_pre, (size_t)n_hc * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_post, (size_t)n_hc * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_comb, (size_t)n_hc * n_hc * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_cur, (size_t)n * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_part, (size_t)2u * kHcnChunks * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_norm_ref, (size_t)n * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_norm_got, (size_t)n * nb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_q8_ref, (size_t)nb * n) == hipSuccess);
        assert(hipMalloc(&d_q8_got, (size_t)nb * n) == hipSuccess);
        assert(hipMalloc(&d_qs_ref, (size_t)nb * nkb * sizeof(float)) == hipSuccess);
        assert(hipMalloc(&d_qs_got, (size_t)nb * nkb * sizeof(float)) == hipSuccess);

        assert(hipMemcpy(d_hc, h_hc.data(), h_hc.size() * sizeof(uint16_t), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_mix, h_mix.data(), h_mix.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_sc, h_sc.data(), h_sc.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_bs, h_bs.data(), h_bs.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_nw, h_nw.data(), h_nw.size() * sizeof(float), hipMemcpyHostToDevice) ==
               hipSuccess);
        assert(hipMemcpy(d_pre_in, h_pre_in.data(), h_pre_in.size() * sizeof(float),
                         hipMemcpyHostToDevice) == hipSuccess);

        for (bool use_v41 : {false, true}) {
          const float* pin = use_v41 ? d_pre_in : nullptr;
          assert(hipMemset(d_part, 0, (size_t)2u * kHcnChunks * nb * sizeof(float)) == hipSuccess);
          batch_hc_fused_narrow_hip(d_hc, d_mix, d_sc, d_bs, d_nw, d_pre, d_post, d_comb, d_cur,
                                    d_norm_ref, d_part, nullptr, nullptr, n, n_hc, nb, 4u, eps,
                                    eps, 1, nullptr, pin);
          if ((n % 256u) == 0u) {
            batch_quant_fp8_hip(d_norm_ref, d_q8_ref, d_qs_ref, n, nb, 0u, nullptr);
          }

          assert(hipMemset(d_part, 0, (size_t)2u * kHcnChunks * nb * sizeof(float)) == hipSuccess);
          uint8_t* const q8_arg = ((n % 256u) == 0u) ? d_q8_got : nullptr;
          float* const qs_arg = ((n % 256u) == 0u) ? d_qs_got : nullptr;
          batch_hc_fused_narrow_hip(d_hc, d_mix, d_sc, d_bs, d_nw, d_pre, d_post, d_comb, d_cur,
                                    d_norm_got, d_part, q8_arg, qs_arg, n, n_hc, nb, 4u, eps, eps,
                                    1, nullptr, pin);
          assert(hipDeviceSynchronize() == hipSuccess);

          std::vector<float> n_ref((size_t)n * nb), n_got((size_t)n * nb);
          assert(hipMemcpy(n_ref.data(), d_norm_ref, n_ref.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(hipMemcpy(n_got.data(), d_norm_got, n_got.size() * sizeof(float),
                           hipMemcpyDeviceToHost) == hipSuccess);
          assert(std::memcmp(n_got.data(), n_ref.data(), n_ref.size() * sizeof(float)) == 0);
          if ((n % 256u) == 0u) {
            std::vector<uint8_t> q8_ref((size_t)nb * n), q8_got((size_t)nb * n);
            std::vector<float> qs_ref((size_t)nb * nkb), qs_got((size_t)nb * nkb);
            assert(hipMemcpy(q8_ref.data(), d_q8_ref, q8_ref.size(), hipMemcpyDeviceToHost) ==
                   hipSuccess);
            assert(hipMemcpy(q8_got.data(), d_q8_got, q8_got.size(), hipMemcpyDeviceToHost) ==
                   hipSuccess);
            assert(hipMemcpy(qs_ref.data(), d_qs_ref, qs_ref.size() * sizeof(float),
                             hipMemcpyDeviceToHost) == hipSuccess);
            assert(hipMemcpy(qs_got.data(), d_qs_got, qs_got.size() * sizeof(float),
                             hipMemcpyDeviceToHost) == hipSuccess);
            assert(std::memcmp(q8_got.data(), q8_ref.data(), q8_ref.size()) == 0);
            assert(std::memcmp(qs_got.data(), qs_ref.data(), qs_ref.size() * sizeof(float)) == 0);
          }
        }

        for (void* p : {(void*)d_hc, (void*)d_mix, (void*)d_sc, (void*)d_bs, (void*)d_nw,
                        (void*)d_pre_in, (void*)d_pre, (void*)d_post, (void*)d_comb, (void*)d_cur,
                        (void*)d_part, (void*)d_norm_ref, (void*)d_norm_got, (void*)d_q8_ref,
                        (void*)d_q8_got, (void*)d_qs_ref, (void*)d_qs_got})
          (void)hipFree(p);
      }
    }
    std::printf("PASS: F8 2-wide batch_rms_norm_q8 + hcn_norm_kernel + hcn_norm_v41_kernel parity\n");
  }

  std::printf("ALL WAVE 2 PHASE 2B KERNEL TESTS PASSED.\n");
  return 0;
}
