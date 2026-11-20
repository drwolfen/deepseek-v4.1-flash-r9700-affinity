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
      }

      for (const FuseCase& fc : fcases) {
        const size_t nh_hc = (size_t)fc.nh * E * nb;
        std::vector<uint16_t> h_hc(nh_hc), h_poison(nh_hc, 0x7FC0u);
        std::vector<float> h_blk0(max_blk), h_blk1(max_blk), h_post((size_t)fc.nh * nb),
            h_comb((size_t)fc.nh * fc.nh * nb);
        for (size_t i = 0; i < nh_hc; ++i)
          h_hc[i] = f32_to_bf16_cpu(0.05f * (float)((int)(i % 37u) - 18));
        for (uint32_t d = 0; d < E; ++d) {
          for (uint32_t b = 0; b < nb; ++b) {
            const float p0 = std::sin((float)(d * nb + b) * 0.03f);
            const float p1 = std::cos((float)(d * nb + b) * 0.05f);
            h_blk0[(size_t)d * nb + b] = (b < fc.live) ? p0 : 123.0f;
            h_blk1[(size_t)d * nb + b] = (b < fc.live) ? p1 : -456.0f;
          }
        }
        for (size_t i = 0; i < h_post.size(); ++i)
          h_post[i] = 0.25f + 0.05f * (float)(i % 9u);
        for (size_t i = 0; i < h_comb.size(); ++i)
          h_comb[i] = 0.02f * (float)((int)(i % 13u) - 6);

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

        // Fused live-only AR + HC expand/add
        uint16_t* const out_f[aff::kTpMaxRanks] = {d_out_fuse[0], d_out_fuse[1]};
        float* const blk_f[aff::kTpMaxRanks] = {d_blk_fuse[0], d_blk_fuse[1]};
        const uint16_t* const hc_f[aff::kTpMaxRanks] = {d_hc[0], d_hc[1]};
        const float* const post_f[aff::kTpMaxRanks] = {d_post[0], d_post[1]};
        const float* const comb_f[aff::kTpMaxRanks] = {d_comb[0], d_comb[1]};
        float* const tap_f[aff::kTpMaxRanks] = {fc.tap ? d_tap_fuse[0] : nullptr,
                                                fc.tap ? d_tap_fuse[1] : nullptr};
        assert(aff::tp_ar_batch_hc_expand(ctx, out_f, blk_f, hc_f, post_f, comb_f, E, fc.nh, nb,
                                          fc.live, fc.ctl, /*zb=*/1, fc.tap ? tap_f : nullptr,
                                          fc.b_lo, fc.keep, 0u, nb, E, 0u));
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
        }
      }

      for (int r = 0; r < 2; ++r) {
        assert(hipSetDevice(ctx.dev[r]) == hipSuccess);
        for (void* p : {(void*)d_hc[r], (void*)d_out_ref[r], (void*)d_out_fuse[r],
                        (void*)d_blk_ref[r], (void*)d_blk_fuse[r], (void*)d_post[r],
                        (void*)d_comb[r], (void*)d_tap_ref[r], (void*)d_tap_fuse[r]})
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

  std::printf("ALL WAVE 2 PHASE 2B KERNEL TESTS PASSED.\n");
  return 0;
}
