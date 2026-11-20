#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float bf16_to_f32_cpu(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

static uint16_t f32_to_bf16_cpu(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
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

// Verify that packing 20 bytes into g32[0..4] and extracting via coalesced
// (u32, u32, u16)/(u16, u32, u32) AND branchless 4B-aligned g32+(sblk?2:0) is
// bit-for-byte identical to the byte-by-byte loop for every possible byte pattern.
static void test_tr_mt1_pack_identity() {
  uint8_t raw[20];
  for (int i = 0; i < 20; ++i) raw[i] = (uint8_t)(0xA5u ^ (i * 19u + 7u));
  const uint32_t* g32 = reinterpret_cast<const uint32_t*>(raw);
  const uint16_t* g16 = reinterpret_cast<const uint16_t*>(raw);

  for (uint32_t sblk : {0u, 2u}) {
    uint32_t tw0 = 0, tw1 = 0, tw2 = 0;
    if (sblk == 0u) {
      tw0 = g32[0];
      tw1 = g32[1];
      tw2 = (uint32_t)g16[4];
    } else {
      tw0 = (uint32_t)g16[5];
      tw1 = g32[3];
      tw2 = g32[4];
    }
    const uint32_t* gp = g32 + (sblk ? 2u : 0u);
    const uint32_t bw0 = gp[0], bw1 = gp[1], bw2 = gp[2];
    for (uint32_t bb = 0; bb < 2u; ++bb) {
      uint64_t ref_win = 0;
      const uint8_t* cp = raw + sblk * 5u + bb * 5u;
      for (uint32_t i = 0; i < 5u; ++i) ref_win |= (uint64_t)cp[i] << (8u * i);

      uint64_t got_win = 0, got_br = 0;
      if (sblk == 0u) {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFu) << 32))
                      : ((uint64_t)(tw1 >> 8) | ((uint64_t)tw2 << 24));
        got_br = (bb == 0u)
                     ? ((uint64_t)bw0 | ((uint64_t)(bw1 & 0xFFu) << 32))
                     : ((uint64_t)(bw1 >> 8) | ((uint64_t)(bw2 & 0xFFFFu) << 24));
      } else {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFFFFFu) << 16))
                      : ((uint64_t)(tw1 >> 24) | ((uint64_t)tw2 << 8));
        got_br = (bb == 0u)
                     ? ((uint64_t)(bw0 >> 16) | ((uint64_t)(bw1 & 0xFFFFFFu) << 16))
                     : ((uint64_t)(bw1 >> 24) | ((uint64_t)bw2 << 8));
      }
      if (got_win != ref_win || got_br != ref_win) {
        std::fprintf(stderr, "FAIL: sblk=%u bb=%u got=0x%010llx br=0x%010llx ref=0x%010llx\n",
                     sblk, bb, (unsigned long long)got_win, (unsigned long long)got_br,
                     (unsigned long long)ref_win);
        std::exit(1);
      }
    }
  }
}

static void test_quad2s_tr_parity() {
  const uint32_t rows = 128u, cols = 512u, sc_cols = cols / 128u;
  const float wscale = 0.75f;
  const size_t code_bytes = (size_t)rows * (cols / 2u * 5u / 8u);
  const size_t field_bytes = (size_t)rows * (cols / 16u * 6u / 8u);

  std::vector<uint8_t> h_codes(code_bytes), h_fields(field_bytes);
  for (size_t i = 0; i < code_bytes; ++i) h_codes[i] = (uint8_t)((i * 73u + 19u) & 0xFFu);
  for (size_t i = 0; i < field_bytes; ++i) h_fields[i] = (uint8_t)((i * 41u + 7u) & 0xFFu);

  std::vector<uint8_t> tr_codes(code_bytes), tr_fields(field_bytes);
  aff::expert_transpose_plane_host(h_codes.data(), h_fields.data(),
                                   tr_codes.data(), tr_fields.data(), rows, cols);

  std::vector<uint16_t> h_rscales(rows);
  for (uint32_t r = 0; r < rows; ++r) {
    const float rs = 0.5f + 0.01f * (float)(r % 37u);
    h_rscales[r] = f32_to_bf16_cpu((r & 1u) ? -rs : rs);
  }

  // Codebook: 2 variants x 1024 quads, exponents in [1..3] so +scale(0..7) stays in [1..10].
  std::vector<uint32_t> h_cb(2u * 1024u);
  for (uint32_t idx = 0; idx < 2u * 1024u; ++idx) {
    uint32_t q = 0;
    for (uint32_t p = 0; p < 4u; ++p) {
      const uint32_t sign = ((idx * 3u + p) & 1u) << 7;
      const uint32_t ex = 1u + ((idx + p * 2u) % 3u);
      const uint32_t mn = (idx * 5u + p * 3u) & 7u;
      q |= (sign | (ex << 3) | mn) << (8u * p);
    }
    h_cb[idx] = q;
  }

  const uint32_t max_chunk_tok = 160u;
  std::vector<uint8_t> h_x8((size_t)max_chunk_tok * cols);
  std::vector<float> h_xs((size_t)max_chunk_tok * sc_cols);
  for (uint32_t t = 0; t < max_chunk_tok; ++t) {
    for (uint32_t c = 0; c < cols; ++c) {
      const uint32_t sign = ((t + c) & 1u) << 7;
      const uint32_t ex = 4u + ((t * 3u + c) % 4u); // [2^-3 .. 1.875]
      const uint32_t mn = (t * 7u + c * 5u) & 7u;
      h_x8[(size_t)t * cols + c] = (uint8_t)(sign | (ex << 3) | mn);
    }
    for (uint32_t kb = 0; kb < sc_cols; ++kb) {
      h_xs[(size_t)t * sc_cols + kb] = 0.25f + 0.125f * (float)((t + kb) % 5u);
    }
  }

  // Pre-decode weights on CPU for exact reference matmul.
  const uint32_t cbr = cols / 2u * 5u / 8u;
  const uint32_t fbr = cols / 16u * 6u / 8u;
  std::vector<float> h_w_ref((size_t)rows * cols);
  for (uint32_t r = 0; r < rows; ++r) {
    const uint8_t* crow = h_codes.data() + (size_t)r * cbr;
    const uint8_t* frow = h_fields.data() + (size_t)r * fbr;
    for (uint32_t b = 0; b < cols / 16u; ++b) {
      const uint32_t fb = b * 6u, fi = fb >> 3;
      const uint32_t fw = (uint32_t)frow[fi] |
                          ((fb & 7u) > 2u ? ((uint32_t)frow[fi + 1] << 8) : 0u);
      const uint32_t field = (fw >> (fb & 7u)) & 63u;
      const uint32_t add = (field & 7u) * 0x08080808u;
      const uint32_t neg = (field & 8u) * 0x10101010u;
      const uint32_t rsh = (field & 16u) ? 16u : 0u;
      const uint32_t var = (field >> 5) & 1u;
      uint64_t win = 0;
      for (uint32_t i = 0; i < 5u; ++i) win |= (uint64_t)crow[(size_t)b * 5u + i] << (8u * i);
      for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t code = (uint32_t)((win >> (10u * i)) & 1023u);
        uint32_t q = (h_cb[var * 1024u + code] + add) ^ neg;
        if (rsh) q = (q >> 16) | (q << 16);
        for (uint32_t p = 0; p < 4u; ++p) {
          h_w_ref[(size_t)r * cols + b * 16u + i * 4u + p] =
              e4m3_to_f32_cpu((uint8_t)((q >> (8u * p)) & 0xFFu));
        }
      }
    }
  }

  uint8_t *d_codes = nullptr, *d_fields = nullptr, *d_x8 = nullptr;
  uint16_t* d_rscales = nullptr;
  uint32_t* d_tok = nullptr;
  float *d_xs = nullptr, *d_cb = nullptr, *d_y0 = nullptr, *d_y1 = nullptr;
  aff::ExpertTile* d_tiles = nullptr;
  AFF_HIP_CHECK(hipMalloc(&d_codes, code_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_fields, field_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_rscales, rows * sizeof(uint16_t)));
  AFF_HIP_CHECK(hipMalloc(&d_x8, h_x8.size()));
  AFF_HIP_CHECK(hipMalloc(&d_xs, h_xs.size() * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_cb, h_cb.size() * sizeof(uint32_t)));
  AFF_HIP_CHECK(hipMalloc(&d_tok, 128u * sizeof(uint32_t)));
  AFF_HIP_CHECK(hipMalloc(&d_y0, (size_t)rows * 256u * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_y1, (size_t)rows * 256u * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_tiles, 2u * sizeof(aff::ExpertTile)));

  AFF_HIP_CHECK(hipMemcpy(d_codes, tr_codes.data(), code_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_fields, tr_fields.data(), field_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_rscales, h_rscales.data(), rows * sizeof(uint16_t), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_x8, h_x8.data(), h_x8.size(), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_xs, h_xs.data(), h_xs.size() * sizeof(float), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_cb, h_cb.data(), h_cb.size() * sizeof(uint32_t), hipMemcpyHostToDevice));

  aff::expert_set_plane_layout_colgroup(true);
  aff::expert_set_wire_format(6u); // kFmtQuad2S
  aff::expert_set_recon_tiles(0u);

  for (uint32_t nt : {1u, 5u, 16u, 32u, 64u, 128u}) {
    std::vector<uint32_t> h_tok(nt);
    for (uint32_t t = 0; t < nt; ++t) h_tok[t] = (t * 7u + 13u) % max_chunk_tok;
    AFF_HIP_CHECK(hipMemcpy(d_tok, h_tok.data(), nt * sizeof(uint32_t), hipMemcpyHostToDevice));

    // Tile 0 uses permuted `tok`, tile 1 uses identity (`tok == nullptr`), sharing ld = 2 * nt.
    const uint32_t ld = 2u * nt;
    std::vector<aff::ExpertTile> h_tiles(2);
    for (uint32_t ti = 0; ti < 2u; ++ti) {
      h_tiles[ti].data = d_codes;
      h_tiles[ti].fields = d_fields;
      h_tiles[ti].rscales = d_rscales;
      h_tiles[ti].tok = (ti == 0u) ? d_tok : nullptr;
      h_tiles[ti].y = d_y0 + ti * nt;
      h_tiles[ti].nt = nt;
      h_tiles[ti].ld = ld;
    }
    AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), 2u * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
    aff::expert_gemm_wmma_hip(d_tiles, 2u, nt, d_x8, d_xs, d_cb, rows, cols, wscale, nullptr, false);
    AFF_HIP_CHECK(hipDeviceSynchronize());

    for (uint32_t ti = 0; ti < 2u; ++ti) h_tiles[ti].y = d_y1 + ti * nt;
    AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), 2u * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
    aff::expert_gemm_wmma_hip(d_tiles, 2u, nt, d_x8, d_xs, d_cb, rows, cols, wscale, nullptr, false);
    AFF_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> y0((size_t)rows * ld), y1((size_t)rows * ld);
    AFF_HIP_CHECK(hipMemcpy(y0.data(), d_y0, y0.size() * sizeof(float), hipMemcpyDeviceToHost));
    AFF_HIP_CHECK(hipMemcpy(y1.data(), d_y1, y1.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) != 0) {
      std::fprintf(stderr, "FAIL: non-deterministic repeat at nt=%u\n", nt);
      std::exit(1);
    }

    float max_err = 0.0f;
    for (uint32_t ti = 0; ti < 2u; ++ti) {
      for (uint32_t r = 0; r < rows; ++r) {
        const float rs = bf16_to_f32_cpu(h_rscales[r]) * wscale;
        for (uint32_t t = 0; t < nt; ++t) {
          const uint32_t tid = (ti == 0u) ? h_tok[t] : t;
          double acc = 0.0;
          for (uint32_t kb = 0; kb < sc_cols; ++kb) {
            double blk = 0.0;
            for (uint32_t c = kb * 128u; c < (kb + 1u) * 128u; ++c) {
              blk += (double)h_w_ref[(size_t)r * cols + c] *
                     (double)e4m3_to_f32_cpu(h_x8[(size_t)tid * cols + c]);
            }
            acc += blk * (double)h_xs[(size_t)tid * sc_cols + kb];
          }
          const float ref = (float)(acc * (double)rs);
          const float got = y0[(size_t)r * ld + ti * nt + t];
          const float err = std::fabs(got - ref);
          if (err > max_err) max_err = err;
        }
      }
    }
    if (max_err > 1e-4f) {
      std::fprintf(stderr, "FAIL: Quad2S TR parity at nt=%u max_abs_err=%g > 1e-4\n", nt, max_err);
      std::exit(1);
    }
  }

  (void)hipFree(d_codes); (void)hipFree(d_fields); (void)hipFree(d_rscales);
  (void)hipFree(d_x8); (void)hipFree(d_xs); (void)hipFree(d_cb); (void)hipFree(d_tok);
  (void)hipFree(d_y0); (void)hipFree(d_y1); (void)hipFree(d_tiles);
}

static void test_scatter_slots_parity() {
  for (uint32_t dim : {131u, 256u}) {
    for (uint32_t nb_tok : {1u, 6u, 16u, 64u, 256u}) {
      const uint32_t nb_stride = nb_tok + 5u;
      std::vector<uint32_t> h_run(nb_tok + 1u, 0u);
      for (uint32_t t = 0; t < nb_tok; ++t) {
        const uint32_t cnt = (t % 7u == 3u) ? 0u : (1u + (t % 5u));
        h_run[t + 1u] = h_run[t] + cnt;
      }
      const uint32_t total = h_run.back();
      std::vector<uint32_t> h_ord(total);
      for (uint32_t i = 0; i < total; ++i) h_ord[i] = i;
      for (uint32_t i = 0; i + 1u < total; i += 2u) {
        std::swap(h_ord[i], h_ord[(i * 7u + 1u) % total]);
      }

      std::vector<uint16_t> h_y((size_t)dim * total);
      for (size_t i = 0; i < h_y.size(); ++i) {
        const float v = ((float)((int)(i % 97u) - 48)) * 0.125f;
        h_y[i] = f32_to_bf16_cpu(v);
      }

      std::vector<float> h_block_init((size_t)dim * nb_stride);
      for (size_t i = 0; i < h_block_init.size(); ++i) {
        h_block_init[i] = ((float)((int)(i % 31u) - 15)) * 0.25f;
      }
      std::vector<float> h_block_ref = h_block_init;
      for (uint32_t d = 0; d < dim; ++d) {
        for (uint32_t t = 0; t < nb_tok; ++t) {
          const uint32_t a = h_run[t], b = h_run[t + 1u];
          if (a == b) continue;
          float acc = 0.0f;
          for (uint32_t i = a; i < b; ++i) {
            acc += bf16_to_f32_cpu(h_y[(size_t)d * total + h_ord[i]]);
          }
          h_block_ref[(size_t)d * nb_stride + t] += acc;
        }
      }

      uint16_t* d_y = nullptr;
      float* d_block = nullptr;
      uint32_t *d_ord = nullptr, *d_run = nullptr;
      AFF_HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(uint16_t)));
      AFF_HIP_CHECK(hipMalloc(&d_block, h_block_init.size() * sizeof(float)));
      AFF_HIP_CHECK(hipMalloc(&d_ord, std::max<size_t>(1u, h_ord.size()) * sizeof(uint32_t)));
      AFF_HIP_CHECK(hipMalloc(&d_run, h_run.size() * sizeof(uint32_t)));

      AFF_HIP_CHECK(hipMemcpy(d_y, h_y.data(), h_y.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
      AFF_HIP_CHECK(hipMemcpy(d_block, h_block_init.data(), h_block_init.size() * sizeof(float), hipMemcpyHostToDevice));
      AFF_HIP_CHECK(hipMemcpy(d_ord, h_ord.data(), h_ord.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
      AFF_HIP_CHECK(hipMemcpy(d_run, h_run.data(), h_run.size() * sizeof(uint32_t), hipMemcpyHostToDevice));

      aff::expert_scatter_slots_hip(d_y, d_block, d_ord, d_run, dim, nb_tok, nb_stride, total);
      AFF_HIP_CHECK(hipDeviceSynchronize());

      std::vector<float> h_block_got(h_block_init.size());
      AFF_HIP_CHECK(hipMemcpy(h_block_got.data(), d_block, h_block_got.size() * sizeof(float), hipMemcpyDeviceToHost));

      float max_err = 0.0f;
      for (size_t i = 0; i < h_block_ref.size(); ++i) {
        const float err = std::fabs(h_block_got[i] - h_block_ref[i]);
        if (err > max_err) max_err = err;
      }
      if (max_err != 0.0f) {
        std::fprintf(stderr, "FAIL: scatter parity dim=%u nb_tok=%u max_abs_err=%g\n",
                     dim, nb_tok, max_err);
        std::exit(1);
      }

      (void)hipFree(d_y); (void)hipFree(d_block); (void)hipFree(d_ord); (void)hipFree(d_run);
    }
  }
}

int main() {
  test_tr_mt1_pack_identity();
  if (!aff::hip_available()) {
    std::printf("PASS: test_moe_recon_tiles (CPU bit-identity verified; no gfx12 GPU)\n");
    return 0;
  }

  // Test GPU expert_gemm_wmma_hip with and without AFF_MOE_RECON_TILES=2 across 6 tiles
  const uint32_t rows = 128, cols = 512, n_tiles = 6, nt = 8;
  const size_t code_bytes = (size_t)rows * (cols / 2 * 5 / 8);
  const size_t field_bytes = (size_t)rows * (cols / 16 * 6 / 8);

  std::vector<uint8_t> h_codes(code_bytes), h_fields(field_bytes);
  for (size_t i = 0; i < code_bytes; ++i) h_codes[i] = (uint8_t)((i * 37 + 13) & 0xFF);
  for (size_t i = 0; i < field_bytes; ++i) h_fields[i] = (uint8_t)((i * 17 + 3) & 0x3F);

  std::vector<uint8_t> tr_codes(code_bytes), tr_fields(field_bytes);
  aff::expert_transpose_plane_host(h_codes.data(), h_fields.data(),
                                   tr_codes.data(), tr_fields.data(), rows, cols);

  std::vector<uint16_t> h_rscales(rows, 0x3F80u); // 1.0 in bf16

  std::vector<uint8_t> h_x8(nt * cols, 0x38u);
  std::vector<float> h_xs(nt * (cols / 128), 1.0f);
  std::vector<uint32_t> h_cb(2 * 1024, 0x38383838u);

  uint8_t *d_codes = nullptr, *d_fields = nullptr, *d_x8 = nullptr;
  uint16_t *d_rscales = nullptr;
  float *d_xs = nullptr, *d_cb = nullptr, *d_y_full = nullptr, *d_y_sliced = nullptr;
  aff::ExpertTile* d_tiles = nullptr;

  AFF_HIP_CHECK(hipMalloc(&d_codes, code_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_fields, field_bytes));
  AFF_HIP_CHECK(hipMalloc(&d_rscales, rows * sizeof(uint16_t)));
  AFF_HIP_CHECK(hipMalloc(&d_x8, h_x8.size()));
  AFF_HIP_CHECK(hipMalloc(&d_xs, h_xs.size() * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_cb, h_cb.size() * sizeof(uint32_t)));
  AFF_HIP_CHECK(hipMalloc(&d_y_full, (size_t)rows * n_tiles * nt * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_y_sliced, (size_t)rows * n_tiles * nt * sizeof(float)));
  AFF_HIP_CHECK(hipMalloc(&d_tiles, n_tiles * sizeof(aff::ExpertTile)));

  AFF_HIP_CHECK(hipMemcpy(d_codes, tr_codes.data(), code_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_fields, tr_fields.data(), field_bytes, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_rscales, h_rscales.data(), rows * sizeof(uint16_t), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_x8, h_x8.data(), h_x8.size(), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_xs, h_xs.data(), h_xs.size() * sizeof(float), hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_cb, h_cb.data(), h_cb.size() * sizeof(uint32_t), hipMemcpyHostToDevice));

  aff::expert_set_plane_layout_colgroup(true);
  aff::expert_set_wire_format(6u); // kFmtQuad2S

  std::vector<aff::ExpertTile> h_tiles(n_tiles);
  for (uint32_t t = 0; t < n_tiles; ++t) {
    h_tiles[t].data = d_codes;
    h_tiles[t].fields = d_fields;
    h_tiles[t].rscales = d_rscales;
    h_tiles[t].tok = nullptr;
    h_tiles[t].y = d_y_full + t * nt;
    h_tiles[t].nt = nt;
    h_tiles[t].ld = n_tiles * nt;
  }
  AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), n_tiles * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
  aff::expert_set_recon_tiles(0u);
  aff::expert_gemm_wmma_hip(d_tiles, n_tiles, nt, d_x8, d_xs, d_cb, rows, cols, 1.0f, nullptr, false);
  AFF_HIP_CHECK(hipDeviceSynchronize());

  for (uint32_t t = 0; t < n_tiles; ++t) h_tiles[t].y = d_y_sliced + t * nt;
  AFF_HIP_CHECK(hipMemcpy(d_tiles, h_tiles.data(), n_tiles * sizeof(aff::ExpertTile), hipMemcpyHostToDevice));
  aff::expert_set_recon_tiles(2u);
  aff::expert_gemm_wmma_hip(d_tiles, n_tiles, nt, d_x8, d_xs, d_cb, rows, cols, 1.0f, nullptr, false);
  AFF_HIP_CHECK(hipDeviceSynchronize());
  aff::expert_set_recon_tiles(0u);

  std::vector<float> y0((size_t)rows * n_tiles * nt), y1((size_t)rows * n_tiles * nt);
  AFF_HIP_CHECK(hipMemcpy(y0.data(), d_y_full, y0.size() * sizeof(float), hipMemcpyDeviceToHost));
  AFF_HIP_CHECK(hipMemcpy(y1.data(), d_y_sliced, y1.size() * sizeof(float), hipMemcpyDeviceToHost));

  if (std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: sliced recon tiles output differs from full launch\n");
    return 1;
  }

  (void)hipFree(d_codes); (void)hipFree(d_fields); (void)hipFree(d_rscales);
  (void)hipFree(d_x8); (void)hipFree(d_xs); (void)hipFree(d_cb);
  (void)hipFree(d_y_full); (void)hipFree(d_y_sliced); (void)hipFree(d_tiles);
  test_quad2s_tr_parity();
  test_scatter_slots_parity();
  std::printf("PASS: test_moe_recon_tiles\n");
  return 0;
}
