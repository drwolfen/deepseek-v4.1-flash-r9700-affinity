#include "gpu/expert_kernel_hip.h"
#include "gpu/hip_common.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Verify that packing 20 bytes into g32[0..4] and extracting via coalesced
// (u32, u32, u16) for sblk=0 and (u16, u32, u32) for sblk=2 is bit-for-byte
// identical to the byte-by-byte loop for every possible byte pattern.
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
    for (uint32_t bb = 0; bb < 2u; ++bb) {
      uint64_t ref_win = 0;
      const uint8_t* cp = raw + sblk * 5u + bb * 5u;
      for (uint32_t i = 0; i < 5u; ++i) ref_win |= (uint64_t)cp[i] << (8u * i);

      uint64_t got_win = 0;
      if (sblk == 0u) {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFu) << 32))
                      : ((uint64_t)(tw1 >> 8) | ((uint64_t)tw2 << 24));
      } else {
        got_win = (bb == 0u)
                      ? ((uint64_t)tw0 | ((uint64_t)(tw1 & 0xFFFFFFu) << 16))
                      : ((uint64_t)(tw1 >> 24) | ((uint64_t)tw2 << 8));
      }
      if (got_win != ref_win) {
        std::fprintf(stderr, "FAIL: sblk=%u bb=%u got=0x%010llx ref=0x%010llx\n",
                     sblk, bb, (unsigned long long)got_win, (unsigned long long)ref_win);
        std::exit(1);
      }
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
  std::printf("PASS: test_moe_recon_tiles\n");
  return 0;
}
