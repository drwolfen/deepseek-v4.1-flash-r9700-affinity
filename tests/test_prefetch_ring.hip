#include "gpu/moe_plan.h"
#include "gpu/hip_common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  if (!aff::hip_available()) {
    std::printf("PASS: test_prefetch_ring (no gfx12 GPU)\n");
    return 0;
  }

  const uint32_t stride = 4096;
  const uint32_t pref_cap = 4;
  std::vector<uint8_t> h_pool0(stride), h_pool1(stride);
  for (uint32_t i = 0; i < stride; ++i) {
    h_pool0[i] = (uint8_t)((i * 13 + 5) & 0xFF);
    h_pool1[i] = (uint8_t)((i * 29 + 11) & 0xFF);
  }

  uint8_t *d_pool0 = nullptr, *d_pool1 = nullptr, *d_pref_slab = nullptr;
  const uint8_t** d_pref_src = nullptr;
  aff::MoePlanStatus* d_st = nullptr;

  AFF_HIP_CHECK(hipMalloc(&d_pool0, stride));
  AFF_HIP_CHECK(hipMalloc(&d_pool1, stride));
  AFF_HIP_CHECK(hipMalloc(&d_pref_slab, (size_t)pref_cap * stride));
  AFF_HIP_CHECK(hipMalloc(&d_pref_src, pref_cap * sizeof(uint8_t*)));
  AFF_HIP_CHECK(hipMalloc(&d_st, sizeof(aff::MoePlanStatus)));

  AFF_HIP_CHECK(hipMemcpy(d_pool0, h_pool0.data(), stride, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemcpy(d_pool1, h_pool1.data(), stride, hipMemcpyHostToDevice));
  AFF_HIP_CHECK(hipMemset(d_pref_slab, 0, (size_t)pref_cap * stride));

  const uint8_t* h_srcs[4] = {d_pool0, d_pool1, nullptr, nullptr};
  AFF_HIP_CHECK(hipMemcpy(d_pref_src, h_srcs, sizeof(h_srcs), hipMemcpyHostToDevice));

  aff::MoePlanStatus h_st{};
  h_st.n_res = 3;
  h_st.n_uniq = 5; // 2 pooled experts -> should copy slots 0 and 1
  AFF_HIP_CHECK(hipMemcpy(d_st, &h_st, sizeof(h_st), hipMemcpyHostToDevice));

  aff::moe_prefetch_pack_hip(d_pref_slab, d_pref_src, d_st, stride, pref_cap, nullptr);
  AFF_HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint8_t> got0(stride), got1(stride);
  AFF_HIP_CHECK(hipMemcpy(got0.data(), d_pref_slab, stride, hipMemcpyDeviceToHost));
  AFF_HIP_CHECK(hipMemcpy(got1.data(), d_pref_slab + stride, stride, hipMemcpyDeviceToHost));

  if (std::memcmp(got0.data(), h_pool0.data(), stride) != 0 ||
      std::memcmp(got1.data(), h_pool1.data(), stride) != 0) {
    std::fprintf(stderr, "FAIL: moe_prefetch_pack_hip did not copy pooled shards accurately\n");
    return 1;
  }

  (void)hipFree(d_pool0); (void)hipFree(d_pool1);
  (void)hipFree(d_pref_slab); (void)hipFree(d_pref_src); (void)hipFree(d_st);
  std::printf("PASS: test_prefetch_ring\n");
  return 0;
}
