ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:10.1.0-full
FROM ${BASE_IMAGE}
ARG JOBS=16
ARG AFF_GIT_SHA=unknown
ARG AFF_VERSION=v5.11.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake ninja-build git bash sudo passwd python3 procps iproute2 curl ca-certificates rustc && rm -rf /var/lib/apt/lists/*
COPY . /opt/affinity-src
RUN ln -s /opt/affinity-src /opt/affinity
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PATH=/opt/affinity-src/build:/opt/rocm/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/lib
RUN cd /opt/affinity-src \
 && cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DAFF_HIP=ON \
      -DAFF_GPU_ARCH=gfx1201 \
      -DGPU_TARGETS=gfx1201 \
      -DAFF_VERSION=${AFF_VERSION} \
      -DAFF_GIT_SHA=${AFF_GIT_SHA} \
      -DAFF_CPU_ARCH=avx \
      -DAFF_HIPCC=/opt/rocm/bin/hipcc \
      -DCMAKE_CXX_COMPILER=g++ \
 && cmake --build build -j "$JOBS" \
 && (cp build/aff-quantize build/aff_quantize 2>/dev/null || true) \
 && mkdir -p /opt/affinity/bin \
 && for f in /opt/affinity-src/build/*; do \
      if [ -x "$f" ] && [ ! -d "$f" ]; then \
        ln -sf "$f" /usr/local/bin/$(basename "$f"); \
        ln -sf "$f" /opt/affinity/bin/$(basename "$f"); \
      fi; \
    done \
 && ls -la build/affinity \
      build/aff-info \
      build/aff-quantize \
      build/aff-cfgcheck \
      build/aff-fp8check \
      build/aff-exphash \
      build/aff-vitcheck \
      build/aff-engramcheck \
      build/aff-bench \
      build/test_prefix_cache \
      build/test_tool_namespace \
      build/test_cactus_accept \
      build/test_gpu_ports \
      build/test_moe_recon_tiles \
      build/test_o_fp8_qdq \
      build/test_hadamard_block32 \
      build/test_w4_head \
      build/test_thinking_budget \
      build/test_suffix_draft \
      build/test_wave2_kernels \
      build/test_v41_indexer \
      build/test_engram_split_gate \
      build/test_v510_kernels \
      build/test_v511_kernels \
      build/test_cpu_dispatch
RUN if getent passwd ubuntu >/dev/null; then userdel -r ubuntu 2>/dev/null || userdel ubuntu; fi \
 && if getent group ubuntu >/dev/null; then groupdel ubuntu 2>/dev/null || true; fi \
 && rm -rf /home/ubuntu /root/.bash_history /tmp/* /var/tmp/*
LABEL org.opencontainers.image.title="affinity" \
      org.opencontainers.image.version="${AFF_VERSION}" \
      org.opencontainers.image.revision="${AFF_GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/drwolfen/deepseek-v4.1-flash-r9700-affinity"
WORKDIR /opt
ENTRYPOINT ["/opt/affinity-src/build/affinity"]
