# syntax=docker/dockerfile:1.7
#
# Universal dsx build. Three backend targets, one recipe:
#
#   nvidia (default)  dsx-server, dsx, prefill-bench, t16-prefill-audit with a CUDA fatbin for CUDA_ARCHS
#   amd               dsx-amd (HIP, gfx942) with the ROCm runtime libraries it links against
#   cpu               dsx-server / dsx / cpubench without CUDA (CPU reference, GGUF tools, `--mock` server)
#
# Examples:
#   docker build -t dsx:nvidia .                                             # sm_86 + sm_89 + sm_120
#   docker build -t dsx:3090 --build-arg CUDA_ARCHS=86 .                     # one arch, smaller fatbin
#   docker build -t dsx:nvidia --build-arg STRIP=1 .                         # drop debug symbols
#   docker build -t dsx:amd --target amd .
#     NOTE: the ROCm dev image is ~8 GB compressed (one 7.77 GB layer). Pull it on a host with a fast link
#     or in CI; over a slow single-connection line the pull alone takes hours. The AMD stage stays in this
#     file and is not built by a plain `docker build .` (default target is nvidia).
#
# Portability: .cargo/config.toml pins `-C target-cpu=native` for local builds, so an image built on one box
# can SIGILL on another. The image replaces it with CPU_TARGET=x86-64-v3 (AVX2/FMA/BMI2), the documented
# baseline of this project. The AVX-VNNI / AVX-512 paths stay available: those kernels carry
# #[target_feature] and are picked per CPU by is_x86_feature_detected!. CPU_TARGET=native reproduces the
# old host-specific behaviour; CPU_TARGET=x86-64-v2 widens the baseline further.
#
# Build-arg knobs forwarded to nvidia/build.rs (same as a local build):
#   --build-arg DSX_KDEFS="-DABL=5" --build-arg DSX_KS=2 --build-arg DSX_ROWT=...
# Leave them unset for the defaults; an empty value is passed through as empty and is rejected by nvcc.

ARG UBUNTU_VERSION=24.04
ARG RUST_VERSION=1.97.1
ARG CUDA_VERSION=13.2.0
ARG ROCM_VERSION=7.14.1
ARG CUDA_ARCHS=86,89,120
ARG CPU_TARGET=x86-64-v3
ARG STRIP=0

###############################################################################
# Source stage: the manifests only, so a dependency fetch stays cached when code changes.
###############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS source

WORKDIR /dsx
COPY Cargo.toml Cargo.lock ./
COPY core/Cargo.toml core/
COPY nvidia/Cargo.toml nvidia/
COPY amd/Cargo.toml amd/
# cargo fetch parses the workspace members, which need at least one target; the real sources overwrite these.
RUN for c in core nvidia amd; do mkdir -p "$c/src"; : > "$c/src/lib.rs"; done

###############################################################################
# Builder: NVIDIA (CUDA fatbin via nvcc from the devel image)
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder-nvidia

ARG RUST_VERSION
ARG CUDA_ARCHS
ARG CPU_TARGET
ARG STRIP
ARG DSX_KDEFS
ARG DSX_KS
ARG DSX_ROWT

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TERM_COLOR=always \
    CARGO_INCREMENTAL=0 \
    RUSTFLAGS="-C target-cpu=${CPU_TARGET}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libc6-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# pinned toolchain (edition 2024 needs >= 1.85); no rust-toolchain.toml in the repo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" --no-modify-path \
    && rustc --version && cargo --version

WORKDIR /dsx
COPY --from=source /dsx /dsx
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo fetch --locked

COPY core core
COPY nvidia nvidia
COPY amd amd

# CUDA_ARCHS/DSX_* are build args; the ones left unset stay unset for build.rs. The target directory is a
# cache mount (not part of the layer), so the binaries are copied out in the same step.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/dsx/target,sharing=locked \
    cargo build --locked --release -p dsx --features cuda \
    && mkdir -p /out \
    && cp target/release/dsx-server target/release/dsx \
          target/release/prefill-bench target/release/t16-prefill-audit /out/ \
    && if [ "${STRIP}" = "1" ]; then strip --strip-debug /out/*; fi

###############################################################################
# Builder: AMD (hipcc + hipBLAS from the ROCm dev image)
#
# Pull cost: rocm/dev-ubuntu-24.04:<ver>-full is ~8 GB compressed, one 7.77 GB layer. Build this target on
# a host with a fast link or in CI; it is deliberately not reachable from the default `docker build .`.
###############################################################################
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-full AS builder-amd

ARG RUST_VERSION
ARG CPU_TARGET
ARG STRIP

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/opt/rocm/bin:$PATH \
    CARGO_TERM_COLOR=always \
    CARGO_INCREMENTAL=0 \
    HIPCC=/opt/rocm/bin/hipcc \
    RUSTFLAGS="-C target-cpu=${CPU_TARGET}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" --no-modify-path \
    && rustc --version && cargo --version

WORKDIR /dsx
COPY --from=source /dsx /dsx
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo fetch --locked

COPY core core
COPY nvidia nvidia
COPY amd amd

# amd/build.rs compiles kernels/dsxa.hip for gfx942 (the MI325X); the host links -lamdhip64 and -lhipblas.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/dsx/target,sharing=locked \
    cargo build --locked --release -p dsx-amd \
    && mkdir -p /out \
    && cp target/release/dsx-amd /out/ \
    && if [ "${STRIP}" = "1" ]; then strip --strip-debug /out/*; fi

###############################################################################
# Builder: CPU-only (no CUDA: nvidia/build.rs writes an empty fatbin, no nvcc needed)
###############################################################################
FROM rust:${RUST_VERSION}-bookworm AS builder-cpu

ARG CPU_TARGET
ARG STRIP

ENV CARGO_TERM_COLOR=always \
    CARGO_INCREMENTAL=0 \
    RUSTFLAGS="-C target-cpu=${CPU_TARGET}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /dsx
COPY --from=source /dsx /dsx
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo fetch --locked

COPY core core
COPY nvidia nvidia
COPY amd amd

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/dsx/target,sharing=locked \
    cargo build --locked --release -p dsx-core -p dsx \
    && mkdir -p /out \
    && cp target/release/dsx-server target/release/dsx target/release/cpubench /out/ \
    && if [ "${STRIP}" = "1" ]; then strip --strip-debug /out/*; fi

###############################################################################
# Runtime: AMD
###############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS amd

ARG ROCM_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# libamdhip64/libhipblas dlopen their own deps from /opt/rocm/lib; the Ubuntu packages cover libelf/libnuma/libdrm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libstdc++6 \
        libelf1 \
        libnuma1 \
        libdrm2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-amd /opt/rocm/lib /opt/rocm/lib
COPY --from=builder-amd /out/dsx-amd /usr/local/bin/dsx-amd

ENV LD_LIBRARY_PATH=/opt/rocm/lib
WORKDIR /models
EXPOSE 8090
VOLUME ["/models"]
ENTRYPOINT ["dsx-amd"]
CMD ["--help"]
LABEL org.opencontainers.image.title="dsx-amd" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash inference on one AMD Instinct MI325X (ROCm ${ROCM_VERSION})" \
      org.opencontainers.image.source="https://github.com/lxe/dsx" \
      org.opencontainers.image.licenses="MIT"
RUN dsx-amd --help >/dev/null

###############################################################################
# Runtime: CPU-only
###############################################################################
FROM ubuntu:${UBUNTU_VERSION} AS cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cpu /out/ /usr/local/bin/

WORKDIR /models
EXPOSE 8090
VOLUME ["/models"]
ENTRYPOINT ["dsx-server"]
CMD ["--help"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/health || exit 1
LABEL org.opencontainers.image.title="dsx-cpu" \
      org.opencontainers.image.description="dsx without CUDA: CPU reference, GGUF tools and the mock server" \
      org.opencontainers.image.source="https://github.com/lxe/dsx" \
      org.opencontainers.image.licenses="MIT"
RUN dsx-server --help >/dev/null && dsx --help >/dev/null

###############################################################################
# Runtime: NVIDIA (default target; the last stage is what a plain `docker build .` produces)
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS nvidia

ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# cudarc loads these with dlopen and looks for the unversioned names; the runtime image ships only
# lib*.so.<major>. Derive the major from CUDA_VERSION instead of hardcoding 13.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && CUDA_MAJOR="${CUDA_VERSION%%.*}" \
    && for lib in libcublas libcublasLt libcudart; do \
           ln -sf "${lib}.so.${CUDA_MAJOR}" "/usr/local/cuda/lib64/${lib}.so"; \
       done \
    && ldconfig

COPY --from=builder-nvidia /out/ /usr/local/bin/

# The driver (libcuda.so.1) comes from the host through --gpus all; the kernel fatbin covers CUDA_ARCHS.
ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64
WORKDIR /models
EXPOSE 8090
VOLUME ["/models"]
ENTRYPOINT ["dsx-server"]
CMD ["--help"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/health || exit 1
LABEL org.opencontainers.image.title="dsx" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash inference engine (CUDA ${CUDA_VERSION})" \
      org.opencontainers.image.source="https://github.com/lxe/dsx" \
      org.opencontainers.image.licenses="MIT"
# smoke test: no GPU needed, --help exits before cudarc dlopens anything
RUN dsx-server --help >/dev/null && dsx --help >/dev/null
