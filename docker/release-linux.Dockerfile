# The machine that builds every published Linux artifact.
#
# WHY IT EXISTS AT ALL. mdbx.c (crate mdbx-sys) does `#define _GNU_SOURCE`
# itself. In glibc >= 2.38 `features.h` turns `_GNU_SOURCE` into
# `_ISOC2X_SOURCE`, so `stdlib.h` redirects `strtol` to
# `__isoc23_strtol@GLIBC_2.38`. A node built on Ubuntu 24.04 therefore imports
# a 2.38 symbol and refuses to start on Ubuntu 22.04, Debian 12 and Rocky 9 —
# the three bases most miners rent. That exact binary shipped in v1.1.0 and was
# produced again twice since.
#
# The guard is `_GNU_SOURCE`, not `__STDC_VERSION__`, so no `-std=` value and
# no CFLAG prevents it; `-DMDBX_DISABLE_GNU_SOURCE=1` does not compile. The only
# lever that works is compiling against glibc headers older than 2.38, which is
# what this image is: Ubuntu 22.04, glibc 2.35, producing a GLIBC_2.34 floor.
#
# This file used to exist only as an image on one machine. It is in the
# repository now so the release can be rebuilt from a clone.
#
# Build and use:
#   docker build -t jetsam-build:22.04 -f docker/release-linux.Dockerfile docker
#   docker run --rm -v "$PWD":/src -w /src jetsam-build:22.04 \
#     ./scripts/build_release.sh --pack /packs/v1 --pack-v1-3 /packs/v1-3
#
# scripts/build_release.sh verifies the result with objdump regardless of where
# it ran, so a binary that escapes this image is still caught before packaging.

# Pinned by digest: "22.04" moves, and the glibc version is the whole point.
FROM ubuntu:22.04@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751

ENV DEBIAN_FRONTEND=noninteractive

# build-essential brings gcc and binutils (objdump, used by the release gate).
# clang/libclang-dev are what bindgen needs for mdbx-sys.
# appstream (appstreamcli) and dpkg-deb are what scripts/release/package_linux_gui.sh
# requires to build and validate the GUI .deb.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      build-essential \
      curl \
      pkg-config \
      clang \
      libclang-dev \
      git \
      ca-certificates \
      appstream \
      dpkg-dev \
 && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/cargo-home
ENV PATH=/cargo-home/bin:$PATH

# The toolchain matches rust-toolchain.toml, including its rustfmt component,
# so the container needs no network once built.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y \
      --default-toolchain 1.96.0 \
      --profile minimal \
      --component rustfmt

# Fail the image build, not a release twelve minutes in, if either of the two
# things this image is for is wrong.
RUN rustc --version \
 && ldd --version | head -1 \
 && test "$(ldd --version | head -1 | awk '{print $NF}')" = "2.35" \
 && command -v objdump >/dev/null \
 && command -v appstreamcli >/dev/null

WORKDIR /src
CMD ["/bin/bash"]
