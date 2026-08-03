FROM debian:stretch

COPY sources.list /etc/apt/sources.list

# stretch is archived (archive.debian.org): its Release files are expired, so the
# valid-until check must be disabled for every apt call.
ENV APT_OPTS="-o Acquire::Check-Valid-Until=false"

# System toolchain + qemu + the aarch64/armhf cross compilers cross.sh needs for the
# -stretch targets. Baked once so CI neither reinstalls it nor hits archive.debian.org
# per run.
RUN apt-get $APT_OPTS update && apt-get $APT_OPTS install -y --no-install-recommends \
        ca-certificates curl wget build-essential sudo clang llvm \
        python3 python3-numpy jshon libssl-dev pkg-config \
        qemu-system-arm qemu-user \
        binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
        binutils-arm-linux-gnueabihf gcc-arm-linux-gnueabihf libc6-dev-armhf-cross \
    && rm -rf /var/lib/apt/lists/*

# rustup + current stable (matches rust-toolchain.toml's channel = "stable") + the stretch
# cross targets. Rebuilt weekly (see .github/workflows/cross-image.yml) so "stable" tracks
# the current release within a week; nothing here is frozen long-term.
ENV RUSTUP_HOME=/root/.rustup \
    CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain stable -c rustfmt -c clippy \
    && rustup target add aarch64-unknown-linux-gnu armv7-unknown-linux-gnueabihf

# cargo-dinghy, pinned to the version cross.sh downloads at runtime otherwise.
RUN curl -sSfL https://github.com/snipsco/dinghy/releases/download/0.8.0/cargo-dinghy-linux-0.8.0.tgz \
      | tar xz -C /tmp \
    && mv /tmp/cargo-dinghy-0.8.0/cargo-dinghy /root/.cargo/bin/ \
    && rm -rf /tmp/cargo-dinghy-0.8.0

# Tells ci-system-setup.sh / cross.sh the toolchain + apt deps are already present, so they
# skip the per-run install. Only ever set inside this image; every other path is unaffected.
ENV TRACT_PREBUILT_CI=1
