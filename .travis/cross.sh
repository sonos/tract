#!/bin/sh

set -ex

ROOT=$(dirname $(dirname $(realpath $0)))
. $ROOT/.travis/ci-system-setup.sh

ensure_cargo_dinghy() {
    if which cargo-dinghy
    then
        return
    fi
    if [ `arch` = x86_64 -o `arch` = i386 -o `arch` = arm64 ]
    then
         if [ `uname` = "Darwin" ]
         then
             NAME=macos
         else
             NAME=linux
         fi
         VERSION=0.8.0
         URL=https://github.com/snipsco/dinghy/releases/download/$VERSION/cargo-dinghy-$NAME-$VERSION.tgz
         mkdir -p /tmp/cargo-dinghy
         # Subshell: the callers below all use paths relative to the checkout.
         ( cd /tmp/cargo-dinghy
           # wget gives up on a TLS error whatever --tries says, so retry from the shell.
           wget -nv $URL -O cargo-dinghy.tgz || ( sleep 5 ; wget -nv $URL -O cargo-dinghy.tgz )
           tar vzxf cargo-dinghy.tgz --strip-components 1
           mv cargo-dinghy $HOME/.cargo/bin )
    else
        cargo install cargo-dinghy
    fi
}

if [ -z "$PLATFORM" -a -n "$1" ]
then
    PLATFORM=$1
fi

case "$PLATFORM" in
    "raspbian")
        ensure_cargo_dinghy
        [ -e $HOME/cached/raspitools ] || git clone --depth 1 https://github.com/raspberrypi/tools $HOME/cached/raspitools
        TOOLCHAIN=$HOME/cached/raspitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf
        export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
        rustup target add $RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'\ntoolchain='$TOOLCHAIN'" > .dinghy.toml
        cargo dinghy --platform $PLATFORM build --release -p tract-ffi --no-default-features
        cargo dinghy --platform $PLATFORM build --release -p tract-cli \
            --no-default-features \
            --features "onnx,tf,pulse,pulse-opl,tflite,transformers,extra"
        ;;

    "aarch64-linux-android"|"armv7-linux-androideabi"|"i686-linux-android"|"x86_64-linux-android")
        ensure_cargo_dinghy
        case "$PLATFORM" in
            "aarch64-linux-android")
                ANDROID_CPU=aarch64
                RUSTC_TRIPLE=aarch64-linux-android
                ;;
            "armv7-linux-androideabi")
                ANDROID_CPU=armv7
                RUSTC_TRIPLE=armv7-linux-androideabi
                ;;
            "i686-linux-android")
                ANDROID_CPU=i686
                RUSTC_TRIPLE=i686-linux-android
                ;;
            "x86_64-linux-android")
                ANDROID_CPU=x86_64
                RUSTC_TRIPLE=x86_64-linux-android
                ;;
        esac

        export TARGET_AR=ar

        if [ -e /usr/local/lib/android/sdk/ndk-bundle ]
        then
            export ANDROID_NDK_HOME=/usr/local/lib/android/sdk/ndk-bundle
        else
            export ANDROID_SDK_HOME=$HOME/cached/android-sdk
            [ -e $ANDROID_SDK_HOME ] || ./.travis/android-ndk.sh
        fi

        rustup target add $RUSTC_TRIPLE
        cargo dinghy --platform auto-android-$ANDROID_CPU build -p tract-linalg -p tract-ffi
        ;;

    "aarch64-apple-ios")
        ensure_cargo_dinghy
        rustup target add aarch64-apple-ios
        cargo dinghy --platform auto-ios-aarch64 check -p tract-linalg -p tract-ffi
        ;;

    "aarch64-apple-darwin" | "x86_64-unknown-linux-gnu")
        RUSTC_TRIPLE=$PLATFORM
        rustup target add $RUSTC_TRIPLE
        cargo build --target $RUSTC_TRIPLE -p tract-cli --release
        ;;

    "aarch64-unknown-linux-gnu-stretch" | "armv7-unknown-linux-gnueabihf-stretch" | "x86_64-unknown-linux-gnu-stretch")
        INNER_PLATFORM=${PLATFORM%-stretch}
        # aarch64 stretch bench targets Jetson-class boxes that ship with CUDA 12;
        # the default CUDA 13 cudarc binding wouldn't run there (cudart symbol
        # rename across the 12/13 boundary).  Force cuda-12000 for that build only.
        # The Jetson leg wants cuda-12000; the dinghy boards reuse the same aarch64 stretch
        # platform but pass an explicit TRACT_CLI_FEATURES (no cuda), so only force cuda when
        # the caller has not chosen its own feature set.
        CUDA_FEATURE_ENV=""
        if [ "$PLATFORM" = "aarch64-unknown-linux-gnu-stretch" ] && [ -z "$TRACT_CLI_FEATURES" ]
        then
            CUDA_FEATURE_ENV="-e TRACT_CUDA_FEATURE=cuda-12000"
        fi
        # Prefer the prebuilt toolchain image (private ghcr package); log in when a token is
        # present (CI), otherwise build the bare image locally. A failed pull also falls back
        # to a local build, so forks and offline runs still work.
        STRETCH_IMAGE="${TRACT_CROSS_IMAGE:-ghcr.io/sonos/tract/cross-debian-stretch:latest}"
        [ -n "$GITHUB_TOKEN" ] && echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_ACTOR:-x}" --password-stdin
        if docker pull "$STRETCH_IMAGE"
        then
            STRETCH_TAG="$STRETCH_IMAGE"
        else
            (cd .travis/docker-debian-stretch; docker build --tag debian-stretch .)
            STRETCH_TAG=debian-stretch
        fi
        mkdir -p "$HOME/.cargo/registry" "$HOME/.cargo/git"
        docker run -v `pwd`:/tract -w /tract \
            -v "$HOME/.cargo/registry":/root/.cargo/registry \
            -v "$HOME/.cargo/git":/root/.cargo/git \
            -e CI=true \
            -e SKIP_QEMU_TEST=skip \
            -e CARGO_NET_RETRY \
            -e CARGO_HTTP_MULTIPLEXING \
            -e CARGO_REGISTRIES_CRATES_IO_PROTOCOL \
            -e TRACT_CLI_FEATURES \
            ${CARGO_TARGET_DIR:+-e CARGO_TARGET_DIR=$CARGO_TARGET_DIR} \
            -e PLATFORM=$INNER_PLATFORM $CUDA_FEATURE_ENV "$STRETCH_TAG" \
            ./.travis/cross.sh
        sudo chown -R `whoami` "$HOME/.cargo" .
        export RUSTC_TRIPLE=$INNER_PLATFORM
        ;;

    "aarch64-unknown-linux-gnu" | "armv6vfp-unknown-linux-gnueabihf" | "armv7-unknown-linux-gnueabihf" | \
        "aarch64-unknown-linux-musl" | "armv7-unknown-linux-musl" | "cortexa53-unknown-linux-musl" | \
        "riscv64gc-unknown-linux-musl" | \
        "riscv64gc-unknown-linux-gnu" | "rvv128-unknown-linux-gnu" )

        ensure_cargo_dinghy
        case "$PLATFORM" in
            "aarch64-unknown-linux-gnu")
                export ARCH=aarch64
                export QEMU_ARCH=aarch64
                export LIBC_ARCH=arm64
                export TRACT_CPU_AARCH64_KIND=a55
                export RUSTC_TRIPLE=$ARCH-unknown-linux-gnu
                export DEBIAN_TRIPLE=$ARCH-linux-gnu
                ;;
            "armv6vfp-unknown-linux-gnueabihf")
                export ARCH=armv6vfp
                export LIBC_ARCH=armhf
                export QEMU_ARCH=arm
                export QEMU_OPTS="-cpu cortex-a15"
                export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
                export DEBIAN_TRIPLE=arm-linux-gnueabihf
                ;;
            "armv7-unknown-linux-gnueabihf")
                export ARCH=armv7
                export QEMU_ARCH=arm
                export LIBC_ARCH=armhf
                export QEMU_OPTS="-cpu cortex-a15"
                export RUSTC_TRIPLE=armv7-unknown-linux-gnueabihf
                export DEBIAN_TRIPLE=arm-linux-gnueabihf
                export TARGET_CC=$DEBIAN_TRIPLE-gcc
                export TRACT_CPU_ARM32_NEON=true
                export DINGHY_TEST_ARGS="--env TRACT_CPU_ARM32_NEON=true"
                ;;
            "aarch64-unknown-linux-musl")
                export ARCH=aarch64
                export QEMU_ARCH=aarch64
                export LIBC_ARCH=arm64
                export RUSTC_TRIPLE=$ARCH-unknown-linux-musl
                export DEBIAN_TRIPLE=$ARCH-linux-gnu
                export TRACT_CPU_AARCH64_KIND=a55
                export CUSTOM_TC=`pwd`/aarch64-linux-musl-cross
                [ -d "$CUSTOM_TC" ] || curl -s https://tract-test-assets.tract.rs/toolchains/aarch64-linux-musl-cross.tgz | tar zx
                ;;
            "cortexa53-unknown-linux-musl")
                export ARCH=aarch64
                export QEMU_ARCH=aarch64
                export LIBC_ARCH=arm64
                export RUSTC_TRIPLE=$ARCH-unknown-linux-musl
                export DEBIAN_TRIPLE=$ARCH-linux-gnu
                export TRACT_CPU_AARCH64_KIND=a53
                export QEMU_OPTS="-cpu cortex-a53"
                export CUSTOM_TC=`pwd`/aarch64-linux-musl-cross
                [ -d "$CUSTOM_TC" ] || curl -s https://tract-test-assets.tract.rs/toolchains/aarch64-linux-musl-cross.tgz | tar zx
                ;;
            "armv7-unknown-linux-musl")
                export ARCH=armv7
                export QEMU_ARCH=arm
                export LIBC_ARCH=armhf
                export RUSTC_TRIPLE=armv7-unknown-linux-musleabihf
                export DEBIAN_TRIPLE=arm-linux-gnueabihf
                export CUSTOM_TC=`pwd`/armv7l-linux-musleabihf-cross
                export TRACT_CPU_ARM32_NEON=true
                export DINGHY_TEST_ARGS="--env TRACT_CPU_ARM32_NEON=true"
                [ -d "$CUSTOM_TC" ] || curl -s https://tract-test-assets.tract.rs/toolchains/armv7l-linux-musleabihf-cross.tgz | tar zx
                export TARGET_CFLAGS="-mfpu=neon"
                ;;
            "riscv64gc-unknown-linux-musl")
                export ARCH=riscv64
                export QEMU_ARCH=riscv64
                export RUSTC_TRIPLE=riscv64gc-unknown-linux-musl
                export CUSTOM_TC=`pwd`/riscv64-linux-musl-cross
                [ -d "$CUSTOM_TC" ] || curl -s https://tract-test-assets.tract.rs/toolchains/riscv64-linux-musl-cross.tgz | tar zx
                export TARGET_CC=$CUSTOM_TC/bin/riscv64-linux-musl-gcc
                # riscv64 musl defaults to dynamic linking (unlike the aarch64/armv7 musl
                # targets here); force a static binary so it runs on the glibc boards, which
                # have no musl loader.
                export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C target-feature=+crt-static"
                ;;
            # RVV is vector-length agnostic and the mmm kernels are gated on the
            # hart's VLEN, so the two entries below differ only in vlen: 256 is
            # the SpacemiT K1/X100 shape, 128 the Sophgo SG2044 one, and they
            # select disjoint halves of the kernel set.
            #
            # -cpu max rather than a profile model: rva23u64 would describe real
            # silicon more closely but predates neither the CI image's qemu nor
            # its glibc safely, and the generic rv64 model cannot run Debian's
            # riscv64 glibc at all (it SIGILLs on a trivial static binary).
            "riscv64gc-unknown-linux-gnu")
                export ARCH=riscv64
                export QEMU_ARCH=riscv64
                export LIBC_ARCH=riscv64
                export QEMU_OPTS="-cpu max,vlen=256"
                export RUSTC_TRIPLE=riscv64gc-unknown-linux-gnu
                export DEBIAN_TRIPLE=riscv64-linux-gnu
                ;;
            "rvv128-unknown-linux-gnu")
                export ARCH=riscv64
                export QEMU_ARCH=riscv64
                export LIBC_ARCH=riscv64
                export QEMU_OPTS="-cpu max,vlen=128"
                export RUSTC_TRIPLE=riscv64gc-unknown-linux-gnu
                export DEBIAN_TRIPLE=riscv64-linux-gnu
                ;;
            *)
                echo "unsupported platform $PLATFORM"
                exit 1
                ;;
        esac

        mkdir -p $ROOT/target/$RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'" > .dinghy.toml
        if [ -n "$DEBIAN_TRIPLE" ]
        then
            PACKAGES="$PACKAGES binutils-$DEBIAN_TRIPLE gcc-$DEBIAN_TRIPLE libc6-dev-$LIBC_ARCH-cross"
            echo "deb_multiarch='$DEBIAN_TRIPLE'" >> .dinghy.toml
        fi

        if [ -n "$CUSTOM_TC" ]
        then
            echo "toolchain='$CUSTOM_TC'" >> .dinghy.toml
        fi

        echo "[script_devices.qemu-$PLATFORM]\nplatform='$PLATFORM'\npath='$ROOT/target/$RUSTC_TRIPLE/qemu-$PLATFORM'" >> .dinghy.toml
        echo "#!/bin/sh\nexe=\$1\nshift\n/usr/bin/qemu-$QEMU_ARCH $QEMU_OPTS -L /usr/$DEBIAN_TRIPLE/ \$exe --test-threads 1 \"\$@\"" > $ROOT/target/$RUSTC_TRIPLE/qemu-$PLATFORM
        chmod +x $ROOT/target/$RUSTC_TRIPLE/qemu-$PLATFORM

        DINGHY_TEST_ARGS="$DINGHY_TEST_ARGS --env PROPTEST_MAX_SHRINK_ITERS=100000000"

        # The prebuilt image (TRACT_PREBUILT_CI) already carries qemu + the cross toolchains.
        if [ -z "$TRACT_PREBUILT_CI" ]
        then
            apt_retry apt-get -y install --no-install-recommends qemu-system-arm qemu-user libssl-dev pkg-config $PACKAGES
        fi
        rustup target add $RUSTC_TRIPLE
        if [ -z "$SKIP_QEMU_TEST" ]
        then
            qemu-$QEMU_ARCH --version
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS test --profile opt-no-lto -p tract-linalg -- --nocapture
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS test --profile opt-no-lto -p tract-core
        fi

        # TRACT_CUDA_FEATURE is the only signal that a leg targets a CUDA board; every
        # other linux cross target here is GPU-less, so keep cudarc out of its binaries.
        if [ -n "$TRACT_CUDA_FEATURE" ]
        then
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS check -p tract-ffi \
                --no-default-features --features "$TRACT_CUDA_FEATURE"
        else
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS check -p tract-ffi \
                --no-default-features
        fi
        # keep lto for these two are they're going to devices.
        if [ -n "$TRACT_CUDA_FEATURE" ]
        then
            cargo dinghy --platform $PLATFORM build --release \
                --no-default-features \
                --features "onnx,tf,pulse,pulse-opl,tflite,transformers,extra,bench-suite,$TRACT_CUDA_FEATURE" \
                -p tract-cli
        elif [ -n "$TRACT_CLI_FEATURES" ]
        then
            cargo dinghy --platform $PLATFORM build --release \
                --no-default-features --features "$TRACT_CLI_FEATURES" \
                -p tract-cli
        else
            cargo dinghy --platform $PLATFORM build --release \
                --no-default-features \
                --features "onnx,tf,pulse,pulse-opl,tflite,transformers,extra" \
                -p tract-cli
        fi
        ;;

    wasm32-wasi)
        PLATFORM=wasm32-wasip1
        wasmtime --version

        rustup target add $PLATFORM
        cargo check --target $PLATFORM --features getrandom-js -p tract-onnx -p tract-tensorflow
        RUSTFLAGS='-C target-feature=+simd128' CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
            cargo test --target=$PLATFORM -p tract-linalg -p tract-core -p test-unit-core
        # The wasm backend picks its multiply-add form, its int8 packing and its
        # sigmoid/tanh kernels at compile time on +relaxed-simd, so the run above
        # leaves that half of it untested.
        RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
            cargo test --target=$PLATFORM -p tract-linalg -p tract-core -p test-unit-core
        ;;
    wasm32-*)
        rustup target add $PLATFORM
        cargo check --target $PLATFORM --features getrandom-js -p tract-onnx -p tract-tensorflow
        ;;
    *)
        echo "Don't know what to do for platform: $PLATFORM"
        exit 2
        ;;
esac
