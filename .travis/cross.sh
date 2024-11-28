#!/bin/sh

set -ex

export DEBIAN_FRONTEND=noninteractive
export RUSTUP_TOOLCHAIN=1.75.0

if [ `whoami` != "root" ]
then
    SUDO=sudo
fi

if [ `uname` = "Linux" ]
then
    $SUDO rm -f /etc/apt/sources.list.d/dotnetdev.list /etc/apt/sources.list.d/microsoft-prod.list
    $SUDO apt-get update
    if [ -z "$TRAVIS" -a -z "$GITHUB_WORKFLOW" ]
    then
        $SUDO apt-get -y upgrade
        $SUDO apt-get install -y --no-install-recommends unzip wget curl python awscli build-essential sudo
    fi
else
    sysctl -n machdep.cpu.brand_string
    brew install coreutils
fi

ROOT=$(dirname $(dirname $(realpath $0)))

PATH=$PATH:$HOME/.cargo/bin

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y

which cargo-dinghy || ( mkdir -p /tmp/cargo-dinghy
if [ `arch` = x86_64 -o `arch` = i386 -o `arch` = arm64 ]
then
     cd /tmp/cargo-dinghy
     if [ `uname` = "Darwin" ]
     then
         NAME=macos
     else
         NAME=linux
     fi
     VERSION=0.8.0
     wget -q https://github.com/snipsco/dinghy/releases/download/$VERSION/cargo-dinghy-$NAME-$VERSION.tgz -O cargo-dinghy.tgz
     tar vzxf cargo-dinghy.tgz --strip-components 1
     mv cargo-dinghy $HOME/.cargo/bin
else
    cargo install cargo-dinghy
fi
)

case "$PLATFORM" in
    "raspbian")
        [ -e $HOME/cached/raspitools ] || git clone --depth 1 https://github.com/raspberrypi/tools $HOME/cached/raspitools
        TOOLCHAIN=$HOME/cached/raspitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf
        export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
        rustup target add $RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'\ntoolchain='$TOOLCHAIN'" > .dinghy.toml
        cargo dinghy --platform $PLATFORM build --release -p tract -p example-tensorflow-mobilenet-v2 -p tract-ffi
        ;;

    "aarch64-linux-android"|"armv7-linux-androideabi"|"i686-linux-android"|"x86_64-linux-android")
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
        rustup target add aarch64-apple-ios
        cargo dinghy --platform auto-ios-aarch64 build -p tract-linalg -p tract-ffi
        ;;

    "aarch64-apple-darwin")
        rustup target add aarch64-apple-darwin
        cargo build --target aarch64-apple-darwin -p tract
        ;;

    "aarch64-unknown-linux-gnu-stretch" | "armv7-unknown-linux-gnueabihf-stretch" )
        INNER_PLATFORM=${PLATFORM%-stretch}
        (cd .travis/docker-debian-stretch; docker build --tag debian-stretch .)
        docker run -v `pwd`:/tract -w /tract \
            -e SKIP_QEMU_TEST=skip \
            -e PLATFORM=$INNER_PLATFORM debian-stretch \
            ./.travis/cross.sh
        sudo chown -R `whoami` .
        export RUSTC_TRIPLE=$INNER_PLATFORM
        ;;

    "aarch64-unknown-linux-gnu" | "armv6vfp-unknown-linux-gnueabihf" | "armv7-unknown-linux-gnueabihf" | \
        "aarch64-unknown-linux-musl" | "armv7-unknown-linux-musl" | "cortexa53-unknown-linux-musl" )

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
                [ -d "$CUSTOM_TC" ] || curl -s https://s3.amazonaws.com/tract-ci-builds/toolchains/aarch64-linux-musl-cross.tgz | tar zx
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
                [ -d "$CUSTOM_TC" ] || curl -s https://s3.amazonaws.com/tract-ci-builds/toolchains/aarch64-linux-musl-cross.tgz | tar zx
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
                [ -d "$CUSTOM_TC" ] || curl -s https://s3.amazonaws.com/tract-ci-builds/toolchains/armv7l-linux-musleabihf-cross.tgz | tar zx
                export TARGET_CFLAGS="-mfpu=neon"
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

        $SUDO apt-get -y install --no-install-recommends qemu-system-arm qemu-user libssl-dev pkg-config $PACKAGES
        rustup target add $RUSTC_TRIPLE
        if [ -z "$SKIP_QEMU_TEST" ]
        then
            qemu-$QEMU_ARCH --version
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS test --profile opt-no-lto -p tract-linalg -- --nocapture
            cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS test --profile opt-no-lto -p tract-core
        fi

        cargo dinghy --platform $PLATFORM $DINGHY_TEST_ARGS check -p tract-ffi
        # keep lto for these two are they're going to devices.
        cargo dinghy --platform $PLATFORM build --release -p tract -p example-tensorflow-mobilenet-v2
        ;;

    wasm32-wasi)
        rustup target add $PLATFORM
        cargo check --target $PLATFORM --features getrandom-js -p tract-onnx -p tract-tensorflow
        WASMTIME=$HOME/.wasmtime/bin/wasmtime
        [ -e $WASMTIME ] || curl https://wasmtime.dev/install.sh -sSf | bash
        $WASMTIME --version
        RUSTFLAGS='-C target-feature=+simd128' CARGO_TARGET_WASM32_WASI_RUNNER=$WASMTIME \
            cargo test --target=wasm32-wasi -p tract-linalg -p tract-core -p test-unit-core
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

if [ -e "target/$RUSTC_TRIPLE/release/tract" ]
then
    export RUSTC_TRIPLE
    TASK_NAME=`.travis/make_bundle.sh`
    echo bench task: $TASK_NAME 
    if [ -n "$AWS_ACCESS_KEY_ID" ]
    then
        aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
    fi
fi
