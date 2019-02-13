#!/bin/sh

set -ex

mkdir -p $HOME/cached/bin
PATH=$HOME/cached/bin:$HOME/.cargo/bin:/tmp/cargo-dinghy:$HOME/cached/android-sdk/platform-tools:$PATH

if [ -z "$TRAVIS" -a `uname` = "Linux" ]
then
    apt-get update
    apt-get -y upgrade
    apt-get install -y unzip wget curl python awscli build-essential
fi

which rustup || curl https://sh.rustup.rs -sSf | sh -s -- -y

( mkdir -p /tmp/cargo-dinghy
cd /tmp/cargo-dinghy
if [ `uname` = "Darwin" ]
then
    NAME=macos
else
    NAME=travis
fi
wget -q https://github.com/snipsco/dinghy/releases/download/0.4.5/cargo-dinghy-$NAME.tgz -O cargo-dinghy.tgz
tar vzxf cargo-dinghy.tgz --strip-components 1
)

case "$PLATFORM" in
    "raspbian")
        [ -e $HOME/cached/raspitools ] || git clone https://github.com/raspberrypi/tools $HOME/cached/raspitools
        TOOLCHAIN=$HOME/cached/raspitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf
        export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
        rustup target add $RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'\ntoolchain='$TOOLCHAIN'" > $HOME/.dinghy.toml
        cargo dinghy --platform $PLATFORM build --release -p tract
        cargo dinghy --platform $PLATFORM bench --no-run -p tract-linalg
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

        export ANDROID_SDK_HOME=$HOME/cached/android-sdk
        [ -e $ANDROID_SDK_HOME ] || ./.travis/android-ndk.sh

        rustup target add $RUSTC_TRIPLE
        ls $ANDROID_SDK_HOME
        ls $ANDROID_SDK_HOME/ndk-bundle
        export ANDROID_NDK_HOME=$ANDROID_SDK_HOME/ndk-bundle
        cargo dinghy --platform auto-android-$ANDROID_CPU build -p tract-linalg
    ;;

    "aarch64-apple-ios")
        rustup target add aarch64-apple-ios
        cargo dinghy --platform auto-ios-aarch64 build -p tract-linalg 
    ;;

    "aarch64-unknown-linux-gnu" | "armv6vfp-unknown-linux-gnueabihf" | "armv7-unknown-linux-gnueabihf")
        case "$PLATFORM" in 
            "aarch64-unknown-linux-gnu")
                export ARCH=aarch64
                export QEMU_ARCH=aarch64
                export RUSTC_TRIPLE=$ARCH-unknown-linux-gnu
                export DEBIAN_TRIPLE=$ARCH-linux-gnu
            ;;
            "armv6vfp-unknown-linux-gnueabihf")
                export ARCH=armv6vfp
                export QEMU_ARCH=arm
                export QEMU_OPTS="-cpu cortex-a15"
                export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
                export DEBIAN_TRIPLE=arm-linux-gnueabihf
            ;;
            "armv7-unknown-linux-gnueabihf")
                export ARCH=armv7
                export QEMU_ARCH=arm
                export QEMU_OPTS="-cpu cortex-a15"
                export RUSTC_TRIPLE=armv7-unknown-linux-gnueabihf
                export DEBIAN_TRIPLE=arm-linux-gnueabihf
            ;;
            *)
                echo "unsupported platform $PLATFORM"
                exit 1
            ;;
        esac

        export TARGET_CC=$DEBIAN_TRIPLE-gcc

        echo "[platforms.$PLATFORM]\ndeb_multiarch='$DEBIAN_TRIPLE'\nrustc_triple='$RUSTC_TRIPLE'" > $HOME/.dinghy.toml
        echo "[script_devices.qemu-$ARCH]\nplatform='$PLATFORM'\npath='$HOME/qemu-$ARCH'" >> $HOME/.dinghy.toml

        echo "#!/bin/sh\nexe=\$1\nshift\n/usr/bin/qemu-$QEMU_ARCH $QEMU_OPTS -L /usr/$DEBIAN_TRIPLE/ \$exe --test-threads 1 \"\$@\"" > $HOME/qemu-$ARCH
        chmod +x $HOME/qemu-$ARCH

        sudo apt-get -y install binutils-$DEBIAN_TRIPLE gcc-$DEBIAN_TRIPLE qemu-system-arm qemu-user libssl-dev pkg-config
        rustup target add $RUSTC_TRIPLE
        cargo dinghy --platform $PLATFORM test --release -p tract-linalg -- --nocapture
        cargo dinghy --platform $PLATFORM test --release -p tract-core
        cargo dinghy --platform $PLATFORM build --release -p tract
        cargo dinghy --platform $PLATFORM bench --no-run -p tract-linalg
    ;;
    *)
esac

if [ -n "$AWS_ACCESS_KEY_ID" -a -e "target/$RUSTC_TRIPLE/release/tract" ]
then
    export RUSTC_TRIPLE
    TASK_NAME=`.travis/make_bundle.sh`
    aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
fi
