#!/bin/sh

set -ex

ARCH=$1

case $ARCH in
    aarch64)
        MUSL_TRIPLE=aarch64-linux-musl
        RUST_TRIPLE=aarch64-unknown-linux-musl
    ;;
    armv7)
        MUSL_TRIPLE=armv7l-linux-musleabihf
        RUST_TRIPLE=armv7-unknown-linux-musleabi
    ;;
    *)
        exit "Can't build with musl for $ARCH"
    ;;
esac

rustup update
rustup target add $RUST_TRIPLE

curl -s https://musl.cc/${MUSL_TRIPLE}-cross.tgz | tar zx

MUSL_BIN=`pwd`/${MUSL_TRIPLE}-cross/bin
export PATH=$MUSL_BIN:$PATH

export TARGET_CC=$MUSL_BIN/${MUSL_TRIPLE}-gcc

RUST_TRIPLE_ENV=$(echo $RUST_TRIPLE | tr 'a-z-' 'A-Z_')
export CARGO_TARGET_${RUST_TRIPLE_ENV}_CC=$TARGET_CC
export CARGO_TARGET_${RUST_TRIPLE_ENV}_LINKER=$TARGET_CC

( cd linalg/cost_model ; cargo build --target $RUST_TRIPLE --release )

dates=`date -u +"%Y%m%dT%H%M%S %s"`
date_iso=`echo $dates | cut -f 1 -d ' '`
TASK_NAME=cost-model-dataset-$date_iso

mkdir $TASK_NAME
mv linalg/cost_model/target/${RUST_TRIPLE}/release/cost_model $TASK_NAME
echo "#!/bin/sh" > $TASK_NAME/entrypoint.sh
echo "./cost_mmodel $TASK_NAME.txt" >> $TASK_NAME/entrypoint.sh
chmod +x $TASK_NAME/entrypoint.sh
tar czf $TASK_NAME.tgz $TASK_NAME

if [ -n "$AWS_ACCESS_KEY_ID" ]
then
    aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
fi
