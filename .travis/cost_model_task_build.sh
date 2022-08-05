#!/bin/sh

set -ex

ARCH=$1
ID=$2

case $ARCH in
    aarch64)
        MUSL_TRIPLE=aarch64-linux-musl
        RUST_TRIPLE=aarch64-unknown-linux-musl
        PLATFORM=aarch64-unknown-linux-musl
    ;;
    armv7)
        MUSL_TRIPLE=armv7l-linux-musleabihf
        RUST_TRIPLE=armv7-unknown-linux-musleabihf
        PLATFORM=armv7-unknown-linux-musl
    ;;
    *)
        exit "Can't build with musl for $ARCH"
    ;;
esac

rustup update
rustup target add $RUST_TRIPLE

#curl -s https://musl.cc/${MUSL_TRIPLE}-cross.tgz | tar zx
curl -s https://s3.amazonaws.com/tract-ci-builds/toolchains/${MUSL_TRIPLE}-cross.tgz | tar zx

MUSL_BIN=`pwd`/${MUSL_TRIPLE}-cross/bin
export PATH=$MUSL_BIN:$PATH

export TARGET_CC=$MUSL_BIN/${MUSL_TRIPLE}-gcc

RUST_TRIPLE_ENV=$(echo $RUST_TRIPLE | tr 'a-z-' 'A-Z_')
export CARGO_TARGET_${RUST_TRIPLE_ENV}_CC=$TARGET_CC
export CARGO_TARGET_${RUST_TRIPLE_ENV}_LINKER=$TARGET_CC

( cd linalg/cost_model ; cargo build --target $RUST_TRIPLE --release )

TASK_NAME=cost-model-dataset-$ID

mkdir $TASK_NAME
mv linalg/cost_model/target/${RUST_TRIPLE}/release/cost_model $TASK_NAME
echo "export TIMEOUT=$((86400*4))" > $TASK_NAME/vars
echo "#!/bin/sh" > $TASK_NAME/entrypoint.sh
echo "mkdir product" >> $TASK_NAME/entrypoint.sh
echo "./cost_model ds --size 10000 product/$TASK_NAME.txt" >> $TASK_NAME/entrypoint.sh
# echo "./cost_model ds --size 2000 -k 128 -n 16 product/$TASK_NAME-small-k-tiny-n.txt" >> $TASK_NAME/entrypoint.sh
# echo "./cost_model ds --size 5000 -m 1-512 -k 16,64,256 -n 1-20 product/$TASK_NAME-multiple-k-tiny-n.txt" >> $TASK_NAME/entrypoint.sh
# echo "./cost_model ds --size 1000 -m 1-512 -k 256,1024 -n 1-512 product/$TASK_NAME-bigmn" >> $TASK_NAME/entrypoint.sh
chmod +x $TASK_NAME/entrypoint.sh
tar czf $TASK_NAME.tgz $TASK_NAME

if [ -n "$AWS_ACCESS_KEY_ID" ]
then
    aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
fi
