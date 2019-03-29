
set -ex

cd /work/tensorflow
bazel build \
    --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" \
     --copt=-fvisibility=hidden \
    //native_client:libdeepspeech.so \
    //native_client:generate_trie

cd  /work/DeepSpeech/native_client
make install
ldconfig
