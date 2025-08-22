#!/bin/sh

# Model generated with the following configuration
# LlamaConfig(
#     hidden_size=8,
#     intermediate_size=16,
#     num_attention_heads=4,
#     num_key_value_heads=2,
#     num_hidden_layers=1,
#     vocab_size=100,
#     max_position_embeddings=256,
# )

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract $CARGO_OPTS --}

rm -rf found
$TRACT_RUN --nnef-tract-core --nnef-tract-transformers --transform transformers-detect-all model.nnef.tgz run --input-from-bundle io.npz --steps --assert-output-bundle io.npz
