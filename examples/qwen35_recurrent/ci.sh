#!/bin/bash
set -euo pipefail
set -x

cd "$(dirname "$0")"
rm -rf assets
mkdir -p assets/model

# A deterministic miniature recurrent graph keeps the weekly example gate fast
# while exercising NNEF registration, the public API, runtime selection, state
# feedback, and the platform GPU kernel.
cat > assets/model/graph.nnef <<'NNEF'
version 1.0;

extension tract_registry tract_transformers;
extension tract_registry tract_core;

graph qwen35_recurrent_smoke(input_ids, position_ids, query, key, value, log_decay, beta, initial_state) ->
    (logits, query_out, key_out, value_out, log_decay_out, beta_out, final_state)
{
    input_ids = tract_core_external(shape = [1, 1], datum_type = 'i64');
    position_ids = tract_core_external(shape = [1, 1], datum_type = 'i64');
    query = tract_core_external(shape = [1, 1, 2, 128], datum_type = 'f16');
    key = tract_core_external(shape = [1, 1, 2, 128], datum_type = 'f16');
    value = tract_core_external(shape = [1, 1, 2, 128], datum_type = 'f16');
    log_decay = tract_core_external(shape = [1, 1, 2], datum_type = 'f32');
    beta = tract_core_external(shape = [1, 1, 2], datum_type = 'f16');
    initial_state = tract_core_external(shape = [1, 2, 128, 128], datum_type = 'f32');
    (logits, final_state) = tract_transformers_gdn_recurrent(
        query, key, value, log_decay, beta, initial_state);
    query_out = copy(query);
    key_out = copy(key);
    value_out = copy(value);
    log_decay_out = copy(log_decay);
    beta_out = copy(beta);
}
NNEF

python3 - <<'PY'
import numpy as np

np.savez(
    "assets/inputs.npz",
    input_ids=np.zeros((1, 1), dtype=np.int64),
    position_ids=np.zeros((1, 1), dtype=np.int64),
    query=np.zeros((1, 1, 2, 128), dtype=np.float32),
    key=np.zeros((1, 1, 2, 128), dtype=np.float32),
    value=np.zeros((1, 1, 2, 128), dtype=np.float32),
    log_decay=np.zeros((1, 1, 2), dtype=np.float32),
    beta=np.zeros((1, 1, 2), dtype=np.float32),
    initial_state=np.zeros((1, 2, 128, 128), dtype=np.float32),
)
PY

cargo check -p tract-qwen35-recurrent
cargo run --release -p tract-qwen35-recurrent -- assets/model assets/inputs.npz 1 255 0

case "$(uname -s)" in
    Darwin) cargo test -p tract-metal qwen35_ -- --nocapture ;;
    Linux) cargo test -p tract-cuda qwen35_ -- --nocapture ;;
esac

rm -rf assets
