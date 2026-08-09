#include <metal_stdlib>
using namespace metal;

// Gated delta rule over the whole sequence. Layout matches the CPU op:
// query/key [b, S, hk, w], value/output [b, S, hv, w] with hv = G * hk
// (GQA: value head h reads query/key head h / G; hk == hv is the
// ungrouped case), log_decay/beta [b, S, hv], state [b, hv, w, w].
// One thread per (batch, value head, column); each thread owns its
// state column exclusively, so the sequential S loop is race-free
// (later steps read this thread's own prior writes in final_state).
kernel void gdn_recurrent_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const float *log_decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device const float *initial_state [[buffer(5)]],
    device half *output [[buffer(6)]],
    device float *final_state [[buffer(7)]],
    constant int &heads [[buffer(8)]],
    constant int &width [[buffer(9)]],
    constant int &s_len [[buffer(10)]],
    constant int &batch [[buffer(11)]],
    constant int &k_heads [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  const int column = gid % width;
  const int head = (gid / width) % heads;
  const int b = gid / (width * heads);
  if (b >= batch) return;
  const int groups = heads / k_heads;
  const int qk_head = head / groups;
  const int matrix_base = (b * heads + head) * width * width;
  const float out_scale = rsqrt(float(width));
  for (int s = 0; s < s_len; ++s) {
    const int vector_base = ((b * s_len + s) * heads + head) * width;
    const int qk_base = ((b * s_len + s) * k_heads + qk_head) * width;
    const int gate_ix = (b * s_len + s) * heads + head;
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int row = 0; row < width; ++row) {
      const float q = float(query[qk_base + row]);
      const float k = float(key[qk_base + row]);
      q_norm += q * q;
      k_norm += k * k;
    }
    const float q_inv = rsqrt(q_norm + 1.0e-6f);
    const float k_inv = rsqrt(k_norm + 1.0e-6f);
    const float decay = exp(log_decay[gate_ix]);
    device const float *state_in = (s == 0) ? initial_state : final_state;
    float predicted = 0.0f;
    for (int row = 0; row < width; ++row) {
      predicted += float(key[qk_base + row]) * k_inv
          * state_in[matrix_base + row * width + column] * decay;
    }
    const float residual =
        (float(value[vector_base + column]) - predicted)
        * float(beta[gate_ix]);
    float result = 0.0f;
    for (int row = 0; row < width; ++row) {
      const int offset = matrix_base + row * width + column;
      const float next = state_in[offset] * decay
          + float(key[qk_base + row]) * k_inv * residual;
      final_state[offset] = next;
      result += float(query[qk_base + row]) * q_inv * next;
    }
    output[vector_base + column] = half(result * out_scale);
  }
}

// Variant with an f16 recurrent state (graph exported with -idt f16):
// identical math, f32 accumulation, state roundtrips through half like
// the CPU op does for an f16-state graph.
kernel void gdn_recurrent_f16_state_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const float *log_decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device const half *initial_state [[buffer(5)]],
    device half *output [[buffer(6)]],
    device half *final_state [[buffer(7)]],
    constant int &heads [[buffer(8)]],
    constant int &width [[buffer(9)]],
    constant int &s_len [[buffer(10)]],
    constant int &batch [[buffer(11)]],
    constant int &k_heads [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  const int column = gid % width;
  const int head = (gid / width) % heads;
  const int b = gid / (width * heads);
  if (b >= batch) return;
  const int groups = heads / k_heads;
  const int qk_head = head / groups;
  const int matrix_base = (b * heads + head) * width * width;
  const float out_scale = rsqrt(float(width));
  for (int s = 0; s < s_len; ++s) {
    const int vector_base = ((b * s_len + s) * heads + head) * width;
    const int qk_base = ((b * s_len + s) * k_heads + qk_head) * width;
    const int gate_ix = (b * s_len + s) * heads + head;
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int row = 0; row < width; ++row) {
      const float q = float(query[qk_base + row]);
      const float k = float(key[qk_base + row]);
      q_norm += q * q;
      k_norm += k * k;
    }
    const float q_inv = rsqrt(q_norm + 1.0e-6f);
    const float k_inv = rsqrt(k_norm + 1.0e-6f);
    const float decay = exp(log_decay[gate_ix]);
    device const half *state_in = (s == 0) ? initial_state : final_state;
    float predicted = 0.0f;
    for (int row = 0; row < width; ++row) {
      predicted += float(key[qk_base + row]) * k_inv
          * float(state_in[matrix_base + row * width + column]) * decay;
    }
    const float residual =
        (float(value[vector_base + column]) - predicted)
        * float(beta[gate_ix]);
    float result = 0.0f;
    for (int row = 0; row < width; ++row) {
      const int offset = matrix_base + row * width + column;
      const float next = float(state_in[offset]) * decay
          + float(key[qk_base + row]) * k_inv * residual;
      final_state[offset] = half(next);
      result += float(query[qk_base + row]) * q_inv * next;
    }
    output[vector_base + column] = half(result * out_scale);
  }
}


// Stateful causal depthwise conv1d + SiLU over the whole sequence.
// Layout matches the CPU op: input/output [b, C, S], weight [C, k],
// state [b, C, k]. One thread per (batch, channel): the sequential S
// loop reads the thread's own sliding window.
kernel void causal_conv1d_update_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *initial_state [[buffer(2)]],
    device half *output [[buffer(3)]],
    device half *final_state [[buffer(4)]],
    constant int &channels [[buffer(5)]],
    constant int &kernel_width [[buffer(6)]],
    constant int &s_len [[buffer(7)]],
    constant int &batch [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const int channel = gid % channels;
  const int b = gid / channels;
  if (b >= batch) return;
  const int state_base = (b * channels + channel) * kernel_width;
  const int input_base = (b * channels + channel) * s_len;
  const int weight_base = channel * kernel_width;
  // Sliding window over concat(state, input); kernel_width is small
  // (4 for Qwen3.5), keep the window in registers.
  const int MAX_K = 8;
  float window[MAX_K];
  if (kernel_width > MAX_K) return;
  for (int tap = 0; tap < kernel_width; ++tap) {
    window[tap] = float(initial_state[state_base + tap]);
  }
  for (int t = 0; t < s_len; ++t) {
    // shift left, append the new sample: window becomes full[t+1 .. t+k]
    for (int tap = 0; tap < kernel_width - 1; ++tap) {
      window[tap] = window[tap + 1];
    }
    window[kernel_width - 1] = float(input[input_base + t]);
    float sum = 0.0f;
    for (int tap = 0; tap < kernel_width; ++tap) {
      sum += window[tap] * float(weight[weight_base + tap]);
    }
    output[input_base + t] = half(sum / (1.0f + exp(-sum)));
  }
  for (int tap = 0; tap < kernel_width; ++tap) {
    final_state[state_base + tap] = half(window[tap]);
  }
}

// Threadgroup-parallel variant: one threadgroup per (batch, value head),
// laid out [width columns x R row-chunks]. The row loops of the original
// kernel (three serial passes of `width` iterations per thread, with only
// b*heads*width threads in flight) are split across R chunks and reduced
// through threadgroup memory, which multiplies occupancy by R and divides
// the per-thread dependency chains by R. Each thread still owns its
// (column, chunk-rows) slice of the state exclusively, so the sequential S
// loop only ever reads back its own device writes.
template <typename ST>
kernel void gdn_recurrent_tg(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const float *log_decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device const ST *initial_state [[buffer(5)]],
    device half *output [[buffer(6)]],
    device ST *final_state [[buffer(7)]],
    constant int &heads [[buffer(8)]],
    constant int &width [[buffer(9)]],
    constant int &s_len [[buffer(10)]],
    constant int &batch [[buffer(11)]],
    constant int &k_heads [[buffer(12)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]) {
  const int col = tpitg.x;
  const int r = tpitg.y;
  const int rchunks = tptg.y;
  const int head = tgpig.x % heads;
  const int b = tgpig.x / heads;
  if (b >= batch) return;
  const int rows_per_chunk = width / rchunks;
  const int row0 = r * rows_per_chunk;
  const int groups = heads / k_heads;
  const int qk_head = head / groups;
  const int matrix_base = (b * heads + head) * width * width;
  const float out_scale = rsqrt(float(width));

  // scratch layout: pred[rchunks][width], res[rchunks][width],
  // qpart[rchunks], kpart[rchunks]
  threadgroup float *pred_part = scratch;
  threadgroup float *res_part = scratch + rchunks * width;
  threadgroup float *q_part = res_part + rchunks * width;
  threadgroup float *k_part = q_part + rchunks;

  for (int s = 0; s < s_len; ++s) {
    const int vector_base = ((b * s_len + s) * heads + head) * width;
    const int qk_base = ((b * s_len + s) * k_heads + qk_head) * width;
    const int gate_ix = (b * s_len + s) * heads + head;
    float q2 = 0.0f;
    float k2 = 0.0f;
    for (int row = row0; row < row0 + rows_per_chunk; ++row) {
      const float q = float(query[qk_base + row]);
      const float k = float(key[qk_base + row]);
      q2 += q * q;
      k2 += k * k;
    }
    if (col == 0) {
      q_part[r] = q2;
      k_part[r] = k2;
    }
    device const ST *state_in = (s == 0) ? initial_state : final_state;
    const float decay = exp(log_decay[gate_ix]);
    float pred = 0.0f;
    for (int row = row0; row < row0 + rows_per_chunk; ++row) {
      pred += float(key[qk_base + row])
          * float(state_in[matrix_base + row * width + col]) * decay;
    }
    pred_part[r * width + col] = pred;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    float predicted = 0.0f;
    for (int rr = 0; rr < rchunks; ++rr) {
      q_norm += q_part[rr];
      k_norm += k_part[rr];
      predicted += pred_part[rr * width + col];
    }
    const float q_inv = rsqrt(q_norm + 1.0e-6f);
    const float k_inv = rsqrt(k_norm + 1.0e-6f);
    predicted *= k_inv;
    const float residual =
        (float(value[vector_base + col]) - predicted) * float(beta[gate_ix]);
    float res = 0.0f;
    for (int row = row0; row < row0 + rows_per_chunk; ++row) {
      const int offset = matrix_base + row * width + col;
      const float next = float(state_in[offset]) * decay
          + float(key[qk_base + row]) * k_inv * residual;
      final_state[offset] = ST(next);
      res += float(query[qk_base + row]) * next;
    }
    res_part[r * width + col] = res;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0) {
      float result = 0.0f;
      for (int rr = 0; rr < rchunks; ++rr) {
        result += res_part[rr * width + col];
      }
      output[vector_base + col] = half(result * q_inv * out_scale);
    }
    // pred_part/q_part/k_part are rewritten next step: make sure every
    // thread is done reading them (and r==0 done reading res_part).
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template [[host_name("gdn_recurrent_f16_tg")]] [[kernel]] void
gdn_recurrent_tg<float>(
    device const half *, device const half *, device const half *,
    device const float *, device const half *, device const float *,
    device half *, device float *, constant int &, constant int &,
    constant int &, constant int &, constant int &, threadgroup float *,
    uint2, uint2, uint2);

template [[host_name("gdn_recurrent_f16_state_f16_tg")]] [[kernel]] void
gdn_recurrent_tg<half>(
    device const half *, device const half *, device const half *,
    device const float *, device const half *, device const half *,
    device half *, device half *, constant int &, constant int &,
    constant int &, constant int &, constant int &, threadgroup float *,
    uint2, uint2, uint2);
