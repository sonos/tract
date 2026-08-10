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

// ---------------------------------------------------------------------------
// Chunked gated delta rule (prefill path). Mathematically the standard
// chunk-parallel decomposition of the recurrence (HF transformers
// torch_chunk_gated_delta_rule semantics): the sequence is cut into chunks
// of GDN_CHUNK steps; everything state-independent is computed for all
// chunks in parallel (gdn_chunk_prepare_f16), then a scan kernel walks the
// chunks sequentially with dense matrix work instead of one step at a time
// (gdn_chunk_scan). All compute f32, matching the CPU op which keeps an f32
// state across the whole call and rounds once on output.
//
// Per chunk, with qn/kn the L2-normalized rows, G = exp(cumsum(log_decay))
// inclusive within the chunk, k_beta = kn * beta, v_beta = v * beta:
//   A[i][j] = -(k_beta_i . kn_j) * G_i / G_j          (strictly lower)
//   T = (I - A)^-1 via the forward substitution loop  (unit lower)
//   value' = T @ v_beta
//   k_cumdecay = T @ (k_beta * G)
//   attn_local[i][j] = (qn_i . kn_j) * scale * G_i / G_j  (lower incl diag)
//   q_g = qn * scale * G ; w_t = kn * G_last / G ; eg_last = G_last
// and the sequential scan per head:
//   v_new = value' - k_cumdecay @ S
//   out   = q_g @ S + attn_local @ v_new
//   S     = eg_last * S + w_t^T @ v_new
// ---------------------------------------------------------------------------

constant int GDN_CHUNK = 64;

// One threadgroup per (batch, value head, chunk); everything here is
// state-independent so the whole grid runs in parallel.
[[kernel]] void gdn_chunk_prepare_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const float *log_decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device float *value_p [[buffer(5)]],
    device float *k_cumdecay [[buffer(6)]],
    device float *attn_local [[buffer(7)]],
    device float *q_g [[buffer(8)]],
    device float *w_t [[buffer(9)]],
    device float *eg_last [[buffer(10)]],
    constant int &heads [[buffer(11)]],
    constant int &width [[buffer(12)]],
    constant int &s_len [[buffer(13)]],
    constant int &batch [[buffer(14)]],
    constant int &k_heads [[buffer(15)]],
    constant int &nch [[buffer(16)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
  const int C = GDN_CHUNK;
  threadgroup float A[GDN_CHUNK * GDN_CHUNK];
  threadgroup float row_copy[GDN_CHUNK];
  threadgroup float q_inv[GDN_CHUNK];
  threadgroup float k_inv[GDN_CHUNK];
  threadgroup float g_cum[GDN_CHUNK];
  threadgroup float kmul[GDN_CHUNK]; // k_inv * beta * exp(g_cum)
  threadgroup float bmul[GDN_CHUNK]; // beta

  const int chunk = tgid % nch;
  const int head = (int)(tgid / nch) % heads;
  const int b = (int)tgid / (nch * heads);
  if (b >= batch) return;
  const int groups = heads / k_heads;
  const int qk_head = head / groups;
  const int s0 = chunk * C;
  const int cn = min(C, s_len - s0);
  const float out_scale = rsqrt(float(width));

  const int lane = tid % 32;
  const int sgix = tid / 32;
  const int n_sg = max(int(tptg) / 32, 1);

  // Per-row L2 norms of q/k: one simdgroup per row, lanes split the width
  // (coalesced; a thread-per-row layout makes every lane read a different
  // 2*width-byte-distant row and serializes the loads).
  for (int i = sgix; i < cn; i += n_sg) {
    const int qk_base = ((b * s_len + s0 + i) * k_heads + qk_head) * width;
    float q2 = 0.0f, k2 = 0.0f;
    for (int c = lane; c < width; c += 32) {
      const float qv = float(query[qk_base + c]);
      const float kv = float(key[qk_base + c]);
      q2 += qv * qv;
      k2 += kv * kv;
    }
    q2 = simd_sum(q2);
    k2 = simd_sum(k2);
    if (lane == 0) {
      q_inv[i] = rsqrt(q2 + 1.0e-6f);
      k_inv[i] = rsqrt(k2 + 1.0e-6f);
    }
  }
  // Inclusive within-chunk decay cumsum (C serial adds, one thread).
  if (tid == 0) {
    float acc = 0.0f;
    for (int i = 0; i < cn; ++i) {
      acc += log_decay[(b * s_len + s0 + i) * heads + head];
      g_cum[i] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < cn; i += tptg) {
    const float beta_i = float(beta[(b * s_len + s0 + i) * heads + head]);
    bmul[i] = beta_i;
    kmul[i] = k_inv[i] * beta_i * exp(g_cum[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // A = -(k_beta @ kn^T) * decay, strictly lower triangular. One simdgroup
  // per (i, j) pair: lanes split the dot (coalesced row reads + simd_sum).
  for (int e = sgix; e < cn * cn; e += n_sg) {
    const int i = e / cn;
    const int j = e % cn;
    float a = 0.0f;
    if (j < i) {
      const int bi = ((b * s_len + s0 + i) * k_heads + qk_head) * width;
      const int bj = ((b * s_len + s0 + j) * k_heads + qk_head) * width;
      float dot = 0.0f;
      for (int c = lane; c < width; c += 32) {
        dot += float(key[bi + c]) * float(key[bj + c]);
      }
      dot = simd_sum(dot);
      a = -dot * k_inv[i] * k_inv[j] * bmul[i] * exp(g_cum[i] - g_cum[j]);
    }
    if (lane == 0) {
      A[i * C + j] = a;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Forward substitution: row i gains sum_p old_row[p] * A[p][:] over the
  // already-transformed rows p < i (A stays strictly lower throughout).
  for (int i = 1; i < cn; ++i) {
    for (int j = tid; j < i; j += tptg) {
      row_copy[j] = A[i * C + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tid; j < i; j += tptg) {
      float acc = row_copy[j];
      for (int p = j + 1; p < i; ++p) {
        acc += row_copy[p] * A[p * C + j];
      }
      A[i * C + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = tid; i < cn; i += tptg) {
    A[i * C + i] = 1.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float g_last = g_cum[cn - 1];
  if (tid == 0) {
    eg_last[(b * heads + head) * nch + chunk] = exp(g_last);
  }

  // value' = T @ v_beta and k_cumdecay = T @ (k_beta * exp(g_cum)).
  const int row_base = ((b * heads + head) * nch + chunk) * C;
  for (int e = tid; e < cn * width; e += tptg) {
    const int i = e / width;
    const int c = e % width;
    float vp = 0.0f, kc = 0.0f;
    for (int j = 0; j <= i; ++j) {
      const float t = A[i * C + j];
      const int vbase = ((b * s_len + s0 + j) * heads + head) * width;
      const int kbase = ((b * s_len + s0 + j) * k_heads + qk_head) * width;
      vp += t * float(value[vbase + c]) * bmul[j];
      kc += t * float(key[kbase + c]) * kmul[j];
    }
    const int out_ix = (row_base + i) * width + c;
    value_p[out_ix] = vp;
    k_cumdecay[out_ix] = kc;
  }

  // attn_local = (qn @ kn^T) * decay, lower triangular INCLUDING diagonal.
  // Same simdgroup-cooperative dot as the A matrix.
  for (int e = sgix; e < cn * cn; e += n_sg) {
    const int i = e / cn;
    const int j = e % cn;
    float a = 0.0f;
    if (j <= i) {
      const int bi = ((b * s_len + s0 + i) * k_heads + qk_head) * width;
      const int bj = ((b * s_len + s0 + j) * k_heads + qk_head) * width;
      float dot = 0.0f;
      for (int c = lane; c < width; c += 32) {
        dot += float(query[bi + c]) * float(key[bj + c]);
      }
      dot = simd_sum(dot);
      a = dot * q_inv[i] * out_scale * k_inv[j] * exp(g_cum[i] - g_cum[j]);
    }
    if (lane == 0) {
      attn_local[(row_base + i) * C + j] = a;
    }
  }

  // q_g = qn * scale * exp(g_cum) ; w_t = kn * exp(g_last - g_cum).
  for (int e = tid; e < cn * width; e += tptg) {
    const int i = e / width;
    const int c = e % width;
    const int qk_base = ((b * s_len + s0 + i) * k_heads + qk_head) * width;
    const int out_ix = (row_base + i) * width + c;
    q_g[out_ix] = float(query[qk_base + c]) * q_inv[i] * out_scale * exp(g_cum[i]);
    w_t[out_ix] = float(key[qk_base + c]) * k_inv[i] * exp(g_last - g_cum[i]);
  }
}

// Columns of the state each scan threadgroup owns. The whole recurrence is
// column-separable (rows mix only through the precomputed per-chunk row
// vectors), so the scan parallelizes as (batch, head, column-block) with
// the state block held in threadgroup memory in f32 for ALL chunks: no
// device state traffic inside the loop at all, and heads*width/32
// threadgroups instead of heads.
constant int GDN_COL_BLOCK = 16;

// Sequential scan over chunks with dense per-chunk matrix work on one
// column block of one head's state.
template <typename ST>
kernel void gdn_chunk_scan(
    device const float *value_p [[buffer(0)]],
    device const float *k_cumdecay [[buffer(1)]],
    device const float *attn_local [[buffer(2)]],
    device const float *q_g [[buffer(3)]],
    device const float *w_t [[buffer(4)]],
    device const float *eg_last [[buffer(5)]],
    device const ST *initial_state [[buffer(6)]],
    device half *output [[buffer(7)]],
    device ST *final_state [[buffer(8)]],
    constant int &heads [[buffer(9)]],
    constant int &width [[buffer(10)]],
    constant int &s_len [[buffer(11)]],
    constant int &batch [[buffer(12)]],
    constant int &nch [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
  const int C = GDN_CHUNK;
  const int CB = GDN_COL_BLOCK;
  // S block [width x CB] f32 (16KB at width 128) + v_new block [C x CB]
  // f32 (8KB): resident for the whole sequence.
  threadgroup float s_blk[128 * GDN_COL_BLOCK];
  threadgroup float vn_blk[GDN_CHUNK * GDN_COL_BLOCK];

  const int col_blocks = (width + CB - 1) / CB;
  const int cblk = tgid % col_blocks;
  const int head = (int)(tgid / col_blocks) % heads;
  const int b = (int)tgid / (col_blocks * heads);
  if (b >= batch) return;
  const int c0 = cblk * CB;
  const int cb = min(CB, width - c0);
  const int mat_base = (b * heads + head) * width * width;

  for (int e = tid; e < width * cb; e += tptg) {
    const int r = e / cb;
    const int c = e % cb;
    s_blk[r * CB + c] = float(initial_state[mat_base + r * width + c0 + c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int ch = 0; ch < nch; ++ch) {
    const int s0 = ch * C;
    const int cn = min(C, s_len - s0);
    const int row_base = ((b * heads + head) * nch + ch) * C;
    const float egl = eg_last[(b * heads + head) * nch + ch];

    // v_new = value' - k_cumdecay @ S (this head's column block).
    for (int e = tid; e < cn * cb; e += tptg) {
      const int i = e / cb;
      const int c = e % cb;
      device const float *kc = k_cumdecay + (row_base + i) * width;
      float acc = 0.0f;
#pragma clang loop unroll_count(16)
      for (int r = 0; r < width; ++r) {
        acc += kc[r] * s_blk[r * CB + c];
      }
      vn_blk[i * CB + c] = value_p[(row_base + i) * width + c0 + c] - acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // out = q_g @ S + attn_local @ v_new
    for (int e = tid; e < cn * cb; e += tptg) {
      const int i = e / cb;
      const int c = e % cb;
      device const float *qg = q_g + (row_base + i) * width;
      float acc = 0.0f;
#pragma clang loop unroll_count(16)
      for (int r = 0; r < width; ++r) {
        acc += qg[r] * s_blk[r * CB + c];
      }
      device const float *al = attn_local + (row_base + i) * C;
#pragma clang loop unroll_count(8)
      for (int j = 0; j <= i; ++j) {
        acc += al[j] * vn_blk[j * CB + c];
      }
      output[((b * s_len + s0 + i) * heads + head) * width + c0 + c] = half(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = eg_last * S + w_t^T @ v_new
    for (int e = tid; e < width * cb; e += tptg) {
      const int r = e / cb;
      const int c = e % cb;
      device const float *wt = w_t + row_base * width;
      float acc = s_blk[r * CB + c] * egl;
#pragma clang loop unroll_count(16)
      for (int i = 0; i < cn; ++i) {
        acc += wt[i * width + r] * vn_blk[i * CB + c];
      }
      s_blk[r * CB + c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = tid; e < width * cb; e += tptg) {
    const int r = e / cb;
    const int c = e % cb;
    final_state[mat_base + r * width + c0 + c] = ST(s_blk[r * CB + c]);
  }
}

template [[host_name("gdn_chunk_scan_f32_state")]] [[kernel]] void
gdn_chunk_scan<float>(
    device const float *, device const float *, device const float *,
    device const float *, device const float *, device const float *,
    device const float *, device half *, device float *,
    constant int &, constant int &, constant int &,
    constant int &, constant int &, uint, uint, uint);

template [[host_name("gdn_chunk_scan_f16_state")]] [[kernel]] void
gdn_chunk_scan<half>(
    device const float *, device const float *, device const float *,
    device const float *, device const float *, device const float *,
    device const half *, device half *, device half *,
    constant int &, constant int &, constant int &,
    constant int &, constant int &, uint, uint, uint);
