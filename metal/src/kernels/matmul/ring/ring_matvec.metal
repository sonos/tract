#include <metal_stdlib>
using namespace metal;

#define RING_MATVEC_MAX_M 8
#define RING_MATVEC_COLS 32
#define RING_MATVEC_SLICES 8

// out[c, r, j] = sum_kk a[c, r, kk] * w(c, kk, j) for r < m <= 8, where w is
// channel c's window held as a ring: element (kk, j) sits at
// kk * k_stride + j * n_stride from table[2c + 1], turned by table[2c] along k
// (rotate_k) or along n. A
// threadgroup covers 32 columns of one channel, its simdgroups (up to 8) each
// walking a slice of k, so a simdgroup reads a row of a [k, n] window at
// consecutive addresses; the slices are summed in threadgroup memory.
template <typename T>
void ring_matvec(device const T *a, device const T *w, device T *out,
                 device const int *table, constant int &m, constant int &k,
                 constant int &n, constant int &rotate_k,
                 constant int &k_stride, constant int &n_stride, uint2 tg,
                 uint2 tid, uint2 tg_size, threadgroup float *partial) {
  const int c = tg.y;
  const int lane = tid.x;
  const int slice = tid.y;
  const int slices = tg_size.y;
  const int j = tg.x * RING_MATVEC_COLS + lane;
  const int turn = table[2 * c];
  const device T *window = w + table[2 * c + 1];
  const device T *act = a + c * m * k;
  float acc[RING_MATVEC_MAX_M] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (j < n) {
    int col = j;
    if (!rotate_k) {
      col += turn;
      if (col >= n) {
        col -= n;
      }
    }
    const device T *column = window + col * n_stride;
    int row = slice + (rotate_k ? turn : 0);
    if (row >= k) {
      row -= k;
    }
    // Four rows in flight per step: the loads are independent of each other.
    int kk = slice;
    for (; kk + 3 * slices < k; kk += 4 * slices) {
      int r1 = row + slices;
      r1 -= r1 >= k ? k : 0;
      int r2 = r1 + slices;
      r2 -= r2 >= k ? k : 0;
      int r3 = r2 + slices;
      r3 -= r3 >= k ? k : 0;
      const float w0 = float(column[row * k_stride]);
      const float w1 = float(column[r1 * k_stride]);
      const float w2 = float(column[r2 * k_stride]);
      const float w3 = float(column[r3 * k_stride]);
      for (int r = 0; r < m; r++) {
        const device T *x = act + r * k + kk;
        acc[r] += float(x[0]) * w0 + float(x[slices]) * w1 +
                  float(x[2 * slices]) * w2 + float(x[3 * slices]) * w3;
      }
      row = r3 + slices;
      row -= row >= k ? k : 0;
    }
    for (; kk < k; kk += slices) {
      const float wv = float(column[row * k_stride]);
      for (int r = 0; r < m; r++) {
        acc[r] += float(act[r * k + kk]) * wv;
      }
      row += slices;
      row -= row >= k ? k : 0;
    }
  }
  if (slices == 1) {
    if (j < n) {
      device T *dst = out + c * m * n + j;
      for (int r = 0; r < m; r++) {
        dst[r * n] = T(acc[r]);
      }
    }
    return;
  }
  for (int r = 0; r < m; r++) {
    partial[(slice * RING_MATVEC_MAX_M + r) * RING_MATVEC_COLS + lane] = acc[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slice == 0 && j < n) {
    device T *dst = out + c * m * n + j;
    for (int r = 0; r < m; r++) {
      float sum = 0;
      for (int s = 0; s < slices; s++) {
        sum += partial[(s * RING_MATVEC_MAX_M + r) * RING_MATVEC_COLS + lane];
      }
      dst[r * n] = T(sum);
    }
  }
}

#define INSTANTIATE_RING_MATVEC(name, T)                                       \
  kernel void ring_matvec_##name(                                              \
      device const T *a [[buffer(0)]], device const T *w [[buffer(1)]],        \
      device T *out [[buffer(2)]], device const int *table [[buffer(3)]],      \
      constant int &m [[buffer(4)]], constant int &k [[buffer(5)]],            \
      constant int &n [[buffer(6)]], constant int &rotate_k [[buffer(7)]],     \
      constant int &k_stride [[buffer(8)]],                                    \
      constant int &n_stride [[buffer(9)]],                                    \
      uint2 tg [[threadgroup_position_in_grid]],                               \
      uint2 tid [[thread_position_in_threadgroup]],                            \
      uint2 tg_size [[threads_per_threadgroup]]) {                             \
    threadgroup float partial[RING_MATVEC_SLICES * RING_MATVEC_MAX_M *         \
                              RING_MATVEC_COLS];                               \
    ring_matvec<T>(a, w, out, table, m, k, n, rotate_k, k_stride, n_stride,   \
                   tg, tid, tg_size, partial);                                 \
  }

INSTANTIATE_RING_MATVEC(f32, float)
INSTANTIATE_RING_MATVEC(f16, half)
