#include "common.cuh"
#include <math_constants.h>

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle_guarded(uint32_t dst,
                                      const half *src, int src_stride,
                                      int tid,
                                      int valid_rows)
{
  constexpr int BYTES_PER_COPY = 16;
  constexpr int elems_per_copy = BYTES_PER_COPY / sizeof(half);    // 8 f16
  constexpr int total_elems    = HEIGHT * WIDTH;
  constexpr int num_iters      = total_elems / (TB_SIZE * elems_per_copy);

  // zero value to use for st.shared zero-fill
  int z = 0;

  #pragma unroll
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * elems_per_copy;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const bool in_row = (row < valid_rows);       // WIDTH (DIM) is always complete
    const uint32_t dst_addr = swizzle<WIDTH * sizeof(half)>(
        dst + (row * WIDTH + col) * sizeof(half));

    if (in_row) {
      const half* src_addr = src + (row * src_stride + col);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(dst_addr), "l"(src_addr));
    } else {
      // write 16B of zeros into shared for this 8 halfs slot
      asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};"
                    :: "r"(dst_addr), "r"(z));
    }
  }
}

// --- tensor core loads / mma -----------------------------------

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}
__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}
__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}
__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}
__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// --- kernel ----------------------------------------------------------------

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool is_causal, bool use_mask>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v5_kernel(
  const half* __restrict__ Q,  // [bs, len_q, DIM]
  const half* __restrict__ K,  // [bs, len_kv, DIM]
  const half* __restrict__ V,  // [bs, len_kv, DIM]
  const half* __restrict__ M,  // [bs, len_q, len_kv] if use_mask
  half* __restrict__ O,        // [bs, len_q, DIM]
  int bs,
  int qh,
  int head_ratio,
  int len_q,
  int len_kv,
  float scale)
{
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.z;
  const int hid = blockIdx.y;
  const int q_block_id = blockIdx.x;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int q_block_base = q_block_id * BLOCK_Q;
  const int q_valid = max(0, min(BLOCK_Q, len_q - q_block_base)); // tail-safe

  const int past = len_kv - len_q;

  const int q_heads   = qh;
  const int kv_heads  = q_heads / head_ratio;
  const int kv_head_id = hid / head_ratio;

  Q += (( (size_t)bid * q_heads + hid) * (size_t)len_q + q_block_base) * DIM;
  K += (((size_t)bid * kv_heads + kv_head_id) * (size_t)len_kv) * DIM;
  V += (((size_t)bid * kv_heads + kv_head_id) * (size_t)len_kv) * DIM;
  O += (( (size_t)bid * q_heads + hid) * (size_t)len_q + q_block_base) * DIM;

  const half* __restrict__ MaskBase = nullptr;
  if constexpr (use_mask) {
    MaskBase = M ? (M + (size_t)q_block_base * len_kv) : nullptr;
  }

  // Shared memory layout:
  // [Q_smem (BLOCK_Q x DIM)] overlapped with [K_smem 2x (BLOCK_KV x DIM)] + [V_smem (BLOCK_KV x DIM)]
  extern __shared__ half smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(half);

  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = (lane_id / 16) * 8;
    Q_smem_thread = swizzle<DIM * sizeof(half)>(Q_smem + (row_off * DIM + col_off) * sizeof(half));
  }
  {
    const int row_off = lane_id % 8;
    const int col_off = (lane_id / 8) * 8;
    K_smem_thread = swizzle<DIM * sizeof(half)>(K_smem + (row_off * DIM + col_off) * sizeof(half));
  }
  {
    const int row_off = lane_id % 16;
    const int col_off = (lane_id / 16) * 8;
    V_smem_thread = swizzle<DIM * sizeof(half)>(V_smem + (row_off * DIM + col_off) * sizeof(half));
  }

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};
  #pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi) {
    rowmax[qi][0] = -FLT_MAX;
    rowmax[qi][1] = -FLT_MAX;
  }

  // Load Q (tail-safe on rows)
  global_to_shared_swizzle_guarded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Q: shared -> registers
  #pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
    #pragma unroll
    for (int dk = 0; dk < DIM / MMA_K; ++dk) {
      uint32_t addr = Q_smem_thread;
      addr += qi * MMA_M * DIM * sizeof(half);      // row
      addr ^= dk * MMA_K * sizeof(half);            // col
      ldmatrix_x4(Q_rmem[qi][dk], addr);
    }
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id, int kv_valid_rows) {
    if (kv_id < num_kv_iter) {
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(half));
      global_to_shared_swizzle_guarded<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid, kv_valid_rows);
      K += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_valid_rows) {
    const uint32_t dst = V_smem;
    global_to_shared_swizzle_guarded<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid, kv_valid_rows);
    V += BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");
  };

  // Prefetch first K tile (tail-safe)
  int kv0_valid = min(BLOCK_KV, len_kv);
  load_K(0, kv0_valid);

  for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {
    const int kv_tile_base = kv_id * BLOCK_KV;
    const int kv_valid = max(0, min(BLOCK_KV, len_kv - kv_tile_base));

    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // Prefetch V (single buffer). Ensure previous V use finished.
    __syncthreads();
    load_V(kv_valid);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    #pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
      #pragma unroll
      for (int dk = 0; dk < DIM / MMA_K; dk += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(half));
        addr += kvt * MMA_N * DIM * sizeof(half);     // row
        addr ^= dk  * MMA_K * sizeof(half);           // col
        ldmatrix_x4(K_rmem[kvt][dk], addr);
      }

    // S = Q @ K^T
    #pragma unroll
    for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
      #pragma unroll
      for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
        #pragma unroll
        for (int dk = 0; dk < DIM / MMA_K; ++dk)
          mma_m16n8k16(Q_rmem[qi][dk], K_rmem[kvt][dk], S_rmem[qi][kvt]);

    // Prefetch next K (tail-safe)
    load_K(kv_id + 1, max(0, min(BLOCK_KV, len_kv - (kv_id + 1) * BLOCK_KV)));

    // Prepare masking / scaling, track rowmax and rowsumexp
    const int lane_row0 = (lane_id / 4);     // 0..7
    const int lane_row1 = lane_row0 + 8;     // 8..15
    const int j_pair0   = 2 * (lane_id % 4); // {0,2,4,6}

    #pragma unroll
    for (int qi = 0; qi < WARP_Q / MMA_M; ++qi) {
      // scale logits
      #pragma unroll
      for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt) {
        float* regs = S_rmem[qi][kvt];
        regs[0] *= scale; regs[1] *= scale;
        regs[2] *= scale; regs[3] *= scale;

        const int base_i_block  = warp_id * (BLOCK_Q / NUM_WARPS) + qi * MMA_M;   // 16 rows for this MMA
        const int col_base_tile = kv_tile_base + kvt * MMA_N;                     // 8 cols for this MMA
        const int i0 = base_i_block + lane_row0;
        const int i1 = base_i_block + lane_row1;
        const int j0 = col_base_tile + j_pair0 + 0;
        const int j1 = col_base_tile + j_pair0 + 1;

        const int i0_global = q_block_base + i0;
        const int i1_global = q_block_base + i1;
        const int j_limit0 = past + i0_global;
        const int j_limit1 = past + i1_global;

        // Tail-safe row checks (Q)
        if (i0_global >= len_q) { regs[0] = -CUDART_INF_F; regs[1] = -CUDART_INF_F; }
        if (i1_global >= len_q) { regs[2] = -CUDART_INF_F; regs[3] = -CUDART_INF_F; }

        // Tail-safe col checks (KV)
        if (j0 >= len_kv) { regs[0] = -CUDART_INF_F; regs[2] = -CUDART_INF_F; }
        if (j1 >= len_kv) { regs[1] = -CUDART_INF_F; regs[3] = -CUDART_INF_F; }

        // causal / explicit mask
        if constexpr (is_causal) {
          if (j0 > j_limit0) regs[0] = -CUDART_INF_F;
          if (j1 > j_limit0) regs[1] = -CUDART_INF_F;
          if (j0 > j_limit1) regs[2] = -CUDART_INF_F;
          if (j1 > j_limit1) regs[3] = -CUDART_INF_F;
        } else if constexpr (use_mask) {
          if (MaskBase) {
            if (i0_global < len_q && j0 < len_kv) regs[0] += (float) MaskBase[(size_t)i0 * len_kv + j0];
            if (i0_global < len_q && j1 < len_kv) regs[1] += (float) MaskBase[(size_t)i0 * len_kv + j1];
            if (i1_global < len_q && j0 < len_kv) regs[2] += (float) MaskBase[(size_t)i1 * len_kv + j0];
            if (i1_global < len_q && j1 < len_kv) regs[3] += (float) MaskBase[(size_t)i1 * len_kv + j1];
          }
        }
      }

      // rowmax across KV tiles (2 lanes per output row group)
      float this_rowmax[2];
      #pragma unroll
      for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt) {
        float* r = S_rmem[qi][kvt];
        if (kvt == 0) {
          this_rowmax[0] = fmaxf(r[0], r[1]);
          this_rowmax[1] = fmaxf(r[2], r[3]);
        } else {
          this_rowmax[0] = fmaxf(this_rowmax[0], fmaxf(r[0], r[1]));
          this_rowmax[1] = fmaxf(this_rowmax[1], fmaxf(r[2], r[3]));
        }
      }
      this_rowmax[0] = fmaxf(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
      this_rowmax[0] = fmaxf(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
      this_rowmax[1] = fmaxf(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
      this_rowmax[1] = fmaxf(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

      this_rowmax[0] = fmaxf(this_rowmax[0], rowmax[qi][0]);
      this_rowmax[1] = fmaxf(this_rowmax[1], rowmax[qi][1]);

      const float rescale0 = __expf(rowmax[qi][0] - this_rowmax[0]);
      const float rescale1 = __expf(rowmax[qi][1] - this_rowmax[1]);

      #pragma unroll
      for (int d = 0; d < DIM / MMA_N; ++d) {
        O_rmem[qi][d][0] *= rescale0;
        O_rmem[qi][d][1] *= rescale0;
        O_rmem[qi][d][2] *= rescale1;
        O_rmem[qi][d][3] *= rescale1;
      }

      rowmax[qi][0] = this_rowmax[0];
      rowmax[qi][1] = this_rowmax[1];

      float this_rowsumexp0 = 0.f, this_rowsumexp1 = 0.f;

      #pragma unroll
      for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt) {
        float* r = S_rmem[qi][kvt];
        r[0] = __expf(r[0] - rowmax[qi][0]);
        r[1] = __expf(r[1] - rowmax[qi][0]);
        r[2] = __expf(r[2] - rowmax[qi][1]);
        r[3] = __expf(r[3] - rowmax[qi][1]);

        this_rowsumexp0 += (r[0] + r[1]);
        this_rowsumexp1 += (r[2] + r[3]);

        // pack P (m16n8 -> m16k16)
        half2 *Ppack = reinterpret_cast<half2 *>(P_rmem[qi][kvt / 2]);
        Ppack[(kvt % 2) * 2]     = __float22half2_rn({r[0], r[1]});
        Ppack[(kvt % 2) * 2 + 1] = __float22half2_rn({r[2], r[3]});
      }

      this_rowsumexp0 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp0, 1);
      this_rowsumexp0 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp0, 2);
      this_rowsumexp1 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp1, 1);
      this_rowsumexp1 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp1, 2);

      rowsumexp[qi][0] = rowsumexp[qi][0] * rescale0 + this_rowsumexp0;
      rowsumexp[qi][1] = rowsumexp[qi][1] * rescale1 + this_rowsumexp1;
    }

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    #pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_K; ++kvt)
      #pragma unroll
      for (int d = 0; d < DIM / MMA_N; d += 2) {
        uint32_t addr = V_smem_thread;
        addr += kvt * MMA_K * DIM * sizeof(half);   // row
        addr ^= d * MMA_N * sizeof(half);           // col
        ldmatrix_x4_trans(V_rmem[kvt][d], addr);
      }

    // O += P @ V
    #pragma unroll
    for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
      #pragma unroll
      for (int d = 0; d < DIM / MMA_N; ++d)
        #pragma unroll
        for (int kvt = 0; kvt < BLOCK_KV / MMA_K; ++kvt)
          mma_m16n8k16(P_rmem[qi][kvt], V_rmem[kvt][d], O_rmem[qi][d]);
  }

  // write O back (tail-safe on rows)
  #pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
    #pragma unroll
    for (int d = 0; d < DIM / MMA_N; ++d) {
      const int row0 = warp_id * WARP_Q + qi * MMA_M + (lane_id / 4);
      const int col  = d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[qi][d];
      // divide by denom
      regs[0] /= rowsumexp[qi][0];
      regs[1] /= rowsumexp[qi][0];
      regs[2] /= rowsumexp[qi][1];
      regs[3] /= rowsumexp[qi][1];

      const int g_row0 = q_block_base + row0;
      const int g_row1 = g_row0 + 8;

      if (g_row0 < len_q) {
        reinterpret_cast<half2 *>(O + (row0 + 0) * DIM + col)[0] =
            __float22half2_rn({regs[0], regs[1]});
      }
      if (g_row1 < len_q) {
        reinterpret_cast<half2 *>(O + (row0 + 8) * DIM + col)[0] =
            __float22half2_rn({regs[2], regs[3]});
      }
    }
}

#define INSTANTIATE_MINIMAL_FLASH_FOR_MASK_STRATEGY(BLOCK_Q, BLOCK_KV, D, is_causal, use_mask) \
  extern "C" __global__ void attention_v5_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, const half* __restrict__ M, half* __restrict__ O, \
    int bs, int qh, int head_ratio, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask>(Q, K, V, M, O, bs, qh, head_ratio, len_q, len_kv, scale); \
  }

#define INSTANTIATE_MINIMAL_FLASH_FOR_D(block_q, block_kv, D) \
  INSTANTIATE_MINIMAL_FLASH_FOR_MASK_STRATEGY(block_q, block_kv, D, false, false) \
  INSTANTIATE_MINIMAL_FLASH_FOR_MASK_STRATEGY(block_q, block_kv, D, false, true) \
  INSTANTIATE_MINIMAL_FLASH_FOR_MASK_STRATEGY(block_q, block_kv, D, true, false) \

#define INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, block_kv) \
  INSTANTIATE_MINIMAL_FLASH_FOR_D(block_q, block_kv, 64) \
  INSTANTIATE_MINIMAL_FLASH_FOR_D(block_q, block_kv, 128) \

#define INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_Q(block_q) \
  INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 32) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 48) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 64) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 80) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 96) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 112) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 128) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_KV(block_q, 16) \

#define INSTANTIATE_MINIMAL_FLASH() \
  INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_Q(64) \
  //INSTANTIATE_MINIMAL_FLASH_FOR_BLOCK_Q(128)

INSTANTIATE_MINIMAL_FLASH()