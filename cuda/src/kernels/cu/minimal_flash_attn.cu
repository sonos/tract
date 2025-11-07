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
void global_to_shared(uint32_t dst, const half *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(half);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(half);
    const half *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const half *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(half);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(half)>(dst + (row * WIDTH + col) * sizeof(half));
    const half *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

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

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool is_causal, bool use_mask>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v5_kernel(
  const half* __restrict__ Q,  // [bs, len_q, DIM]
  const half* __restrict__ K,  // [bs, len_kv, DIM]
  const half* __restrict__ V,  // [bs, len_kv, DIM]
  const half* __restrict__ M,  // [bs, len_kv, DIM]
  half* __restrict__ O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv,
  float scale) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;
  const int q_block_base = q_block_id * BLOCK_Q;
  
  const int past = len_kv - len_q; 

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;
  O += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  
  const half* __restrict__ MaskBase;
  if (use_mask) {
    MaskBase = M ? (M + q_block_base * len_kv) : nullptr;
  }
  
  // we overlap Q_smem with (K_smem + V_smem), since we only need to load Q_smem once
  extern __shared__ half smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(half);

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  // set up registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

  // let compiler decide register reuse?
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  // rescale O_rmem once we obtain new rowmax, then accumulate to O_rmem for P @ V
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  // pre-compute address and swizzling for ldmatrix
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(half)>(Q_smem + (row_off * DIM + col_off) * sizeof(half));
  }
  {
    // B tile
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(half)>(K_smem + (row_off * DIM + col_off) * sizeof(half));
  }
  {
    // B tile trans
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(half)>(V_smem + (row_off * DIM + col_off) * sizeof(half));
  }

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM]
  global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // shared -> registers
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(half);  // row
      addr ^= mma_id_d * MMA_K * sizeof(half);  // col
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      // double buffer for K
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(half));
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);
      K += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_id) {
    // single buffer for V
    const uint32_t dst = V_smem;
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid);
    V += BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");
  };

  // prefetch K
  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // prefetch V
    // __syncthreads() here is required to make sure we finish using V_smem
    // from the previous iteration, since there is only 1 shared buffer for V.
    __syncthreads();
    load_V(kv_id);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(half));
        addr += mma_id_kv * MMA_N * DIM * sizeof(half);  // row
        addr ^= mma_id_d * MMA_K * sizeof(half);  // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                       K_rmem[mma_id_kv][mma_id_d],
                       S_rmem[mma_id_q][mma_id_kv]);

    // prefetch K
    load_K(kv_id + 1);
    
    const int lane_row0 = (lane_id / 4);     // 0..7
    const int lane_row1 = lane_row0 + 8;     // 8..15
    const int j_pair0   = 2 * (lane_id % 4); // {0,2,4,6}

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // apply softmax scale
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        // 1) scale logits
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] *= scale;
        regs[1] *= scale;
        regs[2] *= scale;
        regs[3] *= scale;

        // 3) causal OR explicit mask
        if (is_causal) {
          const int base_i_block  = warp_id * (BLOCK_Q / NUM_WARPS) + mma_id_q * MMA_M;   // 16 rows in this MMA
          const int col_base_tile = kv_id * BLOCK_KV + mma_id_kv * MMA_N;                 // 8 cols in this MMA
          const int i0 = base_i_block + lane_row0;      // row for regs[0], regs[1]
          const int i1 = base_i_block + lane_row1;      // row for regs[2], regs[3]
          const int j0 = col_base_tile + j_pair0 + 0;   // col for regs[0], regs[2]
          const int j1 = col_base_tile + j_pair0 + 1;   // col for regs[1], regs[3]

          const int i0_global = q_block_base + i0;
          const int i1_global = q_block_base + i1;
          const int j_limit0 = past + i0_global; // last allowed key for row i0
          const int j_limit1 = past + i1_global;

          //// Optional tail handling if not padded:
          //// If you always pad to multiples, you can drop these checks.
          //if (i0 >= len_q) { regs[0] = -CUDART_INF_F; regs[1] = -CUDART_INF_F; }
          //if (i1 >= len_q) { regs[2] = -CUDART_INF_F; regs[3] = -CUDART_INF_F; }
          //if (j0 >= len_kv) { regs[0] = -CUDART_INF_F; regs[2] = -CUDART_INF_F; }
          //if (j1 >= len_kv) { regs[1] = -CUDART_INF_F; regs[3] = -CUDART_INF_F; }
          // zero out (−INF) future positions
          if (j0 > j_limit0) regs[0] = -CUDART_INF_F;
          if (j1 > j_limit0) regs[1] = -CUDART_INF_F;
          if (j0 > j_limit1) regs[2] = -CUDART_INF_F;
          if (j1 > j_limit1) regs[3] = -CUDART_INF_F;
        } else if (use_mask) {
          const int base_i_block  = warp_id * (BLOCK_Q / NUM_WARPS) + mma_id_q * MMA_M;   // 16 rows in this MMA
          const int col_base_tile = kv_id * BLOCK_KV + mma_id_kv * MMA_N;                 // 8 cols in this MMA
          const int i0 = base_i_block + lane_row0;      // row for regs[0], regs[1]
          const int i1 = base_i_block + lane_row1;      // row for regs[2], regs[3]
          const int j0 = col_base_tile + j_pair0 + 0;   // col for regs[0], regs[2]
          const int j1 = col_base_tile + j_pair0 + 1;   // col for regs[1], regs[3]

          // Mask layout: [bs, len_q, len_kv], MaskBase already offset to this (bs, q-block)
          // Values should be 0.0f (keep) or −INF (block) for branchless addition
          //if (i0 < len_q && j0 < len_kv) 
          regs[0] += (float) MaskBase[(size_t)i0 * len_kv + j0];
          //if (i0 < len_q && j1 < len_kv) 
          regs[1] += (float) MaskBase[(size_t)i0 * len_kv + j1];
          //if (i1 < len_q && j0 < len_kv) 
          regs[2] += (float) MaskBase[(size_t)i1 * len_kv + j0];
          //if (i1 < len_q && j1 < len_kv) 
          regs[3] += (float) MaskBase[(size_t)i1 * len_kv + j1];
        }
      }

      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]);  // c0 and c1
          this_rowmax[1] = max(regs[2], regs[3]);  // c2 and c3
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
        }
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        half2 *this_P_rmem = reinterpret_cast<half2 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2]     = __float22half2_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22half2_rn({regs[2], regs[3]});
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread;
        addr += mma_id_kv * MMA_K * DIM * sizeof(half);  // row
        addr ^= mma_id_d * MMA_N * sizeof(half);  // col
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                       V_rmem[mma_id_kv][mma_id_d],
                       O_rmem[mma_id_q][mma_id_d]);
  }

  // write to O
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

      reinterpret_cast<half2 *>(O + (row + 0) * DIM + col)[0] = __float22half2_rn({regs[0], regs[1]});
      reinterpret_cast<half2 *>(O + (row + 8) * DIM + col)[0] = __float22half2_rn({regs[2], regs[3]});
    }
}

#define INSTANTIATE_MINIMAL_FLASH_FOR_MASK_STRATEGY(BLOCK_Q, BLOCK_KV, D, is_causal, use_mask) \
  extern "C" __global__ void attention_v5_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, half* __restrict__ M, half* __restrict__ O, \
    int bs, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask>(Q, K, V, M, O, bs, len_q, len_kv, scale); \
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