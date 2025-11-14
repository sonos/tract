#include "common.cuh"
#include <math_constants.h>

// NOTE: stride in bytes
template <int STRIDE>
static __device__ __forceinline__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

static __device__ inline void cp_async_cg_16B(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
}

static __device__ inline void cp_async_cg_16B_pred(uint32_t dst, const void* src, bool pred) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    ".reg .b32 z;\n\t"
    "mov.b32 z, 0;\n\t"
    "setp.ne.b32 p, %2, 0;\n\t"                           // p = (pred != 0)
    "@p   cp.async.cg.shared.global [%0], [%1], 16;\n\t"  // valid: async copy 16B
    "@!p  st.shared.v4.b32 [%0], {z, z, z, z};\n\t"       // invalid: zero-fill 16B
    "}\n"
    :: "r"(dst), "l"(src), "r"((int)pred));
#else
  // Fallback path if you ever build for pre-SM80
  if (pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
  } else {
    int z = 0;
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(dst), "r"(z));
  }
#endif
}

// ============================================================================
// Global->shared loaders
// ============================================================================
template <int HEIGHT, int WIDTH, int TB_SIZE>
static __device__ __forceinline__
void global_to_shared_swizzle(uint32_t dst, const half* __restrict__ src, int src_stride, int tid) {
  constexpr int elems_per_copy = 16 / sizeof(half);   // 8 f16
  constexpr int iters = HEIGHT * WIDTH / (TB_SIZE * elems_per_copy);
  #pragma unroll
  for (int it = 0; it < iters; ++it) {
    const int idx = (it * TB_SIZE + tid) * elems_per_copy;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const uint32_t dst_addr = swizzle<WIDTH * sizeof(half)>(dst + (row * WIDTH + col) * sizeof(half));
    const half* __restrict__ src_addr = src + (row * src_stride + col);
    cp_async_cg_16B(dst_addr, src_addr);
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
static __device__ __forceinline__
void global_to_shared_swizzle_pred(uint32_t dst, const half* __restrict__ src, int src_stride, int tid, int valid_rows) {
  constexpr int elems_per_copy = 16 / sizeof(half);
  constexpr int iters = HEIGHT * WIDTH / (TB_SIZE * elems_per_copy);
  #pragma unroll
  for (int it = 0; it < iters; ++it) {
    const int idx = (it * TB_SIZE + tid) * elems_per_copy;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const uint32_t dst_addr = swizzle<WIDTH * sizeof(half)>(dst + (row * WIDTH + col) * sizeof(half));
    const half* __restrict__ src_addr = src + (row * src_stride + col);
    const bool in_row = (row < valid_rows);
    cp_async_cg_16B_pred(dst_addr, src_addr, in_row);
  }
}

// ============================================================================
// Tensor Core helpers
// ============================================================================
static __device__ __forceinline__
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(addr));
}
static __device__ __forceinline__
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(addr));
}
static __device__ __forceinline__
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

// ============================================================================
// Select and predicated global store
// ============================================================================
static __device__ __forceinline__ float fsel_neg_inf(bool keep, float x) {
  float out;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %2, 0;\n\t"
    "selp.f32 %0, %1, %3, p;\n\t"
    "}\n"
    : "=f"(out) : "f"(x), "r"((int)keep), "f"(-CUDART_INF_F));
  return out;
}

static __device__ __forceinline__ void st_global_half2_pred(half2* addr, half2 val, bool pred) {
#if __CUDA_ARCH__ >= 700
  uint32_t v;
  memcpy(&v, &val, sizeof(v)); // pack half2 to 32b
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %2, 0;\n\t"
    "@p st.global.b32 [%0], %1;\n\t"
    "}\n"
    :: "l"(addr), "r"(v), "r"((int)pred));
#else
  if (pred) { *addr = val; }
#endif
}

// ======= kv_iter_body (invariants hoisted, tight scopes) =====================
template<
  int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS,
  bool is_causal, bool use_mask, bool full_q_tile, bool kv_tail
>
static __device__ __forceinline__
void kv_iter_body(
    const int kv_tile_base,
    const int len_q, const int len_kv, const int past,
    const int q_block_base, const int warp_id, const int lane_id,
    float scale, const half* __restrict__ MaskBase,
    float S_rmem[ (BLOCK_Q/NUM_WARPS) / 16 ][ BLOCK_KV / 8 ][4],
    uint32_t (&P_rmem)[ (BLOCK_Q/NUM_WARPS) / 16 ][ BLOCK_KV / 16 ][4],
    float (&O_rmem)[ (BLOCK_Q/NUM_WARPS) / 16 ][ DIM / 8 ][4],
    float (&rowmax)[ (BLOCK_Q/NUM_WARPS) / 16 ][2],
    float (&rowsumexp)[ (BLOCK_Q/NUM_WARPS) / 16 ][2]
) {
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
  constexpr int MMA_M = 16, MMA_N = 8;

  // hoisted per-lane constants
  const int lane_row0 = (lane_id >> 2);      // 0..7
  const int lane_row1 = lane_row0 + 8;
  const int j_pair0   = (lane_id & 3) << 1;

  // per-qi
  #pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi) {
    // hoisted per-qi invariants
    const int base_i_block = warp_id * (BLOCK_Q / NUM_WARPS) + qi * MMA_M;
    const int i0           = base_i_block + lane_row0;
    const int i1           = base_i_block + lane_row1;
    const int i0g          = q_block_base + i0;
    const int i1g          = q_block_base + i1;
    int jlim0 = 0, jlim1 = 0;
    if constexpr (is_causal) { jlim0 = past + i0g; jlim1 = past + i1g; }

    // ---- scale, (mask/causal), rowmax (tight scope for S_rmem live range) ---
    float this_rowmax0 = -CUDART_INF_F;
    float this_rowmax1 = -CUDART_INF_F;

    #pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt) {
      float* r = S_rmem[qi][kvt];
      // scale
      r[0] *= scale; r[1] *= scale; r[2] *= scale; r[3] *= scale;

      const int col_base_tile = kv_tile_base + kvt * MMA_N;
      const int j0 = col_base_tile + j_pair0 + 0;
      const int j1 = col_base_tile + j_pair0 + 1;

      if constexpr (kv_tail) {
        // tail validity
        const bool i0_valid = full_q_tile ? true : (i0g < len_q);
        const bool i1_valid = full_q_tile ? true : (i1g < len_q);
        const bool j0_valid = (j0 < len_kv);
        const bool j1_valid = (j1 < len_kv);
        bool c00 = j0_valid & i0_valid;
        bool c01 = j1_valid & i0_valid;
        bool c10 = j0_valid & i1_valid;
        bool c11 = j1_valid & i1_valid;

        if constexpr (is_causal) {
          c00 &= (j0 <= jlim0); c01 &= (j1 <= jlim0);
          c10 &= (j0 <= jlim1); c11 &= (j1 <= jlim1);
        } else if constexpr (use_mask) {
          if (i0_valid && j0_valid) r[0] += (float)MaskBase[(size_t)i0 * len_kv + j0];
          if (i0_valid && j1_valid) r[1] += (float)MaskBase[(size_t)i0 * len_kv + j1];
          if (i1_valid && j0_valid) r[2] += (float)MaskBase[(size_t)i1 * len_kv + j0];
          if (i1_valid && j1_valid) r[3] += (float)MaskBase[(size_t)i1 * len_kv + j1];
        }

        r[0] = fsel_neg_inf(c00, r[0]);
        r[1] = fsel_neg_inf(c01, r[1]);
        r[2] = fsel_neg_inf(c10, r[2]);
        r[3] = fsel_neg_inf(c11, r[3]);
      } else {
        // full tiles: no row/col bounds
        if constexpr (is_causal) {
          if (j0 > jlim0) r[0] = -CUDART_INF_F;
          if (j1 > jlim0) r[1] = -CUDART_INF_F;
          if (j0 > jlim1) r[2] = -CUDART_INF_F;
          if (j1 > jlim1) r[3] = -CUDART_INF_F;
        } else if constexpr (use_mask) {
            r[0] += (float)MaskBase[(size_t)i0 * len_kv + j0];
            r[1] += (float)MaskBase[(size_t)i0 * len_kv + j1];
            r[2] += (float)MaskBase[(size_t)i1 * len_kv + j0];
            r[3] += (float)MaskBase[(size_t)i1 * len_kv + j1];
        }
      }

      // rowmax accumulate
      this_rowmax0 = fmaxf(this_rowmax0, fmaxf(r[0], r[1]));
      this_rowmax1 = fmaxf(this_rowmax1, fmaxf(r[2], r[3]));
    }

    // small warp reduce (xor tree 1,2)
    this_rowmax0 = fmaxf(this_rowmax0, __shfl_xor_sync(0xFFFFFFFF, this_rowmax0, 1));
    this_rowmax0 = fmaxf(this_rowmax0, __shfl_xor_sync(0xFFFFFFFF, this_rowmax0, 2));
    this_rowmax1 = fmaxf(this_rowmax1, __shfl_xor_sync(0xFFFFFFFF, this_rowmax1, 1));
    this_rowmax1 = fmaxf(this_rowmax1, __shfl_xor_sync(0xFFFFFFFF, this_rowmax1, 2));

    this_rowmax0 = fmaxf(this_rowmax0, rowmax[qi][0]);
    this_rowmax1 = fmaxf(this_rowmax1, rowmax[qi][1]);
    // rescale accumulators with EXACT same factor you’ll use for denominators
    const float rescale0 = __expf(rowmax[qi][0] - this_rowmax0);
    const float rescale1 = __expf(rowmax[qi][1] - this_rowmax1);

    #pragma unroll
    for (int d = 0; d < DIM / MMA_N; ++d) {
      O_rmem[qi][d][0] *= rescale0; O_rmem[qi][d][1] *= rescale0;
      O_rmem[qi][d][2] *= rescale1; O_rmem[qi][d][3] *= rescale1;
    }
    rowmax[qi][0] = this_rowmax0;
    rowmax[qi][1] = this_rowmax1;
    float this_rowsumexp0 = 0.f, this_rowsumexp1 = 0.f;

    // exponentiate, pack directly into P_rmem (keep S_rmem live only here)
    #pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt) {
      float* r = S_rmem[qi][kvt];
      r[0] = __expf(r[0] - this_rowmax0);
      r[1] = __expf(r[1] - this_rowmax0);
      r[2] = __expf(r[2] - this_rowmax1);
      r[3] = __expf(r[3] - this_rowmax1);

      this_rowsumexp0 += (r[0] + r[1]);
      this_rowsumexp1 += (r[2] + r[3]);

      half2* Ppack = reinterpret_cast<half2*>(P_rmem[qi][kvt / 2]);
      Ppack[(kvt & 1) * 2    ] = __float22half2_rn({r[0], r[1]});
      Ppack[(kvt & 1) * 2 + 1] = __float22half2_rn({r[2], r[3]});
    }

    // reduce sums (xor 1,2)
    this_rowsumexp0 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp0, 1);
    this_rowsumexp0 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp0, 2);
    this_rowsumexp1 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp1, 1);
    this_rowsumexp1 += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp1, 2);

    rowsumexp[qi][0] = rowsumexp[qi][0] * rescale0 + this_rowsumexp0;
    rowsumexp[qi][1] = rowsumexp[qi][1] * rescale1 + this_rowsumexp1;
  }
}


// ============================================================================
// Kernel. Adapted from: https://github.com/gau-nernst/learn-cuda/blob/main/07_attention/attention_v5.cu
// ============================================================================
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, bool is_causal, bool use_mask, bool full_q_tile, bool kv_rem>
static __device__ void attention_v5_kernel(
  const half* __restrict__ Q,  // [bs, len_q, DIM]
  const half* __restrict__ K,  // [bs, len_kv, DIM]
  const half* __restrict__ V,  // [bs, len_kv, DIM]
  const half* __restrict__ M,  // [bs, len_q, len_kv]
  half* __restrict__ O,        // [bs, len_q, DIM]
  int bs, int qh, int head_ratio, int len_q, int len_kv, float scale)
{
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int WARP_Q  = BLOCK_Q / NUM_WARPS;
  constexpr int MMA_M   = 16, MMA_N = 8, MMA_K = 16;

  const int bid = blockIdx.z;
  const int hid = blockIdx.y;
  const int q_block_id = blockIdx.x;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  int q_block_id_offset = 0;
  if constexpr(!full_q_tile) {
    q_block_id_offset = (len_q / BLOCK_Q);
  }
  const int q_block_base = (q_block_id + q_block_id_offset) * BLOCK_Q;
  const int q_valid = max(0, min(BLOCK_Q, len_q - q_block_base));
  const int past = len_kv - len_q; // causal base

  const int q_heads   = qh;
  const int kv_heads  = q_heads / head_ratio;
  const int kv_head_id = hid / head_ratio;

  // Base pointers
  const half* Qptr = Q + (((size_t)bid * q_heads + hid) * (size_t)len_q + q_block_base) * DIM;
  const half* Kptr = K + (((size_t)bid * kv_heads + kv_head_id) * (size_t)len_kv) * DIM;
  const half* Vptr = V + (((size_t)bid * kv_heads + kv_head_id) * (size_t)len_kv) * DIM;
  half*       Optr = O + (((size_t)bid * q_heads + hid) * (size_t)len_q + q_block_base) * DIM;

  const half* __restrict__ MaskBase =
    (use_mask ? (M ? (M + (size_t)q_block_base * len_kv) : nullptr) : nullptr);

  // Shared memory layout:
  // Q_smem (BLOCK_Q x DIM) overlaps K_smem (2 * BLOCK_KV x DIM), plus V_smem (BLOCK_KV x DIM)
  extern __shared__ half smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(half);

  // Per-thread swizzled bases
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

  // Registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  // Softmax accumulators
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};
#pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi) {
    rowmax[qi][0] = -FLT_MAX;
    rowmax[qi][1] = -FLT_MAX;
  }

  // ------------------ Load Q (tail-safe for last block only) -----------------
  if constexpr (full_q_tile) {
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Qptr, DIM, tid);
  } else {
    global_to_shared_swizzle_pred<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Qptr, DIM, tid, q_valid);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Q: shared -> regs
#pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
#pragma unroll
    for (int dk = 0; dk < DIM / MMA_K; ++dk) {
      uint32_t addr = Q_smem_thread + qi * MMA_M * DIM * sizeof(half);
      addr ^= dk * MMA_K * sizeof(half);
      ldmatrix_x4(Q_rmem[qi][dk], addr);
    }
  __syncthreads();

  // ------------------ KV split: full tiles then optional tail ----------------
  const int kv_full_iters = len_kv / BLOCK_KV;      // exact full tiles

  // ---- Prefetch K0 for FULL loop (unguarded) ----
  const half* Kcur = Kptr;
  const half* Vcur = Vptr;

  if (kv_full_iters > 0) {
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem + 0, Kcur, DIM, tid);
    Kcur += (size_t)BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");
  }

  // ----------------------------- FULL KV LOOP -------------------------------
  for (int kv_id = 0; kv_id < kv_full_iters; ++kv_id) {
    const int kv_tile_base = kv_id * BLOCK_KV; // columns start for this tile

    // Prefetch V (unguarded)
    __syncthreads();
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, Vcur, DIM, tid);
    Vcur += (size_t)BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");

    // Wait K, load K into regs
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
#pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
#pragma unroll
      for (int dk = 0; dk < DIM / MMA_K; dk += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(half));
        addr += kvt * MMA_N * DIM * sizeof(half);
        addr ^=  dk * MMA_K * sizeof(half);
        ldmatrix_x4(K_rmem[kvt][dk], addr);
      }
    
    {
      // S = Q @ K^T
      float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
#pragma unroll
      for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
#pragma unroll
        for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
#pragma unroll
          for (int dk = 0; dk < DIM / MMA_K; ++dk)
            mma_m16n8k16(Q_rmem[qi][dk], K_rmem[kvt][dk], S_rmem[qi][kvt]);

      // Prefetch next FULL K (unguarded)
      if (kv_id + 1 < kv_full_iters) {
        const uint32_t Kdst = K_smem + ((kv_id + 1) % 2) * (BLOCK_KV * DIM * sizeof(half));
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(Kdst, Kcur, DIM, tid);
        Kcur += (size_t)BLOCK_KV * DIM;
        asm volatile("cp.async.commit_group;");
      }

      kv_iter_body<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, is_causal, use_mask, full_q_tile, false>(
        kv_tile_base, len_q, len_kv, past, q_block_base, warp_id, lane_id, scale,
        MaskBase, S_rmem, P_rmem, O_rmem, rowmax, rowsumexp
      );
    }

    // Wait V, load V and do O += P@V
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

#pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_K; ++kvt)
#pragma unroll
      for (int d = 0; d < DIM / MMA_N; d += 2) {
        uint32_t addr = V_smem_thread + kvt * MMA_K * DIM * sizeof(half);
        addr ^= d * MMA_N * sizeof(half);
        ldmatrix_x4_trans(V_rmem[kvt][d], addr);
      }

#pragma unroll
    for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
#pragma unroll
      for (int d = 0; d < DIM / MMA_N; ++d)
#pragma unroll
        for (int kvt = 0; kvt < BLOCK_KV / MMA_K; ++kvt) {
          mma_m16n8k16(P_rmem[qi][kvt], V_rmem[kvt][d], O_rmem[qi][d]);
        }
  }

  // ----------------------------- KV TAIL (optional) --------------------------
  if constexpr(kv_rem) {
    const int kv_tile_base = kv_full_iters * BLOCK_KV;

    // Prefetch tail K (predicated)
    const uint32_t Kdst = K_smem + (kv_full_iters % 2) * (BLOCK_KV * DIM * sizeof(half));
    global_to_shared_swizzle_pred<BLOCK_KV, DIM, TB_SIZE>(Kdst, Kcur, DIM, tid, len_kv % BLOCK_KV);
    Kcur += (size_t)BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");

    __syncthreads();
    // Prefetch tail V (predicated)
    global_to_shared_swizzle_pred<BLOCK_KV, DIM, TB_SIZE>(V_smem, Vcur, DIM, tid, len_kv % BLOCK_KV);
    Vcur += (size_t)BLOCK_KV * DIM;
    asm volatile("cp.async.commit_group;");

    // Wait K, load K regs
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
#pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
#pragma unroll
      for (int dk = 0; dk < DIM / MMA_K; dk += 2) {
        uint32_t addr = K_smem_thread + (kv_full_iters % 2) * (BLOCK_KV * DIM * sizeof(half));
        addr += kvt * MMA_N * DIM * sizeof(half);
        addr ^=  dk * MMA_K * sizeof(half);
        ldmatrix_x4(K_rmem[kvt][dk], addr);
      }
    
    {
      // S = Q @ K^T
      float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
  #pragma unroll
      for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
  #pragma unroll
        for (int kvt = 0; kvt < BLOCK_KV / MMA_N; ++kvt)
  #pragma unroll
          for (int dk = 0; dk < DIM / MMA_K; ++dk)
            mma_m16n8k16(Q_rmem[qi][dk], K_rmem[kvt][dk], S_rmem[qi][kvt]);

      kv_iter_body<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, is_causal, use_mask, full_q_tile, true>(
        kv_tile_base, len_q, len_kv, past, q_block_base, warp_id, lane_id, scale,
        MaskBase, S_rmem, P_rmem, O_rmem, rowmax, rowsumexp
      );
    }
    // Wait V and finish O += P@V (tail)
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
#pragma unroll
    for (int kvt = 0; kvt < BLOCK_KV / MMA_K; ++kvt)
#pragma unroll
      for (int d = 0; d < DIM / MMA_N; d += 2) {
        uint32_t addr = V_smem_thread + kvt * MMA_K * DIM * sizeof(half);
        addr ^= d * MMA_N * sizeof(half);
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

  // ----------------------------- Writeback (Q tail-safe) ---------------------
#pragma unroll
  for (int qi = 0; qi < WARP_Q / MMA_M; ++qi)
#pragma unroll
    for (int d = 0; d < DIM / MMA_N; ++d) {
      const int row0 = warp_id * WARP_Q + qi * MMA_M + (lane_id / 4);
      const int col  = d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[qi][d];
      regs[0] /= rowsumexp[qi][0];
      regs[1] /= rowsumexp[qi][0];
      regs[2] /= rowsumexp[qi][1];
      regs[3] /= rowsumexp[qi][1];

      half2 v0 = __float22half2_rn({regs[0], regs[1]});
      half2 v1 = __float22half2_rn({regs[2], regs[3]});

      half2* p0 = reinterpret_cast<half2 *>(Optr + (row0 + 0) * DIM + col);
      half2* p1 = reinterpret_cast<half2 *>(Optr + (row0 + 8) * DIM + col);

      if constexpr (full_q_tile) {
        // No predicates in full Q tiles
        *p0 = v0;
        *p1 = v1;
      } else {
        const bool valid0 = (q_block_base + row0)     < len_q;
        const bool valid1 = (q_block_base + row0 + 8) < len_q;
        st_global_half2_pred(p0, v0, valid0);
        st_global_half2_pred(p1, v1, valid1);
      }
    }
}

#define INSTANTIATE_FLASH_ATTN_FOR_MASK_STRATEGY(BLOCK_Q, BLOCK_KV, D, is_causal, use_mask) \
extern "C" {  \
    __launch_bounds__(4 * WARP_SIZE) \
    __global__ void attention_v5_full_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask ( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, const half* __restrict__ M, half* __restrict__ O, \
    int bs, int qh, int head_ratio, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask, true, false>( \
        Q, K, V, M, O, bs, qh, head_ratio, len_q, len_kv, scale); \
  } \
\
    __launch_bounds__(4 * WARP_SIZE) \
    __global__ void attention_v5_tail_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask ( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, const half* __restrict__ M, half* __restrict__ O, \
    int bs, int qh, int head_ratio, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask, false, false>( \
        Q, K, V, M, O, bs, qh, head_ratio, len_q, len_kv, scale); \
  } \
\
    __launch_bounds__(4 * WARP_SIZE) \
    __global__ void attention_v5_full_kv_rem_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask ( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, const half* __restrict__ M, half* __restrict__ O, \
    int bs, int qh, int head_ratio, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask, true, true>( \
        Q, K, V, M, O, bs, qh, head_ratio, len_q, len_kv, scale); \
  } \
\
    __launch_bounds__(4 * WARP_SIZE) \
    __global__ void attention_v5_tail_kv_rem_##BLOCK_Q##_##BLOCK_KV##_##D##_##is_causal##_##use_mask ( \
    const half* __restrict__ Q, const half* __restrict__ K, \
    const half* __restrict__ V, const half* __restrict__ M, half* __restrict__ O, \
    int bs, int qh, int head_ratio, int len_q, int len_kv, float scale) { \
      attention_v5_kernel<BLOCK_Q, BLOCK_KV, D, 4, is_causal, use_mask, false, true>( \
        Q, K, V, M, O, bs, qh, head_ratio, len_q, len_kv, scale); \
  } \
}

#define INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, D) \
  INSTANTIATE_FLASH_ATTN_FOR_MASK_STRATEGY(block_q, block_kv, D, false, false) \
  INSTANTIATE_FLASH_ATTN_FOR_MASK_STRATEGY(block_q, block_kv, D, false, true) \
  INSTANTIATE_FLASH_ATTN_FOR_MASK_STRATEGY(block_q, block_kv, D, true, false) \

#define INSTANTIATE_FLASH_ATTN_FOR_BLOCK_KV(block_q, block_kv) \
  INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 64) \
  INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 128) \
  INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 80) \
  // Other supported D value. 
  //Never encountered in practice so commented to keep compilation fast
  //INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 96) \
  //INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 112) \
  //INSTANTIATE_FLASH_ATTN_FOR_D(block_q, block_kv, 256) \

#define INSTANTIATE_FLASH_ATTN_FOR_BLOCK_Q(block_q) \
  INSTANTIATE_FLASH_ATTN_FOR_BLOCK_KV(block_q, 32)

#define INSTANTIATE_FLASH_ATTN() \
  INSTANTIATE_FLASH_ATTN_FOR_BLOCK_Q(64)

INSTANTIATE_FLASH_ATTN()