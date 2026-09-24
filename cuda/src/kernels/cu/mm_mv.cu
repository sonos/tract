
#include <cuda_runtime.h>
#include "common.cuh"

template <typename T, int ncols_dst, int block_size>
static __device__ void
mul_mat_vec(const T *__restrict__ x, const T *__restrict__ y,
            T *__restrict__ dst, const int32_t ncols2, const int32_t nchannels_y,
            const int32_t stride_row, const int32_t stride_col_y2,
            const int32_t stride_col_dst, const int32_t channel_ratio,
            const int32_t stride_channel_x, const int32_t stride_channel_y,
            const int32_t stride_channel_dst,
            const int32_t *__restrict__ x_ring,
            const int32_t ring_rotates_k) {
  const int row = blockIdx.x;
  const int channel_dst = blockIdx.y;
  const int channel_x = channel_dst / channel_ratio;
  const int channel_y = channel_dst;
  const int tid = threadIdx.x;

  // How far this channel's window has turned, in pairs of columns, when the
  // ring turns along the contracted axis rather than along the rows.
  int ring_off2 = 0;
  if (x_ring == nullptr) {
    x += channel_x * stride_channel_x + row * stride_row;
  } else if (ring_rotates_k) {
    // x holds each channel's contracted axis as a ring of whole slots, and more
    // channels than the grid covers: the table gives how far the window has
    // turned and where this channel's rows start.
    ring_off2 = x_ring[2 * channel_dst];
    x += (x_ring[2 * channel_dst + 1] + row) * stride_row;
  } else {
    // x holds each channel's gridDim.x rows as a ring, and more channels than
    // the grid covers: the table gives this channel's first row and where its
    // rows start.
    int row_x = row + x_ring[2 * channel_dst];
    if (row_x >= (int)gridDim.x) {
      row_x -= (int)gridDim.x;
    }
    x += (x_ring[2 * channel_dst + 1] + row_x) * stride_row;
  }
  y += channel_y * stride_channel_y;
  dst += channel_dst * stride_channel_dst;

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int *)&__shm[0]);
  constexpr int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
  // buf_iw needs max(WARP_SIZE, num_warps) entries: indexed by both lane_id
  // (up to WARP_SIZE-1) and tid/WARP_SIZE (up to num_warps-1).
  constexpr int buf_iw_size = (WARP_SIZE > num_warps) ? WARP_SIZE : num_warps;
  float (&buf_iw)[buf_iw_size] = al.allocate<float, buf_iw_size>();

  if (block_size > WARP_SIZE) {
    if (tid < WARP_SIZE) {
      buf_iw[tid] = 0.0f;
    }
    __syncthreads();
  }

  float sumf[ncols_dst] = {0.0f};

  if constexpr (cuda::std::is_same_v<T, float>) {
    const float2 *x2 = (const float2 *)x;
    const float2 *y2 = (const float2 *)y;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
      int col2_x = col2 + ring_off2;
      if (col2_x >= ncols2) {
        col2_x -= ncols2;
      }
      const float2 tmpx = x2[col2_x];

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = y2[j * stride_col_y2 + col2];
        sumf[j] += tmpx.x * tmpy.x;
        sumf[j] += tmpx.y * tmpy.y;
      }
    }
  } else if constexpr (cuda::std::is_same_v<T, half>) {
    const half2 *x2 = (const half2 *)x;
    const half2 *y2 = (const half2 *)y;
    half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
      int col2_x = col2 + ring_off2;
      if (col2_x >= ncols2) {
        col2_x -= ncols2;
      }
      const half2 tmpx = x2[col2_x];

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const half2 tmpy = y2[j * stride_col_y2 + col2];
        sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
      }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
    }
  } else {
    static_assert(cuda::std::is_same_v<T, void>, "unsupported type");
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);

    if (block_size > WARP_SIZE) {
      buf_iw[tid / WARP_SIZE] = sumf[j];
      __syncthreads();
      if (tid < WARP_SIZE) {
        sumf[j] = buf_iw[tid];
        sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
      }
      if (j < ncols_dst) {
        __syncthreads();
      }
    }
  }

  if (tid >= ncols_dst) {
    return;
  }

  dst[tid * stride_col_dst + row] = sumf[tid];
}

#define INSTANTIATE_MAT_VEC(type_name, T, ncols_dst, block_size)               \
  extern "C" __global__ void                                                   \
      ggml_matvec_##type_name##_ncols_##ncols_dst##_bs_##block_size(           \
          const T *__restrict__ x, const T *__restrict__ y,                    \
          T *__restrict__ dst, const int32_t ncols2, const int32_t nchannels_y,        \
          const int32_t stride_row, const int32_t stride_col_y2,                       \
          const int32_t stride_col_dst, const int32_t channel_ratio,                   \
          const int32_t stride_channel_x, const int32_t stride_channel_y,              \
          const int32_t stride_channel_dst,                                    \
          const int32_t *__restrict__ x_ring,                                  \
          const int32_t ring_rotates_k) {                                      \
    mul_mat_vec<T, ncols_dst, block_size>(                                     \
        x, y, dst, ncols2, nchannels_y, stride_row, stride_col_y2,             \
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,     \
        stride_channel_dst, x_ring, ring_rotates_k);                           \
  }

#define INSTANTIATE_MAT_VEC_FOR_BS(name, T, blocksize)                         \
  INSTANTIATE_MAT_VEC(name, T, 1, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 2, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 3, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 4, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 5, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 6, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 7, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 8, blocksize)

#define INSTANTIATE_MAT_VEC_FOR_T(name, T)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 32)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 64)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 96)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 128)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 160)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 192)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 224)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 256)

INSTANTIATE_MAT_VEC_FOR_T(f32, float)
INSTANTIATE_MAT_VEC_FOR_T(f16, half)

// ============================================================================
// WGMMA-accelerated GEMM (SM90+) — ThunderKittens Level-05 pattern.
//
// C = A @ B^T   where  A: (m, k) fp16 (activations),  B: (n, k) fp16 (weights),
//                    C: (m, n) fp16 or fp32
//
// Uses WGMMA m64n32k16.f32.f16.f16 (32 768 MACs / instruction vs 2 048 for
// mma_m16n8k16) on a 4-warps × 32-lane block (128 threads).  Each block
// computes one 64×32 output tile, advancing along K in chunks of 16.
//
// WGMMA accumulator layout (m64n32k16, 128 threads):
//   thread tid → row = tid/2, col_base = (tid%2)*16
//   D[i].x → (row, col_base + i*2)
//   D[i].y → (row, col_base + i*2 + 1)
//
// Kernel name matches the dispatch in matmul/mod.rs → dispatch_wgmma_gemm.
// ============================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#define WGMMA_M_TILE 64
#define WGMMA_N_TILE 32
#define WGMMA_K_CHUNK 16
#define WGMMA_TILE_BYTES ((WGMMA_M_TILE * WGMMA_K_CHUNK + WGMMA_N_TILE * WGMMA_K_CHUNK) * sizeof(half))
#define WGMMA_SHARED_BYTES_DB (2 * WGMMA_TILE_BYTES)  // double-buffered

template <typename T>
__launch_bounds__(128) __global__
void ggml_matmul_wgmma_impl(const half *__restrict__ A, // acts     (m, k)
                            const half *__restrict__ B, // weights  (n, k)
                            T *__restrict__ C,          // output   (m, n)
                            int32_t m, int32_t n, int32_t k,
                            int32_t lda, int32_t ldb, int32_t ldc,
                            int32_t stride_a, int32_t stride_b, int32_t stride_c) {
    using namespace cuda_wgmma;
    using cuda::std::is_same_v;

    constexpr int M_TILE = WGMMA_M_TILE, N_TILE = WGMMA_N_TILE, K_CHUNK = WGMMA_K_CHUNK;

    const int row_base = blockIdx.x * M_TILE;
    const int col_base = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    (void)lane_id;

    // Batch dimension
    const int batch = blockIdx.z;
    A += size_t(batch) * stride_a;
    B += size_t(batch) * stride_b;
    C += size_t(batch) * stride_c;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);

    // Double-buffered shared memory: tile 0 and tile 1
    auto &a_smem0 = al.allocate<half, M_TILE * K_CHUNK>();
    auto &b_smem0 = al.allocate<half, N_TILE * K_CHUNK>();
    auto &a_smem1 = al.allocate<half, M_TILE * K_CHUNK>();
    auto &b_smem1 = al.allocate<half, N_TILE * K_CHUNK>();
    half *buf_a[2] = { &a_smem0[0], &a_smem1[0] };
    half *buf_b[2] = { &b_smem0[0], &b_smem1[0] };
    int cur = 0;

    // Accumulator: 16 f32 per thread = 8 float2.
    float2 D[8] = {};

    // --- Preload K-chunk 0 (cp.async A + direct B) ---
    {
        const int idx = tid * 8;
        const int a_row = idx / K_CHUNK;
        const int a_col = idx % K_CHUNK;
        const bool va = (row_base + a_row < m) && (a_col + 8 <= k);
        const half *sa = B + size_t(row_base + a_row) * ldb + a_col;
        cp_async_ca_16B_pred(__cvta_generic_to_shared(&buf_a[0][idx]),
                             va ? sa : (const half *)B, va);

        const int bidx = tid * 4;
        half *bd = &buf_b[0][bidx];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int e = bidx + i;
            const int bt_row = e / N_TILE;
            const int bt_col = e % N_TILE;
            bd[i] = (col_base + bt_col < n && bt_row < k)
                        ? A[size_t(col_base + bt_col) * lda + bt_row]
                        : __float2half(0.0f);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // --- K dimension loop with cp.async double-buffering ---
    // Pipeline: issue WGMMA on current buffer, then preload next buffer
    // (cp.async A + direct B) while WGMMA computes in the background.
    for (int kt = 0; kt < k; kt += K_CHUNK) {
        const int nxt = 1 - cur;
        const int next_kt = kt + K_CHUNK;

        // WGMMA descriptors for current tile
        const uint32_t a_addr = __cvta_generic_to_shared(buf_a[cur]);
        const uint32_t b_addr = __cvta_generic_to_shared(buf_b[cur]);
        const uint32_t a_stride = K_CHUNK * sizeof(half);
        const uint64_t b_desc = make_b_desc(b_addr, N_TILE * sizeof(half), 0);

        // Issue WGMMA MMA (async — hardware starts computing)
        mma_m64n32k16_fp16_fp32(D, a_addr, a_stride, b_desc, 1, 0, 1);
        fence();
        commit_group();

        // Preload next K-chunk (overlaps with WGMMA compute)
        if (next_kt < k) {
            const int idx = tid * 8;
            const int a_row = idx / K_CHUNK;
            const int a_col = idx % K_CHUNK;
            const bool va = (row_base + a_row < m) && (next_kt + a_col + 8 <= k);
            const half *sa = B + size_t(row_base + a_row) * ldb + next_kt + a_col;
            cp_async_ca_16B_pred(__cvta_generic_to_shared(&buf_a[nxt][idx]),
                                 va ? sa : (const half *)B, va);

            const int bidx = tid * 4;
            half *bd = &buf_b[nxt][bidx];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e = bidx + i;
                const int bt_row = e / N_TILE;
                const int bt_col = e % N_TILE;
                bd[i] = (col_base + bt_col < n && next_kt + bt_row < k)
                            ? A[size_t(col_base + bt_col) * lda + next_kt + bt_row]
                            : __float2half(0.0f);
            }
            cp_async_commit();
        }

        // Wait for WGMMA to complete
        wait_group(0);

        // Wait for preload, sync, and swap buffers
        if (next_kt < k) {
            cp_async_wait_all();
            __syncthreads();
            cur = nxt;
        } else {
            __syncthreads();
        }
    }

    // --- Store accumulator D to global output C ---
    const int row = tid / 2;
    const int col_off = (tid % 2) * 16;
    if constexpr (is_same_v<T, float>) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int c = col_base + col_off + i * 2;
            if (row_base + row < m) {
                if (c < n)      C[size_t(row_base + row) * ldc + c]     = D[i].x;
                if (c + 1 < n)  C[size_t(row_base + row) * ldc + c + 1] = D[i].y;
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int c = col_base + col_off + i * 2;
            if (row_base + row < m) {
                if (c < n) {
                    C[size_t(row_base + row) * ldc + c] =
                        __float2half(D[i].x);
                }
                if (c + 1 < n) {
                    C[size_t(row_base + row) * ldc + c + 1] =
                        __float2half(D[i].y);
                }
            }
        }
    }
}

#define DEFINE_WGMMA_GEMM(name, T) \
    extern "C" __global__ void name( \
        const half *__restrict__ A, const half *__restrict__ B, \
        T *__restrict__ C, int32_t m, int32_t n, int32_t k, \
        int32_t lda, int32_t ldb, int32_t ldc, \
        int32_t stride_a, int32_t stride_b, int32_t stride_c) { \
        ggml_matmul_wgmma_impl<T>(A, B, C, m, n, k, \
            lda, ldb, ldc, stride_a, stride_b, stride_c); \
    }

DEFINE_WGMMA_GEMM(ggml_matmul_wgmma_f16_f32, float)
DEFINE_WGMMA_GEMM(ggml_matmul_wgmma_f16_f16, half)
#undef DEFINE_WGMMA_GEMM

// ============================================================================
// TMA WGMMA GEMM: uses TMA bulk-tensor loads for both A (weights) and B
// (activations) matrices, with mbarrier synchronization for both tiles.
//
// Kernel name matches the dispatch in matmul/mod.rs → dispatch_wgmma_gemm_tma.
// ============================================================================

/// Preload a K-chunk of both A (weights) and B (activations) via TMA.
/// Both loads are issued by tid==0 and synchronized via separate mbarriers.
template <int M_TILE, int N_TILE, int K_CHUNK>
static __device__ __forceinline__ void
tma_load_ab(const half *weights, const half *acts,
            const int row_base, const int col_base, const int kt,
            half *dst_a, half *dst_b,
            const cuda_tensor_map *a_desc, const cuda_tensor_map *b_desc,
            cuda_mbar &mbar_a, cuda_mbar &mbar_b) {
    constexpr uint32_t a_bytes = M_TILE * K_CHUNK * sizeof(half);
    constexpr uint32_t b_bytes = N_TILE * K_CHUNK * sizeof(half);
    if (threadIdx.x == 0) {
        const uint32_t a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst_a));
        const uint32_t b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst_b));
        const uint32_t a_mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_a));
        const uint32_t b_mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_b));
        // A tile: c=row_base (WGMMA M dim), r=kt (K dim)
        mbarrier_arrive_expect_tx(mbar_a, a_bytes);
        cp_async_bulk_tensor_4d(a_smem, a_desc,
                                (uint32_t)row_base, (uint32_t)kt, 0, 0, a_mbar);
        // B tile: c=col_base (WGMMA N dim → activations rows), r=kt (K dim)
        mbarrier_arrive_expect_tx(mbar_b, b_bytes);
        cp_async_bulk_tensor_4d(b_smem, b_desc,
                                (uint32_t)col_base, (uint32_t)kt, 0, 0, b_mbar);
    }
    if (threadIdx.x == 0) {
        mbarrier_wait_parity(mbar_a, 0);
        mbarrier_wait_parity(mbar_b, 0);
    }
    __syncthreads();
}

template <typename T>
__launch_bounds__(128) __global__
void ggml_matmul_wgmma_tma_impl(const half *__restrict__ A, // acts     (m, k)
                                const half *__restrict__ B, // weights  (n, k)
                                T *__restrict__ C,          // output   (m, n)
                                int32_t m, int32_t n, int32_t k,
                                int32_t lda, int32_t ldb, int32_t ldc,
                                int32_t stride_a, int32_t stride_b, int32_t stride_c,
                                const cuda_tensor_map *a_tma_desc,
                                const cuda_tensor_map *b_tma_desc) {
    using namespace cuda_wgmma;
    using cuda::std::is_same_v;

    constexpr int M_TILE = WGMMA_M_TILE, N_TILE = WGMMA_N_TILE, K_CHUNK = WGMMA_K_CHUNK;

    const int row_base = blockIdx.x * M_TILE;
    const int col_base = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    (void)lane_id;

    const int batch = blockIdx.z;
    A += size_t(batch) * stride_a;
    B += size_t(batch) * stride_b;
    C += size_t(batch) * stride_c;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);

    auto &a_smem0 = al.allocate<half, M_TILE * K_CHUNK>();
    auto &b_smem0 = al.allocate<half, N_TILE * K_CHUNK>();
    auto &a_smem1 = al.allocate<half, M_TILE * K_CHUNK>();
    auto &b_smem1 = al.allocate<half, N_TILE * K_CHUNK>();
    half *buf_a[2] = { &a_smem0[0], &a_smem1[0] };
    half *buf_b[2] = { &b_smem0[0], &b_smem1[0] };
    int cur = 0;

    // 4 mbarriers: 2 for A tiles, 2 for B tiles (double-buffered)
    auto &mbar_a0 = al.allocate<cuda_mbar, 1>();
    auto &mbar_a1 = al.allocate<cuda_mbar, 1>();
    auto &mbar_b0 = al.allocate<cuda_mbar, 1>();
    auto &mbar_b1 = al.allocate<cuda_mbar, 1>();
    cuda_mbar *mbar_a[2] = { &mbar_a0[0], &mbar_a1[0] };
    cuda_mbar *mbar_b[2] = { &mbar_b0[0], &mbar_b1[0] };

    float2 D[8] = {};

    // --- Preload K-chunk 0: TMA A + TMA B ---
    {
        if (tid == 0) {
            mbarrier_init(*mbar_a0, 0);
            mbarrier_init(*mbar_b0, 0);
        }
        __syncthreads();
        tma_load_ab<M_TILE, N_TILE, K_CHUNK>(
            B, A, row_base, col_base, 0,
            buf_a[0], buf_b[0], a_tma_desc, b_tma_desc,
            *mbar_a0, *mbar_b0);
    }

    // --- K dimension loop with TMA + WGMMA ---
    for (int kt = 0; kt < k; kt += K_CHUNK) {
        const int nxt = 1 - cur;
        const int next_kt = kt + K_CHUNK;

        const uint32_t a_addr = __cvta_generic_to_shared(buf_a[cur]);
        const uint32_t b_addr = __cvta_generic_to_shared(buf_b[cur]);
        const uint32_t a_stride = K_CHUNK * sizeof(half);
        const uint64_t b_desc = make_b_desc(b_addr, N_TILE * sizeof(half), 0);

        mma_m64n32k16_fp16_fp32(D, a_addr, a_stride, b_desc, 1, 0, 1);
        fence();
        commit_group();

        // Preload next K-chunk via TMA (overlaps with WGMMA compute)
        if (next_kt < k) {
            if (tid == 0) {
                mbarrier_init(*mbar_a[nxt], 0);
                mbarrier_init(*mbar_b[nxt], 0);
            }
            __syncthreads();
            tma_load_ab<M_TILE, N_TILE, K_CHUNK>(
                B, A, row_base, col_base, next_kt,
                buf_a[nxt], buf_b[nxt], a_tma_desc, b_tma_desc,
                *mbar_a[nxt], *mbar_b[nxt]);
        }

        wait_group(0);
        __syncthreads();
        if (next_kt < k) {
            cur = nxt;
        }
    }

    // --- Store accumulator D to global output C ---
    const int row = tid / 2;
    const int col_off = (tid % 2) * 16;
    if constexpr (is_same_v<T, float>) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int c = col_base + col_off + i * 2;
            if (row_base + row < m) {
                if (c < n)      C[size_t(row_base + row) * ldc + c]     = D[i].x;
                if (c + 1 < n)  C[size_t(row_base + row) * ldc + c + 1] = D[i].y;
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int c = col_base + col_off + i * 2;
            if (row_base + row < m) {
                if (c < n) {
                    C[size_t(row_base + row) * ldc + c] = __float2half(D[i].x);
                }
                if (c + 1 < n) {
                    C[size_t(row_base + row) * ldc + c + 1] = __float2half(D[i].y);
                }
            }
        }
    }
}

#define DEFINE_WGMMA_TMA_GEMM(name, T) \
    extern "C" __global__ void name( \
        const half *__restrict__ A, const half *__restrict__ B, \
        T *__restrict__ C, int32_t m, int32_t n, int32_t k, \
        int32_t lda, int32_t ldb, int32_t ldc, \
        int32_t stride_a, int32_t stride_b, int32_t stride_c, \
        const cuda_tensor_map *a_tma_desc, \
        const cuda_tensor_map *b_tma_desc) { \
        ggml_matmul_wgmma_tma_impl<T>(A, B, C, m, n, k, \
            lda, ldb, ldc, stride_a, stride_b, stride_c, \
            a_tma_desc, b_tma_desc); \
    }

DEFINE_WGMMA_TMA_GEMM(ggml_matmul_wgmma_tma_f16_f32, float)
DEFINE_WGMMA_TMA_GEMM(ggml_matmul_wgmma_tma_f16_f16, half)
#undef DEFINE_WGMMA_TMA_GEMM

#endif // __CUDA_ARCH__ >= 900
