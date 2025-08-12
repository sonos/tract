#include <cuda_fp16.h>
#include <cstdint>

// Check CC Version
#define CUDA_CC_TURING 750 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < CUDA_CC_TURING)
#error "Requires GPU with compute capability 7.5 or higher"
#endif

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

#define QK4_0 32
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define WARP_SIZE 32
#define N_WARPS 8

#define MMQ_Y 128

#define QK8_0 32
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QK8_1 32
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define MMQ_ITER_K 256
#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)

#define PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define MMQ_MMA_TILE_X_K_Q8_0 (2*WARP_SIZE + 2*WARP_SIZE/QI8_0 + 4)

typedef struct {
    half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");

namespace cuda_mma {

    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / WARP_SIZE;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && (J == 4 || J == 8)) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l / 2) * 8 + threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 2) % 2) * 8 + threadIdx.x / 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return 4 * l + threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return 2 * (threadIdx.x % 4) + l % 2;
            } else if constexpr (I == 16 && J == 16) {
                return 8 * (l / 4) + 2 * (threadIdx.x % 4) + l % 2;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2> {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return l * 8 + threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l % 2) * 8 + threadIdx.x / 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return l * 4 + threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l / 2) * 4 + threadIdx.x % 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
    };

    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    template <int I, int J, typename T>
    static __device__ __forceinline__ void load_generic(tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "l"(xs));
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
    }
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

using namespace cuda_mma;

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
    float * __restrict__ dst, const int stride, const int i_max, const int j_max);

static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*WARP_SIZE);

    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

_Pragma("unroll")
    for (int i0 = 0; i0 < mmq_y; i0 += N_WARPS) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

_Pragma("unroll")
    for (int i0 = 0; i0 < MMQ_Y; i0 += N_WARPS * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

        x_df[i*MMQ_MMA_TILE_X_K_Q8_0       + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_B::I*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*WARP_SIZE;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][WARP_SIZE/QI8_0];
    float dA[ntx][tile_C::ne/2][WARP_SIZE/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

_Pragma("unroll")
    for (int n = 0; n < ntx; ++n) {
_Pragma("unroll")
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
        }

_Pragma("unroll")
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

_Pragma("unroll")
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + k0/QI8_0];
            }
        }
    }

_Pragma("unroll")
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
_Pragma("unroll")
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

_Pragma("unroll")
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

_Pragma("unroll")
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

_Pragma("unroll")
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
    typedef tile<16, 8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              qk         = QK4_0;
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, MMQ_Y, N_WARPS, need_check>::load_tiles;

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + PAD(mmq_x*(WARP_SIZE + WARP_SIZE/QI8_1), N_WARPS*WARP_SIZE);

    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, MMQ_Y, N_WARPS, need_check>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<mmq_x, MMQ_Y, N_WARPS, need_check>;

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x*MMQ_Y / (N_WARPS*WARP_SIZE)] = {0.0f};

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        {
            const int * by0 = y + ncols_y*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 0*sizeof(block_q8_1_mmq)/sizeof(int));
_Pragma("unroll")
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += N_WARPS*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 1*sizeof(block_q8_1_mmq)/sizeof(int));
_Pragma("unroll")
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += N_WARPS*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, WARP_SIZE);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*MMQ_Y), MMQ_Y, MMQ_Y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <int mmq_x, int nwarps, bool need_check>
static __device__ __forceinline__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst) {
    constexpr int qk    = QK4_0;

    const int ntx = (ncols_dst + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (nrows_x   + MMQ_Y - 1) / MMQ_Y; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
_Pragma("unroll")
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
        const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

        if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const     int64_t blocks_per_ne00 = ncols_x / qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        int tmp = kbc;
        const int it = tmp / (nchannels_y*ntx*blocks_per_ne00);
        tmp -= it * (nchannels_y*ntx*blocks_per_ne00);
        const int zt = tmp / (ntx*blocks_per_ne00);
        tmp -= zt * (ntx*blocks_per_ne00);
        const int jt = tmp / blocks_per_ne00;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y   = zt*stride_channel_y;
        int offset_dst = zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;

            if (jt*mmq_x >= col_diff) {
                kbc += blocks_per_ne00;
                kbc -= kbc % blocks_per_ne00;

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);

                continue;
            }

            __syncthreads();
_Pragma("unroll")
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
                const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
            }
            __syncthreads();

        }

        offset_y   += (col_low + jt*mmq_x)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*MMQ_Y;

        const int tile_x_max_i = nrows_x  - it*MMQ_Y - 1;
        const int tile_y_max_j = col_diff - jt*mmq_x - 1;

        const int offset_x = (zt/channel_ratio)*stride_channel_x + it*MMQ_Y*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<mmq_x, nwarps, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = kbc;
    const int it = tmp / (nchannels_y*ntx*blocks_per_ne00);
    tmp -= it * (nchannels_y*ntx*blocks_per_ne00);
    const int zt = tmp / (ntx*blocks_per_ne00);
    tmp -= zt * (ntx*blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y   = zt*stride_channel_y;
    int offset_dst = zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;

        if (jt*mmq_x >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
_Pragma("unroll")
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
            const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

            if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y   += (col_low + jt*mmq_x)*(sizeof(block_q8_1_mmq)/sizeof(int));
    offset_dst += it*MMQ_Y;

    const int tile_x_max_i = nrows_x  - it*MMQ_Y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;

    const int offset_x = (zt/channel_ratio)*stride_channel_x + it*MMQ_Y*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<mmq_x, nwarps, need_check, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}


template <int mmq_x, int nwarps, bool need_check>
static __device__ __forceinline__ void mul_mat_q_stream_k_fixup(
        const int32_t * ids_dst, const int32_t * expert_bounds, float * __restrict__ dst, const float * __restrict__ tmp_last_tile,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_col_dst,
        const int nchannels_y, const int stride_channel_dst) {
    constexpr int     qk              = QK4_0;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;
    const     int64_t blocks_per_ne00 = ncols_x / qk;

    float sum[mmq_x*MMQ_Y / (nwarps*WARP_SIZE)] = {0.0f};

    const int ntx  = (ncols_dst + mmq_x - 1) / mmq_x;
    const int nty  = (nrows_x   + MMQ_Y - 1) / MMQ_Y;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc0      = (int64_t) bidx0     *nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
    int64_t kbc0_stop = (int64_t)(bidx0 + 1)*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;

    kbc0      -= (kbc0      % blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
    const bool did_not_write_last      = kbc0/blocks_per_ne00 == kbc0_stop/blocks_per_ne00 && kbc0_stop % blocks_per_ne00 != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int64_t bidx = bidx0 - 1;
    int64_t kbc_stop = kbc0;
    while(true) {
        int64_t kbc = bidx*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
        kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

_Pragma("unroll")
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

_Pragma("unroll")
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                sum[(j0/nwarps) * (MMQ_Y/WARP_SIZE) + i0/WARP_SIZE] += tmp_last_tile[bidx*(mmq_x*MMQ_Y) + j*MMQ_Y + i];
            }
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % blocks_per_ne00 == 0 || kbc/blocks_per_ne00 < kbc0/blocks_per_ne00) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = kbc0;
    const int it = tmp / (nchannels_y*ntx*blocks_per_ne00);
    tmp -= it * (nchannels_y*ntx*blocks_per_ne00);
    const int zt = tmp / (ntx*blocks_per_ne00);
    tmp -= zt * (ntx*blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    if (!ids_dst) {
        const int offset_dst = zt*stride_channel_dst + jt*mmq_x*stride_col_dst + it*MMQ_Y;
        dst += offset_dst;

        const int i_max = nrows_x   - it*MMQ_Y - 1;
        const int j_max = ncols_dst - jt*mmq_x - 1;

_Pragma("unroll")
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

_Pragma("unroll")
            for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j*stride_col_dst + i] += sum[(j0/nwarps) * (MMQ_Y/WARP_SIZE) + i0/WARP_SIZE];
            }
        }
        return;
    }

    __shared__ int ids_dst_shared[mmq_x];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y*WARP_SIZE + threadIdx.x; j < mmq_x; j += nwarps*WARP_SIZE) {
        ids_dst_shared[j] = ids_dst[col_low + j];
    }
    __syncthreads();

    const int offset_dst = it*MMQ_Y;
    dst += offset_dst;

    const int i_max = nrows_x  - it*MMQ_Y - 1;
    const int j_max = col_diff - jt*mmq_x - 1;

_Pragma("unroll")
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

_Pragma("unroll")
        for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst_shared[j]*stride_col_dst + i] += sum[(j0/nwarps) * (MMQ_Y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

#define INSTANTIATE_MMQ_KERNEL(mmq_x, nwarps, need_check) \
extern "C" { \
    __launch_bounds__(WARP_SIZE * nwarps, 1) \
    __global__ void mul_mat_q40##_##mmq_x##_##nwarps##_##need_check( \
    const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst, \
    const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup, \
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, \
    const int ncols_y, const int stride_col_dst, const int channel_ratio, const int nchannels_y, \
    const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst) {                       \
        mul_mat_q<mmq_x, nwarps, need_check> \
            (x, y, ids_dst, expert_bounds, dst, tmp_fixup, \
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
            channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst); \
    } \
\
    __global__ void mul_mat_q40_stream_k_fixup##_##mmq_x##_##nwarps##_##need_check( \
    const int32_t * ids_dst, const int32_t * expert_bounds, float * __restrict__ dst, const float * __restrict__ tmp_last_tile, \
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_col_dst, \
    const int nchannels_y, const int stride_channel_dst) { \
        mul_mat_q_stream_k_fixup<mmq_x, nwarps, need_check> \
            (ids_dst, expert_bounds, dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst, \
            nchannels_y, stride_channel_dst); \
    } \
} 

#define INSTANTIATE_MMQ_KERNEL_FOR_T(n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(8,   n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(16,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(24,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(32,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(40,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(48,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(64,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(80,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(96,  n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(112, n_warps, needs_check) \
    INSTANTIATE_MMQ_KERNEL(128, n_warps, needs_check) \


INSTANTIATE_MMQ_KERNEL_FOR_T(N_WARPS, true)
INSTANTIATE_MMQ_KERNEL_FOR_T(N_WARPS, false)

extern "C" __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = 32;
    constexpr int vals_per_sum   = 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    sum = xi.x + xi.y + xi.z + xi.w;

    // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
    for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;
    y[ib].ds4[iqs/32] = make_half2(d, sum);
}
