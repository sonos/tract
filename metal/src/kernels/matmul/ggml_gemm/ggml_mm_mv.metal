#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

typedef struct {
    int32_t batch;
    int32_t m;
    int32_t k;
    int32_t n;
    uint64_t a_strides[4];
    uint64_t b_strides[4];
    int32_t channel_broadcast_ratio;
    int32_t batch_broadcast_ratio;
} ggml_metal_kargs_mul;

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x;
    const int rb = tgpig.y*N_MV_T_T;
    const int im = tgpig.z;

    const uint i12 = im%args.batch;
    const uint i13 = im/args.batch;

    const uint64_t offset0 = r0*args.b_strides[2] + (i12/args.channel_broadcast_ratio)*args.b_strides[1] + (i13/args.batch_broadcast_ratio)*args.b_strides[0];

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.m*args.n;

    if (args.k < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.m) {
                break;
            }

            const uint64_t offset1 = r1*args.a_strides[2] + (i12   )*args.a_strides[1] + (i13   )*args.a_strides[0];

            device const T1 * y = (device const T1 *) (src1 + offset1);

            float sumf = 0;
            for (int i = tiisg; i < args.k; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst_f32[(uint64_t)r1*args.n + r0] = all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.m) {
                break;
            }

            const uint64_t offset1 = r1*args.a_strides[2] + (i12   )*args.a_strides[1] + (i13   )*args.a_strides[0];

            device const T1  * y  = (device const T1  *) (src1 + offset1);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < args.k/4; i += 32) {
                sumf += dot((float4) x4[i], (float4) y4[i]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(args.k/4); i < args.k; ++i) all_sum += (float) (x[i] * y[i]);
                dst_f32[(uint64_t)r1*args.n + r0] = all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        constant ggml_metal_kargs_mul & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14, constant ggml_metal_kargs_mul &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv<half, half4, half, half4>) mul_mv_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t kernel_mul_mv<float,  float4,  float,  float4>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   float,  float4>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   half,   half4>;

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        constant ggml_metal_kargs_mul & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.batch;
    const uint i13 = im/args.batch;

    const uint64_t offset0 = r0*args.b_strides[2] + (i12/args.channel_broadcast_ratio)*args.b_strides[1] + (i13/args.batch_broadcast_ratio)*args.b_strides[0];
    const uint64_t offset1 = r1*args.a_strides[2] + (i12        )*args.a_strides[1] + (i13        )*args.a_strides[0];

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + offset1);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.n*args.m + (uint64_t)r1*args.n;

    float sumf = 0;
    if (args.k < 128) {
        for (int i = tiisg; i < args.k; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < args.k/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(args.k/4); i < args.k; ++i) all_sum += (float) (x[i] * y[i]);
            dst_f32[r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;

// Assumes row size (k) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        constant ggml_metal_kargs_mul & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = args.m;
    const int r0 = tgpig.x;
    const int im = tgpig.z;

    const uint i12 = im%args.batch;
    const uint i13 = im/args.batch;

    const uint64_t offset0 = r0*args.b_strides[2] + (i12/args.channel_broadcast_ratio)*args.b_strides[1] + (i13/args.batch_broadcast_ratio)*args.b_strides[0];

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.n*args.m;

    for (int r1 = 0; r1 < nrows; ++r1) {
        const uint64_t offset1 = r1*args.a_strides[2] + (i12   )*args.a_strides[1] + (i13   )*args.a_strides[0];

        device const float4 * y4 = (device const float4 *) (src1 + offset1);

        float sumf = 0;
        for (int i = tiisg; i < args.k/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[(uint64_t)r1*args.n + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;
    const int im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    const short n_rows = (args.n - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.n - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (args.m - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (args.m - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const int i12 = im%args.batch;
    const int i13 = im/args.batch;

    const uint64_t offset0 = (i12/args.channel_broadcast_ratio)*args.b_strides[1] + (i13/args.batch_broadcast_ratio)*args.b_strides[0];
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.b_strides[2]*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;

    device const float   * y = (device const float   *)(src1
        + args.a_strides[0]*i13
        + args.a_strides[1]*i12
        + args.a_strides[2]*(r1*BLOCK_SIZE_N + thread_col)
        + args.a_strides[3]*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.k; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2)); // 2 -> (BLOCK_SIZE_M / SG_MAT_ROW) / THREAD_MAT_M
        threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2)); // 2 -> (BLOCK_SIZE_N / SG_MAT_ROW) / THREAD_MAT_N

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < THREAD_MAT_M; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(2)
            for (short i = 0; i < THREAD_MAT_N; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){ // THREAD_MAT_M * THREAD_MAT_N
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= args.n && (r1 + 1) * BLOCK_SIZE_N <= args.m) {
        device float * C = (device float *) dst +
            (BLOCK_SIZE_M * r0 + 32*(sgitg &  1)) + \
            (BLOCK_SIZE_N * r1 + 16*(sgitg >> 1)) * args.n + im*args.m*args.n;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * args.n * (i/4), args.n);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) \
                                     + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*args.n + im*args.m*args.n;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;