#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

#define QK4_0 32
typedef struct {
    half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct {
    half d;             // delta
    int8_t qs[QK8_0];   // quants
} block_q8_0;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int16_t  out_f16; // q4_0 path: when set, src1 and dst are f16 rather than f32
} ggml_metal_kargs_mul_mv;

typedef struct {
    int32_t  k;
    int32_t  n;
    int32_t  route_count;
    int32_t  input_mode; // 0: route_token_ids indexes input rows, 1: route row == input row
    uint64_t weight_expert_stride;
    uint64_t weight_row_stride;
    uint64_t input_row_stride;
    uint64_t output_route_stride;
} routed_q40_f32_args;

// Activation epilogue of the fused routed w1/w3 pair (see
// kernel_routed_q4_0_swiglu_f32): mode 0 is plain swiglu silu(g)*u, mode 1
// is the clamped variant (same math as clamped_swiglu_f32 in moe.metal).
typedef struct {
    int32_t act_mode;
    int32_t has_bias;
    float   alpha;
    float   limit;
} routed_swiglu_args;

inline float routed_swiglu_apply(routed_swiglu_args sargs, float g, float u) {
    if (sargs.act_mode == 1) {
        const float gate = min(g, sargs.limit);
        const float up = clamp(u, -sargs.limit, sargs.limit);
        const float glu = gate / (1.0f + exp(-sargs.alpha * gate));
        return (up + 1.0f) * glu;
    }
    return (g / (1.0f + exp(-g))) * u;
}

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14, typename args_t, typename TO = T1>
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

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    // Output dtype defaults to the activation dtype (T1): f32 activations
    // keep f32 outputs, f16 activations produce f16 outputs directly,
    // removing the f32->activation cast tract would otherwise insert after
    // the matmul. TO overrides it for mixed cases (f16 activations with
    // full-precision f32 outputs, e.g. MoE router scores).
    device TO * dst_o = (device TO *) dst + (uint64_t)im*args.ne0*args.ne1;

    if (args.ne00 < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.ne11) {
                break;
            }

            const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

            device const T1 * y = (device const T1 *) (src1 + offset1);

            float sumf = 0;
            for (int i = tiisg; i < args.ne00; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst_o[(uint64_t)r1*args.ne0 + r0] = (TO) all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.ne11) {
                break;
            }

            const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

            device const T1  * y  = (device const T1  *) (src1 + offset1);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < args.ne00/4; i += 32) {
                sumf += dot((float4) x4[i], (float4) y4[i]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(args.ne00/4); i < args.ne00; ++i) all_sum += (float) (x[i] * y[i]);
                dst_o[(uint64_t)r1*args.ne0 + r0] = (TO) all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14, typename TO = T1>
kernel void kernel_mul_mv(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &, TO>(
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
// Mixed variant: f32 weights, f16 activations, full-precision f32 output
// (bit-identical to upcasting the activations then running f32_f32: the
// half->float element conversion is exact and the accumulation order is the
// same). Used by the MoE router so the expert scores stay exact f32 while
// the block input skips its f16->f32 cast dispatch.
template [[host_name("kernel_mul_mv_f32_f16_of32")]] kernel mul_mv_t kernel_mul_mv<float, float4, half, half4, float>;

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + offset1);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    float sumf = 0;
    if (args.ne00 < 128) {
        for (int i = tiisg; i < args.ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < args.ne00/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(args.ne00/4); i < args.ne00; ++i) all_sum += (float) (x[i] * y[i]);
            dst_f32[r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;

// Assumes row size (ne00) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = args.ne11;
    const int r0 = tgpig.x;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    for (int r1 = 0; r1 < nrows; ++r1) {
        const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

        device const float4 * y4 = (device const float4 *) (src1 + offset1);

        float sumf = 0;
        for (int i = tiisg; i < args.ne00/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[(uint64_t)r1*args.ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
// Accumulate the q4_0 GEMV partials for one activation type. Templated on the
// activation type so each instantiation issues a direct typed load -- the f32
// instantiation matches the original f32-only kernel byte for byte. The caller
// selects the instantiation once via a uniform runtime branch, so this hot
// inner loop carries no per-element dtype test.
template<int nr, int nw, typename T_y>
inline void mul_vec_q_n_accumulate(
        thread device const block_q4_0 * const * ax,
        device const T_y * yb,
        int nb,
        short ix,
        short il,
        thread float * sumf) {
    float yl[16]; // src1 vector cache

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy[2] = { 0.f, 0.f };

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            // Accumulate activations in f32 (yl is f32) so f16 activations match
            // the f32 path's precision for the q4_0 zero-point (sumy) correction.
            sumy[0]  += (float) yb[i +  0] + (float) yb[i +  1];
            yl[i + 0] = (float) yb[i +  0];
            yl[i + 1] = (float) yb[i +  1]/256.f;

            sumy[1]  += (float) yb[i + 16] + (float) yb[i + 17];
            yl[i + 8] = (float) yb[i + 16]/16.f;
            yl[i + 9] = (float) yb[i + 17]/4096.f;
        }

#pragma unroll
        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }
}

template<typename block_q_type, int nr, int nsg, int nw, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = args.ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    const bool out_f16 = args.out_f16 != 0;

    // pointers to src0 rows
    device const block_q_type * ax[nr];
    for (int row = 0; row < nr; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[nr] = {0.f};

    const short ix = (tiisg/2);
    const short il = (tiisg%2)*8;

    // Branch once on the uniform activation dtype, then run a fully typed inner
    // loop (no per-element dtype test on the hot path).
    if (out_f16) {
        device const half  * yb = (device const half  *) (src1 + offset1) + ix*QK4_0 + il;
        mul_vec_q_n_accumulate<nr, nw>(ax, yb, nb, ix, il, sumf);
    } else {
        device const float * yb = (device const float *) (src1 + offset1) + ix*QK4_0 + il;
        mul_vec_q_n_accumulate<nr, nw>(ax, yb, nb, ix, il, sumf);
    }

    device char * dst_o = dst
        + (im*args.ne0*args.ne1 + r1*args.ne0) * (out_f16 ? sizeof(half) : sizeof(float));

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            if (out_f16) ((device half  *) dst_o)[first_row + row] = (half) tot;
            else         ((device float *) dst_o)[first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// q8_0 GEMV, same shape contract as kernel_mul_mv_q4_0: each simdgroup owns
// N_DST rows of src0 (q8_0 blocks along k), each thread covers 8 consecutive
// elements of a block (4 threads per block, blocks strided by 8).
template<int nr, typename T_y>
inline void mul_vec_q8_accumulate(
        thread device const block_q8_0 * const * ax,
        device const T_y * yb,
        int nb,
        short ix,
        short il,
        thread float * sumf) {
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        float yl[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            yl[i] = (float) yb[i];
        }
#pragma unroll
        for (int row = 0; row < nr; row++) {
            device const block_q8_0 * qb = ax[row] + ib;
            device const int8_t * qs = qb->qs + il;
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                acc += yl[i] * (float) qs[i];
            }
            sumf[row] += (float) qb->d * acc;
        }
        yb += QK8_0 * (N_SIMDWIDTH/4);
    }
}

kernel void kernel_mul_mv_q8_0(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    const bool out_f16 = args.out_f16 != 0;

    device const block_q8_0 * ax[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        const uint64_t offset0 =
            (first_row + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[N_DST] = {0.f};

    const short ix = tiisg/4;
    const short il = (tiisg%4)*8;

    if (out_f16) {
        device const half  * yb = (device const half  *) (src1 + offset1) + ix*QK8_0 + il;
        mul_vec_q8_accumulate<N_DST>(ax, yb, nb, ix, il, sumf);
    } else {
        device const float * yb = (device const float *) (src1 + offset1) + ix*QK8_0 + il;
        mul_vec_q8_accumulate<N_DST>(ax, yb, nb, ix, il, sumf);
    }

    device char * dst_o = dst
        + (im*args.ne0*args.ne1 + r1*args.ne0) * (out_f16 ? sizeof(half) : sizeof(float));

    for (int row = 0; row < N_DST; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            if (out_f16) ((device half  *) dst_o)[first_row + row] = (half) tot;
            else         ((device float *) dst_o)[first_row + row] = tot;
        }
    }
}

// ---- Expert-grouped routed Q4_0 matmul (prefill path) ----
//
// The per-route kernel above re-reads an expert's full weight slice for
// every route hitting it (a 512-token prefill chunk with top-4 routing reads
// each expert ~64x: ~9.5 GB of weight traffic per matmul). The grouped pair
// below first bins routes by expert (single-threadgroup counting sort), then
// processes GROUPED_ROUTES routes per threadgroup so each weight block is
// read once per 32 routes, with the x tile staged in threadgroup memory.

#define GROUPED_ROUTES 32
#define GROUPED_COLS 256
#define GROUPED_MAX_EXPERTS 256

// Also emits a compact work-chunk list (chunks of GROUPED_ROUTES routes,
// one expert each) so the grouped matmul launches exactly the threadgroups
// that have work: `chunks` rows of [expert, base, len], sentinel-terminated.
kernel void route_sort_by_expert(
        device const long * route_expert_ids,
        device uint * expert_offsets,   // [num_experts + 1]
        device uint * sorted_routes,    // [route_count], expert-grouped
        device uint * chunks,           // [3 * max_chunks]
        constant uint & route_count,
        constant uint & num_experts,
        constant uint & max_chunks,
        uint lane [[thread_position_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]])
{
    threadgroup atomic_uint hist[GROUPED_MAX_EXPERTS];
    for (uint e = lane; e < num_experts; e += tptg) {
        atomic_store_explicit(&hist[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = lane; r < route_count; r += tptg) {
        atomic_fetch_add_explicit(&hist[(uint)route_expert_ids[r]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint run = 0;
        uint chunk = 0;
        for (uint e = 0; e < num_experts; e++) {
            const uint c = atomic_load_explicit(&hist[e], memory_order_relaxed);
            expert_offsets[e] = run;
            atomic_store_explicit(&hist[e], run, memory_order_relaxed);
            for (uint b = 0; b < c && chunk < max_chunks; b += GROUPED_ROUTES, chunk++) {
                chunks[3 * chunk + 0] = e;
                chunks[3 * chunk + 1] = run + b;
                chunks[3 * chunk + 2] = min((uint)GROUPED_ROUTES, c - b);
            }
            run += c;
        }
        expert_offsets[num_experts] = run;
        for (; chunk < max_chunks; chunk++) {
            chunks[3 * chunk + 0] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = lane; r < route_count; r += tptg) {
        const uint slot =
            atomic_fetch_add_explicit(&hist[(uint)route_expert_ids[r]], 1u, memory_order_relaxed);
        sorted_routes[slot] = r;
    }
}

kernel void kernel_routed_q4_0_grouped_f32(
        constant routed_q40_f32_args & args,
        device const char * weights,
        device const float * input,
        device const long * route_token_ids,
        device const uint * chunks,
        device const uint * sorted_routes,
        device float * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],   // (colgroup64, chunk, 1)
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Lane = route, simdgroup = 8-column slice: each lane keeps its route's
    // 32 x values for the current k-block in registers and reuses them for
    // 8 output columns, so the hot loop runs ~8 FMAs per x load. Weight
    // blocks are read at the same address by all lanes (broadcast) and are
    // shared across the whole 32-route chunk.
    const uint e = chunks[3 * tgpig.y + 0];
    if (e == 0xffffffffu) {
        return;
    }
    const uint base = chunks[3 * tgpig.y + 1];
    const uint nr = chunks[3 * tgpig.y + 2];

    const uint route_slot = tiisg;              // 0..31
    const bool live = route_slot < nr;
    long xrow = 0;
    uint orig = 0;
    if (live) {
        orig = sorted_routes[base + route_slot];
        xrow = args.input_mode == 0 ? route_token_ids[orig] : (long)orig;
    }
    device const float * y = (device const float *)(
        (device const char *)input + (uint64_t)xrow * args.input_row_stride);

    const uint col0 = tgpig.x * 64 + sgitg * 8; // 8 sgs x 8 cols = 64 cols/tg
    device const block_q4_0 * w[8];
    bool colv[8];
#pragma unroll
    for (uint c = 0; c < 8; c++) {
        const uint col = col0 + c;
        colv[c] = col < (uint)args.n;
        w[c] = (device const block_q4_0 *)(
            weights + (uint64_t)e * args.weight_expert_stride
                    + (uint64_t)(colv[c] ? col : 0) * args.weight_row_stride);
    }

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int nb = args.k / QK4_0;
    for (int ib = 0; ib < nb; ib++) {
        // This route's 32 x values for the block, in registers.
        float xr[QK4_0];
        device const float4 * y4 = (device const float4 *)(y + ib * QK4_0);
#pragma unroll
        for (int i = 0; i < QK4_0 / 4; i++) {
            const float4 v = live ? y4[i] : float4(0.f);
            xr[4 * i + 0] = v.x;
            xr[4 * i + 1] = v.y;
            xr[4 * i + 2] = v.z;
            xr[4 * i + 3] = v.w;
        }
#pragma unroll
        for (uint c = 0; c < 8; c++) {
            const block_q4_0 blk = w[c][ib];
            const float d = (float)blk.d;
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const uint q = blk.qs[j];
                sum = fma((float)(q & 0xF) - 8.0f, xr[j],
                      fma((float)(q >> 4) - 8.0f, xr[j + 16], sum));
            }
            acc[c] = fma(d, sum, acc[c]);
        }
    }

    if (live) {
        device float * out = (device float *)(
            (device char *)dst + (uint64_t)orig * args.output_route_stride);
#pragma unroll
        for (uint c = 0; c < 8; c++) {
            if (colv[c]) {
                out[col0 + c] = acc[c];
            }
        }
    }
}

kernel void kernel_routed_q4_0_f32(
        constant routed_q40_f32_args & args,
        device const char * weights,
        device const float * input,
        device const long * route_token_ids,
        device const long * route_expert_ids,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nb = args.k/QK4_0;

    const int r0 = tgpig.x;
    const int route = tgpig.y;
    if (route >= args.route_count) {
        return;
    }

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const long expert = route_expert_ids[route];
    const long input_row = args.input_mode == 0 ? route_token_ids[route] : route;
    if (expert < 0 || input_row < 0) {
        return;
    }

    const uint64_t expert_offset = (uint64_t) expert * args.weight_expert_stride;
    device const float * y = (device const float *) ((device const char *) input + (uint64_t) input_row * args.input_row_stride);

    device const block_q4_0 * ax[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        const int weight_row = first_row + row;
        const uint64_t offset = expert_offset + (uint64_t) weight_row * args.weight_row_stride;
        ax[row] = (device const block_q4_0 *) (weights + offset);
    }

    float yl[16];
    float sumf[N_DST] = {0.f};

    const short ix = tiisg/2;
    const short il = (tiisg%2)*8;

    device const float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy[2] = { 0.f, 0.f };

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

#pragma unroll
        for (int row = 0; row < N_DST; ++row) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    device float * dst_route = (device float *) ((device char *) dst + (uint64_t) route * args.output_route_stride);
    for (int row = 0; row < N_DST; ++row) {
        const int weight_row = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && weight_row < args.n) {
            dst_route[weight_row] = tot;
        }
    }
}

// Fused routed w1/w3 gemv pair + swiglu epilogue: one dispatch computes
// g = w1 x (+bias1), u = w3 x (+bias3) and writes act(g, u). Same shape
// contract and route indexing as kernel_routed_q4_0_f32; both weight
// tensors share expert/row strides (same [experts, n, k] q4_0 layout).
// Reading both weight rows in the same k-block loop reuses the activation
// registers, so the extra cost over the single-weight kernel is only the
// second weight stream.
// TX is the activation dtype: every element converts to float before any
// arithmetic (exact for f16), so the f16-input instantiation is bit-identical
// to upcasting the activations and running the f32 one.
template<typename TX>
kernel void kernel_routed_q4_0_swiglu(
        constant routed_q40_f32_args & args,
        constant routed_swiglu_args & sargs,
        device const char * w1,
        device const char * w3,
        device const char * input,
        device const long * route_token_ids,
        device const long * route_expert_ids,
        device const float * bias1,
        device const float * bias3,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nb = args.k/QK4_0;

    const int r0 = tgpig.x;
    const int route = tgpig.y;
    if (route >= args.route_count) {
        return;
    }

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const long expert = route_expert_ids[route];
    const long input_row = args.input_mode == 0 ? route_token_ids[route] : route;
    if (expert < 0 || input_row < 0) {
        return;
    }

    const uint64_t expert_offset = (uint64_t) expert * args.weight_expert_stride;
    device const TX * y = (device const TX *) (input + (uint64_t) input_row * args.input_row_stride);

    device const block_q4_0 * ax1[N_DST];
    device const block_q4_0 * ax3[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        const int weight_row = first_row + row;
        const uint64_t offset = expert_offset + (uint64_t) weight_row * args.weight_row_stride;
        ax1[row] = (device const block_q4_0 *) (w1 + offset);
        ax3[row] = (device const block_q4_0 *) (w3 + offset);
    }

    float yl[16];
    float sumf1[N_DST] = {0.f};
    float sumf3[N_DST] = {0.f};

    const short ix = tiisg/2;
    const short il = (tiisg%2)*8;

    device const TX * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy[2] = { 0.f, 0.f };

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float y00 = (float) yb[i +  0];
            const float y01 = (float) yb[i +  1];
            sumy[0]  += y00 + y01;
            yl[i + 0] = y00;
            yl[i + 1] = y01/256.f;

            const float y16 = (float) yb[i + 16];
            const float y17 = (float) yb[i + 17];
            sumy[1]  += y16 + y17;
            yl[i + 8] = y16/16.f;
            yl[i + 9] = y17/4096.f;
        }

#pragma unroll
        for (int row = 0; row < N_DST; ++row) {
            sumf1[row] += block_q_n_dot_y(ax1[row] + ib, sumy[0] + sumy[1], yl, il);
            sumf3[row] += block_q_n_dot_y(ax3[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    device float * dst_route = (device float *) ((device char *) dst + (uint64_t) route * args.output_route_stride);
    for (int row = 0; row < N_DST; ++row) {
        const int weight_row = first_row + row;
        float g = simd_sum(sumf1[row]);
        float u = simd_sum(sumf3[row]);
        if (tiisg == 0 && weight_row < args.n) {
            if (sargs.has_bias != 0) {
                g += bias1[(uint64_t) expert * (uint)args.n + weight_row];
                u += bias3[(uint64_t) expert * (uint)args.n + weight_row];
            }
            dst_route[weight_row] = routed_swiglu_apply(sargs, g, u);
        }
    }
}

typedef decltype(kernel_routed_q4_0_swiglu<float>) routed_q4_0_swiglu_t;

template [[host_name("kernel_routed_q4_0_swiglu_f32")]]      kernel routed_q4_0_swiglu_t kernel_routed_q4_0_swiglu<float>;
template [[host_name("kernel_routed_q4_0_swiglu_f16x_f32")]] kernel routed_q4_0_swiglu_t kernel_routed_q4_0_swiglu<half>;

// Grouped-path epilogue of the fused routed swiglu: reads the two
// expert-sorted mm results, applies bias + activation, scatters back to
// original route order (one dispatch replacing bias adds, activation and
// two scatters).
kernel void routed_swiglu_scatter_f32(
        constant routed_q40_f32_args & args,
        constant routed_swiglu_args & sargs,
        device const float * c1_sorted,
        device const float * c3_sorted,
        device const uint * sorted_routes,
        device const long * route_expert_ids,
        device const float * bias1,
        device const float * bias3,
        device float * dst,
        uint gid [[thread_position_in_grid]])
{
    const uint n4 = ((uint)args.n + 3) / 4;
    const uint total = (uint)args.route_count * n4;
    if (gid >= total) {
        return;
    }
    const uint i = gid / n4;
    const uint col = (gid - i * n4) * 4;
    const uint orig = sorted_routes[i];
    const uint end = min(col + 4, (uint)args.n);
    device const float * c1 = c1_sorted + (uint64_t)i * (uint)args.n;
    device const float * c3 = c3_sorted + (uint64_t)i * (uint)args.n;
    const long e = sargs.has_bias != 0 ? route_expert_ids[orig] : 0;
    device float * out = (device float *)(
        (device char *)dst + (uint64_t)orig * args.output_route_stride);
    for (uint c = col; c < end; c++) {
        float g = c1[c];
        float u = c3[c];
        if (sargs.has_bias != 0) {
            g += bias1[(uint64_t) e * (uint)args.n + c];
            u += bias3[(uint64_t) e * (uint)args.n + c];
        }
        out[c] = routed_swiglu_apply(sargs, g, u);
    }
}

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
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &), typename T_y>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
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
    const short n_rows = (args.ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (args.ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (args.ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

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

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.nb01*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;

    device const T_y     * y = (device const T_y     *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1*BLOCK_SIZE_N + thread_col)
        + args.nb10*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.ne00; loop_k += BLOCK_SIZE_K) {
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

        // Activations are kept in f32 shared memory for the simdgroup matmul;
        // convert on load so f16 activations need no separate upcast pass.
        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = float2x4(*((device matrix<T_y, 2, 4> *) y));

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    // `simdgroup_store` can only target a buffer of the accumulator's element
    // type (float). For f16 output we route the tile through f32 threadgroup
    // memory and convert on the device write, so the matmul output lands as f16
    // directly (no f32->f16 cast pass after the matmul).
    if (sizeof(T_y) != sizeof(float)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_all = (threadgroup float *) shmem;
        threadgroup float * temp_str = temp_all + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All threads cooperatively convert+write the n_rows x n_cols tile.
        device T_y * D = (device T_y *) dst
            + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N)*args.ne0 + im*args.ne1*args.ne0;
        for (int idx = tiitg; idx < n_rows * n_cols; idx += THREAD_PER_BLOCK) {
            const int col = idx / n_rows;
            const int row = idx % n_rows;
            D[col*args.ne0 + row] = (T_y) temp_all[col*BLOCK_SIZE_M + row];
        }
    } else if ((r0 + 1) * BLOCK_SIZE_M <= args.ne0 && (r1 + 1) * BLOCK_SIZE_N <= args.ne1) {
        device float * C = (device float *) dst +
            (BLOCK_SIZE_M * r0 + 32*(sgitg &  1)) + \
            (BLOCK_SIZE_N * r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * args.ne0 * (i/4), args.ne0);
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
                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*args.ne0 + im*args.ne1*args.ne0;
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

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,  float>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,  float>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0, float>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,  half>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0, half>;

// Gather activation rows into expert-sorted order, always staging as f32.
// One thread moves 4 consecutive elements (vectorized load/store, 4x fewer
// index divisions than the old per-element scheme, which ran ~6x off memory
// roofline). TX is the source dtype: the f16 instantiation converts each
// element exactly, so downstream mm consumers see the same f32 rows as if
// the activations had been upcast beforehand.
template<typename TX, typename TX4>
kernel void routed_gather_rows(
        constant routed_q40_f32_args & args,
        device const char * src,
        device float * dst,
        device const uint * sorted_routes,
        device const long * route_token_ids,
        uint gid [[thread_position_in_grid]])
{
    const uint k4 = ((uint)args.k + 3) / 4;
    const uint total = (uint)args.route_count * k4;
    if (gid >= total) {
        return;
    }
    const uint i = gid / k4;
    const uint kk = (gid - i * k4) * 4;
    const uint orig = sorted_routes[i];
    const long xrow = args.input_mode == 0 ? route_token_ids[orig] : (long)orig;
    device const TX * y = (device const TX *)(
        src + (uint64_t)xrow * args.input_row_stride);
    device float * d = dst + (uint64_t)i * (uint)args.k;
    if (kk + 4 <= (uint)args.k) {
        *(device packed_float4 *)(d + kk) = (packed_float4)(*(device const TX4 *)(y + kk));
    } else {
        for (uint c = kk; c < (uint)args.k; c++) {
            d[c] = (float) y[c];
        }
    }
}

typedef decltype(routed_gather_rows<float, packed_float4>) routed_gather_rows_t;

template [[host_name("routed_gather_rows_f32")]]  kernel routed_gather_rows_t routed_gather_rows<float, packed_float4>;
template [[host_name("routed_gather_rows_f16x")]] kernel routed_gather_rows_t routed_gather_rows<half, packed_half4>;

// Scatter expert-sorted result rows back to original route order (f32).
// Vectorized like the gather.
kernel void routed_scatter_rows_f32(
        constant routed_q40_f32_args & args,
        device const float * src,
        device float * dst,
        device const uint * sorted_routes,
        uint gid [[thread_position_in_grid]])
{
    const uint n4 = ((uint)args.n + 3) / 4;
    const uint total = (uint)args.route_count * n4;
    if (gid >= total) {
        return;
    }
    const uint i = gid / n4;
    const uint col = (gid - i * n4) * 4;
    const uint orig = sorted_routes[i];
    device const float * s = src + (uint64_t)i * (uint)args.n;
    device float * out = (device float *)(
        (device char *)dst + (uint64_t)orig * args.output_route_stride);
    if (col + 4 <= (uint)args.n) {
        *(device packed_float4 *)(out + col) = *(device const packed_float4 *)(s + col);
    } else {
        for (uint c = col; c < (uint)args.n; c++) {
            out[c] = s[c];
        }
    }
}

// Expert-sorted tiled matmul over the routed experts: the simdgroup-matrix
// mm (kernel_mul_mm) with the activation-row tile driven by the sort
// kernel's chunk list ([expert, base, len] per 32-route chunk). A is the
// gathered f32 activations [route_count, k]; C is written expert-sorted
// [route_count, n] and scattered afterwards. This is what fixes prefill:
// the scalar per-route gemv runs at ~1.7 TFLOPS while this path uses the
// same simdgroup-matrix pipeline as the dense q40 matmuls.
kernel void kernel_mul_mm_q4_0_routed_f32(
        constant routed_q40_f32_args & args,
        device const char * weights,
        device const float * a_sorted,
        device const uint * chunks,
        device float * c_sorted,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],   // (chunk, n_tile, 1)
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const uint expert = chunks[3 * tgpig.x + 0];
    if (expert == 0xffffffffu) {
        return;
    }
    const uint base = chunks[3 * tgpig.x + 1];
    const uint len = chunks[3 * tgpig.x + 2];

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;

    const short n_rows = ((uint)args.n - r0*BLOCK_SIZE_M < BLOCK_SIZE_M)
        ? (args.n - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (short)min((uint)BLOCK_SIZE_N, len);

    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    // Each simdgroup owns a 16-wide slice of the N (route) axis. Ragged
    // expert chunks (len <= 16, the common case: ~tokens*top_k/n_experts
    // routes per expert per prompt chunk) leave the upper slice entirely
    // garbage: its columns are never stored. Skip its matmul work.
    const bool n_slice_active = (short)(16*(sgitg >> 1)) < n_cols;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const short nl = 2;
    short il = (tiitg % THREAD_PER_ROW);

    device const block_q4_0 * x = (device const block_q4_0 *)(weights
        + (uint64_t)expert * args.weight_expert_stride
        + args.weight_row_stride*(r0*BLOCK_SIZE_M + thread_row)) + (il/nl);

    device const float * y = a_sorted
        + (uint64_t)(base + thread_col) * (uint)args.k
        + (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL));

    for (int loop_k = 0; loop_k < args.k; loop_k += BLOCK_SIZE_K) {
        half4x4 temp_a;
        dequantize_q4_0(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = *((device const float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (n_slice_active) {
            threadgroup const half  * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
            threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

            #pragma unroll(4)
            for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
                #pragma unroll(4)
                for (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
                }

                simdgroup_barrier(mem_flags::mem_none);

                #pragma unroll(2)
                for (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
                }

                #pragma unroll(8)
                for (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
                lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
            }
        }
    }

    // C rows are the expert-sorted routes: row (activation) index = base+j,
    // col = weight row. Always go through threadgroup memory (the ragged
    // len < 32 case is common: every expert's tail chunk).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_all = (threadgroup float *) shmem;
    threadgroup float * temp_str = temp_all \
                                 + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
    if (n_slice_active) {
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads cooperatively write the n_rows x n_cols tile (the old
    // single-simdgroup scalar loop serialized the store and idled 3/4 of
    // the threadgroup on every tile).
    for (int idx = tiitg; idx < n_rows * n_cols; idx += THREAD_PER_BLOCK) {
        const int j = idx / n_rows;
        const int i = idx % n_rows;
        c_sorted[(uint64_t)(base + j) * (uint)args.n + r0*BLOCK_SIZE_M + i]
            = temp_all[j*BLOCK_SIZE_M + i];
    }
}

