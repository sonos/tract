#include "common.cuh"

// Check CC Version
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < CUDA_CC_TURING)
#error "Requires GPU with compute capability 7.5 or higher"
#endif // __CUDA_ARCH__ >= CUDA_CC_TURING

#if __CUDA_ARCH__ >= CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // __CUDA_ARCH__ >= CUDA_CC_AMPERE

#define FATTN_KQ_STRIDE       256
#define SOFTMAX_FTZ_THRESHOLD -20.0f

#define CUDA_UNUSED(x) (void)(x)

template<typename... Args>
__host__ __device__ constexpr inline void cuda_unused_vars_impl(Args&&...) noexcept {}
#define CUDA_UNUSED_VARS(...) cuda_unused_vars_impl(__VA_ARGS__)

enum cuda_type {
    CUDA_TYPE_F32     = 0,
    CUDA_TYPE_F16     = 1,
    CUDA_TYPE_Q4_0    = 2,
};

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);
typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (cuda::std::is_same_v<T, half>) {
        cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (cuda::std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        half2 tmp[ne/2];
        cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(cuda::std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

    if constexpr (cuda::std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else

    if constexpr (cuda::std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(cuda::std::is_same_v<T, void>, "bad type");
    }
}

static __device__ __forceinline__ void cuda_mad(float & acc, const half2 v, const half2 u) {
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
}

//static __device__ __forceinline__ void cuda_mad(half2 & acc, const half2 v, const half2 u) {
//    acc += v*u;
//}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    CUDA_UNUSED(Q_q8);
    CUDA_UNUSED(Q_ds_v);

    constexpr int cpy_ne = 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        half2 tmp[cpy_ne];
        cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            cuda_mad(sum, tmp[k_KQ_1], ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    CUDA_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = __dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template <cuda_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == CUDA_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == CUDA_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <cuda_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == CUDA_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == CUDA_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (cuda::std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

template<int D, int ncols, cuda_type type_K, cuda_type type_V> // D == head size
static __device__ void flash_attn_ext_vec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int32_t nb13,
                            const int32_t nb21, const int32_t nb22, const int32_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int32_t nb33) {
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.
    constexpr int cpy_nb = 16;
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);

    constexpr int nthreads    = 128;
    constexpr int nthreads_KQ = type_K == CUDA_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = type_V == CUDA_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = type_V == CUDA_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != CUDA_TYPE_F16;

    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;

    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.

    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= D/sizeof(int) || i < D/sizeof(int)) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < ne01) {
                    cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (mask) {
                    sum += __half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == i_KQ_0) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
        }

        __syncwarp();

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= ne01) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*ne02 + ic0 + j_VKQ)*ne01 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
}

template<int D> // D == head size
static __device__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

#define INSTANTIATE_FLASH_ATTN_VEC_FOR_TYPES(D, K_type, K_type_name, V_type, V_type_name) \
    extern "C" {                                                                                  \
        __launch_bounds__(128, 1)                                                                 \
        __global__ void flash_attn_vec_##D##_1##_##K_type_name##_##V_type_name(            \
            const char * __restrict__ Q, \
            const char * __restrict__ K, \
            const char * __restrict__ V, \
            const char * __restrict__ mask, \
            const int  * __restrict__ KV_max, \
            float      * __restrict__ dst, \
            float2     * __restrict__ dst_meta, \
            const float scale, \
            const int32_t ne03, const int32_t ne02, const int32_t ne01, const int32_t ne00,  \
                                const int32_t nb03, const int32_t nb02, const int32_t nb01,  \
            const int32_t ne13, const int32_t ne12, const int32_t ne11, const int32_t ne10,  \
                                const int32_t nb13, const int32_t nb12, const int32_t nb11,  \
                                const int32_t nb23, const int32_t nb22, const int32_t nb21,  \
                                const int32_t ne33, const int32_t ne32, const int32_t ne31,  \
                                const int32_t nb33, const int32_t nb32, const int32_t nb31){ \
            flash_attn_ext_vec<D, 1, K_type, V_type>(                                   \
                            Q, K, V, mask, KV_max, dst, dst_meta, scale,                     \
                            ne00, ne01, ne02, ne03,                                          \
                            nb01, nb02, nb03,                                                \
                            ne10, ne11, ne12, ne13,                                          \
                            nb11, nb12, nb13,                                                \
                            nb21, nb22, nb23,                                                \
                            ne31, ne32, ne33,                                                \
                            nb31, nb32, nb33);                                               \
        }                                                                                    \
                                                                                             \
        __launch_bounds__(128, 1)                                                            \
        __global__ void flash_attn_vec_##D##_2##_##K_type_name##_##V_type_name(            \
            const char * __restrict__ Q, \
            const char * __restrict__ K, \
            const char * __restrict__ V, \
            const char * __restrict__ mask, \
            const int  * __restrict__ KV_max, \
            float      * __restrict__ dst, \
            float2     * __restrict__ dst_meta, \
            const float scale, \
            const int32_t ne03, const int32_t ne02, const int32_t ne01, const int32_t ne00,  \
                                const int32_t nb03, const int32_t nb02, const int32_t nb01,  \
            const int32_t ne13, const int32_t ne12, const int32_t ne11, const int32_t ne10,  \
                                const int32_t nb13, const int32_t nb12, const int32_t nb11,  \
                                const int32_t nb23, const int32_t nb22, const int32_t nb21,  \
                                const int32_t ne33, const int32_t ne32, const int32_t ne31,  \
                                const int32_t nb33, const int32_t nb32, const int32_t nb31){ \
            flash_attn_ext_vec<D, 2, K_type, V_type>(                                        \
                            Q, K, V, mask, KV_max, dst, dst_meta, scale,                     \
                            ne00, ne01, ne02, ne03,                                          \
                            nb01, nb02, nb03,                                                \
                            ne10, ne11, ne12, ne13,                                          \
                            nb11, nb12, nb13,                                                \
                            nb21, nb22, nb23,                                                \
                            ne31, ne32, ne33,                                                \
                            nb31, nb32, nb33);                                               \
        }                                                                                    \
                                                                                             \
        __launch_bounds__(D, 1)                                                              \
        __global__ void flash_attn_combine_results_##D(                                      \
            const float  * __restrict__ VKQ_parts, const float2 * __restrict__ VKQ_meta,     \
            float * __restrict__ dst, const int parallel_blocks){                            \
            flash_attn_combine_results<D>(                                                   \
                VKQ_parts, VKQ_meta, dst, parallel_blocks);                                  \
            }                                                                                \
    }

#define INSTANTIATE_FLASH_ATTN_VEC_FOR_D_NCOLS1(D) \
    INSTANTIATE_FLASH_ATTN_VEC_FOR_TYPES(D, CUDA_TYPE_F16, f16, CUDA_TYPE_F16, f16) \
    //INSTANTIATE_FLASH_ATTN_VEC_FOR_TYPES(D, CUDA_TYPE_Q4_0, q40, CUDA_TYPE_F16, f16) \
    //INSTANTIATE_FLASH_ATTN_VEC_FOR_TYPES(D, CUDA_TYPE_F16, f16, CUDA_TYPE_Q4_0, q40) \
    //INSTANTIATE_FLASH_ATTN_VEC_FOR_TYPES(D, CUDA_TYPE_Q4_0, q40, CUDA_TYPE_Q4_0, q40) \


#define INSTANTIATE_FLASH_ATTN_VEC() \
    INSTANTIATE_FLASH_ATTN_VEC_FOR_D_NCOLS1(64) \
    INSTANTIATE_FLASH_ATTN_VEC_FOR_D_NCOLS1(128) \
    //INSTANTIATE_FLASH_ATTN_VEC_FOR_D_NCOLS1(256) \

INSTANTIATE_FLASH_ATTN_VEC()

/*--------------------------------------------------------------------------------------------------------------------------*/
using namespace cuda_mma;

static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
    tile<8, 8, half2> ret;
    ret.x[0] = cuda_movmatrix(t.x[0]);
    ret.x[1] = cuda_movmatrix(t.x[1]);

    return ret;
}

typedef tile<16,  8, half2> tile_A;
typedef tile< 8,  8, half2> tile_B;
typedef tile<16,  8, half2> tile_B_16;
typedef tile<16,  8, float> tile_C_KQ;
typedef tile<16, 16, float> tile_C_KQ_16;
typedef tile<16,  4, half2> tile_C_VKQ;
typedef tile<16,  8, half2> tile_C_VKQ_16;

// Config options for specific head sizes.
// Should not affect results, only speed/register pressure/shared memory use.
//
// nbatch_fa:      number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
// nwarps_max:     maximum number of warps per CUDA block, up to 8 warps in total can run per SM (given enough shared memory).
// Q_in_reg:       whether the Q values should be kept permanently in registers.
// nstages_target: targeted number of pipeline stages for cp_async (if available), 0 means synchronous data loading.
// nbatch_K2:      number of K half2 values in direction of D to load in parallel.
// nbatch_V2:      number of V half2 values in direction of D to load in parallel.
// nbatch_combine: number of VKQ half2 values in direction of D to combine in parallel.

template <int D>
struct fattn_mma_f16_config;

template <>
struct fattn_mma_f16_config<64> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;
    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 32;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 32;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 32;
    }
};

template <>
struct fattn_mma_f16_config<80> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;


    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 40;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 40;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 40;
    }
};

template <>
struct fattn_mma_f16_config<96> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static __device__ constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 48;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 48;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 48;
    }
};

template <>
struct fattn_mma_f16_config<112> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 56;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 56;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 56;
    }
};

template <>
struct fattn_mma_f16_config<128> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 64;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 64;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 64;
    }
};

template <>
struct fattn_mma_f16_config<256> {
    static constexpr int  nbatch_fa      = 32;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 128;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 128;
    }

    static constexpr __device__ int get_nbatch_combine_device(int ncols) {
#if __CUDA_ARCH__ == cuda_CC_TURING
        return ncols <= 16 ? 128 : 64;
#else
        CUDA_UNUSED(ncols);
        return 128;
#endif // __CUDA_ARCH__ == CUDA_CC_TURING
    }
};

static __device__ __forceinline__ unsigned int cuda_cvta_generic_to_shared(void * generic_ptr) {
    return __cvta_generic_to_shared(generic_ptr);
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}


template<int stride_tile, int nwarps, int nbatch_fa, int N>
static __device__ __forceinline__
void load_cp_async_once(const half2* __restrict__ KV,
                        unsigned int tile_KV_32,
                        int chunks_per_row, int stride_KV) {
    constexpr int preload       = 64;
    constexpr int h2_per_chunk  = 16 / sizeof(half2);
    constexpr int stride_k      = WARP_SIZE >> N;
    constexpr int stride_i      = WARP_SIZE / stride_k;

    const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
    const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
    if (k0_start == k0_stop) return;

    #pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);
        if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) break;

        #pragma unroll
        for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
            const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
            cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16,
                                    KV + i*stride_KV + k*h2_per_chunk);
        }
    }
}

template<int stride_tile, int nwarps, int nbatch_fa, int N, int MAX>
static __device__ __forceinline__
void unroll_cp_async(const half2* __restrict__ KV, unsigned int tile_KV_32,
                     int chunks_per_row, int stride_KV) {
    if constexpr (N < MAX) {
        load_cp_async_once<stride_tile, nwarps, nbatch_fa, N>(KV, tile_KV_32, chunks_per_row, stride_KV);
        unroll_cp_async<stride_tile, nwarps, nbatch_fa, N+1, MAX>(KV, tile_KV_32, chunks_per_row, stride_KV);
    }
}

template<int stride_tile, int nwarps, int nbatch_fa, int N>
static __device__ __forceinline__
void load_sync_once(const half2* __restrict__ KV, half2* __restrict__ tile_KV,
                    int D2, int stride_KV) {
    constexpr int stride_k = WARP_SIZE >> N;
    constexpr int stride_i = WARP_SIZE / stride_k;

    const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
    const int k0_stop  =                             D2 - D2 % (1*stride_k);
    if (k0_start == k0_stop) return;

    #pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);
        if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) break;

        #pragma unroll
        for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
            const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
            tile_KV[i*stride_tile + k] = KV[i*stride_KV + k];
        }
    }
}

template<int stride_tile, int nwarps, int nbatch_fa, int N, int MAX>
static __device__ __forceinline__
void unroll_sync(const half2* __restrict__ KV, half2* __restrict__ tile_KV,
                 int D2, int stride_KV) {
    if constexpr (N < MAX) {
        load_sync_once<stride_tile, nwarps, nbatch_fa, N>(KV, tile_KV, D2, stride_KV);
        unroll_sync<stride_tile, nwarps, nbatch_fa, N+1, MAX>(KV, tile_KV, D2, stride_KV);
    }
}

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async>
static __device__ __forceinline__
void flash_attn_ext_f16_load_tile(const half2* __restrict__ KV,
                                  half2* __restrict__ tile_KV,
                                  int D2, int stride_KV) {
    if constexpr (use_cp_async) {
        const int  chunks_per_row  = D2 / (16 / sizeof(half2));
        const unsigned tile_KV_32  = cuda_cvta_generic_to_shared(tile_KV);
        unroll_cp_async<stride_tile, nwarps, nbatch_fa, 0, 5>(KV, tile_KV_32, chunks_per_row, stride_KV);
    } else {
        static_assert(nbatch_fa % (4*nwarps) == 0, "out of bounds");
        unroll_sync<stride_tile, nwarps, nbatch_fa, 0, 3>(KV, tile_KV, D2, stride_KV);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half2 * const __restrict__ mask_h2, half2 * const __restrict__ tile_mask, const int stride_mask) {
    static_assert(nbatch_fa == 2*WARP_SIZE || WARP_SIZE % nbatch_fa == 0, "bad KQ_per_iter");

    if (use_cp_async) {
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
            const int j = j0 + threadIdx.y*cols_per_warp +
                (nbatch_fa == 2*WARP_SIZE ? threadIdx.x / (WARP_SIZE/4) : threadIdx.x / (WARP_SIZE/cols_per_warp));

            if (j0 + stride_j > ncols1 && j >= ncols1) {
                break;
            }

            const int i = 4 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half2), mask_h2 + j*stride_mask + i);
        }
        return;
    }

    constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
    constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp + (nbatch_fa == 2*WARP_SIZE ? 0 : threadIdx.x / (WARP_SIZE/cols_per_warp));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = nbatch_fa == 2*WARP_SIZE ? threadIdx.x : threadIdx.x % (WARP_SIZE/cols_per_warp);

        tile_mask[j*(nbatch_fa/2 + 4) + i] = mask_h2[j*stride_mask + i];
    }
}

template<int D, int ncols1, int ncols2, int nwarps, int ntiles,
    bool needs_fixup, bool is_fixup, bool last_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const int ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half2        * const __restrict__ tile_mask,
        const tile_B * const __restrict__ Q_B,
        tile_C_VKQ   * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0) {
    typedef fattn_mma_f16_config<D> c;

#ifdef CP_ASYNC_AVAILABLE
    constexpr int nstages = c::nstages_target;
#else
    constexpr int nstages = 0;
#endif // CP_ASYNC_AVAILABLE

    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int ncols           = ncols1 * ncols2;
    constexpr int nbatch_K2       = c::get_nbatch_K2_device(ncols);
    constexpr int nbatch_V2       = c::get_nbatch_V2_device(ncols);

    constexpr int stride_tile_Q = D/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    constexpr int stride_tile_V = nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * c::nbatch_fa;
    tile_C_KQ KQ_C[c::nbatch_fa/(np*tile_C_KQ::I) * ntiles];

    // Use wide variants of tiles if ntiles >= 2.
    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;
    tile_C_KQ_16  * KQ_C_16  = (tile_C_KQ_16  *) KQ_C;

    if constexpr (nstages > 1) {
        static_assert(nbatch_K2 == D/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, c::nbatch_fa, use_cp_async>
            (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V);
    } else {
        constexpr bool use_cp_async = nstages == 1;
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>(mask_h2 + k_VKQ_0/2, tile_mask, stride_mask);
        }
    }

#pragma unroll
    for (int k0_start = 0; k0_start < D/2; k0_start += nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < D/2 ? k0_start + nbatch_K2 : D/2;
        const int k0_diff = k0_stop - k0_start;

        if (nstages <= 1) {
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
                (K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, k0_diff, stride_K);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (c::Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < c::nbatch_fa; i_KQ_00 += np*tile_A::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += tile_A::J) {
                    tile_A K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                    if (ntiles == 1) {
                        mma(KQ_C[i_KQ_00/(np*tile_A::I)], K_A, Q_B[k_KQ_0/tile_A::J]);
                    } else {
#pragma unroll
                        for (int t = 0; t < ntiles/2; ++t) {
                            // Wide version of KQ_C is column-major => swap A and B.
                            mma(KQ_C_16[i_KQ_00/(np*tile_A::I) * ntiles/2 + t], Q_B_16[k_KQ_0/tile_A::J * ntiles/2 + t], K_A);
                        }
                    }
                }
            }
        } else {
            static_assert(ntiles == 2, "ntiles != 2 not implemented");
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += tile_A::J) {
                load_ldmatrix(Q_B_16[0], tile_Q + (threadIdx.y / np)*(tile_B_16::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < c::nbatch_fa; i_KQ_00 += np*tile_A::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;

                    tile_A K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);

                    // Wide version of KQ_C is column-major => swap A and B.
                    mma(KQ_C_16[i_KQ_00/(np*tile_A::I)], Q_B_16[0], K_A);
                }
            }
        }

        if (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if (ntiles == 1) {
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < c::nbatch_fa; i00 += np*tile_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ::I;
#pragma unroll
                for (int l = 0; l < tile_C_KQ::ne; ++l) {
                    const int i = i0 + tile_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*tile_C_KQ::J + tile_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*tile_C_KQ::I)].x[l] +=
                        __half2float(((const half *) tile_mask)[j*(c::nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k].x[l]);
            }
        }

        // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[k].x[l] = expf(KQ_C[k].x[l] - KQ_max_new[l % 2]);

                KQ_rowsum_add[l % 2] += KQ_C[k].x[l];
            }
        }
    } else { // ntiles > 1
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < c::nbatch_fa; i00 += np*tile_C_KQ_16::J) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ_16::J;
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_KQ_16::ne; l0 += 2) {
                        const int i = (i0 + tile_C_KQ_16::get_j(l0)) / 2;
                        const int j = ((threadIdx.y / np)*cols_per_warp + t*tile_C_KQ_16::I + tile_C_KQ_16::get_i(l0)) / ncols2;

                        const float2 tmp = __half22float2(tile_mask[j*(c::nbatch_fa/2 + 4) + i]);
                        const int KQ_index = i00/(np*tile_C_KQ_16::J) * ntiles/2 + t;
                        KQ_C_16[KQ_index].x[l0 + 0] += tmp.x;
                        KQ_C_16[KQ_index].x[l0 + 1] += tmp.y;
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;
                    KQ_max_new[KQ_index] = fmaxf(KQ_max_new[KQ_index], KQ_C_16[k*ntiles/2 + t].x[l]);
                }
            }
        }

        // Values per KQ column are spread across 4 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(c::nbatch_fa % (np*tile_C_KQ_16::J) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;

                    KQ_C_16[k*ntiles/2 + t].x[l] = expf(KQ_C_16[k*ntiles/2 + t].x[l] - KQ_max_new[KQ_index]);

                    KQ_rowsum_add[KQ_index] += KQ_C_16[k*ntiles/2 + t].x[l];
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < D/tile_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < D/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    tile_B B[c::nbatch_fa/(np*2*tile_B::J) * ntiles];
    tile_B_16 * B_16 = (tile_B_16 *) B;
    static_assert(c::nbatch_fa % (np*2*tile_B::J) == 0, "bad loop size");
    if (ntiles == 1) {
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*2*tile_B::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < c::nbatch_fa/(np*2*tile_B_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                B_16[k*ntiles/2 + t] = get_half2(KQ_C_16[k*ntiles/2 + t]);
            }
        }
    }

    if (nstages > 1) {
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h2) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>
                    (mask_h2 + (k_VKQ_0 + c::nbatch_fa)/2, tile_mask, stride_mask);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
                (K_h2 + int64_t(k_VKQ_0 + c::nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K);
        }
    }

    constexpr int reusable_cutoff = D;
#pragma unroll
    for (int i0_stop = D; i0_stop > 0; i0_stop -= 2*nbatch_V2) {
        const int i0_start = i0_stop - 2*nbatch_V2 > 0 ? i0_stop - 2*nbatch_V2 : 0;
        const int i0_diff  = i0_stop - i0_start;

        if (nstages <= 1 && i0_start < reusable_cutoff) {
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, c::nbatch_fa, use_cp_async>
                (V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }
        const half2 * tile_V_i = i0_start < reusable_cutoff ? tile_V : tile_V + (i0_start - reusable_cutoff)/2;

        // Calculate VKQ tile:
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += tile_C_VKQ::I) {
            static_assert((c::nbatch_fa/2) % (np*tile_A::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < c::nbatch_fa/2; k00 += np*tile_A::J) {
                const int k0 = k00 + (threadIdx.y % np)*tile_A::J;

                tile_A A;
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                if (ntiles == 1) {
                    mma(VKQ_C[i_VKQ_0/tile_C_VKQ::I], A, B[k00/(np*tile_A::J)]);
                } else {
#pragma unroll
                    for (int t = 0; t < ntiles/2; ++t) {
                        // Wide version of VKQ_C is column-major => swap A and B.
                        mma(VKQ_C_16[i_VKQ_0/tile_C_VKQ::I * ntiles/2 + t], B_16[k00/(np*tile_A::J) * ntiles/2 + t], A);
                    }
                }
            }
        }

        if (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
}

template<int D, int ncols1, int ncols2, int nwarps, int ntiles, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const int ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    typedef fattn_mma_f16_config<D> c;

#ifdef CP_ASYNC_AVAILABLE
    constexpr int nstages = c::nstages_target;
#else
    constexpr int nstages = 0;
#endif // CP_ASYNC_AVAILABLE

    constexpr int ncols           = ncols1 * ncols2;
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int nbatch_K2       = c::get_nbatch_K2_device(ncols);
    constexpr int nbatch_V2       = c::get_nbatch_V2_device(ncols);

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = D/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    constexpr int stride_tile_V = nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = c::Q_in_reg ? tile_Q                                : tile_Q + ncols        * stride_tile_Q;
    half2 * tile_V    = nstages > 1 ? tile_K + c::nbatch_fa * stride_tile_K : tile_K;
    half2 * tile_mask = nstages > 1 ? tile_V + c::nbatch_fa * stride_tile_V : tile_V + c::nbatch_fa * stride_tile_KV_max;

    tile_B       Q_B[(c::Q_in_reg ? D/(2*tile_B::J) : 1) * ntiles];
    tile_C_VKQ VKQ_C[D/tile_C_VKQ::I  * ntiles];

    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if (jt*ncols1 + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (c::Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            if (ntiles == 1) {
                load_ldmatrix(Q_B[k0/tile_B::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    load_ldmatrix(Q_B_16[k0/tile_B_16::J * ntiles/2 + t],
                        tile_Q + (j0 + t*tile_B_16::I)*stride_tile_Q + k0, stride_tile_Q);
                }
            }
        }
    }

    __syncthreads();

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(nbatch_K2 == D/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>
                (mask_h2 + kb0_start*c::nbatch_fa/2, tile_mask, stride_mask);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
            (K_h2 + int64_t(kb0_start)*c::nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K);
    }

    // Iterate over ne11 == previous tokens:
    int kb0 = kb0_start;
    for (; kb0 < kb0_stop-1; ++kb0) {
        constexpr bool last_iter = false;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, ntiles, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }
    { // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, ntiles, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if (nstages > 1 && nwarps*cols_per_warp > c::nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8/4 threads each, does not need full reduce.
    {
        constexpr int offset_first = ntiles == 1 ? 16 : 2;
        constexpr int offset_last  = ntiles == 1 ?  4 : 1;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int nbatch_combine = c::get_nbatch_combine_device(ncols);
    constexpr int tile_stride    = nbatch_combine + 4;
    static_assert((D/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (ntiles == 1) {
        const int jc_cwmo = (threadIdx.x % (2*tile_C_VKQ::J)) / tile_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*tile_C_VKQ::J) + 2*tile_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*tile_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        static_assert(ntiles == 2 || ntiles == 4, "bad ntiles");
        const int jc_cwm = threadIdx.y*cols_per_warp // jc combine write meta
            + (ntiles == 4 ? ((threadIdx.x % 4) / 2) * tile_C_VKQ_16::I : 0)
            + tile_C_VKQ_16::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && (ntiles == 4 || threadIdx.x % 4 < cols_per_thread)) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    static_assert(np == 1 || ntiles == 1 || ntiles == 2, "bad ntiles");
    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < D/2; k00 += nbatch_combine) {
        if (ntiles == 1) {
            const int jc_cwd = threadIdx.y*tile_B::I + tile_B::get_i(-1); // jc combine write data
#pragma unroll
            for (int k0 = 0; k0 < nbatch_combine; k0 += tile_B::J) {
                const tile_B B = get_transposed(VKQ_C[(k00 + k0)/tile_B::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < tile_B::ne; ++l) {
                    const int k = k0 + tile_B::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                const int j0 = threadIdx.y*cols_per_warp + t*tile_C_VKQ_16::I;
#pragma unroll
                for (int k0 = 0; k0 < nbatch_combine; k0 += tile_C_VKQ_16::J) {
#pragma unroll
                    for (int l = 0; l < tile_C_VKQ_16::ne; ++l) {
                        const int j = j0 + tile_C_VKQ_16::get_i(l);
                        const int k = k0 + tile_C_VKQ_16::get_j(l);

                        tile_Q[j*tile_stride + k] = VKQ_C_16[(k00 + k0)/tile_C_VKQ_16::J * ntiles/2 + t].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && jt*ncols1 + j_dst >= ne01) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(D/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((c_dst * ne01) + (jt*ncols1 + j_dst))*(D/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
}


template<int D, int ncols1, int ncols2, int nwarps, int ntiles>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int32_t nb13,
                            const int32_t nb21, const int32_t nb22, const int32_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int32_t nb33) {
    typedef fattn_mma_f16_config<D> c;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half2);

    const int stride_V = nb21 / sizeof(half2);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    constexpr int kb_niter = FATTN_KQ_STRIDE / c::nbatch_fa; // Number of kernel iterations per assigned KQ slice.

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
        const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;

        const float2 * Q_f2    = (const float2 *) (Q + nb03*sequence + nb02* head0);
        const half2  * K_h2    = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
        const half2  * mask_h2 = ncols2 == 1 && !mask ? nullptr :
            (const half2  *) (mask + nb33*(sequence % ne33) + nb31*jt*ncols1);
        float2       * dstk    = ((float2 *) dst) + ((sequence*ne02 + head0) * ne01) * (D/2);

        const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));

        const int kb0_start_kernel = kb0_start * kb_niter;
        int       kb0_stop_kernel  = kb0_stop  * kb_niter;

        if (KV_max) {
            kb0_stop_kernel = min(kb0_stop_kernel, KV_max[sequence*iter_j + jt] / c::nbatch_fa);
        }

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, ntiles, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale,
                 ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, ntiles, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale,
                 ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
    const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
    const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

    const int head0 = zt * ncols2;

    const float2 * Q_f2    = (const float2 *) (Q + nb03*sequence + nb02* head0);
    const half2  * K_h2    = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2  * mask_h2 = ncols2 == 1 && !mask ? nullptr :
        (const half2  *) (mask + nb33*(sequence % ne33) + nb31*jt*ncols1);
    float2       * dstk    = ((float2 *) dst) + ((sequence*ne02 + head0) * ne01) * (D/2);

    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));

    const int kb0_start_kernel = kb0_start * kb_niter;
    int       kb0_stop_kernel  = kb0_stop  * kb_niter;

    if (KV_max) {
        kb0_stop_kernel = min(kb0_stop_kernel, KV_max[sequence*iter_j + jt] / c::nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, ntiles, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale,
         ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
}

template<int D, int ncols1, int ncols2> // D == head size
static __device__ void flash_attn_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int32_t ne01, const int32_t ne02, const int32_t ne03, const int32_t ne11) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    const int kbc0      = (bidx0 + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc0_stop = (bidx0 + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const int sequence = kbc0 / (iter_k*iter_j*(ne02/ncols2));
    const int head = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j);
    const int jt = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*head) / iter_k; // j index of current tile.

    if (jt*ncols1 + j >= ne01) {
        return;
    }

    dst += sequence*ne02*ne01*D
        + (head*ncols2 + c)*ne01*D
        + (jt*ncols1 + j)*D
        + tid;
    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = bidx*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

#define INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols, ncols2) \
    extern "C" {                                                                                                   \
        __global__ void flash_attn_ext_f16_##D##_##ncols##_##ncols2(                                               \
        const char * __restrict__ Q,                                                                               \
        const char * __restrict__ K,                                                                               \
        const char * __restrict__ V,                                                                               \
        const char * __restrict__ mask,                                                                            \
        const int  * __restrict__ KV_max,                                                                          \
        float      * __restrict__ dst,                                                                             \
        float2     * __restrict__ dst_meta,                                                                        \
        const float scale,                                                                                         \
        const int32_t ne03, const int32_t ne02, const int32_t ne01, const int32_t ne00,                            \
                            const int32_t nb03, const int32_t nb02, const int32_t nb01,                            \
        const int32_t ne13, const int32_t ne12, const int32_t ne11, const int32_t ne10,                            \
                            const int32_t nb13, const int32_t nb12, const int32_t nb11,                            \
                            const int32_t nb23, const int32_t nb22, const int32_t nb21,                            \
                            const int32_t ne33, const int32_t ne32, const int32_t ne31,                            \
                            const int32_t nb33, const int32_t nb32, const int32_t nb31)    { \
            typedef fattn_mma_f16_config<D> c; \
 \
            constexpr int ncols1         = ncols / ncols2; \
            constexpr int ntiles        = ncols <= 8 ? 1 : 2; \
            constexpr int cols_per_warp = ntiles * tile_B::I; \
            constexpr int nwarps_max_x  = ncols / cols_per_warp; \
            constexpr int nwarps_max_y  = c::nbatch_fa / tile_A::I; \
            constexpr int nwarps        = nwarps_max_x*nwarps_max_y <= c::nwarps_max ? nwarps_max_x*nwarps_max_y : c::nwarps_max; \
            flash_attn_ext_f16<D, ncols1, ncols2, nwarps, ntiles>(                                   \
                            Q, K, V, mask, KV_max, dst, dst_meta, scale,                     \
                            ne00, ne01, ne02, ne03,                                          \
                            nb01, nb02, nb03,                                                \
                            ne10, ne11, ne12, ne13,                                          \
                            nb11, nb12, nb13,                                                \
                            nb21, nb22, nb23,                                                \
                            ne31, ne32, ne33,                                                \
                            nb31, nb32, nb33);                                               \
        }                                                                                    \
                                                                                             \
        __launch_bounds__(D, 1)                                                              \
        __global__ void flash_attn_stream_k_fixup_##D##_##ncols##_##ncols2(                  \
            float * __restrict__ dst, const float2 * __restrict__ dst_fixup,                 \
            const int32_t ne03, const int32_t ne02, const int32_t ne01, const int32_t ne11){ \
            constexpr int ncols1         = ncols / ncols2;                                   \
            flash_attn_stream_k_fixup<D, ncols1, ncols2>(                                    \
                dst, dst_fixup, ne01, ne02, ne03, ne11);                                     \
            }                                                                                \
    }

#define INSTANTIATE_FLASH_ATTN_FIXUP_FOR_NCOLS(D, ncols, ncols2)                      \
extern "C" {                                                                            \
                                                                             \
    }

#define INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(ncols1)              \
extern "C" {                                                                  \
    __launch_bounds__(FATTN_KQ_STRIDE/2, 1)                                   \
    __global__ void flash_attn_mask_to_KV_max_##ncols1(                       \
        const half2 * __restrict__ mask, int * __restrict__ KV_max,           \
        const int ne30, const int s31, const int s33) {                       \
            flash_attn_mask_to_KV_max<ncols1>(mask, KV_max, ne30, s31, s33);  \
        }                                                                     \
}

#define INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS(D, ncols) \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols , 1)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols , 2)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols , 4)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols , 8)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, ncols , 16)  \

#define INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(D)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, 8 , 1)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, 8 , 2)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, 8 , 4)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS2(D, 8 , 8)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS(D, 16)     \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS(D, 32)     \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_NCOLS(D, 64)     \

#define INSTANTIATE_FLASH_ATTN_MMA_F16()      \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(64)  \
    INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(128) \
    //INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(80)  \
    //INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(96)  \
    //INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(112) \
    //INSTANTIATE_FLASH_ATTN_MMA_F16_FOR_D(256) \

INSTANTIATE_FLASH_ATTN_MMA_F16()

/*----------------------------------------------------------------------------------------------------------------------*/
template <int ncols1>
static __device__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    // For batched mask support: mask += sequence*s33 + jt*ncols1*s31;
    mask += jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

#define INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX()          \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(1)  \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(2)  \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(4)  \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(8)  \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(16) \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(32) \
    INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX_FOR_NCOLS1(64) \

INSTANTIATE_FLASH_ATTN_MASK_TO_KV_MAX()