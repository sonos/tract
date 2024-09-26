#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

#define NUM_SIMDGROUP 32

#define INSTANTIATE_BASIC_MATMUL(tname, type)                    \
template [[host_name("matmul::basic_matvec_" #tname)]]           \
[[kernel]] void basic_matvec<type>(                              \
    device const type *lhs [[buffer(0)]],                        \
    device const type *rhs [[buffer(1)]],                        \
    device type *output [[buffer(2)]],                           \
    constant   int32_t & m,                                      \
    constant   int32_t & k,                                      \
    uint3 tgpig[[threadgroup_position_in_grid]],                 \
    uint  tiisg[[thread_index_in_simdgroup]],                    \
    uint  sgitg[[simdgroup_index_in_threadgroup]]                \
);                                                               \
template [[host_name("matmul::basic_matmul_" #tname)]]           \
[[kernel]] void basic_matmul<type>(                              \
    device const type *lhs [[buffer(0)]],                        \
    device const type *rhs [[buffer(1)]],                        \
    device type *output [[buffer(2)]],                           \
    constant   int32_t & m,                                      \
    constant   int32_t & k,                                      \
    constant   int32_t & n,                                      \
    constant   int32_t & transpose_lhs,                          \
    constant   int32_t & transpose_rhs,                          \
    uint3 tgpig[[threadgroup_position_in_grid]],                 \
    uint  tiisg[[thread_index_in_simdgroup]],                    \
    uint  sgitg[[simdgroup_index_in_threadgroup]]                \
);                                                           


template<typename T>  
[[kernel]]  void basic_matvec(device const T *lhs [[buffer(0)]],
                              device const T *rhs [[buffer(1)]],
                              device T *output [[buffer(2)]],
                              constant   int32_t & m,
                              constant   int32_t & k,
                              uint3 tgpig[[threadgroup_position_in_grid]],
                              uint  tiisg[[thread_index_in_simdgroup]],
                              uint  sgitg[[simdgroup_index_in_threadgroup]]
                              ) {
    
    const int32_t m_group_size = 4;
    const int32_t _batch_idx = tgpig.x;
    const int32_t m_group_start = tgpig.y*m_group_size;
    
    for (int m_group_idx = 0; m_group_idx < m_group_size; ++m_group_idx) {
        int m_idx = m_group_start + m_group_idx;
        if (m_idx >= m) {
            break;
        }
        device const T * lhs_m = (device const T *) (lhs + m_idx * k);
        T sumf = 0;
        // Accumulate per simd

        for (int i = tiisg; i < k; i += NUM_SIMDGROUP) {
            sumf +=  rhs[i] * lhs_m[i];
        }
        T all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            output[m_idx] = all_sum;
        }
    }
}


template<typename T>  
[[kernel]]  void basic_matmul(device const T  *lhs [[buffer(0)]],
                              device const T *rhs [[buffer(1)]],
                              device T *output [[buffer(2)]],
                              constant   int32_t & m,
                              constant   int32_t & k,
                              constant   int32_t & n,
                              constant   int32_t & transpose_lhs,
                              constant   int32_t & transpose_rhs, 
                              uint3 tgpig[[threadgroup_position_in_grid]],
                              uint  tiisg[[thread_index_in_simdgroup]],
                              uint  sgitg[[simdgroup_index_in_threadgroup]]
                              ) {
    
    const int32_t group_size = 4;
    const int32_t n_group_start = tgpig.x * group_size;
    const int32_t m_group_start = tgpig.y * group_size;
    
    // [m_idx, n_idx] = m_idx * n + n_idx
    
    for (int m_group_idx = 0; m_group_idx < group_size; ++m_group_idx) {
        int m_idx = m_group_start + m_group_idx;
        if (m_idx >= m) {
            break;
        }
        for (int n_group_idx = 0; n_group_idx < group_size; ++n_group_idx) {
            int n_idx = n_group_start + n_group_idx;
            
            if (n_idx >= n) {
                break;
            }
            
            T sumf = 0;
            // Accumulate per simd
            if(transpose_lhs == 0 && transpose_rhs == 0) {
                for (int i = tiisg; i < k; i += NUM_SIMDGROUP) {
                    // lhs[m_idx, i] = m_idx * k + i
                    // rhs[i, n_idx] = i * n + n_idx
                    sumf += rhs[i * n + n_idx] * lhs[m_idx * k + i];
                }
            } else if(transpose_lhs == 0 && transpose_rhs != 0) {
                for (int i = tiisg; i < k; i += NUM_SIMDGROUP) {
                    // lhs[m_idx, i] = m_idx * k + i
                    // rhs[n_idx, i] = n_idx * k + i
                    sumf += rhs[n_idx * k + i] * lhs[m_idx * k + i];
                }
            } else if(transpose_lhs != 0 && transpose_rhs != 0) {
                for (int i = tiisg; i < k; i += NUM_SIMDGROUP) {
                    // lhs[i, m_idx] = i * m + m_idx
                    // rhs[n_idx, i] = n_idx * k + i
                    sumf += rhs[n_idx * k + i] * lhs[i * m + m_idx];
                }
            } else if(transpose_lhs != 0 && transpose_rhs == 0) {
                for (int i = tiisg; i < k; i += NUM_SIMDGROUP) {
                    // lhs[i, m_idx] = i * m + m_idx
                    // rhs[i, n_idx] = i * n + n_idx
                    sumf += rhs[i * n + n_idx] * lhs[i * m + m_idx];
                }
            }
            
            T all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                output[m_idx * n + n_idx] = all_sum;
            }
        }
    }
}

INSTANTIATE_BASIC_MATMUL(f32, float)
INSTANTIATE_BASIC_MATMUL(f16, half)
