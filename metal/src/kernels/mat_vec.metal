#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

#define NUM_SIMDGROUP 32

kernel void op_mat_vec_f32(device float *lhs [[buffer(0)]],
                           device float *rhs [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant   int32_t & nrows,
                           constant   int32_t & ncols,
                           uint3 tgpig[[threadgroup_position_in_grid]],
        				   uint  tiisg[[thread_index_in_simdgroup]],
        				   uint  sgitg[[simdgroup_index_in_threadgroup]]
                           ) {

    const int32_t row_group_size = 4;
    const int32_t _batch_idx = tgpig.x;
    const int32_t row_group_start = tgpig.y*row_group_size;

    for (int row_group_idx = 0; row_group_idx < row_group_size; ++row_group_idx) {
        int row_idx = row_group_start + row_group_idx;
        if (row_idx >= nrows) {
            break;
        }
        device const float * lhs_row = (device const float *) (lhs + row_idx * ncols);
        float sumf = 0;
        // Accumulate per simd
        for (int i = tiisg; i < ncols; i += NUM_SIMDGROUP) {
            sumf += (float) rhs[i] * (float) lhs_row[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            output[row_idx] = all_sum;
        }
    }
}
