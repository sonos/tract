// From https://github.com/cyrusmsk/gemm_apple/blob/main/gemm_metal.py

#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

constant uint LID [[function_constant(0)]];
constant uint dim [[function_constant(1)]];

kernel void mmm_tile_8x8(device float *a [[buffer(0)]], // output
                 device const float *data1 [[buffer(1)]],
                 device const float *data2 [[buffer(2)]],
                 uint3 gid [[threadgroup_position_in_grid]],
                 uint3 lid [[thread_position_in_threadgroup]]) {{
  a += gid.x * 32 * dim + (gid.y * LID + lid.y) * 32;
  data1 += gid.x * 32 * dim;
  data2 += (gid.y * LID + lid.y) * 32;

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++) {{
    for (uint j = 0; j < 4; j++) {{
      acc[i][j] = simdgroup_float8x8(0);
    }}
  }}

  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  for (uint k = 0; k < dim; k+=8) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A[0], data1+k+(0*dim), dim, ulong2(0, 0));
    simdgroup_load(A[1], data1+k+(8*dim), dim, ulong2(0, 0));
    simdgroup_load(A[2], data1+k+(16*dim), dim, ulong2(0, 0));
    simdgroup_load(A[3], data1+k+(24*dim), dim, ulong2(0, 0));
    simdgroup_load(B[0], data2+0+k*dim, dim, ulong2(0, 0));
    simdgroup_load(B[1], data2+8+k*dim, dim, ulong2(0, 0));
    simdgroup_load(B[2], data2+16+k*dim, dim, ulong2(0, 0));
    simdgroup_load(B[3], data2+24+k*dim, dim, ulong2(0, 0));

    simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
    simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
    simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
    simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
    simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
    simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
    simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
    simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
    simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
    simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
    simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
    simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
    simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
    simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
    simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
    simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
  }}
  simdgroup_store(acc[0][0], a+(0+0*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[1][0], a+(8+0*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[2][0], a+(16+0*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[3][0], a+(24+0*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[0][1], a+(0+8*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[1][1], a+(8+8*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[2][1], a+(16+8*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[3][1], a+(24+8*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[0][2], a+(0+16*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[1][2], a+(8+16*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[2][2], a+(16+16*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[3][2], a+(24+16*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[0][3], a+(0+24*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[1][3], a+(8+24*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[2][3], a+(16+24*dim), dim, ulong2(0, 0));
  simdgroup_store(acc[3][3], a+(24+24*dim), dim, ulong2(0, 0));
}}
