#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

extern "C" __global__ void conv2d_f32_generic(
    const float *input,
    size_t in_n, size_t in_c, size_t in_y, size_t in_x,
    size_t in_n_stride, size_t in_c_stride, size_t in_y_stride, size_t in_x_stride,

    const float *kernel,
    size_t ker_o, size_t ker_i, size_t ker_y, size_t ker_x,
    size_t ker_o_stride, size_t ker_i_stride, size_t ker_y_stride, size_t ker_x_stride,

    const float *bias,
    size_t bias_stride,

    size_t ci_per_group, size_t co_per_group,
    
    size_t pad_y, size_t pad_x,
    size_t stride_y, size_t stride_x,
    size_t dil_y, size_t dil_x,
    
    float *output,
    size_t out_n, size_t out_c, size_t out_y, size_t out_x,
    size_t out_n_stride, size_t out_c_stride, size_t out_y_stride, size_t out_x_stride
) {
  assert(in_n == gridDim.z);
  assert(out_n == gridDim.z);
  assert(blockDim.z == 1);

  assert(ker_o == gridDim.y);
  assert(blockDim.y == 1);
  
  size_t n = blockIdx.z;
  size_t co = blockIdx.y;
  size_t xy = blockIdx.x * blockDim.x + threadIdx.x;
  size_t oy = xy / out_x;
  size_t ox = xy % out_x;

  if (ox >= out_x || oy >= out_y) {
    return;
  }

  char *pci = (char*) input + n * in_n_stride;
  char *pck = (char*)  kernel + co * ker_o_stride;

  float sum = 0;
  if(bias) {
    *(float*) ((char*) bias + co * bias_stride);
  }

  for(int ci = 0; ci < ker_i; ci++ ) {
    for(int ky = 0; ky < ker_y; ky++) {
      int y = oy * stride_y + ky * dil_y - pad_y;
      if(y < 0 || y >= in_y) {
        continue;
      }
      for(int kx = 0; kx < ker_x; kx++) {
        int x = ox * stride_x + kx * dil_x - pad_x;
        if(x < 0 || x >= in_x) {
          continue;
        }
        float i = *(float*) (pci + ci * in_c_stride + x * in_x_stride + y * in_y_stride);
        float k = *(float*) (pck + ci * ker_i_stride + kx * ker_x_stride + ky * ker_y_stride);
        sum += i*k;
      }
    }
  }

  size_t poffset = n * out_n_stride + co * out_c_stride + oy * out_y_stride + ox * out_x_stride;

  float *store = (float*) ((char*) output + poffset);
  *store = sum;
  
}
