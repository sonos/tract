#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

// liquid:true

{% for georank in (1..4) %}

extern "C" __global__ void conv{{georank}}d_f32_generic(
    const float *input,
    size_t in_n, size_t in_c,
    {% for i in (1..georank) %} size_t in_{{i}}, {% endfor %}
    size_t in_n_stride, size_t in_c_stride,
    {% for i in (1..georank) %} size_t in_{{i}}_stride, {% endfor %}

    const float *kernel,
    size_t groups, size_t co_per_group, size_t ci_per_group,
    {% for i in (1..georank) %} size_t ker_{{i}}, {% endfor %}
    size_t ker_g_stride, size_t ker_o_stride, size_t ker_i_stride,
    {% for i in (1..georank) %} size_t ker_{{i}}_stride, {% endfor %}

    const float *bias,
    size_t bias_stride,
  
    {% for i in (1..georank) %} size_t pad_{{i}}, {% endfor %}
    {% for i in (1..georank) %} size_t stride_{{i}}, {% endfor %}
    {% for i in (1..georank) %} size_t dil_{{i}}, {% endfor %}
    
    float *output,
    size_t out_n, size_t out_c,
    {% for i in (1..georank) %} size_t out_{{i}}, {% endfor %}
    size_t out_n_stride, size_t out_c_stride
    {% for i in (1..georank) %}, size_t out_{{i}}_stride {% endfor %}
) {
  assert(in_n == gridDim.z);
  assert(out_n == gridDim.z);
  assert(blockDim.z == 1);

  assert(blockDim.y == 1);
  
  size_t n = blockIdx.z;
  size_t co = blockIdx.y;
  size_t group = co / co_per_group;
  size_t xyz = blockIdx.x * blockDim.x + threadIdx.x;
  {% capture georank_minus_1 %}{{georank|minus:1}}{%endcapture%}
  {% for i in (1..georank_minus_1) reversed %}
     size_t ox_{{i}} = xyz % out_{{i}};
     xyz = xyz / out_{{i}};
  {% endfor %}
  size_t ox_{{georank}} = xyz;


  {% for i in (1..georank) %}
     if (ox_{{i}} >= out_{{i}}) {
        return;
     }
  {% endfor %}

  // printf("co={} group={} groups={} co_per_group={}\n", co, group, co_per_group);

  const float *pfi = input + n * in_n_stride + ci_per_group * group * in_c_stride;
  const float *pfk = kernel + co * ker_o_stride; 

  float sum = 0;
  if(bias) {
    sum = *(bias + co * bias_stride);
  }

  for(int ci = 0; ci < ci_per_group; ci++ ) {
  {% for i in (1..georank) %}
    for(int k_{{i}} = 0; k_{{i}} < ker_{{i}}; k_{{i}}++) {
      int x_{{i}} = ox_{{i}} * stride_{{i}} + k_{{i}} * dil_{{i}} - pad_{{i}};
      if (x_{{i}} < 0 || x_{{i}} >= in_{{i}}) {
        continue;
      }
  {% endfor %}

        float i = *(pfi + ci * in_c_stride
        {% for i in (1..georank) %} + x_{{i}} * in_{{i}}_stride {%endfor%});
        float k = *(pfk + ci * ker_i_stride +
        {% for i in (1..georank) %} + k_{{i}} * ker_{{i}}_stride {%endfor%});
        sum += i*k;
    {% for i in (1..georank) %} } {%endfor%} // nested georank loops
  } // ci loop

  size_t poffset = n * out_n_stride + co * out_c_stride
      {% for i in (1..georank) %} + ox_{{i}} * out_{{i}}_stride {%endfor%};
  *(output + poffset) = sum;
  
}

{% endfor %}
