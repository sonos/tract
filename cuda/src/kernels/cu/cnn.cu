#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

// liquid:true

{% assign types = "f32,f16" | split: "," %}

{% for type in types %}
{% if type == "f32" %}
  {% assign T = "float" %}
  {% assign load = "" %}
  {% assign store = "" %}
{% else %}
  {% assign T = "__half" %}
  {% assign load = "__half2float(" %}
  {% assign store = "__float2half(" %}
{% endif %}

{% for georank in (1..4) %}

extern "C" __global__ void conv{{georank}}d_{{type}}_generic(
    const {{T}} *input,
    int32_t in_n, int32_t in_c,
    {% for i in (1..georank) %} int32_t in_{{i}}, {% endfor %}
    int32_t in_n_stride, int32_t in_c_stride,
    {% for i in (1..georank) %} int32_t in_{{i}}_stride, {% endfor %}

    const {{T}} *kernel,
    int32_t groups, int32_t co_per_group, int32_t ci_per_group,
    {% for i in (1..georank) %} int32_t ker_{{i}}, {% endfor %}
    int32_t ker_g_stride, int32_t ker_o_stride, int32_t ker_i_stride,
    {% for i in (1..georank) %} int32_t ker_{{i}}_stride, {% endfor %}

    const {{T}} *bias,
    int32_t bias_stride,

    {% for i in (1..georank) %} int32_t pad_{{i}}, {% endfor %}
    {% for i in (1..georank) %} int32_t stride_{{i}}, {% endfor %}
    {% for i in (1..georank) %} int32_t dil_{{i}}, {% endfor %}

    {{T}} *output,
    int32_t out_n, int32_t out_c,
    {% for i in (1..georank) %} int32_t out_{{i}}, {% endfor %}
    int32_t out_n_stride, int32_t out_c_stride
    {% for i in (1..georank) %}, int32_t out_{{i}}_stride {% endfor %}
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

  const {{T}} *pfi = input + n * in_n_stride + ci_per_group * group * in_c_stride;
  const {{T}} *pfk = kernel + co * ker_o_stride;

  float sum = 0;
  if(bias) {
    sum = {{load}}*(bias + co * bias_stride){% if type == "f16" %}){% endif %};
  }

  for(int ci = 0; ci < ci_per_group; ci++ ) {
  {% for i in (1..georank) %}
    for(int k_{{i}} = 0; k_{{i}} < ker_{{i}}; k_{{i}}++) {
      int x_{{i}} = ox_{{i}} * stride_{{i}} + k_{{i}} * dil_{{i}} - pad_{{i}};
      if (x_{{i}} < 0 || x_{{i}} >= in_{{i}}) {
        continue;
      }
  {% endfor %}

        float i = {{load}}*(pfi + ci * in_c_stride
        {% for i in (1..georank) %} + x_{{i}} * in_{{i}}_stride {%endfor%}){% if type == "f16" %}){% endif %};
        float k = {{load}}*(pfk + ci * ker_i_stride +
        {% for i in (1..georank) %} + k_{{i}} * ker_{{i}}_stride {%endfor%}){% if type == "f16" %}){% endif %};
        sum += i*k;
    {% for i in (1..georank) %} } {%endfor%} // nested georank loops
  } // ci loop

  size_t poffset = n * out_n_stride + co * out_c_stride
      {% for i in (1..georank) %} + ox_{{i}} * out_{{i}}_stride {%endfor%};
  {% if type == "f16" %}
  *(output + poffset) = __float2half(sum);
  {% else %}
  *(output + poffset) = sum;
  {% endif %}

}

{% endfor %}
{% endfor %}
