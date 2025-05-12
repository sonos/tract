#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

namespace utils {
    METAL_FUNC uint indices_to_idx_2(uint2 indices, constant const size_t strides[2]) {
      return indices.x * strides[1] + indices.y * strides[0];
    }

    // Returns offset for iterating over most inner axis
    METAL_FUNC uint indices_to_outer_idx(uint3 indices,
                                  constant const size_t * shape, 
                                  constant const size_t * strides,
                                  size_t rank) {
      if (rank == 1) {
        return 0;
      } else if (rank == 2) {
        return indices.x * strides[0];
      } else {
        auto idx = indices.x * strides[rank - 2] + indices.y * strides[rank - 3];

        for (int32_t i = rank - 4; i >= 0; i--) {
          idx += (indices.z % shape[i]) * strides[i];
          indices.z /= shape[i];
        }
        return idx;
      }
      }
}



#define INSTANTIATE_COPY(tname, type)                         \
template [[host_name("array_ops::copy_nd1_" #tname)]] [[kernel]] copy_nd1_t copy_nd1<type>;  \
template [[host_name("array_ops::copy_nd2_" #tname)]] [[kernel]] copy_nd2_t copy_nd2<type>;  \
template [[host_name("array_ops::copy_nd3_" #tname)]] [[kernel]] copy_nd3_t copy_nd3<type>;  \
template [[host_name("array_ops::copy_nd4_" #tname)]] [[kernel]] copy_nd4_t copy_nd4<type>;  \
template [[host_name("array_ops::copy_nd5_" #tname)]] [[kernel]] copy_nd5_t copy_nd5<type>;  \
template [[host_name("array_ops::copy_nd6_" #tname)]] [[kernel]] copy_nd6_t copy_nd6<type>;  \
template [[host_name("array_ops::copy_unicast_" #tname)]] [[kernel]] copy_unicast_t copy_unicast<type>;

#define INSTANTIATE_CAST_OP(tname, itype, otype)     \
template [[host_name("array_ops::cast_" #tname)]] [[kernel]] cast_t cast<itype, otype>;
    
template<typename In, typename Out> [[kernel]] void cast(             
      device const void *input_b [[buffer(0)]],                 
    device void *output_b [[buffer(1)]],                        
    uint tpig[[thread_position_in_grid]]                    
) {
  device const In *input = (device const In *)input_b;
  device Out* output = (device Out *) output_b;
  output[tpig] = static_cast<Out>(input[tpig]);
}

typedef decltype(cast<float, float>) cast_t;

template<typename T> [[kernel]] void copy_unicast(             
    device const void *input_b [[buffer(0)]],                 
    device void *output_b [[buffer(1)]],                        
    uint tpig[[thread_position_in_grid]]                    
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;
  output[tpig] = input[tpig];
}

typedef decltype(copy_unicast<float>) copy_unicast_t;


template<typename T>  [[kernel]] void copy_nd1(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],
    constant const size_t * out_strides [[buffer(4)]],             
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]                      
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;
  for (size_t i = tpitg.x; i < out_shape[0]; i += ntg.x) {
    output[i] = input[i * input_strides[0]];
  }
}

typedef decltype(copy_nd1<float>) copy_nd1_t;

template<typename T>  
[[kernel]] void copy_nd2(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],
    constant const size_t * out_strides [[buffer(4)]],              
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]                  
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  auto idx = utils::indices_to_outer_idx(tgpig, out_shape, input_strides, 2);
  auto out_idx = utils::indices_to_outer_idx(tgpig, out_shape, out_strides, 2);
  for (size_t i = tpitg.x; i < out_shape[1]; i += ntg.x) {
    output[out_idx + i] = input[idx + i * input_strides[1]];
  }
}

typedef decltype(copy_nd2<float>) copy_nd2_t;

template<typename T>  
[[kernel]] void copy_nd3(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],  
    constant const size_t * out_strides [[buffer(4)]],                
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]                           
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  auto idx = utils::indices_to_outer_idx(tgpig, out_shape, input_strides, 3);
  auto out_idx = utils::indices_to_outer_idx(tgpig, out_shape, out_strides, 3);
  for (size_t i = tpitg.x; i < out_shape[2]; i += ntg.x) {
    output[out_idx + i] = input[idx + i * input_strides[2]];
  }
}

typedef decltype(copy_nd3<float>) copy_nd3_t;

template<typename T>  
[[kernel]] void copy_nd4(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],  
    constant const size_t * out_strides [[buffer(4)]],           
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]                       
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  auto idx = utils::indices_to_outer_idx(tgpig, out_shape, input_strides, 4);
  auto out_idx = utils::indices_to_outer_idx(tgpig, out_shape, out_strides, 4);
  for (size_t i = tpitg.x; i < out_shape[3]; i += ntg.x) {
    output[out_idx + i] = input[idx + i * input_strides[3]];
  }
}

typedef decltype(copy_nd4<float>) copy_nd4_t;

template<typename T>  
[[kernel]] void copy_nd5(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],
    constant const size_t * out_strides [[buffer(4)]],              
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]               
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  auto idx = utils::indices_to_outer_idx(tgpig, out_shape, input_strides, 5);
  auto out_idx = utils::indices_to_outer_idx(tgpig, out_shape, out_strides, 5);
  for (size_t i = tpitg.x; i < out_shape[4]; i += ntg.x) {
    output[out_idx + i] = input[idx + i * input_strides[4]];
  }
}

typedef decltype(copy_nd5<float>) copy_nd5_t;

template<typename T>  
[[kernel]] void copy_nd6(             
      device const void *input_b [[buffer(0)]],                 
    constant const size_t * input_strides [[buffer(1)]],         
    device void *output_b [[buffer(2)]],                        
    constant const size_t * out_shape [[buffer(3)]],
    constant const size_t * out_strides [[buffer(4)]],              
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]                     
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  auto idx = utils::indices_to_outer_idx(tgpig, out_shape, input_strides, 6);
  auto out_idx = utils::indices_to_outer_idx(tgpig, out_shape, out_strides, 6);
  for (size_t i = tpitg.x; i < out_shape[5]; i += ntg.x) {
    output[out_idx + i] = input[idx + i * input_strides[5]];
  }
}

typedef decltype(copy_nd6<float>) copy_nd6_t;

// Rotate half of the input buffer 
//
// Y = Concat(Neg(Slice(X, X.shape[-1]/2.., -1)), Slice(X, ..X.shape[-1]/2, -1))
//
template<typename T>  
[[kernel]] void rotate_half_nd2(             
      device const void *input_b [[buffer(0)]],                 
      device void *output_b [[buffer(1)]],                        
      constant const size_t * shape [[buffer(2)]],
      constant const size_t * strides [[buffer(3)]],              
      uint2 tpig[[thread_position_in_grid]]                   
) {
  device const T *input = (device const T *)input_b;
  device T* output = (device T *) output_b;

  uint2 rotated_tpig = tpig;
  rotated_tpig.x += shape[1] / 2;

  // output[tpig] = -1 * input[rotated_tpig]
  // output[rotated_tpig] = input[tpig]

  auto rotated_idx = utils::indices_to_idx_2(rotated_tpig, strides);
  auto out_idx = utils::indices_to_idx_2(tpig, strides);
  
  output[out_idx] = -input[rotated_idx];

  auto idx = utils::indices_to_idx_2(tpig, strides);
  auto rotated_out_idx = utils::indices_to_idx_2(rotated_tpig, strides);

  output[rotated_out_idx] = input[idx];
}

typedef decltype(rotate_half_nd2<float>) rotate_half_nd2_t;

#define INSTANTIATE_ROTATE_HALF_OP(tname, type)     \
template [[host_name("array_ops::rotate_half_nd2_" #tname)]] [[kernel]] rotate_half_nd2_t rotate_half_nd2<type>;



#define INSTANTIATE_CAST_AND_COPY(tname, type)    \
INSTANTIATE_CAST_OP(tname ##_bool, type, bool)    \
INSTANTIATE_CAST_OP(tname ##_f32, type, float)    \
INSTANTIATE_CAST_OP(tname ##_f16, type, half)     \
INSTANTIATE_CAST_OP(tname ##_u8, type, uint8_t)   \
INSTANTIATE_CAST_OP(tname ##_u16, type, uint16_t) \
INSTANTIATE_CAST_OP(tname ##_u32, type, uint32_t) \
INSTANTIATE_CAST_OP(tname ##_u64, type, uint64_t) \
INSTANTIATE_CAST_OP(tname ##_i8, type, int8_t)    \
INSTANTIATE_CAST_OP(tname ##_i16, type, int16_t)  \
INSTANTIATE_CAST_OP(tname ##_i32, type, int32_t)  \
INSTANTIATE_CAST_OP(tname ##_i64, type, int64_t)  \
INSTANTIATE_COPY(tname, type)                     

#define INSTANTIATE_ALL(tname, type)              \
INSTANTIATE_CAST_AND_COPY(tname, type)            \
INSTANTIATE_ROTATE_HALF_OP(tname, type)

INSTANTIATE_CAST_AND_COPY(bool, bool)
INSTANTIATE_ALL(f32, float)
INSTANTIATE_ALL(f16, half)
INSTANTIATE_ALL(i8, int8_t)
INSTANTIATE_ALL(i16, int16_t)
INSTANTIATE_ALL(i32, int32_t)
INSTANTIATE_ALL(i64, int64_t)
INSTANTIATE_CAST_AND_COPY(u8, uint8_t)
INSTANTIATE_CAST_AND_COPY(u16, uint16_t)
INSTANTIATE_CAST_AND_COPY(u32, uint32_t)
INSTANTIATE_CAST_AND_COPY(u64, uint64_t)


