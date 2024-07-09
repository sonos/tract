#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

namespace utils {

    METAL_FUNC uint indices_to_idx_1(uint index, constant const uint strides[1]) {
      return index * strides[0];
    }

    METAL_FUNC uint indices_to_idx_2(uint2 indices, constant const uint strides[2]) {
      return indices.x * strides[1] + indices.y * strides[0];
    }

    METAL_FUNC uint indices_to_idx_3(uint3 indices, constant const uint strides[3]) {
      return indices.x * strides[2] + indices.y * strides[1] + indices.z * strides[0];
    }

    METAL_FUNC uint indices_to_idx_4(uint3 indices,
                                     constant const uint shape[4], 
                                     constant const uint strides[4]) {
      auto idx = indices.x * strides[3] + indices.y * strides[2];
      idx += (indices.z % shape[1]) * strides[1];
      indices.z /= shape[1];
      idx += indices.z * strides[0];
      return idx;
    }

    METAL_FUNC uint indices_to_idx_5(uint3 indices,
                                     constant const uint shape[5], 
                                     constant const uint strides[5]) {
      auto idx = indices.x * strides[4] + indices.y * strides[3];
      idx += (indices.z % shape[2]) * strides[2];
      indices.z /= shape[2];
      idx += (indices.z % shape[1]) * strides[1];
      indices.z /= shape[1];
      idx += indices.z * strides[0];
      return idx;
    }

    METAL_FUNC uint indices_to_idx_6(uint3 indices,
                                     constant const uint shape[6], 
                                     constant const uint strides[6]) {
      auto idx = indices.x * strides[5] + indices.y * strides[4];
      idx += (indices.z % shape[3]) * strides[3];
      indices.z /= shape[3];
      idx += (indices.z % shape[2]) * strides[2];
      indices.z /= shape[2];
      idx += (indices.z % shape[1]) * strides[1];
      indices.z /= shape[1];
      idx += indices.z * strides[0];
      return idx;
    }
}

namespace array_ops {

    #define INSTANTIATE_BROADCAST(tname, type)                       \
    template [[host_name("array_ops::broadcast_nd1_" #tname)]] [[kernel]] void broadcast_nd1<type>(                                                           \
        device const type *input [[buffer(0)]],                     \
        constant const uint * input_strides [[buffer(1)]],           \
        device type *output [[buffer(2)]],                          \
        constant const uint * out_shape [[buffer(3)]],               \
        uint tpig[[thread_position_in_grid]],                        \
        uint grid_dim [[threads_per_grid]]                           \
    );                                                               \
    template [[host_name("array_ops::broadcast_nd2_" #tname)]] [[kernel]] void broadcast_nd2<type>(                                                        \
        device const type *input [[buffer(0)]],                   \
        constant const uint * input_strides [[buffer(1)]],         \
        device type *output [[buffer(2)]],                        \
        constant const uint * out_shape [[buffer(3)]],             \
        uint2 tpig[[thread_position_in_grid]],                     \
        uint2 grid_dim [[threads_per_grid]]                        \
    );                                                             \
    template [[host_name("array_ops::broadcast_nd3_" #tname)]] [[kernel]] void broadcast_nd3<type>(                                 \
          device const type *input [[buffer(0)]],                  \
        constant const uint * input_strides [[buffer(1)]],         \
        device type *output [[buffer(2)]],                         \
        constant const uint * out_shape [[buffer(3)]],             \
        uint3 tpig[[thread_position_in_grid]],                     \
        uint3 grid_dim [[threads_per_grid]]                        \
    );                                                             \
    template [[host_name("array_ops::broadcast_nd4_" #tname)]] [[kernel]] void broadcast_nd4<type>(                                 \
          device const type *input [[buffer(0)]],                  \
        constant const uint * input_strides [[buffer(1)]],         \
        device type *output [[buffer(2)]],                         \
        constant const uint * out_shape [[buffer(3)]],             \
        uint3 tpig[[thread_position_in_grid]],                     \
        uint3 grid_dim [[threads_per_grid]]                        \
    );                                                             \
    template [[host_name("array_ops::broadcast_nd5_" #tname)]] [[kernel]] void broadcast_nd5<type>(                                 \
          device const type *input [[buffer(0)]],                  \
        constant const uint * input_strides [[buffer(1)]],         \
        device type *output [[buffer(2)]],                         \
        constant const uint * out_shape [[buffer(3)]],             \
        uint3 tpig[[thread_position_in_grid]],                     \
        uint3 grid_dim [[threads_per_grid]]                        \
    );                                                             \
    template [[host_name("array_ops::broadcast_nd6_" #tname)]] [[kernel]] void broadcast_nd6<type>(                                 \
          device const type *input [[buffer(0)]],                 \
        constant const uint * input_strides [[buffer(1)]],         \
        device type *output [[buffer(2)]],                        \
        constant const uint * out_shape [[buffer(3)]],             \
        uint3 tpig[[thread_position_in_grid]],                     \
        uint3 grid_dim [[threads_per_grid]]                        \
    );

    #define INSTANTIATE_COPY_OP(tname, type)     \
    template [[host_name("array_ops::copy_" #tname)]] [[kernel]] void mem_copy<type>(                                                 \
        device const type *input [[buffer(0)]],                     \
        device type *output [[buffer(1)]],                          \
        uint tpig[[thread_position_in_grid]]                        \
    );

    #define INSTANTIATE_CAST_OP(tname, itype, otype)     \
    template [[host_name("array_ops::cast_" #tname)]] [[kernel]] void cast<itype, otype>(                               \
        device const itype *input [[buffer(0)]],                     \
        device otype *output [[buffer(1)]],                          \
        uint tpig[[thread_position_in_grid]]                         \
    );       
    
    template<typename T> [[kernel]] void mem_copy(             
        device const T *input [[buffer(0)]],                 
        device T *output [[buffer(1)]],                        
        uint tpig[[thread_position_in_grid]]                    
    ) {
      output[tpig] = input[tpig];
    }

    template<typename In, typename Out> [[kernel]] void cast(             
          device const In *input [[buffer(0)]],                 
        device Out *output [[buffer(1)]],                        
        uint tpig[[thread_position_in_grid]]                    
    ) {
      output[tpig] = static_cast<Out>(input[tpig]);
    }

    template<typename T>  [[kernel]] void broadcast_nd1(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint tpig[[thread_position_in_grid]],                     
        uint grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_1(tpig, input_strides);
      output[idx] = input[idx];
    }

    template<typename T>  
    [[kernel]] void broadcast_nd2(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint2 tpig[[thread_position_in_grid]],                     
        uint2 grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_2(tpig, input_strides);
      auto out_idx = tpig.x + grid_dim.x * tpig.y;
      output[out_idx] = input[idx];
    }

    template<typename T>  
    [[kernel]] void broadcast_nd3(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint3 tpig[[thread_position_in_grid]],                     
        uint3 grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_3(tpig, input_strides);
      auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
      output[out_idx] = input[idx];
    }

    template<typename T>  
    [[kernel]] void broadcast_nd4(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint3 tpig[[thread_position_in_grid]],                     
        uint3 grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_4(tpig, out_shape, input_strides);
      auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
      output[out_idx] = input[idx];
    }

    template<typename T>  
    [[kernel]] void broadcast_nd5(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint3 tpig[[thread_position_in_grid]],                     
        uint3 grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_5(tpig, out_shape, input_strides);
      auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
      output[out_idx] = input[idx];
    }

    template<typename T>  
    [[kernel]] void broadcast_nd6(             
          device const T *input [[buffer(0)]],                 
        constant const uint * input_strides [[buffer(1)]],         
        device T *output [[buffer(2)]],                        
        constant const uint * out_shape [[buffer(3)]],             
        uint3 tpig[[thread_position_in_grid]],                     
        uint3 grid_dim [[threads_per_grid]]                        
    ) {
      auto idx = utils::indices_to_idx_6(tpig, out_shape, input_strides);
      auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
      output[out_idx] = input[idx];
    }

    #define INSTANTIATE_ALL(tname, type)                             \
    INSTANTIATE_COPY_OP(tname, type)                                 \
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
    INSTANTIATE_BROADCAST(tname, type)

    INSTANTIATE_ALL(bool, bool)
    INSTANTIATE_ALL(f32, float)
    INSTANTIATE_ALL(f16, half)
    INSTANTIATE_ALL(i8, int8_t)
    INSTANTIATE_ALL(i16, int16_t)
    INSTANTIATE_ALL(i32, int32_t)
    INSTANTIATE_ALL(i64, int64_t)
    INSTANTIATE_ALL(u8, uint8_t)
    INSTANTIATE_ALL(u16, uint16_t)
    INSTANTIATE_ALL(u32, uint32_t)
    INSTANTIATE_ALL(u64, uint64_t)

}
