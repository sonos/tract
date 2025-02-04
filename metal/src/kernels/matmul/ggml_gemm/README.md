### Kernel list

Outputs are always F32. Accumulation is F32. Weights are in first position.
- kernel_mul_mv: (f16_f16, f32_f32, f16_f32)
- kernel_mul_mv_1row (f16_f32)
- kernel_mul_mv_l4 (f16_f32)

- kernel_mul_mm (f32_f32, f16_f32)

### Tensor layout

Channel is a batch-like dimension. It is NOT as in the matmul operational convention (==m or  k)

- A: [a_batch, a_channel, m, k]
- B: [b_batch, b_channel, n, k]
- Out: [b_batch, b_channel, n, m]

### Kernel prototype

Kernel params. Used for all MatVec and MatMul kernels
```
typedef struct {
    int32_t batch;
    int32_t m;
    int32_t k;
    int32_t n;
    uint64_t a_strides[4];
    uint64_t b_strides[4];
    int32_t channel_broadcast_ratio;
    int32_t batch_broadcast_ratio;
} ggml_metal_kargs_mul;
````

Channel_broadcast_ratio and batch_broadcast_ratio are currently always 1!
These are for multiplying multiple Bs with a single A.
Kernel should support it if frame tries to call them with values != 1.

```
template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg)
```

```
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
```

### GGML ref params

```
ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00, // Inner axis len of input A: k
        /*.ne01 =*/ ne01, // m
        /*.ne02 =*/ ne02, // a_channel
        /*.nb00 =*/ nb00, // Inner stride of input A: sizeof(floatxx)
        /*.nb01 =*/ nb01, // k * sizeof(floatxx)
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10, // Inner axis len of input B: k
        /*.ne11 =*/ ne11, // n
        /*.ne12 =*/ ne12, //b_channel
        /*.nb10 =*/ nb10, // Inner stride of input B: sizeof(floatxx)
        /*.nb11 =*/ nb11, 
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0, // Inner axis len of Output: n
        /*.ne1  =*/ ne1, // m
        /*.r2   =*/ r2, // channel_broadcast_ratio
        /*.r3   =*/ r3, // batch_broadcast_ratio
    };
```

When refactoring the params, we removed some redundancy in params and switch to outer-first axis order for strides.
