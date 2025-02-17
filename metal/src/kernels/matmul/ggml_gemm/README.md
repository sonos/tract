### Kernel list

Outputs are always F32. Accumulation is F32. Weights are in first position.
- kernel_mul_mv: (f16_f16, f32_f32, f16_f32)
- kernel_mul_mv_1row (f16_f32)
- kernel_mul_mv_l4 (f16_f32)

- kernel_mul_mm (f32_f32, f16_f32)

### Tensor layout

Channel is a batch-like dimension. It is NOT as in the matmul operational convention (==m or k)

- A: [a_batch, a_channel, m, k]
- B: [b_batch, b_channel, n, k]
- Out: [b_batch, b_channel, n, m]

**We actually swap A and B when calling the kernel to have the untransposed output!**

### Kernel prototype

Matvec Kernel params. Matmul params are a subset of this struct
```
ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00, // Inner axis len of input A: k
        /*.ne01 =*/ ne01, // m
        /*.ne02 =*/ ne02, // a_channel
        /*.nb00 =*/ nb00, // Inner stride of input A
        /*.nb01 =*/ nb01, // k
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10, // Inner axis len of input B: k
        /*.ne11 =*/ ne11, // n
        /*.ne12 =*/ ne12, //b_channel
        /*.nb10 =*/ nb10, // Inner stride of input B
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0, // Inner axis len of Output: m
        /*.ne1  =*/ ne1, // n
        /*.r2   =*/ r2, // channel_broadcast_ratio
        /*.r3   =*/ r3, // batch_broadcast_ratio
    };
```

r2 and r3 are currently always 1!
These are for multiplying multiple Bs with a single A.
Kernel should support it if frame tries to call them with values != 1.

```
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]])
```

```
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
```
