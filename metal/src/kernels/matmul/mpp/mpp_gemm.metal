#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// GEMM through Metal 4's matmul2d tensor op, which the OS lowers to the GPU's
// matrix hardware. A group of four simdgroups computes a 64x64 tile of C,
// accumulating in f32 and overwriting C. The tensors are built over the bound
// buffers: Metal tensors list extents innermost first, so a row-major M x K
// matrix is a (K, M) tensor, and a transposed operand swaps its two extents.

#define MPP_TILE 64

// Mirrors tract_metal::kernels::matmul::mpp::MppGemmParams.
struct MppGemmParams {
    int m;
    int n;
    int k;
    int batch_stride_a;
    int batch_stride_b;
    int batch_stride_c;
};

template <typename T, bool TA, bool TB>
[[kernel]] void mpp_gemm(device T *a [[buffer(0)]],
                         device T *b [[buffer(1)]],
                         device T *c [[buffer(2)]],
                         constant MppGemmParams &p [[buffer(3)]],
                         uint3 tgid [[threadgroup_position_in_grid]]) {
    a += (int)tgid.z * p.batch_stride_a;
    b += (int)tgid.z * p.batch_stride_b;
    c += (int)tgid.z * p.batch_stride_c;
    using extents = dextents<int32_t, 2>;
    auto A = tensor<device T, extents, tensor_inline>(
        a, TA ? extents(p.m, p.k) : extents(p.k, p.m));
    auto B = tensor<device T, extents, tensor_inline>(
        b, TB ? extents(p.k, p.n) : extents(p.n, p.k));
    auto C = tensor<device T, extents, tensor_inline>(c, extents(p.n, p.m));
    constexpr auto desc = matmul2d_descriptor(
        MPP_TILE, MPP_TILE, static_cast<int>(dynamic_extent), TA, TB);
    matmul2d<desc, execution_simdgroups<4>> op;
    const int row = tgid.y * MPP_TILE;
    const int col = tgid.x * MPP_TILE;
    auto tile_a = TA ? A.slice(row, 0) : A.slice(0, row);
    auto tile_b = TB ? B.slice(0, col) : B.slice(col, 0);
    auto tile_c = C.slice(col, row);
    op.run(tile_a, tile_b, tile_c);
}

#define INSTANTIATE_MPP_GEMM(tname, type, ta, tb, tn)                          \
    template [[host_name("mpp_gemm_" #tname "_" #tn)]] [[kernel]] void         \
    mpp_gemm<type, ta, tb>(device type *, device type *, device type *,       \
                           constant MppGemmParams &, uint3);

#define INSTANTIATE_MPP_GEMM_ALL(tname, type)                                  \
    INSTANTIATE_MPP_GEMM(tname, type, false, false, nn)                        \
    INSTANTIATE_MPP_GEMM(tname, type, false, true, nt)                         \
    INSTANTIATE_MPP_GEMM(tname, type, true, false, tn)                         \
    INSTANTIATE_MPP_GEMM(tname, type, true, true, tt)

INSTANTIATE_MPP_GEMM_ALL(f32, float)
INSTANTIATE_MPP_GEMM_ALL(f16, half)
