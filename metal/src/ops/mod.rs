pub mod apply_rope;
pub mod fused_axis_op;
pub mod gemm;
pub mod scaled_masked_softmax;

pub use apply_rope::MetalApplyRope;
pub use fused_axis_op::MetalFusedAxisOp;
pub use gemm::MetalGemm;
pub use scaled_masked_softmax::MetalScaledMaskedSoftmax;

// Re-export shared GPU ops for backward compatibility
pub use tract_gpu::ops::cast::GpuCast as MetalCast;
pub use tract_gpu::ops::gelu_approximate::GpuGeluApproximate as MetalGeluApproximate;
pub use tract_gpu::ops::leaky_relu::GpuLeakyRelu as MetalLeakyRelu;
pub use tract_gpu::ops::rms_norm::GpuRmsNorm as MetalRmsNorm;
pub use tract_gpu::ops::rotate_half::GpuRotateHalf as MetalRotateHalf;
pub use tract_gpu::ops::softmax::GpuSoftmax as MetalSoftmax;
