mod apply_rope;
mod conv;
mod flash_attn;
mod fused_axis_op;
mod gemm;
mod quant_q81;
mod scaled_masked_softmax;
pub use apply_rope::CudaApplyRope;
pub use conv::{CudaConv, wire_cuda_conv};
pub use flash_attn::CudaFlashAttention;
pub use fused_axis_op::CudaFusedAxisOp;
pub use gemm::CudaGgmlGemm;
pub use quant_q81::{CudaGgmlQuantQ81, GgmlQuantQ81Fact};
pub use scaled_masked_softmax::CudaScaledMaskedSoftmax;

// Re-export shared GPU ops for backward compatibility
pub use tract_gpu::ops::cast::GpuCast as CudaCast;
pub use tract_gpu::ops::gelu_approximate::GpuGeluApproximate as CudaGeluApproximate;
pub use tract_gpu::ops::leaky_relu::GpuLeakyRelu as CudaLeakyRelu;
pub use tract_gpu::ops::rms_norm::GpuRmsNorm as CudaRmsNorm;
pub use tract_gpu::ops::rotate_half::GpuRotateHalf as CudaRotateHalf;
pub use tract_gpu::ops::softmax::GpuSoftmax as CudaSoftmax;
