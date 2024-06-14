pub mod binary;
pub mod element_wise;
pub mod gemm;
pub mod sync;

pub use binary::MetalBinOp;
pub use element_wise::MetalElementWiseOp;
pub use gemm::MetalGemm;
pub use sync::MetalSync;
