pub mod apply_rope;
pub mod dyn_kv_cache;
pub mod flash_sdpa;
pub mod scaled_masked_softmax;
pub mod sdpa;
pub mod streamed_sdpa;

// Re-export ops that moved to core
pub mod rms_norm {
    pub use tract_nnef::tract_core::ops::nn::RmsNorm;
    pub use tract_nnef::tract_core::ops::nn::rms_norm::*;
}
pub mod silu {
    pub use tract_nnef::tract_core::ops::nn::Silu;
    pub use tract_nnef::tract_core::ops::nn::silu::*;
}
pub mod gelu_approximate {
    pub use tract_nnef::tract_core::ops::nn::GeluApproximate;
    pub use tract_nnef::tract_core::ops::nn::gelu_approximate::*;
}

pub use apply_rope::{apply_rope_rule, rotate_half_rule};
pub use dyn_kv_cache::replace_kv_cache;
pub use scaled_masked_softmax::scaled_masked_softmax_rule;
pub use sdpa::fuse_kv_cache_broadcast_rule;
