//use crate::frame::element_wise::ElementWiseKer;
use crate::frame::mmm::*;
// #[cfg(not(feature="no_fp16"))]
// use tract_data::half::f16;

MMMKernel!(f32, apple_amx_mmm_f32_32x32; 32, 32; 128, 128; 1, 1; no_prefetch, true,
           can_fuse: |f| !matches!(f, &FusedSpec::LeakyRelu(_))
);
