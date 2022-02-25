use crate::frame::mmm::*;

MMMKernel!(f32, armvfpv2_mmm_f32_4x4; 4, 4; 4, 4; 0, 0; no_prefetch, true);
