mod mat_vec;
mod mfa_gemm;
mod mmm_tile_8x8;

pub use mat_vec::{mat_vec, mat_vec_with_slice, metal_mat_vec};
pub use mfa_gemm::{
    dispatch_metal_mfa_gemm, mfa_dispatch_gemm_with_slice, mfa_gemm, mfa_gemm_with_slice,
    GemmPrecision,
};
pub use mmm_tile_8x8::{metal_mmm_tile_8x8, mmm_tile_8x8};
