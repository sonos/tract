mod mfa_gemm;
mod mmm_tile_8x8;

pub use mfa_gemm::{metal_gemm, GemmPrecision};
pub use mmm_tile_8x8::{metal_mmm_tile_8x8, mmm_tile_8x8};

#[cfg(target_os = "ios")]
pub const METAL_FLASH_ATTENTION_LIB: &[u8] = include_bytes!("libMetalFlashAttention-ios.metallib");
#[cfg(target_os = "macos")]
pub const METAL_FLASH_ATTENTION_LIB: &[u8] =
    include_bytes!("libMetalFlashAttention-macos.metallib");
pub const MMM_TILE_8X8_METAL_SOURCE: &str = include_str!("mmm_tile_8x8.metal");

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LibraryContent<'a> {
    Data(&'a [u8]),
    Source(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    MfaLib,
    MmmTile8x8,
}

impl LibraryName {
    pub fn content(&self) -> LibraryContent<'static> {
        match self {
            Self::MfaLib => LibraryContent::Data(METAL_FLASH_ATTENTION_LIB),
            Self::MmmTile8x8 => LibraryContent::Source(MMM_TILE_8X8_METAL_SOURCE),
        }
    }
}
