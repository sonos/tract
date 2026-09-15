//! Differential testing infrastructure for ACE software model vs Intel SDE.
//!
//! This module provides the framework for comparing the portable software model
//! in `isa.rs` against Intel SDE's ACE emulation. When SDE is available and
//! the ACE assembler probe passes, these tests will validate bit-exactness
//! between the two implementations.
//!
//! Currently, this infrastructure is prepared but not actively used since:
//! 1. Intel SDE 10.13.1 (with ACE support) is not available for macOS
//! 2. No assembler can encode ACE instructions yet
//! 3. The cfg(tract_ace) build probe fails on all current toolchains
//!
//! The framework is ready for when:
//! - SDE ACE emulation becomes accessible (via x86_64 Linux VM or other means)
//! - Binutils/LLVM gain ACE assembler support
//! - The build probe passes and cfg(tract_ace) is set

use super::format::{bf16_to_f32, fp8_e4m3_to_f32, mx_scale_decode};
use super::isa::{
    ACE_BF16_LANES, ACE_I8_BLOCK_BYTES, ACE_MX_BLOCK_ELEMS, ACE_MX_BLOCK_K, ACE_TILE_DIM,
    AceTileF32, AceTileI32, ace_top_mxfp8_block, ace_top_mxint8_block, ace_top2bf16ps,
    ace_top4bssd,
};

/// Placeholder for SDE-based differential testing.
///
/// When Intel SDE is available and ACE assembler support exists, this function
/// will:
/// 1. Compile small ACE test programs with the target mnemonics
/// 2. Run them under SDE with the `-future-ag` flag
/// 3. Compare SDE output against the software model results
/// 4. Report any discrepancies in numeric handling or tile operations
#[cfg(test)]
mod sde_differential {
    use super::*;

    /// Check if Intel SDE is available and supports ACE.
    /// This is a placeholder - actual implementation would check for SDE
    /// installation and ACE support via the `-future-ag` flag.
    fn sde_available() -> bool {
        // TODO(ace): Implement SDE availability check
        // - Check if `sde` command exists
        // - Verify SDE version >= 10.13.1 (ACE support)
        // - Test ACE availability with `sde -future-ag -- echo "test"`
        false
    }

    /// Run a small ACE test program under SDE and capture results.
    /// This is a placeholder - actual implementation would:
    /// - Write assembly test file with ACE mnemonics
    /// - Compile with ACE-enabled assembler
    /// - Run under SDE: `sde -future-ag -- ./test_program`
    /// - Parse output and return results
    fn run_sde_test(_mnemonic: &str, _inputs: &[u8]) -> Option<Vec<f32>> {
        // TODO(ace): Implement SDE test execution
        // - Write temporary .S file with ACE instruction
        // - Assemble with ACE-supporting toolchain
        // - Execute under SDE with appropriate flags
        // - Capture and parse output
        None
    }

    /// Differential test for ACE INT8 outer product.
    ///
    /// Compares `ace_top4bssd` results against SDE emulation.
    #[test]
    #[ignore] // Ignored until SDE is available
    fn differential_top4bssd() {
        if !sde_available() {
            return;
        }

        // Test data: deterministic 16x4 i8 matrices
        let mut a = [0i8; ACE_I8_BLOCK_BYTES];
        let mut b = [0i8; ACE_I8_BLOCK_BYTES];
        for i in 0..ACE_I8_BLOCK_BYTES {
            a[i] = ((i as i32 * 37 - 61) % 127) as i8;
            b[i] = ((i as i32 * 13 + 5) % 127 - 40) as i8;
        }

        // Software model result
        let mut tile_model = AceTileI32::zero();
        ace_top4bssd(&mut tile_model, &a, &b);

        // SDE result (placeholder)
        if let Some(tile_sde) = run_sde_test("top4bssd", &[]) {
            // Compare results
            for m in 0..ACE_TILE_DIM {
                for n in 0..ACE_TILE_DIM {
                    assert_eq!(
                        tile_model.e[m][n],
                        tile_sde[m * ACE_TILE_DIM + n] as i32,
                        "SDE mismatch at ({m},{n})"
                    );
                }
            }
        }
    }

    /// Differential test for ACE BF16 outer product.
    ///
    /// Compares `ace_top2bf16ps` results against SDE emulation.
    #[test]
    #[ignore] // Ignored until SDE is available
    fn differential_top2bf16ps() {
        if !sde_available() {
            return;
        }

        // Test data: 16x2 bf16 matrices
        let mut a = [0u16; ACE_BF16_LANES];
        let mut b = [0u16; ACE_BF16_LANES];
        for i in 0..ACE_BF16_LANES {
            let val = (i as f32 * 0.1).sin();
            a[i] = ((val as f32).to_bits() >> 16) as u16;
            b[i] = ((-val as f32).to_bits() >> 16) as u16;
        }

        // Software model result
        let mut tile_model = AceTileF32::zero();
        ace_top2bf16ps(&mut tile_model, &a, &b);

        // SDE result (placeholder)
        if let Some(tile_sde) = run_sde_test("top2bf16ps", &[]) {
            // Compare results
            for m in 0..ACE_TILE_DIM {
                for n in 0..ACE_TILE_DIM {
                    assert!(
                        (tile_model.e[m][n] - tile_sde[m * ACE_TILE_DIM + n]).abs() < 1e-6,
                        "SDE mismatch at ({m},{n})"
                    );
                }
            }
        }
    }

    /// Differential test for ACE MXFP8 block-scaled outer product.
    ///
    /// Compares `ace_top_mxfp8_block` results against SDE emulation.
    #[test]
    #[ignore] // Ignored until SDE is available
    fn differential_top_mxfp8_block() {
        if !sde_available() {
            return;
        }

        // Test data: 16x32 FP8 blocks with scales
        let a: [u8; ACE_MX_BLOCK_ELEMS] = std::array::from_fn(|i| (i % 120) as u8);
        let b: [u8; ACE_MX_BLOCK_ELEMS] = std::array::from_fn(|i| ((i * 3) % 120) as u8);
        let a_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|i| 127 + (i % 4) as u8);
        let b_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|i| 127 - (i % 3) as u8);

        // Software model result
        let mut tile_model = AceTileF32::zero();
        ace_top_mxfp8_block(&mut tile_model, &a, &a_scale, &b, &b_scale);

        // SDE result (placeholder)
        if let Some(tile_sde) = run_sde_test("top_mxfp8_block", &[]) {
            // Compare results
            for m in 0..ACE_TILE_DIM {
                for n in 0..ACE_TILE_DIM {
                    assert!(
                        (tile_model.e[m][n] - tile_sde[m * ACE_TILE_DIM + n]).abs() < 1e-4,
                        "SDE mismatch at ({m},{n})"
                    );
                }
            }
        }
    }

    /// Differential test for ACE MXINT8 block-scaled outer product.
    ///
    /// Compares `ace_top_mxint8_block` results against SDE emulation.
    #[test]
    #[ignore] // Ignored until SDE is available
    fn differential_top_mxint8_block() {
        if !sde_available() {
            return;
        }

        // Test data: 16x32 INT8 blocks with scales
        let a: [i8; ACE_MX_BLOCK_ELEMS] = std::array::from_fn(|i| ((i % 23) - 11) as i8);
        let b: [i8; ACE_MX_BLOCK_ELEMS] = std::array::from_fn(|i| ((i * 2 % 19) - 9) as i8);
        let a_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|i| 120 + (i % 8) as u8);
        let b_scale: [u8; ACE_TILE_DIM] = std::array::from_fn(|i| 125 + (i % 5) as u8);

        // Software model result
        let mut tile_model = AceTileF32::zero();
        ace_top_mxint8_block(&mut tile_model, &a, &a_scale, &b, &b_scale);

        // SDE result (placeholder)
        if let Some(tile_sde) = run_sde_test("top_mxint8_block", &[]) {
            // Compare results
            for m in 0..ACE_TILE_DIM {
                for n in 0..ACE_TILE_DIM {
                    assert!(
                        (tile_model.e[m][n] - tile_sde[m * ACE_TILE_DIM + n]).abs() < 1e-4,
                        "SDE mismatch at ({m},{n})"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod validation_infrastructure {
    use super::*;

    /// Validate that the differential testing infrastructure is properly structured.
    /// This test runs unconditionally to ensure the framework is ready for SDE.
    #[test]
    fn sde_infrastructure_ready() {
        // Verify constants match expectations
        assert_eq!(ACE_TILE_DIM, 16);
        assert_eq!(ACE_I8_BLOCK_BYTES, 64);
        assert_eq!(ACE_BF16_LANES, 32);
        assert_eq!(ACE_MX_BLOCK_K, 32);
        assert_eq!(ACE_MX_BLOCK_ELEMS, 512);

        // Verify helper functions work correctly
        assert_eq!(mx_scale_decode(127), 1.0);
        assert_eq!(mx_scale_decode(128), 2.0);
        assert!(mx_scale_decode(0xFF).is_nan());

        // Verify software model functions are callable
        let mut tile = AceTileI32::zero();
        let a = [1i8; ACE_I8_BLOCK_BYTES];
        let b = [2i8; ACE_I8_BLOCK_BYTES];
        ace_top4bssd(&mut tile, &a, &b);
        // Should have accumulated 16x16x4 = 1024 MACs of 1*2 = 2, total 2048
        for row in &tile.e {
            for &val in row {
                assert_eq!(val, 2048);
            }
        }
    }
}
