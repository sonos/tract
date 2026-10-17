//! Dequant panel extractors so Q8_0 / K-quant weights reach the f32 and f16
//! matmul kernels. The body is portable; one registration per architecture so
//! each host's extractor set includes it.
use super::*;
use crate::pack::Packing;
use crate::panel_extractor;

unsafe fn panel_f32(fmt: &dyn BlockQuant, r: usize, input: *const u8, output: *mut u8, k: usize) {
    unsafe { super::opaque::dequant_panel_f32(fmt, r, input, k, output as *mut f32) }
}

unsafe fn panel_f16(fmt: &dyn BlockQuant, r: usize, input: *const u8, output: *mut u8, k: usize) {
    unsafe { super::opaque::dequant_panel_f16(fmt, r, input, k, output as *mut f16) }
}

macro_rules! apex_panel {
    ($arch:ident, $func:ident, $id:ident, $fmt:expr, $dt:ty) => {
        unsafe fn $func(input: *const u8, output: *mut u8, k: usize) {
            unsafe {
                if <$dt>::datum_type() == f32::datum_type() {
                    panel_f32(&$fmt, 4, input, output, k);
                } else {
                    panel_f16(&$fmt, 4, input, output, k);
                }
            }
        }
        panel_extractor!($arch; $func as $id(
            Box::new(PackedBlockQuantFormat::new(&$fmt, 4, 0, false)),
            <$dt>::packing(4)
        ));
    };
}

macro_rules! apex_panels_for {
    ($fmt:expr, $tag:ident) => {
        pastey::paste! {
            apex_panel!(arm, [<$tag _f32_arm>], [<$tag _r4_f32_arm>], $fmt, f32);
            apex_panel!(arm, [<$tag _f16_arm>], [<$tag _r4_f16_arm>], $fmt, f16);
            apex_panel!(aarch64, [<$tag _f32_aarch64>], [<$tag _r4_f32_aarch64>], $fmt, f32);
            apex_panel!(aarch64, [<$tag _f16_aarch64>], [<$tag _r4_f16_aarch64>], $fmt, f16);
            apex_panel!(x86_64, [<$tag _f32_x86>], [<$tag _r4_f32_x86>], $fmt, f32);
            apex_panel!(x86_64, [<$tag _f16_x86>], [<$tag _r4_f16_x86>], $fmt, f16);
            apex_panel!(riscv64, [<$tag _f32_riscv>], [<$tag _r4_f32_riscv>], $fmt, f32);
            apex_panel!(riscv64, [<$tag _f16_riscv>], [<$tag _r4_f16_riscv>], $fmt, f16);
            apex_panel!(wasm32, [<$tag _f32_wasm>], [<$tag _r4_f32_wasm>], $fmt, f32);
            apex_panel!(wasm32, [<$tag _f16_wasm>], [<$tag _r4_f16_wasm>], $fmt, f16);
        }
    };
}

apex_panels_for!(Q8_0, q80);
apex_panels_for!(Q3_K, q3k);
apex_panels_for!(Q4_K, q4k);
apex_panels_for!(Q5_K, q5k);
apex_panels_for!(Q6_K, q6k);
apex_panels_for!(IQ4_XS, iq4xs);
apex_panels_for!(IQ2_S, iq2s);

/// Wider panels for the aarch64 f32 and f16 tiles. `r` is that tile's M, and
/// the alignment is the one the kernel packed its A panel with (16 for NEON
/// and SVE, 128 for AMX and SME). The r=4 extractors above stay for the
/// portable kernel.
macro_rules! apex_wide {
    ($fmt:expr, $tag:ident, $dt:ty, $dtname:ident, $r:literal, $align:literal) => {
        pastey::paste! {
            unsafe fn [<$tag _ $dtname _r $r _a $align>](
                input: *const u8,
                output: *mut u8,
                k: usize,
            ) {
                unsafe {
                    if <$dt>::datum_type() == f32::datum_type() {
                        super::opaque::dequant_panel_f32(&$fmt, $r, input, k, output as *mut f32);
                    } else {
                        super::opaque::dequant_panel_f16(&$fmt, $r, input, k, output as *mut f16);
                    }
                }
            }
            panel_extractor!(aarch64; [<$tag _ $dtname _r $r _a $align>] as [<$tag _r $r _ $dtname _a $align>](
                Box::new(PackedBlockQuantFormat::new(&$fmt, $r, 0, false)),
                <$dt>::packing($r).align($align)
            ));
        }
    };
}

macro_rules! apex_host_tiles {
    ($fmt:expr, $tag:ident) => {
        apex_wide!($fmt, $tag, f32, f32, 8, 16);
        apex_wide!($fmt, $tag, f32, f32, 12, 16);
        apex_wide!($fmt, $tag, f32, f32, 16, 16);
        apex_wide!($fmt, $tag, f32, f32, 24, 16);
        apex_wide!($fmt, $tag, f32, f32, 32, 16);
        apex_wide!($fmt, $tag, f32, f32, 64, 16);
        apex_wide!($fmt, $tag, f32, f32, 32, 128);
        apex_wide!($fmt, $tag, f32, f32, 64, 128);
        apex_wide!($fmt, $tag, f16, f16, 8, 16);
        apex_wide!($fmt, $tag, f16, f16, 16, 16);
        apex_wide!($fmt, $tag, f16, f16, 32, 16);
        apex_wide!($fmt, $tag, f16, f16, 64, 16);
        apex_wide!($fmt, $tag, f16, f16, 128, 16);
        apex_wide!($fmt, $tag, f16, f16, 64, 128);
    };
}

apex_host_tiles!(Q8_0, q80);
apex_host_tiles!(Q3_K, q3k);
apex_host_tiles!(Q4_K, q4k);
apex_host_tiles!(Q5_K, q5k);
apex_host_tiles!(Q6_K, q6k);
apex_host_tiles!(IQ4_XS, iq4xs);
apex_host_tiles!(IQ2_S, iq2s);

/// Wider panels for the x86_64 f32 tiles. `r` is that tile's M, and the
/// alignment is the one the kernel packed its A panel with (256 for AVX and
/// FMA, 512 for AVX-512, 64 for AMX). There is no x86_64 f16 matmul tile.
/// Names carry the arch so they do not collide with the aarch64 extractors.
macro_rules! apex_x86_wide {
    ($fmt:expr, $tag:ident, $r:literal, $align:literal) => {
        pastey::paste! {
            unsafe fn [<$tag _x86_f32_r $r _a $align>](
                input: *const u8,
                output: *mut u8,
                k: usize,
            ) {
                unsafe {
                    super::opaque::dequant_panel_f32(&$fmt, $r, input, k, output as *mut f32);
                }
            }
            panel_extractor!(x86_64; [<$tag _x86_f32_r $r _a $align>] as [<$tag _x86_r $r _f32_a $align>](
                Box::new(PackedBlockQuantFormat::new(&$fmt, $r, 0, false)),
                f32::packing($r).align($align)
            ));
        }
    };
}

macro_rules! apex_x86_tiles {
    ($fmt:expr, $tag:ident) => {
        apex_x86_wide!($fmt, $tag, 8, 256);
        apex_x86_wide!($fmt, $tag, 16, 256);
        apex_x86_wide!($fmt, $tag, 24, 256);
        apex_x86_wide!($fmt, $tag, 32, 256);
        apex_x86_wide!($fmt, $tag, 40, 256);
        apex_x86_wide!($fmt, $tag, 64, 256);
        apex_x86_wide!($fmt, $tag, 16, 512);
        apex_x86_wide!($fmt, $tag, 32, 512);
        apex_x86_wide!($fmt, $tag, 48, 512);
        apex_x86_wide!($fmt, $tag, 64, 512);
        apex_x86_wide!($fmt, $tag, 80, 512);
        apex_x86_wide!($fmt, $tag, 128, 512);
        apex_x86_wide!($fmt, $tag, 16, 64);
    };
}

apex_x86_tiles!(Q8_0, q80);
apex_x86_tiles!(Q3_K, q3k);
apex_x86_tiles!(Q4_K, q4k);
apex_x86_tiles!(Q5_K, q5k);
apex_x86_tiles!(Q6_K, q6k);
apex_x86_tiles!(IQ4_XS, iq4xs);
apex_x86_tiles!(IQ2_S, iq2s);

/// Wider panels for the RISC-V vector tiles. `r` is that tile's M. Every RVV
/// f32 and f16 kernel packs its A panel at alignment 16. The names carry the
/// arch: those widths are already used by the aarch64 extractors.
macro_rules! apex_riscv_wide {
    ($fmt:expr, $tag:ident, $dt:ty, $dtname:ident, $r:literal) => {
        pastey::paste! {
            unsafe fn [<$tag _riscv_ $dtname _r $r _a16>](
                input: *const u8,
                output: *mut u8,
                k: usize,
            ) {
                unsafe {
                    if <$dt>::datum_type() == f32::datum_type() {
                        panel_f32(&$fmt, $r, input, output, k);
                    } else {
                        panel_f16(&$fmt, $r, input, output, k);
                    }
                }
            }
            panel_extractor!(riscv64; [<$tag _riscv_ $dtname _r $r _a16>] as [<$tag _riscv_r $r _ $dtname _a16>](
                Box::new(PackedBlockQuantFormat::new(&$fmt, $r, 0, false)),
                <$dt>::packing($r).align(16)
            ));
        }
    };
}

macro_rules! apex_riscv_tiles {
    ($fmt:expr, $tag:ident) => {
        apex_riscv_wide!($fmt, $tag, f32, f32, 8);
        apex_riscv_wide!($fmt, $tag, f32, f32, 16);
        apex_riscv_wide!($fmt, $tag, f32, f32, 32);
        apex_riscv_wide!($fmt, $tag, f32, f32, 64);
        apex_riscv_wide!($fmt, $tag, f16, f16, 16);
        apex_riscv_wide!($fmt, $tag, f16, f16, 32);
        apex_riscv_wide!($fmt, $tag, f16, f16, 64);
        apex_riscv_wide!($fmt, $tag, f16, f16, 128);
    };
}

apex_riscv_tiles!(Q8_0, q80);
apex_riscv_tiles!(Q3_K, q3k);
apex_riscv_tiles!(Q4_K, q4k);
apex_riscv_tiles!(Q5_K, q5k);
apex_riscv_tiles!(Q6_K, q6k);
apex_riscv_tiles!(IQ4_XS, iq4xs);
apex_riscv_tiles!(IQ2_S, iq2s);

/// Wider panels for the arm f32 tiles. `r` is that tile's M, and the
/// alignment is the one the kernel packed its A panel with (4 for VFP, 16
/// for NEON). There is no arm f16 matmul tile. The r=4 extractor above uses
/// the host vector alignment, which is not the VFP kernel's 4.
macro_rules! apex_arm_wide {
    ($fmt:expr, $tag:ident, $r:literal, $align:literal) => {
        pastey::paste! {
            unsafe fn [<$tag _arm_f32_r $r _a $align>](
                input: *const u8,
                output: *mut u8,
                k: usize,
            ) {
                unsafe {
                    panel_f32(&$fmt, $r, input, output, k);
                }
            }
            panel_extractor!(arm; [<$tag _arm_f32_r $r _a $align>] as [<$tag _arm_r $r _f32_a $align>](
                Box::new(PackedBlockQuantFormat::new(&$fmt, $r, 0, false)),
                f32::packing($r).align($align)
            ));
        }
    };
}

macro_rules! apex_arm_tiles {
    ($fmt:expr, $tag:ident) => {
        apex_arm_wide!($fmt, $tag, 4, 4);
        apex_arm_wide!($fmt, $tag, 8, 16);
        apex_arm_wide!($fmt, $tag, 32, 16);
    };
}

apex_arm_tiles!(Q8_0, q80);
apex_arm_tiles!(Q3_K, q3k);
apex_arm_tiles!(Q4_K, q4k);
apex_arm_tiles!(Q5_K, q5k);
apex_arm_tiles!(Q6_K, q6k);
apex_arm_tiles!(IQ4_XS, iq4xs);
apex_arm_tiles!(IQ2_S, iq2s);

/// Wider panels for the wasm32 simd128 f32 tiles. `r` is that tile's M, and
/// the alignment is the one the kernel packed its A panel with. There is no
/// wasm f16 matmul tile. The r=4 extractor above uses the host vector
/// alignment, which is not these kernels' alignment.
macro_rules! apex_wasm_wide {
    ($fmt:expr, $tag:ident, $r:literal, $align:literal) => {
        pastey::paste! {
            unsafe fn [<$tag _wasm_f32_r $r _a $align>](
                input: *const u8,
                output: *mut u8,
                k: usize,
            ) {
                unsafe {
                    panel_f32(&$fmt, $r, input, output, k);
                }
            }
            panel_extractor!(wasm32; [<$tag _wasm_f32_r $r _a $align>] as [<$tag _wasm_r $r _f32_a $align>](
                Box::new(PackedBlockQuantFormat::new(&$fmt, $r, 0, false)),
                f32::packing($r).align($align)
            ));
        }
    };
}

macro_rules! apex_wasm_tiles {
    ($fmt:expr, $tag:ident) => {
        apex_wasm_wide!($fmt, $tag, 4, 4);
        apex_wasm_wide!($fmt, $tag, 8, 8);
        apex_wasm_wide!($fmt, $tag, 16, 16);
        apex_wasm_wide!($fmt, $tag, 32, 32);
    };
}

apex_wasm_tiles!(Q8_0, q80);
apex_wasm_tiles!(Q3_K, q3k);
apex_wasm_tiles!(Q4_K, q4k);
apex_wasm_tiles!(Q5_K, q5k);
apex_wasm_tiles!(Q6_K, q6k);
apex_wasm_tiles!(IQ4_XS, iq4xs);
apex_wasm_tiles!(IQ2_S, iq2s);

#[cfg(test)]
mod host_tiles {
    use crate::frame::block_quant::{BlockQuant, Q8_0};
    use crate::frame::weights::WeightType;
    use crate::mmm::{MmmDispatch, Query, retain_best};
    use tract_data::prelude::*;

    fn kept(fmt: &dyn BlockQuant, activation: DatumType, n: usize) -> Vec<String> {
        let query = Query {
            weight: WeightType::BlockQuant(dyn_clone::clone_box(fmt)),
            activation,
            accumulators: tvec!(activation),
            store: Some(activation),
            allow_extractor: true,
            m: None,
            k: Some(fmt.block_len()),
            n: Some(n),
        };
        let mut suitable = MmmDispatch::native().suitable(&query);
        retain_best(&mut suitable);
        suitable.into_iter().map(|(mmm, _, _)| mmm.name().to_string()).collect()
    }

    #[test]
    fn aarch64_f32_q8_0_leaves_the_portable_kernel() {
        let names = kept(&Q8_0, f32::datum_type(), 1);
        assert!(!names.is_empty(), "no kernel for Q8_0");
        if cfg!(target_arch = "aarch64") {
            assert!(
                names.iter().all(|name| !name.starts_with("generic_")),
                "portable kernel still selected: {names:?}"
            );
        }
    }

    #[test]
    fn x86_64_f32_q8_0_leaves_the_portable_kernel() {
        let names = kept(&Q8_0, f32::datum_type(), 1);
        assert!(!names.is_empty(), "no kernel for Q8_0");
        if cfg!(target_arch = "x86_64") {
            assert!(
                names.iter().all(|name| !name.starts_with("generic_")),
                "portable kernel still selected: {names:?}"
            );
        }
    }

    #[test]
    fn riscv64_f32_q8_0_leaves_the_portable_kernel() {
        let names = kept(&Q8_0, f32::datum_type(), 1);
        assert!(!names.is_empty(), "no kernel for Q8_0");
        if cfg!(target_arch = "riscv64") {
            assert!(
                names.iter().all(|name| !name.starts_with("generic_")),
                "portable kernel still selected: {names:?}"
            );
        }
    }

    #[test]
    fn arm_f32_q8_0_leaves_the_portable_kernel() {
        let names = kept(&Q8_0, f32::datum_type(), 1);
        assert!(!names.is_empty(), "no kernel for Q8_0");
        if cfg!(target_arch = "arm") {
            assert!(
                names.iter().all(|name| !name.starts_with("generic_")),
                "portable kernel still selected: {names:?}"
            );
        }
    }

    #[test]
    fn wasm32_f32_q8_0_leaves_the_portable_kernel() {
        let names = kept(&Q8_0, f32::datum_type(), 1);
        assert!(!names.is_empty(), "no kernel for Q8_0");
        if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
            assert!(
                names.iter().all(|name| !name.starts_with("generic_")),
                "portable kernel still selected: {names:?}"
            );
        }
    }

    /// Each AVX, AVX-512, and AMX panel width is registered for every APEX block.
    /// The match is the packing the kernel stores, so it can be checked on any host.
    #[test]
    fn x86_64_f32_tiles_are_registered() {
        use crate::isa::Arch;
        use crate::mmm_routines::declared_extractors;
        use crate::pack::Packing;

        let tiles = [
            (8usize, 256usize),
            (16, 256),
            (24, 256),
            (32, 256),
            (40, 256),
            (64, 256),
            (16, 512),
            (32, 512),
            (48, 512),
            (64, 512),
            (80, 512),
            (128, 512),
            (16, 64),
        ];
        let formats = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ2_S"];
        let extractors: Vec<_> = declared_extractors()
            .filter(|extractor| extractor.target == Arch::X86_64)
            .map(|extractor| (extractor.make)())
            .collect();
        for &(r, align) in &tiles {
            let want = f32::packing(r).align(align);
            for format in formats {
                let from = format!("Packed{format}[{r}]");
                let found =
                    extractors.iter().any(|pe| pe.from.to_string() == from && pe.to == want);
                assert!(found, "no {from} extractor for align {align}");
            }
        }
    }

    /// simd128 f32 tiles pack 4, 8, 16, or 32 rows, and the alignment equals
    /// that width. The r=4 extractor uses the host vector alignment, so it
    /// does not reach these kernels.
    #[test]
    fn wasm32_f32_tiles_are_registered() {
        use crate::isa::Arch;
        use crate::mmm_routines::declared_extractors;
        use crate::pack::Packing;

        let tiles = [(4usize, 4usize), (8, 8), (16, 16), (32, 32)];
        let formats = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ2_S"];
        let extractors: Vec<_> = declared_extractors()
            .filter(|extractor| extractor.target == Arch::Wasm32Simd128)
            .map(|extractor| (extractor.make)())
            .collect();
        for &(r, align) in &tiles {
            let want = f32::packing(r).align(align);
            for format in formats {
                let from = format!("Packed{format}[{r}]");
                let found =
                    extractors.iter().any(|pe| pe.from.to_string() == from && pe.to == want);
                assert!(found, "no {from} extractor for align {align}");
            }
        }
    }

    /// VFP packs a 4-row panel at alignment 4. NEON packs 8- and 32-row panels
    /// at alignment 16. The r=4 extractor uses the host vector alignment, so
    /// it does not reach the VFP kernel.
    #[test]
    fn arm_f32_tiles_are_registered() {
        use crate::isa::Arch;
        use crate::mmm_routines::declared_extractors;
        use crate::pack::Packing;

        let tiles = [(4usize, 4usize), (8, 16), (32, 16)];
        let formats = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ2_S"];
        let extractors: Vec<_> = declared_extractors()
            .filter(|extractor| extractor.target == Arch::Arm)
            .map(|extractor| (extractor.make)())
            .collect();
        for &(r, align) in &tiles {
            let want = f32::packing(r).align(align);
            for format in formats {
                let from = format!("Packed{format}[{r}]");
                let found =
                    extractors.iter().any(|pe| pe.from.to_string() == from && pe.to == want);
                assert!(found, "no {from} extractor for align {align}");
            }
        }
    }

    /// Every RVV f32 and f16 tile packs A at alignment 16. The widths overlap
    /// the aarch64 extractors, so these registrations are the RISC-V ones.
    #[test]
    fn riscv64_tiles_are_registered() {
        use crate::isa::Arch;
        use crate::mmm_routines::declared_extractors;
        use crate::pack::Packing;

        let tiles = [
            (f32::datum_type(), 8usize),
            (f32::datum_type(), 16),
            (f32::datum_type(), 32),
            (f32::datum_type(), 64),
            (f16::datum_type(), 16),
            (f16::datum_type(), 32),
            (f16::datum_type(), 64),
            (f16::datum_type(), 128),
        ];
        let formats = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ2_S"];
        let extractors: Vec<_> = declared_extractors()
            .filter(|extractor| extractor.target == Arch::RiscV64)
            .map(|extractor| (extractor.make)())
            .collect();
        for (dt, r) in tiles {
            let want = if dt == f32::datum_type() {
                f32::packing(r).align(16)
            } else {
                f16::packing(r).align(16)
            };
            for format in formats {
                let from = format!("Packed{format}[{r}]");
                let found =
                    extractors.iter().any(|pe| pe.from.to_string() == from && pe.to == want);
                assert!(found, "no {from} extractor for {dt:?}");
            }
        }
    }

    /// SME's f32 matvec is 64 rows at alignment 128. SVE's f16 gemm is 8 rows
    /// at the default 16-byte packing. Neither pair is in the NEON or AMX list.
    #[test]
    fn aarch64_sme_and_sve_tiles_are_registered() {
        use crate::isa::Arch;
        use crate::mmm_routines::declared_extractors;
        use crate::pack::Packing;

        let tiles = [(f32::datum_type(), 64usize, 128usize), (f16::datum_type(), 8, 16)];
        let formats = ["Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ2_S"];
        let extractors: Vec<_> = declared_extractors()
            .filter(|extractor| extractor.target == Arch::Aarch64)
            .map(|extractor| (extractor.make)())
            .collect();
        for (dt, r, align) in tiles {
            let want = if dt == f32::datum_type() {
                f32::packing(r).align(align)
            } else {
                f16::packing(r).align(align)
            };
            for format in formats {
                let from = format!("Packed{format}[{r}]");
                let found =
                    extractors.iter().any(|pe| pe.from.to_string() == from && pe.to == want);
                assert!(found, "no {from} extractor for {dt:?} align {align}");
            }
        }
    }
}
