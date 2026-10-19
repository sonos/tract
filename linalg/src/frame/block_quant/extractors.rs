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

/// Wider panels for the aarch64 f32, f16, and Apple AMX tiles. `r` is that
/// tile's M, and the alignment is the one the kernel packed its A panel with
/// (16 for NEON, 128 for AMX). The r=4 extractors above stay for the portable
/// kernel.
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
}
