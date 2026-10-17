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
