#![allow(clippy::excessive_precision)]
#[cfg(any(target_os = "macos", all(target_os = "ios", feature = "apple-amx-ios")))]
mod apple_amx;
mod arm64simd;
pub mod cortex_a53;
mod cortex_a55;
//mod cortex_a72;
//mod cortex_a73;
pub use arm64simd::*;

#[cfg(not(feature = "no_fp16"))]
pub mod arm64fp16;
#[cfg(not(feature = "no_fp16"))]
pub use arm64fp16::*;

use crate::f16;
use crate::{BinOp, DatumType, LinalgRegistry, Ops};

use crate::frame::by_scalar::ByScalarKer;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::frame::unicast::UnicastKer;

// https://en.wikipedia.org/wiki/Comparison_of_ARMv8-A_cores
const PART_A53: &str = "0xd03";
const PART_A55: &str = "0xd05";
#[allow(dead_code)]
const PART_A72: &str = "0xd08";
#[allow(dead_code)]
const PART_A73: &str = "0xd09";
#[allow(dead_code)]
const PART_A75: &str = "0xd0a";

fn max_cpuid() -> std::io::Result<String> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let max = cpu_info
        .lines()
        .filter(|line| line.starts_with("CPU part"))
        .map(|line| line.split_whitespace().last().unwrap_or(""))
        .max();
    Ok(max.unwrap_or("").to_string())
}

lazy_static::lazy_static! {
    static ref KIND: Kind = Kind::choose();

    static ref CPU_FEATURES: Vec<String> = {
        #[cfg(test)] crate::setup_test_logger();
        let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") else {
            log::warn!("Could not read /proc/cpuinfo. CPU Features detection may be impaired.");
            return vec!();
        };
        if let Some(line) = cpu_info
            .lines()
                .find(|line| line.starts_with("Features")) {
                    line.split_once(':').unwrap().1.split_whitespace().map(|s| s.to_string()).collect()
                } else {
                    log::warn!("Could not find \"Features  :\" lines in /proc/cpuinfo. CPU Features detection may be impaired.");
                    vec!()
        }
    };

    static ref HAS_FP16: bool = {
        CPU_FEATURES.iter().any(|s| &**s == "asimdhp")
    };
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn apple_get_syscall(key: &str) -> String {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::null_mut;

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut isize,
            newp: *mut c_void,
            newlen: isize,
        );
    }

    unsafe {
        let mut len: isize = 0;
        let name = CString::new(key).unwrap();
        sysctlbyname(name.as_ptr(), null_mut(), &mut len, null_mut(), 0);
        let mut buf = vec![0u8; len as _];
        sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as _, &mut len, null_mut(), 0);
        CStr::from_bytes_with_nul(&buf).unwrap().to_string_lossy().into_owned()
    }
}

#[cfg(target_os = "macos")]
pub fn has_amx() -> bool {
    !apple_get_syscall("machdep.cpu.brand_string").contains("(Virtual)")
}

#[cfg(target_os = "ios")]
lazy_static::lazy_static! {
    static ref IPHONE_MODEL_MAJOR:Option<usize> = {
        let version = apple_get_syscall("hw.machine");
        let Some((major, _)) = version.trim_start_matches("iPhone").split_once(",") else { return None };
        major.parse::<usize>().ok()
    };
}

#[cfg(all(target_os = "ios", feature = "apple-amx-ios"))]
fn has_amx() -> bool {
    // iPhone12,1 is the one branded "iPhone 11", with Apple A13 bionic, first CPU featuring amx
    IPHONE_MODEL_MAJOR.map(|it| it >= 12).unwrap_or(false)
}

#[inline]
#[cfg(target_os = "ios")]
pub fn has_fp16() -> bool {
    // iPhone10,1 is the one branded "iPhone 8", with Apple A11 bionic, first CPU featuring fp16
    IPHONE_MODEL_MAJOR.map(|it| it >= 10).unwrap_or(false)
}

#[inline]
#[cfg(not(target_os = "ios"))]
pub fn has_fp16() -> bool {
    cfg!(target_os = "macos")
        || cfg!(feature_cpu = "fp16")
        || *KIND == Kind::CortexA55
        || *KIND == Kind::CortexA75
        || *HAS_FP16
}

#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn add_f16(a: f16, b: f16) -> f16 {
    let result: u16;
    std::arch::asm!(
        "fadd {0:h}, {1:h}, {2:h}",
        lateout(vreg) result,
        in(vreg) a.to_bits(),
        in(vreg) b.to_bits(),
        options(pure, nomem, nostack, preserves_flags));
    f16::from_bits(result)
}

#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn mul_f16(a: f16, b: f16) -> f16 {
    let result: u16;
    std::arch::asm!(
        "fmul {0:h}, {1:h}, {2:h}",
        lateout(vreg) result,
        in(vreg) a.to_bits(),
        in(vreg) b.to_bits(),
        options(pure, nomem, nostack, preserves_flags));
    f16::from_bits(result)
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Kind {
    Generic,
    AppleM,
    CortexA53,
    CortexA55,
    CortexA72,
    CortexA73,
    CortexA75,
}

impl Kind {
    fn choose() -> Kind {
        #[cfg(test)]
        crate::setup_test_logger();
        let kind = if let Ok(kind) = std::env::var("TRACT_CPU_AARCH64_KIND") {
            log::info!("CPU kind forced with TRACT_CPU_AARCH64_KIND: {}", kind);
            let kind = kind.to_lowercase();
            if kind.contains("a53") {
                Kind::CortexA53
            } else if kind.contains("a55") {
                Kind::CortexA55
            } else if kind.contains("a72") {
                Kind::CortexA72
            } else if kind.contains("a73") {
                Kind::CortexA73
            } else if kind.contains("a75") {
                Kind::CortexA75
            } else if kind.contains("applem") {
                Kind::AppleM
            } else {
                Kind::Generic
            }
        } else if cfg!(target_os = "macos") {
            Kind::AppleM
        } else {
            let part = if let Ok(part) = std::env::var("TRACT_CPU_AARCH64_OVERRIDE_CPU_PART") {
                log::info!("CPU part forced with TRACT_CPU_AARCH64_OVERRIDE_CPU_PART: {}", part);
                part
            } else if cfg!(target_os = "linux") {
                let part = max_cpuid().unwrap_or_else(|_| "0x00".to_string());
                log::info!("CPU part auto detected: {}", part);
                part
            } else {
                log::info!("Unknown CPU part");
                "0x00".to_string()
            };
            match &*part {
                PART_A53 => Kind::CortexA53,
                PART_A55 => Kind::CortexA55,
                PART_A72 => Kind::CortexA72,
                PART_A73 => Kind::CortexA73,
                PART_A75 => Kind::CortexA75,
                _ => Kind::Generic,
            }
        };
        log::info!("CPU optimisation: {:?}", kind);
        kind
    }
}

pub(crate) fn register_all_unicast(registry: &mut LinalgRegistry) {
    registry
        .insert((BinOp::Mul, DatumType::F32), Box::new(|| arm64simd_unicast_mul_f32_16n::bin()));
    registry
        .insert((BinOp::Mul, DatumType::F16), Box::new(|| arm64fp16_unicast_mul_f16_32n::bin()));
    registry
        .insert((BinOp::Add, DatumType::F32), Box::new(|| arm64simd_unicast_add_f32_16n::bin()));
    registry
        .insert((BinOp::Add, DatumType::F16), Box::new(|| arm64fp16_unicast_add_f16_32n::bin()));
    registry
        .insert((BinOp::Sub, DatumType::F32), Box::new(|| arm64simd_unicast_sub_f32_16n::bin()));
    registry
        .insert((BinOp::Sub, DatumType::F16), Box::new(|| arm64fp16_unicast_sub_f16_32n::bin()));
    registry
        .insert((BinOp::SubF, DatumType::F32), Box::new(|| arm64simd_unicast_subf_f32_16n::bin()));
    registry
        .insert((BinOp::SubF, DatumType::F16), Box::new(|| arm64fp16_unicast_subf_f16_32n::bin()));
    registry
        .insert((BinOp::Min, DatumType::F32), Box::new(|| arm64simd_unicast_min_f32_16n::bin()));
    registry
        .insert((BinOp::Min, DatumType::F16), Box::new(|| arm64fp16_unicast_min_f16_32n::bin()));
    registry
        .insert((BinOp::Max, DatumType::F32), Box::new(|| arm64simd_unicast_max_f32_16n::bin()));
    registry
        .insert((BinOp::Max, DatumType::F16), Box::new(|| arm64fp16_unicast_max_f16_32n::bin()));
}

pub(crate) fn register_all_by_scalar(registry: &mut LinalgRegistry) {
    registry
        .insert((BinOp::Mul, DatumType::F32), Box::new(|| arm64simd_mul_by_scalar_f32_16n::bin()));
    registry
        .insert((BinOp::Mul, DatumType::F16), Box::new(|| arm64fp16_mul_by_scalar_f16_32n::bin()));
    registry
        .insert((BinOp::Add, DatumType::F32), Box::new(|| arm64simd_add_by_scalar_f32_16n::bin()));
    registry
        .insert((BinOp::Add, DatumType::F16), Box::new(|| arm64fp16_add_by_scalar_f16_32n::bin()));
    registry
        .insert((BinOp::Sub, DatumType::F32), Box::new(|| arm64simd_sub_by_scalar_f32_16n::bin()));
    registry
        .insert((BinOp::Sub, DatumType::F16), Box::new(|| arm64fp16_sub_by_scalar_f16_32n::bin()));
    registry.insert(
        (BinOp::SubF, DatumType::F32),
        Box::new(|| arm64simd_subf_by_scalar_f32_16n::bin()),
    );
    registry.insert(
        (BinOp::SubF, DatumType::F16),
        Box::new(|| arm64fp16_subf_by_scalar_f16_32n::bin()),
    );
    registry
        .insert((BinOp::Min, DatumType::F32), Box::new(|| arm64simd_min_by_scalar_f32_16n::bin()));
    registry
        .insert((BinOp::Min, DatumType::F16), Box::new(|| arm64fp16_min_by_scalar_f16_32n::bin()));
    registry
        .insert((BinOp::Max, DatumType::F32), Box::new(|| arm64simd_max_by_scalar_f32_16n::bin()));
    registry
        .insert((BinOp::Max, DatumType::F16), Box::new(|| arm64fp16_max_by_scalar_f16_32n::bin()));
}

pub fn plug(ops: &mut Ops) {
    arm64simd::plug(ops);

    #[cfg(not(feature = "no_fp16"))]
    if has_fp16() {
        ops.mmm_impls.push(arm64fp16_mmm_f16_16x8_a55.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_16x8_gen.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_32x4_a55.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_32x4_gen.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_32x6_gen.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_128x1_a55.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_128x1_gen.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_64x1_gen.mmm());
        ops.mmm_impls.push(arm64fp16_mmm_f16_64x3_gen.mmm());
        arm64fp16::plug(ops);
    }

    ops.qmmm_i32 = Box::new(|_, _, _| arm64simd_mmm_i32_8x8.mmm());
    ops.qmmv_i32 = Box::new(|_, _| arm64simd_mmm_i32_64x1.mmm());
    ops.mmv_f32 = match *KIND {
        Kind::CortexA53 => Box::new(|_, _| arm64simd_mmm_f32_64x1_a53.mmm()),
        Kind::CortexA55 => Box::new(|_, _| arm64simd_mmm_f32_64x1_a55.mmm()),
        _ => Box::new(|_, _| arm64simd_mmm_f32_64x1_gen.mmm()),
    };
    let model = match *KIND {
        Kind::CortexA53 => Some(cortex_a53::model()),
        Kind::CortexA55 => Some(cortex_a55::model()),
        _ => None,
    };
    let impls = ops.mmm_impls.clone();
    ops.mmm_f32 = if let Some(model) = model {
        Box::new(move |m, k, n| model.pick(&impls, m, k, n))
    } else {
        Box::new(move |_, _, n| {
            if n.unwrap_or(8) < 8 {
                arm64simd_mmm_f32_16x4_gen.mmm()
            } else {
                arm64simd_mmm_f32_8x8_gen.mmm()
            }
        })
    };
    #[cfg(feature = "no_fp16")]
    if has_fp16() {
        log::warn!(
            "This is a build with fp16 disabled, while your platform CPU seems to support it."
        );
    }
    #[cfg(not(feature = "no_fp16"))]
    if has_fp16() {
        if *KIND == Kind::CortexA55 {
            log::info!("Cortex-A55 mmm_f16 and mmv_f16 activated");
            ops.mmm_f16 = Box::new(|_, _, n| {
                use tract_data::internal::DimLike;
                if n.unwrap_or(1024).divceil(4) * 4 < n.unwrap_or(1024).divceil(8) * 8 {
                    arm64fp16_mmm_f16_32x4_a55.mmm()
                } else {
                    arm64fp16_mmm_f16_16x8_a55.mmm()
                }
            });
            ops.mmv_f16 = Box::new(|_, _| arm64fp16_mmm_f16_128x1_a55.mmm());
        } else {
            log::info!("ARMv8.2 mmm_f16 and mmv_f16 activated");
            ops.mmm_f16 = Box::new(|_, _, n| {
                use tract_data::internal::DimLike;
                if n.unwrap_or(1024).divceil(4) * 4 < n.unwrap_or(1024).divceil(8) * 8 {
                    arm64fp16_mmm_f16_32x4_gen.mmm()
                } else {
                    arm64fp16_mmm_f16_16x8_gen.mmm()
                }
            });
            ops.mmv_f16 = Box::new(|_, _| arm64fp16_mmm_f16_128x1_gen.mmm());
        }
    }
    ops.leaky_relu_f32 = Box::new(|| arm64simd_leaky_relu_f32_8n::ew());
    ops.sigmoid_f32 = Box::new(|| arm64simd_sigmoid_f32_4n::ew());
    ops.tanh_f32 = Box::new(|| arm64simd_tanh_f32_4n::ew());
    ops.max_f32 = Box::new(|| arm64simd_max_f32_16n::red());
    ops.sum_f32 = Box::new(|| arm64simd_sum_f32_16n::red());
    ops.mul_by_scalar_f32 = Box::new(|| arm64simd_mul_by_scalar_f32_16n::ew());
    ops.softmax2_fastcompact_f32 = Box::new(|| arm64simd_softmax2_fastcompact_f32_16n::red());
    #[cfg(not(feature = "no_fp16"))]
    if has_fp16() {
        log::info!("ARMv8.2 tanh_f16 and sigmoid_f16 activated");
        ops.leaky_relu_f16 = Box::new(|| arm64fp16_leaky_relu_f16_16n::ew());
        ops.tanh_f16 = Box::new(|| arm64fp16_tanh_f16_8n::ew());
        ops.sigmoid_f16 = Box::new(|| arm64fp16_sigmoid_f16_8n::ew());
        ops.max_f16 = Box::new(|| arm64fp16_max_f16_32n::red());
        ops.sum_f16 = Box::new(|| arm64fp16_sum_f16_32n::red());
        ops.mul_by_scalar_f16 = Box::new(|| arm64fp16_mul_by_scalar_f16_32n::ew());
    } else {
        log::info!("No native fp16 support");
    }
    #[cfg(any(target_os = "macos", all(target_os = "ios", feature = "apple-amx-ios")))]
    {
        apple_amx::plug(ops);
    }
}
