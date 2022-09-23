mod arm64simd;
pub mod cortex_a53;
mod cortex_a55;
//mod cortex_a72;
//mod cortex_a73;
pub use arm64simd::*;

use crate::Ops;

use crate::frame::element_wise::ElementWiseKer;
use crate::frame::mmm::kernel::MatMatMulKer;

lazy_static::lazy_static! {
    static ref KIND: Kind = Kind::choose();
}

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
        .split("\n")
        .filter(|line| line.starts_with("CPU part"))
        .map(|line| line.split_whitespace().last().unwrap_or(""))
        .max();
    Ok(max.unwrap_or("").to_string())
}

pub fn has_fp16() -> bool {
    KIND.has_fp16()
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
    fn has_fp16(&self) -> bool {
        match self {
            Kind::CortexA55 | Kind::CortexA75 | Kind::AppleM => true,
            _ => false,
        }
    }

    fn choose() -> Kind {
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
        } else {
            if cfg!(target_os = "macos") {
                Kind::AppleM
            } else {
                let part = if let Ok(part) = std::env::var("TRACT_CPU_AARCH64_OVERRIDE_CPU_PART") {
                    log::info!(
                        "CPU part forced with TRACT_CPU_AARCH64_OVERRIDE_CPU_PART: {}",
                        part
                    );
                    part
                } else {
                    let part = max_cpuid().unwrap_or("0x00".to_string());
                    log::info!("CPU part auto detected: {}", part);
                    part
                };
                match &*part {
                    PART_A53 => Kind::CortexA53,
                    PART_A55 => Kind::CortexA55,
                    PART_A72 => Kind::CortexA72,
                    PART_A73 => Kind::CortexA73,
                    PART_A75 => Kind::CortexA75,
                    _ => Kind::Generic,
                }
            }
        };
        log::info!("CPU optimisation: {:?}", kind);
        kind
    }
}

pub fn plug(ops: &mut Ops) {
    let impls = vec![
        arm64simd_mmm_f32_12x8_gen::mmm(),
        arm64simd_mmm_f32_12x8_a53::mmm(),
        arm64simd_mmm_f32_12x8_a55::mmm(),
        arm64simd_mmm_f32_8x8_gen::mmm(),
        arm64simd_mmm_f32_8x8_a53::mmm(),
        arm64simd_mmm_f32_8x8_a55::mmm(),
        arm64simd_mmm_f32_16x4_gen::mmm(),
        arm64simd_mmm_f32_16x4_a53::mmm(),
        arm64simd_mmm_f32_16x4_a55::mmm(),
        arm64simd_mmm_f32_24x4_gen::mmm(),
        arm64simd_mmm_f32_24x4_a53::mmm(),
        arm64simd_mmm_f32_24x4_a55::mmm(),
        crate::generic::mmm::generic_f32_4x4::mmm(),
    ];
    ops.mmm_f32_impls = impls.clone();
    ops.qmmm_i32 = Box::new(|_, _, _| arm64simd_mmm_i32_8x8::mmm());
    ops.qmmv_i32 = Box::new(|_, _| arm64simd_mmm_i32_64x1::mmm());
    ops.mmv_f32 = match *KIND {
        Kind::CortexA53 => Box::new(|_, _| arm64simd_mmm_f32_64x1_a53::mmm()),
        Kind::CortexA55 => Box::new(|_, _| arm64simd_mmm_f32_64x1_a55::mmm()),
        _ => Box::new(|_, _| arm64simd_mmm_f32_64x1_gen::mmm()),
    };
    let model = match *KIND {
        Kind::CortexA53 => Some(cortex_a53::model()),
        Kind::CortexA55 => Some(cortex_a55::model()),
        _ => None,
    };
    if let Some(model) = model {
        ops.mmm_f32 = Box::new(move |m, k, n| model.pick(&impls, m, k, n));
    }
    #[cfg(feature = "no_fp16")]
    if has_fp16() {
        log::warn!("This is a build with fp16 disabled, while your platform CPU seems to support it.");
    }
    #[cfg(not(feature = "no_fp16"))]
    if has_fp16() {
        if *KIND == Kind::CortexA55 {
            ops.mmm_f16 = Box::new(|_, _, n| {
                use tract_data::internal::DimLike;
                if n.unwrap_or(1024).divceil(4) * 4 < n.unwrap_or(1024).divceil(8) * 8 {
                    arm64fp16_mmm_f16_32x4_a55::mmm()
                } else {
                    arm64fp16_mmm_f16_16x8_a55::mmm()
                }
            });
            ops.mmv_f16 = Box::new(|_, _| arm64fp16_mmm_f16_128x1_a55::mmm());
        } else {
            ops.mmm_f16 = Box::new(|_, _, n| {
                use tract_data::internal::DimLike;
                if n.unwrap_or(1024).divceil(4) * 4 < n.unwrap_or(1024).divceil(8) * 8 {
                    arm64fp16_mmm_f16_32x4_gen::mmm()
                } else {
                    arm64fp16_mmm_f16_16x8_gen::mmm()
                }
            });
            ops.mmv_f16 = Box::new(|_, _| arm64fp16_mmm_f16_128x1_gen::mmm());
        }
    }
    ops.sigmoid_f32 = Box::new(|| arm64simd_sigmoid_f32_4n::ew());
    ops.tanh_f32 = Box::new(|| arm64simd_tanh_f32_4n::ew());
    #[cfg(not(feature = "no_fp16"))]
    if has_fp16() {
        ops.tanh_f16 = Box::new(|| arm64fp16_tanh_f16_8n::ew());
        ops.sigmoid_f16 = Box::new(|| arm64fp16_sigmoid_f16_8n::ew());
    }
}
