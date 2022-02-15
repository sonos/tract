mod arm64simd;
mod cortex_a53;
mod cortex_a55;
mod cortex_a72;
mod cortex_a73;
pub use arm64simd::*;

use crate::Ops;

use crate::frame::mmm::kernel::MatMatMulKer;
use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMulImpl;

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

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Kind {
    Generic,
    CortexA53,
    CortexA55,
    CortexA72,
    CortexA73,
    CortexA75,
}

impl Kind {
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
            } else {
                Kind::Generic
            }
        } else {
            let part = if let Ok(part) = std::env::var("TRACT_CPU_AARCH64_OVERRIDE_CPU_PART") {
                log::info!("CPU part forced with TRACT_CPU_AARCH64_OVERRIDE_CPU_PART: {}", part);
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
        };
        log::info!("CPU optimisation: {:?}", kind);
        kind
    }
}

pub fn plug(ops: &mut Ops) {
    let impls = vec![
        arm64simd_mmm_f32_12x8_a53::mmm(),
        arm64simd_mmm_f32_12x8_gen::mmm(),
        arm64simd_mmm_f32_8x8_a53::mmm(),
        arm64simd_mmm_f32_8x8_gen::mmm(),
        arm64simd_mmm_f32_16x4_a53::mmm(),
        arm64simd_mmm_f32_16x4_gen::mmm(),
        arm64simd_mmm_f32_24x4_a53::mmm(),
        arm64simd_mmm_f32_24x4_gen::mmm(),
    ];
    match *KIND {
        Kind::CortexA53 => {
            ops.mmv_f32 =
                Box::new(|_, _| arm64simd_mmm_f32_64x1_a53::mmm());
        }
        _ => {
            ops.mmv_f32 =
                Box::new(|_, _| arm64simd_mmm_f32_64x1_gen::mmm());
        }
    }
    ops.qmmm_i32 = Box::new(|_, _, _| arm64simd_mmm_i32_8x8::mmm());
    ops.qmmv_i32 = Box::new(|_, _| arm64simd_mmm_i32_64x1::mmm());
    ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<TanhF32x4n, f32>::new()));
    /*
    match *KIND {
    Kind::CortexA53 => ops.set_cost_models(cortex_a53::models()),
    Kind::CortexA55 => ops.set_cost_models(cortex_a55::models()),
    Kind::CortexA72 => ops.set_cost_models(cortex_a72::models()),
    Kind::CortexA73 => ops.set_cost_models(cortex_a73::models()),
    _ => ops.set_cost_models(cortex_a53::models()),
    }
    */
    if *KIND == Kind::CortexA55 {
        ops.mmm_f32 = Box::new(|_, _, n| {
            if n.unwrap_or(8) < 8 {
                arm64simd_mmm_f32_16x4_a55::mmm()
            } else {
                arm64simd_mmm_f32_8x8_a55::mmm()
            }
        });
    }
}
