mod arm64simd;
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

#[derive(Debug)]
enum Kind {
    Generic,
    CortexA53Like,
}

impl Kind {
    fn choose() -> Kind {
        let kind = if let Ok(kind) = std::env::var("TRACT_CPU_AARCH64_KIND") {
            log::info!("CPU kind forced with TRACT_CPU_AARCH64_KIND: {}", kind);
            if kind.to_lowercase().contains("a53") {
                Kind::CortexA53Like
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
            if [PART_A55, PART_A53].contains(&&*part) {
                Kind::CortexA53Like
            } else {
                Kind::Generic
            }
        };
        log::info!("CPU optimisation: {:?}", kind);
        kind
    }
}

pub fn plug(ops: &mut Ops) {
    ops.mmm_f32_impls.push(MatMatMulF32x12x8A53::mmm());
    ops.mmm_f32_impls.push(MatMatMulF32x8x8A53::mmm());
    ops.mmm_f32_impls.push(MatMatMulF32x16x4A53::mmm());
    ops.mmm_f32_impls.push(MatMatMulF32x12x8::mmm());
    ops.mmm_f32_impls.push(MatMatMulF32x8x8::mmm());
    ops.mmm_f32_impls.push(MatMatMulF32x16x4::mmm());
    match *KIND {
        Kind::CortexA53Like => {
            ops.mmv_f32 =
                Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulF32x64x1A53, f32>::new()));
        }
        Kind::Generic => {
            ops.mmv_f32 =
                Box::new(
                    |_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1, f32>::new()),
                );
        }
    }
    ops.qmmm_i32 = Box::new(|_, _, _| Box::new(MatMatMulImpl::<MatMatMulI32x8x8, i32>::new()));
    ops.qmmv_i32 = Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulI32x64x1, i32>::new()));
    ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<TanhF32x4n, f32>::new()));
}

