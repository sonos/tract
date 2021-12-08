mod arm64simd;
pub use arm64simd::*;

use crate::Ops;

use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMul;
use crate::frame::MatMatMulImpl;

use tract_data::internal::DimLike;

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
    match *KIND {
        Kind::CortexA53Like => {
            ops.mmv_f32 =
                Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulF32x64x1A53, f32>::new()));
            ops.mmm_f32 = Box::new(|m, k, n| {
                best_of(
                    m,
                    k,
                    n,
                    &[
                        Box::new(MatMatMulImpl::<MatMatMulF32x12x8A53, f32>::new()),
                        Box::new(MatMatMulImpl::<MatMatMulF32x8x8A53, f32>::new()),
                        Box::new(MatMatMulImpl::<MatMatMulF32x16x4A53, f32>::new()),
                    ],
                )
            })
        }
        Kind::Generic => {
            ops.mmv_f32 =
                Box::new(
                    |_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1, f32>::new()),
                );
            ops.mmm_f32 = Box::new(|m, k, n| {
                best_of(
                    m,
                    k,
                    n,
                    &[
                        Box::new(MatMatMulImpl::<MatMatMulF32x12x8, f32>::new()),
                        Box::new(MatMatMulImpl::<MatMatMulF32x8x8, f32>::new()),
                        Box::new(MatMatMulImpl::<MatMatMulF32x16x4, f32>::new()),
                    ],
                )
            });
        }
    }
    ops.qmmm_i32 = Box::new(|_, _, _| Box::new(MatMatMulImpl::<MatMatMulI8x8x8, i32>::new()));
    ops.qmmv_i32 = Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulI8x64x1, i32>::new()));
    ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<TanhF32x4n, f32>::new()));
}

fn best_of(
    m: Option<usize>,
    k: Option<usize>,
    n: Option<usize>,
    kernels: &[Box<dyn MatMatMul>],
) -> Box<dyn MatMatMul> {
    if let (Some(m), Some(k), Some(n)) = (m, k, n) {
        // eprintln!("{}x{}x{}", m, k, n);
        let ker = kernels
            .iter()
            .min_by_key(|ker| {
                let rows = m.divceil(ker.mr());
                let cols = n.divceil(ker.nr());
                let tiles = rows * cols;
                let cost = match *KIND {
                    Kind::CortexA53Like => match (ker.kernel_name(), ker.mr(), ker.nr()) {
                        ("arm64simd (generic)", 16, 4) => 31043 * k + 937000,
                        ("arm64simd (generic)", 12, 8) => 37448 * k + 990000,
                        ("arm64simd (generic)", 8, 8) => 28228 * k + 990000,
                        ("arm64simd (cortex A53)", 16, 4) => 32239 * k + 990000,
                        ("arm64simd (cortex A53)", 12, 8) => 36823 * k + 990000,
                        ("arm64simd (cortex A53)", 8, 8) => 28333 * k + 990000,
                        _ => panic!("uncosted kernel"),
                    },
                    Kind::Generic => match (ker.kernel_name(), ker.mr(), ker.nr()) {
                        ("arm64simd (generic)", 16, 4) => 1500 * k + 83000,
                        ("arm64simd (generic)", 12, 8) => 2083 * k + 83000,
                        ("arm64simd (generic)", 8, 8) => 1458 * k + 83000,
                        ("arm64simd (cortex A53)", 16, 4) => 4792 * k + 83000,
                        ("arm64simd (cortex A53)", 12, 8) => 5625 * k + 83000,
                        ("arm64simd (cortex A53)", 8, 8) => 4834 * k + 41000,
                        _ => panic!("uncosted kernel"),
                    },
                };
                let indirect_tiles =
                    (rows * ker.mr() > m) as usize * cols + (cols * ker.nr() > n) as usize * rows;
                let score = tiles * cost + indirect_tiles * ker.nr() * ker.mr() * 25000;
                /*
                eprintln!(
                " {} {}x{} cost: {} tiles: {} indirect_tiles: {} score: {}",
                ker.kernel_name(),
                ker.mr(),
                ker.nr(),
                cost,
                tiles,
                indirect_tiles,
                score,
                );
                */
                score
            })
            .unwrap()
            .clone();
        ker
    } else {
        kernels.iter().max_by_key(|k| k.mr() * k.nr()).unwrap().clone()
    }
}
