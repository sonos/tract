mod arm64simd;
pub use arm64simd::*;

use crate::Ops;

use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMul;
use crate::frame::MatMatMulImpl;

use tract_data::internal::DimLike;

fn is_cortex_a53() -> std::io::Result<bool> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let a53 =
        cpu_info.split("\n").any(|line| line.starts_with("CPU part") && line.contains("0xd03"));
    Ok(a53)
}

pub fn plug(ops: &mut Ops) {
    if is_cortex_a53().unwrap_or(false) {
        log::info!("arm64simd activated for smmm (cortex A53)");
        ops.mmv_f32 = Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulF32x64x1A53, f32>::new()));
        ops.mmm_f32 = Box::new(|m, _, n| {
            best_of(
                m,
                n,
                &[
                    Box::new(MatMatMulImpl::<MatMatMulF32x12x8A53, f32>::new()),
                    Box::new(MatMatMulImpl::<MatMatMulF32x8x8A53, f32>::new()),
                    Box::new(MatMatMulImpl::<MatMatMulF32x16x4A53, f32>::new()),
                ],
            )
        })
    } else {
        log::info!("arm64simd activated for smmm (generic)");
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1, f32>::new()));
        ops.mmm_f32 = Box::new(|m, _, n| {
            best_of(
                m,
                n,
                &[
                    Box::new(MatMatMulImpl::<MatMatMulF32x12x8, f32>::new()),
                    Box::new(MatMatMulImpl::<MatMatMulF32x8x8, f32>::new()),
                    Box::new(MatMatMulImpl::<MatMatMulF32x16x4, f32>::new()),
                ],
            )
        })
    }

    ops.qmmm_i8_i8 = Box::new(|_, _, _| Box::new(MatMatMulImpl::<MatMatMulI8x8x8, i32>::new()));
    ops.qmmv_i8_i8 = Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulI8x64x1, i32>::new()));
    ops.qmmm_i8_i32 =
        Box::new(|_, _, _| Box::new(MatMatMulImpl::<MatMatMulI8xI32x8x8, i32>::new()));
    ops.qmmv_i8_i32 = Box::new(|_, _| Box::new(MatMatMulImpl::<MatMatMulI8xI32x64x1, i32>::new()));

    ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<TanhF32x4n, f32>::new()));
}

fn best_of(
    m: Option<usize>,
    n: Option<usize>,
    kernels: &[Box<dyn MatMatMul>],
) -> Box<dyn MatMatMul> {
    if let (Some(m), Some(n)) = (m, n) {
//        eprintln!("{}x{}", m, n);
        let a53 = is_cortex_a53().unwrap_or(false);
        let k = kernels
            .iter()
            .min_by_key(|k| {
                let rows = m.div_ceil(k.mr());
                let cols = n.div_ceil(k.nr());
                let tiles = rows * cols;
                //        let cost = 10 + k.mr() * k.nr() + 4 * (k.nr() + k.mr());
                let cost = match (a53, k.mr(), k.nr()) {
                    (true, 16, 4) => 15703,
                    (true, 12, 8) => 18770,
                    (true, 8, 8) => 14152,
                    (false, 16, 4) => 86726,
                    (false, 12, 8) => 12863,
                    (false, 8, 8) => 87252,
                    _ => 1,
                };
                let score = tiles * cost;
                /*
                eprintln!(
                    "  k:{:2}x{} tiles:{:3} cost:{:3} => {:5}",
                    k.mr(),
                    k.nr(),
                    tiles,
                    cost,
                    tiles * cost
                );
                */
                score
            })
            .unwrap()
            .clone();
        k
    } else {
        kernels.iter().max_by_key(|k| k.mr() * k.nr()).unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best(m: usize, n: usize) -> (usize, usize) {
        let k = best_of(
            Some(m),
            Some(n),
            &[
                Box::new(MatMatMulImpl::<MatMatMulF32x12x8, f32>::new()),
                Box::new(MatMatMulImpl::<MatMatMulF32x8x8, f32>::new()),
                Box::new(MatMatMulImpl::<MatMatMulF32x16x4, f32>::new()),
            ],
        );
        (k.mr(), k.nr())
    }

    #[test]
    #[ignore]
    fn kernel_choice() {
        assert_eq!(best(128, 40), (12, 8)); // hey_snips_v1
        assert_eq!(best(32, 24), (8, 8)); // hey_snips_v3
        assert_eq!(best(200, 12), (12, 8)); // 15M
        assert_eq!(best(768, 12), (12, 8)); // 15M
        assert_eq!(best(768, 4), (16, 4)); // 15M
        assert_eq!(best(2000, 4), (16, 4)); // 15M
    }
}
