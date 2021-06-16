mod arm64simd;
use arm64simd::*;

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
        let k = kernels
            .iter()
            .min_by_key(|k| {
                let rows = m.div_ceil(k.mr());
                let cols = n.div_ceil(k.nr());
                (rows * cols) * (50 + k.mr() * k.nr() + 10 * (k.nr() + k.mr()))
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
    fn kernel_choice() {
        assert_eq!(best(128, 40), (12, 8)); // hey_snips_v1 layer_0_2
        assert_eq!(best(200, 12), (12, 8)); // 15M h_new
        assert_eq!(best(768, 12), (12, 8)); // 15M h_new
        assert_eq!(best(768, 4), (16, 4)); // 15M h_new
        assert_eq!(best(2000, 4), (16, 4)); // 15M output
    }
}
