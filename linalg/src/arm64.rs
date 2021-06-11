mod arm64simd;

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
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x64x1A53, f32>::new()));
        ops.mmm_f32 = Box::new(|m, _, n| {
            best_of(
                m,
                n,
                &[
                    Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x12x8A53, f32>::new()),
                    Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8A53, f32>::new()),
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
                    Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x12x8, f32>::new()),
                    Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8, f32>::new()),
                ],
            )
        })
    }

    ops.qmmm_i8_i8 =
        Box::new(|_, _, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8x8x8, i32>::new()));
    ops.qmmv_i8_i8 =
        Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8x64x1, i32>::new()));
    ops.qmmm_i8_i32 =
        Box::new(|_, _, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8xI32x8x8, i32>::new()));
    ops.qmmv_i8_i32 =
        Box::new(|_, _| Box::new(MatMatMulImpl::<arm64simd::MatMatMulI8xI32x64x1, i32>::new()));

    ops.sigmoid_f32 =
        Box::new(|| Box::new(ElementWiseImpl::<arm64simd::SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<arm64simd::TanhF32x4n, f32>::new()));
}

fn best_of(
    m: Option<usize>,
    n: Option<usize>,
    kernels: &[Box<dyn MatMatMul>],
) -> Box<dyn MatMatMul> {
    if let (Some(m), Some(n)) = (m, n) {
        let k = kernels
            .iter()
            .min_by_key(|k| (m.div_ceil(k.mr()) * n.div_ceil(k.nr())) * (10 + k.mr() * k.nr()))
            .unwrap()
            .clone();
        k
    } else {
        kernels.iter().max_by_key(|k| k.mr() * k.nr()).unwrap().clone()
    }
}
