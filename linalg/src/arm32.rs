use std::{env, fs};
mod armv7neon;
mod armvfpv2;
use crate::frame::MatMatMulImpl;
use crate::frame::QMatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

use crate::Ops;

fn has_neon_cpuinfo() -> std::io::Result<bool> {
    let cpu_info = fs::read_to_string("/proc/cpuinfo")?;
    let neon =
        cpu_info.split("\n").any(|line| line.starts_with("Features") && line.contains("neon"));
    Ok(neon)
}

fn has_neon() -> bool {
    if let Ok(v) = env::var("TRACT_CPU_ARM32_NEON") {
        return v == "true";
    }
    has_neon_cpuinfo().unwrap_or(false)
}

pub fn plug(ops: &mut Ops) {
    if has_neon() {
        log::info!("armv7neon activated (smmm, ssigmoid), stanh)");
        ops.smmm = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armv7neon::SMatMatMul8x4, f32, f32, f32, f32>::new(m, k, n))
        });
        ops.qmmm_i8_i8 = Box::new(|m, k, n| {
            Box::new(QMatMatMulImpl::from(MatMatMulImpl::<
                armv7neon::I8MatMatMul8x4,
                i8,
                i8,
                i8,
                i32,
            >::new(m, k, n)))
        });
        ops.ssigmoid = Box::new(|| Box::new(SigmoidImpl::<armv7neon::SSigmoid4, f32>::new()));
        ops.stanh = Box::new(|| Box::new(TanhImpl::<armv7neon::STanh4, f32>::new()));
    } else {
        log::info!("armvfpv2 activated for smmm");
        ops.smmm = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<armvfpv2::SMatMatMul4x4, f32, f32, f32, f32>::new(m, k, n))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn may_have_neon() {
        println!("Has neon ? {:?}", has_neon());
        if let Ok(neon) = env::var("TRACT_CPU_EXPECT_ARM32_NEON") {
            assert_eq!(neon == "true", has_neon());
        }
    }
}
