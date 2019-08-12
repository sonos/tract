use std::{env, fs};
mod armv7neon;
mod armvfpv2;
use crate::frame::TileOp;

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
        log::info!("armv7neon activated for stile");
        ops.stile = Box::new(|m, k, n| {
            Box::new(TileOp::<armv7neon::STile8x4, f32>::new(m, k, n))
        });
    } else {
        log::info!("armvfpv2 activated for stile");
        ops.stile = Box::new(|m, k, n| {
            Box::new(TileOp::<armvfpv2::STile4x4, f32>::new(m, k, n))
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
