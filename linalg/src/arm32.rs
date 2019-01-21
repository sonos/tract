mod armvfpv2;
mod armv7neon;

use Ops;
use crate::frame::PackedMatMul;

pub fn has_neon_cpuinfo() -> std::io::Result<bool>  {
    use std::fs;
    let cpu_info = fs::read_to_string("/proc/cpuinfo")?;
    let neon = cpu_info.split("\n").any(|line| line.starts_with("Features") && line.contains("neon"));
    Ok(neon)
}

pub fn has_neon() -> bool {
    has_neon_cpuinfo().unwrap_or(false)
}

pub fn plug(ops: &mut Ops) {
    if has_neon() {
        ops.smm = Box::new(|m, k, n| {
            log::info!("armv7neon activated for smm");
            Box::new(PackedMatMul::<armv7neon::SMatMul4x4, f32>::new(m, k, n))
        });
    } else {
        ops.smm = Box::new(|m, k, n| {
            log::info!("armvfpv2 activated for smm");
            Box::new(PackedMatMul::<armvfpv2::SMatMul4x4, f32>::new(m, k, n))
        });
    }
}
