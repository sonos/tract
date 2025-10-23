pub mod rewire_sdpa;
pub mod rewire_syncs;
pub mod rms_norm;

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}
