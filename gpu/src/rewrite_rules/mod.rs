pub mod rewire_sdpa;
pub mod rewire_syncs;

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}
