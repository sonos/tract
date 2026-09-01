#![cfg(test)]
use tract_core::internal::*;

mod optimized {
    use super::*;

    pub fn optimized() -> &'static DefaultRuntime {
        &DefaultRuntime
    }
    include!(concat!(env!("OUT_DIR"), "/tests/optimized.rs"));
}
