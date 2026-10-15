//! Process-local scheduling limits for routed Metal execution.
use std::sync::OnceLock;

pub(crate) struct MetalTuning {
    pub commit_every_n_dispatches: usize,
    pub max_command_buffers_in_flight: usize,
    pub moe_commit_min_routes: usize,
    pub moe_grouped_min_routes: usize,
}

pub(crate) fn tuning() -> &'static MetalTuning {
    static TUNING: OnceLock<MetalTuning> = OnceLock::new();
    TUNING.get_or_init(|| {
        let value = |key: &str, default| {
            std::env::var(key).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
        };
        MetalTuning {
            commit_every_n_dispatches: value("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES", 0),
            max_command_buffers_in_flight: value("TRACT_METAL_MAX_IN_FLIGHT", 8).max(1),
            moe_commit_min_routes: value("TRACT_METAL_MOE_COMMIT_MIN_ROUTES", 64).min(64),
            moe_grouped_min_routes: 64,
        }
    })
}
