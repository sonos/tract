//! Scheduling limits for Metal execution.

tract_core::tract_data::declare_knob!(
    TRACT_METAL_COMMIT_EVERY_N_DISPATCHES,
    usize,
    0,
    "Metal: submit on the next command-buffer acquisition after this many encoded kernel launches; zero disables dispatch-based submission. Resolved when a stream is created."
);

tract_core::tract_data::declare_knob!(
    TRACT_METAL_MAX_IN_FLIGHT,
    usize,
    8,
    "Metal: maximum outstanding command buffers before waiting for the oldest; clamped to at least one. Resolved when a stream is created."
);

pub(crate) const MOE_COMMIT_MIN_ROUTES: usize = 64;
pub(crate) const MOE_GROUPED_MIN_ROUTES: usize = 64;
