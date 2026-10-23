//! Scheduling limits for routed Metal execution.

tract_core::tract_data::declare_knob!(
    TRACT_METAL_COMMIT_EVERY_N_DISPATCHES,
    usize,
    0,
    "Metal: submit after this many command-buffer acquisitions; zero disables cadence submission. Resolved when a stream is created."
);

tract_core::tract_data::declare_knob!(
    TRACT_METAL_MAX_IN_FLIGHT,
    usize,
    8,
    "Metal: maximum outstanding command buffers before waiting for the oldest; clamped to at least one. Resolved when a stream is created."
);

pub(crate) const MOE_COMMIT_MIN_ROUTES: usize = 64;
pub(crate) const MOE_GROUPED_MIN_ROUTES: usize = 64;
