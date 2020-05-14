use ansi_term::Colour::*;
use std::time::Duration;

#[allow(unused_imports)]
use tract_core::itertools::Itertools;

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg(measure: Duration) -> String {
    format!("Wallclock time: {}", White.bold().paint(format!("{:.3} ms/i", measure.as_secs_f64() * 1e3)),)
}

/// Format a rusage::Duration showing avgtime in ms, with percentage to a global
/// one.
pub fn dur_avg_ratio(measure: Duration, global: Duration) -> String {
    format!(
        "Wallclock time: {} {}",
        White.bold().paint(format!("{:7.3} ms/i", measure.as_secs_f64() * 1e3)),
        Yellow.bold().paint(format!("{:2.0}%", measure.as_secs_f64() / global.as_secs_f64() * 100.)),
    )
}
