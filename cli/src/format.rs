use crate::rusage::Duration;
use ansi_term::Colour::*;

#[allow(unused_imports)]
use tract_core::itertools::Itertools;

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg_oneline(measure: Duration) -> String {
    format!(
        "Real: {} User: {} Sys: {}",
        White.bold().paint(format!("{:.3} ms/i", measure.avg_real().as_secs_f64() * 1e3)),
        White.bold().paint(format!("{:.3} ms/i", measure.avg_user().as_secs_f64() * 1e3)),
        White.bold().paint(format!("{:.3} ms/i", measure.avg_sys().as_secs_f64() * 1e3)),
    )
}

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg_multiline(measure: Duration) -> String {
    format!(
        "Real: {}\nUser: {}\nSys: {}",
        White.bold().paint(format!("{:.3} ms/i", measure.avg_real().as_secs_f64() * 1e3)),
        White.bold().paint(format!("{:.3} ms/i", measure.avg_user().as_secs_f64() * 1e3)),
        White.bold().paint(format!("{:.3} ms/i", measure.avg_sys().as_secs_f64() * 1e3)),
    )
}

/// Format a rusage::Duration showing avgtime in ms, with percentage to a global
/// one.
pub fn dur_avg_oneline_ratio(measure: Duration, global: Duration) -> String {
    format!(
        "Real: {} {} User: {} {} Sys: {} {}",
        White.bold().paint(format!("{:7.3} ms/i", measure.avg_real().as_secs_f64() * 1e3)),
        Yellow.bold().paint(format!(
            "{:2.0}%",
            measure.avg_real().as_secs_f64() / global.avg_real().as_secs_f64() * 100.
        )),
        Yellow.bold().paint(format!("{:7.3} ms/i", measure.avg_user().as_secs_f64() * 1e3)),
        Yellow.bold().paint(format!(
            "{:2.0}%",
            measure.avg_user().as_secs_f64() / global.avg_user().as_secs_f64() * 100.
        )),
        Yellow.bold().paint(format!("{:7.3} ms/i", measure.avg_sys().as_secs_f64() * 1e3)),
        Yellow.bold().paint(format!(
            "{:2.0}%",
            measure.avg_sys().as_secs_f64() / global.avg_sys().as_secs_f64() * 100.
        )),
    )
}
