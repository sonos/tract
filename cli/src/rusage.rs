use std::time::Duration as StdDuration;
use std::time::Instant as StdInstant;

#[derive(Debug, Copy, Clone)]
pub struct Instant(StdInstant, StdDuration, StdDuration);

impl Instant {
    /// Returns the current instant.
    pub fn now() -> Instant {
        let usage = readings_probe::get_os_readings().unwrap();
        let elapsed_user = usage.user_time;
        let elapsed_sys = usage.system_time;

        Instant(StdInstant::now(), elapsed_user, elapsed_sys)
    }

    /// Returns the number of elapsed real seconds since the instant.
    pub fn elapsed_real(&self) -> StdDuration {
        self.0.elapsed()
    }

    /// Returns the number of elapsed user seconds since the instant.
    pub fn elapsed_user(&self) -> StdDuration {
        let usage = readings_probe::get_os_readings().unwrap();
        usage.user_time - self.1
    }

    /// Returns the number of elapsed system seconds since the instant.
    pub fn elapsed_sys(&self) -> StdDuration {
        let usage = readings_probe::get_os_readings().unwrap();
        usage.system_time - self.2
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Duration {
    pub total_real: StdDuration,
    pub total_user: StdDuration,
    pub total_sys: StdDuration,
}

impl Duration {
    /// Returns a measure from a given instant and iterations.
    pub fn since(start: &Instant) -> Duration {
        let total_real = start.elapsed_real();
        let total_user = start.elapsed_user();
        let total_sys = start.elapsed_sys();

        Duration { total_real, total_user, total_sys }
    }

    pub fn avg_real(&self) -> StdDuration {
        self.total_real
    }

    pub fn avg_user(&self) -> StdDuration {
        self.total_user
    }

    pub fn avg_sys(&self) -> StdDuration {
        self.total_sys
    }
}

impl std::ops::AddAssign for Duration {
    fn add_assign(&mut self, other: Duration) {
        *self = Duration {
            total_real: self.total_real + other.total_real,
            total_user: self.total_user + other.total_user,
            total_sys: self.total_sys + other.total_sys,
        };
    }
}

impl std::ops::SubAssign for Duration {
    fn sub_assign(&mut self, other: Duration) {
        *self = Duration {
            total_real: self.total_real.max(other.total_real) - other.total_real,
            total_user: self.total_user.max(other.total_user) - other.total_user,
            total_sys: self.total_sys.max(other.total_sys) - other.total_sys,
        };
    }
}

impl std::ops::MulAssign<f64> for Duration {
    fn mul_assign(&mut self, other: f64) {
        *self = Duration {
            total_real: StdDuration::from_secs_f64(self.total_real.as_secs_f64() * other),
            total_user: StdDuration::from_secs_f64(self.total_user.as_secs_f64() * other),
            total_sys: StdDuration::from_secs_f64(self.total_sys.as_secs_f64() * other),
        };
    }
}

impl std::ops::DivAssign<f64> for Duration {
    fn div_assign(&mut self, other: f64) {
        *self = Duration {
            total_real: StdDuration::from_secs_f64(self.total_real.as_secs_f64() / other),
            total_user: StdDuration::from_secs_f64(self.total_user.as_secs_f64() / other),
            total_sys: StdDuration::from_secs_f64(self.total_sys.as_secs_f64() / other),
        };
    }
}
