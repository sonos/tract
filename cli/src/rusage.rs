#![allow(dead_code)]

use libc::{getrusage, rusage, RUSAGE_SELF};
use std::time::Instant as StdInstant;

use crate::CliResult;

#[derive(Debug, Copy, Clone)]
pub struct Instant(StdInstant, f64, f64);

impl Instant {
    /// Returns the current instant.
    pub fn now() -> Instant {
        let elapsed_user = get_usage().unwrap().user_time;
        let elapsed_sys = get_usage().unwrap().system_time;

        Instant(StdInstant::now(), elapsed_user, elapsed_sys)
    }

    /// Returns the number of elapsed real seconds since the instant.
    pub fn elapsed_real(&self) -> f64 {
        let duration = self.0.elapsed();
        duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1.0e-9
    }

    /// Returns the number of elapsed user seconds since the instant.
    pub fn elapsed_user(&self) -> f64 {
        get_usage().unwrap().user_time - self.1
    }

    /// Returns the number of elapsed system seconds since the instant.
    pub fn elapsed_sys(&self) -> f64 {
        get_usage().unwrap().system_time - self.2
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Duration {
    pub total_real: f64,
    pub total_user: f64,
    pub total_sys: f64,
}

impl Duration {
    /// Returns an empty measure.
    pub fn new() -> Duration {
        Duration { ..Default::default() }
    }

    /// Returns a measure from a given instant and iterations.
    pub fn since(start: &Instant) -> Duration {
        let total_real = start.elapsed_real();
        let total_user = start.elapsed_user();
        let total_sys = start.elapsed_sys();

        Duration { total_real, total_user, total_sys }
    }

    pub fn avg_real(&self) -> f64 {
        self.total_real
    }

    pub fn avg_user(&self) -> f64 {
        self.total_user
    }

    pub fn avg_sys(&self) -> f64 {
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
            total_real: self.total_real - other.total_real,
            total_user: self.total_user - other.total_user,
            total_sys: self.total_sys - other.total_sys,
        };
    }
}

impl std::ops::MulAssign<f64> for Duration {
    fn mul_assign(&mut self, other: f64) {
        *self = Duration {
            total_real: self.total_real * other,
            total_user: self.total_user * other,
            total_sys: self.total_sys * other,
        };
    }
}

impl std::ops::DivAssign<f64> for Duration {
    fn div_assign(&mut self, other: f64) {
        *self = Duration {
            total_real: self.total_real / other,
            total_user: self.total_user / other,
            total_sys: self.total_sys / other,
        };
    }
}

#[derive(Debug)]
pub struct ResourceUsage {
    pub virtual_size: u64,
    pub resident_size: u64,
    pub resident_size_max: u64,
    pub user_time: f64,
    pub system_time: f64,
    pub minor_fault: u64,
    pub major_fault: u64,
}

#[cfg(target_os = "macos")]
mod darwin {
    use libc::*;
    #[repr(C)]
    pub struct BasicTaskInfo {
        pub virtual_size: u64,
        pub resident_size: u64,
        pub resident_size_max: u64,
        pub user_time: timeval,
        pub system_time: timeval,
        pub policy: c_int,
        pub suspend_count: c_uint,
    }

    impl BasicTaskInfo {
        pub fn empty() -> BasicTaskInfo {
            BasicTaskInfo {
                virtual_size: 0,
                resident_size: 0,
                resident_size_max: 0,
                user_time: timeval { tv_sec: 0, tv_usec: 0 },
                system_time: timeval { tv_sec: 0, tv_usec: 0 },
                policy: 0,
                suspend_count: 0,
            }
        }
    }
    mod ffi {
        use libc::*;
        extern "C" {
            pub fn mach_task_self() -> c_uint;
            pub fn task_info(
                task: c_uint,
                flavor: c_int,
                task_info: *mut super::BasicTaskInfo,
                count: *mut c_uint,
            ) -> c_uint;
        }
    }
    pub fn task_self() -> c_uint {
        unsafe { ffi::mach_task_self() }
    }
    pub fn task_info() -> BasicTaskInfo {
        let mut info = BasicTaskInfo::empty();
        let mut count: c_uint =
            (::std::mem::size_of::<BasicTaskInfo>() / ::std::mem::size_of::<c_uint>()) as c_uint;
        unsafe {
            ffi::task_info(task_self(), 20, &mut info, &mut count);
        }
        info
    }
}

#[cfg(target_os = "macos")]
pub fn get_usage() -> CliResult<ResourceUsage> {
    let info = darwin::task_info();
    let rusage = get_rusage();
    Ok(ResourceUsage {
        virtual_size: info.virtual_size,
        resident_size: info.resident_size,
        resident_size_max: info.resident_size_max,
        user_time: rusage.ru_utime.tv_sec as f64 + rusage.ru_utime.tv_usec as f64 / 1_000_000f64,
        system_time: rusage.ru_stime.tv_sec as f64 + rusage.ru_stime.tv_usec as f64 / 1_000_000f64,
        minor_fault: rusage.ru_minflt as u64,
        major_fault: rusage.ru_majflt as u64,
    })
}

#[cfg(target_os = "linux")]
pub fn get_usage() -> CliResult<ResourceUsage> {
    use std::fs::File;
    use std::io::Read;
    let mut proc_stat = String::new();
    let _ = r#try!(r#try!(File::open("/proc/self/stat")).read_to_string(&mut proc_stat));
    let mut tokens = proc_stat.split(" ");
    let rusage = get_rusage();
    Ok(ResourceUsage {
        virtual_size: tokens.nth(22).unwrap().parse().unwrap_or(0),
        resident_size: 4 * 1024 * tokens.next().unwrap().parse().unwrap_or(0),
        resident_size_max: 1024 * rusage.ru_maxrss as u64,
        user_time: rusage.ru_utime.tv_sec as f64 + rusage.ru_utime.tv_usec as f64 / 1_000_000f64,
        system_time: rusage.ru_stime.tv_sec as f64 + rusage.ru_stime.tv_usec as f64 / 1_000_000f64,
        minor_fault: rusage.ru_minflt as u64,
        major_fault: rusage.ru_majflt as u64,
    })
}

fn get_rusage() -> rusage {
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut usage);
        usage
    }
}
