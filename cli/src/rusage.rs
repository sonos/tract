#![allow(dead_code)]
use Result;

use libc::{getrusage, RUSAGE_SELF, rusage, timeval};

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

#[cfg(target_os="macos")]
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
                user_time: timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                },
                system_time: timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                },
                policy: 0,
                suspend_count: 0,
            }
        }
    }
    mod ffi {
        use libc::*;
        extern "C" {
            pub fn mach_task_self() -> c_uint;
            pub fn task_info(task: c_uint,
                             flavor: c_int,
                             task_info: *mut super::BasicTaskInfo,
                             count: *mut c_uint)
                             -> c_uint;
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

#[cfg(target_os="macos")]
pub fn get_memory_usage() -> Result<ResourceUsage> {
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

#[cfg(target_os="linux")]
pub fn get_memory_usage() -> Result<ResourceUsage> {
    use std::fs::File;
    use std::io::Read;
    let mut proc_stat = String::new();
    let _ = try!(try!(File::open("/proc/self/stat")).read_to_string(&mut proc_stat));
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

pub fn get_rusage() -> rusage {
    let mut usage = rusage {
        ru_idrss: 0,
        ru_nvcsw: 0,
        ru_ixrss: 0,
        ru_isrss: 0,
        ru_inblock: 0,
        ru_minflt: 0,
        ru_oublock: 0,
        ru_nivcsw: 0,
        ru_stime: timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        ru_nswap: 0,
        ru_maxrss: 0,
        ru_majflt: 0,
        ru_msgrcv: 0,
        ru_msgsnd: 0,
        ru_utime: timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        ru_nsignals: 0,
    };
    unsafe {
        getrusage(RUSAGE_SELF, &mut usage);
    }
    usage
}
