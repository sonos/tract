#![allow(clippy::collapsible_if)]
#[macro_use]
extern crate log;

pub mod annotations;
pub mod display_params;
pub mod draw;
pub mod export;
pub mod model;
pub mod profile;
pub mod tensor;
pub mod terminal;
pub mod time;

use tract_core::internal::*;
#[allow(unused_imports)]
use tract_cuda::utils::is_culib_present;

pub fn capture_gpu_trace<F>(matches: &clap::ArgMatches, func: F) -> TractResult<()>
where
    F: FnOnce() -> TractResult<()>,
{
    if matches.is_present("metal-gpu-trace") {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let gpu_trace_path =
                std::path::Path::new(matches.value_of("metal-gpu-trace").unwrap()).to_path_buf();
            ensure!(gpu_trace_path.is_absolute(), "Metal GPU trace file has to be absolute");
            ensure!(
                !gpu_trace_path.exists(),
                format!("Given Metal GPU trace file {:?} already exists.", gpu_trace_path)
            );

            log::info!("Capturing Metal GPU trace at : {gpu_trace_path:?}");
            tract_metal::METAL_STREAM.with_borrow(move |stream| {
                stream.capture_trace(gpu_trace_path, move |_stream| func())
            })
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            bail!("`--metal-gpu-trace` present but it is only available on MacOS and iOS")
        }
    } else if matches.is_present("cuda-gpu-trace") {
        if !is_culib_present() {
            bail!("`--cuda-gpu-trace` present but no CUDA insatllation has been found")
        }

        let _prof = cudarc::driver::safe::Profiler::new()?;
        func()
    } else {
        func()
    }
}
