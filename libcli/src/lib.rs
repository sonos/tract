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
#[cfg(any(target_os = "linux", target_os = "windows"))]
use tract_cuda::utils::ensure_cuda_runtime_dependencies;

pub fn capture_gpu_trace<F>(matches: &clap::ArgMatches, func: F) -> TractResult<()>
where
    F: FnOnce() -> TractResult<()>,
{
    if matches.contains_id("metal-gpu-trace")
        && matches.get_one::<String>("metal-gpu-trace").is_some()
    {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let gpu_trace_path =
                std::path::Path::new(matches.get_one::<String>("metal-gpu-trace").unwrap())
                    .to_path_buf();
            ensure!(gpu_trace_path.is_absolute(), "Metal GPU trace file has to be absolute");
            ensure!(
                !gpu_trace_path.exists(),
                format!("Given Metal GPU trace file {:?} already exists.", gpu_trace_path)
            );

            log::info!("Capturing Metal GPU trace at : {gpu_trace_path:?}");
            tract_metal::with_metal_stream(move |stream| {
                stream.capture_trace(gpu_trace_path, move |_stream| func())
            })
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            bail!("`--metal-gpu-trace` present but it is only available on MacOS and iOS")
        }
    } else if matches.contains_id("cuda-gpu-trace")
        && matches.get_one::<String>("cuda-gpu-trace").is_some()
    {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            ensure_cuda_runtime_dependencies(
                "`--cuda-gpu-trace` present but no CUDA installation has been found",
            )?;
            let _prof = cudarc::driver::safe::Profiler::new()?;
            func()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            bail!("`--cuda-gpu-trace` present but it is only available on Linux and Windows")
        }
    } else {
        func()
    }
}
