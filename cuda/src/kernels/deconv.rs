use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_gpu::ops::deconv::DeconvGeometry;
use tract_gpu::tensor::DeviceTensor;

const BLOCK: usize = 256;

pub fn cuda_deconv_launch(
    geometry: &DeconvGeometry,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        let dt = if input.datum_type() == DatumType::F16 { "f16" } else { "f32" };
        let func = cuda_context().load_pipeline(LibraryName::Cnn, format!("deconv2d_{dt}"))?;
        let input = get_cuda_view(input);
        let weights = get_cuda_view(weights);
        let output = get_cuda_view(output);
        let mut launcher = TractLaunchArgs::new(stream, &func);
        launcher.push_view(&input);
        launcher.push_view(&weights);
        launcher.push_view(&output);
        for arg in geometry.args() {
            launcher.push_i32(arg);
        }
        launcher.launch(LaunchConfig {
            grid_dim: (geometry.output_len().div_ceil(BLOCK) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    })
}
