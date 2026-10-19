use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, Q8_0};
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::get_quant_fact;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

pub fn kernel_name(dt: DatumType) -> TractResult<&'static str> {
    match dt {
        DatumType::F32 => Ok("dequant_q8_0_f32"),
        DatumType::F16 => Ok("dequant_q8_0_f16"),
        dt => bail!("Q8_0 dequant writes f16 or f32, got {dt:?}"),
    }
}

/// The block fact on a host fact, when this backend dequantizes that format.
pub fn is_q8_0(fact: &TypedFact) -> bool {
    fact.exotic_fact
        .as_ref()
        .and_then(|of| of.downcast_ref::<BlockQuantFact>())
        .is_some_and(|bqf| bqf.format.as_ref() == (&Q8_0 as &dyn BlockQuant))
}

pub fn dispatch(
    stream: &TractCudaStream,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape());
    ensure!(output.datum_type() == input.datum_type());
    let block_len = Q8_0.block_len();
    ensure!(
        input.len().is_multiple_of(block_len),
        "Q8_0 dequant length {} is not a multiple of {block_len}",
        input.len()
    );
    let n_blocks = input.len() / block_len;
    ensure!(n_blocks <= u32::MAX as usize, "Q8_0 dequant block count {n_blocks} exceeds u32");
    get_quant_fact(input, &Q8_0).context("Q8_0 dequant input is not Q8_0")?;
    if n_blocks == 0 {
        return Ok(());
    }

    let func = cuda_context()
        .load_pipeline(LibraryName::BlockDequant, kernel_name(output.datum_type())?.to_string())?;
    let quant = get_cuda_view(input);
    let out = get_cuda_view(output);
    let mut launch_args = TractLaunchArgs::new(stream, &func);
    launch_args.push_view(&quant);
    launch_args.push_view(&out);
    launch_args.push::<u32>(n_blocks as u32);
    launch_args.launch(LaunchConfig::for_num_elems(n_blocks as u32))
}

pub fn eval(stream: &TractCudaStream, input: &DeviceTensor) -> TractResult<DeviceTensor> {
    let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
    dispatch(stream, input, &output)?;
    stream.synchronize()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantStorage};
    use tract_gpu::tensor::IntoDevice;

    fn assert_dequant(shape: &[usize], dt: DatumType) -> TractResult<()> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) / 7.0).collect();
        let blob = Q8_0.quant_f32(&data)?;
        let bytes = blob.as_bytes().to_vec();
        let m = shape[..shape.len() - 1].iter().product();
        let k = *shape.last().unwrap();
        let storage = BlockQuantStorage::new(
            tract_core::dyn_clone::clone_box(&Q8_0),
            m,
            k,
            std::sync::Arc::new(blob),
        )?;
        let quant = storage.into_tensor_with_shape(dt, shape);
        let cpu = if dt == DatumType::F16 {
            Q8_0.dequant_f16(&bytes)?.into_shape(shape)?
        } else {
            Q8_0.dequant_f32(&bytes)?.into_shape(shape)?
        };
        let cuda = crate::with_cuda_stream(|stream| {
            let device = quant.into_device()?;
            Ok(eval(stream, &device)?.to_host()?.into_tensor())
        })?;
        cpu.close_enough(&cuda, Approximation::Approximate)
            .with_context(|| format!("Q8_0 {shape:?} {dt:?}"))?;
        Ok(())
    }

    #[test]
    fn q8_0_matches_cpu_dequant() -> TractResult<()> {
        for shape in [[1usize, 32], [3, 64], [4, 256]] {
            assert_dequant(&shape, DatumType::F32)?;
            assert_dequant(&shape, DatumType::F16)?;
        }
        Ok(())
    }
}
