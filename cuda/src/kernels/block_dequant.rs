use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, Q6_K, Q8_0};
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::get_quant_fact;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

/// APEX block this dequant kernel can expand. Variant names are the ggml type names.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApexBlock {
    Q8_0,
    Q6_K,
}

impl ApexBlock {
    pub fn format(self) -> &'static dyn BlockQuant {
        match self {
            ApexBlock::Q8_0 => &Q8_0,
            ApexBlock::Q6_K => &Q6_K,
        }
    }

    pub fn kernel_name(self, dt: DatumType) -> TractResult<&'static str> {
        match (self, dt) {
            (ApexBlock::Q8_0, DatumType::F32) => Ok("dequant_q8_0_f32"),
            (ApexBlock::Q8_0, DatumType::F16) => Ok("dequant_q8_0_f16"),
            (ApexBlock::Q6_K, DatumType::F32) => Ok("dequant_q6_k_f32"),
            (ApexBlock::Q6_K, DatumType::F16) => Ok("dequant_q6_k_f16"),
            _ => bail!("APEX dequant {self:?} does not write {dt:?}"),
        }
    }
}

fn same_format(format: &dyn BlockQuant, expected: &dyn BlockQuant) -> bool {
    format == expected
}

/// The block fact on a host fact, when this backend dequantizes that format.
pub fn apex_block(fact: &TypedFact) -> Option<ApexBlock> {
    let bqf = fact.exotic_fact.as_ref()?.downcast_ref::<BlockQuantFact>()?;
    let format = bqf.format.as_ref();
    if same_format(format, &Q8_0) {
        Some(ApexBlock::Q8_0)
    } else if same_format(format, &Q6_K) {
        Some(ApexBlock::Q6_K)
    } else {
        None
    }
}

pub fn dispatch(
    stream: &TractCudaStream,
    format: ApexBlock,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.shape() == input.shape());
    ensure!(output.datum_type() == input.datum_type());
    let block_len = format.format().block_len();
    ensure!(
        input.len().is_multiple_of(block_len),
        "APEX dequant length {} is not a multiple of {block_len}",
        input.len()
    );
    let n_blocks = input.len() / block_len;
    ensure!(n_blocks <= u32::MAX as usize, "APEX dequant block count {n_blocks} exceeds u32");
    get_quant_fact(input, format.format())
        .with_context(|| format!("APEX dequant input is not {}", format.format()))?;
    if n_blocks == 0 {
        return Ok(());
    }

    let func = cuda_context().load_pipeline(
        LibraryName::BlockDequant,
        format.kernel_name(output.datum_type())?.to_string(),
    )?;
    let quant = get_cuda_view(input);
    let out = get_cuda_view(output);
    let mut launch_args = TractLaunchArgs::new(stream, &func);
    launch_args.push_view(&quant);
    launch_args.push_view(&out);
    launch_args.push::<u32>(n_blocks as u32);
    launch_args.launch(LaunchConfig::for_num_elems(n_blocks as u32))
}

pub fn eval(
    stream: &TractCudaStream,
    format: ApexBlock,
    input: &DeviceTensor,
) -> TractResult<DeviceTensor> {
    let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
    dispatch(stream, format, input, &output)?;
    stream.synchronize()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::tract_linalg::block_quant::BlockQuantStorage;
    use tract_gpu::tensor::IntoDevice;

    fn assert_dequant(format: ApexBlock, shape: &[usize], dt: DatumType) -> TractResult<()> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) / 7.0).collect();
        let bq = format.format();
        let blob = bq.quant_f32(&data)?;
        let bytes = blob.as_bytes().to_vec();
        let m = shape[..shape.len() - 1].iter().product();
        let k = *shape.last().unwrap();
        let storage = BlockQuantStorage::new(
            tract_core::dyn_clone::clone_box(bq),
            m,
            k,
            std::sync::Arc::new(blob),
        )?;
        let quant = storage.into_tensor_with_shape(dt, shape);
        let cpu = if dt == DatumType::F16 {
            bq.dequant_f16(&bytes)?.into_shape(shape)?
        } else {
            bq.dequant_f32(&bytes)?.into_shape(shape)?
        };
        let cuda = crate::with_cuda_stream(|stream| {
            let device = quant.into_device()?;
            Ok(eval(stream, format, &device)?.to_host()?.into_tensor())
        })?;
        cpu.close_enough(&cuda, Approximation::Approximate)
            .with_context(|| format!("{format:?} {shape:?} {dt:?}"))?;
        Ok(())
    }

    #[test]
    fn q8_0_matches_cpu_dequant() -> TractResult<()> {
        for shape in [[1usize, 32], [3, 64], [4, 256]] {
            assert_dequant(ApexBlock::Q8_0, &shape, DatumType::F32)?;
            assert_dequant(ApexBlock::Q8_0, &shape, DatumType::F16)?;
        }
        Ok(())
    }

    #[test]
    fn q6_k_matches_cpu_dequant() -> TractResult<()> {
        for shape in [[1usize, 256], [3, 256], [4, 512]] {
            assert_dequant(ApexBlock::Q6_K, &shape, DatumType::F32)?;
            assert_dequant(ApexBlock::Q6_K, &shape, DatumType::F16)?;
        }
        Ok(())
    }
}
