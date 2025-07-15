use cudarc::cublas::{self, CudaBlas, Gemm};
use cudarc::driver::{CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use derive_new::new;
use num_traits::{Float, One};
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::utils::as_q40_tensor;
use std::{default, fmt};
use tract_core::internal::*;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::tensor::DeviceTensor::Owned;

use crate::context::cuda_context;
use crate::kernels::{get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, LibraryName};
use crate::tensor::CudaTensor;

use DatumType::{F16, F32};

static WARP_SIZE: usize = 32;

#[derive(Debug)]
struct MatMulParams {
    pub a_batch: usize,
    pub b_batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a_strides: TVec<isize>,
    pub b_strides: TVec<isize>,
    pub c_strides: TVec<isize>,
}

// Squeeze batch axes and return a shape with a rank of 3.
fn squeeze_batch_axes(s: &[usize]) -> TractResult<TVec<usize>> {
    ensure!(s.len() >= 2);
    let rank = s.len();
    if s.len() == 2 {
        return Ok(tvec![1, s[rank - 2], s[rank - 1]]);
    }
    let rank = s.len();
    Ok(tvec![s[..rank - 2].iter().product(), s[rank - 2], s[rank - 1],])
}

impl MatMulParams {
    fn from_inputs(a: &DeviceTensor, b: &DeviceTensor, c: &DeviceTensor) -> TractResult<MatMulParams>{
        let rank = c.rank();
        let squeezed_a_shape = squeeze_batch_axes(a.shape())?;
        let squeezed_b_shape = squeeze_batch_axes(b.shape())?;
        let squeezed_c_shape = squeeze_batch_axes(c.shape())?;

        let a_batch = squeezed_a_shape[0];
        let b_batch = squeezed_b_shape[0];

        ensure!(squeezed_c_shape[0] == a_batch || squeezed_c_shape[0] == b_batch);

        let m = c.shape()[rank - 2];
        let n = c.shape()[rank - 1];
        let k = a.shape()[a.rank() - 1];

        ensure!(a_batch % b_batch == 0);
        let a_strides = natural_strides(&[a_batch, m, k]);
        let b_strides = natural_strides(&[b_batch, n, k]);
        let c_strides = natural_strides(&[a_batch, m, n]);

        Ok(MatMulParams {
            a_batch,
            b_batch,
            m,
            n,
            k,
            a_strides,
            b_strides,
            c_strides
        })
    }
}

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Matmul;

impl fmt::Display for Matmul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Matmul {
    pub fn is_supported_dt(a_dt: DatumType, b_dt: DatumType) -> bool {
        let regular_types_support = matches!(
            (a_dt, b_dt),
            (F32, F32) | (F16, F16)
        );

        regular_types_support
    }

    pub fn output_shape<D: DimLike + One>(a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2].clone());
        output.push(b[rank - 2].clone());
        output
    }

    fn find_block_size(k: usize) -> usize {
        let mut block_size_best = WARP_SIZE;
        let mut best_niter = (k + 2 * WARP_SIZE - 1) / (2 * WARP_SIZE);

        for block_size in (2 * WARP_SIZE..=256).step_by(WARP_SIZE as usize) {
            let niter = (k + 2 * block_size - 1) / (2 * block_size);
            if niter < best_niter {
                best_niter = niter;
                block_size_best = block_size;
            }
        }

        block_size_best
    }

    pub fn kernel_name(&self, a_dt: DatumType, b_dt: DatumType, block_size: usize) -> TractResult<String> {
        ensure!(Self::is_supported_dt(a_dt, b_dt), "Unsupported dts ({:?}, {:?}) for Cuda MatMul", a_dt, b_dt);
        let tname = DeviceTensor::tname(a_dt)?;
        // Note: Currently always accumulate in F32
        let acc_type = DeviceTensor::tname(F32)?;
        Ok(format!("ggml_matvec_{tname}_acc_{acc_type}_bs_{block_size}"))
    }

    pub fn dispatch_eval<F: Datum + Float>(
        &self,
        stream: Arc<CudaStream>,
        a: &DeviceTensor,
        b: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()>
    where CudaBlas: Gemm<F>
     {
        let q40_b = if let Owned(t) = b {
            t.downcast_ref::<CudaTensor>().expect("Non Cuda Tensor in Cuda context").block_quant_fact()
        } else { None };
        ensure!((a.datum_type() == b.datum_type()) || q40_b.is_some());

        let b_shape = q40_b
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        ensure!(output.shape() == Self::output_shape(a.shape(), &b_shape).as_slice());

        if output.shape().iter().product::<usize>() == 0 {
            return Ok(());
        }

        let a_view = get_cuda_view(a);
        let b_view = get_cuda_view(b);

        let params = MatMulParams::from_inputs(a, b, output)?;  
        if (params.k % 2 == 0) && (params.a_strides[1] % 2 == 0) && params.m == 1 {
            let mut out_view = get_cuda_view(output);

            let k_div_2 = params.k / 2;
            let block_size = Self::find_block_size(params.k);

            let batch_ratio = params.a_batch / params.b_batch;

            let kernel_name = self.kernel_name(a.datum_type(), b.datum_type(), block_size)?;
            let mut func = cuda_context().load_pipeline(LibraryName::Ggml, kernel_name)?;
            let mut launch_args = stream.launch_builder(&func);

            launch_args.arg(&b_view);
            launch_args.arg(&a_view);
            launch_args.arg(&out_view);
            launch_args.arg(&k_div_2);
            launch_args.arg(&params.a_batch);
            launch_args.arg(&params.b_strides[1]);  
            launch_args.arg(&batch_ratio);
            launch_args.arg(&params.b_strides[0]);
            launch_args.arg(&params.a_strides[0]);
            launch_args.arg(&params.c_strides[0]);

            let cfg = LaunchConfig { 
                grid_dim: (params.n as _, params.a_batch as _, 1),
                block_dim: (block_size as _, 1, 1),
                shared_mem_bytes: (WARP_SIZE * size_of::<f32>()) as u32 };
            unsafe { launch_args.launch(cfg) };
        } else {
            let mut out_view = get_cuda_view_mut(output);

            let cublas_gemm_cfg = cublas::GemmConfig {
                transa: cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: params.n as i32,
                n: params.m as i32,
                k: params.k as i32,
                alpha: F::from(1.0f32).unwrap(),
                lda: params.k as i32,
                ldb: params.k as i32,
                beta:F::from(0.0f32).unwrap(),
                ldc: params.n as i32,
            };
            let cublas = CudaBlas::new(stream)?;
            unsafe { cublas.gemm(cublas_gemm_cfg, &b_view.transmute::<F>(b.len()).unwrap(), &a_view.transmute::<F>(a.len()).unwrap(), &mut out_view.transmute_mut::<F>(output.len()).unwrap()) };
        }
        
        Ok(())
    }

    pub fn eval(
        &self,
        stream: Arc<CudaStream>,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(lhs.datum_type(), &Self::output_shape(lhs.shape(), rhs.shape()))? };
        if lhs.datum_type() == DatumType::F32 {
            self.dispatch_eval::<f32>(stream.clone(), lhs, rhs, &output)?;
        }
        else {
            ensure!(lhs.datum_type() == F16);
            self.dispatch_eval::<f16>(stream.clone(), lhs, rhs, &output)?;
        }
        stream.synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;

    use super::*;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::tract_data::itertools::Itertools;
    use tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0,
    };
    use tract_gpu::tensor::IntoDevice;

    pub(crate) fn run_mmm_test_case(
        (batch, m, k, n): (usize, usize, usize, usize),
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_shape = [batch, m, k];
            let b_shape = [batch, n, k];
            let mut a = if a_dt == DatumType::F16 {
                Tensor::from_shape(
                    &a_shape,
                    &(0..batch * m * k)
                        .map(|f| f16::from_f32(f as f32 / (batch * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &a_shape,
                    &(0..batch * m * k)
                        .map(|f| f as f32 / (batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let mut b = if b_dt == DatumType::F16 {
                Tensor::from_shape(
                    &b_shape,
                    &(0..batch * k * n)
                        .map(|f| f16::from_f32(f as f32 / (batch * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &b_shape,
                    &(0..batch * k * n)
                        .map(|f| f as f32 / (batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let cuda_output = Matmul.eval(
                stream.clone(),
                &a.clone().into_device()?,
                &b.clone().into_device()?,
            )?;

            let matmul = PrefixMatMul {
                transpose_a: false,
                transpose_b: true,
                transpose_c: false,
                quantize_output: None,
            };

            // Compare to full precision
            if a_dt == DatumType::F16 && !(b_dt == DatumType::F16) {
                a = a.clone().cast_to_dt(DatumType::F32).unwrap().into_owned();
            }
            if b_dt == DatumType::F16 && !(a_dt == DatumType::F16) {
                b = b.clone().cast_to_dt(DatumType::F32).unwrap().into_owned();
            }

            let output = args_1!(matmul.eval(tvec![a.into_tvalue(), b.into_tvalue()])?);
            cuda_output.to_host()?.close_enough(&output, Approximation::SuperApproximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case((1, 1, 2, 1), F32, F32)?;
        run_mmm_test_case((1, 1, 60, 2), F32, F32)?;
        run_mmm_test_case((2, 1, 128, 7), F32, F32)?;

        ////// f16_f16
        run_mmm_test_case((1, 1, 2, 1), F16, F16)?;
        run_mmm_test_case((1, 1, 61, 2), F16, F16)?;
        run_mmm_test_case((2, 1, 128, 9), F16, F16)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case((1, 2, 4, 2), F32, F32)?;
        run_mmm_test_case((1, 2, 2, 3), F32, F32)?;
        //run_mmm_test_case((2, 3, 128, 7), F32, F32)?;

        //////// f16_f16
        run_mmm_test_case((1, 2, 7, 2), F16, F16)?;
        run_mmm_test_case((1, 2, 61, 2), F16, F16)?;
        //run_mmm_test_case((2, 1, 128, 9), F16, F16)?;
        Ok(())
    }
}