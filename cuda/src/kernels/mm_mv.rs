use cudarc::cublas::{self, CudaBlas, Gemm};
use cudarc::driver::result::stream::null;
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::runtime::result::device::get_device_prop;
use cudarc::runtime::sys::cudaGetLastError;
use derive_new::new;
use num_traits::{Float, One};
use std::{default, fmt};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::tensor::DeviceTensor::Owned;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::{as_q40_fact, as_q40_tensor};

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::{
    get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, get_sliced_cuda_view_mut, launch_args, LibraryName
};
use crate::tensor::CudaTensor;
use crate::utils::get_q40_fact;
use crate::Q40_ROW_PADDING;

use DatumType::{F16, F32};

static N_WARPS: usize = 8;

static WARP_SIZE: usize = 32;
static MMQ_X_MAX: usize = 128;

static QK8_0: usize = 32;
static QI8_0: usize = QK8_0 / (4 * QR8_0);
static QR8_0: usize =  1;

static MMQ_MMA_TILE_X_K_Q8_0:usize = (2*WARP_SIZE + 2*WARP_SIZE/QI8_0 + 4);

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

fn mmq_get_nbytes_shared_q40(mmq_x: usize, mmq_y: usize)-> usize {
    let nb_ids = mmq_x * size_of::<i32>();
    let mmq_tile_x_l = MMQ_MMA_TILE_X_K_Q8_0;
    let nbs_x = mmq_y * mmq_tile_x_l * size_of::<i32>();
    let nbs_y = mmq_x * 144;

    let pad  = N_WARPS * WARP_SIZE * size_of::<i32>();
    return nb_ids + nbs_x + nbs_y.next_multiple_of(pad)
}

impl MatMulParams {
    fn from_inputs(
        a_shape: &[usize],
        b_shape: &[usize],
        c_shape: &[usize],
    ) -> TractResult<MatMulParams> {
        let rank = c_shape.len();
        let squeezed_a_shape = squeeze_batch_axes(a_shape)?;
        let squeezed_b_shape = squeeze_batch_axes(b_shape)?;
        let squeezed_c_shape = squeeze_batch_axes(c_shape)?;

        let a_batch = squeezed_a_shape[0];
        let b_batch = squeezed_b_shape[0];

        ensure!(squeezed_c_shape[0] == a_batch || squeezed_c_shape[0] == b_batch);

        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a_shape[a_shape.len() - 1];

        ensure!(a_batch % b_batch == 0 || b_batch == 1 || a_batch == 1);
        let a_strides = natural_strides(&[a_batch, m, k]);
        let b_strides = natural_strides(&[b_batch, n, k]);
        let c_strides = natural_strides(&[a_batch, m, n]);

        Ok(MatMulParams { a_batch, b_batch, m, n, k, a_strides, b_strides, c_strides })
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
    pub fn is_supported_dts(facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Ggml: Expected 2 inputs for Matmul");

        let regular_types_support =
            matches!((facts[0].datum_type, facts[1].datum_type), (F32, F32) | (F16, F16));


        regular_types_support || (as_q40_fact(&facts[1]).is_some() && facts[0].datum_type == F32)
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
        let mut best_niter = k.div_ceil(2 * WARP_SIZE);

        for block_size in (2 * WARP_SIZE..=256).step_by(WARP_SIZE) {
            let niter = k.div_ceil(2 * block_size);
            if niter < best_niter {
                best_niter = niter;
                block_size_best = block_size;
            }
        }

        block_size_best
    }

    fn dispatch_ggml_matvec(
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
        output: &DeviceTensor,
        params: MatMulParams,
    ) -> TractResult<()> {
        let a_view = get_cuda_view(a);
        let b_view = get_cuda_view(b);

        let mut out_view = get_cuda_view(output);

        let k_div_2 = params.k / 2;
        let block_size = Self::find_block_size(params.k);

        let batch_ratio = params.a_batch / params.b_batch;

        let kernel_name =
            format!("ggml_matvec_{}_bs_{block_size}", DeviceTensor::tname(a.datum_type())?);
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
            shared_mem_bytes: (WARP_SIZE * size_of::<f32>()) as u32,
        };
        unsafe { launch_args.launch(cfg) };
        Ok(())
    }

    fn dispatch_cublas_gemm<F: Datum + Float>(
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
        output: &DeviceTensor,
        params: MatMulParams,
    ) -> TractResult<()>
    where
        CudaBlas: Gemm<F>,
    {
        let cublas_gemm_cfg = cublas::GemmConfig {
            transa: cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: params.n as i32,
            n: params.m as i32,
            k: params.k as i32,
            alpha: F::from(1.0f32).unwrap(),
            lda: params.k as i32,
            ldb: params.k as i32,
            beta: F::from(0.0f32).unwrap(),
            ldc: params.n as i32,
        };

        let a_batch_stride = params.a_strides[0] as usize;
        let b_batch_stride = params.b_strides[0] as usize;
        let c_batch_stride = params.c_strides[0] as usize;

        if params.a_batch == params.b_batch {
            let a_view = get_cuda_view(a);
            let b_view = get_cuda_view(b);
            let mut out_view = get_cuda_view_mut(output);

            let gemm_batched_strided_cfg = cublas::StridedBatchedConfig {
                gemm: cublas_gemm_cfg,
                batch_size: params.a_batch as i32,
                stride_a: b_batch_stride as _,
                stride_b: a_batch_stride as _,
                stride_c: c_batch_stride as _,
            };
            unsafe {
                stream.cublas().gemm_strided_batched(
                    gemm_batched_strided_cfg,
                    &b_view.transmute::<F>(b.len()).unwrap(),
                    &a_view.transmute::<F>(a.len()).unwrap(),
                    &mut out_view.transmute_mut::<F>(output.len()).unwrap(),
                )
            };
        } else {
            let dt_size = size_of::<F>();
            let (iter_batch, a_offset, b_offset) = if params.b_batch == params.a_batch {
                (params.b_batch, a_batch_stride * dt_size, b_batch_stride * dt_size)
            } else if params.a_batch == 1 {
                (params.b_batch, 0, b_batch_stride * dt_size)
            } else if params.b_batch == 1 {
                (params.a_batch, a_batch_stride * dt_size, 0)
            } else {
                bail!(
                    "Unsupported batches config: A: {} and B: {}",
                    params.a_batch,
                    params.b_batch
                );
            };
            let c_offset = params.c_strides[0] as usize * dt_size;

            for i in 0..iter_batch {
                let a_view =
                    get_sliced_cuda_view(a, i * a_offset, a_batch_stride * size_of::<F>())?;
                let b_view =
                    get_sliced_cuda_view(b, i * b_offset, b_batch_stride * size_of::<F>())?;
                let mut out_view = get_sliced_cuda_view_mut(
                    output,
                    i * c_offset,
                    c_batch_stride * size_of::<F>(),
                )?;

                unsafe {
                    stream.cublas().gemm(
                        cublas_gemm_cfg,
                        &b_view.transmute::<F>(b_batch_stride).unwrap(),
                        &a_view.transmute::<F>(a_batch_stride).unwrap(),
                        &mut out_view.transmute_mut::<F>(c_batch_stride).unwrap(),
                    )?
                };
            }
        }
        Ok(())
    }

    fn kernel_name_q40(params: &MatMulParams, mmq_x: usize, mmq_y: usize) -> TractResult<String> {
        let need_check = params.n % mmq_y != 0;
        Ok(format!("mul_mat_q_GGML_TYPE_Q4_0_{mmq_x}_8_{need_check}"))
    }

    fn fixup_kernel_name_q40(params: &MatMulParams, mmq_x: usize, mmq_y: usize) -> TractResult<String> {
        let need_check = params.n % mmq_y != 0;
        Ok(format!("mul_mat_q_stream_k_fixup_GGML_TYPE_Q4_0_{mmq_x}_8_{need_check}"))
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let b_shape = get_q40_fact(b)
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        ensure!(output.shape() == Self::output_shape(a.shape(), &b_shape).as_slice());

        if output.shape().iter().product::<usize>() == 0 {
            return Ok(());
        }

        let params = MatMulParams::from_inputs(a.shape(), &b_shape, output.shape())?;
        if get_q40_fact(b).is_some() {
            Self::dispatch_ggml_matmul_q40(stream, a, b, output, params)?;
        } else if (params.k % 2 == 0) && params.m == 1 && params.a_batch >= params.b_batch {
            Self::dispatch_ggml_matvec(stream, a, b, output, params)?;
        } else if a.datum_type() == DatumType::F32 {
            Self::dispatch_cublas_gemm::<f32>(stream, a, b, output, params)?;
        } else {
            ensure!(a.datum_type() == F16);
            Self::dispatch_cublas_gemm::<f16>(stream, a, b, output, params)?;
        }
        Ok(())
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        lhs: &DeviceTensor,
        rhs: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let rhs_shape = get_q40_fact(rhs)
            .map(|bqf| rhs.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(rhs.shape().to_vec());
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                lhs.datum_type(),
                &Self::output_shape(lhs.shape(), &rhs_shape),
            )?
        };
        self.dispatch_eval(stream, lhs, rhs, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    fn dispatch_ggml_matmul_q40(
    stream: &TractCudaStream,
    a: &DeviceTensor,
    b: &DeviceTensor,
    output: &DeviceTensor,
    params: MatMulParams,
    ) -> TractResult<()> {
        let a_view = get_cuda_view(a);
        let b_view = get_cuda_view(b);

        let mut out_view = get_cuda_view(output);

        let null_ptr = stream.null::<u8>()?;

        let padded_k = params.k.next_multiple_of(Q40_ROW_PADDING);
        let n_blocks = (padded_k / Q4_0.block_len()); // Q40 weights have also been padded during transform

        let nbytes_a_q8_1 = a.len() * padded_k * (QK8_0 + 4) / (params.k * QK8_0) + MMQ_X_MAX * 144;

        let sample_stride_a = params.a_strides[0] * params.a_batch as isize;
        let quant_a = unsafe {
            DeviceTensor::uninitialized_dt(
                DatumType::U8,
                &[nbytes_a_q8_1],
            )?
        };

        let quant_a_view = get_cuda_view(&quant_a);

        let func = cuda_context().load_pipeline(LibraryName::GgmlQ, "quantize_mmq_q8_1".to_string())?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&a_view);
        launch_args.arg(&null_ptr);
        launch_args.arg(&quant_a_view);
        launch_args.arg(&params.k);
        launch_args.arg(&params.a_strides[1]);
        launch_args.arg(&params.a_strides[0]);
        launch_args.arg(&sample_stride_a);
        launch_args.arg(&padded_k);
        launch_args.arg(&params.m);
        launch_args.arg(&params.a_batch);

        let cfg = LaunchConfig {
            grid_dim: (params.m as _, params.k.div_ceil(4 * 128) as _, params.a_batch as _),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch_args.launch(cfg) };

        //dbg!(quant_a.to_host()?.to_array_view::<u8>()?);
        let a_stride_0 = padded_k * params.m * 36 / 128;

        let mut mmq_x_best  = 0;
        let mut ntiles_x_best = usize::max_value();

        let mut mmq_x = 0;
        while mmq_x <= MMQ_X_MAX && ntiles_x_best > 1 {
            mmq_x += 8;
            let granularity = if mmq_x >= 48 { 16 } else { 8 };
            if (mmq_x % granularity != 0 || mmq_get_nbytes_shared_q40(mmq_x, MMQ_X_MAX) > get_device_prop(0)?.sharedMemPerBlockOptin) {
                continue;
            }
            
            let ntiles_x = (params.m + mmq_x - 1) / mmq_x;
            if (ntiles_x < ntiles_x_best) {
                mmq_x_best = mmq_x;
                ntiles_x_best = ntiles_x;
            }
        }

        let nbytes_shared = mmq_get_nbytes_shared_q40(mmq_x_best, MMQ_X_MAX);
        let kernel_name = Self::kernel_name_q40(&params, mmq_x_best, MMQ_X_MAX)?;

        let batch_ratio = params.a_batch / params.b_batch;
        let b_stride_0 = n_blocks * params.n;

        let nty = params.n.div_ceil(MMQ_X_MAX);
        let ntx = params.m.div_ceil(mmq_x_best);

        let fixup_needed = ((ntx * nty * params.a_batch) % get_device_prop(0)?.multiProcessorCount as usize) != 0;
        let fixup_shape = if fixup_needed { get_device_prop(0)?.multiProcessorCount as usize * mmq_x_best * MMQ_X_MAX } else { 0 };

        let fixup_tensor = unsafe {
                DeviceTensor::uninitialized_dt(
                    DatumType::F32,
                    &[fixup_shape],
                )?
            };
        
        let fixup_view = get_cuda_view(&fixup_tensor);
        let func = cuda_context().load_pipeline(LibraryName::GgmlQ, kernel_name)?;
        func.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes_shared as i32)?;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&b_view);
        launch_args.arg(&quant_a_view);
        launch_args.arg(&null_ptr);
        launch_args.arg(&null_ptr);
        launch_args.arg(&out_view);
        launch_args.arg(&fixup_view);
        launch_args.arg(&params.k);
        launch_args.arg(&params.n);
        launch_args.arg(&params.m);
        launch_args.arg(&n_blocks);
        launch_args.arg(&params.m);
        launch_args.arg(&params.n);
        launch_args.arg(&batch_ratio);
        launch_args.arg(&params.a_batch);
        launch_args.arg(&b_stride_0);
        launch_args.arg(&a_stride_0);
        launch_args.arg(&params.c_strides[0]);
        
        let cfg = LaunchConfig {
            grid_dim: (get_device_prop(0)?.multiProcessorCount as _, 1, 1),
            block_dim: (WARP_SIZE as _, N_WARPS as _, 1),
            shared_mem_bytes: nbytes_shared as _,
        };

        unsafe { launch_args.launch(cfg); }

        if fixup_needed {
            let kernel_name = Self::fixup_kernel_name_q40(&params, mmq_x_best, MMQ_X_MAX)?;
            let func = cuda_context().load_pipeline(LibraryName::GgmlQ, kernel_name)?;
            let mut launch_args = stream.launch_builder(&func);
                launch_args.arg(&null_ptr);
                launch_args.arg(&null_ptr);
                launch_args.arg(&out_view);
                launch_args.arg(&fixup_view);
                launch_args.arg(&params.k);
                launch_args.arg(&params.n);
                launch_args.arg(&params.m);
                launch_args.arg(&params.n);
                launch_args.arg(&params.a_batch);
                launch_args.arg(&params.c_strides[0]);

            let cfg = LaunchConfig {
                grid_dim: (get_device_prop(0)?.multiProcessorCount as _, 1, 1),
                block_dim: (WARP_SIZE as _, N_WARPS as _, 1),
                shared_mem_bytes: 0,
            };

            unsafe { launch_args.launch(cfg); }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use crate::context::CUDA_STREAM;

    use super::*;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::ops::array::MultiBroadcastTo;
    use tract_core::ops::cast::Cast;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::tract_data::itertools::Itertools;
    use tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0,
    };
    use tract_gpu::tensor::IntoDevice;

    pub(crate) fn run_mmm_test_case(
        (batch_a, batch_b, m, k, n): (usize, usize, usize, usize, usize),
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<()> {
        ensure!((batch_a % batch_b == 0) || (batch_b % batch_a == 0));
        CUDA_STREAM.with(|stream| {
            let a_shape = [batch_a, m, k];
            let b_shape = [batch_b, n, k];
            let mut a = if a_dt == DatumType::F16 {
                Tensor::from_shape(
                    &a_shape,
                    &(0..batch_a * m * k)
                        .map(|f| f16::from_f32(f as f32 / (batch_a * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &a_shape,
                    &(0..batch_a * m * k)
                        .map(|f| (f + 1) as f32 / (batch_a * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let mut b = if b_dt == DatumType::F16 {
                Tensor::from_shape(
                    &b_shape,
                    &(0..batch_b * k * n)
                        .map(|f| f16::from_f32(f as f32 / (batch_b * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &b_shape,
                    &(0..batch_b * k * n)
                        .map(|f| (f + 1) as f32 / (batch_b * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let cuda_output =
                Matmul.eval(stream, &a.clone().into_device()?, &b.clone().into_device()?)?;

            //let mut b = b.broadcast_to_shape(&[batch_a, n, k])?;
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
        run_mmm_test_case((1, 1, 1, 2, 1), F32, F32)?;
        run_mmm_test_case((2, 1, 1, 60, 2), F32, F32)?;
        run_mmm_test_case((2, 2, 1, 128, 7), F32, F32)?;
        run_mmm_test_case((4, 1, 1, 2, 1), F32, F32)?;

        // f16_f16
        run_mmm_test_case((1, 1, 1, 2, 1), F16, F16)?;
        run_mmm_test_case((2, 1, 1, 61, 2), F16, F16)?;
        run_mmm_test_case((2, 2, 1, 128, 9), F16, F16)?;
        run_mmm_test_case((4, 1, 1, 128, 9), F16, F16)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case((1, 1, 2, 4, 2), F32, F32)?;
        run_mmm_test_case((1, 1, 2, 2, 3), F32, F32)?;
        run_mmm_test_case((2, 2, 1, 1, 2), F32, F32)?;
        run_mmm_test_case((1, 2, 1, 1, 2), F32, F32)?;

        // f16_f16
        run_mmm_test_case((1, 1, 2, 7, 2), F16, F16)?;
        run_mmm_test_case((1, 1, 2, 61, 2), F16, F16)?;
        run_mmm_test_case((2, 1, 1, 127, 9), F16, F16)?;
        run_mmm_test_case((1, 2, 1, 127, 9), F16, F16)?;
        Ok(())
    }

    fn reference(a: Tensor, b: Tensor) -> TractResult<Tensor> {
        let batch = b.shape()[0];
        let batch_ratio = a.shape()[0] / batch;
        let matmul = PrefixMatMul {
            transpose_a: false,
            transpose_b: true,
            transpose_c: false,
            quantize_output: None,
        };

        let mut model = TypedModel::default();

        let lhs = model.add_source("lhs", TypedFact::shape_and_dt_of(&a))?;
        let mut rhs = model.add_source("rhs", TypedFact::shape_and_dt_of(&b))?;

        if b.datum_type() == DatumType::F16 {
            rhs = model.wire_node("cast", Cast { to: DatumType::F32 }, &[rhs])?[0];
        }
        if batch_ratio > 1 {
            let add_axis_out = model.wire_node("add_axis", AxisOp::Add(1), &[rhs])?[0];
            let mut broadcast_shape = b.shape().to_vec();

            broadcast_shape.insert(1, batch_ratio);
            let broadcast_out = model.wire_node(
                "broadcast",
                MultiBroadcastTo { shape: ShapeFact::from_dims(broadcast_shape) },
                &[add_axis_out],
            )?[0];
            rhs = model.wire_node(
                "reshape",
                AxisOp::Reshape(
                    0,
                    tvec![batch.into(), batch_ratio.into()],
                    tvec![(batch * batch_ratio).into()],
                ),
                &[broadcast_out],
            )?[0]
        }
        let output = model.wire_node("matmul", matmul, &[lhs, rhs])?;

        model.set_output_outlets(&output)?;
        model = model.into_decluttered()?;
        let mut output =
            DefaultRuntime.prepare(model)?.run(tvec!(a.into_tvalue(), b.into_tvalue()))?;
        Ok(output.remove(0).into_tensor())
    }

    fn run_ggml_mat_mul_test<F: Datum + Float>(
        batch: usize,
        broadcast_ratio: usize,
        m: usize,
        k: usize,
        n: usize,
        q40: bool,
    ) -> TractResult<()>
    where
        f32: From<F>,
    {
        CUDA_STREAM.with(|stream| {
            let a_shape = [batch * broadcast_ratio, m, k];
            let b_shape = [batch, n, k];

            let a_data = (0..batch * broadcast_ratio * k * m)
                .map(|f| f as f32 / (batch * broadcast_ratio * m * k) as f32)
                .collect::<Vec<_>>();

            let a = Tensor::from_shape(&a_shape, &a_data)?;

            let b_data = (0..batch * n * k)
                .map(|f| F::from(f).unwrap() / F::from(batch * n * k).unwrap())
                .collect::<Vec<_>>();

            let (ref_b, metal_b) = if q40 {
                ensure!(TypeId::of::<F>() == TypeId::of::<f32>());
                let b_data: Vec<f32> = b_data.into_iter().map(|x| x.into()).collect();
                let b_tensor =
                    Q4_0.simulate_precision_loss(Tensor::from_shape(&b_shape, &b_data)?, 2)?;

                ensure!(k % 32 == 0);
                let b_q4_0_tensor = tensor0(Opaque(Arc::new(BlockQuantValue {
                    fact: BlockQuantFact::new(Box::new(Q4_0), tvec![batch, n, k]),
                    value: Arc::new(Q4_0.quant_f32(&b_data)?),
                })));
                (b_tensor, b_q4_0_tensor)
            } else {
                let b_tensor = Tensor::from_shape(&b_shape, &b_data)?;
                (b_tensor.clone(), b_tensor)
            };

            let cuda_output = Matmul.eval(
                stream,
                &a.clone().into_device()?,
                &metal_b.clone().into_device()?,
            )?;
            let output = reference(a, ref_b)?;
            cuda_output.to_host()?.close_enough(&output, Approximation::VeryApproximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_q4() -> TractResult<()> {
        run_ggml_mat_mul_test::<f32>(32, 1, 1, 256, 32, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 320, 2048, 1, true)?;
        run_ggml_mat_mul_test::<f32>(4, 1, 15, 2048, 320, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 12, 512, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 1, 8, 4096, 512, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 1, 1024, 32, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 61, 1280, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 1, 13, 2048, 128, true)?;
        run_ggml_mat_mul_test::<f32>(1, 3, 1, 256, 32, true)?;
        run_ggml_mat_mul_test::<f32>(4, 2, 7, 512, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 2, 6, 512, 256, true)?;
        Ok(())
    }
}
