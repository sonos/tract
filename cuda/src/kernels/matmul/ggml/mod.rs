use cudarc::cublas::{self, CudaBlas, Gemm};
use cudarc::driver::result::stream::null;
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::runtime::result::device::get_device_prop;
use cudarc::runtime::sys::cudaGetLastError;
use derive_new::new;
use num_traits::{Float, One};
use std::{default, fmt};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::device::get_context;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::as_quant_fact;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::kernels::{LibraryName, get_cuda_view, launch_args};
use crate::tensor::CudaTensor;
use crate::utils::get_quant_fact;
use crate::{Q40_ROW_PADDING, context};

use DatumType::{F16, F32};

static N_WARPS: usize = 8;
static WARP_SIZE: usize = 32;

static QUANTIZE_BLOCK_SIZE: usize = 256;
static QUANTIZE_BLOCK_SIZE_MMQ: usize = 128;

static MMQ_X_MAX: usize = 128;

static QK8_0: usize = 32;
static QI8_0: usize = QK8_0 / (4 * QR8_0);
static QR8_0: usize = 1;

static QK8_1: usize = 32;

static MMQ_MMA_TILE_X_K_Q8_0: usize = (2 * WARP_SIZE + 2 * WARP_SIZE / QI8_0 + 4);

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

fn mmq_get_nbytes_shared_q40(mmq_x: usize, mmq_y: usize) -> usize {
    let nb_ids = mmq_x * size_of::<i32>();
    let mmq_tile_x_l = MMQ_MMA_TILE_X_K_Q8_0;
    let nbs_x = mmq_y * mmq_tile_x_l * size_of::<i32>();
    let nbs_y = mmq_x * 144;

    let pad = N_WARPS * WARP_SIZE * size_of::<i32>();
    nb_ids + nbs_x + nbs_y.next_multiple_of(pad)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GgmlGemm;

impl fmt::Display for GgmlGemm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GgmlGemm")
    }
}

impl GemmKernel for GgmlGemm {
    fn name() -> &'static str {
        "ggml"
    }

    fn supports_broadcast(
        a_batch: usize,
        b_batch: usize,
        m: usize,
        k: usize,
        _n: usize,
        is_q40: bool,
    ) -> bool {
        (a_batch % b_batch == 0) && (is_q40 || ((k % 2 == 0) && m <= 8))
    }

    fn is_supported_dts(&self, facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Ggml: Expected 2 inputs for Matmul");

        let regular_types_support = matches!(
            (facts[0].datum_type, facts[1].datum_type),
            (F32, F32) | (F16, F16) | (F16, F32)
        );

        regular_types_support
            || (as_quant_fact(&facts[1], &Q4_0).is_some()
                && matches!(facts[0].datum_type, F16 | F32))
    }

    fn output_dt(&self, a_dt: DatumType, _b_dt: DatumType) -> TractResult<DatumType> {
        Ok(a_dt)
    }

    fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        params: GemmDispatchParams,
        a_view: &CudaView<'_, u8>,
        b_view: &CudaView<'_, u8>,
        c_view_mut: &mut CudaViewMut<'_, u8>,
    ) -> TractResult<()> {
        let GemmDispatchParams {
            dts,
            a_batch,
            m,
            k,
            transpose_a,
            a_offset,
            transpose_b,
            b_offset,
            c_offset,
            q40_b,
            ..
        } = params;

        ensure!(!transpose_a && transpose_b);

        let c_view = &c_view_mut.as_view();
        if q40_b {
            if params.m <= 8 {
                dispatch_ggml_matvec_q40(stream, a_view, b_view, c_view, params)?;
            } else {
                dispatch_ggml_matmul_q40(stream, a_view, b_view, c_view, params)?;
            }
        } else if (params.k % 2 == 0) && params.m <= 8 {
            dispatch_ggml_matvec(stream, a_view, b_view, c_view, params)?;
        } else if dts[0] == DatumType::F32 {
            dispatch_cublas_gemm::<f32>(stream, a_view, b_view, c_view_mut, params)?;
        } else {
            ensure!(dts[0] == F16);
            dispatch_cublas_gemm::<f16>(stream, a_view, b_view, c_view_mut, params)?;
        }
        Ok(())
    }
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
    a: &CudaView<'_, u8>,
    b: &CudaView<'_, u8>,
    output: &CudaView<'_, u8>,
    params: GemmDispatchParams,
) -> TractResult<()> {
    let k_div_2 = params.k / 2;
    let ncols_y_div_2 = params.a_strides[1] / 2;
    let block_size = find_block_size(params.k);

    let batch_ratio = params.a_batch / params.b_batch;

    let kernel_name = format!(
        "ggml_matvec_{}_ncols_{}_bs_{}",
        DeviceTensor::tname(params.dts[0])?,
        params.m,
        block_size
    );
    let mut func = cuda_context().load_pipeline(LibraryName::Ggml, kernel_name)?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(b);
    launch_args.arg(a);
    launch_args.arg(output);
    launch_args.arg(&k_div_2);
    launch_args.arg(&params.a_batch);
    launch_args.arg(&params.b_strides[1]);
    launch_args.arg(&ncols_y_div_2);
    launch_args.arg(&params.c_strides[1]);
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
    a: &CudaView<'_, u8>,
    b: &CudaView<'_, u8>,
    output: &mut CudaViewMut<'_, u8>,
    params: GemmDispatchParams,
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

    let gemm_batched_strided_cfg = cublas::StridedBatchedConfig {
        gemm: cublas_gemm_cfg,
        batch_size: params.a_batch as i32,
        stride_a: b_batch_stride as _,
        stride_b: a_batch_stride as _,
        stride_c: c_batch_stride as _,
    };

    ensure!((a.len() % size_of::<F>() == 0) && (b.len() % size_of::<F>() == 0));
    unsafe {
        stream.cublas().gemm_strided_batched(
            gemm_batched_strided_cfg,
            &b.transmute::<F>(b.len() / size_of::<F>()).unwrap(),
            &a.transmute::<F>(a.len() / size_of::<F>()).unwrap(),
            &mut output.transmute_mut::<F>(output.len() / size_of::<F>()).unwrap(),
        )
    };
    Ok(())
}

fn kernel_name_q40(
    params: &GemmDispatchParams,
    mmq_x: usize,
    mmq_y: usize,
    fixup: bool,
) -> TractResult<String> {
    let need_check = params.n % mmq_y != 0;
    let fixup_str = if fixup { "stream_k_fixup_" } else { "" };
    Ok(format!("mul_mat_q40_{fixup_str}{mmq_x}_8_{need_check}"))
}

fn find_best_mmq_x(smbpo: usize, m: usize) -> usize {
    let mut mmq_x_best = 0;
    let mut ntiles_x_best = usize::MAX;

    let mut mmq_x = 0;
    while mmq_x <= MMQ_X_MAX && ntiles_x_best > 1 {
        mmq_x += 8;
        let granularity = if mmq_x >= 48 { 16 } else { 8 };
        if (mmq_x % granularity != 0 || mmq_get_nbytes_shared_q40(mmq_x, MMQ_X_MAX) > smbpo) {
            continue;
        }
        let ntiles_x = m.div_ceil(mmq_x);
        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }
    mmq_x_best
}

fn launch_quantize_q8_1(
    stream: &TractCudaStream,
    a_view: &CudaView<'_, u8>,
    quant_a_view: &CudaView<'_, u8>,
    params: &GemmDispatchParams,
    sample_stride_a: isize,
    padded_k: usize,
) -> TractResult<()> {
    let func = cuda_context().load_pipeline(LibraryName::GgmlQ, "quantize_mmq_q8_1".to_string())?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(a_view);
    launch_args.arg(quant_a_view);
    launch_args.arg(&params.k);
    launch_args.arg(&params.a_strides[1]);
    launch_args.arg(&params.a_strides[0]);
    launch_args.arg(&sample_stride_a);
    launch_args.arg(&padded_k);
    launch_args.arg(&params.m);
    launch_args.arg(&params.a_batch);

    let cfg = LaunchConfig {
        grid_dim: (
            params.m as _,
            padded_k.div_ceil(4 * QUANTIZE_BLOCK_SIZE_MMQ) as _,
            params.a_batch as _,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { launch_args.launch(cfg) };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_matmul_q40(
    stream: &TractCudaStream,
    weights: &CudaView<'_, u8>,
    quant_activ: &CudaView<'_, u8>,
    output: &CudaView<'_, u8>,
    fixup_tens: &CudaView<'_, u8>,
    params: &GemmDispatchParams,
    a_stride_0: usize,
    b_stride_0: usize,
    batch_ratio: usize,
    mmq_x_best: usize,
    nbytes_shared: usize,
) -> TractResult<()> {
    let n_blocks = b_stride_0 / params.n;
    let kernel_name = kernel_name_q40(params, mmq_x_best, MMQ_X_MAX, false)?;

    let context = cuda_context();
    let props = context.properties();
    let func = context.load_pipeline(LibraryName::GgmlQ, kernel_name)?;
    func.set_attribute(
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        nbytes_shared as i32,
    )?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(weights);
    launch_args.arg(quant_activ);
    launch_args.arg(output);
    launch_args.arg(fixup_tens);
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
        grid_dim: (props.multiProcessorCount as usize as _, 1, 1),
        block_dim: (WARP_SIZE as _, N_WARPS as _, 1),
        shared_mem_bytes: nbytes_shared as _,
    };

    unsafe {
        launch_args.launch(cfg);
    }
    Ok(())
}

fn launch_fixup_q40(
    stream: &TractCudaStream,
    output: &CudaView<'_, u8>,
    fixup_tens: &CudaView<'_, u8>,
    params: &GemmDispatchParams,
    mmq_x_best: usize,
) -> TractResult<()> {
    let kernel_name = kernel_name_q40(params, mmq_x_best, MMQ_X_MAX, true)?;

    let context = cuda_context();
    let props = context.properties();
    let func = context.load_pipeline(LibraryName::GgmlQ, kernel_name)?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(output);
    launch_args.arg(fixup_tens);
    launch_args.arg(&params.k);
    launch_args.arg(&params.n);
    launch_args.arg(&params.m);
    launch_args.arg(&params.n);
    launch_args.arg(&params.a_batch);
    launch_args.arg(&params.c_strides[0]);

    let cfg = LaunchConfig {
        grid_dim: (props.multiProcessorCount as usize as _, 1, 1),
        block_dim: (WARP_SIZE as _, N_WARPS as _, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_args.launch(cfg);
    }
    Ok(())
}

fn dispatch_ggml_matmul_q40(
    stream: &TractCudaStream,
    a: &CudaView<'_, u8>,
    b: &CudaView<'_, u8>,
    output: &CudaView<'_, u8>,
    params: GemmDispatchParams,
) -> TractResult<()> {
    let context = cuda_context();
    let props = context.properties();

    let null_ptr = stream.null::<u8>()?;

    let padded_k = params.k.next_multiple_of(Q40_ROW_PADDING);
    let n_blocks = padded_k / Q4_0.block_len(); // padded Q40 weights
    let sample_stride_a = params.a_strides[0] * params.a_batch as isize;

    let nbytes_a_q8_1 = (a.len() / params.dts[0].size_of()) * padded_k * (QK8_0 + 4)
        / (params.k * QK8_0)
        + MMQ_X_MAX * 144;

    let quant_a = unsafe { DeviceTensor::uninitialized_dt(DatumType::U8, &[nbytes_a_q8_1])? };
    let quant_a_view = get_cuda_view(&quant_a);

    launch_quantize_q8_1(stream, a, &quant_a_view, &params, sample_stride_a, padded_k)?;

    let a_stride_0 = padded_k * params.m * 36 / 128;
    let b_stride_0 = n_blocks * params.n;
    let batch_ratio = params.a_batch / params.b_batch;

    let mmq_x_best = find_best_mmq_x(props.sharedMemPerBlockOptin, params.m);
    let nbytes_shared = mmq_get_nbytes_shared_q40(mmq_x_best, MMQ_X_MAX);

    let ntx = params.m.div_ceil(mmq_x_best);
    let nty = params.n.div_ceil(MMQ_X_MAX);

    let fixup_tensor = {
        let needs_fixup = (ntx * nty * params.a_batch) % props.multiProcessorCount as usize != 0;

        needs_fixup
            .then(|| {
                let fixup_shape = props.multiProcessorCount as usize * mmq_x_best * MMQ_X_MAX;
                unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[fixup_shape]) }
            })
            .transpose()?
    };

    let fixup_view = fixup_tensor.as_ref().map(|t| get_cuda_view(t)).unwrap_or(null_ptr.as_view());

    launch_matmul_q40(
        stream,
        b,
        &quant_a_view,
        output,
        &fixup_view,
        &params,
        a_stride_0,
        b_stride_0,
        batch_ratio,
        mmq_x_best,
        nbytes_shared,
    )?;

    if let Some(ref fixup) = fixup_tensor {
        launch_fixup_q40(stream, output, &fixup_view, &params, mmq_x_best)?;
    }

    Ok(())
}

fn dispatch_ggml_matvec_q40(
    stream: &TractCudaStream,
    a: &CudaView<'_, u8>,
    b: &CudaView<'_, u8>,
    output: &CudaView<'_, u8>,
    params: GemmDispatchParams,
) -> TractResult<()> {
    let context = cuda_context();
    let props = context.properties();

    let null_ptr = stream.null::<u8>()?;

    let padded_k = params.k.next_multiple_of(Q40_ROW_PADDING);
    let n_blocks = padded_k / Q4_0.block_len(); // padded Q40 weights
    let sample_stride_a = params.a_strides[0] * params.a_batch as isize;

    let nbytes_a_q8_1 = a.len() * padded_k * (QK8_1 + 4) / (params.k * QK8_1);

    let quant_a = unsafe { DeviceTensor::uninitialized_dt(DatumType::I8, &[nbytes_a_q8_1])? };
    let quant_a_view = get_cuda_view(&quant_a);

    let func = context.load_pipeline(LibraryName::GgmlQ, "quantize_q8_1".to_string())?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(a);
    launch_args.arg(&quant_a_view);
    launch_args.arg(&params.k);
    launch_args.arg(&params.a_strides[1]);
    launch_args.arg(&params.a_strides[0]);
    launch_args.arg(&sample_stride_a);
    launch_args.arg(&padded_k);
    launch_args.arg(&params.m);
    launch_args.arg(&params.a_batch);

    let cfg = LaunchConfig {
        grid_dim: (padded_k.div_ceil(QUANTIZE_BLOCK_SIZE) as _, params.m as _, params.a_batch as _),
        block_dim: (QUANTIZE_BLOCK_SIZE as _, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { launch_args.launch(cfg) };

    let stride_col_y = padded_k / QK8_1;
    let stride_col_dst = params.n;
    let stride_channel_x = n_blocks * params.n;
    let stride_channel_y = stride_col_y * params.m;
    let stride_channel_dst = params.m * params.n;

    let batch_ratio = params.a_batch / params.b_batch;

    let func = context.load_pipeline(LibraryName::GgmlQ, format!("mul_vec_q40_m_{}", params.m))?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(b);
    launch_args.arg(&quant_a_view);
    launch_args.arg(output);
    launch_args.arg(&params.k);
    launch_args.arg(&params.a_batch);
    launch_args.arg(&n_blocks);
    launch_args.arg(&stride_col_y);
    launch_args.arg(&stride_col_dst);
    launch_args.arg(&batch_ratio);
    launch_args.arg(&stride_channel_x);
    launch_args.arg(&stride_channel_y);
    launch_args.arg(&stride_channel_dst);

    let rows_per_block = if params.m == 1 { 1 } else { 2 };
    let n_warps = if params.m <= 4 { 4 } else { 2 };
    let cfg = LaunchConfig {
        grid_dim: (params.n.div_ceil(rows_per_block) as _, params.a_batch as _, 1 as _),
        block_dim: (WARP_SIZE as _, n_warps, 1),
        shared_mem_bytes: 0,
    };

    unsafe { launch_args.launch(cfg) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use crate::context::CUDA_STREAM;
    use crate::kernels::matmul::GemmImpl;
    use crate::kernels::matmul::tests::run_mmm_test_case;

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

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case::<GgmlGemm>((1, 1, 1, 2, 1), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 1, 1, 60, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 2, 1, 128, 7), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((4, 1, 1, 2, 1), false, true, F32, F32)?;

        //// f16_f16
        run_mmm_test_case::<GgmlGemm>((1, 1, 1, 2, 1), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 1, 1, 61, 2), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 2, 1, 128, 9), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((4, 1, 1, 128, 9), false, true, F16, F16)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case::<GgmlGemm>((1, 1, 9, 4, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 1, 11, 2, 3), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 2, 15, 1, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 1, 10, 32, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 2, 12, 1, 2), false, true, F32, F32)?;

        // f16_f16
        run_mmm_test_case::<GgmlGemm>((1, 1, 12, 7, 2), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 1, 9, 61, 2), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 1, 10, 127, 9), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 2, 16, 127, 9), false, true, F16, F16)?;
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

    fn run_q40_mat_mul_test(
        batch: usize,
        broadcast_ratio: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_shape = [batch * broadcast_ratio, m, k];
            let b_shape = [batch, n, k];

            let a_data = (0..batch * broadcast_ratio * k * m)
                .map(|f| f as f32 / (batch * broadcast_ratio * m * k) as f32)
                .collect::<Vec<_>>();

            let a = Tensor::from_shape(&a_shape, &a_data)?;

            let b_data =
                (0..batch * n * k).map(|f| f as f32 / (batch * n * k) as f32).collect::<Vec<_>>();

            let b_data: Vec<f32> = b_data.into_iter().map(|x| x.into()).collect();
            let b_tensor =
                Q4_0.simulate_precision_loss(Tensor::from_shape(&b_shape, &b_data)?, 2)?;

            ensure!(k % 512 == 0);
            let b_q4_0_tensor = tensor0(Opaque(Arc::new(BlockQuantValue {
                fact: BlockQuantFact::new(Box::new(Q4_0), tvec![batch, n, k]),
                value: Arc::new(Q4_0.quant_f32(&b_data)?),
            })));

            let cuda_output = GemmImpl::<GgmlGemm>::new(false, true).eval(
                stream,
                &a.clone().into_device()?,
                &b_q4_0_tensor.clone().into_device()?,
            )?;
            let output = reference(a, b_tensor)?;
            cuda_output.to_host()?.close_enough(&output, Approximation::VeryApproximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_q4() -> TractResult<()> {
        // MV
        run_q40_mat_mul_test(1, 1, 2, 512, 3)?;
        run_q40_mat_mul_test(3, 1, 8, 4096, 512)?;
        run_q40_mat_mul_test(1, 1, 1, 1024, 32)?;
        run_q40_mat_mul_test(1, 3, 1, 512, 32)?;
        run_q40_mat_mul_test(4, 2, 7, 512, 4)?;
        run_q40_mat_mul_test(3, 2, 6, 512, 256)?;
        //// MM
        run_q40_mat_mul_test(1, 1, 320, 2048, 1)?;
        run_q40_mat_mul_test(4, 1, 15, 2048, 320)?;
        run_q40_mat_mul_test(1, 1, 12, 512, 4)?;
        run_q40_mat_mul_test(1, 1, 61, 1024, 4)?;
        run_q40_mat_mul_test(3, 1, 13, 2048, 128)?;
        Ok(())
    }
}
