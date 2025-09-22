pub(crate) mod quant_act_q81;

use cudarc::cublas::{self, CudaBlas, Gemm};
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0, Q8_1};

use num_traits::{Float, One};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::as_quant_fact;

use crate::Q40_ROW_PADDING;
use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::matmul::quant_act_q81::{QUANTIZE_BLOCK_SIZE, QUANTIZE_BLOCK_SIZE_MMQ};
use crate::kernels::{
    LibraryName, get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, get_sliced_cuda_view_mut,
};
use crate::utils::{get_ggml_q81_fact, get_quant_fact};

use DatumType::{F16, F32};

static N_WARPS: usize = 8;
static WARP_SIZE: usize = 32;

static MMQ_X_MAX: usize = 128;

static QK8_0: usize = 32;
static QI8_0: usize = QK8_0 / (4 * QR8_0);
static QR8_0: usize = 1;

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

pub fn get_concrete_shapes(
    a: &DeviceTensor,
    b: &DeviceTensor,
) -> TractResult<(Vec<usize>, Vec<usize>)> {
    let q81_a = get_ggml_q81_fact(a);
    let q40_b = get_quant_fact(b, &Q4_0);
    ensure!(q40_b.is_none() || (q40_b.is_some() && q81_a.is_some()));

    let a_shape = match q81_a {
        Some(bqf) => {
            let concrete = bqf.concrete_in_shape()?;
            a.shape().iter().copied().chain(concrete.iter().copied()).collect()
        }
        None => a.shape().to_vec(),
    };

    let b_shape = match q40_b {
        Some(bqf) => b.shape().iter().copied().chain(bqf.shape().iter().copied()).collect(),
        None => b.shape().to_vec(),
    };

    Ok((a_shape, b_shape))
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct GemmParams {
    pub dts: [DatumType; 3],
    pub a_batch: usize,
    pub b_batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a_strides: TVec<isize>,
    pub b_strides: TVec<isize>,
    pub c_strides: TVec<isize>,
}

impl GemmParams {
    pub fn compute_gemm_params(
        dts: [DatumType; 3],
        a_shape: &[usize],
        b_shape: &[usize],
        c_shape: &[usize],
    ) -> TractResult<GemmParams> {
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

        ensure!((a_batch % b_batch == 0) || (a_batch == 1));
        let a_strides = natural_strides(&[a_batch, m, k]);
        let b_strides = natural_strides(&[b_batch, n, k]);
        let c_strides = natural_strides(&[a_batch.max(b_batch), m, n]);

        Ok(GemmParams { dts, a_batch, b_batch, m, n, k, a_strides, b_strides, c_strides })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GgmlGemm;

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

fn kernel_name_mat_vec(dt: DatumType, n_cols: usize, block_size: usize) -> TractResult<String> {
    Ok(format!("ggml_matvec_{}_ncols_{}_bs_{}", DeviceTensor::tname(dt)?, n_cols, block_size))
}

fn dispatch_ggml_matvec(
    stream: &TractCudaStream,
    a: &DeviceTensor,
    b: &DeviceTensor,
    output: &DeviceTensor,
    params: GemmParams,
) -> TractResult<()> {
    let a_view = get_cuda_view(a);
    let b_view = get_cuda_view(b);
    let output_view = get_cuda_view(output);

    let k_div_2 = params.k / 2;
    let ncols_y_div_2 = params.a_strides[1] / 2;
    let block_size = find_block_size(params.k);

    let batch_ratio = params.a_batch / params.b_batch;

    let kernel_name = kernel_name_mat_vec(params.dts[0], params.m, block_size)?;
    let mut func = cuda_context().load_pipeline(LibraryName::Ggml, kernel_name)?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(&b_view);
    launch_args.arg(&a_view);
    launch_args.arg(&output_view);
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

pub struct CublasDispatchParams {
    pub a_batch: usize,
    pub b_batch: usize,
    pub m: usize,
    pub a_offset: usize,
    pub b_offset: usize,
    pub c_offset: usize,
}

impl CublasDispatchParams {
    pub fn compute_dispatch_params(params: &GemmParams) -> TractResult<Vec<CublasDispatchParams>> {
        match (params.a_batch, params.b_batch) {
            (a_batch, 1) if a_batch != 1 => Ok(vec![CublasDispatchParams {
                a_batch: 1,
                b_batch: 1,
                m: params.m * params.a_batch,
                a_offset: 0,
                b_offset: 0,
                c_offset: 0,
            }]),
            (1, b_batch) if b_batch != 1 => Ok((0..b_batch)
                .map(|b_batch_idx| CublasDispatchParams {
                    a_batch: 1,
                    b_batch: 1,
                    m: params.m,
                    a_offset: 0,
                    b_offset: b_batch_idx * params.b_strides[0] as usize,
                    c_offset: b_batch_idx * params.m * params.n * params.dts[2].size_of(),
                })
                .collect()),
            (a_batch, b_batch) => {
                ensure!(
                    a_batch == b_batch,
                    "Only support equal batches or either batch == 1 for Cublas MM"
                );
                Ok(vec![CublasDispatchParams {
                    a_batch: params.a_batch,
                    b_batch: params.b_batch,
                    m: params.m,
                    a_offset: 0,
                    b_offset: 0,
                    c_offset: 0,
                }])
            }
        }
    }
}

fn dispatch_cublas_gemm<F: Datum + Float>(
    stream: &TractCudaStream,
    a: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
    params: GemmParams,
) -> TractResult<()>
where
    CudaBlas: Gemm<F>,
{
    let dispatch_params = CublasDispatchParams::compute_dispatch_params(&params)?;
    for d in dispatch_params {
        let a_len = params.a_strides[0] as usize * d.a_batch * params.dts[0].size_of();
        let b_len = params.b_strides[0] as usize * d.b_batch * params.dts[1].size_of();
        let a_view = get_sliced_cuda_view(a, d.a_offset, a_len)?;
        let b_view = get_sliced_cuda_view(b, d.b_offset, b_len)?;
        let mut c_view = get_sliced_cuda_view_mut(
            c,
            d.c_offset,
            params.c_strides[0] as usize * d.a_batch.max(d.b_batch) * params.dts[2].size_of(),
        )?;
        let cublas_gemm_cfg = cublas::GemmConfig {
            transa: cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: params.n as i32,
            n: d.m as i32,
            k: params.k as i32,
            alpha: F::from(1.0f32).unwrap(),
            lda: params.k as i32,
            ldb: params.k as i32,
            beta: F::from(0.0f32).unwrap(),
            ldc: params.n as i32,
        };

        let gemm_batched_strided_cfg = cublas::StridedBatchedConfig {
            gemm: cublas_gemm_cfg,
            batch_size: params.a_batch as i32,
            stride_a: params.b_strides[0] as _,
            stride_b: params.a_strides[0] as _,
            stride_c: params.c_strides[0] as _,
        };

        unsafe {
            stream.cublas().gemm_strided_batched(
                gemm_batched_strided_cfg,
                &b_view.transmute::<F>(b_view.len() / size_of::<F>()).unwrap(),
                &a_view.transmute::<F>(a_view.len() / size_of::<F>()).unwrap(),
                &mut c_view.transmute_mut::<F>(c_view.len() / size_of::<F>()).unwrap(),
            )
        };
    }

    Ok(())
}

fn kernel_name_q40(
    params: &GemmParams,
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

#[allow(clippy::too_many_arguments)]
fn launch_matmul_q40(
    stream: &TractCudaStream,
    weights: &CudaView<'_, u8>,
    quant_activ: &CudaView<'_, u8>,
    output: &CudaView<'_, u8>,
    fixup_tens: &CudaView<'_, u8>,
    params: &GemmParams,
    a_batch_stride: usize,
    b_batch_stride: usize,
    batch_ratio: usize,
    mmq_x_best: usize,
    nbytes_shared: usize,
) -> TractResult<()> {
    let n_blocks = b_batch_stride / params.n;
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
    launch_args.arg(&b_batch_stride);
    launch_args.arg(&a_batch_stride);
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
    params: &GemmParams,
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
    params: GemmParams,
) -> TractResult<()> {
    let context = cuda_context();
    let props = context.properties();

    let null_ptr = stream.null::<u8>()?;

    let padded_k = params.k.next_multiple_of(Q40_ROW_PADDING);
    let n_blocks = padded_k / Q4_0.block_len(); // padded Q40 weights

    let n_mmq_blocks = padded_k / (Q8_1.block_len() * 4);
    let a_batch_stride = n_mmq_blocks * params.m * Q8_1.block_bytes();
    let b_batch_stride = n_blocks * params.n;
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
        a,
        output,
        &fixup_view,
        &params,
        a_batch_stride,
        b_batch_stride,
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
    params: GemmParams,
) -> TractResult<()> {
    let context = cuda_context();
    let props = context.properties();
    let null_ptr = stream.null::<u8>()?;

    let padded_k = params.k.next_multiple_of(Q40_ROW_PADDING);

    let n_blocks = padded_k / Q4_0.block_len();
    let stride_col_y = padded_k / Q8_1.block_len();
    let stride_col_dst = params.n;
    let stride_channel_x = n_blocks * params.n;
    let stride_channel_y = stride_col_y * params.m;
    let stride_channel_dst = params.m * params.n;

    let batch_ratio = params.a_batch / params.b_batch;

    let func = context.load_pipeline(LibraryName::GgmlQ, format!("mul_vec_q40_m_{}", params.m))?;
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(b);
    launch_args.arg(a);
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

impl GgmlGemm {
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

    pub fn is_supported_dts(&self, facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Ggml: Expected 2 inputs for Matmul");

        let regular_types_support = matches!(
            (facts[0].datum_type, facts[1].datum_type),
            (F32, F32) | (F16, F16) | (F16, F32)
        );

        regular_types_support
            || (as_quant_fact(&facts[1], &Q4_0).is_some()
                && matches!(facts[0].datum_type, F16 | F32))
    }

    fn output_dt(&self, a_dt: DatumType, b_dt: DatumType) -> TractResult<DatumType> {
        ensure!(b_dt == a_dt);
        if a_dt == DatumType::Opaque {
            // Q40 MM -> F32 output
            Ok(DatumType::F32)
        } else {
            Ok(a_dt)
        }
    }

    pub fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        // A: [b, n, k] B: [b, m, k]
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2].clone());
        output.push(b[rank - 2].clone());
        output
    }

    pub fn output_facts(
        &self,
        shape: &[TDim],
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<TVec<TypedFact>> {
        let out_dt = self.output_dt(a_dt, b_dt)?;
        ensure!([DatumType::F16, DatumType::F32].contains(&out_dt));
        Ok(tvec!(out_dt.fact(shape)))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let (a_shape, b_shape) = get_concrete_shapes(a, b)?;

        let c_dt = self.output_dt(a.datum_type(), b.datum_type())?;
        let c_shape = self.output_shape(&a_shape, &b_shape);
        let c = unsafe { DeviceTensor::uninitialized_dt(c_dt, &c_shape)? };

        self.dispatch_eval(stream, a, b, &c)?;
        stream.synchronize()?;
        Ok(c)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
        c: &DeviceTensor,
    ) -> TractResult<()> {
        let (a_shape, b_shape) = get_concrete_shapes(a, b)?;

        ensure!(c.shape() == self.output_shape(&a_shape, &b_shape).as_slice());

        if c.shape().iter().product::<usize>() == 0 {
            return Ok(());
        }

        let params = GemmParams::compute_gemm_params(
            [a.datum_type(), b.datum_type(), c.datum_type()],
            &a_shape,
            &b_shape,
            c.shape(),
        )?;
        if get_quant_fact(b, &Q4_0).is_some() {
            let a_view = get_cuda_view(a);
            let b_view = get_cuda_view(b);
            let c_view = get_cuda_view(c);
            if params.m <= 8 {
                dispatch_ggml_matvec_q40(stream, &a_view, &b_view, &c_view, params)?;
            } else {
                dispatch_ggml_matmul_q40(stream, &a_view, &b_view, &c_view, params)?;
            }
        } else if (params.k % 2 == 0) && params.m <= 8 {
            dispatch_ggml_matvec(stream, a, b, c, params)?;
        } else if a.datum_type() == DatumType::F32 {
            dispatch_cublas_gemm::<f32>(stream, a, b, c, params)?;
        } else {
            ensure!(a.datum_type() == F16);
            dispatch_cublas_gemm::<f16>(stream, a, b, c, params)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CUDA_STREAM;
    use crate::kernels::matmul::quant_act_q81::GgmlQuantQ81;
    use crate::ops::GgmlQuantQ81Fact;
    use crate::utils::pad_q40;
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
        (a_batch, b_batch, m, k, n): (usize, usize, usize, usize, usize),
        transpose_a: bool,
        transpose_b: bool,
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_shape = if !transpose_a { [a_batch, m, k] } else { [a_batch, k, m] };
            let b_shape = if !transpose_b { [b_batch, k, n] } else { [b_batch, n, k] };
            let mut a = if a_dt == DatumType::F16 {
                Tensor::from_shape(
                    &a_shape,
                    &(0..a_batch * m * k)
                        .map(|f| f16::from_f32(f as f32 / (a_batch * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &a_shape,
                    &(0..a_batch * m * k)
                        .map(|f| f as f32 / (a_batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let mut b = if b_dt == DatumType::F16 {
                Tensor::from_shape(
                    &b_shape,
                    &(0..b_batch * k * n)
                        .map(|f| f16::from_f32(f as f32 / (b_batch * n * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &b_shape,
                    &(0..b_batch * k * n)
                        .map(|f| f as f32 / (b_batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let cuda_output =
                GgmlGemm.eval(stream, &a.clone().into_device()?, &b.clone().into_device()?)?;

            let matmul = PrefixMatMul {
                transpose_a,
                transpose_b,
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
            cuda_output.to_host()?.close_enough(&output, Approximation::VeryApproximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_squeeze_batch_axes() -> TractResult<()> {
        assert_eq!(squeeze_batch_axes(&[1, 2, 3, 4])?, tvec![2, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 2, 3, 4])?, tvec![6, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 1, 2, 3, 4])?, tvec![6, 3, 4]);
        assert!(squeeze_batch_axes(&[1]).is_err());
        assert_eq!(squeeze_batch_axes(&[1, 1, 3, 4])?, tvec![1, 3, 4]);
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn mmm_ggml_prop_f32(pb in <MmmProblem<f32>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: false,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::Approximate).is_ok())
        }

        #[test]
        fn mmm_ggml_prop_f16(pb in <MmmProblem<f16>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: false,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::VeryApproximate).is_ok())
        }

        #[test]
        fn mmm_ggml_prop_q4(pb in <MmmProblem<f32>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: true,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::VeryApproximate).is_ok())
        }
    }

    #[derive(Default, Debug, Clone)]
    pub struct MmmProblemParams {
        pub force_k_as_inner_axis: bool,
        pub q4_0_weights: bool,
    }

    #[derive(Debug)]
    pub struct MmmProblem<F: Datum + Float>
    where
        F: Datum + Float,
        f32: AsPrimitive<F>,
    {
        pub b: usize,
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub lhs: Vec<F>,
        pub transpose_lhs: bool,
        pub rhs: Vec<F>,
        pub transpose_rhs: bool,
        pub q4_0: bool,
    }

    impl<F> Arbitrary for MmmProblem<F>
    where
        F: Datum + Float,
        f32: AsPrimitive<F>,
    {
        type Parameters = MmmProblemParams;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(params: MmmProblemParams) -> Self::Strategy {
            (1usize..4, 1usize..16, 1usize..128, 1usize..16)
                .prop_flat_map(move |(b, m, mut k, n)| {
                    if params.q4_0_weights {
                        k = k.div_ceil(32) * 32
                    };

                    let lhs_len = b * m * k;
                    let rhs_len = b * n * k;
                    let datum = (0f32..1f32).prop_map(|x| x.as_());
                    (
                        Just(b),
                        Just(m),
                        Just(k),
                        Just(n),
                        vec(datum.clone(), lhs_len..=lhs_len),
                        proptest::bool::ANY,
                        vec(datum, rhs_len..=rhs_len),
                        proptest::bool::ANY,
                    )
                })
                .prop_map(move |(b, m, k, n, lhs, mut transpose_lhs, rhs, mut transpose_rhs)| {
                    if params.force_k_as_inner_axis {
                        (transpose_lhs, transpose_rhs) = (false, true);
                    }
                    Self {
                        b,
                        m,
                        k,
                        n,
                        lhs,
                        transpose_lhs,
                        rhs,
                        transpose_rhs,
                        q4_0: params.q4_0_weights,
                    }
                })
                .boxed()
        }
    }

    impl<F> MmmProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let matmul = PrefixMatMul {
                transpose_a: self.transpose_lhs,
                transpose_b: self.transpose_rhs,
                transpose_c: false,
                quantize_output: None,
            };

            let lhs_tensor = if self.transpose_lhs {
                Tensor::from_shape(&[self.b, self.k, self.m], &self.lhs)?
            } else {
                Tensor::from_shape(&[self.b, self.m, self.k], &self.lhs)?
            };
            let mut rhs_tensor = if self.transpose_rhs {
                Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?
            } else {
                Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?
            };

            if self.q4_0 {
                rhs_tensor = Q4_0.simulate_precision_loss(rhs_tensor, 2)?
            };
            let output = matmul.eval(tvec![lhs_tensor.into_tvalue(), rhs_tensor.into_tvalue()])?;

            Ok(output[0].clone().into_tensor())
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let lhs = if self.transpose_lhs {
                    Tensor::from_shape(&[self.b, self.k, self.m], &self.lhs)?.into_device()?
                } else {
                    let mut lhs =
                        Tensor::from_shape(&[self.b, self.m, self.k], &self.lhs)?.into_device()?;
                    if self.q4_0 {
                        let a_shape_tdim: ShapeFact = tvec![
                            TDim::Val(self.b as i64),
                            TDim::Val(self.m as i64),
                            TDim::Val(self.k as i64)
                        ]
                        .into();

                        let io_facts = GgmlQuantQ81Fact {
                            in_fact: a_shape_tdim.clone(),
                            out_fact: GgmlQuantQ81::output_shape_fact(&a_shape_tdim)?,
                        };

                        lhs = GgmlQuantQ81.eval(stream, &lhs, io_facts)?;
                    }
                    lhs
                };
                let rhs = if self.transpose_rhs {
                    if !self.q4_0 {
                        Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?
                    } else {
                        let b_quant = Q4_0.quant_f32(
                            &self
                                .rhs
                                .clone()
                                .into_iter()
                                .map(|x| x.to_f32().unwrap())
                                .collect_vec(),
                        )?;
                        let bqv = BlockQuantValue {
                            fact: BlockQuantFact::new(
                                Box::new(Q4_0),
                                tvec![self.b, self.n, self.k],
                            ),
                            value: Arc::new(b_quant),
                        };
                        let padded_q40 = pad_q40(&bqv)?;
                        tensor0(Opaque(Arc::new(padded_q40)))
                    }
                } else {
                    Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?
                }
                .into_device()?;

                let c = GgmlGemm.eval(stream, &lhs, &rhs)?;
                Ok(c.to_host()?.into_tensor())
            })
        }
    }
}
