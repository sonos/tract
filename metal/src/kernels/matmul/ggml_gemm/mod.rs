use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::encoder::EncoderExt;
use crate::utils::get_metal_buffer;
use crate::{LibraryName, MetalStream};
use DatumType::{F16, F32};
use anyhow::ensure;
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::{as_quant_fact, get_quant_fact};

#[derive(Debug)]
#[repr(C)]
struct GgmlGemmParams {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

impl From<GemmDispatchParams> for GgmlGemmParams {
    fn from(params: GemmDispatchParams) -> Self {
        assert!(params.a_strides.len() == 3 && params.b_strides.len() == 3);
        let a_el_size = params.dts[0].size_of();

        let b_el_size = if params.q40_b { Q4_0.block_bytes() } else { params.dts[1].size_of() };
        let mut b_strides = params.b_strides;
        if params.q40_b {
            b_strides[0] /= Q4_0.block_len() as isize;
            b_strides[1] /= Q4_0.block_len() as isize;
        };

        // Kernel produced transposed output so we swap the inputs
        GgmlGemmParams {
            ne00: params.k as i32,
            ne02: params.b_batch as i32,
            nb01: (b_strides[1] as usize * b_el_size) as u64,
            nb02: (b_strides[0] as usize * b_el_size) as u64,
            nb03: (b_strides[0] as usize * params.b_batch * b_el_size) as u64,
            ne12: params.a_batch as i32,
            nb10: (params.a_strides[2] as usize * a_el_size) as u64,
            nb11: (params.a_strides[1] as usize * a_el_size) as u64,
            nb12: (params.a_strides[0] as usize * a_el_size) as u64,
            nb13: (params.a_strides[0] as usize * params.a_batch * a_el_size) as u64,
            ne0: params.n as i32,
            ne1: params.m as i32,
            r2: (params.a_batch / params.b_batch) as i16,
            r3: 1,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
struct GgmlGemvParams {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    out_f16: i16,
}

#[derive(Debug)]
#[repr(C)]
struct RoutedQ40F32Params {
    k: i32,
    n: i32,
    route_count: i32,
    input_mode: i32,
    weight_expert_stride: u64,
    weight_row_stride: u64,
    input_row_stride: u64,
    output_route_stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutedQ40InputMode {
    TokenRows,
    RouteRows,
}

/// Activation epilogue of the fused routed w1/w3 swiglu (must match
/// routed_swiglu_args in ggml_mm_mv.metal).
#[derive(Debug)]
#[repr(C)]
struct RoutedSwigluParams {
    act_mode: i32,
    has_bias: i32,
    alpha: f32,
    limit: f32,
}

/// Activation of the fused routed w1/w3 pair: plain swiglu silu(g)*u or the
/// clamped gpt-oss variant ((clamp(u)+1) * min(g,limit)*sigmoid(alpha*g)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutedSwigluAct {
    Plain,
    Clamped { alpha: f32, limit: f32 },
}

impl RoutedSwigluAct {
    fn params(&self, has_bias: bool) -> RoutedSwigluParams {
        match self {
            RoutedSwigluAct::Plain => RoutedSwigluParams {
                act_mode: 0,
                has_bias: has_bias as i32,
                alpha: 0.,
                limit: 0.,
            },
            RoutedSwigluAct::Clamped { alpha, limit } => RoutedSwigluParams {
                act_mode: 1,
                has_bias: has_bias as i32,
                alpha: *alpha,
                limit: *limit,
            },
        }
    }
}

impl From<GemmDispatchParams> for GgmlGemvParams {
    fn from(params: GemmDispatchParams) -> Self {
        assert!(params.a_strides.len() == 3 && params.b_strides.len() == 3);
        let a_el_size = params.dts[0].size_of();

        let b_el_size = if params.q40_b { Q4_0.block_bytes() } else { params.dts[1].size_of() };
        let mut b_strides = params.b_strides;
        if params.q40_b {
            b_strides[0] /= Q4_0.block_len() as isize;
            b_strides[1] /= Q4_0.block_len() as isize;
        };

        // Kernel produced transposed output so we swap the inputs
        GgmlGemvParams {
            ne00: params.k as i32,
            ne01: params.n as i32,
            ne02: params.b_batch as i32,
            nb00: (b_strides[2] as usize * b_el_size) as u64,
            nb01: (b_strides[1] as usize * b_el_size) as u64,
            nb02: (b_strides[0] as usize * b_el_size) as u64,
            nb03: (b_strides[0] as usize * params.b_batch * b_el_size) as u64,
            ne10: params.k as i32,
            ne11: params.m as i32,
            ne12: params.a_batch as i32,
            nb10: (params.a_strides[2] as usize * a_el_size) as u64,
            nb11: (params.a_strides[1] as usize * a_el_size) as u64,
            nb12: (params.a_strides[0] as usize * a_el_size) as u64,
            nb13: (params.a_strides[0] as usize * params.a_batch * a_el_size) as u64,
            ne0: params.n as i32,
            ne1: params.m as i32,
            r2: (params.a_batch / params.b_batch) as i16,
            r3: 1,
            out_f16: (params.dts[0] == F16) as i16,
        }
    }
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

    fn supports_broadcast() -> bool {
        true
    }

    fn is_supported_dts(&self, facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Ggml: Expected 2 inputs for Matmul");

        let regular_types_support = facts.iter().all(|f| f.is_plain())
            && matches!(
                (facts[0].datum_type, facts[1].datum_type),
                (F32, F32) | (F16, F16) | (F32, F16)
            );

        regular_types_support
            || (as_quant_fact(&facts[1], &Q4_0).is_some()
                && facts[0].is_plain()
                && matches!(facts[0].datum_type, F16 | F32))
    }

    fn output_dt(&self, a_dt: DatumType, _b_dt: DatumType) -> TractResult<DatumType> {
        // Output dtype follows the activation (input[0]).
        Ok(a_dt)
    }

    fn dispatch_eval(
        &self,
        stream: &MetalStream,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
    ) -> TractResult<()> {
        let (mut a_buffer, mut b_buffer) = (a_buffer, b_buffer);
        // A gemv-shaped problem can reach us in weight-first orientation:
        // A = weights [m rows, k], B = a single activation row [n=1, k]
        // (e.g. qwen3.5 linear-attention in-projections at decode, exported
        // as W @ x). The tiled GEMM kernel wastes 63/64 of each 32x64 tile
        // on it. C [m,1] and C^T [1,m] are byte-identical, so swap the
        // operands and let the bandwidth-bound mat-vec kernel handle it.
        let params = if params.n == 1
            && params.m > 4
            && params.a_batch == 1
            && params.b_batch == 1
            && !params.transpose_a
            && params.transpose_b
            && !params.q40_b
            && params.dts[0] == params.dts[1]
        {
            std::mem::swap(&mut a_buffer, &mut b_buffer);
            GemmDispatchParams {
                dts: params.dts,
                a_batch: 1,
                b_batch: 1,
                m: 1,
                n: params.m,
                k: params.k,
                transpose_a: false,
                a_offset: params.b_offset,
                transpose_b: true,
                b_offset: params.a_offset,
                q40_b: false,
                c_offset: params.c_offset,
                a_strides: natural_strides(&[1, 1, params.k]),
                b_strides: natural_strides(&[1, params.m, params.k]),
            }
        } else {
            params
        };

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

        // The q4_0 matrix-vector kernel stays bandwidth-bound (one weight read)
        // for several activation rows, so it beats the tiled GEMM until ~8 rows;
        // the f16/f32 mat-vec kernels are tuned for 4.
        let gemv_max_rows = if q40_b { 8 } else { 4 };
        if matches!(dts[0], F32 | F16)
            && (k % 32 == 0)
            && (k >= 64)
            && ((m > gemv_max_rows) || (q40_b && a_batch > 1))
        {
            dispatch_metal_ggml_gemm(
                stream, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        } else {
            dispatch_metal_ggml_gemv(
                stream, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        }

        Ok(())
    }
}

fn mv_kernel_name_and_dispatch_params(
    params: &GemmDispatchParams,
) -> TractResult<(String, (u64, u64, u64))> {
    if params.q40_b {
        ensure!(matches!(params.dts[0], F32 | F16));
        // Activation/output dtype is carried at runtime by GgmlGemvParams::out_f16.
        Ok(("kernel_mul_mv_q4_0".to_string(), (8, 8, 1)))
    } else if params.dts[1] == F32 {
        if params.dts[0] == F16 {
            // f32 weights, f16 activations, full-precision f32 output (the
            // MoE router score path; bit-identical to upcasting first).
            ensure!(params.dts[2] == F32);
            Ok(("kernel_mul_mv_f32_f16_of32".to_string(), (32, 1, 4)))
        } else {
            ensure!(params.dts[0] == F32);
            Ok(("kernel_mul_mv_f32_f32".to_string(), (32, 1, 4)))
        }
    } else if params.dts[1] == F16 {
        if params.dts[0] == F32 {
            if (params.m * params.a_batch) < 4 {
                Ok(("kernel_mul_mv_f16_f32_1row".to_string(), (32, 1, 1)))
            } else if (params.k >= 128) && params.k.is_multiple_of(4) && (params.n >= 8) {
                Ok(("kernel_mul_mv_f16_f32_l4".to_string(), (32, 1, params.m as u64)))
            } else {
                Ok(("kernel_mul_mv_f16_f32".to_string(), (32, 1, 4)))
            }
        } else {
            // Never used in practice since we upcast input[0] to f32
            ensure!(params.dts[0] == F16);
            Ok(("kernel_mul_mv_f16_f16".to_string(), (32, 1, 4)))
        }
    } else {
        bail!("Unsupported dtype combination for GGML gemv: dts={:?}", params.dts);
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_metal_ggml_gemv(
    stream: &MetalStream,
    params: GemmDispatchParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> TractResult<()> {
    let (name, (nth0, nth1, nrows)) = mv_kernel_name_and_dispatch_params(&params)?;
    //dbg!(&name);
    let pipeline = stream.load_pipeline(LibraryName::Ggml, &name)?;

    let ggml_params: GgmlGemvParams = params.clone().into();
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlGemvParams>() as u64,
            &ggml_params as *const _ as *const _,
        );

        // Kernel produced transposed output so we swap the inputs
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let grid_size = if !params.q40_b {
            MTLSize {
                width: params.n as u64,
                height: (params.m as u64).div_ceil(nrows),
                depth: /* batch_size_out */ params.a_batch as u64,
            }
        } else {
            MTLSize {
                width: (params.n as u64).div_ceil(8),
                height: params.m as u64,
                depth: /* batch_size_out */ params.a_batch as u64,
            }
        };
        let group_size = MTLSize { width: nth0, height: nth1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, group_size);
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_metal_ggml_gemm(
    stream: &MetalStream,
    params: GemmDispatchParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> TractResult<()> {
    let GemmDispatchParams { dts, q40_b, .. } = params;

    ensure!((matches!(dts[1], F32 | F16) || q40_b) && matches!(dts[0], F32 | F16));

    // The GEMM is templated on both weight (i1) and activation/output (i2)
    // dtype: a single fat runtime-branched kernel regressed prefill and PSO
    // load on apple-m1-max (the dead f16-output path inflates the f32 kernel's
    // footprint), so each combination gets its own specialized kernel.
    let i1_tname = if !q40_b { DeviceTensor::tname(dts[1])? } else { "q4_0" };
    let i2_tname = DeviceTensor::tname(dts[0])?;

    let name = format!("kernel_mul_mm_{i1_tname}_{i2_tname}");
    if std::env::var_os("TRACT_METAL_LOG_GEMM").is_some() {
        eprintln!(
            "ggml-gemm {name} m={} n={} k={} a_batch={} b_batch={}",
            params.m, params.n, params.k, params.a_batch, params.b_batch
        );
    }
    let pipeline = stream.load_pipeline(LibraryName::Ggml, &name)?;

    let ggml_params: GgmlGemmParams = params.clone().into();
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlGemmParams>() as u64,
            &ggml_params as *const _ as *const _,
        );

        // Kernel produced transposed output so we swap the inputs
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let grid_size = MTLSize {
            width: (params.m as u64).div_ceil(32),
            height: (params.n as u64).div_ceil(64),
            depth: /* batch_size_out */ params.a_batch as u64,
        };
        let group_size = MTLSize { width: 128, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(grid_size, group_size);
    });

    Ok(())
}

/// Raw q8_0 GEMV against an externally managed block buffer (the GPT-OSS
/// KV-cache q8 shadow): C[batch, m, n] (f16, row stride `c_row_stride`
/// elements) = A[batch, m, k] (f16) x B^T, where B is `n` rows of `k/32`
/// q8_0 blocks. `k` must be a multiple of 32 (callers zero-pad).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mul_mv_q8_0(
    stream: &MetalStream,
    b: &DeviceTensor,
    b_offset: usize,
    b_row_stride_bytes: usize,
    b_batch_stride_bytes: usize,
    n: usize,
    k: usize,
    batch: usize,
    a: &DeviceTensor,
    a_offset: usize,
    a_row_stride_bytes: usize,
    a_batch_stride_bytes: usize,
    m: usize,
    c: &DeviceTensor,
    c_offset: usize,
    c_row_stride: usize,
) -> TractResult<()> {
    ensure!(k % 32 == 0, "q8_0 gemv needs k % 32 == 0, got {k}");
    stream.retain_tensor(a);
    stream.retain_tensor(b);
    stream.retain_tensor(c);

    let params = GgmlGemvParams {
        ne00: k as i32,
        ne01: n as i32,
        ne02: batch as i32,
        nb00: 0,
        nb01: b_row_stride_bytes as u64,
        nb02: b_batch_stride_bytes as u64,
        nb03: (b_batch_stride_bytes * batch) as u64,
        ne10: k as i32,
        ne11: m as i32,
        ne12: batch as i32,
        nb10: 2,
        nb11: a_row_stride_bytes as u64,
        nb12: a_batch_stride_bytes as u64,
        nb13: (a_batch_stride_bytes * batch) as u64,
        ne0: c_row_stride as i32,
        ne1: m as i32,
        r2: 1,
        r3: 1,
        out_f16: 1,
    };
    let pipeline = stream.load_pipeline(LibraryName::Ggml, "kernel_mul_mv_q8_0")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlGemvParams>() as u64,
            &params as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(get_metal_buffer(b)), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(get_metal_buffer(a)), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(get_metal_buffer(c)), c_offset as NSUInteger);
        // N_DST(4) x N_SIMDGROUP(2) rows per threadgroup, one r1 per grid.y.
        let grid = MTLSize {
            width: (n as u64).div_ceil(8),
            height: m as u64,
            depth: batch as u64,
        };
        let group = MTLSize { width: 8, height: 8, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Split-K f16 GEMV: C_partial[head, chunk, m, n] = A[head, m, k_chunk(c)] x
/// B_chunk^T, expressed through the kernel's existing two-level batch dims
/// (i12 = chunk via nb02/nb12, i13 = head via nb03/nb13). The caller reduces
/// the chunk axis afterwards. Strides in bytes; B rows must be k-contiguous.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mul_mv_f16_split_k(
    stream: &MetalStream,
    b: &DeviceTensor,
    b_offset: usize,
    b_row_stride: usize,
    b_head_stride: usize,
    n: usize,
    k_total: usize,
    k_chunk: usize,
    heads: usize,
    a: &DeviceTensor,
    a_offset: usize,
    a_row_stride: usize,
    a_head_stride: usize,
    m: usize,
    partial: &DeviceTensor,
) -> TractResult<()> {
    let chunks = k_total.div_ceil(k_chunk);
    ensure!(k_chunk % 32 == 0, "split-k chunk must be a multiple of 32");
    // The last chunk may be short: the kernel reads exactly ne00 elements, so
    // callers must guarantee k_total % k_chunk == 0 (pad upstream) or accept
    // that we require it here.
    ensure!(k_total % k_chunk == 0, "split-k needs k_total % k_chunk == 0");
    stream.retain_tensor(a);
    stream.retain_tensor(b);
    stream.retain_tensor(partial);

    let params = GgmlGemvParams {
        ne00: k_chunk as i32,
        ne01: n as i32,
        ne02: (heads * chunks) as i32,
        nb00: 2,
        nb01: b_row_stride as u64,
        nb02: (k_chunk * 2) as u64,       // chunk step inside a B row region
        nb03: b_head_stride as u64,       // head step
        ne10: k_chunk as i32,
        ne11: m as i32,
        ne12: chunks as i32,
        nb10: 2,
        nb11: a_row_stride as u64,
        nb12: (k_chunk * 2) as u64,       // chunk step inside an A row region
        nb13: a_head_stride as u64,       // head step
        ne0: n as i32,
        ne1: m as i32,
        r2: 1,
        r3: 1,
        out_f16: 1,
    };
    let pipeline = stream.load_pipeline(LibraryName::Ggml, "kernel_mul_mv_f16_f16")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlGemvParams>() as u64,
            &params as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(get_metal_buffer(b)), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(get_metal_buffer(a)), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(get_metal_buffer(partial)), 0);
        // f16_f16 gemv: (nth0, nth1, nrows) = (32, 1, 4).
        let grid = MTLSize {
            width: n as u64,
            height: (m as u64).div_ceil(4),
            depth: (heads * chunks) as u64,
        };
        let group = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

pub fn eval_routed_q40_f32(
    stream: &MetalStream,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    route_token_ids: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    input_mode: RoutedQ40InputMode,
) -> TractResult<DeviceTensor> {
    ensure!(route_token_ids.rank() == 1);
    let routes = route_token_ids.shape()[0];
    ensure!(weights.rank() == 3);
    let n = weights.shape()[1];
    let output = unsafe { DeviceTensor::uninitialized_dt(F32, &[routes, n])? };
    dispatch_routed_q40_f32(
        stream,
        input,
        weights,
        route_token_ids,
        route_expert_ids,
        input_mode,
        &output,
    )?;
    stream.wait_until_completed()?;
    Ok(output)
}

pub fn dispatch_routed_q40_f32(
    stream: &MetalStream,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    route_token_ids: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    input_mode: RoutedQ40InputMode,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    stream.retain_tensor(route_token_ids);
    stream.retain_tensor(route_expert_ids);
    stream.retain_tensor(output);

    ensure!(input.rank() == 2, "routed q40 input must be [rows,k], got {:?}", input.shape());
    ensure!(input.datum_type() == F32, "routed q40 input must be f32");
    ensure!(
        route_token_ids.rank() == 1 && route_expert_ids.rank() == 1,
        "routed q40 route ids must be rank-1"
    );
    ensure!(
        route_token_ids.datum_type() == i64::datum_type()
            && route_expert_ids.datum_type() == i64::datum_type(),
        "routed q40 route ids must be i64"
    );
    ensure!(route_token_ids.shape() == route_expert_ids.shape());
    ensure!(
        weights.rank() == 3 && get_quant_fact(weights, &Q4_0).is_some(),
        "routed q40 weights must be Q4_0 [experts,n,k], got {:?}",
        weights.shape()
    );
    ensure!(output.rank() == 2 && output.datum_type() == F32);

    let route_count = route_token_ids.shape()[0];
    let n = weights.shape()[1];
    let k = weights.shape()[2];
    ensure!(input.shape()[1] == k, "input k {} != weight k {k}", input.shape()[1]);
    ensure!(output.shape() == [route_count, n]);
    ensure!(k % Q4_0.block_len() == 0, "routed q40 k must be divisible by 32");
    if input_mode == RoutedQ40InputMode::RouteRows {
        ensure!(
            input.shape()[0] == route_count,
            "route-row input has {} rows but route metadata has {route_count}",
            input.shape()[0]
        );
    }
    if route_count == 0 || n == 0 {
        return Ok(());
    }

    let block_count = k / Q4_0.block_len();
    let weight_row_stride = block_count * Q4_0.block_bytes();
    let weight_expert_stride = n * weight_row_stride;
    let input_row_stride = input.strides()[0] as usize * input.datum_type().size_of();
    let output_route_stride = output.strides()[0] as usize * output.datum_type().size_of();

    let params = RoutedQ40F32Params {
        k: k as i32,
        n: n as i32,
        route_count: route_count as i32,
        input_mode: match input_mode {
            RoutedQ40InputMode::TokenRows => 0,
            RoutedQ40InputMode::RouteRows => 1,
        },
        weight_expert_stride: weight_expert_stride as u64,
        weight_row_stride: weight_row_stride as u64,
        input_row_stride: input_row_stride as u64,
        output_route_stride: output_route_stride as u64,
    };

    // Prefill-sized route lists go through the expert-grouped path: bin the
    // routes by expert (single-threadgroup counting sort), then let each
    // threadgroup amortize every weight read across 32 routes of one expert.
    // The per-route kernel re-reads an expert's weights once per route, which
    // multiplies weight traffic by routes-per-expert (~64x on a 512-token
    // top-4 prefill chunk).
    const GROUPED_MIN_ROUTES: usize = 64;
    let n_experts = weights.shape()[0];
    // Default on: halves 2800-token prefill vs the per-route gemv (11.5 ->
    // 5.9 s on gpt-oss-20b) by running the ALU-bound expert matmuls through
    // the simdgroup-matrix pipeline. Weights pass through f16 in that
    // pipeline, same precision as every dense q40 matmul in the model.
    if route_count >= GROUPED_MIN_ROUTES
        && n_experts <= 256
        && k % 32 == 0
        && std::env::var_os("TRACT_METAL_DISABLE_GROUPED_MOE").is_none()
    {
        // Worst case every expert has a ragged tail chunk.
        let max_chunks = route_count.div_ceil(32) + n_experts;
        let offsets = unsafe {
            DeviceTensor::uninitialized_dt(u32::datum_type(), &[n_experts + 1])?
        };
        let sorted = unsafe { DeviceTensor::uninitialized_dt(u32::datum_type(), &[route_count])? };
        let chunks =
            unsafe { DeviceTensor::uninitialized_dt(u32::datum_type(), &[3 * max_chunks])? };
        stream.retain_tensor(&offsets);
        stream.retain_tensor(&sorted);
        stream.retain_tensor(&chunks);
        let sort = stream.load_pipeline(LibraryName::Ggml, "route_sort_by_expert")?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&sort);
            encoder.set_metal_tensor(0, route_expert_ids, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, &offsets, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(2, &sorted, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(3, &chunks, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, &[route_count as u32]);
            encoder.set_slice(5, &[n_experts as u32]);
            encoder.set_slice(6, &[max_chunks as u32]);
            let grid = MTLSize { width: 1, height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        // Gather activations into expert-sorted order, run the
        // simdgroup-matrix tiled mm per 32-route chunk, scatter results back
        // to route order. Two f32 staging buffers, trivial next to the
        // matmul itself.
        let a_sorted = unsafe {
            DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count, k])?
        };
        let c_sorted = unsafe {
            DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count, n])?
        };
        stream.retain_tensor(&a_sorted);
        stream.retain_tensor(&c_sorted);
        let gather = stream.load_pipeline(LibraryName::Ggml, "routed_gather_rows_f32")?;
        let scatter = stream.load_pipeline(LibraryName::Ggml, "routed_scatter_rows_f32")?;
        let mm = stream.load_pipeline(LibraryName::Ggml, "kernel_mul_mm_q4_0_routed_f32")?;
        let set_args = |encoder: &metal::ComputeCommandEncoderRef| {
            encoder.set_bytes(
                0,
                std::mem::size_of::<RoutedQ40F32Params>() as u64,
                &params as *const _ as *const _,
            );
        };
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            set_args(encoder);
            encoder.set_compute_pipeline_state(&gather);
            encoder.set_buffer(1, Some(get_metal_buffer(input)), input.buffer_offset::<u64>());
            encoder.set_metal_tensor(2, &a_sorted, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(3, &sorted, metal::MTLResourceUsage::Read);
            encoder.set_buffer(
                4,
                Some(get_metal_buffer(route_token_ids)),
                route_token_ids.buffer_offset::<u64>(),
            );
            // one thread per 4 elements (vectorized gather)
            let total = (route_count * k.div_ceil(4)) as u64;
            let grid = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        command_buffer.encode(|encoder| {
            set_args(encoder);
            encoder.set_compute_pipeline_state(&mm);
            encoder.set_buffer(1, Some(get_metal_buffer(weights)), weights.buffer_offset::<u64>());
            encoder.set_metal_tensor(2, &a_sorted, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, &chunks, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(4, &c_sorted, metal::MTLResourceUsage::Write);
            encoder.set_threadgroup_memory_length(0, 8192);
            let grid = MTLSize {
                width: max_chunks as u64,
                height: (n as u64).div_ceil(64),
                depth: 1,
            };
            let group = MTLSize { width: 128, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        command_buffer.encode(|encoder| {
            set_args(encoder);
            encoder.set_compute_pipeline_state(&scatter);
            encoder.set_metal_tensor(1, &c_sorted, metal::MTLResourceUsage::Read);
            encoder.set_buffer(2, Some(get_metal_buffer(output)), output.buffer_offset::<u64>());
            encoder.set_metal_tensor(3, &sorted, metal::MTLResourceUsage::Read);
            // one thread per 4 elements (vectorized scatter)
            let total = (route_count * n.div_ceil(4)) as u64;
            let grid = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        return Ok(());
    }

    let pipeline = stream.load_pipeline(LibraryName::Ggml, "kernel_routed_q4_0_f32")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RoutedQ40F32Params>() as u64,
            &params as *const _ as *const _,
        );
        encoder.set_buffer(1, Some(get_metal_buffer(weights)), weights.buffer_offset::<u64>());
        encoder.set_buffer(2, Some(get_metal_buffer(input)), input.buffer_offset::<u64>());
        encoder.set_buffer(
            3,
            Some(get_metal_buffer(route_token_ids)),
            route_token_ids.buffer_offset::<u64>(),
        );
        encoder.set_buffer(
            4,
            Some(get_metal_buffer(route_expert_ids)),
            route_expert_ids.buffer_offset::<u64>(),
        );
        encoder.set_buffer(5, Some(get_metal_buffer(output)), output.buffer_offset::<u64>());

        let grid_size =
            MTLSize { width: (n as u64).div_ceil(8), height: route_count as u64, depth: 1 };
        let group_size = MTLSize { width: 8, height: 8, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// Fused routed expert up-projection: one pass computes g = w1 x (+bias1),
/// u = w3 x (+bias3) and writes act(g, u). Decode-sized route lists run a
/// single gemv-pair dispatch; prefill-sized lists share ONE expert sort and
/// ONE activation gather between the two tiled matmuls, and the scatter
/// applies bias + activation on the way out (replacing 2x(sort, gather,
/// scatter) + bias adds + activation dispatches of the unfused lowering).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_routed_q40_swiglu_f32(
    stream: &MetalStream,
    input: &DeviceTensor,
    w1: &DeviceTensor,
    w3: &DeviceTensor,
    biases: Option<(&DeviceTensor, &DeviceTensor)>,
    route_token_ids: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    input_mode: RoutedQ40InputMode,
    act: RoutedSwigluAct,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(w1);
    stream.retain_tensor(w3);
    stream.retain_tensor(route_token_ids);
    stream.retain_tensor(route_expert_ids);
    stream.retain_tensor(output);
    if let Some((b1, b3)) = biases {
        stream.retain_tensor(b1);
        stream.retain_tensor(b3);
    }

    ensure!(input.rank() == 2, "routed swiglu input must be [rows,k], got {:?}", input.shape());
    ensure!(
        matches!(input.datum_type(), F32 | F16),
        "routed swiglu input must be f32 or f16"
    );
    ensure!(
        route_token_ids.rank() == 1
            && route_expert_ids.rank() == 1
            && route_token_ids.datum_type() == i64::datum_type()
            && route_expert_ids.datum_type() == i64::datum_type()
            && route_token_ids.shape() == route_expert_ids.shape()
    );
    ensure!(
        w1.rank() == 3 && get_quant_fact(w1, &Q4_0).is_some(),
        "routed swiglu w1 must be Q4_0 [experts,n,k], got {:?}",
        w1.shape()
    );
    ensure!(w3.rank() == 3 && get_quant_fact(w3, &Q4_0).is_some());
    ensure!(w1.shape() == w3.shape(), "w1/w3 must share [experts,n,k]");
    ensure!(output.rank() == 2 && output.datum_type() == F32);

    let route_count = route_token_ids.shape()[0];
    let n_experts = w1.shape()[0];
    let n = w1.shape()[1];
    let k = w1.shape()[2];
    ensure!(input.shape()[1] == k);
    ensure!(output.shape() == [route_count, n]);
    ensure!(k % Q4_0.block_len() == 0);
    if let Some((b1, b3)) = biases {
        for b in [b1, b3] {
            ensure!(
                b.datum_type() == F32 && b.shape() == [n_experts, n],
                "routed swiglu bias must be f32 [experts,n], got {:?}",
                b.shape()
            );
        }
    }
    if input_mode == RoutedQ40InputMode::RouteRows {
        ensure!(input.shape()[0] == route_count);
    }
    if route_count == 0 || n == 0 {
        return Ok(());
    }

    let block_count = k / Q4_0.block_len();
    let weight_row_stride = block_count * Q4_0.block_bytes();
    let params = RoutedQ40F32Params {
        k: k as i32,
        n: n as i32,
        route_count: route_count as i32,
        input_mode: match input_mode {
            RoutedQ40InputMode::TokenRows => 0,
            RoutedQ40InputMode::RouteRows => 1,
        },
        weight_expert_stride: (n * weight_row_stride) as u64,
        weight_row_stride: weight_row_stride as u64,
        input_row_stride: (input.strides()[0] as usize * input.datum_type().size_of()) as u64,
        output_route_stride: (output.strides()[0] as usize * output.datum_type().size_of())
            as u64,
    };
    let sparams = act.params(biases.is_some());
    // Unused bias bind points fall back to w1 (the kernel never reads them
    // when has_bias == 0).
    let (b1_buf, b1_off, b3_buf, b3_off) = match biases {
        Some((b1, b3)) => (
            get_metal_buffer(b1),
            b1.buffer_offset::<u64>(),
            get_metal_buffer(b3),
            b3.buffer_offset::<u64>(),
        ),
        None => {
            (get_metal_buffer(w1), w1.buffer_offset::<u64>(), get_metal_buffer(w1), w1.buffer_offset::<u64>())
        }
    };

    const GROUPED_MIN_ROUTES: usize = 64;
    if route_count >= GROUPED_MIN_ROUTES
        && n_experts <= 256
        && k % 32 == 0
        && std::env::var_os("TRACT_METAL_DISABLE_GROUPED_MOE").is_none()
    {
        // Same grouped machinery as dispatch_routed_q40_f32, with the sort
        // and gather shared between the two matmuls.
        let max_chunks = route_count.div_ceil(32) + n_experts;
        let offsets =
            unsafe { DeviceTensor::uninitialized_dt(u32::datum_type(), &[n_experts + 1])? };
        let sorted = unsafe { DeviceTensor::uninitialized_dt(u32::datum_type(), &[route_count])? };
        let chunks =
            unsafe { DeviceTensor::uninitialized_dt(u32::datum_type(), &[3 * max_chunks])? };
        let a_sorted =
            unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count, k])? };
        let c1_sorted =
            unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count, n])? };
        let c3_sorted =
            unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), &[route_count, n])? };
        for t in [&offsets, &sorted, &chunks, &a_sorted, &c1_sorted, &c3_sorted] {
            stream.retain_tensor(t);
        }
        let sort = stream.load_pipeline(LibraryName::Ggml, "route_sort_by_expert")?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&sort);
            encoder.set_metal_tensor(0, route_expert_ids, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, &offsets, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(2, &sorted, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(3, &chunks, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, &[route_count as u32]);
            encoder.set_slice(5, &[n_experts as u32]);
            encoder.set_slice(6, &[max_chunks as u32]);
            let grid = MTLSize { width: 1, height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        let set_args = |encoder: &metal::ComputeCommandEncoderRef| {
            encoder.set_bytes(
                0,
                std::mem::size_of::<RoutedQ40F32Params>() as u64,
                &params as *const _ as *const _,
            );
        };
        // The gather stages activations as f32 whatever the input dtype (the
        // f16 variant converts exactly), so both mms and the scatter below
        // are dtype-blind.
        let gather_name =
            if input.datum_type() == F16 { "routed_gather_rows_f16x" } else { "routed_gather_rows_f32" };
        let gather = stream.load_pipeline(LibraryName::Ggml, gather_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            set_args(encoder);
            encoder.set_compute_pipeline_state(&gather);
            encoder.set_buffer(1, Some(get_metal_buffer(input)), input.buffer_offset::<u64>());
            encoder.set_metal_tensor(2, &a_sorted, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(3, &sorted, metal::MTLResourceUsage::Read);
            encoder.set_buffer(
                4,
                Some(get_metal_buffer(route_token_ids)),
                route_token_ids.buffer_offset::<u64>(),
            );
            // one thread per 4 elements (vectorized gather)
            let total = (route_count * k.div_ceil(4)) as u64;
            let grid = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        let mm = stream.load_pipeline(LibraryName::Ggml, "kernel_mul_mm_q4_0_routed_f32")?;
        for (w, c_sorted) in [(w1, &c1_sorted), (w3, &c3_sorted)] {
            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                set_args(encoder);
                encoder.set_compute_pipeline_state(&mm);
                encoder.set_buffer(1, Some(get_metal_buffer(w)), w.buffer_offset::<u64>());
                encoder.set_metal_tensor(2, &a_sorted, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(3, &chunks, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(4, c_sorted, metal::MTLResourceUsage::Write);
                encoder.set_threadgroup_memory_length(0, 8192);
                let grid = MTLSize {
                    width: max_chunks as u64,
                    height: (n as u64).div_ceil(64),
                    depth: 1,
                };
                let group = MTLSize { width: 128, height: 1, depth: 1 };
                encoder.dispatch_thread_groups(grid, group);
            });
        }
        let scatter = stream.load_pipeline(LibraryName::Ggml, "routed_swiglu_scatter_f32")?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            set_args(encoder);
            encoder.set_bytes(
                1,
                std::mem::size_of::<RoutedSwigluParams>() as u64,
                &sparams as *const _ as *const _,
            );
            encoder.set_compute_pipeline_state(&scatter);
            encoder.set_metal_tensor(2, &c1_sorted, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, &c3_sorted, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(4, &sorted, metal::MTLResourceUsage::Read);
            encoder.set_buffer(
                5,
                Some(get_metal_buffer(route_expert_ids)),
                route_expert_ids.buffer_offset::<u64>(),
            );
            encoder.set_buffer(6, Some(b1_buf), b1_off as NSUInteger);
            encoder.set_buffer(7, Some(b3_buf), b3_off as NSUInteger);
            encoder.set_buffer(8, Some(get_metal_buffer(output)), output.buffer_offset::<u64>());
            // one thread per 4 elements (vectorized swiglu scatter)
            let total = (route_count * n.div_ceil(4)) as u64;
            let grid = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
            let group = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        return Ok(());
    }

    let gemv_name = if input.datum_type() == F16 {
        "kernel_routed_q4_0_swiglu_f16x_f32"
    } else {
        "kernel_routed_q4_0_swiglu_f32"
    };
    let pipeline = stream.load_pipeline(LibraryName::Ggml, gemv_name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RoutedQ40F32Params>() as u64,
            &params as *const _ as *const _,
        );
        encoder.set_bytes(
            1,
            std::mem::size_of::<RoutedSwigluParams>() as u64,
            &sparams as *const _ as *const _,
        );
        encoder.set_buffer(2, Some(get_metal_buffer(w1)), w1.buffer_offset::<u64>());
        encoder.set_buffer(3, Some(get_metal_buffer(w3)), w3.buffer_offset::<u64>());
        encoder.set_buffer(4, Some(get_metal_buffer(input)), input.buffer_offset::<u64>());
        encoder.set_buffer(
            5,
            Some(get_metal_buffer(route_token_ids)),
            route_token_ids.buffer_offset::<u64>(),
        );
        encoder.set_buffer(
            6,
            Some(get_metal_buffer(route_expert_ids)),
            route_expert_ids.buffer_offset::<u64>(),
        );
        encoder.set_buffer(7, Some(b1_buf), b1_off as NSUInteger);
        encoder.set_buffer(8, Some(b3_buf), b3_off as NSUInteger);
        encoder.set_buffer(9, Some(get_metal_buffer(output)), output.buffer_offset::<u64>());

        let grid_size =
            MTLSize { width: (n as u64).div_ceil(8), height: route_count as u64, depth: 1 };
        let group_size = MTLSize { width: 8, height: 8, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use std::any::TypeId;

    use num_traits::Float;
    use tract_core::ops::array::MultiBroadcastTo;
    use tract_core::ops::cast::Cast;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_linalg::block_quant::{BlockQuant, BlockQuantStorage, Q4_0};

    use super::*;
    use crate::kernels::matmul::GemmImpl;
    use crate::kernels::matmul::tests::run_mmm_test_case;
    use tract_gpu::tensor::IntoDevice;
    use tract_ndarray::{Ix2, Ix3};

    #[test]
    fn test_ggml_compilation() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| stream.load_library(LibraryName::Ggml))?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 8, 64, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((3, 8, 64, 200), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((10, 25, 512, 320), false, true, F32, F16)?;
        Ok(())
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case::<GgmlGemm>((1, 8, 32, 3), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, F32, F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 4, 128, 8), false, true, F32, F32)?;

        // f16_f32_1row
        run_mmm_test_case::<GgmlGemm>((1, 1, 32, 2), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 3, 62, 2), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 3, 2, 9), false, true, F32, F16)?;

        // f16_f32_L4
        run_mmm_test_case::<GgmlGemm>((2, 2, 128, 8), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((4, 4, 156, 30), false, true, F32, F16)?;

        // f16_f32
        run_mmm_test_case::<GgmlGemm>((1, 4, 32, 2), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, F32, F16)?;
        run_mmm_test_case::<GgmlGemm>((4, 4, 128, 7), false, true, F32, F16)?;

        // f16_f16
        run_mmm_test_case::<GgmlGemm>((1, 1, 2, 1), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, F16, F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 16, 128, 9), false, true, F16, F16)?;
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
            operating_dt: Some(DatumType::F32),
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

        model.select_output_outlets(&output)?;
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
        with_borrowed_metal_stream(|stream| {
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
                let b_q4_0_tensor = BlockQuantStorage::new(
                    Box::new(Q4_0),
                    batch * n,
                    k,
                    Arc::new(Q4_0.quant_f32(&b_data)?),
                )?
                .into_tensor_with_shape(f32::datum_type(), &[batch, n, k]);
                (b_tensor, b_q4_0_tensor)
            } else {
                let b_tensor = Tensor::from_shape(&b_shape, &b_data)?;
                (b_tensor.clone(), b_tensor)
            };

            let metal_output = GemmImpl::<GgmlGemm>::new(false, true).eval(
                stream,
                &a.clone().into_device()?,
                &metal_b.clone().into_device()?,
            )?;
            let output = reference(a, ref_b)?;
            metal_output.to_host()?.close_enough(&output, Approximation::Approximate)?;
            Ok(())
        })
    }

    fn q40_weights_tensor(shape: &[usize], data: &[f32]) -> TractResult<Tensor> {
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0);
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        Ok(BlockQuantStorage::new(Box::new(Q4_0), rows, k, Arc::new(Q4_0.quant_f32(data)?))?
            .into_tensor_with_shape(f32::datum_type(), shape))
    }

    fn routed_q40_reference(
        input: &Tensor,
        weights: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
        input_mode: RoutedQ40InputMode,
    ) -> TractResult<Tensor> {
        let input = input.to_plain_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
        let weights = weights.to_plain_array_view::<f32>()?.into_dimensionality::<Ix3>()?;
        let routes = route_token_ids.len();
        let n = weights.shape()[1];
        let k = weights.shape()[2];
        let mut output = tract_ndarray::Array2::<f32>::zeros((routes, n));
        for route in 0..routes {
            let input_row = match input_mode {
                RoutedQ40InputMode::TokenRows => route_token_ids[route] as usize,
                RoutedQ40InputMode::RouteRows => route,
            };
            let expert = route_expert_ids[route] as usize;
            for out in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += input[[input_row, kk]] * weights[[expert, out, kk]];
                }
                output[[route, out]] = sum;
            }
        }
        Ok(output.into_tensor())
    }

    fn run_routed_q40_case(input_mode: RoutedQ40InputMode) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let experts = 3;
            let tokens = 5;
            let routes = 6;
            let n = 17;
            let k = 64;
            let input_rows = match input_mode {
                RoutedQ40InputMode::TokenRows => tokens,
                RoutedQ40InputMode::RouteRows => routes,
            };
            let input_data = (0..input_rows * k)
                .map(|i| ((i * 13 % 97) as f32 - 48.0) / 64.0)
                .collect::<Vec<_>>();
            let weight_data = (0..experts * n * k)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 80.0)
                .collect::<Vec<_>>();
            let route_token_ids = match input_mode {
                RoutedQ40InputMode::TokenRows => vec![3, 0, 4, 1, 3, 2],
                RoutedQ40InputMode::RouteRows => (0..routes as i64).collect(),
            };
            let route_expert_ids = vec![1, 0, 2, 1, 2, 0];

            let input = Tensor::from_shape(&[input_rows, k], &input_data)?;
            let weights_plain = Tensor::from_shape(&[experts, n, k], &weight_data)?;
            let weights_dequant = Q4_0.simulate_precision_loss(weights_plain, 2)?;
            let weights = q40_weights_tensor(&[experts, n, k], &weight_data)?;
            let token_ids = Tensor::from_shape(&[routes], &route_token_ids)?;
            let expert_ids = Tensor::from_shape(&[routes], &route_expert_ids)?;

            let expected = routed_q40_reference(
                &input,
                &weights_dequant,
                &route_token_ids,
                &route_expert_ids,
                input_mode,
            )?;
            let actual = eval_routed_q40_f32(
                stream,
                &input.into_device()?,
                &weights.into_device()?,
                &token_ids.into_device()?,
                &expert_ids.into_device()?,
                input_mode,
            )?;
            actual.to_host()?.close_enough(&expected, Approximation::Approximate)
        })
    }

    #[test]
    fn test_routed_q40_token_rows() -> TractResult<()> {
        run_routed_q40_case(RoutedQ40InputMode::TokenRows)
    }

    /// Route counts above GROUPED_MIN_ROUTES take the expert-grouped path
    /// (counting sort + 32-routes-per-threadgroup matmul).
    fn run_routed_q40_grouped_case(input_mode: RoutedQ40InputMode) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let experts = 8;
            let tokens = 40;
            let per_token = 4;
            let routes = tokens * per_token;
            let n = 48;
            let k = 64;
            let input_rows = match input_mode {
                RoutedQ40InputMode::TokenRows => tokens,
                RoutedQ40InputMode::RouteRows => routes,
            };
            let input_data = (0..input_rows * k)
                .map(|i| ((i * 13 % 97) as f32 - 48.0) / 64.0)
                .collect::<Vec<_>>();
            let weight_data = (0..experts * n * k)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 80.0)
                .collect::<Vec<_>>();
            let route_token_ids: Vec<i64> = match input_mode {
                RoutedQ40InputMode::TokenRows => {
                    (0..routes as i64).map(|r| r / per_token as i64).collect()
                }
                RoutedQ40InputMode::RouteRows => (0..routes as i64).collect(),
            };
            let route_expert_ids: Vec<i64> =
                (0..routes as i64).map(|r| (r * 5 + r / 7) % experts as i64).collect();

            let input = Tensor::from_shape(&[input_rows, k], &input_data)?;
            let weights_plain = Tensor::from_shape(&[experts, n, k], &weight_data)?;
            let weights_dequant = Q4_0.simulate_precision_loss(weights_plain, 2)?;
            let weights = q40_weights_tensor(&[experts, n, k], &weight_data)?;
            let token_ids = Tensor::from_shape(&[routes], &route_token_ids)?;
            let expert_ids = Tensor::from_shape(&[routes], &route_expert_ids)?;

            let expected = routed_q40_reference(
                &input,
                &weights_dequant,
                &route_token_ids,
                &route_expert_ids,
                input_mode,
            )?;
            let actual = eval_routed_q40_f32(
                stream,
                &input.into_device()?,
                &weights.into_device()?,
                &token_ids.into_device()?,
                &expert_ids.into_device()?,
                input_mode,
            )?;
            // The grouped path dequantizes q4_0 through f16 inside the
            // simdgroup-matrix pipeline, exactly like the dense q40 matmuls;
            // the reference dequantizes in f32, hence the looser gate.
            actual.to_host()?.close_enough(&expected, Approximation::VeryApproximate)
        })
    }

    #[test]
    fn test_routed_q40_grouped_token_rows() -> TractResult<()> {
        run_routed_q40_grouped_case(RoutedQ40InputMode::TokenRows)
    }

    #[test]
    fn test_routed_q40_grouped_route_rows() -> TractResult<()> {
        run_routed_q40_grouped_case(RoutedQ40InputMode::RouteRows)
    }

    #[test]
    fn test_routed_q40_route_rows() -> TractResult<()> {
        run_routed_q40_case(RoutedQ40InputMode::RouteRows)
    }

    /// The f16-activation variants must be BIT-IDENTICAL to upcasting the
    /// activations to f32 first: same kernels, same accumulation order, the
    /// half->float element conversion is exact. Covers the per-route gemv
    /// (few routes) and the expert-grouped gather+mm path (many routes), and
    /// the router score gemv through dispatch_route_topk_f32.
    #[test]
    fn test_routed_swiglu_and_router_f16_input_bit_exact() -> TractResult<()> {
        use tract_num_traits::AsPrimitive;
        for tokens in [3usize, 40] {
            with_borrowed_metal_stream(|stream| {
                let experts = 8;
                let per_token = 4;
                let routes = tokens * per_token;
                let n = 48;
                let k = 64;
                let input_f16 = Tensor::from_shape(
                    &[tokens, k],
                    &(0..tokens * k)
                        .map(|i| {
                            let v: f32 = ((i * 13 % 97) as f32 - 48.0) / 64.0;
                            let h: f16 = v.as_();
                            h
                        })
                        .collect::<Vec<_>>(),
                )?;
                let input_f32 = input_f16.cast_to::<f32>()?.into_owned();
                let weight_data = (0..experts * n * k)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 80.0)
                    .collect::<Vec<_>>();
                let w1 = q40_weights_tensor(&[experts, n, k], &weight_data)?.into_device()?;
                let w3 = q40_weights_tensor(&[experts, n, k], &weight_data)?.into_device()?;
                let route_token_ids: Vec<i64> =
                    (0..routes as i64).map(|r| r / per_token as i64).collect();
                let route_expert_ids: Vec<i64> =
                    (0..routes as i64).map(|r| (r * 5 + r / 7) % experts as i64).collect();
                let token_ids =
                    Tensor::from_shape(&[routes], &route_token_ids)?.into_device()?;
                let expert_ids =
                    Tensor::from_shape(&[routes], &route_expert_ids)?.into_device()?;

                let x16 = input_f16.clone().into_device()?;
                let x32 = input_f32.clone().into_device()?;
                let mut outs: Vec<Tensor> = vec![];
                for x in [&x32, &x16] {
                    let output = unsafe {
                        DeviceTensor::uninitialized_dt(F32, &[routes, n])?
                    };
                    dispatch_routed_q40_swiglu_f32(
                        stream,
                        x,
                        &w1,
                        &w3,
                        None,
                        &token_ids,
                        &expert_ids,
                        RoutedQ40InputMode::TokenRows,
                        RoutedSwigluAct::Plain,
                        &output,
                    )?;
                    stream.wait_until_completed()?;
                    outs.push(output.to_host()?.into_tensor());
                }
                ensure!(outs[0] == outs[1], "swiglu f16 input diverged from upcast-first");

                // Router scores + topk: same exactness contract.
                let wg = Tensor::from_shape(
                    &[experts, k],
                    &(0..experts * k)
                        .map(|i| ((i * 7 % 89) as f32 - 44.0) / 60.0)
                        .collect::<Vec<_>>(),
                )?
                .into_device()?;
                let kk = 4usize;
                let mut routed: Vec<(Tensor, Tensor, Tensor)> = vec![];
                for x in [&x32, &x16] {
                    let tid = unsafe {
                        DeviceTensor::uninitialized_dt(DatumType::I64, &[tokens * kk])?
                    };
                    let eid = unsafe {
                        DeviceTensor::uninitialized_dt(DatumType::I64, &[tokens * kk])?
                    };
                    let wts = unsafe {
                        DeviceTensor::uninitialized_dt(F32, &[tokens * kk])?
                    };
                    crate::kernels::moe::dispatch_route_topk_f32(
                        stream,
                        x,
                        &wg,
                        None,
                        kk,
                        &tract_transformers::ops::moe_ffn::GateMode::SoftmaxTopk,
                        &tid,
                        &eid,
                        &wts,
                    )?;
                    stream.wait_until_completed()?;
                    routed.push((
                        tid.to_host()?.into_tensor(),
                        eid.to_host()?.into_tensor(),
                        wts.to_host()?.into_tensor(),
                    ));
                }
                ensure!(routed[0].0 == routed[1].0, "router token ids diverged");
                ensure!(routed[0].1 == routed[1].1, "router expert ids diverged");
                ensure!(routed[0].2 == routed[1].2, "router weights diverged");
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_granite_shape_routed_q40_metal() -> TractResult<()> {
        use std::time::Instant;

        with_borrowed_metal_stream(|stream| {
            let experts = 8;
            let tokens = 16;
            let routes = 8;
            let n = 1024;
            let k = 2048;
            let input_data =
                (0..tokens * k).map(|i| ((i * 13 % 97) as f32 - 48.0) / 64.0).collect::<Vec<_>>();
            let weight_data = (0..experts * n * k)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 80.0)
                .collect::<Vec<_>>();
            let route_token_ids = vec![0i64, 1, 2, 3, 4, 5, 6, 7];
            let route_expert_ids = vec![0i64, 1, 2, 3, 4, 5, 6, 7];

            let input = Tensor::from_shape(&[tokens, k], &input_data)?.into_device()?;
            let input_batched =
                Tensor::from_shape(&[experts, 1, k], &input_data[..experts * k])?.into_device()?;
            let weights = q40_weights_tensor(&[experts, n, k], &weight_data)?.into_device()?;
            let token_ids = Tensor::from_shape(&[routes], &route_token_ids)?.into_device()?;
            let expert_ids = Tensor::from_shape(&[routes], &route_expert_ids)?.into_device()?;
            let output = unsafe { DeviceTensor::uninitialized_dt(F32, &[routes, n])? };
            let batched_output = unsafe { DeviceTensor::uninitialized_dt(F32, &[experts, 1, n])? };
            let batched = GemmImpl::<GgmlGemm>::new(false, true);

            for _ in 0..10 {
                dispatch_routed_q40_f32(
                    stream,
                    &input,
                    &weights,
                    &token_ids,
                    &expert_ids,
                    RoutedQ40InputMode::TokenRows,
                    &output,
                )?;
                stream.wait_until_completed()?;
                batched.dispatch_eval(stream, &input_batched, &weights, &batched_output)?;
                stream.wait_until_completed()?;
            }

            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..50 {
                    dispatch_routed_q40_f32(
                        stream,
                        &input,
                        &weights,
                        &token_ids,
                        &expert_ids,
                        RoutedQ40InputMode::TokenRows,
                        &output,
                    )?;
                    stream.wait_until_completed()?;
                }
                best = best.min(start.elapsed().as_secs_f64() / 50.0);
            }
            let mut batched_best = f64::INFINITY;
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..50 {
                    batched.dispatch_eval(stream, &input_batched, &weights, &batched_output)?;
                    stream.wait_until_completed()?;
                }
                batched_best = batched_best.min(start.elapsed().as_secs_f64() / 50.0);
            }
            eprintln!(
                "metal routed q40 token rows: experts={experts} routes={routes} n={n} k={k} routed={:.3}us ggml_batched={:.3}us routed_vs_batched={:.3}x",
                best * 1e6,
                batched_best * 1e6,
                batched_best / best,
            );
            Ok(())
        })
    }

    #[test]
    fn test_broadcast() -> TractResult<()> {
        run_ggml_mat_mul_test::<f32>(2, 2, 1, 8, 4, false)?;
        run_ggml_mat_mul_test::<f32>(6, 3, 26, 22, 1, false)?;
        run_ggml_mat_mul_test::<f16>(1, 2, 1, 64, 10, false)?;
        run_ggml_mat_mul_test::<f16>(2, 2, 1, 128, 8, false)?;
        run_ggml_mat_mul_test::<f16>(4, 4, 6, 64, 10, false)?;
        Ok(())
    }

    #[test]
    fn test_q4() -> TractResult<()> {
        run_ggml_mat_mul_test::<f32>(32, 1, 1, 32, 32, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 320, 2048, 1, true)?;
        run_ggml_mat_mul_test::<f32>(4, 1, 1, 2048, 320, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 1, 32, 32, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 1, 64, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 1, 1, 4096, 512, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 1, 32, 32, true)?;
        run_ggml_mat_mul_test::<f32>(1, 1, 1, 64, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 1, 1, 2048, 128, true)?;
        run_ggml_mat_mul_test::<f32>(1, 3, 1, 32, 32, true)?;
        run_ggml_mat_mul_test::<f32>(4, 2, 1, 64, 4, true)?;
        run_ggml_mat_mul_test::<f32>(3, 2, 1, 512, 256, true)?;
        Ok(())
    }
}
