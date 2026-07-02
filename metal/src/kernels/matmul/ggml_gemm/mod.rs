use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
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

    fn output_dt(&self, _a_dt: DatumType, _b_dt: DatumType) -> TractResult<DatumType> {
        Ok(F32)
    }

    fn dispatch_eval(
        &self,
        stream: &MetalStream,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
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

        if (dts[0] == F32) && (k % 32 == 0) && (k >= 64) && ((m > 4) || (q40_b && a_batch > 1)) {
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
        ensure!(params.dts[0] == F32);
        Ok(("kernel_mul_mv_q4_0_f32".to_string(), (8, 8, 1)))
    } else if params.dts[1] == F32 {
        ensure!(params.dts[0] == F32);
        Ok(("kernel_mul_mv_f32_f32".to_string(), (32, 1, 4)))
    } else if params.dts[1] == F16 {
        if params.dts[0] == F32 {
            if (params.m * params.a_batch) < 4 {
                Ok(("kernel_mul_mv_f16_f32_1row".to_string(), (32, 1, 1)))
            } else if (params.k >= 128) && (params.k % 4 == 0) && (params.n >= 8) {
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

    ensure!((matches!(dts[1], F32 | F16) || q40_b) && dts[0] == F32);

    let i1_tname = if !q40_b { DeviceTensor::tname(dts[1])? } else { "q4_0" };
    let i2_tname = DeviceTensor::tname(dts[0])?;

    let name = format!("kernel_mul_mm_{i1_tname}_{i2_tname}");
    //dbg!(&name);
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

    #[test]
    fn test_routed_q40_route_rows() -> TractResult<()> {
        run_routed_q40_case(RoutedQ40InputMode::RouteRows)
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
