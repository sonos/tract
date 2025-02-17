use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::utils::as_q40_fact;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_linalg::frame::block_quant::{BlockQuant, Q4_0};
use DatumType::{F16, F32};

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
            ne02: params.batch as i32,
            nb01: (b_strides[1] as usize * b_el_size) as u64,
            nb02: (b_strides[0] as usize * b_el_size) as u64,
            nb03: (b_strides[0] as usize * params.batch * b_el_size) as u64,
            ne12: params.batch as i32,
            nb10: (params.a_strides[2] as usize * a_el_size) as u64,
            nb11: (params.a_strides[1] as usize * a_el_size) as u64,
            nb12: (params.a_strides[0] as usize * a_el_size) as u64,
            nb13: (params.a_strides[0] as usize * params.batch * a_el_size) as u64,
            ne0: params.n as i32,
            ne1: params.m as i32,
            r2: 1,
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

impl From<GemmDispatchParams> for GgmlGemvParams {
    fn from(params: GemmDispatchParams) -> Self {
        assert!(params.a_strides.len() == 3 && params.b_strides.len() == 3);
        let a_el_size = params.dts[0].size_of();

        let b_el_size = if params.q40_b { 18 } else { params.dts[1].size_of() };
        let mut b_strides = params.b_strides;
        if params.q40_b {
            b_strides[0] /= 32;
            b_strides[1] /= 32
        };

        // Kernel produced transposed output so we swap the inputs
        GgmlGemvParams {
            ne00: params.k as i32,
            ne01: params.n as i32,
            ne02: params.batch as i32,
            nb00: (b_strides[2] as usize * b_el_size) as u64,
            nb01: (b_strides[1] as usize * b_el_size) as u64,
            nb02: (b_strides[0] as usize * b_el_size) as u64,
            nb03: (b_strides[0] as usize * params.batch * b_el_size) as u64,
            ne10: params.k as i32,
            ne11: params.m as i32,
            ne12: params.batch as i32,
            nb10: (params.a_strides[2] as usize * a_el_size) as u64,
            nb11: (params.a_strides[1] as usize * a_el_size) as u64,
            nb12: (params.a_strides[0] as usize * a_el_size) as u64,
            nb13: (params.a_strides[0] as usize * params.batch * a_el_size) as u64,
            ne0: params.n as i32,
            ne1: params.m as i32,
            r2: 1,
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

    fn is_supported_dts(&self, facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Ggml: Expected 2 inputs for Matmul");

        let regular_types_support = matches!(
            (facts[0].datum_type, facts[1].datum_type),
            (F32, F32) | (F16, F16) | (F16, F32)
        );

        regular_types_support
            || (as_q40_fact(&facts[1]).is_some() && matches!(facts[0].datum_type, F16 | F32))
    }

    fn output_dt(&self, _a_dt: DatumType, _b_dt: DatumType) -> TractResult<DatumType> {
        Ok(F32)
    }

    fn dispatch_eval(
        &self,
        context: &MetalContext,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
    ) -> TractResult<()> {
        let GemmDispatchParams {
            dts,
            batch,
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

        if (dts[0] == F32) && (k % 32 == 0) && (k >= 64) && ((m > 4) || (q40_b && batch > 1)) {
            dispatch_metal_ggml_gemm(
                context, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        } else {
            dispatch_metal_ggml_gemv(
                context, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        }

        Ok(())
    }
}

fn mv_kernel_name_and_dispatch_params(
    params: &GemmDispatchParams,
) -> Result<(String, (u64, u64, u64))> {
    if params.dts[1] == F32 {
        ensure!(params.dts[0] == F32);
        Ok(("kernel_mul_mv_f32_f32".to_string(), (32, 1, 4)))
    } else if params.dts[1] == F16 {
        if params.dts[0] == F32 {
            if (params.m * params.batch) < 4 {
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
        ensure!((params.q40_b) && (params.dts[0] == F32));
        Ok(("kernel_mul_mv_q4_0_f32".to_string(), (8, 8, 1)))
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_metal_ggml_gemv(
    context: &MetalContext,
    params: GemmDispatchParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    let (name, (nth0, nth1, nrows)) = mv_kernel_name_and_dispatch_params(&params)?;
    //dbg!(&name);
    let pipeline = context.shared_context().load_pipeline(LibraryName::Ggml, &name)?;

    let ggml_params: GgmlGemvParams = params.clone().into();
    let command_buffer = context.command_buffer();
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
                depth: /* batch_size_out */ params.batch as u64,
            }
        } else {
            MTLSize {
                width: (params.n as u64).div_ceil(8),
                height: params.m as u64,
                depth: /* batch_size_out */ params.batch as u64,
            }
        };
        let group_size = MTLSize { width: nth0, height: nth1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_metal_ggml_gemm(
    context: &MetalContext,
    params: GemmDispatchParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    let GemmDispatchParams { dts, q40_b, .. } = params;

    ensure!((matches!(dts[1], F32 | F16) || q40_b) && dts[0] == F32);

    let i1_tname = if !q40_b { MetalTensor::tname(dts[1])? } else { "q4_0" };
    let i2_tname = MetalTensor::tname(dts[0])?;

    let name = format!("kernel_mul_mm_{i1_tname}_{i2_tname}");
    //dbg!(&name);
    let pipeline = context.shared_context().load_pipeline(LibraryName::Ggml, &name)?;

    let ggml_params: GgmlGemmParams = params.clone().into();
    let command_buffer = context.command_buffer();
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
            depth: /* batch_size_out */ params.batch as u64,
        };
        let group_size = MTLSize { width: 128, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use tract_core::ops::einsum::BasicMatMul;
    use tract_linalg::frame::block_quant::{BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0};

    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;
    use crate::kernels::matmul::GemmImpl;
    use crate::IntoMetal;

    #[test]
    fn test_ggml_compilation() -> Result<()> {
        crate::METAL_CONTEXT
            .with_borrow(|context| context.shared_context().load_library(LibraryName::Ggml))?;
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

    fn run_mat_mul_q4_test(batch: usize, m: usize, k: usize, n: usize) -> TractResult<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                ensure!(k % 32 == 0);
                let a_shape = [batch, m, k];
                let b_shape = [batch, n, k];

                let a_data = (0..batch * k * m)
                    .map(|f| f as f32 / (batch * m * k) as f32)
                    .collect::<Vec<_>>();

                let a = Tensor::from_shape(&a_shape, &a_data)?;

                let b_data = (0..batch * n * k)
                    .map(|f| f as f32 / (batch * n * k) as f32)
                    .collect::<Vec<_>>();

                let mut b_quant = Q4_0.quant_f32(&b_data)?;

                let b_tensor = Tensor::from_shape(&b_shape, &b_data)?;

                crate::utils::tract_to_gguf_q4_0_packing(&mut b_quant)?;

                let b_q4_0_tensor = tensor0(Opaque(Arc::new(BlockQuantValue {
                    fact: BlockQuantFact { format: Box::new(Q4_0), shape: tvec![batch, n, k] },
                    value: b_quant,
                })));

                let metal_output = GemmImpl::<GgmlGemm>::new(false, true).eval(
                    context,
                    &a.clone().into_metal()?,
                    &b_q4_0_tensor.clone().into_metal()?,
                )?;

                let matmul = BasicMatMul {
                    transpose_a: false,
                    transpose_b: true,
                    transpose_c: false,
                    quantize_output: None,
                };

                let output = args_1!(matmul.eval(tvec![a.into_tvalue(), b_tensor.into_tvalue()])?);
                metal_output.to_cpu()?.close_enough(&output, Approximation::SuperApproximate)?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_q4() -> TractResult<()> {
        run_mat_mul_q4_test(32, 1, 32, 32)?;
        run_mat_mul_q4_test(1, 32003, 2048, 1)?;
        run_mat_mul_q4_test(4, 1, 2048, 32003)?;
        run_mat_mul_q4_test(1, 1, 32, 32)?;
        run_mat_mul_q4_test(1, 1, 64, 4)?;
        run_mat_mul_q4_test(3, 1, 4096, 4096)?;

        Ok(())
    }
}
