use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug)]
#[repr(C)]
struct GgmlParams {
    batch: i32,
    m: i32,
    k: i32,
    n: i32,
    a_strides: [u64; 4],
    b_strides: [u64; 4],
    channel_broadcast_ratio: i32,
    batch_broadcast_ratio: i32,
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

    fn is_supported_dts(&self, dts: &[DatumType]) -> TractResult<bool> {
        ensure!(dts.len() == 2);

        if dts[0] == DatumType::F32 {
            Ok(dts[1] == DatumType::F32)
        } else {
            Ok(dts[0] == DatumType::F16 && matches!(dts[1], DatumType::F32 | DatumType::F16))
        }
    }

    fn output_dt(&self, _a_dt: DatumType, _b_dt: DatumType) -> TractResult<DatumType> {
        Ok(DatumType::F32)
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
            n,
            transpose_a,
            a_offset,
            transpose_b,
            b_offset,
            c_offset,
        } = params;

        ensure!(!transpose_a && transpose_b);

        // Kernel output is transposed so we switch the inputs
        let a_el_size = dts[0].size_of();
        let a_strides = [
            (batch * m * k * a_el_size) as u64,
            (m * k * a_el_size) as u64,
            (k * a_el_size) as u64,
            a_el_size as u64,
        ];

        let b_el_size = dts[1].size_of();
        let b_strides = [
            (batch * n * k * b_el_size) as u64,
            (n * k * b_el_size) as u64,
            (k * b_el_size) as u64,
            b_el_size as u64,
        ];

        let params = GgmlParams {
            batch: batch as i32,
            m: m as i32,
            k: k as i32,
            n: n as i32,
            a_strides,
            b_strides,
            channel_broadcast_ratio: 1,
            batch_broadcast_ratio: 1,
        };

        if (dts[0] == DatumType::F32) && (k % 32 == 0) && (k >= 64) && (m > 4) {
            dispatch_metal_ggml_gemm(
                context, dts, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        } else {
            dispatch_metal_ggml_gemv(
                context, dts, params, a_offset, a_buffer, b_offset, b_buffer, c_buffer, c_offset,
            )?;
        }

        Ok(())
    }
}

fn mv_kernel_name_and_dispatch_params(
    dts: &[DatumType],
    params: &GgmlParams,
) -> Result<(String, (u64, u64, u64))> {
    let (nth0, nth1, nrows): (u64, u64, u64) = (32, 1, 1);

    if dts[1] == DatumType::F32 {
        ensure!(dts[0] == DatumType::F32);
        Ok(("kernel_mul_mv_f32_f32".to_string(), (nth0, nth1, 4)))
    } else {
        ensure!(dts[1] == DatumType::F16);
        if dts[0] == DatumType::F32 {
            if (params.m * params.batch) < 4 {
                Ok(("kernel_mul_mv_f16_f32_1row".to_string(), (nth0, nth1, nrows)))
            } else if (params.k >= 128) && (params.k % 4 == 0) && (params.n >= 8) {
                Ok(("kernel_mul_mv_f16_f32_l4".to_string(), (nth0, nth1, params.m as u64)))
            } else {
                Ok(("kernel_mul_mv_f16_f32".to_string(), (nth0, nth1, 4)))
            }
        } else {
            // Never used in practice since we upcast input[0] to f32
            ensure!(dts[1] == DatumType::F16);
            Ok(("kernel_mul_mv_f16_f16".to_string(), (nth0, nth1, 4)))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_metal_ggml_gemv(
    context: &MetalContext,
    dts: [DatumType; 3],
    params: GgmlParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    let (name, (nth0, nth1, nrows)) = mv_kernel_name_and_dispatch_params(&dts, &params)?;
    //dbg!(&name);

    let pipeline = context.shared_context().load_pipeline(LibraryName::Ggml, &name)?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GgmlParams>() as u64, &params as *const _ as *const _);
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let ny = (params.m as u64).div_ceil(nrows);
        let grid_size = MTLSize {
            width: params.n as u64,
            height: ny,
            depth: /* batch_size_out */ params.batch as u64,
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
    dts: [DatumType; 3],
    params: GgmlParams,
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    ensure!(matches!(dts[1], DatumType::F32 | DatumType::F16) && dts[0] == DatumType::F32);

    let i1_tname = MetalTensor::tname(dts[1])?;
    let i2_tname = MetalTensor::tname(dts[0])?;

    let name = format!("kernel_mul_mm_{i1_tname}_{i2_tname}");

    let pipeline = context.shared_context().load_pipeline(LibraryName::Ggml, &name)?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GgmlParams>() as u64, &params as *const _ as *const _);
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let grid_size = MTLSize {
            width: ((params.m + 31) / 32) as u64,
            height: ((params.n + 63) / 64) as u64,
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
    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;

    #[test]
    fn test_ggml_compilation() -> Result<()> {
        crate::METAL_CONTEXT
            .with_borrow(|context| context.shared_context().load_library(LibraryName::Ggml))?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 1, 32, 2), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>(
            (3, 8, 64, 200),
            false,
            true,
            DatumType::F32,
            DatumType::F16,
        )?;
        run_mmm_test_case::<GgmlGemm>(
            (10, 25, 512, 320),
            false,
            true,
            DatumType::F32,
            DatumType::F16,
        )?;
        Ok(())
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        // f32_f32
        run_mmm_test_case::<GgmlGemm>((1, 8, 32, 3), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((2, 4, 128, 8), false, true, DatumType::F32, DatumType::F32)?;

        // f16_f32_1row
        run_mmm_test_case::<GgmlGemm>((1, 1, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 3, 62, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 3, 2, 9), false, true, DatumType::F32, DatumType::F16)?;

        // f16_f32_L4
        run_mmm_test_case::<GgmlGemm>((2, 2, 128, 8), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>(
            (4, 4, 156, 30),
            false,
            true,
            DatumType::F32,
            DatumType::F16,
        )?;

        // f16_f32
        run_mmm_test_case::<GgmlGemm>((1, 4, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((4, 4, 128, 7), false, true, DatumType::F32, DatumType::F16)?;

        // f16_f16
        run_mmm_test_case::<GgmlGemm>((1, 1, 2, 1), false, true, DatumType::F16, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F16, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>(
            (2, 16, 128, 9),
            false,
            true,
            DatumType::F16,
            DatumType::F16,
        )?;
        Ok(())
    }
}
