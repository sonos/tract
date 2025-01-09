use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

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
    r3: i16
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

    fn is_supported_dt(&self, dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
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
            dt,
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

        dispatch_metal_ggml_gemm(
            context,
            dt,
            (batch, m, n, k),
            a_offset,
            a_buffer,
            b_offset,
            b_buffer,
            c_buffer,
            c_offset,
        )?;
        

        Ok(())
    }
}


#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_ggml_gemm(
    context: &MetalContext,
    dt: DatumType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {

    assert!(k % 32 == 0);
    ensure!(matches!(dt, DatumType::F32));

    let params = GgmlGemmParams {
        ne00: k as i32,
        ne02: b as i32,
        nb01: (k * 4 )as u64,
        nb02: (k * n * 4 )as u64,
        nb03: (k * n * b * 4 )as u64,
        ne12: b as i32,
        nb10: (4)as u64,
        nb11: (k * 4)as u64,
        nb12: (k * m  * 4 )as u64,
        nb13: (k * m * b * 4 )as u64,
        ne0: n as i32,
        ne1: m as i32,
        r2:  1,
        r3: 1
    };

    assert!(params.nb01 % 16 == 0);
    let tname = MetalTensor::tname(dt)?;
    let name = format!("kernel_mul_mm_{tname}_{tname}");

    let pipeline = context.shared_context().load_pipeline(
        LibraryName::Ggml,
        &name,
    )?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, size_of::<GgmlGemmParams>() as u64, &params as *const _ as *const _);        
        encoder.set_buffer(1, Some(rhs_buffer), lhs_offset as NSUInteger);
        encoder.set_buffer(2, Some(lhs_buffer), rhs_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let grid_size = MTLSize {
            width: m.div_ceil(32) as u64,
            height: n.div_ceil(64) as u64,
            depth: /* batch_size_out */ b as u64,
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
    use std::time::{Duration, Instant};

    use super::*;
    use crate::kernels::matmul::tests::{run_mmm_test_case, run_mmm_test_case_ggml};
    use crate::kernels::matmul::{GemmImpl, MlxGemm};
    use crate::{IntoMetal, MetalTensor};


    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<GgmlGemm>((1, 3, 32, 2), false, true)?;
        run_mmm_test_case::<GgmlGemm>((1, 2, 1536, 10), false, false)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 32, 4), false, false)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 64, 200), false, false)?;
        run_mmm_test_case::<GgmlGemm>((1, 25, 1280, 32000), false, false)?;
        Ok(())
    }

    #[test]
    fn test_ggml_vs_mlx() -> TractResult<()> {
        let n_iter = 5;
        
        run_mmm_test_case_ggml((1, 3, 32, 2), false, false)?;

        let mut ggml_duration = Duration::default();
        for _ in 0..n_iter {
            ggml_duration += run_mmm_test_case_ggml((1, 3, 32, 2), false, false)?;
            ggml_duration += run_mmm_test_case_ggml((1, 2, 1536, 10), false, false)?;
            ggml_duration += run_mmm_test_case_ggml((1, 4, 32, 4), false, false)?;
            ggml_duration += run_mmm_test_case_ggml((1, 4, 64, 200), false, false)?;
            ggml_duration += run_mmm_test_case_ggml((1, 25, 1280, 32000), false, false)?;
        }

        run_mmm_test_case::<MlxGemm>((1, 3, 32, 2), false, false)?;

        let mut mlx_duration = Duration::default();
        for _ in 0..n_iter {
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 3, 32, 2), false, false)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 2, 1536, 10), false, false)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 32, 4), false, false)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 64, 200), false, false)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 25, 1280, 32000), false, false)?;
        }

        println!("Mlx duration: {:}. Ggml duration: {}", mlx_duration.as_millis(), ggml_duration.as_millis());
        Ok(())
    }

    #[test]
    fn test_mlx_gemv_compilation() -> Result<()> {
        crate::METAL_CONTEXT
            .with_borrow(|context| context.shared_context().load_library(LibraryName::Ggml))?;
        Ok(())
    }

    #[test]
    fn test_mlx_gemm() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let (batch, m, n, k) = (10, 25, 48, 32);
                let a = Tensor::from_shape(
                    &[batch, m, k],
                    &(0..batch * m * k).map(|_f| 1.0 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let b = Tensor::from_shape(
                    &[batch, k, n],
                    &(0..batch * n * k).map(|_f| 1.0 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;


                let c = GemmImpl::<GgmlGemm>::default().eval(context, &a, &b)?;

                let expected_c = Tensor::from_shape(&[batch, m, n], &vec![k as f32; batch * m * n])?;

                let c = c.to_cpu()?;
                c.close_enough(&expected_c, Approximation::Approximate)?;
                assert!(c.close_enough(&expected_c, Approximation::Approximate).is_ok());

                let (b, m, n, k) = (1, 2, 4, 3);
                let a = MetalTensor::from_shape(
                    &[b, m, k],
                    &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;
                let b = MetalTensor::from_shape(
                    &[b, k, n],
                    &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;

                let c = GemmImpl::<GgmlGemm>::default().eval(context, &a, &b)?;

                let expected_c = Tensor::from_shape(
                    &[2, 2, 4],
                    &[
                        20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0,
                        488.0, 518.0, 548.0, 578.0,
                    ],
                )?;

                assert!(c.to_cpu()?.close_enough(&expected_c, Approximation::Approximate).is_ok());
                Ok(())
            })
        })
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        run_mmm_test_case::<GgmlGemm>((1, 4, 4, 1), false, false)?;
        run_mmm_test_case::<GgmlGemm>((10, 1, 4, 4), false, false)?;
        run_mmm_test_case::<GgmlGemm>((5, 1, 15, 7), false, true)?;
        Ok(())
    }
}
