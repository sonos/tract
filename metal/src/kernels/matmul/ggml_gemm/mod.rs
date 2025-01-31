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
    ne00:  i32,
    ne02:  i32,
    nb01:  u64,
    nb02:  u64,
    nb03:  u64,
    ne12:  i32,
    nb10:  u64,
    nb11:  u64,
    nb12:  u64,
    nb13:  u64,
    ne0:  i32,
    ne1:  i32,
    r2:  i16,
    r3:  i16,
}

#[derive(Debug)]
#[repr(C)]
struct GgmlGemvParams{
    ne00:  i32,
    ne01:  i32,
    ne02:  i32,
    nb00:  u64,
    nb01:  u64,
    nb02:  u64,
    nb03:  u64,
    ne10:  i32,
    ne11:  i32,
    ne12:  i32,
    nb10:  u64,
    nb11:  u64,
    nb12:  u64,
    nb13:  u64,
    ne0:  i32,
    ne1:  i32,
    r2:  i16,
    r3:  i16,
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

        if dts[0] == DatumType::F32 { Ok(dts[1] == DatumType::F32) }
        else { Ok(dts[0] == DatumType::F16 && matches!(dts[1], DatumType::F32 | DatumType::F16))}
    }

    fn output_dt(
            &self,
            _a_dt: DatumType,
            _b_dt: DatumType,
        ) -> TractResult<DatumType> {
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
        if (dts[0] == DatumType::F32) && (k % 32 == 0) && (k >= 64) && (m > 4){
            dispatch_metal_ggml_gemm(
                context,
                dts,
                (batch, m, n, k),
                a_offset,
                a_buffer,
                b_offset,
                b_buffer,
                c_buffer,
                c_offset,
            )?;
        } else {
            dispatch_metal_ggml_gemv(
                context,
                dts,
                (batch, m, n, k),
                a_offset,
                a_buffer,
                b_offset,
                b_buffer,
                c_buffer,
                c_offset,
            )?;
        }

        Ok(())
    }
}

fn mv_kernel_name_and_dispatch_params(dts: &[DatumType], (b, m , k , n): (usize, usize, usize, usize)) -> Result<(String, (u64, u64, u64))> {
    let (nth0, nth1, nrows): (u64, u64, u64) = (32, 1, 1);

    if dts[1] == DatumType::F32 {
        ensure!(dts[0] == DatumType::F32);
        Ok(("kernel_mul_mv_f32_f32".to_string(), (nth0, nth1, 4)))
    }
    else {
        ensure!(dts[1] == DatumType::F16);
        if dts[0] == DatumType::F32 {
            if (m * b) < 4 {
                Ok(("kernel_mul_mv_f16_f32_1row".to_string(), (nth0, nth1, nrows)))
            }
            else if (k >= 128) && (k % 4 == 0) && (n >= 8) { 
                Ok(("kernel_mul_mv_f16_f32_l4".to_string(), (nth0, nth1, m as u64)))
            }
            else {
                Ok(("kernel_mul_mv_f16_f32".to_string(), (nth0, nth1, 4)))
            }
        }
        else {
            ensure!(dts[1] == DatumType::F16);
            Ok(("kernel_mul_mv_f16_f16".to_string(), (nth0, nth1, 4)))
        }

    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_ggml_gemv(
    context: &MetalContext,
    dts: [DatumType; 3],
    (b, m, n, k): (usize, usize, usize, usize),
    a_offset: usize,
    a_buffer: &Buffer,
    b_offset: usize,
    b_buffer: &Buffer,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {

    let el1_size = dts[1].size_of();
    let el2_size = dts[0].size_of();

    let params = GgmlGemvParams{
        ne00:  k as i32,
        ne01:  n as i32,
        ne02:  b as i32,
        nb00:  el1_size as u64,
        nb01:  (k * el1_size) as u64,
        nb02:  (n * k * el1_size) as u64,
        nb03:  (b * n * k * el1_size) as u64,
        ne10:  k as i32,
        ne11:  m as i32,
        ne12:  b as i32,
        nb10:  el2_size as u64,
        nb11:  (k * el2_size) as u64,
        nb12:  (m * k * el2_size) as u64,
        nb13:  (b * m * k * el2_size) as u64,
        ne0:  n as i32,
        ne1:  m as i32,
        r2:  1,
        r3:  1,
    };
    
    let (name, (nth0, nth1, nrows)) = mv_kernel_name_and_dispatch_params(&dts, (b, m, k, n))?;
    //dbg!(&name);

    let pipeline = context.shared_context().load_pipeline(
        LibraryName::Ggml,
        &name,
    )?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, size_of::<GgmlGemvParams>() as u64, &params as *const _ as *const _);        
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);

        let ny = (params.ne11 as u64 + nrows - 1) / nrows;
        let grid_size = MTLSize {
            width: n as u64,
            height: ny as u64,
            depth: /* batch_size_out */ b as u64,
        };
        let group_size = MTLSize { width: nth0, height: nth1, depth: 1 };
        
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_ggml_gemm(
    context: &MetalContext,
    dts: [DatumType; 3],
    (b, m, n, k): (usize, usize, usize, usize),
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

    // GGML transposes the output, so we invert the arguments
    let el1_size = dts[1].size_of();
    let el2_size = dts[0].size_of();

    let params = GgmlGemmParams {
        ne00: k as i32,
        ne02: b as i32,
        nb01: (k * el1_size) as u64,
        nb02: (k * n * el1_size) as u64,
        nb03: (k * n * b * el1_size )as u64,
        ne12: b as i32,
        nb10: el2_size as u64,
        nb11: (k * el2_size) as u64,
        nb12: (k * m * el2_size) as u64,
        nb13: (k * m * b * el2_size )as u64,
        ne0: n as i32,
        ne1: m as i32,
        r2: 1,
        r3: 1,
    };
    
    let pipeline = context.shared_context().load_pipeline(
        LibraryName::Ggml,
        &name,
    )?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, size_of::<GgmlGemmParams>() as u64, &params as *const _ as *const _);        
        encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
        encoder.set_buffer(2, Some(a_buffer), a_offset as NSUInteger);
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
    use std::time::Duration;

    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;
    use crate::kernels::matmul::MlxGemm;

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
        run_mmm_test_case::<GgmlGemm>((3, 8, 64, 200), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((10, 25, 512, 320), false, true, DatumType::F32, DatumType::F16)?;
        Ok(())
    }

    #[warn(dead_code)]
    #[test]
    fn test_ggml_vs_mlx() -> TractResult<()> {
        let n_iter = 10;

        run_mmm_test_case::<GgmlGemm>((2, 1, 32, 2), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F16)?;

        let mut ggml_duration = Duration::default();
        for _ in 0..n_iter {
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F32)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((2, 1, 32, 2), false, true, DatumType::F32, DatumType::F32)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((3, 8, 64, 200), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((10, 25, 512, 320), false, true, DatumType::F32, DatumType::F16)?;
        }

        run_mmm_test_case::<MlxGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((3, 8, 64, 200), false, true, DatumType::F16, DatumType::F16)?;
        let mut mlx_duration = Duration::default();
        for _ in 0..n_iter {
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 5, 64, 2), false, true, DatumType::F32, DatumType::F32)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((2, 1, 32, 2), false, true, DatumType::F32, DatumType::F32)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 5, 64, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((3, 8, 64, 200), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((10, 25, 512, 320), false, true, DatumType::F16, DatumType::F16)?;
        }

        println!("Various matrice' sizes: Mlx duration: {:}. Ggml duration: {}\n", mlx_duration.as_millis(), ggml_duration.as_millis());       
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
       run_mmm_test_case::<GgmlGemm>((4, 4, 156, 30), false, true, DatumType::F32, DatumType::F16)?;

       // f16_f32
       run_mmm_test_case::<GgmlGemm>((1, 4, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
       run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F16)?;
       run_mmm_test_case::<GgmlGemm>((4, 4, 128, 7), false, true, DatumType::F32, DatumType::F16)?;

        // f16_f16
        run_mmm_test_case::<GgmlGemm>((1, 1, 2, 1), false, true, DatumType::F16, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F16, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 16, 128, 9), false, true, DatumType::F16, DatumType::F16)?;
        Ok(())
    }

    #[test]
    fn test_matvec_ggml_vs_mlx() -> TractResult<()> {
        let n_iter = 50;
        run_mmm_test_case::<GgmlGemm>((1, 8, 32, 3), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<GgmlGemm>((1, 1, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((2, 2, 128, 8), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
        run_mmm_test_case::<GgmlGemm>((1, 1, 2, 1), false, true, DatumType::F16, DatumType::F16)?;

        let mut ggml_duration = Duration::default();
        for _ in 0..n_iter {
            // f32_f32
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 8, 32, 3), false, true, DatumType::F32, DatumType::F32)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F32)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((2, 4, 128, 8), false, true, DatumType::F32, DatumType::F32)?;

            // f16_f32_1row
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 1, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 3, 62, 2), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 3, 2, 9), false, true, DatumType::F32, DatumType::F16)?;

            // f16_f32_L4
            ggml_duration += run_mmm_test_case::<GgmlGemm>((2, 2, 128, 8), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((4, 4, 156, 30), false, true, DatumType::F32, DatumType::F16)?;

            // f16_f32
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 4, 32, 2), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((4, 4, 128, 7), false, true, DatumType::F32, DatumType::F16)?;

            // f16_f16
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 1, 2, 1), false, true, DatumType::F16, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 4, 61, 2), false, true, DatumType::F16, DatumType::F16)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((2, 16, 128, 9), false, true, DatumType::F16, DatumType::F16)?;
        }

        println!("Mlx monzbi");
        run_mmm_test_case::<MlxGemm>((1, 8, 32, 3), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((2, 2, 128, 8), false, true, DatumType::F16, DatumType::F16)?;

        let mut mlx_duration = Duration::default();
        for _ in 0..n_iter {
            // f32
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 8, 32, 3), false, true, DatumType::F32, DatumType::F32)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 61, 2), false, true, DatumType::F32, DatumType::F32)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((2, 4, 128, 8), false, true, DatumType::F32, DatumType::F32)?;

            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 1, 32, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 3, 62, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 3, 2, 9), false, true, DatumType::F16, DatumType::F16)?;
            
            // f16
            mlx_duration += run_mmm_test_case::<MlxGemm>((2, 2, 128, 8), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((4, 4, 156, 30), false, true, DatumType::F16, DatumType::F16)?;

            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 32, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 61, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((4, 4, 128, 7), false, true, DatumType::F16, DatumType::F16)?;

            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 1, 2, 1), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 4, 61, 2), false, true, DatumType::F16, DatumType::F16)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((2, 16, 128, 9), false, true, DatumType::F16, DatumType::F16)?;
        }

        println!("Various matrice' sizes: Mlx duration: {:}. Ggml duration: {}\n", mlx_duration.as_millis(), ggml_duration.as_millis());       
     
        Ok(())
    }
}
