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
    m: u32,
    k: u32,
    n: u32,
    channel_a: u32,
    channel_b: u32,
    a_strides: [u64; 4],
    b_strides: [u64; 4],
    channel_broadcast_ratio: i16,
    batch_broadcast_ratio: i16,
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

        if true {
            dispatch_metal_ggml_gemm(
                context,
                dts,
                (batch, m, n, k),
                a_offset,
                a_buffer,
                transpose_a,
                b_offset,
                b_buffer,
                transpose_b,
                c_buffer,
                c_offset,
            )?;
        } else {
            ensure!((!transpose_a || m==1) && (transpose_b || n==1));
            dispatch_metal_ggml_gemv(
                context,
                dts,
                (batch, m, n, k),
                a_offset,
                a_buffer,
                transpose_a,
                b_offset,
                b_buffer,
                transpose_b,
                c_buffer,
                c_offset,
            )?;
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_ggml_gemv(
    context: &MetalContext,
    dts: [DatumType; 3],
    (b, m, n, k): (usize, usize, usize, usize),
    a_offset: usize,
    a_buffer: &Buffer,
    transpose_a: bool,
    b_offset: usize,
    b_buffer: &Buffer,
    transpose_b: bool,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {

    let tname = MetalTensor::tname(dts[0])?;
    let name = format!("kernel_mul_mv_{tname}_{tname}");

    let n_rows = 4;
    let nth0 = 32;
    let nth1 = 1;

    let el_size = dts[0].size_of();
    let params = GgmlGemvParams{
        ne00:  k as i32,
        ne01:  n as i32,
        ne02:  b as i32,
        nb00:  el_size as u64,
        nb01:  (k * el_size) as u64,
        nb02:  (n * k * el_size) as u64,
        nb03:  (b * n * k * el_size) as u64,
        ne10:  k as i32,
        ne11:  m as i32,
        ne12:  b as i32,
        nb10:  el_size as u64,
        nb11:  (k * el_size) as u64,
        nb12:  (m * k * el_size) as u64,
        nb13:  (b * m * k * el_size) as u64,
        ne0:  n as i32,
        ne1:  m as i32,
        r2:  1,
        r3:  1,
    };
    
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

        let ny = (params.ne11 + n_rows - 1)/n_rows;
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
    transpose_a: bool,
    b_offset: usize,
    b_buffer: &Buffer,
    transpose_b: bool,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    assert!(k % 32 == 0);
    ensure!(matches!(dts[0], DatumType::F32));

    let tname = MetalTensor::tname(dts[0])?;
    let name = format!("kernel_mul_mm_{tname}_{tname}");

    // GGML transposes the output, so we invert the arguments

    let a_strides = if !transpose_a { [(k * m * b * 4 )as u64, (k * m * 4) as u64, (k * 4) as u64, 4] } 
                                        else { [(k * m * b * 4 )as u64, (k * m  * 4) as u64, 4, (m * 4) as u64] };
    let b_strides = if !transpose_b { [(k * n * b * 4 )as u64, (k * n * 4) as u64, 4, (n * 4) as u64] } 
                                         else { [(k * m * b * 4 )as u64, (k * m  * 4) as u64, (k * 4) as u64, 4] };
    let params = GgmlGemmParams {
        m: n as u32,
        k: k as u32,
        n: m as u32,
        channel_a: 1,
        channel_b: 1,
        a_strides: b_strides,
        b_strides: a_strides,
        channel_broadcast_ratio: 1,
        batch_broadcast_ratio: 1,
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
        run_mmm_test_case::<GgmlGemm>((1, 3, 32, 2), false, true)?;
        run_mmm_test_case::<GgmlGemm>((1, 3, 1536, 10), false, false)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 32, 4), true, false)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 64, 200), false, true)?;
        run_mmm_test_case::<GgmlGemm>((1, 25, 1280, 32000), true, true)?;
        Ok(())
    }

    //#[warn(dead_code)]
    #[test]
    fn test_ggml_vs_mlx() -> TractResult<()> {
        let n_iter = 50;
        let (b, m, k , n) = (1, 25, 1280, 320);

        for trans_a in [false, true].into_iter() {
            for trans_b in [false, true].into_iter() {

                println!("Profiling with trans_a {trans_a} and trans_b {trans_b}");
                run_mmm_test_case::<GgmlGemm>((b, m, k, n), false, true)?;

                let mut ggml_duration = Duration::default();
                for _ in 0..n_iter {
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((b, m, k, n), trans_a, trans_b)?;
                }

                run_mmm_test_case::<MlxGemm>((b, m, k , n), trans_a, trans_b)?;

                let mut mlx_duration = Duration::default();
                for _ in 0..n_iter {
                    mlx_duration += run_mmm_test_case::<MlxGemm>((b, m, k, n), trans_a, trans_b)?;
                }

                println!("Big B Matrix: Mlx duration: {:}. Ggml duration: {}", mlx_duration.as_millis(), ggml_duration.as_millis());

                run_mmm_test_case::<GgmlGemm>((b, m, k, n), trans_a, trans_b)?;

                let mut ggml_duration = Duration::default();
                for _ in 0..n_iter {
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 20, 256, 15), trans_a, trans_b)?;
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 30, 1536, 30), trans_a, trans_b)?;
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 40, 512, 45), trans_a, trans_b)?;
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 64, 1024, 96), trans_a, trans_b)?;
                    ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 25, 1280, 320), trans_a, trans_b)?;
                }

                run_mmm_test_case::<MlxGemm>((b, m, k , n), trans_a, trans_b)?;

                let mut mlx_duration = Duration::default();
                for _ in 0..n_iter {
                    mlx_duration += run_mmm_test_case::<MlxGemm>((1, 20, 256, 15), trans_a, trans_b)?;
                    mlx_duration += run_mmm_test_case::<MlxGemm>((1, 30, 1536, 30), trans_a, trans_b)?;
                    mlx_duration += run_mmm_test_case::<MlxGemm>((1, 40, 512, 45), trans_a, trans_b)?;
                    mlx_duration += run_mmm_test_case::<MlxGemm>((1, 64, 1024, 96), trans_a, trans_b)?;
                    mlx_duration += run_mmm_test_case::<MlxGemm>((1, 25, 1280, 320), trans_a, trans_b)?;
                }

                println!("Various matrice' sizes: Mlx duration: {:}. Ggml duration: {}\n", mlx_duration.as_millis(), ggml_duration.as_millis());       
            }
        }
        Ok(())
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        run_mmm_test_case::<GgmlGemm>((1, 8, 32, 2), false, true)?;
        run_mmm_test_case::<GgmlGemm>((1, 1, 4, 4), false, true)?;
        run_mmm_test_case::<GgmlGemm>((1, 4, 128, 2), false, true)?;
        Ok(())
    }

    #[test]
    fn test_matvec_ggml_vs_mlx() -> TractResult<()> {
        let n_iter = 50;
        run_mmm_test_case::<GgmlGemm>((1, 20, 256, 15), false, true)?;

        let mut ggml_duration = Duration::default();
        for _ in 0..n_iter {
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 20, 256, 15), false, true)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 30, 1536, 30), false, true)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 40, 512, 45), false, true)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 64, 1024, 96), false, true)?;
            ggml_duration += run_mmm_test_case::<GgmlGemm>((1, 25, 1280, 320), false, true)?;
        }

        run_mmm_test_case::<MlxGemm>((1, 20, 256, 15), false, true)?;

        let mut mlx_duration = Duration::default();
        for _ in 0..n_iter {
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 20, 256, 15), false, true)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 30, 1536, 30), false, true)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 40, 512, 45), false, true)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 64, 1024, 96), false, true)?;
            mlx_duration += run_mmm_test_case::<MlxGemm>((1, 25, 1280, 320), false, true)?;
        }

        println!("Various matrice' sizes: Mlx duration: {:}. Ggml duration: {}\n", mlx_duration.as_millis(), ggml_duration.as_millis());       
     
        Ok(())
    }
}
