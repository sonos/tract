use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::MetalTensor;
use crate::{ConstantValues, LibraryName, MetalContext, Value};
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use std::ffi::c_void;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug)]
#[repr(C)]
struct MlxGemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

#[derive(Debug, Default)]
#[repr(C)]
#[allow(non_snake_case)]
struct GEMMDebug {
    TM_stride: i32,
    TN_stride: i32,
    WM: i32,
    WN: i32,
    TM: i32,
    TN: i32,
    num_threads_in_simd: i32,
    num_simd_group: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MlxGemm;

impl fmt::Display for MlxGemm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MlxGemm")
    }
}

impl GemmKernel for MlxGemm {
    fn name() -> &'static str {
        "mlx"
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

        let a_strides = if transpose_a {
            natural_strides(&[batch, k, m])
        } else {
            natural_strides(&[batch, m, k])
        };
        let b_strides = if transpose_b {
            natural_strides(&[batch, n, k])
        } else {
            natural_strides(&[batch, k, n])
        };

        if m == 1 || n == 1 {
            dispatch_metal_mlx_gemv(
                context,
                dt,
                (batch, m, n, k),
                unsafe { std::mem::transmute::<&[isize], &[usize]>(a_strides.as_slice()) },
                a_offset,
                a_buffer,
                transpose_a,
                unsafe { std::mem::transmute::<&[isize], &[usize]>(b_strides.as_slice()) },
                b_offset,
                b_buffer,
                transpose_b,
                c_buffer,
                c_offset,
            )?;
        } else {
            dispatch_metal_mlx_gemm(
                context,
                dt,
                (batch, m, n, k),
                unsafe { std::mem::transmute::<&[isize], &[usize]>(a_strides.as_slice()) },
                a_offset,
                a_buffer,
                transpose_a,
                unsafe { std::mem::transmute::<&[isize], &[usize]>(b_strides.as_slice()) },
                b_offset,
                b_buffer,
                transpose_b,
                c_buffer,
                c_offset,
                false,
            )?;
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mlx_gemv(
    context: &MetalContext,
    dt: DatumType,
    (b, m, n, k): (usize, usize, usize, usize),
    a_strides: &[usize],
    a_offset: usize,
    a_buffer: &Buffer,
    a_trans: bool,
    b_strides: &[usize],
    b_offset: usize,
    b_buffer: &Buffer,
    b_trans: bool,
    output: &Buffer,
    output_offset: usize,
) -> Result<()> {
    ensure!(m == 1 || n == 1);
    assert!(a_strides.len() >= 2 && b_strides.len() >= 2);
    assert!(a_strides.len() >= 2);
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    let lda = if a_trans { m } else { k };
    let ldb = if b_trans { k } else { n };

    // Determine dispatch kernel
    let (mut tm, mut tn) = (4, 4);
    #[allow(unused_assignments)]
    let (mut sm, mut sn) = (1, 32);
    let (mut bm, mut bn) = (1, 1);

    // Map (m, k, n) to Matrix * Vector

    let is_b_matrix = n != 1;
    let mv_m = if is_b_matrix { n } else { m };
    let mv_k = k;
    let mv_ld = if is_b_matrix { ldb } else { lda };
    let mv_trans = if is_b_matrix { !b_trans } else { a_trans };
    let mat_batch_stride = if is_b_matrix { b_strides[0] } else { a_strides[0] };
    let vec_batch_stride = if is_b_matrix { a_strides[0] } else { b_strides[0] };

    let n_out_per_tgp = if mv_trans {
        (sm, sn) = if mv_k >= 8192 && mv_m >= 2048 { (4, 8) } else { (8, 4) };
        bn = if mv_m >= 2048 {
            16
        } else if mv_m >= 512 {
            4
        } else {
            2
        };
        // Specialized kernel for very small outputs
        tn = if mv_m < tn { 1 } else { tn };

        bn * sn * tn
    } else {
        bm = if mv_m >= 4096 { 8 } else { 4 };
        sn = 32;
        // Specialized kernel for very small outputs
        tm = if mv_m < tm { 1 } else { tm };
        bm * sm * tm
    };

    let n_tgp = mv_m.div_ceil(n_out_per_tgp);

    let group_size = MTLSize { width: 32, height: bn as _, depth: bm as _ };
    let grid_size = MTLSize {
        width: n_tgp as _,
        height: 1,
        depth: /* batch_size_out */ b as u64,
    };

    let t_mat = if mv_trans { "t_" } else { "" };

    let tname = MetalTensor::tname(dt)?;
    let name = format!("gemv_{t_mat}{tname}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc0_axpby0");
    let pipeline = context.shared_context().load_pipeline(LibraryName::MlxGemv, &name)?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        if is_b_matrix {
            encoder.set_buffer(0, Some(b_buffer), b_offset as _);
            encoder.set_buffer(1, Some(a_buffer), a_offset as _);
        } else {
            encoder.set_buffer(0, Some(a_buffer), a_offset as _);
            encoder.set_buffer(1, Some(b_buffer), b_offset as _);
        }
        encoder.set_buffer(3, Some(output), output_offset as _);

        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(mv_k as i32) as *const i32 as *const c_void,
        );

        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &(mv_m as i32) as *const i32 as *const c_void,
        );

        encoder.set_bytes(
            6,
            std::mem::size_of::<i32>() as u64,
            &(mv_ld as i32) as *const i32 as *const c_void,
        );

        encoder.set_bytes(
            9, // batch_ndim
            std::mem::size_of::<i32>() as u64,
            &1_i32 as *const i32 as *const c_void,
        );
        encoder.set_bytes(
            10, // batch_shape
            std::mem::size_of::<i32>() as u64,
            &(b as i32) as *const i32 as *const c_void,
        );
        encoder.set_bytes(
            11, // batch_strides_vec
            std::mem::size_of::<usize>() as u64,
            &vec_batch_stride as *const usize as *const c_void,
        );
        encoder.set_bytes(
            12, // batch_strides_mat
            std::mem::size_of::<usize>() as u64,
            &mat_batch_stride as *const usize as *const c_void,
        );

        encoder.use_resource(a_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(b_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
    });
    Ok(())
}

// From https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/lib.rs
#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mlx_gemm(
    context: &MetalContext,
    dt: DatumType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    lhs_transpose: bool,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    rhs_transpose: bool,
    output: &Buffer,
    output_offset: usize,
    debug: bool,
) -> Result<()> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let a_trans = lhs_transpose;
    let b_trans = rhs_transpose;

    if a_trans {
        // (k, m)
        ensure!(lhs_m1 == 1 && lhs_m2 == m, "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{m}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    } else {
        // (m, k)
        ensure!(lhs_m1 == 1 && lhs_m2 == k, "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    }

    if b_trans {
        // (n, k)
        ensure!(rhs_m1 == 1 && rhs_m2 == k, "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    } else {
        // (k, n)
        ensure!(rhs_m1 == 1 && rhs_m2 == n, "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{n}, 1] {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    }

    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
        (400, Value::Bool(debug)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a =
        if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] } else { m * k };
    let batch_stride_b =
        if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] } else { n * k };

    let gemm_params = MlxGemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if a_trans { m } else { k } as i32,
        ldb: if b_trans { k } else { n } as i32,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    let batch_strides = [gemm_params.batch_stride_a, gemm_params.batch_stride_b];

    let name = kernel_name_gemm(dt, a_trans, b_trans)?;

    let pipeline = context.shared_context().load_pipeline_with_constants(
        LibraryName::MlxGemm,
        &name,
        constants,
    )?;

    let command_buffer = context.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
        encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
        encoder.set_buffer(3, Some(output), output_offset as NSUInteger);
        encoder.set_bytes(
            4,
            std::mem::size_of::<MlxGemmParams>() as u64,
            &gemm_params as *const MlxGemmParams as *const c_void,
        );
        encoder.set_bytes(
            6, // batch_shape
            std::mem::size_of::<i32>() as u64,
            &(b as i32) as *const i32 as *const c_void,
        );
        encoder.set_bytes(
            7,
            (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
            batch_strides.as_ptr() as *const c_void,
        );

        let gemm_debug = Box::<GEMMDebug>::default();
        if debug {
            let gemm_debug_size = core::mem::size_of_val(&gemm_debug) as NSUInteger;
            let gemm_debug_buffer = context.device().new_buffer_with_bytes_no_copy(
                gemm_debug.as_ref() as *const GEMMDebug as *const core::ffi::c_void,
                gemm_debug_size,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            encoder.set_buffer(16, Some(&gemm_debug_buffer), 0);
        }

        let grid_size = MTLSize {
            width: tn as u64,
            height: tm as u64,
            depth: /* batch_size_out */ b as u64,
        };
        let group_size = MTLSize { width: 32, height: wn, depth: wm };
        encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
    });
    if debug {
        context.wait_until_completed()?;
        //log::debug!("{:#?}", gemm_debug);
    }

    Ok(())
}

pub fn kernel_name_gemm(dt: DatumType, transpose_a: bool, transpose_b: bool) -> Result<String> {
    let t_a = if transpose_a { "t" } else { "n" };
    let t_b = if transpose_b { "t" } else { "n" };
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16));
    let tname = MetalTensor::tname(dt)?;
    Ok(format!("gemm_{t_a}{t_b}_{tname}_{tname}_32_32_16_2_2"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;
    use crate::kernels::matmul::GemmImpl;
    use crate::{IntoMetal, MetalTensor};

    #[test]
    fn test_mlx_gemv_compilation() -> Result<()> {
        crate::METAL_CONTEXT
            .with_borrow(|context| context.shared_context().load_library(LibraryName::MlxGemv))?;
        Ok(())
    }

    #[test]
    fn test_mlx_gemm() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let (b, m, n, k) = (10, 32, 32, 16);
                let a = Tensor::from_shape(
                    &[b, m, k],
                    &(0..b * m * k).map(|_f| 1.0 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let b = Tensor::from_shape(
                    &[b, k, n],
                    &(0..b * n * k).map(|_f| 1.0 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let c = GemmImpl::<MlxGemm>::default().eval(context, &a, &b)?;

                let expected_c = Tensor::from_shape(&[10, 32, 32], &vec![16.0; 10 * 32 * 32])?;

                let c = c.to_cpu()?;
                c.close_enough(&expected_c, Approximation::Approximate)?;
                assert!(c.close_enough(&expected_c, Approximation::Approximate).is_ok());

                let (b, m, n, k) = (2, 2, 4, 3);
                let a = MetalTensor::from_shape(
                    &[b, m, k],
                    &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;
                let b = MetalTensor::from_shape(
                    &[b, k, n],
                    &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;

                let c = GemmImpl::<MlxGemm>::default().eval(context, &a, &b)?;

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
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 1), false, false)?;
        run_mmm_test_case::<MlxGemm>((10, 1, 4, 4), false, false)?;
        run_mmm_test_case::<MlxGemm>((5, 1, 15, 7), false, true)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<MlxGemm>((1, 3, 5, 4), false, false)?;
        run_mmm_test_case::<MlxGemm>((1, 2, 5, 10), false, true)?;
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 4), false, true)?;
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 200), false, true)?;
        run_mmm_test_case::<MlxGemm>((1, 25, 1280, 32000), false, true)?;
        Ok(())
    }
}
