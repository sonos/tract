use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::{ConstantValues, LibraryName, MetalStream, Value};
use anyhow::ensure;
use metal::{Buffer, MTLSize, NSUInteger};
use std::ffi::c_void;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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
            n,
            transpose_a,
            a_offset,
            transpose_b,
            b_offset,
            c_offset,
            a_strides,
            b_strides,
            ..
        } = params;

        ensure!(
            matches!(dts[0], DatumType::F32 | DatumType::F16),
            "Unsupported datum type for MlxGemm {:?}",
            dts[0]
        );
        ensure!(
            dts[0] == dts[1] && dts[0] == dts[2],
            "MlxGemm only supports homogeneous datum types. I: {:?}, {:?}. O: {:?}",
            dts[0],
            dts[1],
            dts[2]
        );

        let wide = (dts[0] == DatumType::F16 && !transpose_a && transpose_b && a_batch == 1)
            .then(|| gemv_wide_config(m, n, k, device_arch_gen::get()))
            .flatten();
        if let Some(config) = wide {
            dispatch_metal_mlx_gemv_wide(
                stream,
                dts[0],
                (m, n, k),
                &config,
                a_buffer,
                a_offset,
                b_buffer,
                b_offset,
                c_buffer,
                c_offset,
            )?;
        } else if m == 1 || n == 1 {
            dispatch_metal_mlx_gemv(
                stream,
                dts[0],
                (a_batch, m, n, k),
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
                stream,
                dts[0],
                (a_batch, m, n, k),
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
    stream: &MetalStream,
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
) -> TractResult<()> {
    ensure!(m == 1 || n == 1);
    ensure!(a_strides.len() >= 2 && b_strides.len() >= 2);
    ensure!(a_strides.len() >= 2);

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

    let tname = DeviceTensor::tname(dt)?;
    let name = format!("gemv_{t_mat}{tname}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc0_axpby0");
    let pipeline = stream.load_pipeline(LibraryName::MlxGemv, &name)?;

    let command_buffer = stream.command_buffer();
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
    });
    Ok(())
}

// From https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/lib.rs
#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mlx_gemm(
    stream: &MetalStream,
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
) -> TractResult<()> {
    ensure!(rhs_stride.len() >= 2);
    ensure!(lhs_stride.len() >= 2);

    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let a_trans = lhs_transpose;
    let b_trans = rhs_transpose;

    if a_trans {
        // (k, m)
        ensure!(
            lhs_m1 == 1 && lhs_m2 == m,
            "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{m}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    } else {
        // (m, k)
        ensure!(
            lhs_m1 == 1 && lhs_m2 == k,
            "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    }

    if b_trans {
        // (n, k)
        ensure!(
            rhs_m1 == 1 && rhs_m2 == k,
            "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    } else {
        // (k, n)
        ensure!(
            rhs_m1 == 1 && rhs_m2 == n,
            "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{n}, 1] {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
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

    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxGemm, &name, constants)?;

    let command_buffer = stream.command_buffer();
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
    });
    if debug {
        stream.wait_until_completed()?;
    }

    Ok(())
}

pub fn kernel_name_gemm(
    dt: DatumType,
    transpose_a: bool,
    transpose_b: bool,
) -> TractResult<String> {
    let t_a = if transpose_a { "t" } else { "n" };
    let t_b = if transpose_b { "t" } else { "n" };

    let tname = DeviceTensor::tname(dt)?;
    Ok(format!("gemm_{t_a}{t_b}_{tname}_{tname}_32_32_16_2_2"))
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use crate::kernels::matmul::GemmImpl;
    use crate::kernels::matmul::tests::run_mmm_test_case;
    use tract_gpu::tensor::{DeviceTensor, IntoDevice};

    #[test]
    fn test_mlx_gemv_compilation() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            stream.load_library(LibraryName::MlxGemv)
        })?;
        Ok(())
    }

    #[test]
    fn test_mlx_gemm() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let (b, m, n, k) = (10, 32, 32, 16);
            let a = Tensor::from_shape(
                &[b, m, k],
                &(0..b * m * k).map(|_f| 1.0_f32).collect::<Vec<_>>(),
            )?
            .into_device()?;
            let b = Tensor::from_shape(
                &[b, k, n],
                &(0..b * n * k).map(|_f| 1.0_f32).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let c = GemmImpl::<MlxGemm>::default().eval(stream, &a, &b)?;

            let expected_c = Tensor::from_shape(&[10, 32, 32], &vec![16.0; 10 * 32 * 32])?;

            let c = c.to_host()?;
            c.close_enough(&expected_c, Approximation::Approximate)?;
            assert!(c.close_enough(&expected_c, Approximation::Approximate).is_ok());

            let (b, m, n, k) = (2, 2, 4, 3);
            let a = DeviceTensor::from_shape(
                &[b, m, k],
                &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?;
            let b = DeviceTensor::from_shape(
                &[b, k, n],
                &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?;

            let c = GemmImpl::<MlxGemm>::default().eval(stream, &a, &b)?;

            let expected_c = Tensor::from_shape(
                &[2, 2, 4],
                &[
                    20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0,
                    488.0, 518.0, 548.0, 578.0,
                ],
            )?;

            assert!(c.to_host()?.close_enough(&expected_c, Approximation::Approximate).is_ok());
            Ok(())
        })
    }

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 1), false, false, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((10, 1, 4, 4), false, false, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((5, 1, 15, 7), false, true, DatumType::F32, DatumType::F32)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<MlxGemm>((1, 3, 5, 4), false, false, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((1, 2, 5, 10), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 4), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>((1, 4, 4, 200), false, true, DatumType::F32, DatumType::F32)?;
        run_mmm_test_case::<MlxGemm>(
            (1, 25, 1280, 32000),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        Ok(())
    }
}

/// Apple GPU architecture generation from `-[MTLDevice architecture].name`
/// ("applegpu_g16g" -> 16). mlx gates several kernels on it: M1=13, M3=15,
/// M4=16.
// The objc msg_send!/sel! macros expand to a cargo-clippy cfg check that older
// toolchains report at the call site; the module-level allow covers it.
#[allow(unexpected_cfgs)]
mod device_arch_gen {
    use metal::foreign_types::ForeignTypeRef;
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};

    pub fn get() -> Option<u32> {
        static GEN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *GEN.get_or_init(|| unsafe {
            let device = metal::Device::system_default()?;
            let dev: *mut Object = device.as_ref().as_ptr() as *mut Object;
            let responds: bool = msg_send![dev, respondsToSelector: sel!(architecture)];
            if !responds {
                return None;
            }
            let arch: *mut Object = msg_send![dev, architecture];
            if arch.is_null() {
                return None;
            }
            let name_obj: *mut Object = msg_send![arch, name];
            if name_obj.is_null() {
                return None;
            }
            let cstr: *const std::os::raw::c_char = msg_send![name_obj, UTF8String];
            if cstr.is_null() {
                return None;
            }
            std::ffi::CStr::from_ptr(cstr)
                .to_string_lossy()
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
    }
}

/// Launch parameters for `gemv_wide`, mirroring mlx `gemv_wide_config`.
pub(crate) struct GemvWideConfig {
    vecs_per_tg: usize,
    k_lanes: usize,
    grid_x: usize,
}

/// mlx routes `x[M, K] @ w[N, K]^T` here for M in 2..=15: a padded GEMM tile
/// wastes most of its rows, while this streams the weight block once per
/// register tile of vectors. mlx keeps it off below architecture generation 15
/// (M3), where load issue rate rather than bandwidth is the limit.
pub(crate) fn gemv_wide_config(
    m: usize,
    n: usize,
    k: usize,
    arch_gen: Option<u32>,
) -> Option<GemvWideConfig> {
    if arch_gen.is_none_or(|g| g < 15) {
        return None;
    }
    if !(2..=15).contains(&m) || n <= 1 || !k.is_multiple_of(4) {
        return None;
    }
    let passes = m.div_ceil(5);
    if passes > 3 {
        return None;
    }
    let full_simd = passes == 1 || n <= 64;
    Some(GemvWideConfig {
        vecs_per_tg: m.div_ceil(passes),
        k_lanes: if full_simd { 32 } else { 16 },
        grid_x: if n >= 65536 { 1 } else { passes },
    })
}

/// `c[m, n] = a[m, k] @ b[n, k]^T` for a small number of rows. Contiguous,
/// single-batch, f16/f32 only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_metal_mlx_gemv_wide(
    stream: &MetalStream,
    dt: DatumType,
    (m, n, k): (usize, usize, usize),
    config: &GemvWideConfig,
    a_buffer: &Buffer,
    a_offset: usize,
    b_buffer: &Buffer,
    b_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> TractResult<()> {
    let tname = DeviceTensor::tname(dt)?;
    let name = format!("gemv_wide_{tname}_nv{}_kl{}", config.vecs_per_tg, config.k_lanes);
    let constants = Some(ConstantValues::new(vec![
        (0, Value::Bool(false)), // gemv_wide_has_batch
        (1, Value::Bool(false)), // gemv_wide_do_axpby
    ]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxGemv, &name, constants)?;

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(b_buffer), b_offset as _);
        encoder.set_buffer(1, Some(a_buffer), a_offset as _);
        encoder.set_buffer(3, Some(output), output_offset as _);
        let i32_at = |encoder: &metal::ComputeCommandEncoderRef, ix: u64, v: i32| {
            encoder.set_bytes(
                ix,
                std::mem::size_of::<i32>() as u64,
                &v as *const i32 as *const c_void,
            )
        };
        i32_at(encoder, 4, k as i32); // in_vec_size
        i32_at(encoder, 5, n as i32); // out_vec_size
        i32_at(encoder, 6, m as i32); // M
        i32_at(encoder, 7, k as i32); // matrix_ld
        i32_at(encoder, 8, k as i32); // vector_ld
        i32_at(encoder, 11, 1); // batch_ndim
        i32_at(encoder, 12, 1); // batch_shape
        let zero = 0_i64;
        encoder.set_bytes(13, 8, &zero as *const i64 as *const c_void);
        encoder.set_bytes(14, 8, &zero as *const i64 as *const c_void);
        let group = MTLSize { width: 32, height: (config.k_lanes / 8) as _, depth: 1 };
        let grid = MTLSize { width: config.grid_x as _, height: n.div_ceil(4) as _, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

#[cfg(test)]
mod gemv_wide_tests {
    use super::*;
    use crate::utils::{get_metal_buffer, with_borrowed_metal_stream};
    use tract_gpu::tensor::IntoDevice;

    fn ramp(dt: DatumType, shape: &[usize], seed: usize) -> TractResult<Tensor> {
        let len: usize = shape.iter().product();
        let v: Vec<f32> =
            (0..len).map(|i| (((i * 7 + seed * 13) % 29) as f32 - 14.0) / 64.0).collect();
        Ok(Tensor::from_shape(shape, &v)?.cast_to_dt(dt)?.into_owned())
    }

    // The mlx gate keeps gemv_wide off below architecture generation 15, so the
    // config is built by hand here to exercise the kernel on any device.
    fn check(dt: DatumType, m: usize, n: usize, k: usize) -> TractResult<()> {
        let a = ramp(dt, &[m, k], 1)?;
        let b = ramp(dt, &[n, k], 2)?;
        let expected = {
            let a32 = a.cast_to::<f32>()?.into_owned();
            let b32 = b.cast_to::<f32>()?.into_owned();
            let (av, bv) = unsafe {
                (a32.as_slice_unchecked::<f32>().to_vec(), b32.as_slice_unchecked::<f32>().to_vec())
            };
            let mut out = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = (0..k).map(|x| av[i * k + x] * bv[j * k + x]).sum::<f32>();
                }
            }
            Tensor::from_shape(&[m, n], &out)?.cast_to_dt(dt)?.into_owned()
        };
        let config = gemv_wide_config(m, n, k, Some(15))
            .with_context(|| format!("no config for m={m} n={n} k={k}"))?;
        let got = with_borrowed_metal_stream(|stream| {
            let ad = a.clone().into_device()?;
            let bd = b.clone().into_device()?;
            let cd = unsafe { DeviceTensor::uninitialized_dt(dt, &[m, n])? };
            dispatch_metal_mlx_gemv_wide(
                stream,
                dt,
                (m, n, k),
                &config,
                &get_metal_buffer(&ad),
                ad.buffer_offset(),
                &get_metal_buffer(&bd),
                bd.buffer_offset(),
                &get_metal_buffer(&cd),
                cd.buffer_offset(),
            )?;
            stream.wait_until_completed()?;
            Ok(cd.to_host()?.into_tensor())
        })?;
        expected.close_enough(&got, Approximation::Approximate).with_context(|| {
            format!("dt={dt:?} m={m} n={n} k={k} nv={} kl={}", config.vecs_per_tg, config.k_lanes)
        })
    }

    #[test]
    fn gemv_wide_f16() -> TractResult<()> {
        for m in 2..=15 {
            check(DatumType::F16, m, 128, 256)?;
        }
        Ok(())
    }

    #[test]
    fn gemv_wide_f32() -> TractResult<()> {
        for m in [2usize, 5, 8, 15] {
            check(DatumType::F32, m, 128, 256)?;
        }
        Ok(())
    }

    #[test]
    fn gemv_wide_tail_rows() -> TractResult<()> {
        check(DatumType::F16, 3, 130, 260)?;
        check(DatumType::F16, 7, 63, 2048)?;
        check(DatumType::F32, 4, 2, 512)
    }

    // Skinny f16 through the real MlxGemm dispatch: on an M3-or-later GPU this
    // is the gemv_wide path, elsewhere the pre-existing one.
    #[test]
    fn mlx_gemm_skinny_f16_matches_reference() -> TractResult<()> {
        for m in 2..=15 {
            crate::kernels::matmul::tests::run_mmm_test_case::<MlxGemm>(
                (1, m, 256, 128),
                false,
                true,
                DatumType::F16,
                DatumType::F16,
            )?;
        }
        Ok(())
    }

    #[test]
    fn gemv_wide_config_matches_mlx_gate() {
        assert!(gemv_wide_config(2, 128, 256, Some(13)).is_none()); // pre-M3
        assert!(gemv_wide_config(1, 128, 256, Some(16)).is_none()); // M == 1 -> gemv
        assert!(gemv_wide_config(16, 128, 256, Some(16)).is_none()); // 4 passes
        assert!(gemv_wide_config(2, 128, 254, Some(16)).is_none()); // K % 4
        let c = gemv_wide_config(8, 128, 256, Some(16)).unwrap();
        assert_eq!((c.vecs_per_tg, c.k_lanes, c.grid_x), (4, 16, 2));
        let c = gemv_wide_config(4, 128, 256, Some(16)).unwrap();
        assert_eq!((c.vecs_per_tg, c.k_lanes, c.grid_x), (4, 32, 1));
        let c = gemv_wide_config(4, 131072, 256, Some(16)).unwrap();
        assert_eq!(c.grid_x, 1);
    }
}
