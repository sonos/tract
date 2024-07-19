use crate::kernels::matmul::GemmKernel;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::Result;
use derive_new::new;
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct BasicMatMul;

impl GemmKernel for BasicMatMul {
    fn is_supported_dt(&self, dt: DatumType) -> bool {
        Self::tname(dt).is_ok()
    }

    fn dispatch_eval(
        &self,
        context: &MetalContext,
        dt: DatumType,
        m: usize,
        n: usize,
        k: usize,
        a_buffer: &Buffer,
        a_offset: usize,
        transpose_a: bool,
        b_buffer: &Buffer,
        b_offset: usize,
        transpose_b: bool,
        c_buffer: &Buffer,
        c_offset: usize,
    ) -> TractResult<()> {
        if n == 1 && !transpose_a && !transpose_b {
            Self::metal_mat_vec(
                context, dt, m, k, a_buffer, a_offset, b_buffer, b_offset, c_buffer, c_offset,
            )?;
        } else {
            Self::metal_mat_mul(
                context,
                dt,
                m,
                k,
                n,
                a_buffer,
                a_offset,
                transpose_a,
                b_buffer,
                b_offset,
                transpose_b,
                c_buffer,
                c_offset,
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for BasicMatMul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BasicMatMul")
    }
}

impl BasicMatMul {
    pub fn tname(dt: DatumType) -> Result<&'static str> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            _ => bail!("Unsupport dt {:?} for metal basic matmul", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(dt: DatumType, mat_vec: bool) -> Result<String> {
        let tname = Self::tname(dt)?;
        if mat_vec {
            Ok(format!("matmul::basic_matvec_{tname}"))
        } else {
            Ok(format!("matmul::basic_matmul_{tname}"))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn metal_mat_vec(
        context: &MetalContext,
        dt: DatumType,
        nrows: usize,
        ncols: usize,
        lhs_buffer: &Buffer,
        lhs_offset: usize,
        rhs_buffer: &Buffer,
        rhs_offset: usize,
        output: &Buffer,
        output_offset: usize,
    ) -> Result<()> {
        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::BasicMatMul, &Self::kernel_name(dt, true)?)?;

        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as _);
        encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as _);
        encoder.set_buffer(2, Some(output), output_offset as _);
        encoder.set_bytes(3, 4, &(nrows as i32) as *const i32 as *const _);
        encoder.set_bytes(4, 4, &(ncols as i32) as *const i32 as *const _);

        let grid_size =
            MTLSize { width: 1, height: crate::utils::div_ceil(nrows, 4), depth: 1 as NSUInteger };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn metal_mat_mul(
        context: &MetalContext,
        dt: DatumType,
        m: usize,
        k: usize,
        n: usize,
        lhs_buffer: &Buffer,
        lhs_offset: usize,
        lhs_transpose: bool,
        rhs_buffer: &Buffer,
        rhs_offset: usize,
        rhs_transpose: bool,
        output: &Buffer,
        output_offset: usize,
    ) -> Result<()> {
        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::BasicMatMul, &Self::kernel_name(dt, false)?)?;

        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as _);
        encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as _);
        encoder.set_buffer(2, Some(output), output_offset as _);
        encoder.set_bytes(3, 4, &(m as i32) as *const i32 as *const _);
        encoder.set_bytes(4, 4, &(k as i32) as *const i32 as *const _);
        encoder.set_bytes(5, 4, &(n as i32) as *const i32 as *const _);
        encoder.set_bytes(6, 4, &(lhs_transpose as i32) as *const i32 as *const _);
        encoder.set_bytes(7, 4, &(rhs_transpose as i32) as *const i32 as *const _);

        let grid_size = MTLSize {
            width: crate::utils::div_ceil(n, 4),
            height: crate::utils::div_ceil(m, 4),
            depth: 1 as NSUInteger,
        };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::GemmImpl;
    use crate::IntoMetal;
    use tract_core::internal::Tensor;
    use tract_core::ops::einsum::BasicMatMul as TractBasicMatMul;

    fn run_test_case(
        (m, k, n): (usize, usize, usize),
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_shape = if !transpose_a { [m, k] } else { [k, m] };
                let b_shape = if !transpose_b { [k, n] } else { [n, k] };
                let a = Tensor::from_shape(
                    &a_shape,
                    &(0..m * k).map(|f| f as f32 / 100.0).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let b = Tensor::from_shape(
                    &b_shape,
                    &(0..k * n).map(|f| f as f32 / 100.0).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let metal_output =
                    GemmImpl::<BasicMatMul>::new(transpose_a, transpose_b).eval(context, &a, &b)?;
                let matmul = TractBasicMatMul {
                    transpose_a,
                    transpose_b,
                    transpose_c: false,
                    quantize_output: None,
                };
                let output = args_1!(
                    matmul.eval(tvec![a.to_cpu().into_tvalue(), b.to_cpu().into_tvalue()])?
                );
                output.close_enough(&metal_output.to_cpu(), Approximation::Approximate)?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_mat_vec() -> Result<()> {
        run_test_case((4, 4, 1), false, false)?;
        run_test_case((1, 4, 4), false, false)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> Result<()> {
        run_test_case((3, 5, 4), false, false)?;
        run_test_case((2, 5, 10), false, true)?;
        run_test_case((4, 4, 4), false, true)?;
        run_test_case((4, 4, 200), false, true)?;
        run_test_case((25, 1280, 32000), false, true)?;
        Ok(())
    }
}
