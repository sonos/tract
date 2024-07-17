use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::Result;
use derive_new::new;
use metal::{Buffer, MTLSize, NSUInteger};
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::Dimension;

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct BasicMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl fmt::Display for BasicMatMul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BasicMatMul")
    }
}

impl BasicMatMul {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::tname(dt).is_ok()
    }

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

    pub fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2 + self.transpose_a as usize].clone());
        output.push(b[rank - 2 + !self.transpose_b as usize].clone());
        output
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        let output = self.dispatch_eval(context, a, b)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        a.retain_until_completion();
        b.retain_until_completion();

        let c_dt = a.datum_type();
        let c_shape = self.output_shape(a.shape(), b.shape());
        let rank = c_shape.len();
        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a.shape()[a.rank() - 2 + !self.transpose_a as usize];

        unsafe {
            let c = MetalTensor::uninitialized_dt(c_dt, &c_shape)?;
            c.retain_until_completion();
            let silent_a_axis = c.rank() - a.rank();
            let silent_b_axis = c.rank() - b.rank();
            for prefix in ndarray::indices(&c_shape[0..rank - 2]) {
                let mut a_offset = 0;
                let mut b_offset = 0;
                let mut c_offset = 0;
                for (axis, x) in prefix.as_array_view().iter().enumerate() {
                    if axis >= silent_a_axis && a.shape()[axis - silent_a_axis] != 1 {
                        a_offset += *x as isize * a.strides()[axis - silent_a_axis];
                    }
                    if axis >= silent_b_axis && b.shape()[axis - silent_b_axis] != 1 {
                        b_offset += *x as isize * b.strides()[axis - silent_b_axis];
                    }
                    c_offset += *x as isize * c.strides()[axis];
                }

                if n == 1 && !self.transpose_a && !self.transpose_b {
                    Self::metal_mat_vec(
                        context,
                        c_dt,
                        m,
                        k,
                        a.metal(),
                        a_offset as usize * c_dt.size_of(),
                        b.metal(),
                        b_offset as usize * c_dt.size_of(),
                        c.metal(),
                        c_offset as usize * c_dt.size_of(),
                    )?;
                } else {
                    Self::metal_mat_mul(
                        context,
                        c_dt,
                        m,
                        k,
                        n,
                        a.metal(),
                        a_offset as usize * c_dt.size_of(),
                        self.transpose_a,
                        b.metal(),
                        b_offset as usize * c_dt.size_of(),
                        self.transpose_b,
                        c.metal(),
                        c_offset as usize * c_dt.size_of(),
                    )?;
                }
            }

            Ok(c)
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
                    BasicMatMul::new(transpose_a, transpose_b).eval(context, &a, &b)?;
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
