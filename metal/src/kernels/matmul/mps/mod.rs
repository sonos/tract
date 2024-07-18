pub mod api;

use api::{MPSDataType, Matrix};

use crate::MetalContext;
use crate::MetalTensor;
use anyhow::bail;
use derive_new::new;
use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::Dimension;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsMatmulKey {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct MpsMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl fmt::Display for MpsMatMul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MpsMatMul")
    }
}

impl MpsMatMul {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
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

        let data_type = match c_dt {
            DatumType::F32 => MPSDataType::Float32,
            DatumType::F16 => MPSDataType::Float16,
            _ => bail!("Unsupported datum type for MPSMatMul {:?}", c_dt),
        };

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

                let a_matrix = Matrix::new(
                    a.metal().to_owned(),
                    (a_offset as usize * c_dt.size_of()) as _,
                    data_type,
                    if self.transpose_a { k } else { m } as _,
                    if self.transpose_a { m } else { k } as _,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

                let b_matrix = Matrix::new(
                    b.metal().to_owned(),
                    (b_offset as usize * c_dt.size_of()) as _,
                    data_type,
                    if self.transpose_b { n } else { k } as _,
                    if self.transpose_b { k } else { n } as _,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

                let c_matrix = Matrix::new(
                    c.metal().to_owned(),
                    (c_offset as usize * c_dt.size_of()) as _,
                    data_type,
                    m as _,
                    n as _,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

                let matmul = context.shared_context().load_mps_matmul(&MpsMatmulKey {
                    transpose_a: self.transpose_a,
                    transpose_b: self.transpose_b,
                    m,
                    n,
                    k,
                })?;

                matmul.encode(context.command_buffer(), a_matrix, b_matrix, c_matrix);
            }

            Ok(c)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use tract_core::ops::einsum::BasicMatMul as TractBasicMatMul;

    fn run_test_case(
        (m, k, n): (usize, usize, usize),
        transpose_a: bool,
        transpose_b: bool,
    ) -> TractResult<()> {
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
                    MpsMatMul::new(transpose_a, transpose_b).eval(context, &a, &b)?;
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
    fn test_mat_vec() -> TractResult<()> {
        run_test_case((4, 4, 1), false, false)?;
        run_test_case((1, 4, 4), false, false)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_test_case((3, 5, 4), false, false)?;
        // run_test_case((2, 5, 10), false, true)?;
        // run_test_case((4, 4, 4), false, true)?;
        // run_test_case((4, 4, 200), false, true)?;
        // run_test_case((25, 1280, 32000), false, true)?;
        Ok(())
    }
}
