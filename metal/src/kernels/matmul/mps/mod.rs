pub mod api;

use crate::kernels::matmul::GemmKernel;
use crate::MetalContext;
use anyhow::bail;
use api::{MPSDataType, Matrix, MatrixVectorMultiplication, Vector};
use derive_new::new;
use metal::Buffer;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsMatmulKey {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct MpsMatMul;

impl fmt::Display for MpsMatMul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MpsMatMul")
    }
}

impl GemmKernel for MpsMatMul {
    fn is_supported_dt(&self, dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
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
        let data_type = match dt {
            DatumType::F32 => MPSDataType::Float32,
            DatumType::F16 => MPSDataType::Float16,
            _ => bail!("Unsupported datum type for MpsMatMul {:?}", dt),
        };

        if m == 1 {
            let a_vector = Vector::new(a_buffer.to_owned(), a_offset as _, data_type, k as _)
                .ok_or_else(|| anyhow!("An error occured when creating MPS vector"))?;

            let b_matrix = Matrix::new(
                b_buffer.to_owned(),
                b_offset as _,
                data_type,
                if transpose_b { n } else { k } as _,
                if transpose_b { k } else { n } as _,
            )
            .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

            let c_vector = Vector::new(c_buffer.to_owned(), c_offset as _, data_type, n as _)
                .ok_or_else(|| anyhow!("An error occured when creating MPS vector"))?;

            let mat_vec = MatrixVectorMultiplication::new(
                context.device().to_owned(),
                !transpose_b,
                n as _,
                k as _,
                1.0,
                0.0,
            )
            .ok_or_else(|| {
                anyhow!("An error occured when createing MPS Matrix vector multiplication")
            })?;

            mat_vec.encode(context.command_buffer(), b_matrix, a_vector, c_vector);
            Ok(())
        } else {
            let a_matrix = Matrix::new(
                a_buffer.to_owned(),
                a_offset as _,
                data_type,
                if transpose_a { k } else { m } as _,
                if transpose_a { m } else { k } as _,
            )
            .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

            let b_matrix = Matrix::new(
                b_buffer.to_owned(),
                b_offset as _,
                data_type,
                if transpose_b { n } else { k } as _,
                if transpose_b { k } else { n } as _,
            )
            .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

            let c_matrix =
                Matrix::new(c_buffer.to_owned(), c_offset as _, data_type, m as _, n as _)
                    .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

            let matmul = context.shared_context().load_mps_matmul(&MpsMatmulKey {
                transpose_a,
                transpose_b,
                m,
                n,
                k,
            })?;

            matmul.encode(context.command_buffer(), a_matrix, b_matrix, c_matrix);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::GemmImpl;
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
                    GemmImpl::<MpsMatMul>::new(transpose_a, transpose_b).eval(context, &a, &b)?;
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
        run_test_case((2, 5, 10), false, true)?;
        run_test_case((4, 4, 4), false, true)?;
        run_test_case((4, 4, 200), false, true)?;
        run_test_case((25, 1280, 32000), false, true)?;
        Ok(())
    }
}
