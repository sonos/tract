#![allow(unexpected_cfgs)]

pub mod api;

use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
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
    fn name() -> &'static str {
        "mps"
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

        let data_type = match params.dt {
            DatumType::F32 => MPSDataType::Float32,
            DatumType::F16 => MPSDataType::Float16,
            _ => bail!("Unsupported datum type for MpsMatMul {:?}", params.dt),
        };

        for b_idx in 0..batch {
            let a_offset = a_offset + b_idx * m * k * dt.size_of();
            let b_offset = b_offset + b_idx * n * k * dt.size_of();
            let c_offset = c_offset + b_idx * m * n * dt.size_of();

            if m == 1 && dt == DatumType::F32 {
                // The F16 integration seems broken while running prop tests.
                // Therefore we limit the integration to F32 for the moment.

                let a_vector =
                    Vector::new(a_buffer.to_owned(), a_offset as _, data_type, k as _)
                        .ok_or_else(|| anyhow!("An error occured when creating MPS vector"))?;

                let b_matrix = Matrix::new(
                    b_buffer.to_owned(),
                    b_offset as _,
                    data_type,
                    if transpose_b { n } else { k } as _,
                    if transpose_b { k } else { n } as _,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MPS matrix"))?;

                let c_vector =
                    Vector::new(c_buffer.to_owned(), c_offset as _, data_type, n as _)
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
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        run_mmm_test_case::<MpsMatMul>((1, 4, 4, 1), false, false)?;
        run_mmm_test_case::<MpsMatMul>((1, 1, 4, 4), false, false)?;
        run_mmm_test_case::<MpsMatMul>((1, 1, 15, 7), false, true)?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<MpsMatMul>((1, 3, 5, 4), false, false)?;
        run_mmm_test_case::<MpsMatMul>((1, 2, 5, 10), false, true)?;
        run_mmm_test_case::<MpsMatMul>((1, 4, 4, 4), false, true)?;
        run_mmm_test_case::<MpsMatMul>((1, 4, 4, 200), false, true)?;
        run_mmm_test_case::<MpsMatMul>((1, 25, 1280, 32000), false, true)?;
        Ok(())
    }
}
