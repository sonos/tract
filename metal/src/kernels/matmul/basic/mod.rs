use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::{LibraryName, MetalStream};
use anyhow::bail;
use derive_new::new;
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct BasicMatMul;

impl GemmKernel for BasicMatMul {
    fn name() -> &'static str {
        "basic"
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
            ..
        } = params;

        ensure!(
            Self::tname(dts[0]).is_ok(),
            "Unsupported datum type for Metal BasicMatmul {:?}",
            dts[0]
        );
        ensure!(
            dts[0] == dts[1] && dts[0] == dts[2],
            "Metal BasicMatmul only supports homogenous datum types. I: {:?}, {:?}. O: {:?}",
            dts[0],
            dts[1],
            dts[2]
        );

        let dt = dts[0];
        for b_idx in 0..a_batch {
            let a_offset = a_offset + b_idx * m * k * dt.size_of();
            let b_offset = b_offset + b_idx * n * k * dt.size_of();
            let c_offset = c_offset + b_idx * m * n * dt.size_of();
            if n == 1 && !transpose_a && !transpose_b {
                Self::metal_mat_vec(
                    stream, dt, m, k, a_buffer, a_offset, b_buffer, b_offset, c_buffer, c_offset,
                )?;
            } else {
                Self::metal_mat_mul(
                    stream,
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
    pub fn tname(dt: DatumType) -> TractResult<&'static str> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            _ => bail!("Unsupport dt {:?} for metal basic matmul", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(dt: DatumType, mat_vec: bool) -> TractResult<String> {
        let tname = Self::tname(dt)?;
        if mat_vec {
            Ok(format!("matmul::basic_matvec_{tname}"))
        } else {
            Ok(format!("matmul::basic_matmul_{tname}"))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn metal_mat_vec(
        stream: &MetalStream,
        dt: DatumType,
        m: usize,
        k: usize,
        lhs_buffer: &Buffer,
        lhs_offset: usize,
        rhs_buffer: &Buffer,
        rhs_offset: usize,
        output: &Buffer,
        output_offset: usize,
    ) -> TractResult<()> {
        let pipeline =
            stream.load_pipeline(LibraryName::BasicMatMul, &Self::kernel_name(dt, true)?)?;

        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as _);
            encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as _);
            encoder.set_buffer(2, Some(output), output_offset as _);
            encoder.set_bytes(3, 4, &(m as i32) as *const i32 as *const _);
            encoder.set_bytes(4, 4, &(k as i32) as *const i32 as *const _);

            // m x k * k * 1
            let grid_size =
                MTLSize { width: 1, height: m.div_ceil(4) as NSUInteger, depth: 1 as NSUInteger };
            let group_size = MTLSize { width: 32, height: 1, depth: 1 };
            encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
            encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
            encoder.use_resource(output, metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn metal_mat_mul(
        stream: &MetalStream,
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
    ) -> TractResult<()> {
        let pipeline =
            stream.load_pipeline(LibraryName::BasicMatMul, &Self::kernel_name(dt, false)?)?;

        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
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
                width: n.div_ceil(4) as NSUInteger,
                height: m.div_ceil(4) as NSUInteger,
                depth: 1 as NSUInteger,
            };
            let group_size: MTLSize = MTLSize { width: 32, height: 1, depth: 1 };
            encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
            encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
            encoder.use_resource(output, metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::tests::run_mmm_test_case;

    #[test]
    fn test_mat_vec() -> TractResult<()> {
        run_mmm_test_case::<BasicMatMul>(
            (1, 4, 4, 1),
            false,
            false,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 1, 4, 4),
            false,
            false,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 1, 15, 7),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> TractResult<()> {
        run_mmm_test_case::<BasicMatMul>(
            (1, 3, 5, 4),
            false,
            false,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 2, 5, 10),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 4, 4, 4),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 4, 4, 200),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (1, 25, 1280, 32000),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;

        run_mmm_test_case::<BasicMatMul>(
            (10, 3, 5, 4),
            false,
            false,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (10, 2, 5, 10),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (10, 4, 4, 4),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (10, 4, 4, 200),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        run_mmm_test_case::<BasicMatMul>(
            (10, 25, 1280, 32000),
            false,
            true,
            DatumType::F32,
            DatumType::F32,
        )?;
        Ok(())
    }
}
