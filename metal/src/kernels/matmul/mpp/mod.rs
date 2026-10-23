//! GEMM through Metal 4's `matmul2d` tensor op (see mpp_gemm.metal), which
//! the OS lowers to whatever matrix hardware the GPU has.
//!
//! The library only compiles where the OS carries Metal Performance
//! Primitives, so [`MppGemm::is_available`] gates it. Products with a side
//! narrower than a tile, matrix-vector ones included, go to [`MlxGemm`].

use crate::encoder::EncoderExt;
use crate::kernels::matmul::{GemmDispatchParams, GemmKernel, MlxGemm};
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{Buffer, MTLSize, NSUInteger};
use std::fmt;
use std::sync::OnceLock;
use tract_core::internal::*;

/// Edge of the C tile one threadgroup computes.
const MPP_TILE: usize = 64;

/// Simdgroups sharing a tile, as `execution_simdgroups` in the kernel.
const MPP_SIMDGROUPS: usize = 4;

/// Mirror of `MppGemmParams` in mpp_gemm.metal. Strides are in elements.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MppGemmParams {
    m: i32,
    n: i32,
    k: i32,
    batch_stride_a: i32,
    batch_stride_b: i32,
    batch_stride_c: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MppGemm;

impl MppGemm {
    /// Whether the device's OS compiles the tensor-op library.
    pub fn is_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            crate::context::metal_context().load_library(LibraryName::MppGemm).is_ok()
        })
    }
}

impl fmt::Display for MppGemm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MppGemm")
    }
}

impl GemmKernel for MppGemm {
    fn name() -> &'static str {
        "mpp"
    }

    fn dispatch_eval(
        &self,
        stream: &MetalStream,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
    ) -> TractResult<()> {
        if params.m < MPP_TILE / 2 || params.n < MPP_TILE / 2 {
            return MlxGemm.dispatch_eval(stream, params, a_buffer, b_buffer, c_buffer);
        }
        let GemmDispatchParams {
            dts,
            a_batch,
            b_batch,
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
            matches!(dts[0], DatumType::F32 | DatumType::F16)
                && dts[0] == dts[1]
                && dts[0] == dts[2],
            "MppGemm takes homogeneous f16 or f32, got {dts:?}"
        );
        ensure!(a_batch == b_batch, "MppGemm does not broadcast batches");
        let tname = if dts[0] == DatumType::F16 { "f16" } else { "f32" };
        let layout = match (transpose_a, transpose_b) {
            (false, false) => "nn",
            (false, true) => "nt",
            (true, false) => "tn",
            (true, true) => "tt",
        };
        let pipeline =
            stream.load_pipeline(LibraryName::MppGemm, &format!("mpp_gemm_{tname}_{layout}"))?;
        let gemm_params = MppGemmParams {
            m: m as i32,
            n: n as i32,
            k: k as i32,
            batch_stride_a: (m * k) as i32,
            batch_stride_b: (k * n) as i32,
            batch_stride_c: (m * n) as i32,
        };
        stream.command_buffer().encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a_buffer), a_offset as NSUInteger);
            encoder.set_buffer(1, Some(b_buffer), b_offset as NSUInteger);
            encoder.set_buffer(2, Some(c_buffer), c_offset as NSUInteger);
            encoder.set_slice(3, std::slice::from_ref(&gemm_params));
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: n.div_ceil(MPP_TILE) as _,
                    height: m.div_ceil(MPP_TILE) as _,
                    depth: a_batch as _,
                },
                MTLSize {
                    width: (pipeline.thread_execution_width() as usize * MPP_SIMDGROUPS) as _,
                    height: 1,
                    depth: 1,
                },
            );
        });
        Ok(())
    }
}
