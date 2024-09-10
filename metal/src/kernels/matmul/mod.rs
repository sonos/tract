mod basic_mat_mul;
mod mfa;
mod mmm_tile_8x8;
pub mod mps;

pub use basic_mat_mul::BasicMatMul;
pub use mfa::{MfaGemm, MfaGemmPrecision};
pub use mmm_tile_8x8::{metal_mmm_tile_8x8, mmm_tile_8x8};
pub use mps::MpsMatMul;

use crate::{MetalContext, MetalTensor};
use metal::Buffer;
use num_traits::One;
use std::fmt;
use tract_core::{internal::*, ndarray, ndarray::Dimension};

#[cfg(target_os = "macos")]
pub type DefaultGemmImpl = GemmImpl<MpsMatMul>;
#[cfg(target_os = "ios")]
pub type DefaultGemmImpl = GemmImpl<MpsMatMul>;

pub trait GemmKernel: fmt::Display + fmt::Debug + Clone + Default {
    fn is_supported_dt(&self, dt: DatumType) -> bool;

    #[allow(clippy::too_many_arguments)]
    fn dispatch_eval(
        &self,
        context: &MetalContext,
        dt: DatumType,
        m: usize,
        k: usize,
        n: usize,
        a_buffer: &Buffer,
        a_offset: usize,
        a_transpose: bool,
        b_buffer: &Buffer,
        b_offset: usize,
        b_transpose: bool,
        c_buffer: &Buffer,
        c_offset: usize,
    ) -> TractResult<()>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GemmImpl<M: GemmKernel> {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub matmul: M,
}

impl<M: GemmKernel> fmt::Display for GemmImpl<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.matmul)
    }
}

impl<M: GemmKernel> GemmImpl<M> {
    pub fn new(transpose_a: bool, transpose_b: bool) -> Self {
        Self { transpose_a, transpose_b, matmul: M::default() }
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        self.matmul.is_supported_dt(dt)
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
            let c = MetalTensor::zero_dt(c_dt, &c_shape)?;
            c.retain_until_completion();

            if k == 0 {
                return Ok(c);
            }

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

                self.matmul
                    .dispatch_eval(
                        context,
                        c_dt,
                        m,
                        n,
                        k,
                        a.metal(),
                        a_offset as usize * c_dt.size_of(),
                        self.transpose_a,
                        b.metal(),
                        b_offset as usize * c_dt.size_of(),
                        self.transpose_b,
                        c.metal(),
                        c_offset as usize * c_dt.size_of(),
                    )
                    .with_context(|| {
                        anyhow!(
                        "Error while performing MatMul with {:?} (a: {:?}), (b: {:?}) = (c: {:?})",
                        self.matmul,
                        a.shape(),
                        b.shape(),
                        c_shape
                    )
                    })?;
            }

            Ok(c)
        }
    }
}
