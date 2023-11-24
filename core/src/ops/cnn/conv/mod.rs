mod depth_wise;
mod im2col;
mod lazy_im2col;
mod q_sum_b;
mod unary;

use crate::internal::*;

pub use self::im2col::Im2Col;
pub(crate) use self::q_sum_b::QSumB;
pub use self::unary::ConvUnary;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum KernelFormat {
    #[default]
    OIHW,
    HWIO,
    OHWI,
}

impl KernelFormat {
    pub fn h_axis(&self) -> usize {
        match self {
            KernelFormat::OIHW => 2,
            KernelFormat::HWIO => 0,
            KernelFormat::OHWI => 1,
        }
    }

    pub fn spatial_shape<'a, D>(&self, full_shape: &'a [D]) -> &'a [D] {
        &full_shape[self.h_axis()..][..full_shape.len() - 2]
    }

    pub fn hw<'a, D>(&self, full_shape: &'a [D]) -> &'a [D] {
        self.spatial_shape(full_shape)
    }

    pub fn i<'a, D>(&self, full_shape: &'a [D]) -> &'a D {
        match self {
            KernelFormat::OIHW => &full_shape[1],
            KernelFormat::HWIO => &full_shape[full_shape.len() - 2],
            KernelFormat::OHWI => &full_shape[full_shape.len() - 1],
        }
    }

    pub fn o_axis<D>(&self, full_shape: &[D]) -> usize {
        match self {
            KernelFormat::OIHW | KernelFormat::OHWI => 0,
            KernelFormat::HWIO => full_shape.len() - 1,
        }
    }

    pub fn o<'a, D>(&self, full_shape: &'a [D]) -> &'a D {
        &full_shape[self.o_axis(full_shape)]
    }

    pub fn input_channels<'s, D: DimLike>(
        &self,
        full_kernel_shape: &'s [D],
        group: usize,
    ) -> Cow<'s, D> {
        match self {
            KernelFormat::OIHW => Cow::Owned(self.i(full_kernel_shape).clone() * group),
            KernelFormat::HWIO | KernelFormat::OHWI => Cow::Borrowed(self.i(full_kernel_shape)),
        }
    }

    pub fn output_channels<'s, D: DimLike>(
        &self,
        full_kernel_shape: &'s [D],
        group: usize,
    ) -> Cow<'s, D> {
        match self {
            KernelFormat::OIHW => Cow::Borrowed(self.o(full_kernel_shape)),
            KernelFormat::HWIO | KernelFormat::OHWI => {
                Cow::Owned(self.o(full_kernel_shape).clone() * group)
            }
        }
    }

    pub fn kernel_as_group_o_i_hw_ops(
        &self,
        full_shape: &[impl DimLike],
        group: usize,
    ) -> TractResult<TVec<AxisOp>> {
        let mut ops = tvec!();
        ops.push(AxisOp::Reshape(
            self.h_axis(),
            self.hw(full_shape).iter().map(|t| t.to_dim()).collect(),
            tvec!(self.hw(full_shape).iter().map(|t| t.to_dim()).product()),
        ));
        match self {
            // g is on i
            KernelFormat::HWIO => {
                ops.push(AxisOp::Reshape(
                    1,
                    tvec!(self.i(full_shape).to_dim()),
                    tvec!(group.to_dim(), self.i(full_shape).to_dim() / group),
                ));
                ops.push(AxisOp::Move(0, 3));
                ops.push(AxisOp::Move(1, 2));
            }
            // g is on o
            KernelFormat::OIHW => {
                ops.push(AxisOp::Reshape(
                    0,
                    tvec!(self.o(full_shape).to_dim()),
                    tvec!(group.to_dim(), self.o(full_shape).to_dim() / group),
                ));
            }
            // g is on i
            KernelFormat::OHWI => {
                ops.push(AxisOp::Reshape(
                    2,
                    tvec!(self.i(full_shape).to_dim()),
                    tvec!(group.to_dim(), self.i(full_shape).to_dim() / group),
                )); // o hw g i
                ops.push(AxisOp::Move(2, 0)); // g o hw i
                ops.push(AxisOp::Move(2, 3));
            }
        }
        Ok(ops)
    }

    pub fn kernel_as_group_o_i_hw(&self, kernel: &Tensor, group: usize) -> TractResult<Tensor> {
        let mut kernel = kernel.clone();
        let ops = self.kernel_as_group_o_i_hw_ops(&kernel.shape(), group)?;
        for op in &ops {
            op.change_tensor(&mut kernel, false)?;
        }
        Ok(kernel)
    }

    pub fn kernel_as_group_o_ihw(&self, kernel: &Tensor, group: usize) -> TractResult<Tensor> {
        let group_o_i_hw = self.kernel_as_group_o_i_hw(kernel, group)?;
        Ok(group_o_i_hw.collapse_axis_with_next(2))
    }
}
