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

    pub fn input_channels<'s, D: DimLike>(&self, full_kernel_shape: &'s [D], group: usize) -> Cow<'s, D> {
        match self {
            KernelFormat::OIHW => Cow::Owned(self.i(full_kernel_shape).clone() * group),
            KernelFormat::HWIO | KernelFormat::OHWI => Cow::Borrowed(self.i(full_kernel_shape)),
        }
    }

    pub fn output_channels<'s, D: DimLike>(&self, full_kernel_shape: &'s [D], group: usize) -> Cow<'s, D> {
        match self {
            KernelFormat::OIHW => Cow::Borrowed(self.o(full_kernel_shape)),
            KernelFormat::HWIO | KernelFormat::OHWI => Cow::Owned(self.o(full_kernel_shape).clone() * group),
        }
    }

    pub fn kernel_as_group_o_i_hw(&self, kernel: &Tensor, group: usize) -> TractResult<Tensor> {
        let input_channels = self.input_channels(kernel.shape(), group);
        let output_channels = self.output_channels(kernel.shape(), group);
        let shape_g_o_i_hw = [
            group,
            output_channels.into_owned() / group,
            input_channels.into_owned() / group,
            self.hw(kernel.shape()).iter().product(),
        ];
        trace!("kernel shape (group, output, rest) = {:?}", shape_g_o_i_hw);
        let hw_rank = kernel.rank() - 2;
        match self {
            KernelFormat::HWIO => {
                let mut hw_gi_o = kernel.clone();
                for _ in 0..hw_rank - 1 {
                    hw_gi_o = hw_gi_o.collapse_axis_with_next(0);
                }
                let hw_g_i_o = hw_gi_o.split_axis(1, group)?;
                let g_o_i_hw = hw_g_i_o.move_axis(0, 3)?.move_axis(1, 2)?;
                Ok(g_o_i_hw)
            }
            KernelFormat::OIHW => Ok(kernel.clone().into_shape(&shape_g_o_i_hw)?),
            KernelFormat::OHWI => {
                // move I to OIHW, then same as OIHW
                Ok(kernel.clone().move_axis(kernel.rank() - 1, 1)?.into_shape(&shape_g_o_i_hw)?)
            }
        }
    }

    pub fn kernel_as_group_o_ihw(&self, kernel: &Tensor, group: usize) -> TractResult<Tensor> {
        let group_o_i_hw = self.kernel_as_group_o_i_hw(kernel, group)?;
        Ok(group_o_i_hw.collapse_axis_with_next(2))
    }
}
