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

    pub fn input_channels<D: DimLike>(&self, full_shape: &[D], group: usize) -> D {
        match self {
            KernelFormat::OIHW => self.i(full_shape).clone() * group,
            KernelFormat::HWIO | KernelFormat::OHWI => self.i(full_shape).clone(),
        }
    }

    pub fn output_channels<D: DimLike>(&self, full_shape: &[D], group: usize) -> D {
        match self {
            KernelFormat::OIHW => self.o(full_shape).clone(),
            KernelFormat::HWIO | KernelFormat::OHWI => self.o(full_shape).clone() * group,
        }
    }

    pub fn kernel_as_group_o_ihw(
        &self,
        kernel: &Tensor,
        group: usize,
        input_channels: usize,
        output_channels: usize,
    ) -> TractResult<Arc<Tensor>> {
        let final_shape = [group, output_channels / group, kernel.len() / output_channels];
        trace!("kernel shape (group, output, rest) = {:?}", final_shape);
        let hw_rank = kernel.rank() - 2;
        match self {
            KernelFormat::HWIO => {
                // HWGIO
                let tensor = kernel.clone().split_axis(hw_rank, input_channels / group)?;
                // GOIHW
                let mut permutation: Vec<usize> = vec![hw_rank + 1, hw_rank + 2, hw_rank];
                permutation.extend(0..hw_rank);
                let tensor =
                    tensor.permute_axes(&permutation)?.into_shape(&final_shape)?.into_arc_tensor();
                Ok(tensor)
            }
            KernelFormat::OIHW => Ok(kernel.clone().into_shape(&final_shape)?.into_arc_tensor()),
            KernelFormat::OHWI => {
                // move I to OIHW, then same as OIHW
                Ok(kernel
                    .clone()
                    .move_axis(kernel.rank() - 1, 1)?
                    .into_shape(&final_shape)?
                    .into_arc_tensor())
            }
        }
    }
}
