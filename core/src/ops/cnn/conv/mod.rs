mod block_quant;
#[allow(clippy::module_inception)]
mod conv;
mod depth_wise;
mod im2col;
mod lazy_im2col;
mod q_sum_b;

use tract_linalg::block_quant::BlockQuantFact;

use crate::internal::*;

pub use self::conv::Conv;
pub use self::im2col::Im2Col;
pub(crate) use self::q_sum_b::QSumB;

fn block_quant_aware_weight_shape(weights: &TypedFact) -> TractResult<Cow<ShapeFact>> {
    if weights.datum_type.is_number() {
        Ok(Cow::Borrowed(&weights.shape))
    } else if let Some(bqf) =
        weights.opaque_fact().and_then(|of| of.downcast_ref::<BlockQuantFact>())
    {
        Ok(Cow::Owned(bqf.shape().into()))
    } else {
        todo!("Weights are expected to be numbers of BlockQuant");
    }
}

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

    pub fn kernel_as_group_o_i_h_w_ops(
        &self,
        full_shape: &[impl DimLike],
        group: usize,
    ) -> TVec<AxisOp> {
        let geo_rank = full_shape.len() - 2;
        match self {
            // g is on i
            KernelFormat::HWIO => {
                tvec!(
                    AxisOp::Reshape(
                        geo_rank,
                        tvec!(self.i(full_shape).to_dim()),
                        tvec!(group.to_dim(), self.i(full_shape).to_dim() / group),
                    ), // h w g i o
                    AxisOp::Move(geo_rank, 0),     // g h w i o
                    AxisOp::Move(geo_rank + 2, 1), // g o h w i
                    AxisOp::Move(geo_rank + 2, 2)
                ) // g o i h w
            }
            // g is on o
            KernelFormat::OIHW => {
                tvec!(AxisOp::Reshape(
                    0,
                    tvec!(self.o(full_shape).to_dim()),
                    tvec!(group.to_dim(), self.o(full_shape).to_dim() / group),
                ))
            }
            // g is on i
            KernelFormat::OHWI => {
                tvec!(
                    AxisOp::Reshape(
                        geo_rank + 1,
                        tvec!(self.i(full_shape).to_dim()),
                        tvec!(group.to_dim(), self.i(full_shape).to_dim() / group),
                    ), // o h w g i
                    AxisOp::Move(geo_rank + 1, 0), // g o h w i
                    AxisOp::Move(geo_rank + 2, 2)
                )
            }
        }
    }

    pub fn kernel_as_group_o_i_hw_ops(
        &self,
        full_shape: &[impl DimLike],
        group: usize,
    ) -> TVec<AxisOp> {
        let mut ops = self.kernel_as_group_o_i_h_w_ops(full_shape, group);
        if self.hw(full_shape).len() > 1 {
            ops.push(AxisOp::Reshape(
                3,
                self.hw(full_shape).iter().map(|t| t.to_dim()).collect(),
                tvec!(self.hw(full_shape).iter().map(|t| t.to_dim()).product()),
            ));
        }
        ops
    }

    pub fn kernel_as_group_o_ihw_ops(
        &self,
        full_shape: &[impl DimLike],
        group: usize,
    ) -> TVec<AxisOp> {
        let i = (self.input_channels(full_shape, group).into_owned() / group).to_dim();
        let hw = self.hw(full_shape).iter().map(|t| t.to_dim()).product::<TDim>();
        let mut ops = self.kernel_as_group_o_i_hw_ops(full_shape, group);
        ops.push(AxisOp::Reshape(2, tvec!(i.clone(), hw.clone()), tvec!(i * hw)));
        ops
    }

    pub fn kernel_as_group_o_i_hw(&self, kernel: &Tensor, group: usize) -> TractResult<Tensor> {
        let mut kernel = kernel.clone();
        let ops = self.kernel_as_group_o_i_hw_ops(kernel.shape(), group);
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
