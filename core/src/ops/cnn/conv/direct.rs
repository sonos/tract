use crate::internal::*;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use tract_linalg::mmm::*;

#[derive(Debug, Clone, new)]
pub struct Direct {
    tile: Box<dyn MatMatMul<f32>>,
    data_offsets: Vec<isize>,
    kernel_offsets: Vec<isize>,
    input_shape: DataShape,
    output_shape: DataShape,
    packed_filters: Tensor,
    fused_ops: Vec<FusedSpec<f32>>,
}

impl Direct {
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape.shape
    }
}

impl Op for Direct {
    fn name(&self) -> Cow<str> {
        "ConvDirect".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = vec![format!("{:?}", self.tile)];
        for op in &self.fused_ops {
            info.push(format!(" + {:?}", op));
        }
        Ok(info)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        use crate::num_traits::AsPrimitive;
        if let Some(succ) = model.single_succ(node.id)? {
            let fused_micro_op = (|| -> TractResult<Option<TVec<FusedSpec<f32>>>> {
                if let Some(op) = succ.op_as::<crate::ops::binary::UnaryOp>() {
                    if op.a.shape() == &[*self.output_shape.c()] {
                        if op.mini_op.is::<crate::ops::math::Mul>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowMul(
                                op.a.as_slice::<f32>()?.to_vec(),
                            ))));
                        } else if op.mini_op.is::<crate::ops::math::Add>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowAdd(
                                op.a.as_slice::<f32>()?.to_vec(),
                            ))));
                        }
                    }
                } else if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>() {
                    if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMax>() {
                        return Ok(Some(tvec!(FusedSpec::Max(op.max.as_()))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMin>() {
                        return Ok(Some(tvec!(FusedSpec::Min(op.min.as_()))));
                    } else if let Some(op) = op.0.downcast_ref::<ops::math::ScalarMinMax>() {
                        return Ok(Some(tvec!(
                            FusedSpec::Min(op.min.as_()),
                            FusedSpec::Max(op.max.as_()),
                        )));
                    }
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut ops = self.fused_ops.clone();
                ops.extend(op.into_iter());
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    node,
                    Direct { fused_ops: ops, ..self.clone() },
                )?));
            }
        }
        Ok(None)
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            batch * self.tile.n() * self.tile.m() * self.tile.k()
        )))
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl StatelessOp for Direct {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<f32>()?;
            let mut output = ArrayD::<f32>::uninitialized(&*self.output_shape.shape);
            let filters = self.packed_filters.as_ptr::<f32>()?;
            for n in 0..*self.input_shape.n() {
                let input = input.slice_axis(Axis(0), (n..=n).into());
                let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                self.tile.run(
                    &self.tile.a_from_packed(filters),
                    &self.tile.b_from_data_and_offsets(
                        input.as_ptr(),
                        &self.kernel_offsets,
                        &self.data_offsets,
                    ),
                    &mut self.tile.c_from_data_and_strides(
                        output.as_mut_ptr(),
                        *self.output_shape.c_stride() as isize,
                        *self.output_shape.w_stride() as isize,
                    ),
                    &*self.fused_ops,
                );
            }
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}

impl TypedOp for Direct {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*self.output_shape.shape)?))
    }
}
