use crate::internal::*;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use tract_linalg::NonLinearSpec;
use tract_linalg::Tile;

#[derive(CustomDebug, Clone, new)]
pub struct Direct {
    tile: Box<dyn Tile<f32>>,
    data_offsets: Vec<isize>,
    kernel_offsets: Vec<isize>,
    input_shape: DataShape,
    output_shape: DataShape,
    packed_filters: Tensor,
    non_linear_fused_op: Vec<NonLinearSpec<f32>>,
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
        for op in &self.non_linear_fused_op {
            info.push(format!(" + {:?}", op));
        }
        Ok(info)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        if let Some(succ) = model.single_succ(node.id)? {
            let fused_micro_op = (|| -> TractResult<Option<NonLinearSpec<f32>>> {
                if let Some(op) = succ.op_as::<crate::ops::math::Mul::UnaryA>() {
                    if op.b.shape() == &[*self.output_shape.c()] {
                        return Ok(Some(NonLinearSpec::PerRowMul(
                            op.b.as_slice::<f32>()?.to_vec(),
                        )));
                    }
                } else if let Some(op) = succ.op_as::<crate::ops::math::Add::UnaryA>() {
                    if op.b.shape() == &[*self.output_shape.c()] {
                        return Ok(Some(NonLinearSpec::PerRowAdd(
                            op.b.as_slice::<f32>()?.to_vec(),
                        )));
                    }
                } else if succ.op_is::<crate::ops::nn::Relu>() {
                    return Ok(Some(NonLinearSpec::Max(0f32)));
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut ops = self.non_linear_fused_op.clone();
                ops.push(op);
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Direct { non_linear_fused_op: ops, ..self.clone() },
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
                    &[],
                );
            }
            Ok(tvec!(output.into_arc_tensor()))
        }
    }
}
