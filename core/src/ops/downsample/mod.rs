use crate::internal::*;
use crate::ops;
use ndarray::prelude::*;

mod array;
mod conv;
mod scan;

#[derive(Debug, Clone, new, Default, PartialEq)]
pub struct Downsample {
    axis: usize,
    stride: usize,
    modulo: usize,
}

impl Downsample {
    fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Arc<Tensor>> {
        let input = input.to_array_view::<T>()?;
        let sampled = if self.modulo < input.shape()[self.axis] {
            input
                .slice_axis(
                    Axis(self.axis),
                    ndarray::Slice::new(self.modulo as isize, None, self.stride as isize),
                )
                .to_owned()
                .into_arc_tensor()
        } else {
            let mut shape = input.shape().to_vec();
            shape[self.axis] = 0;
            unsafe { Tensor::uninitialized::<T>(&shape)?.into_arc_tensor() }
        };
        Ok(sampled)
    }

    pub(crate) fn transform_dim(&self, input_dim: &TDim) -> TDim {
        (input_dim.clone() - self.modulo).div_ceil(self.stride.into())
    }

    pub(crate) fn transform_fact(
        &self,
        input_fact: &TypedTensorInfo,
    ) -> TractResult<TypedTensorInfo> {
        let mut downed = input_fact.clone();
        let down_len = self.transform_dim(&input_fact.shape.dim(self.axis));
        downed.shape.set_dim(self.axis, down_len.clone())?;
        Ok(downed)
    }
}

impl Op for Downsample {
    fn name(&self) -> Cow<str> {
        "Downsample".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{} stride:{} modulo:{}", self.axis, self.stride, self.modulo)])
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl StatelessOp for Downsample {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, &*input))?))
    }
}

impl InferenceRulesOp for Downsample {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].rank, move |s, r| {
            for i in 0..(r as usize) {
                if i == self.axis {
                    s.given(&inputs[0].shape[i], move |s, d| {
                        s.equals(
                            &outputs[0].shape[i],
                            (d - self.modulo).div_ceil(self.stride.to_dim()),
                        )
                    })?
                } else {
                    s.equals(&inputs[0].shape[i], &outputs[0].shape[i])?
                }
            }
            Ok(())
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Downsample {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let mut downed = inputs[0].clone();
        let down_len = self.transform_dim(&downed.shape.dim(self.axis));
        downed.shape.set_dim(self.axis, down_len.clone())?;
        Ok(tvec!(downed))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.stride == 1 {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        }
        pull_downsample_up(model, node)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        if fact.pulse() % self.stride != 0 {
            bail!("Pulsificaton requires pulse to be a stride multiple")
        }
        fact.shape[self.axis] /= self.stride;
        fact.dim = fact.dim.div_ceil(self.stride.to_dim());
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

fn pull_downsample_up(
    model: &TypedModel,
    down_node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let down_op = down_node.op_as::<Downsample>().unwrap();
    if let Some(prec) = model.single_prec(down_node.id)? {
        let invariants = prec.op.axes_info(model, prec)?;
        debug!("Consider pull {:?} over {:?} (invariants: {:?})", down_op, prec, invariants);
        if let Some(above_axis) = invariants.unary_track_axis_up(down_op.axis, true) {
            let mut patch = TypedModelPatch::default();
            let mut inputs = vec![];
            for (ix, &oo) in prec.inputs.iter().enumerate() {
                let source = patch.tap_model(model, oo)?;
                let mut op = down_op.clone();
                op.axis = above_axis;
                let ds = patch.wire_node(format!("{}-{}", prec.name, ix), op, [source].as_ref())?;
                inputs.push(ds[0]);
            }
            let other = patch.wire_node(&*prec.name, prec.op.clone(), &*inputs)?;
            patch.shunt_outside(OutletId::new(down_node.id, 0), other[0])?;
            return Ok(Some(patch));
        } else if let Some(crop_op) = prec.op_as::<ops::array::Slice<TDim>>() {
            return array::pull_downsample_over_slice(model, prec, crop_op, down_node, down_op);
        } else if let Some(crop_op) = prec.op_as::<ops::array::Slice<usize>>() {
            return array::pull_downsample_over_slice(model, prec, crop_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::array::RmDims>() {
            return array::pull_downsample_over_rmdims(model, prec, other_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::array::AddDims>() {
            return array::pull_downsample_over_adddims(model, prec, other_op, down_node, down_op);
        } else if let Some(conv_op) = prec.op_as::<ops::cnn::conv::ConvUnary>() {
            return conv::fuse_downsample_into_conv(model, prec, conv_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::scan::Typed>() {
            return scan::pull_downsample_over_scan(model, prec, other_op, down_node, down_op);
        }
    }
    Ok(None)
}
