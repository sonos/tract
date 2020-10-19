use crate::internal::*;
use crate::ops;
use ndarray::prelude::*;

mod array;
mod conv;
mod scan;

#[derive(Debug, Clone, new, Default, PartialEq, Hash)]
pub struct Downsample {
    pub axis: usize,
    pub stride: isize,
    pub modulo: usize,
}

impl Downsample {
    pub(crate) fn transform_dim(&self, input_dim: &TDim) -> TDim {
        (input_dim.clone() - self.modulo).div_ceil(self.stride.abs() as _)
    }

    pub(crate) fn transform_fact(&self, input_fact: &TypedFact) -> TractResult<TypedFact> {
        let mut downed = input_fact.clone();
        let down_len = self.transform_dim(&input_fact.shape[self.axis]);
        downed.shape[self.axis] = down_len.clone();
        if let Some(k) = downed.konst {
            let mut outputs = self.eval(tvec!(k))?;
            downed.konst = Some(outputs.remove(0));
        }
        if cfg!(debug_assertions) {
            downed.consistent()?;
        }
        Ok(downed)
    }
}

tract_data::impl_dyn_hash!(Downsample);

impl Op for Downsample {
    fn name(&self) -> Cow<str> {
        "Downsample".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{} stride:{} modulo:{}", self.axis, self.stride, self.modulo)])
    }

    op_core_mir!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Downsample {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let t = if self.modulo > input.shape()[self.axis] {
                let mut shape: TVec<usize> = input.shape().into();
                shape[self.axis] = 0;
                Tensor::uninitialized_dt(input.datum_type(), &*shape)?
            } else {
                let slice = ndarray::Slice::new(self.modulo as isize, None, self.stride);
                unsafe fn do_slice<T: Datum>(
                    t: &Tensor,
                    axis: usize,
                    slice: ndarray::Slice,
                ) -> Tensor {
                    let dt = t.datum_type();
                    let mut t2 = t
                        .to_array_view_unchecked::<T>()
                        .slice_axis(Axis(axis), slice)
                        .into_owned()
                        .into_tensor();
                    t2.set_datum_type(dt);
                    t2
                }
                dispatch_datum_by_size!(do_slice(input.datum_type())(&*input, self.axis, slice))
            };
            Ok(tvec!(t.into_arc_tensor()))
        }
    }
}

impl TypedOp for Downsample {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut downed = inputs[0].clone();
        let down_len = self.transform_dim(&downed.shape[self.axis]);
        downed.shape[self.axis] = down_len.clone();
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

    as_op!();
}

fn pull_downsample_up(
    model: &TypedModel,
    down_node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    #[cfg(debug_assertions)]
    {
        model.check_consistent_facts()?;
    }
    let down_op = down_node.op_as::<Downsample>().unwrap();
    if let Some(prec) = model.single_prec(down_node.id)? {
        let invariants = prec.op.invariants(model, prec)?;
        debug!("Consider pull {:?} over {:?} (invariants: {:?})", down_op, prec, invariants);
        if let Some(crop_op) = prec.op_as::<ops::array::Slice<TDim>>() {
            return array::pull_downsample_over_slice(model, prec, crop_op, down_node, down_op);
        } else if let Some(crop_op) = prec.op_as::<ops::array::Slice<usize>>() {
            return array::pull_downsample_over_slice(model, prec, crop_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<AxisOp>() {
            return array::pull_downsample_over_axis_op(model, prec, other_op, down_node, down_op);
        } else if let Some(conv_op) = prec.op_as::<ops::cnn::conv::ConvUnary>() {
            return conv::fuse_downsample_into_conv(model, prec, conv_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::scan::Scan>() {
            return scan::pull_downsample_over_scan(model, prec, other_op, down_node, down_op);
        } else if let Some(above_axis) = invariants.unary_track_axis_up(down_op.axis, false) {
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
            patch.shunt_outside(model, OutletId::new(down_node.id, 0), other[0])?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
}
