use crate::internal::*;
use crate::ops;
use ndarray::prelude::*;

use super::identity::Identity;

mod array;
mod conv;
mod scan;

#[derive(Debug, Clone, new, Default, PartialEq, Eq, Hash)]
pub struct Downsample {
    pub axis: usize,
    pub stride: isize,
    pub modulo: usize,
}

impl Downsample {
    pub(crate) fn transform_dim(&self, input_dim: &TDim) -> TDim {
        (input_dim.clone() - self.modulo).div_ceil(self.stride.unsigned_abs() as u64)
    }

    pub(crate) fn transform_fact(&self, input_fact: &TypedFact) -> TractResult<TypedFact> {
        let mut downed = input_fact.clone();
        let down_len = self.transform_dim(&input_fact.shape[self.axis]);
        downed.shape.set(self.axis, down_len);
        if let Some(k) = downed.konst {
            let mut outputs = self.eval(tvec!(k.into_tvalue()))?;
            downed.konst = Some(outputs.remove(0).into_arc_tensor())
        }
        if cfg!(debug_assertions) {
            downed.consistent()?;
        }
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

impl EvalOp for Downsample {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        unsafe {
            let t = if self.modulo > input.shape()[self.axis] {
                let mut shape: TVec<usize> = input.shape().into();
                shape[self.axis] = 0;
                Tensor::uninitialized_dt(input.datum_type(), &shape)?
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
            Ok(tvec!(t.into_tvalue()))
        }
    }
}

impl TypedOp for Downsample {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axis < inputs[0].rank());
        ensure!(
            self.modulo == 0 || self.stride > 0,
            "non-zero modulo is only defined with forward strides"
        );
        let mut downed = inputs[0].without_value();
        let down_len = self.transform_dim(&downed.shape[self.axis]);
        downed.shape.set(self.axis, down_len);
        Ok(tvec!(downed))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.stride == 1 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Identity,
            )?));
        }
        pull_downsample_up(model, node)
            .with_context(|| format!("Pulling {} over {}", node, model.node(node.inputs[0].node)))
    }

    as_op!();
}

fn pull_downsample_up(
    model: &TypedModel,
    down_node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    model.check_consistency()?;
    let down_op = down_node.op_as::<Downsample>().unwrap();
    if let Some(prec) = model.single_prec(down_node.id)? {
        let (input_facts, output_facts) = model.node_facts(prec.id)?;
        let axes_mapping = prec.op.axes_mapping(&input_facts, &output_facts)?;
        debug!("Consider pull {down_op:?} over {prec:?} (invariants: {axes_mapping:?})");
        if let Some(slice_op) = prec.op_as::<ops::array::Slice>() {
            if let Some(p) =
                array::pull_downsample_over_slice(model, prec, slice_op, down_node, down_op)?
            {
                return Ok(Some(p));
            }
        } else if let Some(other_op) = prec.op_as::<AxisOp>() {
            return array::pull_downsample_over_axis_op(model, prec, other_op, down_node, down_op);
        } else if let Some(conv_op) = prec.op_as::<ops::cnn::conv::Conv>() {
            return conv::fuse_downsample_into_conv(model, prec, conv_op, down_node, down_op);
        } else if let Some(other_op) = prec.op_as::<ops::scan::Scan>() {
            return scan::pull_downsample_over_scan(model, prec, other_op, down_node, down_op);
        }
        if prec.outputs.len() > 1 || prec.inputs.len() == 0 {
            return Ok(None);
        }
        let axis_info = axes_mapping.axis((InOut::Out(0), down_op.axis))?;
        let mut patch = TypedModelPatch::default();
        let mut inputs = vec![];
        for (ix, (outlet, axis_info)) in prec.inputs.iter().zip(&axis_info.inputs).enumerate() {
            let mut wire = patch.tap_model(model, *outlet)?;
            if let &[axis] = &**axis_info {
                if !patch.outlet_fact(wire)?.shape[axis].is_one() {
                    let mut op = down_op.clone();
                    op.axis = axis;
                    wire = patch.wire_node(
                        format!("{}.{}-{}", down_node.name, prec.name, ix),
                        op,
                        &[wire],
                    )?[0];
                }
            } else {
                return Ok(None);
            }
            inputs.push(wire);
        }
        let other = patch.wire_node(&prec.name, prec.op.clone(), &inputs)?;
        patch.shunt_outside(model, OutletId::new(down_node.id, 0), other[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}
