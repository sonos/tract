use tract_data::itertools::Itertools;
use tract_num_traits::Zero;

use crate::internal::*;

use super::Slice;

#[derive(new, Debug, Clone, Hash)]
pub struct TypedConcat {
    pub axis: usize,
}

impl TypedConcat {
    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        for slice in inputs {
            let len = slice.shape[self.axis].clone();
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl Op for TypedConcat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl TypedOp for TypedConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        for input in inputs {
            if input.rank() != fact.rank()
                || input
                    .shape
                    .iter()
                    .zip(fact.shape.iter())
                    .enumerate()
                    .filter(|(ax, _)| *ax != self.axis)
                    .any(|(_, (i, f))| i != f)
            {
                bail!("Inconsistent concat {:?} inputs: {:?}", self, inputs);
            }
        }
        fact.shape.set(self.axis, self.offsets(inputs)?.pop().unwrap());
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes = AxesMapping::disconnected(inputs, outputs)?;
        for ax in 0..outputs[0].rank() {
            if ax != self.axis {
                for i in 0..inputs.len() {
                    axes = axes.linking((InOut::Out(0), ax), (InOut::In(i), ax))?;
                }
            }
        }
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axis =
            if let Some(axis) = change.transform_axis(self.axis) { axis } else { return Ok(None) };
        let op = TypedConcat { axis };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.inputs.len() == 1 {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        let inputs = model.node_input_facts(node.id)?;
        if let Some(pos) = inputs.iter().position(|f| f.shape.volume().is_zero()) {
            let mut inputs = node.inputs.clone();
            inputs.remove(pos);
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &inputs,
                self.clone(),
            )?));
        }
        Ok(None)
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        _node: &TypedNode,
        prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        start: &TDim,
        end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        if output_axis != self.axis {
            return Ok(Some(patch.wire_node(prefix, self.clone(), inputs)?));
        }
        let facts =
            inputs.iter().map(|o| patch.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
        let offsets = self.offsets(&facts)?;
        std::mem::drop(facts);
        for (ix, (slice_start, slice_end)) in offsets.iter().tuple_windows().enumerate() {
            if (start.clone() - slice_start).prove_positive_or_zero()
                && (slice_end.clone() - end).prove_positive_or_zero()
            {
                return patch
                    .wire_node(
                        format!("{prefix}.slice-{output_axis}.{start}..{end}"),
                        Slice {
                            axis: output_axis,
                            start: (start.clone() - slice_start),
                            end: (end.clone() - slice_start),
                        },
                        &[inputs[ix]],
                    )
                    .map(Some);
            }
        }
        Ok(None)
    }
}

impl EvalOp for TypedConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let result = Tensor::stack_tensors(self.axis, &inputs)?;
        Ok(tvec![result.into_tvalue()])
    }
}
