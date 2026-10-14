use super::MultiBroadcastTo;
use crate::internal::*;
use tract_itertools::Itertools;
use tract_num_traits::Zero;

/// Cumulative sum along `axis`.
///
/// Output `i` is the sum of inputs `0..=i` (`0..i` when `exclusive`), counted from the
/// end of the axis when `reverse`. Shape and type are the input's; every numeric type is
/// accepted. The axes mapping leaves `axis` disconnected so no rule slices, pulses or
/// moves through it as if positions were independent. Declutter folds a sum over a
/// length-1 axis to the identity, or to zeros when `exclusive`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CumSum {
    pub axis: usize,
    pub exclusive: bool,
    pub reverse: bool,
}

impl CumSum {
    fn eval_t<T: Datum + Copy + std::ops::AddAssign + Zero>(
        &self,
        tensor: &mut Tensor,
    ) -> TractResult<()> {
        let shape = tensor.shape().to_vec();
        let len = shape[self.axis];
        let inner: usize = shape[self.axis + 1..].iter().product();
        if len == 0 || inner == 0 {
            return Ok(());
        }
        let mut plain = tensor.try_as_plain_ram_mut()?;
        let data = plain.as_slice_mut::<T>()?;
        for block in data.chunks_exact_mut(len * inner) {
            if self.reverse {
                for i in (0..len - 1).rev() {
                    let (head, tail) = block.split_at_mut((i + 1) * inner);
                    add_row(&mut head[i * inner..], &tail[..inner]);
                }
                if self.exclusive {
                    block.copy_within(inner.., 0);
                    block[(len - 1) * inner..].fill(T::zero());
                }
            } else {
                for i in 1..len {
                    let (head, tail) = block.split_at_mut(i * inner);
                    add_row(&mut tail[..inner], &head[(i - 1) * inner..]);
                }
                if self.exclusive {
                    block.copy_within(..(len - 1) * inner, inner);
                    block[..inner].fill(T::zero());
                }
            }
        }
        Ok(())
    }
}

fn add_row<T: Copy + std::ops::AddAssign>(acc: &mut [T], row: &[T]) {
    acc.iter_mut().zip(row).for_each(|(a, r)| *a += *r);
}

impl Op for CumSum {
    fn name(&self) -> StaticName {
        "CumSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis: {} exclusive: {} reverse: {}",
            self.axis, self.exclusive, self.reverse
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for CumSum {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut tensor = args_1!(inputs).into_tensor();
        dispatch_numbers!(Self::eval_t(tensor.datum_type())(self, &mut tensor))?;
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for CumSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.axis < inputs[0].rank());
        ensure!(!inputs[0].datum_type.is_quantized(), "CumSum does not take quantized inputs");
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.clone())))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut letters = 'a'..;
        let axes = (0..inputs[0].rank())
            .flat_map(|ix| {
                if ix == self.axis {
                    tvec!(
                        Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                            .input(0, ix),
                        Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                            .output(0, ix),
                    )
                } else {
                    tvec!(
                        Axis::new(letters.next().unwrap(), inputs.len(), outputs.len())
                            .input(0, ix)
                            .output(0, ix)
                    )
                }
            })
            .collect_vec();
        AxesMapping::new(1, 1, axes)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let fact = model.outlet_fact(node.inputs[0])?;
        rule_if!(fact.shape[self.axis].is_one());
        if !self.exclusive {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        // An exclusive sum over a single position is all zeros.
        let zero = Tensor::zero_dt(fact.datum_type, &tvec![1; fact.rank()])?;
        let mut patch = TypedModelPatch::default();
        let zero = patch.add_const(format!("{}.zero", node.name), zero)?;
        let wire =
            patch.wire_node(&node.name, MultiBroadcastTo::new(fact.shape.clone()), &[zero])?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        rule_if_some!(axis = change.transform_axis(self.axis));
        let op = Some(Box::new(Self { axis, ..self.clone() }) as _);
        Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        rule_if!(output_axis != self.axis);
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declutter_single_step(exclusive: bool) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let t = model.symbols.sym("T");
        let input = model.add_source("input", f32::fact(&[1.to_dim(), t.to_dim()]))?;
        let op = CumSum { axis: 0, exclusive, reverse: false };
        let output = model.wire_node("cumsum", op, &[input])?;
        model.select_output_outlets(&output)?;
        model.into_decluttered()
    }

    #[test]
    fn single_step_inclusive_is_identity() -> TractResult<()> {
        let model = declutter_single_step(false)?;
        assert_eq!(model.nodes().len(), 1);
        Ok(())
    }

    #[test]
    fn single_step_exclusive_is_zeros() -> TractResult<()> {
        let model = declutter_single_step(true)?;
        assert!(model.nodes().iter().all(|n| !n.op_is::<CumSum>()));
        let input = tensor2(&[[1f32, 2., 3.]]);
        let output = model.into_runnable()?.run(tvec!(input.into_tvalue()))?;
        output[0].close_enough(&tensor2(&[[0f32, 0., 0.]]), Approximation::Exact)?;
        Ok(())
    }
}
