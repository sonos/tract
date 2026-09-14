use crate::model::{ParsingContext, optional_inputs};
use crate::pb::*;
use tract_hir::internal::*;

pub fn dropout(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut inputs = optional_inputs(node).skip(1);
    let ratio_input = inputs.next().unwrap();
    let training_mode_input = inputs.next().unwrap();
    Ok((Box::new(Dropout::new(node.output.len() == 2, ratio_input, training_mode_input)), vec![]))
}

/// Inference-mode Dropout: the output is the input, and the optional mask is all true.
///
/// From opset 12 the op takes optional `ratio` and `training_mode` inputs. Training mode
/// with a non-zero ratio drops elements at random, which tract does not do, so it is
/// rejected rather than silently run as inference.
#[derive(Debug, Clone, new, Default, Hash, PartialEq, Eq)]
pub struct Dropout {
    output_mask: bool,
    ratio_input: Option<usize>,
    training_mode_input: Option<usize>,
}

impl Dropout {
    /// Whether the node behaves as in inference, given a way to read its input values.
    /// `None` when that depends on a value that is not known.
    fn acts_as_inference(&self, value: impl Fn(usize) -> Option<Tensor>) -> Option<bool> {
        let Some(training_mode) = self.training_mode_input else { return Some(true) };
        if !value(training_mode)?.cast_to_scalar::<bool>().ok()? {
            return Some(true);
        }
        // Training mode drops nothing when the ratio is zero. The default ratio is 0.5.
        let ratio = match self.ratio_input {
            Some(ratio) => value(ratio)?.cast_to_scalar::<f32>().ok()?,
            None => 0.5,
        };
        Some(ratio == 0.0)
    }
}

impl Op for Dropout {
    fn name(&self) -> StaticName {
        "Dropout".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Dropout {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if self.acts_as_inference(|ix| inputs.get(ix).map(|t| t.clone().into_tensor()))
            != Some(true)
        {
            bail!("Dropout in training mode with a non-zero ratio is not supported");
        }
        let input = inputs[0].clone();
        if self.output_mask {
            let mask = tract_ndarray::ArrayD::from_elem(input.shape(), true);
            Ok(tvec!(input, mask.into_tvalue()))
        } else {
            Ok(tvec!(input))
        }
    }
}

impl InferenceRulesOp for Dropout {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            1 + self.ratio_input.is_some() as usize + self.training_mode_input.is_some() as usize,
        )?;
        check_output_arity(outputs, 1 + self.output_mask as usize)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if outputs.len() == 2 {
            s.equals(&outputs[1].datum_type, bool::datum_type())?;
            s.equals(&inputs[0].shape, &outputs[1].shape)?;
        }
        Ok(())
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.output_mask as usize)
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Dropout {
    as_op!();
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts = tvec!(inputs[0].without_value());
        if self.output_mask {
            facts.push(bool::datum_type().fact(inputs[0].shape.clone()));
        }
        Ok(facts)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let known_inference = self.acts_as_inference(|ix| {
            model.outlet_fact(node.inputs[ix]).ok()?.konst.as_ref().map(|k| (**k).clone())
        }) == Some(true);
        if !known_inference {
            return Ok(None);
        }
        let mut patch = TypedModelPatch::default();
        let input = patch.tap_model(model, node.inputs[0])?;
        patch.shunt_outside(model, node.id.into(), input)?;
        if self.output_mask {
            let shape = model.outlet_fact(node.inputs[0])?.shape.clone();
            let keep = patch.add_const(format!("{}.keep", node.name), tensor0(true))?;
            let mask = patch.wire_node(
                format!("{}.mask", node.name),
                tract_core::ops::array::MultiBroadcastTo::new(shape),
                &[keep],
            )?[0];
            patch.shunt_outside(model, OutletId::new(node.id, 1), mask)?;
        }
        Ok(Some(patch))
    }
}
