use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn clip(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    match ctx.onnx_operator_set_version {
        6..=10 => clip_6(ctx, node),
        v if v >= 10 => clip_11(ctx, node),
        _ => bail!("Unsupported operator set for Clip operator"),
    }
}

pub fn clip_6(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let min: Option<f32> = node.get_attr_opt("min")?;
    let max: Option<f32> = node.get_attr_opt("max")?;
    Ok((expand(tract_hir::ops::activations::Clip::new(min, max)), vec![]))
}

pub fn clip_11(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    let op = Clip11::new(options.next().unwrap(), options.next().unwrap());
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Clip11 {
    input_min: Option<usize>,
    input_max: Option<usize>,
}

impl Expansion for Clip11 {
    fn name(&self) -> StaticName {
        "Clip".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            1 + self.input_min.is_some() as usize + self.input_max.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        // A bound may be an I64 constant while the clamped value is a symbolic TDim
        // (streaming cache-len clamps); wire() casts each bound to the input dtype.
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        let mut wire = inputs[0];
        if let Some(min) = self.input_min {
            let mut b = inputs[min];
            if model.outlet_fact(b)?.datum_type != dt {
                b = model.wire_node(
                    format!("{name}.min.cast"),
                    tract_core::ops::cast::cast(dt),
                    &[b],
                )?[0];
            }
            wire = wire_with_rank_broadcast(
                format!("{name}.min"),
                model,
                tract_hir::ops::math::max(),
                &[wire, b],
            )?[0];
        }
        if let Some(max) = self.input_max {
            let mut b = inputs[max];
            if model.outlet_fact(b)?.datum_type != dt {
                b = model.wire_node(
                    format!("{name}.max.cast"),
                    tract_core::ops::cast::cast(dt),
                    &[b],
                )?[0];
            }
            wire = wire_with_rank_broadcast(
                format!("{name}.max"),
                model,
                tract_hir::ops::math::min(),
                &[wire, b],
            )?[0];
        }
        Ok(tvec!(wire))
    }
}
