use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn clip(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    match ctx.onnx_operator_set_version {
        6 | 7 | 8 | 9 | 10 => clip_6(ctx, node),
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

#[derive(Debug, Clone, new, Hash)]
pub struct Clip11 {
    input_min: Option<usize>,
    input_max: Option<usize>,
}

tract_data::impl_dyn_hash!(Clip11);

impl Expansion for Clip11 {
    fn name(&self) -> Cow<str> {
        "Clip".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            &inputs,
            1 + self.input_min.is_some() as usize + self.input_max.is_some() as usize,
        )?;
        check_output_arity(&outputs, 1)?;
        if let Some(input) = self.input_min {
            s.equals(&inputs[0].datum_type, &inputs[input].datum_type)?;
            s.equals(&inputs[input].rank, 0)?;
        }
        if let Some(input) = self.input_max {
            s.equals(&inputs[0].datum_type, &inputs[input].datum_type)?;
            s.equals(&inputs[input].rank, 0)?;
        }
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
        let rank = model.outlet_fact(inputs[0])?.rank();
        let mut wire = inputs[0];
        if let Some(min) = self.input_min {
            let mut min = inputs[min];
            for r in 0..rank {
                min =
                    model.wire_node(format!("{}.min.axis-{}", name, r), AxisOp::Add(r), &[min])?[0];
            }
            wire = model.wire_node(
                format!("{}.min", name),
                tract_hir::ops::math::max::bin_typed(),
                &[wire, min],
            )?[0];
        }
        if let Some(max) = self.input_max {
            let mut max = inputs[max];
            for r in 0..rank {
                max =
                    model.wire_node(format!("{}.max.axis-{}", name, r), AxisOp::Add(r), &[max])?[0];
            }
            wire = model.wire_node(
                format!("{}.max", name),
                tract_hir::ops::math::min::bin_typed(),
                &[wire, max],
            )?[0];
        }
        Ok(tvec!(wire))
    }
}
