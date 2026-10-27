use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::nn::GroupNorm;
use tract_hir::internal::*;

pub fn instance_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    Ok((expand(InstanceNorm::new(epsilon)), vec![]))
}

#[derive(Debug, Clone, new, Default)]
pub struct InstanceNorm {
    epsilon: f32,
}

impl Expansion for InstanceNorm {
    fn name(&self) -> StaticName {
        "InstanceNorm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[1].shape, &inputs[2].shape)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[1].shape[0], &inputs[0].shape[1])?;
        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_fact = model.outlet_fact(inputs[0])?.clone();
        let rank = input_fact.rank();
        let divised = if let Ok(channels) = input_fact.shape[1].to_usize() {
            // One group per channel: backends run GroupNorm as one fused normalization.
            let op = GroupNorm {
                axis: 1,
                groups: channels,
                reduced_axes: (2..rank).collect(),
                eps: rctensor0(self.epsilon),
            };
            model.wire_node(format!("{name}.group_norm"), op, &inputs[0..1])?
        } else {
            self.wire_decomposed(name, model, inputs[0], &input_fact)?
        };
        let mut scale =
            model.wire_node(format!("{name}.add-scale-axis-n"), AxisOp::Add(0), &inputs[1..2])?;
        for i in 2..rank {
            scale =
                model.wire_node(format!("{name}.add-scale-axis-{i}"), AxisOp::Add(2), &scale)?;
        }
        let scaled = model.wire_node(
            format!("{name}.scaled"),
            tract_hir::ops::math::mul(),
            &[divised[0], scale[0]],
        )?;
        let mut bias =
            model.wire_node(format!("{name}.add-bias-axis-n"), AxisOp::Add(0), &inputs[2..3])?;
        for i in 2..rank {
            bias = model.wire_node(format!("{name}.add-bias-axis-{i}"), AxisOp::Add(2), &bias)?;
        }
        model.wire_node(name, tract_hir::ops::math::add(), &[scaled[0], bias[0]])
    }
}

impl InstanceNorm {
    fn wire_decomposed(
        &self,
        name: &str,
        model: &mut TypedModel,
        input: OutletId,
        input_fact: &TypedFact,
    ) -> TractResult<TVec<OutletId>> {
        let rank = input_fact.rank();
        // Each (batch, channel) plane is normalized on its own, over the spatial axes.
        let axes: Vec<_> = (2..rank as i64).collect();
        let mean = tract_hir::ops::nn::Reduce::new(
            Some(axes.clone()),
            true,
            tract_hir::ops::nn::Reducer::Mean,
        )
        .wire(&format!("{name}.mean"), model, &[input])?[0];
        let diff =
            model.wire_node(format!("{name}.diff"), tract_hir::ops::math::sub(), &[input, mean])?;
        let sqr_diff =
            model.wire_node(format!("{name}.sqr"), tract_hir::ops::math::square(), &diff)?;
        let vari =
            tract_hir::ops::nn::Reduce::new(Some(axes), true, tract_hir::ops::nn::Reducer::Mean)
                .wire(&format!("{name}.variance"), model, &sqr_diff)?[0];
        let epsilon = model.add_const(
            format!("{name}.epsilon.cst"),
            tensor0(self.epsilon)
                .cast_to_dt(input_fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(rank)?
                .into_arc_tensor(),
        )?;
        let vari_sane = model.wire_node(
            format!("{name}.epsilon"),
            tract_hir::ops::math::add(),
            &[vari, epsilon],
        )?;
        let div =
            model.wire_node(format!("{name}.rsqrt"), tract_hir::ops::math::rsqrt(), &vari_sane)?;
        model.wire_node(format!("{name}.div"), tract_hir::ops::math::mul(), &[diff[0], div[0]])
    }
}
