use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn instance_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    Ok((expand(InstanceNorm::new(epsilon)), vec![]))
}

#[derive(Debug, Clone, new, Default, Educe)]
#[educe(Hash)]
pub struct InstanceNorm {
    #[educe(Hash(method = "hash_f32"))]
    epsilon: f32,
}

tract_data::impl_dyn_hash!(InstanceNorm);

impl Expansion for InstanceNorm {
    fn name(&self) -> Cow<str> {
        "InstanceNorm".into()
    }

    op_onnx!();

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
        let rank = model.outlet_fact(inputs[0])?.rank();
        let axes: Vec<_> = (0..rank as i64).filter(|&axis| axis != 1).collect();
        let mean = tract_hir::ops::nn::Reduce::new(
            Some(axes.clone()),
            true,
            tract_hir::ops::nn::Reducer::Mean,
        )
        .wire(&format!("{}.mean", name), model, &inputs[0..1])?[0];
        let diff = model.wire_node(
            format!("{}.diff", name),
            tract_hir::ops::math::sub::bin_typed(),
            &[inputs[0], mean],
        )?;
        let sqr_diff =
            model.wire_node(format!("{}.sqr", name), tract_hir::ops::math::square(), &diff)?;
        let vari = tract_hir::ops::nn::Reduce::new(
            Some(axes.clone()),
            true,
            tract_hir::ops::nn::Reducer::Mean,
        )
        .wire(&format!("{}.variance", name), model, &sqr_diff)?[0];
        let vari_sane = model.wire_node(
            format!("{}.epsilon", name),
            tract_hir::ops::math::add::unary(
                tensor0(self.epsilon).broadcast_into_rank(rank)?.into_arc_tensor(),
            ),
            &[vari],
        )?;
        let div = model.wire_node(
            format!("{}.rsqrt", name),
            tract_hir::ops::math::rsqrt(),
            &vari_sane,
        )?;
        let divised = model.wire_node(
            format!("{}.div", name),
            tract_hir::ops::math::mul::bin_typed(),
            &[diff[0], div[0]],
        )?;
        let mut scale =
            model.wire_node(format!("{}.add-scale-axis-n", name), AxisOp::Add(0), &inputs[1..2])?;
        for i in 2..rank {
            scale = model.wire_node(
                format!("{}.add-scale-axis-{}", name, i),
                AxisOp::Add(2),
                &scale,
            )?;
        }
        let scaled = model.wire_node(
            format!("{}.scaled", name),
            tract_hir::ops::math::mul::bin_typed(),
            &[divised[0], scale[0]],
        )?;
        let mut bias =
            model.wire_node(format!("{}.add-bias-axis-n", name), AxisOp::Add(0), &inputs[2..3])?;
        for i in 2..rank {
            bias = model.wire_node(
                format!("{}.add-bias-axis-{}", name, i),
                AxisOp::Add(2),
                &bias,
            )?;
        }
        model.wire_node(name, tract_hir::ops::math::add::bin_typed(), &[scaled[0], bias[0]])
    }
}
