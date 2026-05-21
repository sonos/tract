use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::math::{add, erf, mul};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn gelu(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let approximate = node.get_attr_opt::<String>("approximate")?.unwrap_or_default();
    if approximate == "tanh" {
        Ok((tract_core::ops::nn::gelu_approximate::gelu_approximate(false).into_hir(), vec![]))
    } else {
        Ok((expand(GeluExact), vec![]))
    }
}

#[derive(Debug, Clone, Default)]
struct GeluExact;

impl Expansion for GeluExact {
    fn name(&self) -> StaticName {
        "GeluExact".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = model.outlet_fact(inputs[0])?.datum_type;
        // gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        let inv_sqrt2 = tensor0((2.0f32).sqrt().recip()).cast_to_dt(dt)?.into_owned();
        let c_inv_sqrt2 = model.add_const(format!("{prefix}.inv_sqrt2"), inv_sqrt2)?;
        let x_scaled = wire_with_rank_broadcast(
            format!("{prefix}.scale"),
            model,
            mul(),
            &[inputs[0], c_inv_sqrt2],
        )?[0];
        let erf_x = model.wire_node(format!("{prefix}.erf"), erf(), &[x_scaled])?[0];
        let c_one =
            model.add_const(format!("{prefix}.one"), tensor0(1f32).cast_to_dt(dt)?.into_owned())?;
        let one_plus_erf =
            wire_with_rank_broadcast(format!("{prefix}.add_one"), model, add(), &[erf_x, c_one])?
                [0];
        let c_half = model
            .add_const(format!("{prefix}.half"), tensor0(0.5f32).cast_to_dt(dt)?.into_owned())?;
        let half_x = wire_with_rank_broadcast(
            format!("{prefix}.half_x"),
            model,
            mul(),
            &[inputs[0], c_half],
        )?[0];
        wire_with_rank_broadcast(format!("{prefix}.out"), model, mul(), &[half_x, one_plus_erf])
    }
}
