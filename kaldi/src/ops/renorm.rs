use tract_hir::internal::*;
use tract_hir::tract_core::ops::binary::wire_bin;

use crate::model::ParsingContext;

pub fn renorm(ctx: &ParsingContext, name: &str) -> TractResult<Box<dyn InferenceOp>> {
    let component = &ctx.proto_model.components[name];
    let rms = *component
        .attributes
        .get("TargetRms")
        .context("missing attributes TargetRms")?
        .to_scalar::<f32>()?;
    Ok(expand(Renorm::new(rms)))
}

#[derive(Clone, Debug, new)]
struct Renorm {
    target_rms: f32,
}

impl Expansion for Renorm {
    fn name(&self) -> std::borrow::Cow<str> {
        "Renorm".into()
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
        let input = model.outlet_fact(inputs[0])?.clone();
        let sqr =
            model.wire_node(prefix.to_string() + ".sqr", tract_hir::ops::math::square(), inputs)?;
        let sum = model.wire_node(
            prefix.to_string() + ".sum",
            tract_hir::tract_core::ops::nn::Reduce::new(
                tvec![1],
                tract_hir::tract_core::ops::nn::Reducer::Sum,
            ),
            &sqr,
        )?;
        let sqrt =
            model.wire_node(prefix.to_string() + ".sqrt", tract_hir::ops::math::sqrt(), &sum)?;
        let epsilon = tensor0(std::f32::EPSILON).broadcast_into_rank(2)?.into_arc_tensor();
        let epsilon = model.add_const(prefix.to_string() + ".epsilon", epsilon)?;
        let epsilon = wire_bin(
            prefix.to_string() + ".max.epsilon",
            model,
            tract_hir::ops::math::Max,
            &[sqrt[0], epsilon],
        )?;
        let recip = model.wire_node(
            prefix.to_string() + ".recip",
            tract_hir::ops::math::recip(),
            &epsilon,
        )?;
        let rms_sqrt_d = self.target_rms * (input.shape[1].to_isize()? as f32).sqrt();
        let rms_sqrt_d = tensor0(rms_sqrt_d).broadcast_into_rank(2)?.into_arc_tensor();
        let rms_sqrt_d = model.add_const(prefix.to_string() + "rms_sqrt_d", rms_sqrt_d)?;
        let mul = wire_bin(
            prefix.to_string() + ".mul",
            model,
            tract_hir::ops::math::Mul,
            &[rms_sqrt_d, recip[0]],
        )?;
        wire_bin(prefix, model, tract_hir::ops::math::Mul, &[inputs[0], mul[0]])
    }
}
