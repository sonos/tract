use tract_core::ops::math::{add, exp, ln, mul, tanh};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

#[derive(Debug, Clone, Default)]
pub struct Mish;

impl Expansion for Mish {
    fn name(&self) -> StaticName {
        "Mish".into()
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
        // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        let exp_x = model.wire_node(format!("{prefix}.exp"), exp(), &[inputs[0]])?[0];
        let c_one =
            model.add_const(format!("{prefix}.one"), tensor0(1f32).cast_to_dt(dt)?.into_owned())?;
        let one_plus_exp =
            wire_with_rank_broadcast(format!("{prefix}.add_one"), model, add(), &[exp_x, c_one])?
                [0];
        let softplus = model.wire_node(format!("{prefix}.ln"), ln(), &[one_plus_exp])?[0];
        let tanh_sp = model.wire_node(format!("{prefix}.tanh"), tanh(), &[softplus])?[0];
        model.wire_node(prefix, mul(), &[inputs[0], tanh_sp])
    }
}
