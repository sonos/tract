use crate::internal::*;
use tract_core::ops::math::*;

macro_rules! activation {
    ($op: ident, $wire:expr) => {
        tract_data::impl_dyn_hash!($op);

        impl Expansion for $op {
            fn name(&self) -> Cow<str> {
                stringify!($op).into()
            }

            op_hir!();

            fn rules<'r, 'p: 'r, 's: 'r>(
                &'s self,
                s: &mut Solver<'r>,
                inputs: &'p [TensorProxy],
                outputs: &'p [TensorProxy],
            ) -> InferenceResult {
                simple_unary_rules(s, inputs, outputs)
            }

            fn wire(
                &self,
                name: &str,
                model: &mut TypedModel,
                inputs: &[OutletId],
            ) -> TractResult<TVec<OutletId>> {
                let wire: fn(
                    &$op,
                    &str,
                    &mut TypedModel,
                    &[OutletId],
                ) -> TractResult<TVec<OutletId>> = $wire;
                (wire)(self, name, model, inputs)
            }
        }
    };
}

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Clip(
    #[educe(Hash(method = "hash_opt_f32"))] Option<f32>,
    #[educe(Hash(method = "hash_opt_f32"))] Option<f32>,
);

activation!(Clip, |op, name: &str, model: &mut TypedModel, inputs| {
    let mut wire: TVec<OutletId> = inputs.into();
    if let Some(low) = op.0 {
        let low = broadcast_scalar(low, model, inputs)?;
        wire = model.wire_node(name.to_string() + ".low", max::unary(low), &wire)?;
    }
    if let Some(high) = op.1 {
        let high = broadcast_scalar(high, model, inputs)?;
        wire = model.wire_node(name.to_string() + ".high", min::unary(high), &wire)?;
    }
    Ok(wire)
});

#[derive(Debug, Clone, new, Hash)]
pub struct Softplus;

activation!(Softplus, |_op, name: &str, model: &mut TypedModel, inputs| {
    let one = broadcast_scalar(1.0, model, inputs)?;
    let wire = model.wire_node(name.to_string() + ".exp", exp(), inputs)?;
    let wire = model.wire_node(name.to_string() + ".plus_one", add::unary(one), &wire)?;
    let wire = model.wire_node(name.to_string() + ".ln", ln(), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Hash)]
pub struct Softsign;

activation!(Softsign, |_op, name: &str, model: &mut TypedModel, inputs| {
    let one = broadcast_scalar(1.0, model, inputs)?;
    let x_abs = model.wire_node(name.to_string() + ".abs", abs(), inputs)?;
    let denum = model.wire_node(name.to_string() + ".plus_one", add::unary(one), &x_abs)?;
    let wire =
        model.wire_node(name.to_string() + ".div", div::bin_typed(), &[inputs[0], denum[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Elu(#[educe(Hash(method = "hash_f32"))] pub f32);

activation!(Elu, |op, name: &str, model: &mut TypedModel, inputs| {
    let zero = broadcast_scalar(0.0, model, inputs)?;
    let minus_one = broadcast_scalar(-1.0, model, inputs)?;
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let x_exp = model.wire_node(name.to_string() + ".exp", exp(), inputs)?;
    let minus_one =
        model.wire_node(name.to_string() + ".minus_one", add::unary(minus_one), &x_exp)?;
    let neg = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), &minus_one)?;
    let test = model.wire_node(
        name.to_string() + ".test",
        tract_core::ops::logic::lesser::unary(zero),
        &[inputs[0]],
    )?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], neg[0]],
    )?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct HardSigmoid(
    #[educe(Hash(method = "hash_f32"))] pub f32,
    #[educe(Hash(method = "hash_f32"))] pub f32,
);

activation!(HardSigmoid, |op, name: &str, model: &mut TypedModel, inputs| {
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let beta = broadcast_scalar(op.1, model, inputs)?;
    let one = broadcast_scalar(1.0, model, inputs)?;
    let zero = broadcast_scalar(0.0, model, inputs)?;
    let wire = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), inputs)?;
    let wire = model.wire_node(name.to_string() + ".add_beta", add::unary(beta), &wire)?;
    let wire = model.wire_node(name.to_string() + ".sat-one", min::unary(one), &wire)?;
    let wire = model.wire_node(name.to_string() + ".sat-zero", max::unary(zero), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct LeakyRelu(#[educe(Hash(method = "hash_f32"))] pub f32);

activation!(LeakyRelu, |op, name: &str, model: &mut TypedModel, inputs| {
    let zero = broadcast_scalar(0.0, model, inputs)?;
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let neg = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), &inputs)?;
    let test = model.wire_node(
        name.to_string() + ".test",
        tract_core::ops::logic::lesser::unary(zero),
        &[inputs[0]],
    )?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], neg[0]],
    )?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct ParametricSoftplus(
    #[educe(Hash(method = "hash_f32"))] pub f32,
    #[educe(Hash(method = "hash_f32"))] pub f32,
);

activation!(ParametricSoftplus, |op, name: &str, model: &mut TypedModel, inputs| {
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let beta = broadcast_scalar(op.1, model, inputs)?;
    let one = broadcast_scalar(1.0, model, inputs)?;
    let wire = model.wire_node(name.to_string() + ".mul_beta", mul::unary(beta), inputs)?;
    let wire = model.wire_node(name.to_string() + ".exp", exp(), &wire)?;
    let wire = model.wire_node(name.to_string() + ".plus_one", add::unary(one), &wire)?;
    let wire = model.wire_node(name.to_string() + ".ln", ln(), &wire)?;
    let wire = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct ScaledTanh(
    #[educe(Hash(method = "hash_f32"))] pub f32,
    #[educe(Hash(method = "hash_f32"))] pub f32,
);

activation!(ScaledTanh, |op, name: &str, model: &mut TypedModel, inputs| {
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let beta = broadcast_scalar(op.1, model, inputs)?;
    let wire = model.wire_node(name.to_string() + ".mul_beta", mul::unary(beta), inputs)?;
    let wire = model.wire_node(name.to_string() + ".tanh", tanh(), &wire)?;
    let wire = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Selu(
    #[educe(Hash(method = "hash_f32"))] pub f32,
    #[educe(Hash(method = "hash_f32"))] pub f32,
);

activation!(Selu, |op, name: &str, model: &mut TypedModel, inputs| {
    let zero = broadcast_scalar(0.0, model, inputs)?;
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let minus_alpha = broadcast_scalar(-op.0, model, inputs)?;
    let gamma = broadcast_scalar(op.1, model, inputs)?;
    let wire = model.wire_node(name.to_string() + ".exp", exp(), &inputs)?;
    let wire = model.wire_node(name.to_string() + ".mul_alpha", mul::unary(alpha), &wire)?;
    let wire = model.wire_node(name.to_string() + ".sub_alpha", add::unary(minus_alpha), &wire)?;
    let test = model.wire_node(
        name.to_string() + ".test",
        tract_core::ops::logic::lesser::unary(zero),
        &[inputs[0]],
    )?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], wire[0]],
    )?;
    let wire = model.wire_node(name.to_string() + ".mul_gamma", mul::unary(gamma), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct Shrink(
    #[educe(Hash(method = "hash_f32"))] pub f32,
    #[educe(Hash(method = "hash_f32"))] pub f32,
);

activation!(Shrink, |op, name: &str, model: &mut TypedModel, inputs| {
    let bias = broadcast_scalar(op.0, model, inputs)?;
    let lambda = broadcast_scalar(op.1, model, inputs)?;
    let minus_bias = broadcast_scalar(-op.0, model, inputs)?;
    let minus_lambda = broadcast_scalar(-op.1, model, inputs)?;
    let zero =
        model.add_const(name.to_string() + ".zero", broadcast_scalar(0.0, model, inputs)?)?;
    let test_pos = model.wire_node(
        name.to_string() + ".test_pos",
        tract_core::ops::logic::lesser::unary(lambda),
        &inputs,
    )?;
    let pos = model.wire_node(
        name.to_string() + ".pos",
        tract_core::ops::math::add::unary(minus_bias),
        &inputs,
    )?;
    let test_neg = model.wire_node(
        name.to_string() + ".test_neg",
        tract_core::ops::logic::greater::unary(minus_lambda),
        &inputs,
    )?;
    let neg = model.wire_node(
        name.to_string() + ".neg",
        tract_core::ops::math::add::unary(bias),
        &inputs,
    )?;
    let wire = model.wire_node(
        name.to_string() + ".if_pos",
        tract_core::ops::logic::Iff,
        &[test_pos[0], pos[0], zero],
    )?;
    let wire = model.wire_node(
        name.to_string() + ".if_neg",
        tract_core::ops::logic::Iff,
        &[test_neg[0], neg[0], wire[0]],
    )?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Educe)]
#[educe(Hash)]
pub struct ThresholdRelu(#[educe(Hash(method = "hash_f32"))] pub f32);

activation!(ThresholdRelu, |op, name: &str, model: &mut TypedModel, inputs| {
    let zero =
        model.add_const(name.to_string() + ".zero", broadcast_scalar(0.0, model, inputs)?)?;
    let alpha = broadcast_scalar(op.0, model, inputs)?;
    let test = model.wire_node(
        name.to_string() + ".test",
        tract_core::ops::logic::lesser::unary(alpha),
        &[inputs[0]],
    )?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], zero],
    )?;
    Ok(wire)
});

fn simple_unary_rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_input_arity(&inputs, 1)?;
    check_output_arity(&outputs, 1)?;
    s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
    s.equals(&inputs[0].shape, &outputs[0].shape)?;
    Ok(())
}

pub fn broadcast_scalar(f: f32, model: &TypedModel, inputs: &[OutletId]) -> TractResult<Arc<Tensor>> {
    let fact = model.outlet_fact(inputs[0])?;
    let mut tensor = tensor0(f).cast_to_dt(fact.datum_type)?.into_owned();
    while tensor.rank() < fact.rank() {
        tensor.insert_axis(0)?;
    }
    Ok(tensor.into_arc_tensor())
}
