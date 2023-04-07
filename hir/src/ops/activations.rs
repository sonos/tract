use crate::internal::*;
use tract_core::ops::binary::wire_bin;
use tract_core::ops::logic::{Greater, Less};
use tract_core::ops::math::*;

macro_rules! activation {
    ($op: ident, $wire:expr) => {
        impl Expansion for $op {
            fn name(&self) -> Cow<str> {
                stringify!($op).into()
            }

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

macro_rules! cst {
    ($model: expr, $inputs: expr, $name: expr, $id:ident, $value: expr) => {
        let $id = broadcast_scalar($value, $model, $inputs)?;
        let $id = $model.add_const($name.to_string() + "." + stringify!($id), $id)?;
    };
}

#[derive(Debug, Clone, new)]
pub struct Clip(Option<f32>, Option<f32>);

activation!(Clip, |op, name: &str, model: &mut TypedModel, inputs| {
    let mut wire: TVec<OutletId> = inputs.into();
    if let Some(low) = op.0 {
        let low = broadcast_scalar(low, model, inputs)?;
        let low = model.add_const(name.to_string() + ".low.cst", low)?;
        wire = wire_bin(name.to_string() + ".low", model, Max, &[wire[0], low])?;
    }
    if let Some(high) = op.1 {
        let high = broadcast_scalar(high, model, inputs)?;
        let high = model.add_const(name.to_string() + ".high.cst", high)?;
        wire = wire_bin(name.to_string() + ".high", model, Min, &[wire[0], high])?;
    }
    Ok(wire)
});

#[derive(Debug, Clone, new, Hash)]
pub struct Softplus;

activation!(Softplus, |_op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, one, 1.0);
    let wire = model.wire_node(name.to_string() + ".exp", exp(), inputs)?;
    let wire = wire_bin(name.to_string() + ".plus_one", model, Add, &[wire[0], one])?;
    let wire = model.wire_node(name.to_string() + ".ln", ln(), &wire)?;
    Ok(wire)
});

#[derive(Debug, Clone, new, Hash)]
pub struct Softsign;

activation!(Softsign, |_op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, one, 1.0);
    let x_abs = model.wire_node(name.to_string() + ".abs", abs(), inputs)?;
    let denum = wire_bin(name.to_string() + ".plus_one", model, Add, &[x_abs[0], one])?;
    let wire = wire_bin(name.to_string() + ".div", model, Div, &[inputs[0], denum[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct Elu(pub f32);

activation!(Elu, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, zero, 0.0);
    cst!(model, inputs, name, one, 1.0);
    cst!(model, inputs, name, alpha, op.0);
    let x_exp = model.wire_node(name.to_string() + ".exp", exp(), inputs)?;
    let minus_one = wire_bin(name.to_string() + ".minus_one", model, Sub, &[x_exp[0], one])?;
    let neg = wire_bin(name.to_string() + ".mul_alpha", model, Mul, &[alpha, minus_one[0]])?;
    let test = wire_bin(name.to_string() + ".test", model, Less, &[zero, inputs[0]])?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], neg[0]],
    )?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct HardSigmoid(pub f32, pub f32);

activation!(HardSigmoid, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, zero, 0.0);
    cst!(model, inputs, name, one, 1.0);
    cst!(model, inputs, name, alpha, op.0);
    cst!(model, inputs, name, beta, op.1);
    let wire = wire_bin(name.to_string() + ".mul_alpha", model, Mul, &[alpha, inputs[0]])?;
    let wire = wire_bin(name.to_string() + ".add_beta", model, Add, &[beta, wire[0]])?;
    let wire = wire_bin(name.to_string() + ".sat-one", model, Min, &[one, wire[0]])?;
    let wire = wire_bin(name.to_string() + ".sat-zero", model, Max, &[zero, wire[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct LeakyRelu(pub f32);

activation!(LeakyRelu, |op, name: &str, model: &mut TypedModel, inputs| {
    model.wire_node(name, tract_core::ops::nn::leaky_relu(op.0), inputs)
});

#[derive(Debug, Clone, new)]
pub struct ParametricSoftplus(pub f32, pub f32);

activation!(ParametricSoftplus, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, one, 1.0);
    cst!(model, inputs, name, alpha, op.0);
    cst!(model, inputs, name, beta, op.1);
    let wire = wire_bin(name.to_string() + ".mul_beta", model, Mul, &[beta, inputs[0]])?;
    let wire = model.wire_node(name.to_string() + ".exp", exp(), &wire)?;
    let wire = wire_bin(name.to_string() + ".plus_one", model, Add, &[one, wire[0]])?;
    let wire = model.wire_node(name.to_string() + ".ln", ln(), &wire)?;
    let wire = wire_bin(name.to_string() + ".mul_alpha", model, Mul, &[alpha, wire[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct ScaledTanh(pub f32, pub f32);

activation!(ScaledTanh, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, alpha, op.0);
    cst!(model, inputs, name, beta, op.1);
    let wire = wire_bin(name.to_string() + ".mul_beta", model, Mul, &[beta, inputs[0]])?;
    let wire = model.wire_node(name.to_string() + ".tanh", tanh(), &wire)?;
    let wire = wire_bin(name.to_string() + ".mul_alpha", model, Mul, &[alpha, wire[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct Selu(pub f32, pub f32);

activation!(Selu, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, zero, 0.0);
    cst!(model, inputs, name, alpha, op.0);
    cst!(model, inputs, name, gamma, op.1);
    let wire = model.wire_node(name.to_string() + ".exp", exp(), inputs)?;
    let wire = wire_bin(name.to_string() + ".mul_alpha", model, Mul, &[wire[0], alpha])?;
    let wire = wire_bin(name.to_string() + ".sub_alpha", model, Sub, &[wire[0], alpha])?;
    let test = wire_bin(name.to_string() + ".test", model, Less, &[zero, inputs[0]])?;
    let wire = model.wire_node(
        name.to_string() + ".iff",
        tract_core::ops::logic::Iff,
        &[test[0], inputs[0], wire[0]],
    )?;
    let wire = wire_bin(name.to_string() + ".mul_gamma", model, Mul, &[gamma, wire[0]])?;
    Ok(wire)
});

#[derive(Debug, Clone, new)]
pub struct Shrink(pub f32, pub f32);

activation!(Shrink, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, bias, op.0);
    cst!(model, inputs, name, lambda, op.1);
    cst!(model, inputs, name, minus_lambda, -op.1);
    let zero = broadcast_scalar(0.0, model, inputs)?;
    let zero = model.add_const(name.to_string() + ".zero", zero)?;
    let test_pos = wire_bin(name.to_string() + ".test_pos", model, Less, &[lambda, inputs[0]])?;
    let pos =
        wire_bin(name.to_string() + ".pos", model, tract_core::ops::math::Sub, &[inputs[0], bias])?;
    let test_neg =
        wire_bin(name.to_string() + ".test_neg", model, Greater, &[minus_lambda, inputs[0]])?;
    let neg =
        wire_bin(name.to_string() + ".neg", model, tract_core::ops::math::Add, &[bias, inputs[0]])?;
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

#[derive(Debug, Clone, new)]
pub struct ThresholdRelu(pub f32);

activation!(ThresholdRelu, |op, name: &str, model: &mut TypedModel, inputs| {
    cst!(model, inputs, name, zero, 0.0);
    cst!(model, inputs, name, alpha, op.0);
    let test = wire_bin(name.to_string() + ".test", model, Less, &[alpha, inputs[0]])?;
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
    check_input_arity(inputs, 1)?;
    check_output_arity(outputs, 1)?;
    s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
    s.equals(&inputs[0].shape, &outputs[0].shape)?;
    Ok(())
}

pub fn broadcast_scalar(
    f: f32,
    model: &TypedModel,
    inputs: &[OutletId],
) -> TractResult<Arc<Tensor>> {
    let fact = model.outlet_fact(inputs[0])?;
    let mut tensor = tensor0(f).cast_to_dt(fact.datum_type)?.into_owned();
    while tensor.rank() < fact.rank() {
        tensor.insert_axis(0)?;
    }
    Ok(tensor.into_arc_tensor())
}
