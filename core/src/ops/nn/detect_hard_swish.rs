use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::math::{Add, Max, Min, Mul};

/// `x * clamp(x / 6 + 1/2, 0, 1)`, within f16 rounding of the constants.
const ALPHA: f32 = 1.0 / 6.0;
const BETA: f32 = 0.5;
const TOLERANCE: f32 = 2.0e-4;

/// Folds `x * HardSigmoid(x)` when that sigmoid is the h-swish gate
/// (`alpha ≈ 1/6`, `beta ≈ 1/2`). ONNX `HardSwish` is already one op. RVM and
/// PP-HumanSeg spell the same function as `HardSigmoid` then `Mul`, which
/// arrives here already expanded to `mul`, `add`, `min`, `max`.
pub fn detect_hardswish(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let dt = model.node_output_facts(node.id)?[0].datum_type;
    if !matches!(dt, DatumType::F32 | DatumType::F16) {
        return Ok(None);
    }
    let Some(x) = hardswish_input(model, node)? else { return Ok(None) };
    let mut patch = TypedModelPatch::default();
    let tap = patch.tap_model(model, x)?;
    let out = patch.wire_node(format!("{}.hardswish", node.name), super::hard_swish(), &[tap])?[0];
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}

/// The other input, when this `Mul` is `x * hard_sigmoid_sixth(x)`.
fn hardswish_input(model: &TypedModel, node: &TypedNode) -> TractResult<Option<OutletId>> {
    let [a, b] = node.inputs.as_slice() else { return Ok(None) };
    if let Some(x) = hard_sigmoid_sixth(model, *a)?
        && x == *b
    {
        return Ok(Some(x));
    }
    if let Some(x) = hard_sigmoid_sixth(model, *b)?
        && x == *a
    {
        return Ok(Some(x));
    }
    Ok(None)
}

/// `clamp(x * alpha + beta, 0, 1)` with h-swish's constants. The value is the
/// non-constant input of that `max`.
fn hard_sigmoid_sixth(model: &TypedModel, outlet: OutletId) -> TractResult<Option<OutletId>> {
    let Some(saturated) = scalar_gate::<Max>(model, outlet, 0.0)? else { return Ok(None) };
    let Some(sum) = scalar_gate::<Min>(model, saturated, 1.0)? else { return Ok(None) };
    let Some(scaled) = scalar_gate::<Add>(model, sum, BETA)? else { return Ok(None) };
    scalar_gate::<Mul>(model, scaled, ALPHA)
}

/// When `outlet` is `op(value, scalar)` and `scalar` is `expected`, the value.
fn scalar_gate<B: crate::ops::binary::BinMiniOp>(
    model: &TypedModel,
    outlet: OutletId,
    expected: f32,
) -> TractResult<Option<OutletId>> {
    if outlet.slot != 0 {
        return Ok(None);
    }
    let node = model.node(outlet.node);
    if !node.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<B>()) {
        return Ok(None);
    }
    let facts = model.node_input_facts(node.id)?;
    for (slot, fact) in facts.iter().enumerate() {
        let Some(tensor) = fact.uniform.as_ref().or(fact.konst.as_ref()) else { continue };
        if tensor.len() != 1 {
            continue;
        }
        let Ok(value) = tensor.cast_to_scalar::<f32>() else { continue };
        if (value - expected).abs() > TOLERANCE {
            continue;
        }
        return Ok(Some(node.inputs[1 - slot]));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{ALPHA, BETA};
    use crate::internal::*;
    use crate::ops::binary::TypedBinOp;
    use crate::ops::element_wise::ElementWiseOp;
    use crate::ops::math::{Mul, add, max, min, mul};
    use crate::ops::nn::HardSwish;

    fn scalar(
        model: &mut TypedModel,
        name: &str,
        dt: DatumType,
        value: f32,
    ) -> TractResult<OutletId> {
        let tensor = tensor0(value).cast_to_dt(dt)?.into_owned().broadcast_into_rank(4)?;
        model.add_const(name, tensor)
    }

    fn fact(dt: DatumType) -> TypedFact {
        if dt == DatumType::F16 { f16::fact([1, 1, 1, 8]) } else { f32::fact([1, 1, 1, 8]) }
    }

    /// The expanded `HardSigmoid` then `Mul` that RVM and PP-HumanSeg emit.
    fn expanded(dt: DatumType, alpha: f32, gate_first: bool) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", fact(dt))?;
        let alpha = scalar(&mut model, "alpha", dt, alpha)?;
        let beta = scalar(&mut model, "beta", dt, BETA)?;
        let zero = scalar(&mut model, "zero", dt, 0.0)?;
        let one = scalar(&mut model, "one", dt, 1.0)?;
        let scaled = model.wire_node("mul_alpha", mul(), &[alpha, x])?[0];
        let shifted = model.wire_node("add_beta", add(), &[beta, scaled])?[0];
        let capped = model.wire_node("sat_one", min(), &[one, shifted])?[0];
        let gate = model.wire_node("sat_zero", max(), &[zero, capped])?[0];
        let inputs = if gate_first { [gate, x] } else { [x, gate] };
        let y = model.wire_node("mul_x", mul(), &inputs)?[0];
        model.select_output_outlets(&[y])?;
        Ok(model)
    }

    fn reference(dt: DatumType) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", fact(dt))?;
        let y = model.wire_node("h", super::super::hard_swish(), &[x])?[0];
        model.select_output_outlets(&[y])?;
        Ok(model)
    }

    #[test]
    fn folds_hardsigmoid_mul_into_hardswish() -> TractResult<()> {
        let values: Vec<f32> = (-4..4).map(|i| i as f32 * 0.75).collect();
        for dt in [DatumType::F32, DatumType::F16] {
            for gate_first in [false, true] {
                let mut model = expanded(dt, ALPHA, gate_first)?;
                model.declutter()?;
                let out = model.output_outlets()?[0];
                let op = model.node(out.node).op_as::<ElementWiseOp>().context("not folded")?;
                ensure!(op.0.is::<HardSwish>(), "{dt:?} gate_first={gate_first} did not fold");
                let input =
                    Tensor::from_shape(&[1, 1, 1, 8], &values)?.cast_to_dt(dt)?.into_owned();
                let folded = SimplePlan::new(model)?.run(tvec!(input.clone().into()))?.remove(0);
                let direct = SimplePlan::new(reference(dt)?)?.run(tvec!(input.into()))?.remove(0);
                folded.into_tensor().close_enough(&direct.into_tensor(), Approximation::Close)?;
            }
        }
        let mut other = expanded(DatumType::F32, 0.2, true)?;
        other.declutter()?;
        let out = other.output_outlets()?[0];
        ensure!(other.node(out.node).op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Mul>()));
        Ok(())
    }
}
