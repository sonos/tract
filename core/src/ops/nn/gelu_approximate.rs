use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::Add;
use crate::ops::math::Mul;
use crate::ops::math::PowConst;
use crate::ops::math::Square;

use tract_data::half::f16;
use tract_linalg::routines::Func;

fn gelu_approx_f32(x: f32, pow: i32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(pow))))
}

element_wise!(gelu_approximate, GeluApproximate { fast_impl: bool },
    [f16] => |op, xs| {
        let pow = if op.fast_impl { 2 } else { 3 };
        xs.iter_mut().for_each(|x| {
            *x = f16::from_f32(gelu_approx_f32(x.to_f32(), pow));
        });
        Ok(())
    },
    [f32] => |op, xs| {
        if op.fast_impl {
            // pow=2 fast path: no linalg kernel yet, scalar fallback.
            xs.iter_mut().for_each(|x| {
                *x = gelu_approx_f32(*x, 2);
            });
            Ok(())
        } else {
            // pow=3 canonical path: linalg NEON kernel composes with tanh.
            Func::Gelu.ew_f32()?.run(xs)
        }
    };
    cost: |dt| {tvec!((Cost::FMA(dt), 15))}
);

/// A tree of `Mul`s as its non-constant factors and constant coefficient. `Square` and
/// `PowConst(3)` count their operand's factors twice and three times; any other node is a
/// factor of its own.
fn product(
    model: &TypedModel,
    outlet: OutletId,
    depth: usize,
) -> TractResult<(TVec<OutletId>, f32)> {
    let node = model.node(outlet.node);
    if depth > 0 {
        if let Some(bin) = node.op_as::<TypedBinOp>()
            && bin.0.is::<Mul>()
        {
            let (mut factors, mut coef) = (tvec!(), 1.0);
            for input in &node.inputs {
                if let Some(k) = &model.outlet_fact(*input)?.uniform {
                    coef *= k.cast_to_scalar::<f32>()?;
                } else {
                    let (f, k) = product(model, *input, depth - 1)?;
                    factors.extend(f);
                    coef *= k;
                }
            }
            return Ok((factors, coef));
        }
        if let Some(ew) = node.op_as::<ElementWiseOp>() {
            let power = if ew.0.is::<Square>() {
                2
            } else if ew.0.downcast_ref::<PowConst>().is_some_and(|p| p.exponent == 3.0) {
                3
            } else {
                0
            };
            if power > 0 {
                let (f, k) = product(model, node.inputs[0], depth - 1)?;
                return Ok(((0..power).flat_map(|_| f.iter().copied()).collect(), k.powi(power)));
            }
        }
    }
    Ok((tvec!(outlet), 1.0))
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 * b.abs()
}

/// The other operand of an `Add` whose one operand is the uniform constant `konst`.
fn add_const_operand(
    model: &TypedModel,
    node: &TypedNode,
    konst: f32,
) -> TractResult<Option<OutletId>> {
    rule_if_some!(bin = node.op_as::<TypedBinOp>());
    rule_if!(bin.0.is::<Add>() && node.inputs.len() == 2);
    for (i, input) in node.inputs.iter().enumerate() {
        if let Some(k) = &model.outlet_fact(*input)?.uniform
            && close(k.cast_to_scalar::<f32>()?, konst)
        {
            return Ok(Some(node.inputs[1 - i]));
        }
    }
    Ok(None)
}

fn is_only(factors: &[OutletId], x: OutletId, count: usize) -> bool {
    factors.len() == count && factors.iter().all(|f| *f == x)
}

/// The tanh argument `c · (x + a · x^n)`, written expanded or factored as `c · x · (1 + a · x²)`,
/// with its factors in any order: returns `(x, c, a, n)`.
fn tanh_argument(
    model: &TypedModel,
    outlet: OutletId,
) -> TractResult<Option<(OutletId, f32, f32, usize)>> {
    let (factors, c) = product(model, outlet, 4)?;
    if factors.len() == 1 {
        let sum = model.node(factors[0].node);
        rule_if_some!(bin = sum.op_as::<TypedBinOp>());
        rule_if!(bin.0.is::<Add>() && sum.inputs.len() == 2);
        for (lin, pow) in [(0, 1), (1, 0)] {
            let (x, k) = product(model, sum.inputs[lin], 4)?;
            let (p, a) = product(model, sum.inputs[pow], 4)?;
            if x.len() == 1 && k == 1.0 && (is_only(&p, x[0], 3) || is_only(&p, x[0], 2)) {
                return Ok(Some((x[0], c, a, p.len())));
            }
        }
    } else if factors.len() == 2 {
        for (lin, sum) in [(0, 1), (1, 0)] {
            let x = factors[lin];
            if let Some(rest) = add_const_operand(model, model.node(factors[sum].node), 1.0)? {
                let (p, a) = product(model, rest, 4)?;
                if is_only(&p, x, 2) {
                    return Ok(Some((x, c, a, 3)));
                }
            }
        }
    }
    Ok(None)
}

/// Detects the tanh approximation of GELU anchored on its `Tanh`:
/// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x^n)))` with n = 3, or n = 2 for the fast
/// variant, the cubic one also as `√(2/π) · x · (1 + 0.044715 · x²)`, with the products in any
/// grouping and order (`Square` and `PowConst(3)` included). Replaces the whole expression by
/// [`GeluApproximate`]. Only F16 and F32.
pub fn detect_gelu_approx(
    model: &TypedModel,
    tanh: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!((x, c, a, power) = tanh_argument(model, tanh.inputs[0])?);
    rule_if!(matches!(model.outlet_fact(x)?.datum_type, DatumType::F32 | DatumType::F16));
    rule_if!(close(c, (2.0 / std::f32::consts::PI).sqrt()) && close(a, 0.044715));
    rule_if_some!(one_plus = model.single_succ(tanh.id)?);
    rule_if_some!(one_plus_tanh = add_const_operand(model, one_plus, 1.0)?);
    rule_if!(one_plus_tanh == tanh.id.into());
    let mut end = one_plus;
    for _ in 0..2 {
        rule_if_some!(next = model.single_succ(end.id)?);
        end = next;
        let (factors, k) = product(model, end.id.into(), 4)?;
        let mut factors = factors.to_vec();
        factors.sort();
        let mut want = vec![x, one_plus.id.into()];
        want.sort();
        if factors == want && close(k, 0.5) {
            let mut patch = TypedModelPatch::default();
            let input = patch.tap_model(model, x)?;
            let out = patch.wire_node(
                format!("{}.gelu_approx", end.name),
                gelu_approximate(power == 2),
                &[input],
            )?;
            patch.shunt_outside(model, end.id.into(), out[0])?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
}
