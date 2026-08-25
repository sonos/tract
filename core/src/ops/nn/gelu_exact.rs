use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::math::{Add, Mul};
use tract_linalg::routines::Func;

const CHUNK: usize = 1024;

fn inv_sqrt2() -> f32 {
    (2.0f32).sqrt().recip()
}

crate::element_wise!(gelu_exact, GeluExact,
    [f16] => |_, xs| {
        let erf = Func::Erf.ew_f32()?;
        let c = f16::from_f32(inv_sqrt2());
        let half = f16::from_f32(0.5);
        let one = f16::from_f32(1.0);
        let mut scratch = vec![0f32; xs.len().min(CHUNK)];
        for chunk in xs.chunks_mut(CHUNK) {
            let scaled = &mut scratch[..chunk.len()];
            scaled.iter_mut().zip(chunk.iter()).for_each(|(s, x)| *s = (*x * c).to_f32());
            erf.run(scaled)?;
            chunk.iter_mut().zip(scaled.iter()).for_each(|(x, e)| {
                *x = (*x * half) * (f16::from_f32(*e) + one);
            });
        }
        Ok(())
    },
    [f32] => |_, xs| {
        let erf = Func::Erf.ew_f32()?;
        let c = inv_sqrt2();
        let mut scratch = vec![0f32; xs.len().min(CHUNK)];
        for chunk in xs.chunks_mut(CHUNK) {
            let scaled = &mut scratch[..chunk.len()];
            scaled.iter_mut().zip(chunk.iter()).for_each(|(s, x)| *s = *x * c);
            erf.run(scaled)?;
            chunk.iter_mut().zip(scaled.iter()).for_each(|(x, e)| *x = (*x * 0.5) * (*e + 1.0));
        }
        Ok(())
    };
    cost: |dt| {tvec!((Cost::FMA(dt), 14), (Cost::Div(dt), 1))}
);

/// Search pattern => GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
///
/// Anchored on the `Erf`, which the ONNX `Gelu` (without `approximate="tanh"`)
/// and `BiasGelu` expansions both emit as the middle of a five node chain.
pub fn detect_gelu_exact(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let erf_node = node;
    let dt = model.node_input_facts(erf_node.id)?[0].datum_type;
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    // x / sqrt(2)
    let scale = &model.nodes()[erf_node.inputs[0].node];
    rule_if_some!(scale_op = scale.op_as::<TypedBinOp>());
    rule_if!(scale_op.0.is::<Mul>());
    rule_if!(model.matches_single_input_const(scale, inv_sqrt2()));
    rule_if_some!(
        x = scale
            .inputs
            .iter()
            .find(|o| model.outlet_fact(**o).map(|f| f.konst.is_none()).unwrap_or(false))
            .copied()
    );

    // 1 + erf(x / sqrt(2))
    rule_if_some!(one_plus_erf = model.find_succ_bin_with_const::<Add>(erf_node, 1.0));

    // (0.5 * x) * (1 + erf(x / sqrt(2)))
    rule_if_some!(out = model.single_succ(one_plus_erf.id)?);
    rule_if_some!(out_op = out.op_as::<TypedBinOp>());
    rule_if!(out_op.0.is::<Mul>());
    rule_if_some!(
        half_x = out
            .inputs
            .iter()
            .filter_map(|i| {
                let n = &model.nodes()[i.node];
                n.op_as::<TypedBinOp>()?.0.is::<Mul>().then_some(n)
            })
            .next()
    );
    rule_if!(model.matches_single_input_const(half_x, 0.5));
    rule_if!(half_x.inputs.contains(&x));

    let mut patch = TypedModelPatch::default();
    let tap = patch.taps(model, &[x])?;
    let wired =
        patch.wire_node(format!("{}.gelu_exact", erf_node.name), gelu_exact(), &[tap[0]])?;
    patch.shunt_outside(model, out.id.into(), wired[0])?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::element_wise::ElementWiseOp;
    use crate::ops::math::{Erf, add, erf, mul};

    fn chain(dt: DatumType, len: usize) -> TractResult<TypedModel> {
        let mut m = TypedModel::default();
        let x = m.add_source("x", dt.fact([len]))?;
        let c = m.add_const("c", tensor1(&[inv_sqrt2()]).cast_to_dt(dt)?.into_owned())?;
        let scaled = m.wire_node("scale", mul(), &[x, c])?[0];
        let e = m.wire_node("erf", erf(), &[scaled])?[0];
        let one = m.add_const("one", tensor1(&[1f32]).cast_to_dt(dt)?.into_owned())?;
        let ope = m.wire_node("add_one", add(), &[e, one])?[0];
        let half = m.add_const("half", tensor1(&[0.5f32]).cast_to_dt(dt)?.into_owned())?;
        let hx = m.wire_node("half_x", mul(), &[x, half])?[0];
        let out = m.wire_node("out", mul(), &[hx, ope])?;
        m.select_output_outlets(&out)?;
        Ok(m)
    }

    fn is_mini<T: crate::ops::element_wise::ElementWiseMiniOp>(n: &TypedNode) -> bool {
        n.op_as::<ElementWiseOp>().map(|e| e.0.is::<T>()).unwrap_or(false)
    }

    fn input(dt: DatumType, len: usize) -> TractResult<TValue> {
        let values: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin() * 4.0).collect();
        Ok(tensor1(&values).cast_to_dt(dt)?.into_owned().into_tvalue())
    }

    #[test]
    fn fuses_the_chain_and_keeps_the_values() -> TractResult<()> {
        for dt in [DatumType::F32, DatumType::F16] {
            let len = 2050;
            let raw = chain(dt, len)?;
            let reference = raw.clone().into_runnable()?.run(tvec!(input(dt, len)?))?;

            let fused = raw.into_decluttered()?;
            assert!(
                fused.nodes().iter().any(is_mini::<GeluExact>),
                "{dt:?}: no GeluExact after declutter"
            );
            assert!(!fused.nodes().iter().any(is_mini::<Erf>), "{dt:?}: Erf survived");

            let got = fused.into_runnable()?.run(tvec!(input(dt, len)?))?;
            assert_eq!(
                got[0].as_bytes(),
                reference[0].as_bytes(),
                "{dt:?}: fused output differs from the chain"
            );
        }
        Ok(())
    }
}
