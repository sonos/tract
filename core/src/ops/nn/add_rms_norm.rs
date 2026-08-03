use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::math::Add;
use crate::ops::nn::RmsNorm;
use tract_itertools::Itertools;

/// A residual add feeding an [`RmsNorm`] over the trailing axis, evaluated in one
/// pass per row.
///
/// Output 0 is the sum, kept because a transformer residual is read again by the
/// next block; output 1 is the normalised sum. Produced by [`detect_add_rms_norm`]
/// off `RmsNorm`'s declutter, and restricted to the same inputs `RmsNorm`'s fused
/// path accepts: f32 or f16, trailing axis, no broadcasting.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AddRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for AddRmsNorm {
    fn name(&self) -> StaticName {
        "AddRmsNorm".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for AddRmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        let dt = a.datum_type();
        let eps_f32: f32 = self.eps.cast_to_scalar::<f32>()?;
        let row_len = a.shape()[self.axis];

        let mut sum = a.into_tensor();

        let add =
            tract_linalg::bin_unicast(dt, tract_linalg::BinOp::Add).context("no add kernel")?;
        add(&mut sum.view_mut(), &b.view())?;

        let mut normed =
            if dt == DatumType::F32 { sum.clone() } else { sum.cast_to::<f32>()?.into_owned() };
        if row_len > 0 {
            let rms_norm = &tract_linalg::ops().rms_norm_f32;
            let mut plain = normed.try_as_plain_mut()?;
            let data = plain.as_slice_mut::<f32>()?;
            let total = data.len();
            tract_linalg::multithread::par_chunks_mut(data, row_len, total, |_, chunk| {
                for row in chunk.chunks_mut(row_len) {
                    rms_norm(row, eps_f32);
                }
                Ok(())
            })?;
        }
        if dt != DatumType::F32 {
            normed = normed.cast_to_dt(dt)?.into_owned();
        }
        Ok(tvec![sum.into_tvalue(), normed.into_tvalue()])
    }
}

impl TypedOp for AddRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.eps.rank() == 0, "AddRmsNorm: eps must be a rank-0 tensor");
        ensure!(inputs.len() == 2, "AddRmsNorm: expects two inputs");
        ensure!(inputs[0].shape == inputs[1].shape, "AddRmsNorm: inputs must have equal shapes");
        ensure!(
            self.axis < inputs[0].rank(),
            "AddRmsNorm: axis {} is out of bounds for input rank {}",
            self.axis,
            inputs[0].rank()
        );
        let fact = inputs[0].datum_type.fact(inputs[0].shape.clone());
        Ok(tvec!(fact.clone(), fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let rank = inputs[0].rank();
        let mut letters = 'a'..;
        let axes = (0..rank)
            .map(|ix| {
                Axis::new(letters.next().unwrap(), 2, 2)
                    .input(0, ix)
                    .input(1, ix)
                    .output(0, ix)
                    .output(1, ix)
            })
            .collect_vec();
        AxesMapping::new(2, 2, axes)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let dt = inputs[0].datum_type;
        let count: TDim = inputs[0].shape.iter().product();
        let groups: TDim = inputs[0]
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.axis)
            .map(|(_, d)| d)
            .product();
        Ok(tvec!((Cost::FMA(dt), count * 4), (Cost::Div(dt), groups)))
    }

    as_op!();
}

/// Search pattern => RMS_NORM(A + B), keeping A + B available.
///
/// A transformer reads the residual sum again in the next block, so this cannot
/// drop it; the fused node exposes it as output 0 and the normalised value as
/// output 1.
pub fn detect_add_rms_norm(
    op: &RmsNorm,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));
    rule_if!(in_fact.rank() > 0 && op.axis == in_fact.rank() - 1);

    let add = &model.nodes()[node.inputs[0].node];
    rule_if_some!(add_op = add.op_as::<TypedBinOp>());
    rule_if!(add_op.0.is::<Add>());
    rule_if!(add.inputs.len() == 2);

    // Broadcasting would make the sum a different shape from either operand.
    let a = model.outlet_fact(add.inputs[0])?;
    let b = model.outlet_fact(add.inputs[1])?;
    rule_if!(a.shape == b.shape && a.shape == in_fact.shape);
    rule_if!(a.datum_type == dt && b.datum_type == dt);

    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &add.inputs)?;
    let wired = patch.wire_node(
        format!("{}.add_rms_norm", node.name),
        AddRmsNorm { axis: op.axis, eps: op.eps.clone() },
        &taps,
    )?;
    patch.shunt_outside(model, add.id.into(), wired[0])?;
    patch.shunt_outside(model, node.id.into(), wired[1])?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::math::add;

    fn residual_norm(dt: DatumType, rows: usize, len: usize) -> TractResult<TypedModel> {
        let mut m = TypedModel::default();
        let x = m.add_source("x", dt.fact([rows, len]))?;
        let r = m.add_source("r", dt.fact([rows, len]))?;
        let sum = m.wire_node("sum", add(), &[x, r])?[0];
        let eps = tensor0(1e-5f32).cast_to_dt(dt)?.into_owned().into_arc_tensor();
        let normed = m.wire_node("rms", RmsNorm { axis: 1, eps }, &[sum])?[0];
        // The residual is read again downstream, as it is in a transformer.
        let out = m.wire_node("next", add(), &[sum, normed])?;
        m.select_output_outlets(&out)?;
        Ok(m)
    }

    fn inputs(dt: DatumType, rows: usize, len: usize) -> TractResult<TVec<TValue>> {
        let mk = |seed: f32| -> TractResult<TValue> {
            let v: Vec<f32> =
                (0..rows * len).map(|i| ((i as f32 + seed) * 0.37).sin() * 3.0).collect();
            Ok(tensor1(&v).into_shape(&[rows, len])?.cast_to_dt(dt)?.into_owned().into_tvalue())
        };
        Ok(tvec!(mk(0.0)?, mk(11.0)?))
    }

    #[test]
    fn fuses_and_keeps_both_values() -> TractResult<()> {
        for dt in [DatumType::F32, DatumType::F16] {
            let (rows, len) = (8, 320);
            let raw = residual_norm(dt, rows, len)?;
            let reference = raw.clone().into_runnable()?.run(inputs(dt, rows, len)?)?;

            let fused = raw.into_decluttered()?;
            assert!(
                fused.nodes().iter().any(|n| n.op_is::<AddRmsNorm>()),
                "{dt:?}: no AddRmsNorm after declutter"
            );
            assert!(
                !fused.nodes().iter().any(|n| n.op_is::<RmsNorm>()),
                "{dt:?}: RmsNorm survived"
            );

            let got = fused.into_runnable()?.run(inputs(dt, rows, len)?)?;
            assert_eq!(
                got[0].as_bytes(),
                reference[0].as_bytes(),
                "{dt:?}: fused output differs from the unfused graph"
            );
        }
        Ok(())
    }
}
