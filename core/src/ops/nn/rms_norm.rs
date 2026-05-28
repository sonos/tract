use crate::internal::*;
use crate::ops::binary::{BinMiniOp, TypedBinOp};
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::{Add, Mul, Rsqrt};
use crate::ops::nn::{Reduce, Reducer};
use tract_itertools::Itertools;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for RmsNorm {
    fn name(&self) -> StaticName {
        "RmsNorm".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for RmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let in_dt = input.datum_type();

        // Fast path: F32 or F16 input where the normalised axis is the last
        // (contiguous) one. Use the fused tract_linalg::rms_norm_f32 kernel
        // (AVX-512 when available; scalar fallback otherwise) instead of the
        // 4-call MeanOfSquares + Add + Rsqrt + Mul composition below. ~16-18x
        // faster on Cascade Lake AVX-512, ~equivalent on the scalar fallback
        // since the composition is also memory-bandwidth bound.
        if matches!(in_dt, DatumType::F32 | DatumType::F16)
            && input.rank() > 0
            && self.axis == input.rank() - 1
        {
            let eps_f32: f32 = self.eps.cast_to_scalar::<f32>()?;
            let mut buf = input.cast_to::<f32>()?.into_owned();
            let row_len = buf.shape()[self.axis];
            if row_len > 0 {
                let n_rows: usize = buf.shape().iter().take(self.axis).product();
                let data = unsafe { buf.as_slice_mut_unchecked::<f32>() };
                let rms_norm = &tract_linalg::ops().rms_norm_f32;
                for r in 0..n_rows {
                    let start = r * row_len;
                    rms_norm(&mut data[start..start + row_len], eps_f32);
                }
            }
            return Ok(tvec![buf.cast_to_dt(in_dt)?.into_owned().into()]);
        }

        // Slow path: original 4-call composition (kept for non-contiguous axes).
        let input_f32 = input.cast_to::<f32>()?.into_owned();
        // eps inherits the input dtype from the declutter pattern (F16 when the
        // surrounding LayerNorm chain is F16). The MeanOfSquares + Add + Rsqrt
        // + Mul chain below all runs at F32, so eps must be cast to match —
        // otherwise the Add::eval call below panics with
        //   "tensor is F32, accessed as F16"
        // when input is F16.
        let eps = self.eps.cast_to::<f32>()?.into_owned();
        let a1 = Reducer::MeanOfSquares.reduce(&[self.axis], &input_f32)?;
        let mut a2 = Add.eval(a1.into_tvalue(), eps.into_tvalue(), DatumType::F32)?;
        Rsqrt {}.eval_in_place(&mut a2, None)?;
        let a3 = Mul.eval(a2.into_tvalue(), input_f32.into_tvalue(), DatumType::F32)?;
        Ok(tvec![a3.cast_to_dt(in_dt)?.into_owned().into()])
    }
}

impl TypedOp for RmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.eps.rank() == 0, "RmsNorm: eps must be a rank-0 tensor");
        ensure!(
            self.axis < inputs[0].rank(),
            "RmsNorm: axis {} is out of bounds for input rank {}",
            self.axis,
            inputs[0].rank()
        );
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        crate::optim::propagate_roi::bubble_roi(model, node)
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
                Axis::new(letters.next().unwrap(), inputs.len(), 1).input(0, ix).output(0, ix)
            })
            .collect_vec();
        AxesMapping::new(1, 1, axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            let op = Some(Box::new(RmsNorm { axis, eps: self.eps.clone() }) as _);
            Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
        } else {
            Ok(None)
        }
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        rule_if!(output_axis != self.axis);
        patch.wire_node(&node.name, self.clone(), inputs).map(Some)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let dt = inputs[0].datum_type;
        let count: TDim = inputs[0].shape.iter().product();
        // per element: square + accumulate + mul by rsqrt ≈ 3 FMA
        // per reduction group: 1 div (rsqrt)
        let groups: TDim = inputs[0]
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.axis)
            .map(|(_, d)| d)
            .product();
        Ok(tvec!((Cost::FMA(dt), count * 3), (Cost::Div(dt), groups)))
    }

    as_op!();
}

/// Search pattern => A = A * RSQRT(MEAN_OF_SQUARES(A) + EPS)
pub fn detect_rms_norm(
    op: &Reduce,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.reducer == Reducer::MeanOfSquares);
    rule_if!(op.axes.len() == 1);
    let axis = op.axes[0];

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Add operator
    rule_if_some!(add_succ = model.single_succ(node.id)?);
    rule_if_some!(add_succ_op = add_succ.op_as::<TypedBinOp>());
    rule_if!(add_succ_op.0.is::<Add>());

    // Retrieve epsilon
    let add_consts = model.collect_const_inputs(add_succ);
    rule_if!(add_consts.len() == 1);
    let eps = add_consts[0].val().clone();
    rule_if!(eps.len() == 1);
    rule_if!(eps.datum_type() == dt);
    let eps = eps.into_tensor().into_shape(&[])?.into_arc_tensor();

    // Identify Rsqrt
    rule_if_some!(rsqrt_succ = model.single_succ(add_succ.id)?);
    rule_if_some!(rsqrt_succ_op = rsqrt_succ.op_as::<ElementWiseOp>());
    rule_if!(rsqrt_succ_op.0.is::<Rsqrt>());

    // Identify Mul: RSQRT(...) * A
    rule_if_some!(mul_succ = model.find_succ_bin_with_outlet::<Mul>(rsqrt_succ, &node.inputs[0]));

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let out =
        patch.wire_node(format!("{}.rms_norm", node.name), RmsNorm { axis, eps }, &rsm_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::nn::RmsNorm;

    /// Regression: the declutter pattern (`detect_rms_norm`) stores `eps` with
    /// the input dtype (F16 when the surrounding LayerNorm chain is F16) — see
    /// `rule_if!(eps.datum_type() == dt)` above. The eval path runs at F32, so
    /// it must cast `self.eps` to F32 before using it. Without the cast in
    /// `RmsNorm::eval`, this test panics with "tensor is F32, accessed as F16".
    #[test]
    fn eval_with_f16_eps_and_f16_input() {
        let to_h = |x: f32| f16::from_f32(x);
        let input = tensor1(&[to_h(1.0), to_h(2.0), to_h(3.0), to_h(4.0)]);
        let eps = tensor0(to_h(1e-5)).into_arc_tensor();
        let op = RmsNorm { axis: 0, eps };
        let out = op.eval(tvec!(input.clone().into())).expect("eval should not panic");
        let out = out.into_iter().next().unwrap().into_tensor();
        assert_eq!(out.datum_type(), DatumType::F16);
        assert_eq!(out.shape(), &[4]);
        // Reference: rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + 1e-5) ≈ 2.7386
        // normalised: [1, 2, 3, 4] / 2.7386 ≈ [0.365, 0.730, 1.095, 1.461]
        let got = unsafe { out.as_slice_unchecked::<f16>() };
        let expected = [0.365_f32, 0.730, 1.095, 1.461];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g.to_f32() - e).abs();
            assert!(diff < 0.01, "lane {i}: got {} expected {}", g.to_f32(), e);
        }
    }

    /// Slow path: when the normalised axis is NOT the trailing one, the fast
    /// path in `eval` (which dispatches to `tract_linalg::ops().rms_norm_f32`)
    /// is skipped and the original 4-call `MeanOfSquares` + `Add` + `Rsqrt` +
    /// `Mul` composition runs. Asserts the result is identical to a hand-
    /// computed reference, so the slow path stays correct after the fast-path
    /// addition.
    #[test]
    fn eval_with_non_trailing_axis_f32() {
        // 2x3 input, axis=0 means we normalise across the 2 rows for each
        // column independently:
        //   col 0: [1, 4] → mean_sq = (1 + 16) / 2 =  8.5 → 1/√8.5
        //   col 1: [2, 5] → mean_sq = (4 + 25) / 2 = 14.5 → 1/√14.5
        //   col 2: [3, 6] → mean_sq = (9 + 36) / 2 = 22.5 → 1/√22.5
        let input = tensor2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let eps = tensor0(0.0_f32).into_arc_tensor();
        let op = RmsNorm { axis: 0, eps };
        let out = op.eval(tvec!(input.into())).expect("eval should not panic");
        let out = out.into_iter().next().unwrap().into_tensor();
        assert_eq!(out.datum_type(), DatumType::F32);
        assert_eq!(out.shape(), &[2, 3]);
        let got = unsafe { out.as_slice_unchecked::<f32>() };
        let inv = |ms: f32| ms.sqrt().recip();
        let expected: [f32; 6] = [
            1.0 * inv(8.5),
            2.0 * inv(14.5),
            3.0 * inv(22.5),
            4.0 * inv(8.5),
            5.0 * inv(14.5),
            6.0 * inv(22.5),
        ];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(diff < 1e-5, "lane {i}: got {g}, want {e}, diff {diff}");
        }
    }
}
