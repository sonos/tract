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

        let input_f32 = input.cast_to::<f32>()?.into_owned();
        let a1 = Reducer::MeanOfSquares.reduce(&[self.axis], &input_f32)?;
        let mut a2 = Add.eval(a1.into_tvalue(), self.eps.clone().into_tvalue(), DatumType::F32)?;
        Rsqrt {}.eval_in_place(&mut a2, None)?;
        let a3 = Mul.eval(a2.into_tvalue(), input_f32.into_tvalue(), DatumType::F32)?;
        Ok(tvec![a3.cast_to_dt(input.datum_type())?.into_owned().into()])
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
        if output_axis == self.axis {
            return Ok(None);
        }
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
