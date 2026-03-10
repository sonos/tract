use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::konst::Const;
use crate::ops::math::{Add, Mul, Pow, Tanh};

#[derive(Default, Clone, Debug, Hash)]
/// NEW_GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)));
pub struct GeluApproximate {
    pub fast_impl: bool,
}

impl Op for GeluApproximate {
    fn name(&self) -> StaticName {
        if self.fast_impl {
            "GeluApproximateFast".to_string().into()
        } else {
            "GeluApproximate".to_string().into()
        }
    }
    op_as_typed_op!();
}

impl EvalOp for GeluApproximate {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();

        let a_f32 = input.cast_to_dt(DatumType::F32)?;

        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

        let pow = if self.fast_impl { 2 } else { 3 };
        let gelu_approx_f32_data = a_f32
            .try_as_dense()?
            .as_slice::<f32>()?
            .iter()
            .map(|x| 0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(pow)))))
            .collect::<Vec<_>>();

        let gelu_approx_f32 = Tensor::from_shape(input.shape(), &gelu_approx_f32_data)?;
        Ok(tvec![gelu_approx_f32.cast_to_dt(dt)?.into_owned().into_tvalue()])
    }
}

impl TypedOp for GeluApproximate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

// Helper: collect constant inputs of a node
fn collect_const_inputs<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a Const> {
    node.inputs
        .iter()
        .filter_map(|i| {
            let prec = &model.nodes()[i.node];
            prec.op_as::<Const>()
        })
        .collect::<TVec<_>>()
}

// Helper: check if node has exactly one constant input matching a scalar value
fn matches_single_input_const(model: &TypedModel, node: &TypedNode, konst: f32) -> bool {
    let consts = collect_const_inputs(model, node);
    if consts.len() != 1 {
        return false;
    }
    let Ok(in_const) = consts[0].val().cast_to_dt(DatumType::F32) else {
        return false;
    };
    let Ok(in_const) = in_const.to_scalar_tensor() else {
        return false;
    };
    in_const.close_enough(&tensor0(konst), Approximation::Approximate).is_ok()
}

// Helper: find successor binary op of type B where one input is a constant matching the value
fn find_succ_bin_with_const<'a, B: crate::ops::binary::BinMiniOp>(
    model: &'a TypedModel,
    node: &'a TypedNode,
    konst: f32,
) -> Option<&'a TypedNode> {
    let succ = model.single_succ(node.id).ok()??;
    let succ_op = succ.op_as::<TypedBinOp>()?;
    (succ_op.0.is::<B>() && matches_single_input_const(model, succ, konst)).then_some(succ)
}

// Helper: find successor binary op of type B where one input is the given outlet
fn find_succ_bin_with_outlet<'a, B: crate::ops::binary::BinMiniOp>(
    model: &'a TypedModel,
    node: &'a TypedNode,
    outlet_id: &OutletId,
) -> Option<&'a TypedNode> {
    let succ = model.single_succ(node.id).ok()??;
    let succ_op = succ.op_as::<TypedBinOp>()?;
    (succ_op.0.is::<B>() && succ.inputs.contains(outlet_id)).then_some(succ)
}

/// Search pattern => NEW_GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N))); N ∈ {2, 3}
pub fn detect_gelu_approx(
    _op: &Pow,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let pow_node = node;

    let in_fact = model.node_input_facts(pow_node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    rule_if!(
        matches_single_input_const(model, pow_node, 3.0)
            || matches_single_input_const(model, pow_node, 2.0)
    );
    let fast_impl = matches_single_input_const(model, pow_node, 2.0);

    // 0.044715 * x^N
    rule_if_some!(mul_coef_a = find_succ_bin_with_const::<Mul>(model, pow_node, 0.044715));

    // x + 0.044715 * x^N
    rule_if_some!(
        x_plus_mul_coef_a =
            find_succ_bin_with_outlet::<Add>(model, mul_coef_a, &pow_node.inputs[0])
    );

    // sqrt(2/pi) * (x + 0.044715 * x^N)
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    rule_if_some!(
        mul_sqrt_2_over_pi =
            find_succ_bin_with_const::<Mul>(model, x_plus_mul_coef_a, sqrt_2_over_pi)
    );

    // tanh(sqrt(2/pi) * (x + 0.044715 * x^N))
    rule_if_some!(tanh_succ = model.single_succ(mul_sqrt_2_over_pi.id)?);
    rule_if_some!(tanh_succ_op = tanh_succ.op_as::<ElementWiseOp>());
    rule_if!(tanh_succ_op.0.is::<Tanh>());

    // 1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)) N ∈ {2, 3}
    rule_if_some!(tanh_plus_1 = find_succ_bin_with_const::<Add>(model, tanh_succ, 1.0));

    // Identify Mul
    rule_if_some!(mul_succ = model.single_succ(tanh_plus_1.id)?);
    rule_if_some!(mul_succ_op = mul_succ.op_as::<TypedBinOp>());
    rule_if!(mul_succ_op.0.is::<Mul>());

    // Search first
    // tmp = x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)))
    // out = 0.5 * tmp
    let last_node_id = if mul_succ.inputs.contains(&pow_node.inputs[0]) {
        // 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)))
        rule_if_some!(last_mul_with_0_5 = find_succ_bin_with_const::<Mul>(model, mul_succ, 0.5));
        last_mul_with_0_5.id
    } else {
        // tmp = 0.5 * x
        // out = tmp * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N))) N ∈ {2, 3}
        rule_if_some!(
            x_mul_0_5 = mul_succ
                .inputs
                .iter()
                .filter_map(|i| {
                    let n = &model.nodes()[i.node];
                    let op = n.op_as::<TypedBinOp>()?;
                    op.0.is::<Mul>().then_some(n)
                })
                .next()
        );
        rule_if!(matches_single_input_const(model, x_mul_0_5, 0.5));
        rule_if!(x_mul_0_5.inputs.contains(&pow_node.inputs[0]));
        mul_succ.id
    };

    let mut patch = TypedModelPatch::default();
    let gelu_approx_input = patch.taps(model, &pow_node.inputs)?;
    let out = patch.wire_node(
        format!("{}.gelu_approx", pow_node.name),
        GeluApproximate { fast_impl },
        &[gelu_approx_input[0]],
    )?;
    patch.shunt_outside(model, last_node_id.into(), out[0])?;
    Ok(Some(patch))
}
