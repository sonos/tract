use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::array::{DynSlice, Slice, TypedConcat};
use tract_nnef::tract_core::ops::binary::BinMiniOp;
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::math::{Add, Mul, Neg};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_apply_rope);
    registry.register_primitive(
        "tract_transformers_apply_rope",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("cos"),
            TypeName::Scalar.tensor().named("sin"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_apply_rope,
    );
}

fn de_apply_rope(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let cos = invocation.named_arg_as(builder, "cos")?;
    let sin = invocation.named_arg_as(builder, "sin")?;
    builder.wire(ApplyRope, &[input, cos, sin])
}

fn ser_apply_rope(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &ApplyRope,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let cos: Arc<RValue> = ast.mapping[&node.inputs[1]].clone();
    let sin: Arc<RValue> = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation("tract_transformers_apply_rope", &[input, cos, sin], &[])))
}

/// Some exported graphs (e.g. torch_to_nnef partial-rotary attention) express
/// the rotary slices as dyn_slice ops whose bounds are computed from shape_of
/// chains. Those chains are constant once prop_consts has run, but the rope
/// patterns below only match static Slice ops, so fold DynSlice into Slice
/// whenever both bounds have resolved to constants. Bounds are kept as TDim,
/// so symbolic-but-constant expressions fold too.
pub fn fold_const_dyn_slice_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &DynSlice,
) -> TractResult<Option<TypedModelPatch>> {
    let inputs = model.node_input_facts(node.id)?;
    rule_if_some!(start = &inputs[1].konst);
    rule_if_some!(end = &inputs[2].konst);
    let (Ok(start), Ok(end)) = (as_scalar_tdim(start), as_scalar_tdim(end)) else {
        return Ok(None);
    };
    Ok(Some(TypedModelPatch::replace_single_op(
        model,
        node,
        &[node.inputs[0]],
        Slice { axis: op.axis, start, end },
    )?))
}

fn as_scalar_tdim(t: &Tensor) -> TractResult<TDim> {
    Ok(t.cast_to::<TDim>()?.try_as_plain()?.to_scalar::<TDim>()?.clone())
}

/// Exporters may wrap binary-op operands in no-op casts (cast to the dtype
/// the tensor already has). They sit between RotateHalf and its consumers
/// and defeat the apply-rope pattern; drop them (regular declutter does the
/// same, but the rope detection runs on the raw graph).
pub fn fold_identity_cast_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &tract_nnef::tract_core::ops::cast::Cast,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(model.node_input_facts(node.id)?[0].datum_type == op.to);
    let mut patch = TypedModelPatch::default();
    let input = patch.tap_model(model, node.inputs[0])?;
    patch.shunt_outside(model, node.id.into(), input)?;
    Ok(Some(patch))
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RotateHalf;

impl Op for RotateHalf {
    fn name(&self) -> StaticName {
        "RotateHalf".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for RotateHalf {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let shape: TVec<_> = input.shape().into();
        let mut tensor = Tensor::zero_dt(input.datum_type(), &shape)?;

        let axis = shape.len() - 1;
        ensure!(
            shape[axis] % 2 == 0,
            "RotateHalf possible only if the most inner dimension of the shape {:?} is divible by 2",
            shape
        );
        let half = shape[axis] / 2;
        unsafe { tensor.assign_slice_unchecked(0..half, &input, half.., axis) };
        Neg {}.eval_in_place(&mut tensor, None)?;
        unsafe { tensor.assign_slice_unchecked(half.., &input, 0..half, axis) };
        Ok(tvec![tensor.into()])
    }
}

impl TypedOp for RotateHalf {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern:
/// Y = Concat(Neg(Slice(X, X.shape[-1]/2.., -1)), Slice(X, ..X.shape[-1]/2, -1))
pub fn rotate_half_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    let out_fact = model.node_output_facts(node.id)?[0];
    let dt = out_fact.datum_type;
    rule_if!(dt.is_float() || dt.is_integer());
    rule_if!(op.axis == out_fact.rank() - 1);

    let in_concat = model.previous_nodes(node);
    rule_if!(in_concat.len() == 2);

    let neg_half = in_concat[0];
    rule_if_some!(neg_half_op = neg_half.op_as::<ElementWiseOp>());
    rule_if!(neg_half_op.0.is::<Neg>());

    rule_if_some!(neg_half_slice = model.previous_node(neg_half));
    rule_if_some!(neg_half_slice_op = neg_half_slice.op_as::<Slice>());

    rule_if!(neg_half_slice_op.axis == op.axis);

    let pos_half = in_concat[1];
    rule_if_some!(pos_half_op = pos_half.op_as::<Slice>());

    rule_if!(pos_half_op.axis == op.axis);
    rule_if!(pos_half_op.end == neg_half_slice_op.start);
    rule_if!(neg_half_slice_op.end == out_fact.shape[op.axis].clone());

    // Ensure it is a half rotation
    rule_if_some!(pos_half_slice_end = pos_half_op.end.as_i64());
    rule_if_some!(concatenated_last_dim = out_fact.shape[op.axis].as_i64());
    rule_if!(pos_half_slice_end * 2 == concatenated_last_dim);

    let in_fact = model.node_input_facts(neg_half_slice.id)?[0];

    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &neg_half_slice.inputs)?;

    if pos_half_op.start != 0.into() || neg_half_slice_op.end != in_fact.shape[op.axis] {
        inputs = patch.wire_node(
            format!("{node_name}.rotate_half.slice"),
            Slice {
                start: pos_half_op.start.clone(),
                end: neg_half_slice_op.end.clone(),
                axis: op.axis,
            },
            &inputs,
        )?;
    }

    let out = patch.wire_node(format!("{node_name}.rotate_half"), RotateHalf, &inputs)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;

    Ok(Some(patch))
}

/// Search pattern (post-declutter form): PushSliceUp splits the rope input
/// over the rotate-half boundary, leaving a sibling pair
///   C  = Concat(a, b)          (the rope input, cos path)
///   Y  = Concat(Neg(b), a)     (the rotated input, sin path)
/// with a and b of equal width on the concat axis. By definition
/// Y == RotateHalf(C), so rewire Y through RotateHalf; the apply-rope rule
/// can then collapse the surrounding mul/add. The rewrite is sound wherever
/// the sibling pair exists (it is an identity on concat semantics).
pub fn rotate_half_concat_pair_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    let out_fact = model.node_output_facts(node.id)?[0];
    let dt = out_fact.datum_type;
    rule_if!(dt.is_float() || dt.is_integer());
    rule_if!(op.axis == out_fact.rank() - 1);
    rule_if!(node.inputs.len() == 2);

    let neg = model.node(node.inputs[0].node);
    rule_if_some!(neg_op = neg.op_as::<ElementWiseOp>());
    rule_if!(neg_op.0.is::<Neg>());
    rule_if!(neg.inputs.len() == 1);

    let b = neg.inputs[0];
    let a = node.inputs[1];
    rule_if!(a != b);
    rule_if!(model.outlet_fact(a)?.shape[op.axis] == model.outlet_fact(b)?.shape[op.axis]);

    // Find the sibling Concat(a, b) on the same axis.
    let sibling = model.node(a.node).outputs[a.slot].successors.iter().find(|inlet| {
        let s = model.node(inlet.node);
        s.id != node.id
            && s.inputs.len() == 2
            && s.inputs[0] == a
            && s.inputs[1] == b
            && s.op_as::<TypedConcat>().is_some_and(|c| c.axis == op.axis)
    });
    rule_if_some!(sibling = sibling);

    let mut patch = TypedModelPatch::default();
    let rope_input = patch.tap_model(model, OutletId::new(sibling.node, 0))?;
    let out = patch.wire_node(format!("{node_name}.rotate_half"), RotateHalf, &[rope_input])?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ApplyRope;

impl ApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }
}

impl Op for ApplyRope {
    fn name(&self) -> StaticName {
        "ApplyRope".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for ApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, cos, sin) = args_3!(inputs);
        let rotated_input = args_1!(RotateHalf.eval(tvec![input.clone()])?);
        let mul_with_cos = Mul.eval(input.clone(), cos, input.datum_type())?;
        let mul_with_sin = Mul.eval(rotated_input, sin, input.datum_type())?;
        let output = Add.eval(mul_with_cos.into(), mul_with_sin.into(), input.datum_type())?;
        Ok(tvec![output.into()])
    }
}

impl TypedOp for ApplyRope {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern:
/// Y = X * Cos + RotateHalf(X) * Sin
pub fn apply_rope_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.0.is::<Add>());

    let in_add = model.previous_nodes(node);
    rule_if!(in_add.len() == 2);

    rule_if!(in_add.iter().all(|n| n.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Mul>())));

    // The sin operand is the one fed by RotateHalf; it can be on either side
    // of the add.
    let (cos_mul, sin_mul, rotate_half_in_idx, rotate_half) =
        if let Some((idx, rh)) = model.single_prev_node_as::<RotateHalf>(in_add[1]) {
            (in_add[0], in_add[1], idx, rh)
        } else if let Some((idx, rh)) = model.single_prev_node_as::<RotateHalf>(in_add[0]) {
            (in_add[1], in_add[0], idx, rh)
        } else {
            return Ok(None);
        };

    // If cos and rotate half don't share the same input, we check if they don't
    // input node that are the same.
    let (apply_rope_in, cos) = if !cos_mul.inputs.contains(&rotate_half.inputs[0]) {
        rule_if_some!(rotate_half_prev = model.previous_node(rotate_half));
        rule_if_some!(
            (cos_common_input_idx, _) = model
                .previous_nodes(cos_mul)
                .iter()
                .enumerate()
                .find(|(_, n)| n.same_as(rotate_half_prev))
        );
        (rotate_half.inputs[0], cos_mul.inputs[1 - cos_common_input_idx])
    } else {
        let apply_rope_in = rotate_half.inputs[0];
        let cos =
            if cos_mul.inputs[0] == apply_rope_in { cos_mul.inputs[1] } else { cos_mul.inputs[0] };
        (apply_rope_in, cos)
    };

    let sin = sin_mul.inputs[1 - rotate_half_in_idx];

    rule_if!(ApplyRope::is_supported_dt(model.outlet_fact(apply_rope_in)?.datum_type));
    rule_if!(ApplyRope::is_supported_dt(model.outlet_fact(cos)?.datum_type));
    rule_if!(ApplyRope::is_supported_dt(model.outlet_fact(sin)?.datum_type));

    let mut patch = TypedModelPatch::default();
    let input = patch.tap_model(model, apply_rope_in)?;
    let cos = patch.tap_model(model, cos)?;
    let sin = patch.tap_model(model, sin)?;
    let out = patch.wire_node(format!("{node_name}.apply_rope"), ApplyRope, &[input, cos, sin])?;
    patch.shunt_outside(model, node.id.into(), out[0])?;

    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::ops::math::Neg;
    use tract_num_traits::AsPrimitive;
    use tract_num_traits::Zero;

    fn run_test_case<F: Datum + Zero + Copy>(a_shape: &[usize]) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        let a_len = a_shape.iter().product::<usize>();
        let input = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f.as_()).collect::<Vec<F>>())?;
        let rotated = RotateHalf.eval(tvec![input.clone().into()])?;
        let mut back = args_1!(RotateHalf.eval(rotated)?).into_tensor();
        Neg {}.eval_in_place(&mut back, None)?;
        back.close_enough(&input, Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn test_rotate_half() -> TractResult<()> {
        run_test_case::<f32>(&[2, 2])?;
        run_test_case::<f32>(&[512, 512])?;
        run_test_case::<f32>(&[10, 512, 1024])?;

        Ok(())
    }

    /// Builds the partial-rotary rope subgraph the way torch_to_nnef exports
    /// it (qwen3.5 geometry): all slices are DynSlice ops with scalar bound
    /// inputs, rotary_dim < head_dim, cos/sin broadcast over the head axis.
    /// After ApplyRopeTransform the graph must contain an ApplyRope op and
    /// keep the same numerics.
    fn build_partial_rotary_dyn_slice_model(
        heads: usize,
        seq: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> TractResult<TypedModel> {
        use tract_nnef::tract_core::ops::binary::TypedBinOp;
        use tract_nnef::tract_core::ops::math::{Add, Mul};

        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, heads, seq, head_dim]))?;
        let cos_len = (0..seq * rotary_dim).map(|f| (f as f32).cos()).collect::<Vec<_>>();
        let sin_len = (0..seq * rotary_dim).map(|f| (f as f32).sin()).collect::<Vec<_>>();
        let cos = model.add_const("cos", Tensor::from_shape(&[1, 1, seq, rotary_dim], &cos_len)?)?;
        let sin = model.add_const("sin", Tensor::from_shape(&[1, 1, seq, rotary_dim], &sin_len)?)?;

        let half = rotary_dim / 2;
        let mut bound = |name: &str, v: usize| model.add_const(name, tensor0(v as i64));
        let (c0, chalf, crot, chd) = (
            bound("c0", 0)?,
            bound("chalf", half)?,
            bound("crot", rotary_dim)?,
            bound("chd", head_dim)?,
        );

        let rot = model.wire_node(
            "rot",
            DynSlice { axis: 3, len: rotary_dim.to_dim() },
            &[x, c0, crot],
        )?[0];
        let pass = model.wire_node(
            "pass",
            DynSlice { axis: 3, len: (head_dim - rotary_dim).to_dim() },
            &[x, crot, chd],
        )?[0];
        let x1 =
            model.wire_node("x1", DynSlice { axis: 3, len: half.to_dim() }, &[rot, c0, chalf])?[0];
        let x2 = model
            .wire_node("x2", DynSlice { axis: 3, len: half.to_dim() }, &[rot, chalf, crot])?[0];
        let neg = model.wire_node("neg", tract_nnef::tract_core::ops::math::neg(), &[x2])?[0];
        let cat = model.wire_node("cat", TypedConcat::new(3), &[neg, x1])?[0];
        let mul_cos =
            model.wire_node("mul_cos", TypedBinOp(Box::new(Mul), None), &[rot, cos])?[0];
        let mul_sin =
            model.wire_node("mul_sin", TypedBinOp(Box::new(Mul), None), &[cat, sin])?[0];
        let roped =
            model.wire_node("roped", TypedBinOp(Box::new(Add), None), &[mul_cos, mul_sin])?[0];
        let out = model.wire_node("out", TypedConcat::new(3), &[roped, pass])?[0];
        model.select_output_outlets(&[out])?;
        Ok(model)
    }

    #[test]
    fn test_detect_partial_rotary_dyn_slice_rope() -> TractResult<()> {
        use tract_nnef::tract_core::transform::ModelTransform;

        let (heads, seq, head_dim, rotary_dim) = (4, 3, 16, 4);
        let model = build_partial_rotary_dyn_slice_model(heads, seq, head_dim, rotary_dim)?;

        let input = Tensor::from_shape(
            &[1, heads, seq, head_dim],
            &(0..heads * seq * head_dim).map(|f| (f as f32 * 0.17).sin()).collect::<Vec<_>>(),
        )?;

        let reference =
            model.clone().into_runnable()?.run(tvec![input.clone().into()])?[0].clone();

        let mut detected = model;
        crate::rewriter::ApplyRopeTransform.transform(&mut detected)?;
        assert_eq!(
            detected.nodes().iter().filter(|n| n.op_is::<ApplyRope>()).count(),
            1,
            "expected the partial-rotary dyn-slice pattern to fuse into ApplyRope, got: {detected:?}"
        );

        let fused = detected.into_runnable()?.run(tvec![input.into()])?[0].clone();
        fused.close_enough(&reference, Approximation::Close)?;
        Ok(())
    }
}
