use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_nnef::tract_core::ops::logic::Iff;
use tract_nnef::tract_core::ops::math::{Add, Mul};
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_scaled_masked_softmax);
    registry.register_primitive(
        "tract_transformers_scaled_masked_softmax",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("mask"),
            TypeName::Scalar.named("scale"),
            TypeName::Logical.named("post_softmax_mask"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scaled_masked_softmax,
    );
}

fn de_scaled_masked_softmax(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let mask = invocation.named_arg_as(builder, "mask")?;
    let scale = invocation.named_arg_as(builder, "scale")?;
    let post_softmax_mask: bool =
        invocation.get_named_arg_as(builder, "post_softmax_mask")?.unwrap_or(false);
    builder.wire(ScaledMaskedSoftmax { scale, post_softmax_mask }, &[input, mask])
}

fn ser_scaled_masked_softmax(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ScaledMaskedSoftmax,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let mask = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_transformers_scaled_masked_softmax",
        &[input, mask],
        &[
            ("scale", numeric(op.scale.cast_to_scalar::<f32>()?)),
            ("post_softmax_mask", logical(op.post_softmax_mask)),
        ],
    )))
}

/// Fused scale + mask + softmax over the last axis, with optional post-softmax zeroing.
///
/// - Float mask: `A = SOFTMAX(INPUT * SCALE + MASK, axis=-1)`
/// - Bool mask:  `A = SOFTMAX(IFF(MASK, INPUT * SCALE, -inf), axis=-1)`
///
/// If `post_softmax_mask` is true (bool mask only), also applies:
/// `A = IFF(MASK, A, 0)` — zeros out positions where the mask is false.
///
/// The mask dtype determines which mode is used at eval time.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
    pub post_softmax_mask: bool,
}

impl Op for ScaledMaskedSoftmax {
    fn name(&self) -> StaticName {
        "ScaledMaskedSoftmax".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        let mut v = vec![format!("scale: {:?}", self.scale)];
        if self.post_softmax_mask {
            v.push("post_softmax_mask: true".to_string());
        }
        Ok(v)
    }
    op_as_typed_op!();
}

impl EvalOp for ScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, mask) = args_2!(inputs);
        let softmax_axis = tvec!(input.rank() - 1);
        let dt = input.datum_type();
        let scale = self.scale.cast_to_dt(dt)?.into_owned();
        let scaled = Mul.eval(input, scale.into_tvalue(), dt)?;

        let pre_softmax: TValue = if mask.datum_type() == bool::datum_type() {
            // Boolean mask: keep score where mask=true, replace with -inf where mask=false.
            let fill = tensor0(-f32::INFINITY).cast_to_dt(dt)?.into_owned();
            Iff.eval(tvec![mask.clone(), scaled.into(), fill.into_tvalue()])?.remove(0)
        } else {
            Add.eval(scaled.into(), mask.clone(), dt)?.into()
        };

        let softmax_out = Softmax::new(softmax_axis, None, SoftmaxKind::Softmax(SoftmaxExp::Libc))
            .eval(tvec![pre_softmax])?[0]
            .clone();

        if self.post_softmax_mask {
            // Zero out positions where the bool mask is false.
            let zero = tensor0(0f32).cast_to_dt(dt)?.into_owned();
            Ok(Iff.eval(tvec![mask, softmax_out, zero.into_tvalue()])?)
        } else {
            Ok(tvec![softmax_out])
        }
    }
}

impl TypedOp for ScaledMaskedSoftmax {
    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(!self.scale.is_zero()?);
        ensure!(inputs.len() == 2);
        let (input, mask) = (inputs[0], inputs[1]);
        ensure!(
            input.datum_type == mask.datum_type || mask.datum_type == bool::datum_type(),
            "mask must be same dtype as input or bool"
        );
        ensure!(
            input.rank() == mask.rank() || mask.datum_type == bool::datum_type(),
            "float mask must have same rank as input"
        );
        let dt = input.datum_type;
        let fact = dt.fact(input.shape.clone());
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        // Introduction: mask's uniform_tdim defines which positions matter for scores.
        let mask_fact = model.outlet_fact(node.inputs[1])?;
        if let Some(mask_expr) = &mask_fact.uniform_tdim {
            return Ok(Some(tvec![Some(mask_expr.clone()), None]));
        }
        // Bubbling: delegate to the natural blanket implementation.
        tract_nnef::tract_core::optim::propagate_roi::bubble_roi(model, node)
    }

    as_op!();
}

/// Search pattern => A = SOFTMAX(A * SCALE + MASK, AXIS=-1)
pub fn scaled_masked_softmax_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Softmax,
) -> TractResult<Option<TypedModelPatch>> {
    let rank = node.outputs[0].fact.rank();
    rule_if!(op.axes.as_slice() == [rank - 1]);

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;
    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Try boolean-mask pattern first: Softmax(Iff(bool_mask, [Mul(x,scale) | x], fill))
    if let Some(patch) = try_match_bool_iff_softmax(model, node, node_name, op, dt)? {
        return Ok(Some(patch));
    }

    // Identify Add operator (Mask)
    rule_if_some!(add_prev = model.previous_node(node));
    rule_if_some!(add_prev_op = add_prev.op_as::<TypedBinOp>());
    rule_if!(add_prev_op.0.is::<Add>());

    let mut in_add = model.previous_nodes(add_prev);
    rule_if!(in_add.len() == 2);

    in_add.reverse();
    let (left, right) = (in_add.pop().unwrap(), in_add.pop().unwrap());

    let (scale_node, mask_outlet) = if left.op_is::<TypedBinOp>() {
        (left, add_prev.inputs[1])
    } else {
        (right, add_prev.inputs[0])
    };

    rule_if_some!(scale_op = scale_node.op_as::<TypedBinOp>());
    rule_if!(scale_op.0.is::<Mul>());

    // Retrieve Scale
    let mul_consts = model.collect_const_inputs(scale_node);
    rule_if!(mul_consts.len() == 1);
    let scale = mul_consts[0].val().clone();

    rule_if!(scale.len() == 1);
    rule_if!(scale.datum_type() == dt);

    let mut patch = TypedModelPatch::default();
    let input = patch.taps(model, &scale_node.inputs)?[0];
    let mask = patch.taps(model, &[mask_outlet])?[0];

    let out = patch.wire_node(
        format!("{node_name}.scaled_masked_softmax"),
        ScaledMaskedSoftmax { scale, post_softmax_mask: false },
        &[input, mask],
    )?;

    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Pattern: Softmax(Iff(cond, A, B)) where exactly one of A/B is the fill (uniform/const)
/// and the other is the attention scores.
///
/// Two sub-cases handled:
///   - `Iff(mask,  scores, fill)` — cond=True means "valid, keep score"
///   - `Iff(~mask, fill,   scores)` — cond=True means "masked, replace with fill";
///     we look through a BitNot/Not predecessor to recover the non-negated mask.
///
/// Also detects a downstream post-softmax Iff of the form:
///   `Iff(~mask, 0, softmax_out)` and folds it into `post_softmax_mask=true`.
fn try_match_bool_iff_softmax(
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _op: &Softmax,
    dt: DatumType,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(iff_node = model.previous_node(node));
    rule_if!(iff_node.op_is::<Iff>());

    let cond_outlet = iff_node.inputs[0];
    let branch_t = iff_node.inputs[1]; // true branch
    let branch_f = iff_node.inputs[2]; // false branch

    rule_if!(model.outlet_fact(cond_outlet)?.datum_type == bool::datum_type());

    // Decide which branch is the fill (uniform value) and which is the scores.
    let t_is_fill = outlet_is_uniform(model, branch_t);
    let f_is_fill = outlet_is_uniform(model, branch_f);
    // Require exactly one side to be fill.
    rule_if!(t_is_fill ^ f_is_fill);

    let (scores_outlet, bool_mask_outlet) = if f_is_fill {
        // Iff(mask, scores, fill) — condition is the direct "attend" mask (True=valid).
        (branch_t, cond_outlet)
    } else {
        // Iff(~mask, fill, scores) — condition is negated; unwrap the Not/BitNot.
        let original_mask = unwrap_bool_not(model, cond_outlet);
        rule_if_some!(original_mask = original_mask);
        (branch_f, original_mask)
    };

    // Optionally unwrap Mul(x, scale); fall back to identity scale (1.0).
    let (scores_outlet, scale) = try_extract_scale(model, scores_outlet, dt).unwrap_or_else(|| {
        let one = tensor0(1f32).cast_to_dt(dt).unwrap().into_owned().into_arc_tensor();
        (scores_outlet, one)
    });

    // Detect optional downstream post-softmax Iff: Iff(~mask, 0, softmax_out).
    let post_mask_succ = try_detect_post_softmax_iff(model, node)?;

    let mut patch = TypedModelPatch::default();
    let scores = patch.tap_model(model, scores_outlet)?;
    let bool_mask = patch.tap_model(model, bool_mask_outlet)?;

    let out = patch.wire_node(
        format!("{node_name}.scaled_masked_softmax"),
        ScaledMaskedSoftmax { scale, post_softmax_mask: post_mask_succ.is_some() },
        &[scores, bool_mask],
    )?;

    // Shunt the Softmax node, and also the downstream post-softmax Iff if present.
    patch.shunt_outside(model, node.id.into(), out[0])?;
    if let Some(post_iff_outlet) = post_mask_succ {
        patch.shunt_outside(model, post_iff_outlet, out[0])?;
    }
    Ok(Some(patch))
}

/// Checks whether the single successor of `softmax_node` is a post-softmax masking Iff:
///   `Iff(bool_cond, fill=0, softmax_out)` or `Iff(bool_cond, softmax_out, fill=0)`
///
/// Returns the outlet id of that Iff node if the pattern matches, `None` otherwise.
fn try_detect_post_softmax_iff(
    model: &TypedModel,
    softmax_node: &TypedNode,
) -> TractResult<Option<OutletId>> {
    rule_if_some!(succ = model.single_succ(softmax_node.id)?);
    rule_if!(succ.op_is::<Iff>());

    let cond_outlet = succ.inputs[0];
    let branch_t = succ.inputs[1];
    let branch_f = succ.inputs[2];

    rule_if!(model.outlet_fact(cond_outlet)?.datum_type == bool::datum_type());

    // Exactly one branch must be uniform (the zero fill).
    let t_is_fill = outlet_is_uniform(model, branch_t);
    let f_is_fill = outlet_is_uniform(model, branch_f);
    rule_if!(t_is_fill ^ f_is_fill);

    // The non-fill branch must be the softmax output.
    let softmax_outlet = OutletId::new(softmax_node.id, 0);
    let data_branch = if f_is_fill { branch_t } else { branch_f };
    rule_if!(data_branch == softmax_outlet);

    Ok(Some(OutletId::new(succ.id, 0)))
}

/// Returns true if the outlet carries a uniform (all-same-value) tensor —
/// i.e. it is a constant or its `uniform` field is set.
fn outlet_is_uniform(model: &TypedModel, outlet: OutletId) -> bool {
    model.outlet_fact(outlet).map(|f| f.konst.is_some() || f.uniform.is_some()).unwrap_or(false)
}

/// Walk the graph upstream from `outlet`, passing through shape-only ops
/// (AddAxis / RemoveAxis / MultiBroadcastTo), looking for a logical/bitwise NOT.
/// If found, returns the input to that NOT (the non-negated bool wire).
/// Returns `None` if no such NOT is reachable.
fn unwrap_bool_not(model: &TypedModel, outlet: OutletId) -> Option<OutletId> {
    use tract_nnef::tract_core::ops::array::MultiBroadcastTo;
    use tract_nnef::tract_core::ops::change_axes::AxisOp;
    use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
    use tract_nnef::tract_core::ops::logic::{BitNot, Not};

    let node = model.node(outlet.node);

    // Direct Not/BitNot on bool
    if let Some(ew) = node.op_as::<ElementWiseOp>()
        && (ew.0.is::<Not>() || ew.0.is::<BitNot>())
    {
        return Some(node.inputs[0]);
    }

    // Look through shape-transparent ops (AddAxis, RemoveAxis, broadcast)
    if node.op_is::<AxisOp>() || node.op_is::<MultiBroadcastTo>() {
        return unwrap_bool_not(model, node.inputs[0]);
    }

    None
}

/// If the node at `outlet` is `Mul(x, const_scale)`, return `(x_outlet, scale_tensor)`.
fn try_extract_scale(
    model: &TypedModel,
    outlet: OutletId,
    dt: DatumType,
) -> Option<(OutletId, Arc<Tensor>)> {
    let node = model.node(outlet.node);
    let bin = node.op_as::<TypedBinOp>()?;
    if !bin.0.is::<Mul>() {
        return None;
    }
    let consts = model.collect_const_inputs(node);
    if consts.len() != 1 {
        return None;
    }
    let scale = consts[0].val().clone();
    if scale.len() != 1 || scale.datum_type() != dt {
        return None;
    }
    // The non-const input is the scores.
    let scores_outlet = node
        .inputs
        .iter()
        .copied()
        .find(|o| model.outlet_fact(*o).map(|f| f.konst.is_none()).unwrap_or(false))?;
    Some((scores_outlet, scale))
}
