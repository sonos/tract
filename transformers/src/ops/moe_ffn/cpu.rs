use super::*;

/// gpt-oss clamped SwiGLU, as an op so the expert subplans can express it:
///   gate = min(gate, limit); up = clamp(up, -limit, limit)
///   out  = (up + 1) * gate * sigmoid(alpha * gate)
#[derive(Clone, Debug, PartialEq)]
struct ClampedSwiGlu {
    alpha: f32,
    limit: f32,
}

impl Eq for ClampedSwiGlu {}

impl Op for ClampedSwiGlu {
    fn name(&self) -> StaticName {
        "ClampedSwiGlu".into()
    }

    op_as_typed_op!();
}

impl EvalOp for ClampedSwiGlu {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        let gate = inputs[0].cast_to::<f32>()?.into_owned();
        let up = inputs[1].cast_to::<f32>()?.into_owned();
        ensure!(
            gate.shape() == up.shape(),
            "ClampedSwiGlu gate/up shape mismatch: {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
        let mut output = Tensor::zero_dt(f32::datum_type(), gate.shape())?;
        let gate_ram = gate.try_as_plain_ram()?;
        let up_ram = up.try_as_plain_ram()?;
        let mut output_ram = output.try_as_plain_ram_mut()?;
        let gate = gate_ram.as_slice::<f32>()?;
        let up = up_ram.as_slice::<f32>()?;
        let output_slice = output_ram.as_slice_mut::<f32>()?;
        for ((out, &gate), &up) in output_slice.iter_mut().zip(gate).zip(up) {
            let gate = gate.min(self.limit);
            let up = up.clamp(-self.limit, self.limit);
            let glu = gate / (1.0 + (-self.alpha * gate).exp());
            *out = (up + 1.0) * glu;
        }
        Ok(tvec![output.into_tvalue()])
    }
}

impl TypedOp for ClampedSwiGlu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2, "ClampedSwiGlu expects gate and up inputs");
        ensure!(
            inputs[0].shape == inputs[1].shape,
            "ClampedSwiGlu gate/up shape mismatch: {:?} vs {:?}",
            inputs[0].shape,
            inputs[1].shape
        );
        Ok(tvec!(f32::datum_type().fact(inputs[0].shape.clone())))
    }

    as_op!();
}

pub(super) fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(silu())),
        // SwiGLU: the inner activation is silu, w3 provides the gate branch
        "swiglu" if has_w3 => Some(Box::new(silu())),
        "gelu" => Some(Box::new(gelu_approximate(false))),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}

pub(super) fn build_router_plan(
    wg: &Arc<Tensor>,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel { symbols: symbols.clone(), ..Default::default() };
    let n_sym = symbols.sym("moe_t");
    let dt = f32::datum_type();

    let wg_2d = if wg.rank() == 3 {
        wg.slice(0, 0, 1)?.into_shape(&[wg.shape()[1], wg.shape()[2]])?
    } else {
        (**wg).clone()
    };
    let wg_2d = wg_2d.cast_to::<f32>()?.into_owned();
    let d_model = wg_2d.shape()[1];

    let x = model.add_source("x", dt.fact([n_sym.to_dim(), d_model.to_dim()]))?;
    let wg_const = model.add_const("wg", wg_2d)?;
    let axes: AxesMapping = "ij,kj->ik".parse()?;
    let logits = model.wire_node("router_logits", EinSum::new(axes, dt), &[x, wg_const])?[0];

    model.select_output_outlets(&[logits])?;
    SimplePlan::new(model.into_optimized()?)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_expert_plan(
    w1: &Tensor,
    w2: &Tensor,
    w3: Option<&Tensor>,
    w1_bias: Option<&Tensor>,
    w3_bias: Option<&Tensor>,
    w2_bias: Option<&Tensor>,
    activation: &str,
    act_alpha: Option<f32>,
    act_limit: Option<f32>,
    expert_layout: ExpertLayout,
    preserve_block_quant: bool,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel { symbols: symbols.clone(), ..Default::default() };
    let n_sym = symbols.sym("moe_n");
    let dt = f32::datum_type();
    let expert_const = |t: &Tensor| -> TractResult<Tensor> {
        // Keep weights in their exported storage. Upcasting the float
        // projection to f32 doubles its bytes, and for a mixed-precision
        // export that projection is the bulk of the model (gpt-oss keeps w2
        // in f16, ~12.7GB of a 22GB export), so the copy costs both prepare
        // time and decode bandwidth. The matmul still accumulates in f32.
        if t.storage_as::<BlockQuantStorage>().is_some() || t.datum_type() == f16::datum_type() {
            Ok(t.clone())
        } else {
            Ok(t.cast_to::<f32>()?.into_owned())
        }
    };

    let d_model = match expert_layout {
        ExpertLayout::Canonical => w1.shape()[0],
        ExpertLayout::Linear => w1.shape()[1],
    };

    let x = model.add_source("x", dt.fact([n_sym.to_dim(), d_model.to_dim()]))?;
    let axes_canonical: AxesMapping = "ij,jk->ik".parse()?;
    let axes_linear: AxesMapping = "ij,kj->ik".parse()?;
    let axes_in = match expert_layout {
        ExpertLayout::Canonical => axes_canonical.clone(),
        ExpertLayout::Linear => axes_linear.clone(),
    };

    let add_bias = |model: &mut TypedModel,
                    name: &str,
                    wire: OutletId,
                    bias: &Tensor|
     -> TractResult<OutletId> {
        let bias = model.add_const(format!("{name}_const"), bias.cast_to::<f32>()?.into_owned())?;
        let bias = model.wire_node(format!("{name}_add_axis"), AxisOp::Add(0), &[bias])?[0];
        Ok(model.wire_node(name, add(), &[wire, bias])?[0])
    };

    // gpt-oss: biased projections and a clamped SwiGLU. Handled before the
    // fused block-quant shortcut, which only models the bias-free case.
    if act_limit.is_some() || w1_bias.is_some() || w3_bias.is_some() || w2_bias.is_some() {
        let w1_const = model.add_const("w1", expert_const(w1)?)?;
        let mut h =
            model.wire_node("w1_matmul", EinSum::new(axes_in.clone(), dt), &[x, w1_const])?[0];
        if let Some(b) = w1_bias {
            h = add_bias(&mut model, "w1_bias_add", h, b)?;
        }
        let h = if let Some(limit) = act_limit {
            let w3 = w3.context("clamped activation requires w3 (up branch)")?;
            let w3_const = model.add_const("w3", expert_const(w3)?)?;
            let mut up =
                model.wire_node("w3_matmul", EinSum::new(axes_in.clone(), dt), &[x, w3_const])?[0];
            if let Some(b) = w3_bias {
                up = add_bias(&mut model, "w3_bias_add", up, b)?;
            }
            model.wire_node(
                "clamped_swiglu",
                ClampedSwiGlu { alpha: act_alpha.unwrap_or(1.0), limit },
                &[h, up],
            )?[0]
        } else {
            let act_op = activation_op(activation, w3.is_some())
                .ok_or_else(|| format_err!("Unsupported activation: {activation}"))?;
            let h = model.wire_node("activation", act_op, &[h])?[0];
            if let Some(w3) = w3 {
                let w3_const = model.add_const("w3", expert_const(w3)?)?;
                let mut gate = model.wire_node(
                    "w3_matmul",
                    EinSum::new(axes_in.clone(), dt),
                    &[x, w3_const],
                )?[0];
                if let Some(b) = w3_bias {
                    gate = add_bias(&mut model, "w3_bias_add", gate, b)?;
                }
                model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
            } else {
                h
            }
        };
        let w2_const = model.add_const("w2", expert_const(w2)?)?;
        let axes_out = match expert_layout {
            ExpertLayout::Canonical => axes_canonical,
            ExpertLayout::Linear => axes_linear,
        };
        let mut y = model.wire_node("w2_matmul", EinSum::new(axes_out, dt), &[h, w2_const])?[0];
        if let Some(b) = w2_bias {
            y = add_bias(&mut model, "w2_bias_add", y, b)?;
        }
        model.select_output_outlets(&[y])?;
        return SimplePlan::new(model.into_optimized()?);
    }

    let act_op = activation_op(activation, w3.is_some())
        .ok_or_else(|| format_err!("Unsupported activation: {activation}"))?;

    let h = if preserve_block_quant
        && expert_layout == ExpertLayout::Linear
        && w1.storage_as::<BlockQuantStorage>().is_some()
        && w3.is_some_and(|w3| w3.storage_as::<BlockQuantStorage>().is_some())
    {
        let w3 = w3.context("w3 disappeared after fusion eligibility check")?;
        ensure!(
            w1.shape()[0] == w3.shape()[0],
            "fused w1/w3 input matmul requires matching hidden dimensions"
        );
        let fused_w1_w3 = concat_block_quant_rows(w1, w3)?;
        let hidden = w1.shape()[0];
        let fused_const = model.add_const("w1_w3", fused_w1_w3)?;
        let fused =
            model.wire_node("w1_w3_matmul", EinSum::new(axes_in.clone(), dt), &[x, fused_const])?
                [0];
        let h = model.wire_node("w1_slice", Slice::new(1, 0, hidden), &[fused])?[0];
        let gate =
            model.wire_node("w3_slice", Slice::new(1, hidden, hidden + w3.shape()[0]), &[fused])?
                [0];
        let h = model.wire_node("activation", act_op, &[h])?[0];
        model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
    } else {
        let w1_const = model.add_const("w1", expert_const(w1)?)?;
        let h = model.wire_node("w1_matmul", EinSum::new(axes_in.clone(), dt), &[x, w1_const])?[0];
        let h = model.wire_node("activation", act_op, &[h])?[0];

        if let Some(w3) = w3 {
            let w3_const = model.add_const("w3", expert_const(w3)?)?;
            let gate = model.wire_node("w3_matmul", EinSum::new(axes_in, dt), &[x, w3_const])?[0];
            model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
        } else {
            h
        }
    };

    let w2_const = model.add_const("w2", expert_const(w2)?)?;
    let axes_out = match expert_layout {
        ExpertLayout::Canonical => axes_canonical,
        ExpertLayout::Linear => axes_linear,
    };
    let y = model.wire_node("w2_matmul", EinSum::new(axes_out, dt), &[h, w2_const])?[0];

    model.select_output_outlets(&[y])?;
    SimplePlan::new(model.into_optimized()?)
}

#[derive(Clone, Debug)]
pub(super) struct Q40LinearExpertPlan {
    activation: String,
    gate_up: PreparedRoutedMatMul,
    down: PreparedRoutedMatMul,
    gate_up_dim: usize,
}

pub(super) fn q40_direct_activation_supported(activation: &str, has_w3: bool) -> bool {
    matches!(activation, "silu") || (has_w3 && matches!(activation, "swiglu"))
}

pub(super) fn build_q40_linear_expert_plan(
    w1: &Tensor,
    w2: &Tensor,
    w3: Option<&Tensor>,
    activation: &str,
    num_experts: usize,
) -> TractResult<Arc<Q40LinearExpertPlan>> {
    ensure!(q40_direct_activation_supported(activation, w3.is_some()));

    let sample_gate_up = {
        let w1_e = block_quant_group_tensor(w1, 0)?;
        if let Some(w3) = w3 {
            let w3_e = block_quant_group_tensor(w3, 0)?;
            concat_block_quant_rows(&w1_e, &w3_e)?
        } else {
            w1_e
        }
    };
    let sample_down = block_quant_group_tensor(w2, 0)?;
    let gate_up_dim = sample_gate_up.shape()[0];
    let hidden_dim = sample_down.shape()[1];
    ensure!(
        gate_up_dim == hidden_dim || gate_up_dim == hidden_dim * 2,
        "Q40 direct MoE expected gate/up dim {gate_up_dim} to match hidden {hidden_dim} or 2x hidden"
    );

    let mut gate_up_weights = Vec::with_capacity(num_experts);
    let mut down_weights = Vec::with_capacity(num_experts);
    gate_up_weights.push(sample_gate_up);
    down_weights.push(sample_down);
    for eid in 0..num_experts {
        if eid == 0 {
            continue;
        }
        let w1_e = block_quant_group_tensor(w1, eid)?;
        let gate_up_e = if let Some(w3) = w3 {
            let w3_e = block_quant_group_tensor(w3, eid)?;
            concat_block_quant_rows(&w1_e, &w3_e)?
        } else {
            w1_e
        };
        gate_up_weights.push(gate_up_e);

        let w2_e = block_quant_group_tensor(w2, eid)?;
        down_weights.push(w2_e);
    }

    let gate_up = build_block_quant_routed_matmul(gate_up_weights)?;
    let down = build_block_quant_routed_matmul(down_weights)?;

    Ok(Arc::new(Q40LinearExpertPlan {
        activation: activation.to_string(),
        gate_up,
        down,
        gate_up_dim,
    }))
}

fn ensure_f32_tensor_capacity(slot: &mut Option<Tensor>, len: usize) -> TractResult<&mut Tensor> {
    if slot.as_ref().is_none_or(|tensor| tensor.len() < len) {
        *slot = Some(Tensor::zero::<f32>(&[len])?);
    }
    Ok(slot.as_mut().unwrap())
}

fn profile_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

fn profile_elapsed(start: Option<Instant>) -> Duration {
    start.map(|start| start.elapsed()).unwrap_or(Duration::ZERO)
}

fn apply_silu_gate(
    hidden: &mut [f32],
    route_count: usize,
    d_hidden: usize,
    gate_up_dim: usize,
    has_w3: bool,
) {
    if has_w3 {
        for row in 0..route_count {
            let base = row * gate_up_dim;
            let gate_base = base + d_hidden;
            for h in 0..d_hidden {
                let v = hidden[base + h];
                hidden[base + h] = (v / (1.0 + (-v).exp())) * hidden[gate_base + h];
            }
        }
    } else {
        for row in 0..route_count {
            let base = row * gate_up_dim;
            for h in 0..d_hidden {
                let v = hidden[base + h];
                hidden[base + h] = v / (1.0 + (-v).exp());
            }
        }
    }
}

fn push_selected_routes(
    gate: &GateMode,
    full_logits: &[f32],
    selected: &[(usize, f32)],
    token: usize,
    expert_tokens: &mut [Vec<(usize, f32)>],
) {
    match gate {
        GateMode::Raw => {
            for &(eid, logit) in selected {
                expert_tokens[eid].push((token, logit));
            }
        }
        GateMode::Sigmoid => {
            for &(eid, logit) in selected {
                expert_tokens[eid].push((token, 1.0 / (1.0 + (-logit).exp())));
            }
        }
        GateMode::SoftmaxTopk => {
            let max = selected.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = selected.iter().map(|(_, l)| (*l - max).exp()).sum();
            for &(eid, logit) in selected {
                expert_tokens[eid].push((token, (logit - max).exp() / denom));
            }
        }
        GateMode::SoftmaxAll => {
            let max = full_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = full_logits.iter().map(|l| (*l - max).exp()).sum();
            for &(eid, logit) in selected {
                expert_tokens[eid].push((token, (logit - max).exp() / denom));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptMoeFfn {
    pub k: usize,
    pub gate: GateMode,
    pub num_experts: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub router_plan: Arc<TypedSimplePlan>,
    /// Router bias, added to the logits after `router_plan` (which only holds
    /// the `x @ wg.T` matmul). gpt-oss routers carry one.
    pub wg_bias: Option<Tensor>,
    pub expert_plans: Vec<Arc<TypedSimplePlan>>,
    pub(super) q40_linear_plan: Option<Arc<Q40LinearExpertPlan>>,
}

impl Hash for OptMoeFfn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        self.gate.hash(state);
        self.num_experts.hash(state);
        self.d_model.hash(state);
        self.d_hidden.hash(state);
        self.q40_linear_plan.is_some().hash(state);
        // Independently compiled plans are conservatively distinct; clones
        // share plan identity. Comparing geometry alone would merge weights.
        Arc::as_ptr(&self.router_plan).hash(state);
        for plan in &self.expert_plans {
            Arc::as_ptr(plan).hash(state);
        }
        self.q40_linear_plan.as_ref().map(Arc::as_ptr).hash(state);
        // Tensor equality equates signed zeros and NaN payloads, unlike its
        // bitwise hash. Bias is compared below but omitted from this hash.
    }
}

impl PartialEq for OptMoeFfn {
    fn eq(&self, other: &Self) -> bool {
        self.k == other.k
            && self.gate == other.gate
            && self.num_experts == other.num_experts
            && self.d_model == other.d_model
            && self.d_hidden == other.d_hidden
            && self.q40_linear_plan.is_some() == other.q40_linear_plan.is_some()
            && Arc::ptr_eq(&self.router_plan, &other.router_plan)
            && self.expert_plans.len() == other.expert_plans.len()
            && self.expert_plans.iter().zip(&other.expert_plans).all(|(a, b)| Arc::ptr_eq(a, b))
            && match (&self.q40_linear_plan, &other.q40_linear_plan) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
            && self.wg_bias == other.wg_bias
    }
}

impl Eq for OptMoeFfn {}

impl Op for OptMoeFfn {
    fn name(&self) -> StaticName {
        "OptMoeFfn".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for OptMoeFfn {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        let router_state = self.router_plan.spawn()?;
        let q40_state = self.q40_linear_plan.as_ref().map(|_| Q40LinearExpertState::default());
        Ok(Some(Box::new(OptMoeFfnState { op: self.clone(), router_state, q40_state })))
    }
}

#[derive(Default)]
struct Q40LinearExpertState {
    gate_up_state: PreparedRoutedMatMulState,
    down_state: PreparedRoutedMatMulState,
    hidden: Option<Tensor>,
    expert_out: Option<Tensor>,
    expert_tokens: Vec<Vec<(usize, f32)>>,
    route_groups: Vec<RoutedMatMulGroup>,
    shared_rhs_groups: Vec<(usize, usize)>,
    route_tokens: Vec<usize>,
    route_weights: Vec<f32>,
    combine_scale: Option<Tensor>,
    scores: Vec<(usize, f32)>,
}

impl fmt::Debug for Q40LinearExpertState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Q40LinearExpertState")
            .field("gate_up_state", &self.gate_up_state)
            .field("down_state", &self.down_state)
            .field("hidden_len", &self.hidden.as_ref().map(|t| t.len()).unwrap_or(0))
            .field("expert_out_len", &self.expert_out.as_ref().map(|t| t.len()).unwrap_or(0))
            .field("expert_tokens", &self.expert_tokens.len())
            .field("route_groups", &self.route_groups.len())
            .field("route_count", &self.route_tokens.len())
            .finish()
    }
}

impl Clone for Q40LinearExpertState {
    fn clone(&self) -> Self {
        Q40LinearExpertState::default()
    }
}

impl Q40LinearExpertState {
    fn ensure_expert_tokens(&mut self, num_experts: usize) {
        if self.expert_tokens.len() != num_experts {
            self.expert_tokens = vec![Vec::new(); num_experts];
        } else {
            self.expert_tokens.iter_mut().for_each(Vec::clear);
        }
    }

    fn set_combine_scale(&mut self, value: f32) -> TractResult<()> {
        if self.combine_scale.is_none() {
            self.combine_scale = Some(Tensor::zero_dt(f32::datum_type(), &[])?);
        }
        let scale = self.combine_scale.as_mut().unwrap();
        let mut plain = scale.try_as_plain_ram_mut()?;
        plain.as_slice_mut::<f32>()?[0] = value;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn eval(
        &mut self,
        op: &OptMoeFfn,
        plan: &Q40LinearExpertPlan,
        x: ArrayView2<f32>,
        router_logits: ArrayView2<f32>,
        x_ndim: usize,
        x_orig_shape: &[usize],
        dt: DatumType,
        router_elapsed: Duration,
        profile: bool,
    ) -> TractResult<TVec<TValue>> {
        let total_start = profile_start(profile);
        let topk_start = profile_start(profile);

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        ensure!(
            d_model == op.d_model,
            "Q40 direct MoE input dim {} does not match op dim {}",
            d_model,
            op.d_model
        );
        ensure!(
            plan.gate_up_dim == op.d_hidden || plan.gate_up_dim == op.d_hidden * 2,
            "Q40 direct MoE gate/up dim {} does not match hidden {}",
            plan.gate_up_dim,
            op.d_hidden
        );
        let has_w3 = plan.gate_up_dim == op.d_hidden * 2;
        ensure!(
            q40_direct_activation_supported(&plan.activation, has_w3),
            "unsupported Q40 direct activation {}",
            plan.activation
        );

        self.ensure_expert_tokens(op.num_experts);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full = row.as_slice().context("router row is not contiguous")?;
            self.scores.clear();
            self.scores.extend(row.iter().enumerate().map(|(eid, &score)| (eid, score)));
            select_routes(&mut self.scores, op.k);
            push_selected_routes(&op.gate, full, &self.scores, t, &mut self.expert_tokens);
        }
        let topk_elapsed = profile_elapsed(topk_start);

        let output_shape: TVec<usize> = if x_ndim == 3 {
            tvec!(x_orig_shape[0], x_orig_shape[1], d_model)
        } else {
            tvec!(t_tokens, d_model)
        };
        let mut output_tensor = Tensor::zero_dt(f32::datum_type(), &output_shape)?;

        let item_size = f32::datum_type().size_of() as isize;
        let x_base = x.as_ptr();
        let x_row_stride_bytes = x.strides()[0] * item_size;
        let x_k_stride_bytes = x.strides()[1] * item_size;

        let mut gate_up_elapsed = Duration::ZERO;
        let mut activation_elapsed = Duration::ZERO;
        let mut down_elapsed = Duration::ZERO;
        let mut scatter_elapsed = Duration::ZERO;
        let mut active_experts = 0usize;
        let mut routed_rows = 0usize;

        self.route_groups.clear();
        self.shared_rhs_groups.clear();
        self.route_tokens.clear();
        self.route_weights.clear();
        for (eid, tokens) in self.expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            active_experts += 1;
            let output_row_offset = routed_rows;
            let route_count = tokens.len();
            routed_rows += route_count;

            self.route_tokens.extend(tokens.iter().map(|&(token, _)| token));
            self.route_weights.extend(tokens.iter().map(|&(_, weight)| weight));

            let rows = if route_count == 1 {
                RoutedInputRows::single(
                    x_base,
                    x_row_stride_bytes * tokens[0].0 as isize,
                    x_k_stride_bytes,
                )
            } else {
                let row_offsets: Vec<isize> =
                    tokens.iter().map(|&(t, _)| x_row_stride_bytes * t as isize).collect();
                RoutedInputRows::explicit(x_base, row_offsets, x_k_stride_bytes)
            };
            self.route_groups.push(RoutedMatMulGroup { group: eid, rows, output_row_offset });
            if t_tokens == 1 && route_count == 1 {
                self.shared_rhs_groups.push((eid, output_row_offset));
            }
        }

        if routed_rows > 0 {
            let hidden =
                ensure_f32_tensor_capacity(&mut self.hidden, routed_rows * plan.gate_up_dim)?;

            let gate_up_start = profile_start(profile);
            if self.shared_rhs_groups.len() == self.route_groups.len() {
                let rhs = pack_prepared_routed_matmul_rhs(
                    &plan.gate_up,
                    RoutedInputRows::single(x_base, 0, x_k_stride_bytes),
                )?;
                run_prepared_routed_matmul_many_same_rhs(
                    &plan.gate_up,
                    &self.shared_rhs_groups,
                    &*rhs,
                    1,
                    hidden,
                    &mut self.gate_up_state,
                )?;
            } else {
                run_prepared_routed_matmul_many(
                    &plan.gate_up,
                    &self.route_groups,
                    hidden,
                    &mut self.gate_up_state,
                )?;
            }
            gate_up_elapsed += profile_elapsed(gate_up_start);

            let activation_start = profile_start(profile);
            {
                let mut hidden_plain = hidden.try_as_plain_ram_mut()?;
                let hidden_slice = hidden_plain.as_slice_mut::<f32>()?;
                apply_silu_gate(
                    &mut hidden_slice[..routed_rows * plan.gate_up_dim],
                    routed_rows,
                    op.d_hidden,
                    plan.gate_up_dim,
                    has_w3,
                );
            }
            activation_elapsed += profile_elapsed(activation_start);

            let hidden_plain = hidden.try_as_plain_ram()?;
            let hidden_slice = hidden_plain.as_slice::<f32>()?;
            let hidden_base = hidden_slice.as_ptr();
            let hidden_row_stride_bytes = plan.gate_up_dim as isize * item_size;

            for route_group_index in 0..self.route_groups.len() {
                let group_id = self.route_groups[route_group_index].group;
                let output_row_offset = self.route_groups[route_group_index].output_row_offset;
                let route_count = self.route_groups[route_group_index].rows.len();
                if route_count == 1 {
                    let route = output_row_offset;
                    self.set_combine_scale(self.route_weights[route])?;
                    let scale = self.combine_scale.as_ref().unwrap();
                    let down_start = profile_start(profile);
                    run_prepared_routed_matmul_accumulate_one(
                        &plan.down,
                        group_id,
                        RoutedInputRows::single(
                            hidden_base,
                            hidden_row_stride_bytes * route as isize,
                            item_size,
                        ),
                        &mut output_tensor,
                        self.route_tokens[route],
                        scale,
                        &mut self.down_state,
                    )?;
                    down_elapsed += profile_elapsed(down_start);
                } else {
                    let expert_out =
                        ensure_f32_tensor_capacity(&mut self.expert_out, route_count * d_model)?;
                    let down_start = profile_start(profile);
                    run_prepared_routed_matmul(
                        &plan.down,
                        group_id,
                        RoutedInputRows::regular(
                            hidden_base,
                            hidden_row_stride_bytes * output_row_offset as isize,
                            route_count,
                            hidden_row_stride_bytes,
                            item_size,
                        ),
                        expert_out,
                        &mut self.down_state,
                    )?;
                    down_elapsed += profile_elapsed(down_start);

                    let scatter_start = profile_start(profile);
                    let y_plain = expert_out.try_as_plain_ram()?;
                    let y = y_plain.as_slice::<f32>()?;
                    let mut output_plain = output_tensor.try_as_plain_ram_mut()?;
                    let output = output_plain.as_slice_mut::<f32>()?;
                    for route in 0..route_count {
                        let flat_route = output_row_offset + route;
                        let y_row = &y[route * d_model..(route + 1) * d_model];
                        let token = self.route_tokens[flat_route];
                        let weight = self.route_weights[flat_route];
                        let out_row = &mut output[token * d_model..(token + 1) * d_model];
                        for (o, y) in out_row.iter_mut().zip(y_row) {
                            *o += weight * *y;
                        }
                    }
                    scatter_elapsed += profile_elapsed(scatter_start);
                }
            }
        }

        let output_tensor = output_tensor.cast_to_dt(dt)?.into_owned();
        if profile {
            eprintln!(
                "OptMoeFfn(Q40 direct) tokens={t_tokens} routes={routed_rows} experts={active_experts} router={router_elapsed:?} topk={topk_elapsed:?} gate_up={gate_up_elapsed:?} act={activation_elapsed:?} down={down_elapsed:?} scatter={scatter_elapsed:?} total={:?}",
                profile_elapsed(total_start)
            );
        }
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

#[derive(Clone)]
/// Only the router keeps a long-lived state. Expert plans are spawned per eval
/// so they can run on the rayon pool; they are stateless matmuls, so there is
/// nothing to carry between calls anyway.
struct OptMoeFfnState {
    op: OptMoeFfn,
    router_state: TypedSimpleState,
    q40_state: Option<Q40LinearExpertState>,
}

impl fmt::Debug for OptMoeFfnState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptMoeFfnState")
            .field("op", &self.op)
            .field("router_state", &self.router_state)
            .field("q40_state", &self.q40_state)
            .finish()
    }
}

impl OpState for OptMoeFfnState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let profile = std::env::var_os("TRACT_MOE_PROFILE").is_some();
        let total_start = profile_start(profile);
        ensure!(inputs.len() == 1, "OptMoeFfn expects one input");
        let x_input = &inputs[0];
        ensure!(matches!(x_input.rank(), 2 | 3), "OptMoeFfn input must have rank 2 or 3");
        ensure!(
            matches!(x_input.datum_type(), DatumType::F16 | DatumType::F32),
            "OptMoeFfn input must be floating point"
        );
        ensure!(
            x_input.shape()[x_input.rank() - 1] == self.op.d_model,
            "OptMoeFfn input feature dimension disagrees with its compiled weights"
        );
        let dt = x_input.datum_type();
        let x_f32 = x_input.cast_to::<f32>()?;
        let x_view = x_f32.to_plain_array_view::<f32>()?;
        let x_ndim = x_view.ndim();
        let x_orig_shape: Vec<usize> = x_view.shape().to_vec();

        let x: ArrayView2<f32> = if x_ndim == 3 {
            x_view
                .into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?
                .into_dimensionality()?
        } else {
            x_view.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let x_2d_tensor = x_f32.clone().into_owned().into_shape(&[t_tokens, d_model])?;

        let router_start = profile_start(profile);
        let router_result = self.router_state.run(tvec![x_2d_tensor.into_tvalue()])?;
        let router_elapsed = profile_elapsed(router_start);
        let router_logits_f32 = router_result[0].cast_to::<f32>()?;
        let mut router_logits_t = router_logits_f32.into_owned();
        if let Some(bias) = self.op.wg_bias.clone() {
            let num_experts = self.op.num_experts;
            let bias = bias.try_as_plain_ram()?.as_slice::<f32>()?.to_vec();
            let mut logits_ram = router_logits_t.try_as_plain_ram_mut()?;
            let logits = logits_ram.as_slice_mut::<f32>()?;
            for row in logits.chunks_mut(num_experts) {
                row.iter_mut().zip(&bias).for_each(|(v, &b)| *v += b);
            }
        }
        let router_logits: ArrayView2<f32> =
            router_logits_t.to_plain_array_view::<f32>()?.into_dimensionality()?;

        if let Some(plan) = self.op.q40_linear_plan.clone() {
            let q40_state = self
                .q40_state
                .as_mut()
                .context("OptMoeFfn has a Q40 plan but no Q40 runtime state")?;
            return q40_state.eval(
                &self.op,
                &plan,
                x,
                router_logits,
                x_ndim,
                &x_orig_shape,
                dt,
                router_elapsed,
                profile,
            );
        }

        let op = &self.op;
        let topk_start = profile_start(profile);
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            select_routes(&mut scores, op.k);

            let gate_weights = op.gate.weights(&full, &scores);
            assignments
                .push(scores.iter().zip(gate_weights).map(|((eid, _), gw)| (*eid, gw)).collect());
        }
        let topk_elapsed = profile_elapsed(topk_start);

        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); op.num_experts];
        for (t, token_experts) in assignments.iter().enumerate() {
            for &(eid, gw) in token_experts {
                expert_tokens[eid].push((t, gw));
            }
        }

        // Each active expert gathers its own token rows and runs its own plan,
        // so the experts run concurrently. This matters most during prefill:
        // with T tokens and top-k routing a prompt pass lights up nearly every
        // expert in every layer, and each expert's matmuls are too narrow (a
        // handful of token rows) to fill the machine on their own.
        //
        // Plans are spawned per call rather than reusing `expert_states`:
        // `OpState` is deliberately not `Send`, and expert sub-models are
        // stateless matmuls so nothing is carried between calls anyway.
        let run_expert =
            |plan: &Arc<TypedSimplePlan>, tokens: &[(usize, f32)]| -> TractResult<Option<Tensor>> {
                if tokens.is_empty() {
                    return Ok(None);
                }
                let n = tokens.len();
                let mut x_batch = Tensor::zero_dt(f32::datum_type(), &[n, d_model])?;
                {
                    let mut x_batch_plain = x_batch.try_as_plain_ram_mut()?;
                    let x_batch_slice = x_batch_plain.as_slice_mut::<f32>()?;
                    for (i, &(t, _)) in tokens.iter().enumerate() {
                        let src = x.row(t);
                        x_batch_slice[i * d_model..(i + 1) * d_model]
                            .copy_from_slice(src.as_slice().unwrap());
                    }
                }
                let y_expert = plan.run(tvec![x_batch.into_tvalue()])?;
                Ok(Some(y_expert[0].clone().into_tensor()))
            };

        #[cfg(not(target_family = "wasm"))]
        let expert_outputs = {
            use rayon::prelude::*;
            self.op
                .expert_plans
                .par_iter()
                .zip(expert_tokens.par_iter())
                .map(|(plan, tokens)| run_expert(plan, tokens))
                .collect::<TractResult<Vec<_>>>()?
        };
        #[cfg(target_family = "wasm")]
        let expert_outputs = self
            .op
            .expert_plans
            .iter()
            .zip(expert_tokens.iter())
            .map(|(plan, tokens)| run_expert(plan, tokens))
            .collect::<TractResult<Vec<_>>>()?;

        // Scatter serially and in fixed expert order: a token routed to k
        // experts has k contributions landing on the same output row, so the
        // reduction order must stay deterministic.
        let mut output = Array2::<f32>::zeros((t_tokens, d_model));
        for (y_expert, tokens) in expert_outputs.iter().zip(expert_tokens.iter()) {
            let Some(y_expert) = y_expert else { continue };
            let y_view: ArrayView2<f32> =
                y_expert.to_plain_array_view::<f32>()?.into_dimensionality()?;
            scatter_add_weighted(&mut output, &y_view, tokens);
        }

        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(dt)?.into_owned();
        if profile {
            eprintln!(
                "OptMoeFfn(plan) tokens={t_tokens} router={router_elapsed:?} topk={topk_elapsed:?} total={:?}",
                profile_elapsed(total_start)
            );
        }
        Ok(tvec![output_tensor.into_tvalue()])
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        // The router sub-state and the Q40 expert scratch are reused,
        // op-level working buffers, not data keyed per lane, so there is
        // nothing here that needs discarding when handed to another stream.
        Ok(())
    }
}

impl TypedOp for OptMoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());
        Ok(tvec!(output_fact))
    }

    as_op!();
}
