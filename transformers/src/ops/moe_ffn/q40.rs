//! Packed Q40 expert plans and reusable CPU execution scratch.
use super::cpu::{OptMoeFfn, profile_elapsed, profile_start};
use super::{
    GateMode, as_2d_tokens, block_quant_group_tensor, concat_block_quant_rows, select_routes,
};
use crate::ops::routed_matmul::{
    PreparedRoutedMatMul, PreparedRoutedMatMulState, RoutedInputRows, RoutedMatMulGroup,
    build_block_quant_routed_matmul, pack_prepared_routed_matmul_rhs, run_prepared_routed_matmul,
    run_prepared_routed_matmul_accumulate_one, run_prepared_routed_matmul_many,
    run_prepared_routed_matmul_many_same_rhs,
};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tract_ndarray::ArrayView2;
use tract_nnef::internal::*;

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

#[derive(Default)]
pub(super) struct Q40LinearExpertState {
    gate_up_state: PreparedRoutedMatMulState,
    down_state: PreparedRoutedMatMulState,
    hidden: Option<Arc<Tensor>>,
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
    pub(super) fn eval(
        &mut self,
        op: &OptMoeFfn,
        plan: &Q40LinearExpertPlan,
        x_tensor: &Arc<Tensor>,
        router_logits: ArrayView2<f32>,
        x_ndim: usize,
        x_orig_shape: &[usize],
        dt: DatumType,
        router_elapsed: Duration,
        profile: bool,
    ) -> TractResult<TVec<TValue>> {
        let total_start = profile_start(profile);
        let topk_start = profile_start(profile);

        let x = as_2d_tokens(x_tensor.to_plain_array_view::<f32>()?)?;
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

        let x_row_stride = usize::try_from(x.strides()[0])?;
        let x_k_stride = usize::try_from(x.strides()[1])?;

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
                RoutedInputRows::single(x_row_stride * tokens[0].0, x_k_stride)
            } else {
                let row_offsets: Vec<usize> =
                    tokens.iter().map(|&(t, _)| x_row_stride * t).collect();
                RoutedInputRows::explicit(row_offsets, x_k_stride)
            };
            self.route_groups.push(RoutedMatMulGroup { group: eid, rows, output_row_offset });
            if t_tokens == 1 && route_count == 1 {
                self.shared_rhs_groups.push((eid, output_row_offset));
            }
        }

        if routed_rows > 0 {
            let hidden_len = routed_rows * plan.gate_up_dim;
            if self.hidden.as_ref().is_none_or(|tensor| tensor.len() < hidden_len) {
                self.hidden = Some(Arc::new(Tensor::zero::<f32>(&[hidden_len])?));
            }
            let hidden = Arc::get_mut(self.hidden.as_mut().unwrap())
                .context("Q40 hidden scratch is still shared across evaluations")?;

            let gate_up_start = profile_start(profile);
            if self.shared_rhs_groups.len() == self.route_groups.len() {
                let rhs = pack_prepared_routed_matmul_rhs(
                    &plan.gate_up,
                    x_tensor,
                    RoutedInputRows::single(0, x_k_stride),
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
                    x_tensor,
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

            let hidden_source = self.hidden.as_ref().unwrap().clone();
            let hidden_row_stride = plan.gate_up_dim;

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
                        &hidden_source,
                        RoutedInputRows::single(hidden_row_stride * route, 1),
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
                        &hidden_source,
                        RoutedInputRows::regular(
                            hidden_row_stride * output_row_offset,
                            route_count,
                            hidden_row_stride,
                            1,
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
