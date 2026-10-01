use super::activation::activation_op;
use super::q40::{Q40LinearExpertPlan, Q40LinearExpertState};
use super::{ExpertLayout, GateMode, concat_block_quant_rows, scatter_add_weighted, select_routes};
use crate::ops::routed_matmul::f32_input;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tract_ndarray::{Array2, ArrayView2};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::OpState;
use tract_nnef::tract_core::ops::array::Slice;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::math::{add, mul};
use tract_nnef::tract_core::ops::nn::ClampedSwiGlu;
use tract_nnef::tract_core::tract_linalg::block_quant::BlockQuantStorage;

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

pub(super) fn profile_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

pub(super) fn profile_elapsed(start: Option<Instant>) -> Duration {
    start.map(|start| start.elapsed()).unwrap_or(Duration::ZERO)
}

#[derive(Clone, Debug)]
pub struct OptMoeFfn {
    pub(super) k: usize,
    pub(super) gate: GateMode,
    pub(super) num_experts: usize,
    pub(super) d_model: usize,
    pub(super) d_hidden: usize,
    pub(super) router_plan: Arc<TypedSimplePlan>,
    /// Router bias, added to the logits after `router_plan` (which only holds
    /// the `x @ wg.T` matmul). gpt-oss routers carry one.
    pub(super) wg_bias: Option<Tensor>,
    pub(super) experts: CpuExpertPlan,
}

/// Mutually exclusive prepared CPU strategies; neither owns session scratch.
#[derive(Clone, Debug)]
pub(super) enum CpuExpertPlan {
    PerExpert(Vec<Arc<TypedSimplePlan>>),
    PackedQ40(Arc<Q40LinearExpertPlan>),
}

impl PartialEq for CpuExpertPlan {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PerExpert(a), Self::PerExpert(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| Arc::ptr_eq(a, b))
            }
            (Self::PackedQ40(a), Self::PackedQ40(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for CpuExpertPlan {}

impl Hash for CpuExpertPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::PerExpert(plans) => {
                plans.len().hash(state);
                for plan in plans {
                    Arc::as_ptr(plan).hash(state);
                }
            }
            Self::PackedQ40(plan) => Arc::as_ptr(plan).hash(state),
        }
    }
}

impl OptMoeFfn {
    #[cfg(test)]
    pub(super) fn uses_direct_q40(&self) -> bool {
        matches!(self.experts, CpuExpertPlan::PackedQ40(_))
    }
}

impl Hash for OptMoeFfn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        self.gate.hash(state);
        self.num_experts.hash(state);
        self.d_model.hash(state);
        self.d_hidden.hash(state);
        // Independently compiled plans are conservatively distinct; clones
        // share plan identity. Comparing geometry alone would merge weights.
        Arc::as_ptr(&self.router_plan).hash(state);
        self.experts.hash(state);
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
            && Arc::ptr_eq(&self.router_plan, &other.router_plan)
            && self.experts == other.experts
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
        let experts = match &self.experts {
            CpuExpertPlan::PerExpert(_) => CpuExpertState::PerExpert,
            CpuExpertPlan::PackedQ40(_) => CpuExpertState::PackedQ40(Box::default()),
        };
        Ok(Some(Box::new(OptMoeFfnState { router_state, experts })))
    }
}

#[derive(Clone, Debug)]
enum CpuExpertState {
    PerExpert,
    PackedQ40(Box<Q40LinearExpertState>),
}

/// Execution workspace, not model history: retains the router's runnable
/// state and Q40 allocation/kernel scratch between calls. Expert inputs are
/// evaluation-local; cloning the Q40 workspace starts with fresh scratch.
#[derive(Clone, Debug)]
struct OptMoeFfnState {
    router_state: TypedSimpleState,
    experts: CpuExpertState,
}

impl OpState for OptMoeFfnState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op =
            op.downcast_ref::<OptMoeFfn>().context("OptMoeFfn workspace requires OptMoeFfn")?;
        ensure!(
            Arc::ptr_eq(self.router_state.plan(), &op.router_plan),
            "OptMoeFfn router workspace belongs to a different plan"
        );
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
            x_input.shape()[x_input.rank() - 1] == op.d_model,
            "OptMoeFfn input feature dimension disagrees with its compiled weights"
        );
        let dt = x_input.datum_type();
        let x_f32 = f32_input(x_input)?;
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
        let x_2d_tensor = x_f32.as_ref().clone().into_shape(&[t_tokens, d_model])?;

        let router_start = profile_start(profile);
        let router_result = self.router_state.run(tvec![x_2d_tensor.into_tvalue()])?;
        let router_elapsed = profile_elapsed(router_start);
        let router_logits_f32 = router_result[0].cast_to::<f32>()?;
        let mut router_logits_t = router_logits_f32.into_owned();
        if let Some(bias) = op.wg_bias.clone() {
            let num_experts = op.num_experts;
            let bias = bias.try_as_plain_ram()?.as_slice::<f32>()?.to_vec();
            let mut logits_ram = router_logits_t.try_as_plain_ram_mut()?;
            let logits = logits_ram.as_slice_mut::<f32>()?;
            for row in logits.chunks_mut(num_experts) {
                row.iter_mut().zip(&bias).for_each(|(v, &b)| *v += b);
            }
        }
        let router_logits: ArrayView2<f32> =
            router_logits_t.to_plain_array_view::<f32>()?.into_dimensionality()?;

        let expert_plans = match (&op.experts, &mut self.experts) {
            (CpuExpertPlan::PackedQ40(plan), CpuExpertState::PackedQ40(state)) => {
                return state.eval(
                    op,
                    plan,
                    &x_f32,
                    router_logits,
                    x_ndim,
                    &x_orig_shape,
                    dt,
                    router_elapsed,
                    profile,
                );
            }
            (CpuExpertPlan::PerExpert(plans), CpuExpertState::PerExpert) => plans,
            _ => bail!("OptMoeFfn expert workspace does not match its plan"),
        };

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
            expert_plans
                .par_iter()
                .zip(expert_tokens.par_iter())
                .map(|(plan, tokens)| run_expert(plan, tokens))
                .collect::<TractResult<Vec<_>>>()?
        };
        #[cfg(target_family = "wasm")]
        let expert_outputs = expert_plans
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
