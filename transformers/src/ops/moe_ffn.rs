//! Inference-only, token-routed mixture-of-experts feed-forward blocks.
//!
//! # Interchange surface
//!
//! `tract_moe_ffn` is the only NNEF primitive registered here. Its required
//! tensors are `x`, `wg`, `w1`, and `w2`; optional tensors are `w3`, `wg_bias`,
//! `w1_bias`, `w3_bias`, and `w2_bias`. Input tokens have shape `[T,D]` or
//! `[B,S,D]`, and the router is `[E,D]` (or `[1,E,D]`). Canonical expert layout
//! uses `w1/w3: [E,D,H]`, `w2: [E,H,D]`; linear layout uses their per-expert
//! transposes. Router bias is `[E]`, gate/up biases `[E,H]`, down bias `[E,D]`.
//! Facts use f16 or f32; expert tensors can also carry block-quantized storage.
//! The output has the shape and datum type of `x`; accumulation is in f32.
//!
//! `k` selects between one and E experts. `gate` is `softmax_topk` (normalize
//! selected logits), `softmax_all` (normalize all logits before selection),
//! `sigmoid`, or `raw`. Equal scores select lower expert indices first, including
//! signed zero; this is a deterministic local convention, not a promise of
//! matching another framework's unspecified tie order.
//! `activation` accepts `silu`, `gelu`, `relu`, or `swiglu` (requires `w3`).
//! With `w3`, the activated gate is multiplied by the up projection. Optional
//! `act_limit` instead selects clamped SwiGLU, with `act_alpha` defaulting to 1;
//! alpha alone does not change an unclamped activation.
//!
//! These combinations express the routed FFN patterns used by Mixtral and
//! Qwen MoE (normalized top-k), OLMoE (full softmax), Llama4 (sigmoid routing),
//! and gpt-oss (biases and clamped SwiGLU). This is not a full model importer:
//! shared experts, model-specific router preprocessing, and training losses
//! must be represented outside this primitive.
//!
//! # Lowering and serialization
//!
//! [`MoeFfn`] is the typed, serializable operator and reference evaluator.
//! It has no special declutter pass. CPU codegen captures constant weights and
//! biases into [`OptMoeFfn`] router/expert plans. Supported linear block-quant
//! experts have a direct packed CPU path; other supported constant variants
//! use per-expert plans. Plain, canonical, bias-free dynamic weights with
//! concrete weight shapes lower to
//! [`RouteTopK`], [`RoutedMatMul`], activation and [`RoutedCombine`]. Dynamic
//! variants outside that lowering stay on the reference evaluator, preserving
//! runtime cross-projection checks for unresolved weight dimensions. Symbolic
//! token counts do not prevent lowering.
//! [`RoutedQ40MatMul`] is a routed linear block-quant execution helper.
//! These helper operators are execution forms, not additional NNEF primitives.
//! Serialize before codegen: packed [`OptMoeFfn`] plans deliberately reject
//! export rather than retaining a second copy of their source weights.
//!
//! `nnef` contains interchange handling, `routing` the routed operators,
//! `activation` selects core activation operators, `lowering` selects CPU execution
//! forms, and `cpu` owns prepared plans and their state. Public operator paths
//! remain available through this module for backend integrations.

mod activation;
mod cpu;
mod lowering;
mod nnef;
mod routing;
#[cfg(test)]
mod tests;

use activation::activation_op;
pub use cpu::OptMoeFfn;
pub use nnef::register;
pub use routing::{RouteTopK, RoutedCombine, RoutedInputMode, RoutedMatMul, RoutedQ40MatMul};

use tract_ndarray::{Array2, ArrayView2, ArrayViewD, Axis, s};
use tract_nnef::internal::*;
use tract_nnef::tract_core::tract_linalg::block_quant::{
    BlockQuantFact, BlockQuantStorage, block_quant_slice,
};

/// How router logits are turned into the top-k gate weights.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum GateMode {
    /// softmax over the top-k logits (Mixtral, Qwen norm_topk_prob, gpt-oss).
    SoftmaxTopk,
    /// softmax over ALL experts, gather the top-k, no renormalization (OLMoE).
    SoftmaxAll,
    /// per-expert sigmoid of the top-k logits, no normalization (Llama4).
    Sigmoid,
    /// raw top-k logits as weights.
    Raw,
}

impl GateMode {
    fn parse(s: &str) -> TractResult<Self> {
        match s {
            "softmax_topk" => Ok(GateMode::SoftmaxTopk),
            "softmax_all" => Ok(GateMode::SoftmaxAll),
            "sigmoid" => Ok(GateMode::Sigmoid),
            "raw" => Ok(GateMode::Raw),
            other => bail!("unknown tract_moe_ffn gate mode {other:?}"),
        }
    }

    /// Weights for the selected experts given the full logits row and the
    /// (expert_id, logit) pairs chosen by top-k.
    fn weights(&self, full_logits: &[f32], selected: &[(usize, f32)]) -> Vec<f32> {
        match self {
            GateMode::Raw => selected.iter().map(|(_, l)| *l).collect(),
            GateMode::Sigmoid => selected.iter().map(|(_, l)| 1.0 / (1.0 + (-l).exp())).collect(),
            GateMode::SoftmaxTopk => {
                let m = selected.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = selected.iter().map(|(_, l)| (l - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            }
            GateMode::SoftmaxAll => {
                let m = full_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = full_logits.iter().map(|l| (l - m).exp()).sum();
                selected.iter().map(|(_, l)| (l - m).exp() / denom).collect()
            }
        }
    }
}

fn select_routes(scores: &mut Vec<(usize, f32)>, k: usize) {
    scores.sort_unstable_by(|a, b| {
        let order = if a.1 == b.1 { std::cmp::Ordering::Equal } else { b.1.total_cmp(&a.1) };
        order.then_with(|| a.0.cmp(&b.0))
    });
    scores.truncate(k);
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ExpertLayout {
    /// Existing tract_moe_ffn layout: w1/w3 `[E,D,H]`, w2 `[E,H,D]`.
    Canonical,
    /// Natural nn.Linear layout: w1/w3 `[E,H,D]`, w2 `[E,D,H]`.
    Linear,
}

impl ExpertLayout {
    fn parse(s: &str) -> TractResult<Self> {
        match s {
            "canonical" => Ok(ExpertLayout::Canonical),
            "linear" => Ok(ExpertLayout::Linear),
            other => bail!("unknown tract_moe_ffn expert layout {other:?}"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MoeFfn {
    pub k: usize,
    pub activation: String,
    pub gate: GateMode,
    pub has_w3: bool,
    pub has_wg_bias: bool,
    pub has_w1_bias: bool,
    pub has_w3_bias: bool,
    pub has_w2_bias: bool,
    // f32 has no Hash/Eq, so the clamped-SwiGLU params are stored as bit
    // patterns; None means "use the plain activation_op path".
    pub act_alpha_bits: Option<u32>,
    pub act_limit_bits: Option<u32>,
    pub expert_layout: ExpertLayout,
}

/// Resolved positions of the optional inputs within the `inputs` slice,
/// derived from the `has_*` flags in NNEF optional-input order.
struct MoeInputIdx {
    w3: Option<usize>,
    wg_bias: Option<usize>,
    w1_bias: Option<usize>,
    w3_bias: Option<usize>,
    w2_bias: Option<usize>,
}

impl MoeFfn {
    /// Bias-free, plain-activation op for unit tests (kept future-proof
    /// against new optional fields).
    #[cfg(test)]
    fn basic(k: usize, activation: &str, gate: GateMode, has_w3: bool) -> Self {
        Self::basic_with_layout(k, activation, gate, has_w3, ExpertLayout::Canonical)
    }

    #[cfg(test)]
    fn basic_with_layout(
        k: usize,
        activation: &str,
        gate: GateMode,
        has_w3: bool,
        expert_layout: ExpertLayout,
    ) -> Self {
        MoeFfn {
            k,
            activation: activation.to_string(),
            gate,
            has_w3,
            has_wg_bias: false,
            has_w1_bias: false,
            has_w3_bias: false,
            has_w2_bias: false,
            act_alpha_bits: None,
            act_limit_bits: None,
            expert_layout,
        }
    }

    fn is_clamped_act(&self) -> bool {
        self.act_limit_bits.is_some()
    }

    fn has_any_bias(&self) -> bool {
        self.has_wg_bias || self.has_w1_bias || self.has_w3_bias || self.has_w2_bias
    }

    fn act_alpha(&self) -> f32 {
        self.act_alpha_bits.map(f32::from_bits).unwrap_or(1.0)
    }

    fn act_limit(&self) -> f32 {
        self.act_limit_bits.map(f32::from_bits).unwrap_or(f32::INFINITY)
    }

    fn validate_facts(&self, inputs: &[&TypedFact]) -> TractResult<()> {
        let idx = self.input_idx();
        let optional = [idx.w3, idx.wg_bias, idx.w1_bias, idx.w3_bias, idx.w2_bias];
        ensure!(
            inputs.len() == 4 + optional.iter().flatten().count(),
            "MoeFfn input count disagrees with its optional inputs"
        );
        ensure!(self.k > 0, "MoeFfn k must be positive");
        ensure!(!self.has_w3_bias || self.has_w3, "MoeFfn w3_bias requires w3");
        ensure!(!self.is_clamped_act() || self.has_w3, "clamped activation requires w3");
        ensure!(
            activation_op(&self.activation, self.has_w3).is_some(),
            "Unsupported MoE activation {}",
            self.activation
        );
        for bits in [self.act_alpha_bits, self.act_limit_bits].into_iter().flatten() {
            ensure!(f32::from_bits(bits).is_finite(), "MoE activation parameters must be finite");
        }
        if let Some(limit) = self.act_limit_bits {
            ensure!(f32::from_bits(limit) > 0.0, "MoE activation limit must be positive");
        }
        for fact in inputs {
            ensure!(
                matches!(fact.datum_type, DatumType::F16 | DatumType::F32),
                "Unsupported MoE datum type {:?}",
                fact.datum_type
            );
        }
        ensure!(matches!(inputs[0].rank(), 2 | 3), "MoE x must be [T,D] or [B,S,D]");
        ensure!(matches!(inputs[1].rank(), 2 | 3), "MoE wg must be [E,D] or [1,E,D]");
        ensure!(
            inputs[2].rank() == 3 && inputs[3].rank() == 3,
            "MoE expert weights must have rank 3"
        );
        let agree = |a: &TDim, b: &TDim| -> TractResult<()> {
            ensure!(
                (a.clone() - b.clone()).to_i64().map_or(true, |d| d == 0),
                "MoE dimensions disagree: {a} versus {b}"
            );
            Ok(())
        };
        let wg = &inputs[1].shape;
        if inputs[1].rank() == 3 {
            agree(&wg[0], &1.to_dim())?;
        }
        let e = &wg[wg.rank() - 2];
        let d = &inputs[0].shape[inputs[0].rank() - 1];
        agree(d, &wg[wg.rank() - 1])?;
        agree(e, &inputs[2].shape[0])?;
        agree(e, &inputs[3].shape[0])?;
        if let Ok(e) = e.to_i64() {
            ensure!(e > 0 && self.k <= e as usize, "MoE k exceeds expert count {e}");
        }
        let (d_axis, h_axis) = match self.expert_layout {
            ExpertLayout::Canonical => (1, 2),
            ExpertLayout::Linear => (2, 1),
        };
        let h = &inputs[2].shape[h_axis];
        agree(d, &inputs[2].shape[d_axis])?;
        agree(d, &inputs[3].shape[h_axis])?;
        agree(h, &inputs[3].shape[d_axis])?;
        for dim in [d, h] {
            ensure!(
                dim.to_i64().map_or(true, |v| v > 0),
                "MoE feature dimensions must be positive"
            );
        }
        if let Some(w3) = idx.w3 {
            ensure!(inputs[w3].rank() == 3, "MoE w3 must have rank 3");
            for (a, b) in inputs[w3].shape.iter().zip(inputs[2].shape.iter()) {
                agree(a, b)?;
            }
        }
        if let Some(bias) = idx.wg_bias {
            ensure!(inputs[bias].rank() == 1, "MoE router bias must be [E]");
            agree(&inputs[bias].shape[0], e)?;
        }
        for (bias, width) in [(idx.w1_bias, h), (idx.w3_bias, h), (idx.w2_bias, d)] {
            if let Some(bias) = bias {
                ensure!(inputs[bias].rank() == 2, "MoE expert bias must be [E,width]");
                agree(&inputs[bias].shape[0], e)?;
                agree(&inputs[bias].shape[1], width)?;
            }
        }
        Ok(())
    }

    fn input_idx(&self) -> MoeInputIdx {
        let mut i = 4;
        let mut next = |present: bool| {
            if present {
                let v = i;
                i += 1;
                Some(v)
            } else {
                None
            }
        };
        MoeInputIdx {
            w3: next(self.has_w3),
            wg_bias: next(self.has_wg_bias),
            w1_bias: next(self.has_w1_bias),
            w3_bias: next(self.has_w3_bias),
            w2_bias: next(self.has_w2_bias),
        }
    }

    /// Inner activation of the reference paths, matching `activation_op`:
    /// silu for "silu"/"swiglu" (w3 provides the gate branch), the pow-3
    /// tanh-approximate gelu, and relu. The clamped-SwiGLU (gpt-oss) case is
    /// handled by the callers before reaching this.
    fn apply_reference_activation(&self, h: &mut Array2<f32>) -> TractResult<()> {
        match self.activation.as_str() {
            "silu" | "swiglu" => {
                h.iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp()));
            }
            "gelu" => {
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                h.iter_mut().for_each(|v| {
                    let x = *v;
                    *v = 0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(3))));
                });
            }
            "relu" => {
                h.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            other => bail!("MoeFfn reference eval: unsupported activation {other:?}"),
        }
        Ok(())
    }

    fn can_eval_lazy_block_quant(&self, inputs: &[TValue], idx: &MoeInputIdx) -> bool {
        !self.has_w1_bias
            && !self.has_w3_bias
            && !self.has_w2_bias
            && inputs[2].storage_as::<BlockQuantStorage>().is_some()
            && inputs[3].storage_as::<BlockQuantStorage>().is_some()
            && idx.w3.is_none_or(|w3| inputs[w3].storage_as::<BlockQuantStorage>().is_some())
    }

    fn eval_lazy_block_quant(
        &self,
        inputs: TVec<TValue>,
        idx: MoeInputIdx,
        orig_dt: DatumType,
    ) -> TractResult<TVec<TValue>> {
        let x_t = inputs[0].cast_to::<f32>()?.into_owned();
        let wg_t = inputs[1].cast_to::<f32>()?.into_owned();
        let wg_bias_t = idx
            .wg_bias
            .map(|i| inputs[i].cast_to::<f32>().map(|cow| cow.into_owned()))
            .transpose()?;
        let x = x_t.to_plain_array_view::<f32>()?;
        let wg_raw = wg_t.to_plain_array_view::<f32>()?;
        let wg_bias = wg_bias_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;

        let wg: ArrayView2<f32> = if wg_raw.ndim() == 3 {
            wg_raw.index_axis(Axis(0), 0).into_dimensionality()?
        } else {
            wg_raw.into_dimensionality()?
        };

        let x_ndim = x.ndim();
        let x_orig_shape: Vec<usize> = x.shape().to_vec();
        let x: ArrayView2<f32> = if x_ndim == 3 {
            x.into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?
                .into_dimensionality()?
        } else {
            x.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let num_experts = wg.shape()[0];

        let mut router_logits: Array2<f32> = x.dot(&wg.t());
        if let Some(ref b) = wg_bias {
            let bias = b.as_slice().context("wg_bias not contiguous")?;
            for mut row in router_logits.rows_mut() {
                row.iter_mut().zip(bias).for_each(|(v, &bv)| *v += bv);
            }
        }

        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            select_routes(&mut scores, self.k);

            let gate_weights = self.gate.weights(&full, &scores);
            for ((eid, _), gw) in scores.iter().zip(gate_weights) {
                expert_tokens[*eid].push((t, gw));
            }
        }

        let mut output = Array2::<f32>::zeros((t_tokens, d_model));
        for (eid, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let mut x_batch = Array2::<f32>::zeros((tokens.len(), d_model));
            for (i, &(t, _)) in tokens.iter().enumerate() {
                x_batch.row_mut(i).assign(&x.row(t));
            }

            let w1_e_t = block_quant_group_as_2d(&inputs[2], eid)?;
            let w1_e: ArrayView2<f32> =
                w1_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let mut h: Array2<f32> = match self.expert_layout {
                ExpertLayout::Canonical => x_batch.dot(&w1_e),
                ExpertLayout::Linear => x_batch.dot(&w1_e.t()),
            };

            if self.is_clamped_act() {
                let w3_ix = idx.w3.context("clamped activation requires w3 (up branch)")?;
                let w3_e_t = block_quant_group_as_2d(&inputs[w3_ix], eid)?;
                let w3_e: ArrayView2<f32> =
                    w3_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
                let up: Array2<f32> = match self.expert_layout {
                    ExpertLayout::Canonical => x_batch.dot(&w3_e),
                    ExpertLayout::Linear => x_batch.dot(&w3_e.t()),
                };
                let alpha = self.act_alpha();
                let limit = self.act_limit();
                h.iter_mut().zip(up.iter()).for_each(|(g, &u)| {
                    let gate = g.min(limit);
                    let up = u.clamp(-limit, limit);
                    let glu = gate / (1.0 + (-alpha * gate).exp());
                    *g = (up + 1.0) * glu;
                });
            } else if let Some(w3_ix) = idx.w3 {
                let w3_e_t = block_quant_group_as_2d(&inputs[w3_ix], eid)?;
                let w3_e: ArrayView2<f32> =
                    w3_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
                let gate: Array2<f32> = match self.expert_layout {
                    ExpertLayout::Canonical => x_batch.dot(&w3_e),
                    ExpertLayout::Linear => x_batch.dot(&w3_e.t()),
                };
                self.apply_reference_activation(&mut h)?;
                h.iter_mut().zip(gate.iter()).for_each(|(h_val, &g_val)| *h_val *= g_val);
            } else {
                self.apply_reference_activation(&mut h)?;
            }

            let w2_e_t = block_quant_group_as_2d(&inputs[3], eid)?;
            let w2_e: ArrayView2<f32> =
                w2_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let y_expert: Array2<f32> = match self.expert_layout {
                ExpertLayout::Canonical => h.dot(&w2_e),
                ExpertLayout::Linear => h.dot(&w2_e.t()),
            };
            scatter_add_weighted(&mut output, &y_expert.view(), tokens);
        }

        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(orig_dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

impl Op for MoeFfn {
    fn name(&self) -> StaticName {
        "MoeFfn".to_string().into()
    }
    op_as_typed_op!();
}

fn token_count_dim(shape: &ShapeFact) -> TDim {
    let dims = shape.to_tvec();
    dims[..dims.len() - 1].iter().cloned().product()
}

fn as_2d_tokens<'a>(view: ArrayViewD<'a, f32>) -> TractResult<ArrayView2<'a, f32>> {
    let shape = view.shape().to_vec();
    match shape.len() {
        2 => Ok(view.into_dimensionality()?),
        3 => Ok(view
            .into_shape_with_order((shape[0] * shape[1], shape[2]))?
            .into_dimensionality()?),
        _ => bail!("expected rank 2 or 3 token tensor, got {:?}", shape),
    }
}

fn router_weights_as_2d(view: ArrayViewD<'_, f32>) -> TractResult<ArrayView2<'_, f32>> {
    match view.ndim() {
        2 => Ok(view.into_dimensionality()?),
        3 => {
            let shape = view.shape().to_vec();
            ensure!(
                shape[0] == 1,
                "rank-3 router weights must have leading dimension 1, got {:?}",
                shape
            );
            Ok(view.into_shape_with_order((shape[1], shape[2]))?.into_dimensionality()?)
        }
        _ => bail!("expected router weights rank 2 or 3, got {:?}", view.shape()),
    }
}

fn plain_i64_slice<'a>(tensor: &'a Tensor, label: &str) -> TractResult<&'a [i64]> {
    tensor
        .try_as_plain_ram()
        .with_context(|| format!("{label} is not a plain tensor"))?
        .as_slice::<i64>()
        .with_context(|| format!("{label} is not an i64 slice"))
}

fn plain_f32_slice<'a>(tensor: &'a Tensor, label: &str) -> TractResult<&'a [f32]> {
    tensor
        .try_as_plain_ram()
        .with_context(|| format!("{label} is not a plain tensor"))?
        .as_slice::<f32>()
        .with_context(|| format!("{label} is not an f32 slice"))
}

/// Scatter-add expert outputs into `output`, weighted by the gate weights.
/// Accumulation is in f32: the router and experts run in f32 (matching
/// PyTorch's CPU f32-upcast of f16 matmuls), so the whole op is f32 internally
/// and only the final output is cast back to the model dtype.
fn scatter_add_weighted(
    output: &mut Array2<f32>,
    y_view: &ArrayView2<f32>,
    tokens: &[(usize, f32)],
) {
    for (i, &(t, gw)) in tokens.iter().enumerate() {
        let y_row = y_view.row(i);
        let mut out_row = output.row_mut(t);
        out_row.scaled_add(gw, &y_row);
    }
}

/// Add a per-expert bias row (`bias_full[eid]`, length H or D) to every row of
/// `arr` ([n, H] or [n, D]).
fn add_expert_bias(
    arr: &mut Array2<f32>,
    bias_full: &ArrayViewD<f32>,
    eid: usize,
) -> TractResult<()> {
    let brow = bias_full.index_axis(Axis(0), eid);
    let bslice = brow.as_slice().context("expert bias row not contiguous")?;
    for mut row in arr.rows_mut() {
        row.iter_mut().zip(bslice).for_each(|(v, &bv)| *v += bv);
    }
    Ok(())
}

fn block_quant_group_as_2d(input: &Tensor, group: usize) -> TractResult<Tensor> {
    let shape = input.shape();
    ensure!(shape.len() >= 2, "block-quant expert tensor must have rank >= 2");
    let k = *shape.last().context("block-quant tensor has no last axis")?;
    let m = shape[shape.len() - 2];
    let groups = if shape.len() > 2 { shape[..shape.len() - 2].iter().product() } else { 1 };
    ensure!(group < groups, "block-quant group {group} is out of range for {groups} groups");
    let bqs = input.try_storage_as::<BlockQuantStorage>()?;
    let slice = block_quant_slice(bqs.value(), bqs.format(), m, k, group);
    bqs.format().dequant_f32(slice)?.into_shape(&[m, k])
}

fn block_quant_group_tensor(input: &Tensor, group: usize) -> TractResult<Tensor> {
    let shape = input.shape();
    ensure!(shape.len() >= 2, "block-quant expert tensor must have rank >= 2");
    let k = *shape.last().context("block-quant tensor has no last axis")?;
    let m = shape[shape.len() - 2];
    let groups = if shape.len() > 2 { shape[..shape.len() - 2].iter().product() } else { 1 };
    ensure!(group < groups, "block-quant group {group} is out of range for {groups} groups");
    let bqs = input.try_storage_as::<BlockQuantStorage>()?;
    let exotic_fact = input.exotic_fact()?.context("block-quant tensor has no exotic fact")?;
    let bqf = exotic_fact
        .downcast_ref::<BlockQuantFact>()
        .context("block-quant tensor has no BlockQuantFact")?;
    let slice = block_quant_slice(bqs.value(), bqs.format(), m, k, group);
    let storage =
        BlockQuantStorage::new(bqf.format.clone(), m, k, Arc::new(Blob::from_bytes(slice)?))?;
    Ok(storage.into_tensor_with_shape(input.datum_type(), &[m, k]))
}

/// Transposes each expert matrix of a rank-3 block-quant tensor `[G,A,B]` into
/// `[G,B,A]`, requantizing along the new innermost axis. This maps the
/// canonical expert layout (w1/w3 `[E,D,H]`, w2 `[E,H,D]`) to the linear layout
/// (w1/w3 `[E,H,D]`, w2 `[E,D,H]`) the routed Q40 kernels consume. Because Q4_0
/// blocks run along the innermost axis, the transpose regroups values under
/// new block scales: the result carries requantization noise of the same
/// order as the original quantization (it is NOT a pure byte shuffle).
pub fn transpose_block_quant_experts(input: &Tensor) -> TractResult<Tensor> {
    let shape = input.shape();
    ensure!(shape.len() == 3, "expected a rank-3 expert tensor, got {shape:?}");
    let (g, a, b) = (shape[0], shape[1], shape[2]);
    let bqs = input.try_storage_as::<BlockQuantStorage>()?;
    let exotic_fact = input.exotic_fact()?.context("block-quant tensor has no exotic fact")?;
    let bqf = exotic_fact
        .downcast_ref::<BlockQuantFact>()
        .context("block-quant tensor has no BlockQuantFact")?;
    let format = bqf.format.clone();
    let block_len = format.block_len();
    ensure!(
        a % block_len == 0 && b % block_len == 0,
        "cannot transpose block-quant experts {shape:?}: both matrix axes must be multiples of the block length {block_len}"
    );

    let out_row_bytes = a / block_len * format.block_bytes();
    let out_group_bytes = b * out_row_bytes;
    let mut out = vec![0u8; g * out_group_bytes];
    let mut transposed = vec![0f32; a * b];
    for group in 0..g {
        let q = block_quant_slice(bqs.value(), &*format, a, b, group);
        let deq_t = format.dequant_f32(q)?;
        let deq = deq_t.try_as_plain_ram()?.as_slice::<f32>()?;
        for i in 0..a {
            for j in 0..b {
                transposed[j * a + i] = deq[i * b + j];
            }
        }
        let qt = format.quant_f32(&transposed)?;
        out[group * out_group_bytes..][..out_group_bytes].copy_from_slice(&qt);
    }
    let storage =
        BlockQuantStorage::new(format, g * b, a, Arc::new(Blob::from_bytes_alignment(&out, 128)?))?;
    Ok(storage.into_tensor_with_shape(input.datum_type(), &[g, b, a]))
}

fn concat_block_quant_rows(lhs: &Tensor, rhs: &Tensor) -> TractResult<Tensor> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    ensure!(lhs_shape.len() == 2, "lhs block-quant tensor must be rank 2");
    ensure!(rhs_shape.len() == 2, "rhs block-quant tensor must be rank 2");
    ensure!(
        lhs_shape[1] == rhs_shape[1],
        "block-quant tensors must have the same K axis to concatenate rows"
    );
    let lhs_bqs = lhs.try_storage_as::<BlockQuantStorage>()?;
    let rhs_bqs = rhs.try_storage_as::<BlockQuantStorage>()?;
    ensure!(
        lhs_bqs.format().dyn_eq(rhs_bqs.format()),
        "block-quant tensors must use the same format to concatenate rows"
    );
    let lhs_exotic_fact =
        lhs.exotic_fact()?.context("lhs block-quant tensor has no exotic fact")?;
    let lhs_bqf = lhs_exotic_fact
        .downcast_ref::<BlockQuantFact>()
        .context("lhs block-quant tensor has no BlockQuantFact")?;

    let mut bytes = Vec::with_capacity(lhs_bqs.value().len() + rhs_bqs.value().len());
    bytes.extend_from_slice(lhs_bqs.value().as_bytes());
    bytes.extend_from_slice(rhs_bqs.value().as_bytes());

    let m = lhs_shape[0] + rhs_shape[0];
    let k = lhs_shape[1];
    let storage =
        BlockQuantStorage::new(lhs_bqf.format.clone(), m, k, Arc::new(Blob::from_bytes(&bytes)?))?;
    Ok(storage.into_tensor_with_shape(lhs.datum_type(), &[m, k]))
}

impl EvalOp for MoeFfn {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let facts: Vec<_> = inputs.iter().map(|t| t.datum_type().fact(t.shape())).collect();
        self.validate_facts(&facts.iter().collect::<Vec<_>>())?;
        // inputs: x [T,D], wg [E,D] or [1,E,D], w1 [E,D,H], w2 [E,H,D],
        // then optionally w3 [E,D,H], wg_bias [E], w1_bias [E,H],
        // w3_bias [E,H], w2_bias [E,D] (order set in deser_moe_ffn).
        let idx = self.input_idx();
        let orig_dt = inputs[0].datum_type();
        if self.can_eval_lazy_block_quant(&inputs, &idx) {
            return self.eval_lazy_block_quant(inputs, idx, orig_dt);
        }

        // The reference path computes in f32. Cast every input up front; for
        // block-quant tensors, explicitly dequantize each leading group.
        let cast = |i: usize| -> TractResult<Tensor> {
            let input = &inputs[i];
            if input.is_plain() {
                return Ok(input.cast_to::<f32>()?.into_owned());
            }
            let s = input.shape();
            let k = *s.last().context("block-quant tensor has no last axis")?;
            let num_groups: usize = if s.len() > 2 { s[..s.len() - 2].iter().product() } else { 1 };
            let m_per_group: usize = if s.len() >= 2 { s[s.len() - 2] } else { 1 };
            let bqs = input.try_storage_as::<BlockQuantStorage>()?;
            let mut unpacked = (0..num_groups)
                .map(|g| {
                    let slice = block_quant_slice(bqs.value(), bqs.format(), m_per_group, k, g);
                    bqs.format().dequant_f32(slice)
                })
                .collect::<TractResult<Vec<_>>>()?;
            unpacked.iter_mut().try_for_each(|t| t.insert_axis(0))?;
            let stacked = if unpacked.len() > 1 {
                Tensor::stack_tensors(0, &unpacked)?
            } else {
                unpacked.into_iter().next().context("block-quant tensor has no groups to unpack")?
            };
            stacked.into_shape(s)
        };
        let x_t = cast(0)?;
        let wg_t = cast(1)?;
        let w1_t = cast(2)?;
        let w2_t = cast(3)?;
        let w3_t = idx.w3.map(cast).transpose()?;
        let wg_bias_t = idx.wg_bias.map(cast).transpose()?;
        let w1_bias_t = idx.w1_bias.map(cast).transpose()?;
        let w3_bias_t = idx.w3_bias.map(cast).transpose()?;
        let w2_bias_t = idx.w2_bias.map(cast).transpose()?;
        let x = x_t.to_plain_array_view::<f32>()?;
        let wg_raw = wg_t.to_plain_array_view::<f32>()?;
        let w1 = w1_t.to_plain_array_view::<f32>()?;
        let w2 = w2_t.to_plain_array_view::<f32>()?;
        let w3 = w3_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;
        let wg_bias = wg_bias_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;
        let w1_bias = w1_bias_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;
        let w3_bias = w3_bias_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;
        let w2_bias = w2_bias_t.as_ref().map(|t| t.to_plain_array_view::<f32>()).transpose()?;

        // Normalize wg to 2D [E, D] (may be [1, E, D] from unsqueeze)
        let wg: ArrayView2<f32> = if wg_raw.ndim() == 3 {
            wg_raw.index_axis(Axis(0), 0).into_dimensionality()?
        } else {
            wg_raw.into_dimensionality()?
        };

        // Normalize x to 2D [T, D] (may be [B, S, D] with B=1)
        let x_ndim = x.ndim();
        let x_orig_shape: Vec<usize> = x.shape().to_vec();
        let x: ArrayView2<f32> = if x_ndim == 3 {
            x.into_shape_with_order((x_orig_shape[0] * x_orig_shape[1], x_orig_shape[2]))?
                .into_dimensionality()?
        } else {
            x.into_dimensionality()?
        };

        let t_tokens = x.shape()[0];
        let d_model = x.shape()[1];
        let num_experts = wg.shape()[0];
        let _d_hidden = match self.expert_layout {
            ExpertLayout::Canonical => w1.shape()[2],
            ExpertLayout::Linear => w1.shape()[1],
        };

        // ---- Step 1: Router ----
        // logits = x @ wg.T  [T, D] @ [D, E] -> [T, E]  (+ optional bias [E])
        let mut router_logits: Array2<f32> = x.dot(&wg.t());
        if let Some(ref b) = wg_bias {
            let bias = b.as_slice().context("wg_bias not contiguous")?;
            for mut row in router_logits.rows_mut() {
                row.iter_mut().zip(bias).for_each(|(v, &bv)| *v += bv);
            }
        }

        // ---- Step 2: Top-k selection + gate weights per token ----
        // assignments[token] = Vec<(expert_id, gate_weight)>
        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            select_routes(&mut scores, self.k);

            let gate_weights = self.gate.weights(&full, &scores);

            assignments
                .push(scores.iter().zip(gate_weights).map(|((eid, _), gw)| (*eid, gw)).collect());
        }

        // ---- Step 3: Group tokens per expert ----
        // expert_tokens[eid] = Vec<(token_idx, gate_weight)>
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
        for (t, token_experts) in assignments.iter().enumerate() {
            for &(eid, gw) in token_experts {
                expert_tokens[eid].push((t, gw));
            }
        }

        // ---- Step 4: Batched expert computation (conditional!) ----
        let mut output = Array2::<f32>::zeros((t_tokens, d_model));

        for (eid, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue; // Skip unused experts entirely
            }
            let n = tokens.len();

            // Gather: build x_batch [n, D] from selected tokens
            let mut x_batch = Array2::<f32>::zeros((n, d_model));
            for (i, &(t, _)) in tokens.iter().enumerate() {
                x_batch.row_mut(i).assign(&x.row(t));
            }

            // Expert weight slices for this expert.
            // canonical: w1/w3 [D,H], w2 [H,D]
            // linear:    w1/w3 [H,D], w2 [D,H]
            let w1_e = w1.slice(s![eid, .., ..]);
            let w2_e = w2.slice(s![eid, .., ..]);

            // gate branch -> [n, H]
            let mut h: Array2<f32> = match self.expert_layout {
                ExpertLayout::Canonical => x_batch.dot(&w1_e),
                ExpertLayout::Linear => x_batch.dot(&w1_e.t()),
            };
            if let Some(ref b) = w1_bias {
                add_expert_bias(&mut h, b, eid)?;
            }

            if self.is_clamped_act() {
                // gpt-oss clamped SwiGLU: gate = w1 branch, up = w3 branch.
                //   gate = clamp(gate, max=limit)
                //   up   = clamp(up, -limit, limit)
                //   glu  = gate * sigmoid(alpha * gate)
                //   h    = (up + 1) * glu
                let w3 = w3.as_ref().context("clamped activation requires w3 (up branch)")?;
                let w3_e = w3.slice(s![eid, .., ..]);
                let mut up: Array2<f32> = match self.expert_layout {
                    ExpertLayout::Canonical => x_batch.dot(&w3_e),
                    ExpertLayout::Linear => x_batch.dot(&w3_e.t()),
                };
                if let Some(ref b) = w3_bias {
                    add_expert_bias(&mut up, b, eid)?;
                }
                let alpha = self.act_alpha();
                let limit = self.act_limit();
                h.iter_mut().zip(up.iter()).for_each(|(g, &u)| {
                    let gate = g.min(limit);
                    let up = u.clamp(-limit, limit);
                    let glu = gate / (1.0 + (-alpha * gate).exp());
                    *g = (up + 1.0) * glu;
                });
            } else if let Some(ref w3) = w3 {
                // GLU: h = act(h) * (x_batch @ w3_e + w3_bias)
                let w3_e = w3.slice(s![eid, .., ..]);
                let mut gate: Array2<f32> = match self.expert_layout {
                    ExpertLayout::Canonical => x_batch.dot(&w3_e),
                    ExpertLayout::Linear => x_batch.dot(&w3_e.t()),
                };
                if let Some(ref b) = w3_bias {
                    add_expert_bias(&mut gate, b, eid)?;
                }
                self.apply_reference_activation(&mut h)?;
                h.iter_mut().zip(gate.iter()).for_each(|(h_val, &g_val)| *h_val *= g_val);
            } else {
                // Plain declared activation
                self.apply_reference_activation(&mut h)?;
            }

            // y_expert -> [n, D]  (BLAS-backed GEMM)
            let mut y_expert: Array2<f32> = match self.expert_layout {
                ExpertLayout::Canonical => h.dot(&w2_e),
                ExpertLayout::Linear => h.dot(&w2_e.t()),
            };
            if let Some(ref b) = w2_bias {
                add_expert_bias(&mut y_expert, b, eid)?;
            }

            // ---- Step 5: Scatter-add weighted results back ----
            scatter_add_weighted(&mut output, &y_expert.view(), tokens);
        }

        // Restore original rank if input was 3D, and the original dtype.
        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(orig_dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}
