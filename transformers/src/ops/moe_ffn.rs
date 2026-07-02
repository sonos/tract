use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tract_ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewD, Axis, s};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::mul;
use tract_nnef::tract_core::ops::{FrozenOpState, OpState, OpStateFreeze};
use tract_nnef::tract_core::tract_linalg::block_quant::{BlockQuantStorage, block_quant_slice};
use tract_nnef::tract_core::tract_linalg::mmm::{
    AsInputValue, FusedSpec, MMMInputFormat, MMMInputValue, PackedExoticFact,
};
use tract_nnef::tract_core::tract_linalg::pack::{PackedFormat, PackingWriter};

use super::gelu_approximate::gelu_approximate;
use super::silu::silu;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_moe_ffn",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("wg"),
            TypeName::Scalar.tensor().named("w1"),
            TypeName::Scalar.tensor().named("w2"),
            TypeName::Scalar.tensor().named("w3"),
            // Optional biases (e.g. gpt-oss): router bias [E], gate bias
            // (w1) [E, H], up bias (w3) [E, H], down bias (w2) [E, D].
            TypeName::Scalar.tensor().named("wg_bias"),
            TypeName::Scalar.tensor().named("w1_bias"),
            TypeName::Scalar.tensor().named("w3_bias"),
            TypeName::Scalar.tensor().named("w2_bias"),
            TypeName::Integer.named("k"),
            TypeName::String.named("activation"),
            // How router logits become the top-k gate weights:
            //   softmax_topk: softmax over the top-k logits (Mixtral, Qwen w/
            //                 norm_topk_prob, gpt-oss)
            //   softmax_all:  softmax over ALL experts, gather top-k, no
            //                 renormalization (OLMoE, Qwen w/o norm_topk_prob)
            //   sigmoid:      per-expert sigmoid of the top-k logits (Llama4)
            //   raw:          raw top-k logits as weights
            TypeName::String.named("gate"),
            // Optional clamped-SwiGLU params (gpt-oss): when act_limit is set
            // the activation becomes gate.clamp(max=limit) /
            // up.clamp(+-limit) / glu = gate*sigmoid(alpha*gate) /
            // out = (up + 1) * glu.
            TypeName::Scalar.named("act_alpha"),
            TypeName::Scalar.named("act_limit"),
        ],
        // single output: tract is inference-only, so router_logits (a
        // training-time load-balancing-loss signal) is computed internally for
        // expert selection but never surfaced as an output.
        &[("output", TypeName::Scalar.tensor())],
        deser_moe_ffn,
    );
}

fn deser_moe_ffn(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let wg = invocation.named_arg_as(builder, "wg")?;
    let w1 = invocation.named_arg_as(builder, "w1")?;
    let w2 = invocation.named_arg_as(builder, "w2")?;
    let w3: Option<OutletId> = invocation.get_named_arg_as(builder, "w3")?;
    let wg_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "wg_bias")?;
    let w1_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w1_bias")?;
    let w3_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w3_bias")?;
    let w2_bias: Option<OutletId> = invocation.get_named_arg_as(builder, "w2_bias")?;
    let k: i64 = invocation.named_arg_as(builder, "k")?;
    let activation: String = invocation.named_arg_as(builder, "activation")?;
    let gate_str: String = invocation.named_arg_as(builder, "gate")?;
    let gate = GateMode::parse(&gate_str)?;
    let act_alpha: Option<f32> = invocation.get_named_arg_as(builder, "act_alpha")?;
    let act_limit: Option<f32> = invocation.get_named_arg_as(builder, "act_limit")?;

    // Inputs are pushed in a fixed order so eval can recover their positions
    // from the has_* flags: x, wg, w1, w2, [w3], [wg_bias], [w1_bias],
    // [w3_bias], [w2_bias].
    let mut inputs = vec![x, wg, w1, w2];
    let has_w3 = w3.is_some();
    let has_wg_bias = wg_bias.is_some();
    let has_w1_bias = w1_bias.is_some();
    let has_w3_bias = w3_bias.is_some();
    let has_w2_bias = w2_bias.is_some();
    for opt in [w3, wg_bias, w1_bias, w3_bias, w2_bias].into_iter().flatten() {
        inputs.push(opt);
    }

    builder.wire(
        MoeFfn {
            k: k as usize,
            activation,
            gate,
            has_w3,
            has_wg_bias,
            has_w1_bias,
            has_w3_bias,
            has_w2_bias,
            act_alpha_bits: act_alpha.map(f32::to_bits),
            act_limit_bits: act_limit.map(f32::to_bits),
        },
        &inputs,
    )
}

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
}

/// Resolved positions of the optional inputs within the `inputs` slice,
/// derived from the `has_*` flags (see `deser_moe_ffn` for the push order).
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
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(self.k);

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
            let mut h: Array2<f32> = x_batch.dot(&w1_e);

            if self.is_clamped_act() {
                let w3_ix = idx.w3.context("clamped activation requires w3 (up branch)")?;
                let w3_e_t = block_quant_group_as_2d(&inputs[w3_ix], eid)?;
                let w3_e: ArrayView2<f32> =
                    w3_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
                let up: Array2<f32> = x_batch.dot(&w3_e);
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
                let gate: Array2<f32> = x_batch.dot(&w3_e);
                h.iter_mut().zip(gate.iter()).for_each(|(h_val, &g_val)| {
                    let silu = *h_val / (1.0 + (-*h_val).exp());
                    *h_val = silu * g_val;
                });
            } else {
                h.iter_mut().for_each(|h_val| {
                    *h_val = *h_val / (1.0 + (-*h_val).exp());
                });
            }

            let w2_e_t = block_quant_group_as_2d(&inputs[3], eid)?;
            let w2_e: ArrayView2<f32> =
                w2_e_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let y_expert: Array2<f32> = h.dot(&w2_e);
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
        .try_as_plain()
        .with_context(|| format!("{label} is not a plain tensor"))?
        .as_slice::<i64>()
        .with_context(|| format!("{label} is not an i64 slice"))
}

fn plain_f32_slice<'a>(tensor: &'a Tensor, label: &str) -> TractResult<&'a [f32]> {
    tensor
        .try_as_plain()
        .with_context(|| format!("{label} is not a plain tensor"))?
        .as_slice::<f32>()
        .with_context(|| format!("{label} is not an f32 slice"))
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RouteTopK {
    pub k: usize,
    pub gate: GateMode,
}

impl Op for RouteTopK {
    fn name(&self) -> StaticName {
        "RouteTopK".into()
    }
    op_as_typed_op!();
}

impl EvalOp for RouteTopK {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let x_t = inputs[0].cast_to::<f32>()?.into_owned();
        let wg_t = inputs[1].cast_to::<f32>()?.into_owned();
        let x = as_2d_tokens(x_t.to_plain_array_view::<f32>()?)?;
        let wg = router_weights_as_2d(wg_t.to_plain_array_view::<f32>()?)?;

        ensure!(
            x.shape()[1] == wg.shape()[1],
            "router dimension mismatch: x {:?}, wg {:?}",
            x.shape(),
            wg.shape()
        );
        ensure!(self.k <= wg.shape()[0], "top-k {} exceeds expert count {}", self.k, wg.shape()[0]);

        let t_tokens = x.shape()[0];
        let route_count = t_tokens * self.k;
        let mut route_token_ids = Vec::with_capacity(route_count);
        let mut route_expert_ids = Vec::with_capacity(route_count);
        let mut route_weights = Vec::with_capacity(route_count);

        let router_logits: Array2<f32> = x.dot(&wg.t());
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(self.k);

            let gate_weights = self.gate.weights(&full, &scores);
            for ((eid, _), gw) in scores.iter().zip(gate_weights) {
                route_token_ids.push(t as i64);
                route_expert_ids.push(*eid as i64);
                route_weights.push(gw);
            }
        }

        Ok(tvec![
            Tensor::from_shape(&[route_count], &route_token_ids)?.into_tvalue(),
            Tensor::from_shape(&[route_count], &route_expert_ids)?.into_tvalue(),
            Tensor::from_shape(&[route_count], &route_weights)?.into_tvalue(),
        ])
    }
}

impl TypedOp for RouteTopK {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        ensure!(
            inputs[0].rank() == 2 || inputs[0].rank() == 3,
            "RouteTopK expects rank-2 or rank-3 x, got {}",
            inputs[0].rank()
        );
        ensure!(
            inputs[1].rank() == 2 || inputs[1].rank() == 3,
            "RouteTopK expects rank-2 or rank-3 router weights, got {}",
            inputs[1].rank()
        );
        let route_count = token_count_dim(&inputs[0].shape) * self.k;
        Ok(tvec![
            i64::datum_type().fact(&[route_count.clone()]),
            i64::datum_type().fact(&[route_count.clone()]),
            f32::datum_type().fact(&[route_count]),
        ])
    }

    as_op!();
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RoutedInputMode {
    TokenRows,
    RouteRows,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedMatMul {
    pub input_mode: RoutedInputMode,
    pub cache_weights: bool,
}

#[derive(Clone, Debug)]
struct RoutedMatMulWeightsCache {
    k_dim: usize,
    n_dim: usize,
    num_experts: usize,
    packing: usize,
    a_format: PackedFormat,
    packed_by_expert: Vec<Box<dyn MMMInputValue>>,
}

#[derive(Clone, Debug)]
struct RoutedMatMulState {
    op: RoutedMatMul,
    cache: Option<RoutedMatMulWeightsCache>,
}

#[derive(Clone, Debug)]
struct FrozenRoutedMatMulState {
    op: RoutedMatMul,
    cache: Option<RoutedMatMulWeightsCache>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoutedRowsInput {
    base: usize,
    row_byte_offsets: Vec<isize>,
    k: usize,
    k_stride_bytes: isize,
    format: PackedFormat,
    fact: PackedExoticFact,
}

impl RoutedRowsInput {
    fn new(
        base: *const f32,
        row_byte_offsets: Vec<isize>,
        k: usize,
        k_stride_bytes: isize,
        format: PackedFormat,
    ) -> Self {
        let fact = PackedExoticFact {
            format: Box::new(format.clone()),
            mn: row_byte_offsets.len().to_dim(),
            k,
        };
        RoutedRowsInput { base: base as usize, row_byte_offsets, k, k_stride_bytes, format, fact }
    }

    fn read(&self, mn: usize, k: usize) -> f32 {
        unsafe {
            let ptr = (self.base as *const u8)
                .offset(self.row_byte_offsets[mn] + self.k_stride_bytes * k as isize);
            *(ptr as *const f32)
        }
    }
}

impl Hash for RoutedRowsInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
        self.row_byte_offsets.hash(state);
        self.k.hash(state);
        self.k_stride_bytes.hash(state);
        self.format.hash(state);
    }
}

impl std::fmt::Display for RoutedRowsInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RoutedRowsInput(mn={}, k={}, {})", self.mn(), self.k, self.format)
    }
}

unsafe impl Send for RoutedRowsInput {}
unsafe impl Sync for RoutedRowsInput {}

impl MMMInputValue for RoutedRowsInput {
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format
    }

    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.format.single_panel_layout(self.k, f32::datum_type().size_of()))
    }

    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let buffer = buffer.context("RoutedRowsInput requires a scratch panel buffer")?;
        let r = self.format.r();
        let mn_start = i * r;
        let mn_end = (mn_start + r).min(self.row_byte_offsets.len());
        ensure!(mn_start < self.row_byte_offsets.len(), "panel {i} starts past routed rows");

        unsafe {
            std::ptr::write_bytes(buffer, 0, self.format.single_panel_layout(self.k, 4).size());
            let mut writer = self.format.write_with_k_outer(buffer as *mut f32, self.k, r);
            for k in 0..self.k {
                for mn in mn_start..mn_start + r {
                    writer.write(if mn < mn_end { self.read(mn, k) } else { 0.0 });
                }
            }
        }
        Ok(buffer)
    }

    fn mn(&self) -> usize {
        self.row_byte_offsets.len()
    }

    fn k(&self) -> usize {
        self.k
    }

    fn exotic_fact(&self) -> &dyn ExoticFact {
        &self.fact
    }

    fn extract_at_mn_f16(&self, _mn: usize, _slice: &mut [f16]) -> TractResult<()> {
        bail!("RoutedRowsInput only supports f32 extraction")
    }

    fn extract_at_mn_f32(&self, mn: usize, slice: &mut [f32]) -> TractResult<()> {
        ensure!(slice.len() == self.k);
        ensure!(mn < self.mn());
        for (k, slot) in slice.iter_mut().enumerate() {
            *slot = self.read(mn, k);
        }
        Ok(())
    }
}

impl RoutedMatMul {
    fn input_view<'a>(&self, input_t: &'a Tensor) -> TractResult<ArrayView2<'a, f32>> {
        match self.input_mode {
            RoutedInputMode::TokenRows => as_2d_tokens(input_t.to_plain_array_view::<f32>()?),
            RoutedInputMode::RouteRows => {
                let input = input_t.to_plain_array_view::<f32>()?;
                ensure!(
                    input.ndim() == 2,
                    "route-row input must be rank 2, got {:?}",
                    input.shape()
                );
                Ok(input.into_dimensionality()?)
            }
        }
    }

    fn group_routes(
        &self,
        route_count: usize,
        num_experts: usize,
        route_expert_ids: &[i64],
    ) -> TractResult<Vec<Vec<usize>>> {
        let mut expert_routes = vec![Vec::new(); num_experts];
        for (r, &eid) in route_expert_ids.iter().enumerate().take(route_count) {
            ensure!(eid >= 0, "route {r} has negative expert id");
            let eid = eid as usize;
            ensure!(
                eid < num_experts,
                "route {r} references expert {eid}, but weights have {num_experts} experts"
            );
            expert_routes[eid].push(r);
        }
        Ok(expert_routes)
    }

    fn source_row(&self, r: usize, route_token_ids: &[i64]) -> TractResult<usize> {
        match self.input_mode {
            RoutedInputMode::TokenRows => {
                ensure!(route_token_ids[r] >= 0, "route {r} has negative token id");
                Ok(route_token_ids[r] as usize)
            }
            RoutedInputMode::RouteRows => Ok(r),
        }
    }

    fn eval_with_mmm(
        &self,
        input_t: &Tensor,
        weights_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
    ) -> TractResult<Option<Tensor>> {
        let input = self.input_view(input_t)?;
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let route_count = route_token_ids.len();
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let num_experts = weights.shape()[0];
        ensure!(route_count == route_expert_ids.len());
        ensure!(input.shape()[1] == k_dim);
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(input.shape()[0] == route_count);
        }
        if route_count == 0 {
            return Ok(Some(Array2::<f32>::zeros((0, n_dim)).into_tensor()));
        }

        let Some(mmm) = tract_nnef::tract_core::tract_linalg::ops().mmm(
            f32::datum_type(),
            None,
            Some(k_dim),
            Some(n_dim),
        ) else {
            return Ok(None);
        };
        if !mmm.is_supported_here() {
            return Ok(None);
        }
        let Some((packing, a_format, b_format)) =
            mmm.packings().iter().enumerate().find_map(|(ix, (a, b))| {
                let a = a.downcast_ref::<PackedFormat>()?;
                let b = b.downcast_ref::<PackedFormat>()?;
                Some((ix, a.clone(), b.clone()))
            })
        else {
            return Ok(None);
        };

        let expert_routes = self.group_routes(route_count, num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, n_dim));
        let item_size = f32::datum_type().size_of() as isize;
        let base = input.as_ptr();
        let row_stride_bytes = input.strides()[0] * item_size;
        let k_stride_bytes = input.strides()[1] * item_size;

        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut row_byte_offsets = Vec::with_capacity(routes.len());
            for &r in routes {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                row_byte_offsets.push(row_stride_bytes * src as isize);
            }

            let a = RoutedRowsInput::new(
                base,
                row_byte_offsets,
                k_dim,
                k_stride_bytes,
                a_format.clone(),
            );
            let w_e = weights_t.view_at_prefix(&[eid])?;
            let b = b_format.pack_tensor_view(&w_e, 0, 1)?;
            let mut expert_output =
                unsafe { Tensor::uninitialized::<f32>(&[routes.len(), n_dim])? };
            let store = unsafe { mmm.c_view(Some(0), Some(1)).wrap(&expert_output.view_mut()) };
            let uops = tvec![
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&a),
                    b: AsInputValue::Borrowed(&*b),
                    packing,
                },
                FusedSpec::Store(store),
            ];
            unsafe { mmm.run(routes.len(), n_dim, &uops)? };
            let expert_output: ArrayView2<f32> =
                expert_output.to_plain_array_view::<f32>()?.into_dimensionality()?;
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }

        Ok(Some(output.into_tensor()))
    }

    fn build_weights_cache(
        &self,
        weights_input: &Tensor,
    ) -> TractResult<Option<RoutedMatMulWeightsCache>> {
        let weights_t = weights_input.cast_to::<f32>()?.into_owned();
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let num_experts = weights.shape()[0];
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let Some(mmm) = tract_nnef::tract_core::tract_linalg::ops().mmm(
            f32::datum_type(),
            None,
            Some(k_dim),
            Some(n_dim),
        ) else {
            return Ok(None);
        };
        if !mmm.is_supported_here() {
            return Ok(None);
        }
        let Some((packing, a_format, b_format)) =
            mmm.packings().iter().enumerate().find_map(|(ix, (a, b))| {
                let a = a.downcast_ref::<PackedFormat>()?;
                let b = b.downcast_ref::<PackedFormat>()?;
                Some((ix, a.clone(), b.clone()))
            })
        else {
            return Ok(None);
        };

        let mut packed_by_expert = Vec::with_capacity(num_experts);
        for eid in 0..num_experts {
            let w_e = weights_t.view_at_prefix(&[eid])?;
            packed_by_expert.push(b_format.pack_tensor_view(&w_e, 0, 1)?);
        }

        Ok(Some(RoutedMatMulWeightsCache {
            k_dim,
            n_dim,
            num_experts,
            packing,
            a_format,
            packed_by_expert,
        }))
    }

    fn eval_with_cached_mmm(
        &self,
        input_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
        cache: &RoutedMatMulWeightsCache,
    ) -> TractResult<Tensor> {
        let input = self.input_view(input_t)?;
        let route_count = route_token_ids.len();
        ensure!(route_count == route_expert_ids.len());
        ensure!(
            input.shape()[1] == cache.k_dim,
            "routed matmul input dim {} does not match weight dim {}",
            input.shape()[1],
            cache.k_dim
        );
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(
                input.shape()[0] == route_count,
                "route-row input has {} rows but route metadata has {} rows",
                input.shape()[0],
                route_count
            );
        }
        if route_count == 0 {
            return Ok(Array2::<f32>::zeros((0, cache.n_dim)).into_tensor());
        }

        let Some(mmm) = tract_nnef::tract_core::tract_linalg::ops().mmm(
            f32::datum_type(),
            None,
            Some(cache.k_dim),
            Some(cache.n_dim),
        ) else {
            bail!("cached routed matmul MMM is no longer available");
        };
        ensure!(mmm.is_supported_here(), "cached routed matmul MMM is no longer supported here");

        let expert_routes = self.group_routes(route_count, cache.num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, cache.n_dim));
        let item_size = f32::datum_type().size_of() as isize;
        let base = input.as_ptr();
        let row_stride_bytes = input.strides()[0] * item_size;
        let k_stride_bytes = input.strides()[1] * item_size;

        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut row_byte_offsets = Vec::with_capacity(routes.len());
            for &r in routes {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                row_byte_offsets.push(row_stride_bytes * src as isize);
            }

            let a = RoutedRowsInput::new(
                base,
                row_byte_offsets,
                cache.k_dim,
                k_stride_bytes,
                cache.a_format.clone(),
            );
            let mut expert_output =
                unsafe { Tensor::uninitialized::<f32>(&[routes.len(), cache.n_dim])? };
            let store = unsafe { mmm.c_view(Some(0), Some(1)).wrap(&expert_output.view_mut()) };
            let uops = tvec![
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&a),
                    b: AsInputValue::Borrowed(&*cache.packed_by_expert[eid]),
                    packing: cache.packing,
                },
                FusedSpec::Store(store),
            ];
            unsafe { mmm.run(routes.len(), cache.n_dim, &uops)? };
            let expert_output: ArrayView2<f32> =
                expert_output.to_plain_array_view::<f32>()?.into_dimensionality()?;
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }

        Ok(output.into_tensor())
    }

    fn eval_grouped_ndarray(
        &self,
        input_t: &Tensor,
        weights_t: &Tensor,
        route_token_ids: &[i64],
        route_expert_ids: &[i64],
    ) -> TractResult<Tensor> {
        let input = self.input_view(input_t)?;
        let weights: ArrayView3<f32> =
            weights_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        ensure!(route_token_ids.len() == route_expert_ids.len());

        let route_count = route_token_ids.len();
        let k_dim = weights.shape()[1];
        let n_dim = weights.shape()[2];
        let num_experts = weights.shape()[0];
        ensure!(
            input.shape()[1] == k_dim,
            "routed matmul input dim {} does not match weight dim {}",
            input.shape()[1],
            k_dim
        );
        if self.input_mode == RoutedInputMode::RouteRows {
            ensure!(
                input.shape()[0] == route_count,
                "route-row input has {} rows but route metadata has {} rows",
                input.shape()[0],
                route_count
            );
        }

        let expert_routes = self.group_routes(route_count, num_experts, route_expert_ids)?;
        let mut output = Array2::<f32>::zeros((route_count, n_dim));
        for (eid, routes) in expert_routes.iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let mut gathered = Array2::<f32>::zeros((routes.len(), k_dim));
            for (slot, &r) in routes.iter().enumerate() {
                let src = self.source_row(r, route_token_ids)?;
                ensure!(
                    src < input.shape()[0],
                    "route {r} references input row {src}, but input has {} rows",
                    input.shape()[0]
                );
                gathered.row_mut(slot).assign(&input.row(src));
            }

            let expert_output = gathered.dot(&weights.slice(s![eid, .., ..]));
            for (slot, &r) in routes.iter().enumerate() {
                output.row_mut(r).assign(&expert_output.row(slot));
            }
        }
        Ok(output.into_tensor())
    }
}

impl Op for RoutedMatMul {
    fn name(&self) -> StaticName {
        "RoutedMatMul".into()
    }
    op_as_typed_op!();
}

impl OpStateFreeze for RoutedMatMulState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenRoutedMatMulState { op: self.op.clone(), cache: self.cache.clone() })
    }
}

impl FrozenOpState for FrozenRoutedMatMulState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(RoutedMatMulState { op: self.op.clone(), cache: self.cache.clone() })
    }
}

impl OpState for RoutedMatMulState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        if self.cache.is_none() {
            self.cache = self.op.build_weights_cache(&inputs[1])?;
        }
        let Some(cache) = self.cache.as_ref() else {
            return self.op.eval(inputs);
        };

        let input_t = inputs[0].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_expert_ids_t = inputs[3].cast_to::<i64>()?.into_owned();
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_expert_ids = plain_i64_slice(&route_expert_ids_t, "route_expert_ids")?;
        let output =
            self.op.eval_with_cached_mmm(&input_t, route_token_ids, route_expert_ids, cache)?;
        Ok(tvec![output.into_tvalue()])
    }
}

impl EvalOp for RoutedMatMul {
    fn is_stateless(&self) -> bool {
        !self.cache_weights
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        if self.cache_weights {
            Ok(Some(Box::new(RoutedMatMulState { op: self.clone(), cache: None })))
        } else {
            Ok(None)
        }
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: data, weights [E,K,N], route_token_ids [R], route_expert_ids [R].
        let input_t = inputs[0].cast_to::<f32>()?.into_owned();
        let weights_t = inputs[1].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_expert_ids_t = inputs[3].cast_to::<i64>()?.into_owned();
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_expert_ids = plain_i64_slice(&route_expert_ids_t, "route_expert_ids")?;

        let output = if let Some(output) =
            self.eval_with_mmm(&input_t, &weights_t, route_token_ids, route_expert_ids)?
        {
            output
        } else {
            self.eval_grouped_ndarray(&input_t, &weights_t, route_token_ids, route_expert_ids)?
        };
        Ok(tvec![output.into_tvalue()])
    }
}

impl TypedOp for RoutedMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        ensure!(inputs[1].rank() == 3, "RoutedMatMul weights must be rank 3");
        ensure!(inputs[2].rank() == 1, "RoutedMatMul route_token_ids must be rank 1");
        ensure!(inputs[3].rank() == 1, "RoutedMatMul route_expert_ids must be rank 1");
        let route_count = inputs[2].shape.to_tvec()[0].clone();
        let out_dim = inputs[1].shape.to_tvec()[2].clone();
        Ok(tvec![f32::datum_type().fact(&[route_count, out_dim])])
    }

    as_op!();
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedCombine;

impl Op for RoutedCombine {
    fn name(&self) -> StaticName {
        "RoutedCombine".into()
    }
    op_as_typed_op!();
}

impl EvalOp for RoutedCombine {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // inputs: shape_like, route_values [R,D], route_token_ids [R], route_weights [R].
        let output_dt = inputs[0].datum_type();
        let route_values_t = inputs[1].cast_to::<f32>()?.into_owned();
        let route_token_ids_t = inputs[2].cast_to::<i64>()?.into_owned();
        let route_weights_t = inputs[3].cast_to::<f32>()?.into_owned();

        let shape = inputs[0].shape().to_vec();
        ensure!(
            shape.len() == 2 || shape.len() == 3,
            "RoutedCombine shape_like must be rank 2 or 3, got {:?}",
            shape
        );
        let t_tokens: usize = shape[..shape.len() - 1].iter().product();
        let d_model = *shape.last().unwrap();
        let route_values: ArrayView2<f32> =
            route_values_t.to_plain_array_view::<f32>()?.into_dimensionality()?;
        let route_token_ids = plain_i64_slice(&route_token_ids_t, "route_token_ids")?;
        let route_weights = plain_f32_slice(&route_weights_t, "route_weights")?;
        ensure!(route_values.shape()[0] == route_token_ids.len());
        ensure!(route_token_ids.len() == route_weights.len());
        ensure!(
            route_values.shape()[1] == d_model,
            "route value dim {} does not match output dim {}",
            route_values.shape()[1],
            d_model
        );

        let mut output = Array2::<f32>::zeros((t_tokens, d_model));
        for r in 0..route_token_ids.len() {
            ensure!(route_token_ids[r] >= 0, "route {r} has negative token id");
            let token = route_token_ids[r] as usize;
            ensure!(
                token < t_tokens,
                "route {r} references token {token}, but output has {t_tokens} tokens"
            );
            let route = route_values.row(r);
            let mut out = output.row_mut(token);
            out.scaled_add(route_weights[r], &route);
        }

        let output_tensor = if shape.len() == 3 {
            output.into_shape_with_order((shape[0], shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(output_dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
    }
}

impl TypedOp for RoutedCombine {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 4);
        Ok(tvec![inputs[0].datum_type.fact(inputs[0].shape.clone())])
    }

    as_op!();
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

impl EvalOp for MoeFfn {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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
                unpacked.into_iter().next().unwrap()
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
        let _d_hidden = w1.shape()[2];

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
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(self.k);

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

            // Expert weight slices for this expert
            let w1_e = w1.slice(s![eid, .., ..]); // [D, H]
            let w2_e = w2.slice(s![eid, .., ..]); // [H, D]

            // gate branch: h = x_batch @ w1_e (+ w1_bias)  -> [n, H]
            let mut h: Array2<f32> = x_batch.dot(&w1_e);
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
                let w3_e = w3.slice(s![eid, .., ..]); // [D, H]
                let mut up: Array2<f32> = x_batch.dot(&w3_e);
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
                // SwiGLU: h = silu(h) * (x_batch @ w3_e + w3_bias)
                let w3_e = w3.slice(s![eid, .., ..]); // [D, H]
                let mut gate: Array2<f32> = x_batch.dot(&w3_e); // [n, H]
                if let Some(ref b) = w3_bias {
                    add_expert_bias(&mut gate, b, eid)?;
                }
                h.iter_mut().zip(gate.iter()).for_each(|(h_val, &g_val)| {
                    let silu = *h_val / (1.0 + (-*h_val).exp());
                    *h_val = silu * g_val;
                });
            } else {
                // Simple silu activation
                h.iter_mut().for_each(|h_val| {
                    *h_val = *h_val / (1.0 + (-*h_val).exp());
                });
            }

            // y_expert = h @ w2_e (+ w2_bias)  -> [n, D]  (BLAS-backed GEMM)
            let mut y_expert: Array2<f32> = h.dot(&w2_e);
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

impl TypedOp for MoeFfn {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        // single output: same shape as input x
        let x_fact = inputs[0];
        let output_fact = x_fact.datum_type.fact(x_fact.shape.clone());
        Ok(tvec!(output_fact))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        // The routed primitive lowering only models the bias-free, plain
        // activation case. gpt-oss (biases + clamped SwiGLU) stays on the
        // reference eval path, which is correct if slower.
        if self.has_any_bias() || self.is_clamped_act() {
            return Ok(None);
        }
        let Some(act_op) = activation_op(&self.activation, self.has_w3) else {
            return Ok(None);
        };

        let wg_const = model.node(node.inputs[1].node).op_as::<Const>();
        let w1_const = model.node(node.inputs[2].node).op_as::<Const>();
        let w2_const = model.node(node.inputs[3].node).op_as::<Const>();
        let w3_const =
            if self.has_w3 { model.node(node.inputs[4].node).op_as::<Const>() } else { None };
        if wg_const.is_some()
            && w1_const.is_some()
            && w2_const.is_some()
            && (!self.has_w3 || w3_const.is_some())
        {
            let wg_tensor = wg_const.unwrap().val().clone();
            let w1_tensor = w1_const.unwrap().val().clone();
            let w2_tensor = w2_const.unwrap().val().clone();
            let w3_tensor = w3_const.map(|c| c.val().clone());

            if !w1_tensor.is_plain()
                || !w2_tensor.is_plain()
                || w3_tensor.as_ref().is_some_and(|t| !t.is_plain())
            {
                return Ok(None);
            }

            let num_experts = w1_tensor.shape()[0];
            let d_model = w1_tensor.shape()[1];
            let d_hidden = w1_tensor.shape()[2];

            let router_plan =
                build_router_plan(&wg_tensor, &model.symbols).context("Building router plan")?;

            let mut expert_plans = Vec::with_capacity(num_experts);
            for eid in 0..num_experts {
                let w1_e = w1_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?;
                let w2_e = w2_tensor.slice(0, eid, eid + 1)?.into_shape(&[d_hidden, d_model])?;
                let w3_e = if let Some(ref w3) = w3_tensor {
                    Some(w3.slice(0, eid, eid + 1)?.into_shape(&[d_model, d_hidden])?)
                } else {
                    None
                };

                let plan = build_expert_plan(
                    &w1_e,
                    &w2_e,
                    w3_e.as_ref(),
                    &self.activation,
                    &model.symbols,
                )
                .with_context(|| format!("Building expert plan for expert {eid}"))?;
                expert_plans.push(plan);
            }

            let opt_op = OptMoeFfn {
                k: self.k,
                gate: self.gate.clone(),
                num_experts,
                d_model,
                d_hidden,
                router_plan,
                expert_plans,
            };

            let mut patch = TypedModelPatch::default();
            let x_tap = patch.tap_model(model, node.inputs[0])?;
            let wires = patch.wire_node(&node.name, opt_op, &[x_tap])?;
            patch.shunt_outside(model, OutletId::new(node.id, 0), wires[0])?;
            return Ok(Some(patch));
        }

        let expert_inputs: &[usize] = if self.has_w3 { &[2, 3, 4] } else { &[2, 3] };
        for &input_ix in expert_inputs {
            if let Some(konst) = model.node(node.inputs[input_ix].node).op_as::<Const>() {
                if !konst.val().is_plain() {
                    return Ok(None);
                }
            }
        }
        let cache_weights = |input_ix: usize| {
            model
                .node(node.inputs[input_ix].node)
                .op_as::<Const>()
                .is_some_and(|konst| konst.val().is_plain())
        };
        let cache_w1 = cache_weights(2);
        let cache_w2 = cache_weights(3);
        let cache_w3 = self.has_w3 && cache_weights(4);

        let mut patch = TypedModelPatch::default();
        let x = patch.tap_model(model, node.inputs[0])?;
        let wg = patch.tap_model(model, node.inputs[1])?;
        let w1 = patch.tap_model(model, node.inputs[2])?;
        let w2 = patch.tap_model(model, node.inputs[3])?;

        let routes = patch.wire_node(
            format!("{}.route_topk", node.name),
            RouteTopK { k: self.k, gate: self.gate.clone() },
            &[x, wg],
        )?;
        let route_token_ids = routes[0];
        let route_expert_ids = routes[1];
        let route_weights = routes[2];

        let h1 = patch.wire_node(
            format!("{}.w1", node.name),
            RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: cache_w1 },
            &[x, w1, route_token_ids, route_expert_ids],
        )?[0];
        let activated = patch.wire_node(format!("{}.activation", node.name), act_op, &[h1])?[0];

        let hidden = if self.has_w3 {
            let w3 = patch.tap_model(model, node.inputs[4])?;
            let gate = patch.wire_node(
                format!("{}.w3", node.name),
                RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: cache_w3 },
                &[x, w3, route_token_ids, route_expert_ids],
            )?[0];
            patch.wire_node(format!("{}.swiglu_mul", node.name), mul(), &[activated, gate])?[0]
        } else {
            activated
        };

        let route_values = patch.wire_node(
            format!("{}.w2", node.name),
            RoutedMatMul { input_mode: RoutedInputMode::RouteRows, cache_weights: cache_w2 },
            &[hidden, w2, route_token_ids, route_expert_ids],
        )?[0];
        let output = patch.wire_node(
            format!("{}.combine", node.name),
            RoutedCombine,
            &[x, route_values, route_token_ids, route_weights],
        )?[0];
        patch.shunt_outside(model, OutletId::new(node.id, 0), output)?;
        Ok(Some(patch))
    }

    as_op!();
}

// ---------------------------------------------------------------------------
// Activation helper
// ---------------------------------------------------------------------------

fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(silu())),
        // SwiGLU: the inner activation is silu, w3 provides the gate branch
        "swiglu" if has_w3 => Some(Box::new(silu())),
        "gelu" => Some(Box::new(gelu_approximate(false))),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}

fn build_router_plan(wg: &Arc<Tensor>, symbols: &SymbolScope) -> TractResult<Arc<TypedSimplePlan>> {
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

fn build_expert_plan(
    w1: &Tensor,
    w2: &Tensor,
    w3: Option<&Tensor>,
    activation: &str,
    symbols: &SymbolScope,
) -> TractResult<Arc<TypedSimplePlan>> {
    let mut model = TypedModel { symbols: symbols.clone(), ..Default::default() };
    let n_sym = symbols.sym("moe_n");
    let dt = f32::datum_type();
    let to_f32 = |t: &Tensor| -> TractResult<Tensor> { Ok(t.cast_to::<f32>()?.into_owned()) };

    let d_model = w1.shape()[0];

    let x = model.add_source("x", dt.fact([n_sym.to_dim(), d_model.to_dim()]))?;
    let w1_const = model.add_const("w1", to_f32(w1)?)?;
    let axes: AxesMapping = "ij,jk->ik".parse()?;
    let h = model.wire_node("w1_matmul", EinSum::new(axes.clone(), dt), &[x, w1_const])?[0];

    let act_op = activation_op(activation, w3.is_some())
        .ok_or_else(|| format_err!("Unsupported activation: {activation}"))?;
    let h = model.wire_node("activation", act_op, &[h])?[0];

    let h = if let Some(w3) = w3 {
        let w3_const = model.add_const("w3", to_f32(w3)?)?;
        let gate = model.wire_node("w3_matmul", EinSum::new(axes.clone(), dt), &[x, w3_const])?[0];
        model.wire_node("swiglu_mul", mul(), &[h, gate])?[0]
    } else {
        h
    };

    let w2_const = model.add_const("w2", to_f32(w2)?)?;
    let y = model.wire_node("w2_matmul", EinSum::new(axes, dt), &[h, w2_const])?[0];

    model.select_output_outlets(&[y])?;
    SimplePlan::new(model.into_optimized()?)
}

#[derive(Clone, Debug)]
pub struct OptMoeFfn {
    pub k: usize,
    pub gate: GateMode,
    pub num_experts: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub router_plan: Arc<TypedSimplePlan>,
    pub expert_plans: Vec<Arc<TypedSimplePlan>>,
}

impl Hash for OptMoeFfn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        self.gate.hash(state);
        self.num_experts.hash(state);
        self.d_model.hash(state);
        self.d_hidden.hash(state);
    }
}

impl PartialEq for OptMoeFfn {
    fn eq(&self, other: &Self) -> bool {
        self.k == other.k
            && self.gate == other.gate
            && self.num_experts == other.num_experts
            && self.d_model == other.d_model
            && self.d_hidden == other.d_hidden
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
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let router_state = self.router_plan.spawn()?;
        let expert_states =
            self.expert_plans.iter().map(|p| p.spawn()).collect::<TractResult<Vec<_>>>()?;
        Ok(Some(Box::new(OptMoeFfnState { op: self.clone(), router_state, expert_states })))
    }
}

#[derive(Clone, Debug)]
struct OptMoeFfnState {
    op: OptMoeFfn,
    router_state: TypedSimpleState,
    expert_states: Vec<TypedSimpleState>,
}

#[derive(Clone, Debug)]
struct FrozenOptMoeFfnState {
    op: OptMoeFfn,
    router_state: TypedFrozenSimpleState,
    expert_states: Vec<TypedFrozenSimpleState>,
}

impl OpStateFreeze for OptMoeFfnState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenOptMoeFfnState {
            op: self.op.clone(),
            router_state: self.router_state.freeze(),
            expert_states: self.expert_states.iter().map(|s| s.freeze()).collect(),
        })
    }
}

impl FrozenOpState for FrozenOptMoeFfnState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(OptMoeFfnState {
            op: self.op.clone(),
            router_state: self.router_state.unfreeze(),
            expert_states: self.expert_states.iter().map(|s| s.unfreeze()).collect(),
        })
    }
}

impl OpState for OptMoeFfnState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = &self.op;
        let x_input = &inputs[0];
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

        let router_result = self.router_state.run(tvec![x_2d_tensor.into_tvalue()])?;
        let router_logits_f32 = router_result[0].cast_to::<f32>()?;
        let router_logits: ArrayView2<f32> =
            router_logits_f32.to_plain_array_view::<f32>()?.into_dimensionality()?;

        let mut assignments: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t_tokens);
        for t in 0..t_tokens {
            let row = router_logits.row(t);
            let full: Vec<f32> = row.iter().copied().collect();
            let mut scores: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(e, &s)| (e, s)).collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(op.k);

            let gate_weights = op.gate.weights(&full, &scores);
            assignments
                .push(scores.iter().zip(gate_weights).map(|((eid, _), gw)| (*eid, gw)).collect());
        }

        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); op.num_experts];
        for (t, token_experts) in assignments.iter().enumerate() {
            for &(eid, gw) in token_experts {
                expert_tokens[eid].push((t, gw));
            }
        }

        let mut output = Array2::<f32>::zeros((t_tokens, d_model));
        for (eid, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let n = tokens.len();
            let mut x_batch = Tensor::zero_dt(f32::datum_type(), &[n, d_model])?;
            {
                let mut x_batch_plain = x_batch.try_as_plain_mut()?;
                let x_batch_slice = x_batch_plain.as_slice_mut::<f32>()?;
                for (i, &(t, _)) in tokens.iter().enumerate() {
                    let src = x.row(t);
                    x_batch_slice[i * d_model..(i + 1) * d_model]
                        .copy_from_slice(src.as_slice().unwrap());
                }
            }

            let y_expert = self.expert_states[eid].run(tvec![x_batch.into_tvalue()])?;
            let y_view: ArrayView2<f32> =
                y_expert[0].to_plain_array_view::<f32>()?.into_dimensionality()?;
            scatter_add_weighted(&mut output, &y_view, tokens);
        }

        let output_tensor = if x_ndim == 3 {
            output.into_shape_with_order((x_orig_shape[0], x_orig_shape[1], d_model))?.into_tensor()
        } else {
            output.into_tensor()
        };
        let output_tensor = output_tensor.cast_to_dt(dt)?.into_owned();
        Ok(tvec![output_tensor.into_tvalue()])
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use tract_nnef::tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0,
    };

    fn make_moe_model(
        t_tokens: usize,
        d_model: usize,
        d_hidden: usize,
        num_experts: usize,
        k: usize,
        has_w3: bool,
    ) -> TractResult<(TypedModel, Tensor)> {
        let mut model = TypedModel::default();

        let x = model.add_source("x", f32::datum_type().fact([t_tokens, d_model]))?;

        // Deterministic pseudo-random weights
        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng()).collect();
            tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
        };

        let wg_data = make_tensor(&[num_experts, d_model], &mut next_f32);
        let w1_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
        let w2_data = make_tensor(&[num_experts, d_hidden, d_model], &mut next_f32);

        let wg = model.add_const("wg", wg_data)?;
        let w1 = model.add_const("w1", w1_data)?;
        let w2 = model.add_const("w2", w2_data)?;

        let mut inputs = vec![x, wg, w1, w2];

        if has_w3 {
            let w3_data = make_tensor(&[num_experts, d_model, d_hidden], &mut next_f32);
            let w3 = model.add_const("w3", w3_data)?;
            inputs.push(w3);
        }

        let op = MoeFfn::basic(k, "silu", GateMode::SoftmaxTopk, has_w3);
        let outputs = model.wire_node("moe", op, &inputs)?;
        model.select_output_outlets(&outputs)?;

        // Create input tensor
        let x_data = make_tensor(&[t_tokens, d_model], &mut next_f32);

        Ok((model, x_data))
    }

    fn add_q40_const(model: &mut TypedModel, name: &str, tensor: Tensor) -> TractResult<OutletId> {
        let shape = tensor.shape().to_vec();
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
        let m = shape[..shape.len() - 1].iter().product();
        let quant = Q4_0.quant_f32(tensor.try_as_plain()?.as_slice::<f32>()?)?;
        let storage = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant))?;
        let packed = Arc::new(storage.into_tensor_with_shape(f32::datum_type(), &shape));
        let fact = BlockQuantFact::new(Box::new(Q4_0), shape.iter().copied().collect());
        Ok(model.wire_node(name, Const::new_with_exotic_fact(packed, Box::new(fact))?, &[])?[0])
    }

    fn q40_roundtrip(tensor: &Tensor) -> TractResult<Tensor> {
        let shape = tensor.shape().to_vec();
        let k = *shape.last().context("Q40 tensor has no last axis")?;
        ensure!(k % Q4_0.block_len() == 0, "Q40 K axis must be a multiple of 32");
        let quant = Q4_0.quant_f32(tensor.try_as_plain()?.as_slice::<f32>()?)?;
        Q4_0.dequant_f32(&quant)?.into_shape(&shape)
    }

    #[test]
    fn test_codegen_keeps_q40_expert_constants_on_reference_eval() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([2, 32]))?;
        let mut ref_model = TypedModel::default();
        let ref_x = ref_model.add_source("x", f32::datum_type().fact([2, 32]))?;

        let mut rng_state: u64 = 77;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let make_tensor = |shape: &[usize], rng: &mut dyn FnMut() -> f32| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng()).collect();
            tract_ndarray::ArrayD::from_shape_vec(shape, data).unwrap().into_tensor()
        };

        let wg_data = make_tensor(&[2, 32], &mut next_f32);
        let w1_data = make_tensor(&[2, 32, 32], &mut next_f32);
        let w2_data = make_tensor(&[2, 32, 32], &mut next_f32);
        let w3_data = make_tensor(&[2, 32, 32], &mut next_f32);

        let wg = model.add_const("wg", wg_data.clone())?;
        let w1 = add_q40_const(&mut model, "w1", w1_data.clone())?;
        let w2 = add_q40_const(&mut model, "w2", w2_data.clone())?;
        let w3 = add_q40_const(&mut model, "w3", w3_data.clone())?;

        let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, true);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2, w3])?;
        model.select_output_outlets(&outputs)?;

        let ref_wg = ref_model.add_const("wg", wg_data)?;
        let ref_w1 = ref_model.add_const("w1", q40_roundtrip(&w1_data)?)?;
        let ref_w2 = ref_model.add_const("w2", q40_roundtrip(&w2_data)?)?;
        let ref_w3 = ref_model.add_const("w3", q40_roundtrip(&w3_data)?)?;
        let ref_op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, true);
        let ref_outputs =
            ref_model.wire_node("moe", ref_op, &[ref_x, ref_wg, ref_w1, ref_w2, ref_w3])?;
        ref_model.select_output_outlets(&ref_outputs)?;

        let opt_model = model.into_optimized()?;
        let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
        assert!(has_moe, "Expected Q40 experts to stay on MoeFfn reference eval");
        let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
        assert!(!has_opt, "Q40 experts should not lower to OptMoeFfn yet");
        let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
        assert!(!has_routed, "Q40 experts should not lower to RoutedMatMul yet");

        let x_data = make_tensor(&[2, 32], &mut next_f32);
        let plan = SimplePlan::new(opt_model)?;
        let result = plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;
        let ref_result = SimplePlan::new(ref_model)?.spawn()?.run(tvec![x_data.into_tvalue()])?;
        result[0].close_enough(&ref_result[0], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_opt_moe_ffn_matches_reference() -> TractResult<()> {
        // Test with SwiGLU (has_w3=true)
        let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, true)?;

        // Run reference (unoptimized)
        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        // Run optimized
        let opt_model = model.into_optimized()?;

        // Verify constant expert weights keep the fast plan-based path.
        let has_opt = opt_model.nodes().iter().any(|n| n.op_is::<OptMoeFfn>());
        assert!(has_opt, "Expected OptMoeFfn in optimized model");
        let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
        assert!(!has_routed, "Constant expert weights should use OptMoeFfn");

        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        // Compare outputs
        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_opt_moe_ffn_no_w3() -> TractResult<()> {
        // Test without SwiGLU (has_w3=false)
        let (model, x_data) = make_moe_model(8, 16, 32, 4, 2, false)?;

        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        let opt_model = model.into_optimized()?;
        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_opt_moe_ffn_top1() -> TractResult<()> {
        let (model, x_data) = make_moe_model(16, 8, 16, 8, 1, true)?;

        let ref_plan = SimplePlan::new(model.clone())?;
        let ref_result = ref_plan.spawn()?.run(tvec![x_data.clone().into_tvalue()])?;

        let opt_model = model.into_optimized()?;
        let opt_plan = SimplePlan::new(opt_model)?;
        let opt_result = opt_plan.spawn()?.run(tvec![x_data.into_tvalue()])?;

        ref_result[0].close_enough(&opt_result[0], Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_codegen_lowers_non_const_weights() -> TractResult<()> {
        // The routed primitive lowering does not require constant weights.
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([4, 8]))?;
        let wg = model.add_source("wg", f32::datum_type().fact([2, 8]))?;
        let w1 = model.add_source("w1", f32::datum_type().fact([2, 8, 16]))?;
        let w2 = model.add_source("w2", f32::datum_type().fact([2, 16, 8]))?;

        let op = MoeFfn::basic(1, "silu", GateMode::SoftmaxTopk, false);
        let outputs = model.wire_node("moe", op, &[x, wg, w1, w2])?;
        model.select_output_outlets(&outputs)?;

        let opt_model = model.into_optimized()?;

        let has_moe = opt_model.nodes().iter().any(|n| n.op_is::<MoeFfn>());
        assert!(!has_moe, "Expected MoeFfn to lower even when weights are not constants");
        let has_routed = opt_model.nodes().iter().any(|n| n.op_is::<RoutedMatMul>());
        assert!(has_routed, "Expected RoutedMatMul in optimized model");

        Ok(())
    }

    #[test]
    fn test_routed_matmul_groups_by_expert_and_preserves_route_order() -> TractResult<()> {
        let input = Tensor::from_shape(&[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let weights = Tensor::from_shape(
            &[2, 2, 2],
            &[
                1.0f32, 0.0, 0.0, 1.0, // expert 0: identity
                10.0, 0.0, 0.0, 100.0, // expert 1: scale columns differently
            ],
        )?;
        let route_token_ids = Tensor::from_shape(&[4], &[2i64, 0, 1, 2])?;
        let route_expert_ids = Tensor::from_shape(&[4], &[1i64, 0, 1, 0])?;

        let op = RoutedMatMul { input_mode: RoutedInputMode::TokenRows, cache_weights: false };
        let result = op.eval(tvec![
            input.into_tvalue(),
            weights.into_tvalue(),
            route_token_ids.into_tvalue(),
            route_expert_ids.into_tvalue(),
        ])?;

        let expected = Tensor::from_shape(
            &[4, 2],
            &[
                50.0f32, 600.0, // route 0: token 2 through expert 1
                1.0, 2.0, // route 1: token 0 through expert 0
                30.0, 400.0, // route 2: token 1 through expert 1
                5.0, 6.0, // route 3: token 2 through expert 0
            ],
        )?;
        result[0].close_enough(&expected, Approximation::Approximate)?;

        Ok(())
    }

    #[test]
    fn test_e2e_nnef_qwen3_moe() -> TractResult<()> {
        use crate::WithTractTransformers;
        use std::io::Cursor;

        // Load the Qwen3 MoE model exported from transformers
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../harness/nnef-test-cases/moe-ffn/qwen3-tiny");

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let model = nnef.model_for_path(&model_path)?;
        let model = model.into_optimized()?;

        // Verify constant expert weights use the fast plan-based path.
        let has_opt = model.nodes().iter().any(|n: &TypedNode| n.op_is::<OptMoeFfn>());
        assert!(has_opt, "Expected OptMoeFfn in optimized model");
        let has_routed = model.nodes().iter().any(|n: &TypedNode| n.op_is::<RoutedMatMul>());
        assert!(!has_routed, "Constant expert weights should use OptMoeFfn");

        let plan = SimplePlan::new(model)?;

        // Load input and expected output from io.npz
        let npz_path = model_path.join("io.npz");
        let npz_bytes = std::fs::read(&npz_path)?;
        let mut npz = ndarray_npy::NpzReader::new(Cursor::new(npz_bytes))?;

        let input: tract_ndarray::ArrayD<f32> = npz.by_name("input_0.npy")?;
        let expected_output: tract_ndarray::ArrayD<f32> = npz.by_name("output_0.npy")?;

        // Run inference
        let result = plan.spawn()?.run(tvec![input.into_tensor().into_tvalue()])?;

        // Compare against PyTorch reference output
        result[0].close_enough(&expected_output.into_tensor(), Approximation::Approximate)?;

        Ok(())
    }
}
