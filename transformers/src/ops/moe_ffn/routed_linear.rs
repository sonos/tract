use super::{block_quant_group_as_2d, block_quant_group_tensor, plain_i64_slice};
use crate::ops::routed_matmul::{
    PreparedRoutedMatMulState, RoutedInputRows, build_block_quant_routed_matmul, f32_input,
    run_prepared_routed_matmul,
};
use tract_ndarray::{Array2, ArrayView2, Axis, s};
use tract_nnef::internal::*;
use tract_nnef::tract_core::tract_linalg::block_quant::BlockQuantStorage;

use super::{ExpertLayout, MoeFfn, RouteSelect, RoutedCombine, RoutedInputMode};
use tract_nnef::tract_core::ops::cast::cast;
use tract_nnef::tract_core::ops::change_axes::AxisOp;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;

/// The activation a [`RoutedLinear`] applies to its projection, or to the gate of its gate/up
/// pair before the up projection multiplies it.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RoutedActivation {
    None,
    Silu,
    /// The tanh approximation.
    Gelu,
    Relu,
    /// gpt-oss: `gate = min(g, limit)`, `up = clamp(u, -limit, limit)`,
    /// `h = (up + 1) * gate * sigmoid(alpha * gate)`. Pairs only.
    ClampedSwiGlu {
        alpha_bits: u32,
        limit_bits: u32,
    },
}

impl RoutedActivation {
    fn apply(&self, v: f32) -> f32 {
        match self {
            RoutedActivation::None | RoutedActivation::ClampedSwiGlu { .. } => v,
            RoutedActivation::Silu => v / (1.0 + (-v).exp()),
            RoutedActivation::Gelu => {
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                0.5 * v * (1.0 + f32::tanh(sqrt_2_over_pi * (v + 0.044715 * v.powi(3))))
            }
            RoutedActivation::Relu => v.max(0.0),
        }
    }

    fn combine(&self, gate: f32, up: f32) -> f32 {
        if let RoutedActivation::ClampedSwiGlu { alpha_bits, limit_bits } = self {
            let (alpha, limit) = (f32::from_bits(*alpha_bits), f32::from_bits(*limit_bits));
            let gate = gate.min(limit);
            let up = up.clamp(-limit, limit);
            (up + 1.0) * gate / (1.0 + (-alpha * gate).exp())
        } else {
            self.apply(gate) * up
        }
    }
}

/// Each route's input row through the weights of its expert: `[R, N]` in f32.
///
/// Inputs, in order: `data [rows, K]` (f16 or f32), `route_token_ids [R]` and
/// `route_expert_ids [R]` (i64), the weights `w [E, N, K]`, then `w_up [E, N, K]` when `pair`,
/// then per-expert biases `[E, N]` for each projection when `bias`. Weights are plain f16 or
/// f32, or block-quantized in any format, K last.
///
/// The route reads input row `route_token_ids[r]` in [`RoutedInputMode::TokenRows`] and row `r`
/// in [`RoutedInputMode::RouteRows`]. A projection is `x . w[e]^T (+ b[e])`, accumulated in f32.
/// Single: `activation(projection)`. Pair: `activation(gate) * up`, or the clamped SwiGLU.
/// Ids out of range are an error.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RoutedLinear {
    pub input_mode: RoutedInputMode,
    pub pair: bool,
    pub bias: bool,
    pub activation: RoutedActivation,
}

impl RoutedLinear {
    fn input_count(&self) -> usize {
        4 + self.pair as usize + if self.bias { 1 + self.pair as usize } else { 0 }
    }

    fn weight_inputs(&self) -> TVec<usize> {
        if self.pair { tvec!(3, 4) } else { tvec!(3) }
    }

    fn bias_input(&self, projection: usize) -> Option<usize> {
        self.bias.then(|| 4 + self.pair as usize + projection)
    }
}

/// `weights [E, N, K]` applied to `input` rows: row `rows[r]` through expert `experts[r]`.
fn project(
    input: &Arc<Tensor>,
    weights: &Tensor,
    rows: &[usize],
    experts: &[usize],
) -> TractResult<Array2<f32>> {
    let (num_experts, n, k) = (weights.shape()[0], weights.shape()[1], weights.shape()[2]);
    let mut by_expert = vec![vec![]; num_experts];
    for (route, &expert) in experts.iter().enumerate() {
        by_expert[expert].push(route);
    }
    let mut output = Array2::<f32>::zeros((rows.len(), n));
    let packed = if weights.storage_as::<BlockQuantStorage>().is_some() {
        let groups = (0..num_experts)
            .map(|e| block_quant_group_tensor(weights, e))
            .collect::<TractResult<Vec<_>>>()?;
        build_block_quant_routed_matmul(groups).ok()
    } else {
        None
    };
    let plain = if weights.storage_as::<BlockQuantStorage>().is_none() {
        Some(weights.cast_to::<f32>()?.into_owned())
    } else {
        None
    };
    let x = input.to_plain_array_view::<f32>()?.into_dimensionality::<tract_ndarray::Ix2>()?;
    let mut state = PreparedRoutedMatMulState::default();
    let (row_stride, k_stride) =
        (usize::try_from(input.strides()[0])?, usize::try_from(input.strides()[1])?);
    for (expert, routes) in by_expert.iter().enumerate() {
        if routes.is_empty() {
            continue;
        }
        let expert_output = if let Some(plan) = &packed {
            let offsets = routes.iter().map(|&r| row_stride * rows[r]).collect();
            let mut out = Tensor::zero::<f32>(&[routes.len(), n])?;
            run_prepared_routed_matmul(
                plan,
                expert,
                input,
                RoutedInputRows::explicit(offsets, k_stride),
                &mut out,
                &mut state,
            )?;
            out.into_plain_array::<f32>()?.into_dimensionality()?
        } else {
            let gathered = x.select(Axis(0), &routes.iter().map(|&r| rows[r]).collect::<Vec<_>>());
            let w = if let Some(plain) = &plain {
                plain.to_plain_array_view::<f32>()?.slice_move(s![expert, .., ..]).to_owned()
            } else {
                block_quant_group_as_2d(weights, expert)?
                    .into_plain_array::<f32>()?
                    .into_dimensionality()?
            };
            ensure!(w.shape() == [n, k]);
            gathered.dot(&w.t())
        };
        for (slot, &route) in routes.iter().enumerate() {
            output.row_mut(route).assign(&expert_output.row(slot));
        }
    }
    Ok(output)
}

fn add_bias(
    output: &mut Array2<f32>,
    bias: &Tensor,
    experts: &[usize],
    num_experts: usize,
) -> TractResult<()> {
    let bias = bias.cast_to::<f32>()?;
    let bias: ArrayView2<f32> = bias.to_plain_array_view::<f32>()?.into_dimensionality()?;
    ensure!(bias.shape() == [num_experts, output.shape()[1]], "routed bias shape mismatch");
    for (mut row, &expert) in output.rows_mut().into_iter().zip(experts) {
        row += &bias.row(expert);
    }
    Ok(())
}

impl Op for RoutedLinear {
    fn name(&self) -> StaticName {
        "RoutedLinear".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "{:?}, pair: {}, bias: {}, activation: {:?}",
            self.input_mode, self.pair, self.bias, self.activation
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for RoutedLinear {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == self.input_count());
        let input = f32_input(&inputs[0])?;
        let token_ids = inputs[1].cast_to::<i64>()?;
        let expert_ids = inputs[2].cast_to::<i64>()?;
        let token_ids = plain_i64_slice(&token_ids, "route_token_ids")?;
        let expert_ids = plain_i64_slice(&expert_ids, "route_expert_ids")?;
        ensure!(token_ids.len() == expert_ids.len());
        let route_count = token_ids.len();
        let (input_rows, k) = (input.shape()[0], input.shape()[1]);
        let num_experts = inputs[3].shape()[0];

        let rows = (0..route_count)
            .map(|r| {
                let row = match self.input_mode {
                    RoutedInputMode::TokenRows => token_ids[r],
                    RoutedInputMode::RouteRows => r as i64,
                };
                ensure!(
                    (0..input_rows as i64).contains(&row),
                    "route {r} reads input row {row}, but input has {input_rows} rows"
                );
                Ok(row as usize)
            })
            .collect::<TractResult<Vec<_>>>()?;
        let experts = expert_ids
            .iter()
            .enumerate()
            .map(|(r, &e)| {
                ensure!(
                    (0..num_experts as i64).contains(&e),
                    "route {r} references expert {e}, but weights have {num_experts} experts"
                );
                Ok(e as usize)
            })
            .collect::<TractResult<Vec<_>>>()?;

        let mut projections = tvec!();
        for (projection, &w) in self.weight_inputs().iter().enumerate() {
            let weights = inputs[w].clone().into_tensor();
            ensure!(weights.shape()[0] == num_experts && weights.shape()[2] == k);
            let mut out = project(&input, &weights, &rows, &experts)?;
            if let Some(b) = self.bias_input(projection) {
                add_bias(&mut out, &inputs[b], &experts, num_experts)?;
            }
            projections.push(out);
        }
        let mut output = projections.remove(0);
        if let Some(up) = projections.pop() {
            output.zip_mut_with(&up, |g, &u| *g = self.activation.combine(*g, u));
        } else {
            output.mapv_inplace(|v| self.activation.apply(v));
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for RoutedLinear {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == self.input_count(),
            "RoutedLinear expects {} inputs",
            self.input_count()
        );
        ensure!(
            self.pair || !matches!(self.activation, RoutedActivation::ClampedSwiGlu { .. }),
            "clamped SwiGLU needs a gate/up pair"
        );
        ensure!(inputs[0].rank() == 2, "RoutedLinear input must be [rows, K]");
        ensure!(matches!(inputs[0].datum_type, DatumType::F16 | DatumType::F32));
        for ids in &inputs[1..3] {
            ensure!(ids.rank() == 1 && ids.datum_type == i64::datum_type());
        }
        ensure!(inputs[1].shape == inputs[2].shape, "route ids must have the same length");
        let w = inputs[3];
        ensure!(w.rank() == 3, "RoutedLinear weights must be [E, N, K]");
        ensure!(w.shape[2] == inputs[0].shape[1], "RoutedLinear K mismatch");
        for &ix in &self.weight_inputs()[1..] {
            ensure!(inputs[ix].shape == w.shape, "gate and up weights must have the same shape");
        }
        for projection in 0..self.weight_inputs().len() {
            if let Some(b) = self.bias_input(projection) {
                ensure!(inputs[b].shape.dims() == &w.shape.dims()[..2], "bias must be [E, N]");
            }
        }
        Ok(tvec!(f32::datum_type().fact(&[inputs[1].shape[0].clone(), w.shape[1].clone()])))
    }

    as_op!();
}

impl MoeFfn {
    /// `node` as an f32 router [`EinSum`] and [`RouteSelect`], a [`RoutedLinear`] gate (and up) projection with its activation, a
    /// [`RoutedLinear`] down projection and [`RoutedCombine`]: the form GPU transforms lower,
    /// one op at a time. `None` when the experts are block-quantized in the canonical layout,
    /// whose blocks do not run along K, or when weight shapes are symbolic.
    pub fn routed_linear_patch(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let activation = match (self.activation.as_str(), self.act_limit_bits) {
            (_, Some(limit_bits)) if self.has_w3 => RoutedActivation::ClampedSwiGlu {
                alpha_bits: self.act_alpha().to_bits(),
                limit_bits,
            },
            ("silu", None) => RoutedActivation::Silu,
            ("swiglu", None) if self.has_w3 => RoutedActivation::Silu,
            ("gelu", None) => RoutedActivation::Gelu,
            ("relu", None) => RoutedActivation::Relu,
            _ => return Ok(None),
        };
        let idx = self.input_idx();
        let facts = model.node_input_facts(node.id)?;
        let experts = [Some(2), idx.w3, Some(3)];
        for ix in experts.iter().flatten() {
            rule_if!(facts[*ix].shape.as_concrete().is_some() && facts[*ix].rank() == 3);
            rule_if!(
                self.expert_layout == ExpertLayout::Linear || facts[*ix].exotic_fact().is_none()
            );
        }

        let mut patch = TypedModelPatch::default();
        let name = &node.name;
        let tap = |patch: &mut TypedModelPatch, ix: usize| patch.tap_model(model, node.inputs[ix]);
        let x = tap(&mut patch, 0)?;
        let x2 = if facts[0].rank() == 3 {
            let dims = facts[0].shape.dims();
            let tokens = dims[0].clone() * &dims[1];
            let reshape = AxisOp::Reshape(0, dims[..2].into(), tvec!(tokens));
            patch.wire_node(format!("{name}.tokens"), reshape, &[x])?[0]
        } else {
            x
        };
        let f32_of = |patch: &mut TypedModelPatch, wire: OutletId, dt: DatumType, what: &str| {
            if dt == f32::datum_type() {
                Ok(wire)
            } else {
                patch
                    .wire_node(format!("{name}.{what}.f32"), cast(f32::datum_type()), &[wire])
                    .map(|o| o[0])
            }
        };
        let x_f32 = f32_of(&mut patch, x2, facts[0].datum_type, "x")?;
        let mut wg = tap(&mut patch, 1)?;
        if facts[1].rank() == 3 {
            wg = patch.wire_node(format!("{name}.wg.rank2"), AxisOp::Rm(0), &[wg])?[0];
        }
        let wg = f32_of(&mut patch, wg, facts[1].datum_type, "wg")?;
        let scores = patch.wire_node(
            format!("{name}.router"),
            EinSum::new("td,ed->te".parse()?, f32::datum_type()),
            &[x_f32, wg],
        )?[0];
        let mut route_inputs = tvec!(scores);
        let f32_input = |patch: &mut TypedModelPatch, ix: usize| -> TractResult<OutletId> {
            let wire = tap(patch, ix)?;
            f32_of(patch, wire, facts[ix].datum_type, &format!("input{ix}"))
        };
        if let Some(b) = idx.wg_bias {
            route_inputs.push(f32_input(&mut patch, b)?);
        }
        let routes = patch.wire_node(
            format!("{name}.route_select"),
            RouteSelect { k: self.k, gate: self.gate.clone() },
            &route_inputs,
        )?;

        let linear_weights = |patch: &mut TypedModelPatch, ix: usize| -> TractResult<_> {
            let w = tap(patch, ix)?;
            if self.expert_layout == ExpertLayout::Linear {
                Ok(w)
            } else {
                Ok(patch.wire_node(format!("{name}.w{ix}.linear"), AxisOp::Move(1, 2), &[w])?[0])
            }
        };
        let w1 = linear_weights(&mut patch, 2)?;
        let w3 = idx.w3.map(|ix| linear_weights(&mut patch, ix)).transpose()?;
        let w2 = linear_weights(&mut patch, 3)?;

        let hidden_dim = |w: usize| {
            let dims = facts[w].shape.as_concrete().unwrap();
            if self.expert_layout == ExpertLayout::Linear { dims[1] } else { dims[2] }
        };
        let bias_or_zero = |patch: &mut TypedModelPatch, bias: Option<usize>, w: usize| {
            if let Some(ix) = bias {
                return f32_input(patch, ix);
            }
            let zero = Tensor::zero::<f32>(&[facts[w].shape[0].to_usize()?, hidden_dim(w)])?;
            Ok(patch.wire_node(format!("{name}.zero_bias{w}"), Const::new(zero.into())?, &[])?[0])
        };
        let gate_bias = self.has_w1_bias || (self.has_w3 && self.has_w3_bias);
        let mut gate_inputs = tvec!(x_f32, routes[0], routes[1], w1);
        gate_inputs.extend(w3);
        if gate_bias {
            gate_inputs.push(bias_or_zero(&mut patch, idx.w1_bias, 2)?);
            if let Some(w3_ix) = idx.w3 {
                gate_inputs.push(bias_or_zero(&mut patch, idx.w3_bias, w3_ix)?);
            }
        }
        let hidden = patch.wire_node(
            format!("{name}.gate_up"),
            RoutedLinear {
                input_mode: RoutedInputMode::TokenRows,
                pair: self.has_w3,
                bias: gate_bias,
                activation,
            },
            &gate_inputs,
        )?[0];
        let mut down_inputs = tvec!(hidden, routes[0], routes[1], w2);
        if let Some(b) = idx.w2_bias {
            down_inputs.push(f32_input(&mut patch, b)?);
        }
        let values = patch.wire_node(
            format!("{name}.down"),
            RoutedLinear {
                input_mode: RoutedInputMode::RouteRows,
                pair: false,
                bias: self.has_w2_bias,
                activation: RoutedActivation::None,
            },
            &down_inputs,
        )?[0];
        let output = patch.wire_node(
            format!("{name}.combine"),
            RoutedCombine,
            &[x, values, routes[0], routes[2]],
        )?[0];
        patch.shunt_outside(model, node.id.into(), output)?;
        Ok(Some(patch))
    }
}
