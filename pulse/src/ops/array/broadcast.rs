use crate::fact::StreamInfo;
use crate::internal::*;
use tract_pulse_opl::tract_core::ops::array::MultiBroadcastTo;

register_all!(MultiBroadcastTo: pulsify);

fn pulsify(
    op: &MultiBroadcastTo,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    if let Some(axis) = op.shape.iter().position(|dim| dim.symbols().contains(symbol)) {
        let full_dim = op.shape[axis].clone();
        let fact = PulsedFact {
            datum_type: _source.outlet_fact(node.inputs[0])?.datum_type,
            shape: op
                .shape
                .iter()
                .enumerate()
                .map(|(i, dim)| {
                    if i == axis {
                        pulsified_stream_axis_dim(dim, symbol, pulse)
                    } else {
                        dim.substitute(symbol, pulse)
                    }
                })
                .collect::<TractResult<_>>()?,
            stream: Some(StreamInfo { axis, dim: full_dim, delay: 0 }),
        };
        let new_op = PulsedMultibroadcastTo { fact };
        target.wire_node(&node.name, new_op, &[mapping[&node.inputs[0]]]).map(Some)
    } else {
        Ok(None)
    }
}

/// Compute the per-pulse size for a `MultiBroadcastTo` target axis whose
/// shape mentions the streaming symbol.
///
/// The canonical pattern emitted by ONNX `Expand` / `BroadcastTo` against
/// a `shape_of(streaming)` chain is a *linear* dim of the form
/// `pulse_growth(S) + boundary` with `pulse_growth(0) = 0`, e.g. a
/// stride-2 conv's output length `1 + S/2`. For that pattern the per-pulse
/// increment is `dim(P) - dim(0)` and we use it as the pulse-axis size.
///
/// When the expression is constant over `[0, P]` or non-linear in `S`,
/// the same subtraction can collapse `full - base` to `0` while `full`
/// itself is a positive valid shape. That happens for chunked-batch
/// expressions like `1 + -1*min(2, -1+(8·S)/5) + (8·S)/5` (which equals
/// `max(2, (8·S)/5 - 1)`), where every `S ∈ {0, P, 2P}` resolves to the
/// same lower bound. The consumer of such an axis (a Scan body state
/// init, an elementwise meet point) reads the *full* per-pulse shape,
/// not an empty delta — emitting a `0`-volume PulsedFact poisons every
/// downstream fact.
///
/// Heuristic: probe at `S=0`, `S=P`, and `S=2P`. Use the linear
/// subtraction iff the delta is strictly positive and `delta(2P) ==
/// 2·delta(P)` (provably linear over the probe interval). Otherwise
/// fall back to `dim(P)`.
fn pulsified_stream_axis_dim(dim: &TDim, symbol: &Symbol, pulse: &TDim) -> TractResult<TDim> {
    let full = dim.substitute(symbol, pulse)?;
    let base = dim.substitute(symbol, &TDim::Val(0))?;
    let delta = full.clone() - base.clone();
    // Constant on `[0, P]` — this axis is not actually streaming on this
    // pulse window. Use the full value so downstream facts stay
    // non-degenerate.
    if delta == 0.to_dim() {
        return Ok(full);
    }
    // Confirm linearity by sampling at `2P`. Only worthwhile when `P` is
    // a concrete positive integer; for symbolic `pulse` the trick falls
    // back to the existing behavior (treat as linear).
    if let Some(pulse_v) = pulse.as_i64()
        && pulse_v > 0
    {
        let double = dim.substitute(symbol, &TDim::Val(pulse_v * 2))?;
        let delta_double = double - base;
        if delta_double != delta.clone() * 2 {
            return Ok(full);
        }
    }
    Ok(delta)
}

/// Concat with pulse along concat axis
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PulsedMultibroadcastTo {
    fact: PulsedFact,
}

impl Op for PulsedMultibroadcastTo {
    fn name(&self) -> StaticName {
        "PulsedMultibroadcastTo".into()
    }

    op_as_typed_op!();
}

impl TypedOp for PulsedMultibroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(self.fact.to_pulse_fact().shape)))
    }
    as_op!();
}

impl EvalOp for PulsedMultibroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.to_typed().eval(inputs)
    }
}

impl PulsedOp for PulsedMultibroadcastTo {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(MultiBroadcastTo { shape: self.fact.to_pulse_fact().shape })
    }

    as_op!();
}
