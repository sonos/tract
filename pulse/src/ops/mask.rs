/// Pulsifier for `Iff` when the condition wire carries a chunk-window
/// `uniform_tdim` expression.
///
/// **L == 0 (no lookback):** the mask is always all-true.  Elide the Iff and
/// wire the true branch directly.
///
/// **L > 0 (left-chunk lookback):** the mask has a startup phase: for the
/// first L chunks the K Delay buffer is zero-padded, and those L*P positions
/// should be masked to -inf.  We wire a `ChunkWindowMask` stateful op that
/// produces the correct `[P, (L+1)*P]` bool mask at each chunk, then keep the
/// Iff but replace its condition with ChunkWindowMask's output.
///
/// For the false branch (fill): rather than using the pulsed false-branch wire
/// (which may have incorrect shape due to `MultiBroadcastTo([S,S])` substituting
/// S → P in both axes instead of P and (L+1)×P), we extract the scalar fill value
/// from the source model and wire a fresh `Const` scalar.  The `Iff` then
/// broadcasts it correctly against the [P,(L+1)P] true branch.
use crate::internal::*;
use crate::model::{NonPulsingWrappingOp, PulseWrappingOp};
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::{BitNot, Iff, classify_chunk_window};
use tract_nnef::tract_core::trivial_op_state_freeze;

register_all!(Iff: pulsify);

// ── ChunkWindowMask ────────────────────────────────────────────────────────

/// Stateful op that produces a `[P, (L+1)*P]` bool mask for each chunk.
///
/// At chunk c, column j is True iff `j >= max(0, (L - c) * P)`.
/// After `left_chunks` chunks the mask is all-True and stays that way.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ChunkWindowMask {
    pub left_chunks: usize,
    pub pulse_size: usize,
    pub key_window: usize, // (left_chunks + 1) * pulse_size
}

#[derive(Debug, Clone)]
struct ChunkWindowMaskState {
    chunk: usize,
    left_chunks: usize,
    pulse_size: usize,
    key_window: usize,
}

impl OpState for ChunkWindowMaskState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        _op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let c = self.chunk;
        self.chunk += 1;
        // First (L - c)*P positions are zero-padded K from the Delay buffer → mask False.
        let first_valid = self.left_chunks.saturating_sub(c) * self.pulse_size;
        let mut data = vec![false; self.pulse_size * self.key_window];
        for p in 0..self.pulse_size {
            for j in 0..self.key_window {
                data[p * self.key_window + j] = j >= first_valid;
            }
        }
        let tensor = Tensor::from_shape(&[self.pulse_size, self.key_window], &data)?;
        Ok(tvec!(tensor.into_tvalue()))
    }
}

trivial_op_state_freeze!(ChunkWindowMaskState);

impl Op for ChunkWindowMask {
    fn name(&self) -> StaticName {
        "ChunkWindowMask".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "left_chunks={} pulse_size={} key_window={}",
            self.left_chunks, self.pulse_size, self.key_window
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for ChunkWindowMask {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(ChunkWindowMaskState {
            chunk: 0,
            left_chunks: self.left_chunks,
            pulse_size: self.pulse_size,
            key_window: self.key_window,
        })))
    }
}

impl TypedOp for ChunkWindowMask {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(DatumType::Bool.fact([self.pulse_size, self.key_window])))
    }
}

impl PulsedOp for ChunkWindowMask {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(PulsedFact {
            datum_type: DatumType::Bool,
            shape: tvec![self.pulse_size.to_dim(), self.key_window.to_dim()].into(),
            stream: None, // not a stream-shaped output — just a chunk counter
        }))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

// ── fill value extraction ─────────────────────────────────────────────────

/// Walk the source-model subgraph rooted at `outlet` and try to compute the
/// scalar f32 "fill" value that it evaluates to — without any dynamic inputs.
///
/// Handles:
///   - `fact.uniform` (uniform value already annotated on the fact)
///   - `fact.konst` (the whole tensor is a constant)
///   - `TypedBinOp` whose operands both have uniform/constant values (recursively)
///
/// Returns a scalar f32 if successful, `None` otherwise.
fn try_fill_scalar_f32(source: &TypedModel, outlet: OutletId) -> Option<f32> {
    let fact = source.outlet_fact(outlet).ok()?;
    // Fast path: uniform annotation
    if let Some(u) = &fact.uniform {
        return u.cast_to_scalar::<f32>().ok();
    }
    // Fast path: whole tensor is a constant scalar
    if let Some(k) = &fact.konst {
        if k.len() == 1 {
            return k.cast_to_scalar::<f32>().ok();
        }
    }
    // Recurse: TypedBinOp with recursively-computable operands
    let node = &source.nodes()[outlet.node];
    let bin = node.op_as::<TypedBinOp>()?;
    if node.inputs.len() != 2 {
        return None;
    }
    let a = try_fill_scalar_f32(source, node.inputs[0])?;
    let b = try_fill_scalar_f32(source, node.inputs[1])?;
    // Evaluate the binary op on f32 scalars.
    let result = bin.0.eval(tensor0(a).into(), tensor0(b).into(), DatumType::F32).ok()?;
    result.cast_to_scalar::<f32>().ok()
}

// ── Iff pulsifier ─────────────────────────────────────────────────────────

/// Walk back through the source-model condition graph, peeling `AddAxis`
/// (unsqueeze) and `BitNot` (logical NOT) ops that wrap the real mask.
///
/// Returns `(inner_outlet, inverted)` where `inverted` is `true` when the
/// condition has been NOTted an odd number of times — i.e. the condition is
/// True where the attention position should be *masked out* (fill with -∞).
fn peel_condition(source: &TypedModel, mut outlet: OutletId) -> (OutletId, bool) {
    let mut inverted = false;
    loop {
        let node = &source.nodes()[outlet.node];
        if node.inputs.len() != 1 {
            break;
        }
        if node.op_as::<AxisOp>().is_some() {
            outlet = node.inputs[0];
            continue;
        }
        if let Some(ew) = node.op_as::<ElementWiseOp>() {
            if ew.0.is::<BitNot>() {
                inverted = !inverted;
                outlet = node.inputs[0];
                continue;
            }
        }
        break;
    }
    (outlet, inverted)
}

fn pulsify(
    _op: &Iff,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // inputs[0] = condition, inputs[1] = true branch, inputs[2] = false branch
    //
    // Two conventions appear in practice:
    //   Normal   (inverted=false): condition=True → keep score (inputs[1])
    //   Inverted (inverted=true):  condition=True → fill with -∞ (inputs[1]),
    //                              condition=False → keep score (inputs[2])
    //
    // We peel AddAxis / BitNot wrappers from the condition to find the
    // underlying chunk-window mask and detect which convention is in use.
    let raw_cond = node.inputs[0];
    let (inner_outlet, inverted) = peel_condition(source, raw_cond);
    let inner_fact = source.outlet_fact(inner_outlet)?;

    // Try to obtain a chunk-window uniform_tdim expression.
    // The stored output fact may not have it when FoldUniformTDim introduced a
    // UniformTDim node but did not re-propagate uniform_tdim to successor nodes
    // (e.g. attMask_1 = And(padMask, UniformTDim) may have uniform_tdim=None
    // on its output fact even though its UniformTDim input carries it).
    // Fall back to scanning the inner node's inputs.
    let expr_opt = inner_fact.uniform_tdim.as_ref().map(|e| e.clone().simplify()).or_else(|| {
        let inner_node = &source.nodes()[inner_outlet.node];
        inner_node.inputs.iter().find_map(|inp| {
            let f = source.outlet_fact(*inp).ok()?;
            let e = f.uniform_tdim.as_ref()?;
            classify_chunk_window(&e.clone().simplify())?;
            Some(e.clone().simplify())
        })
    });
    let expr = match expr_opt {
        Some(e) => e,
        None => return Ok(None),
    };
    let cw = match classify_chunk_window(&expr) {
        Some(cw) => cw,
        None => return Ok(None),
    };

    let left_chunks = cw.left_chunks as usize;
    let pulse_size = cw.p as usize;

    // For the inverted convention: condition is True when masked.
    // At left_chunks=0 all positions are valid, so condition is all-False →
    // always take the false branch (inputs[2] = scores).
    // For normal convention: all-True → always take inputs[1] (scores).
    if left_chunks == 0 {
        let keep_wire = if inverted { node.inputs[2] } else { node.inputs[1] };
        return Ok(Some(tvec![mapping[&keep_wire]]));
    }

    let key_window = (left_chunks + 1) * pulse_size;

    // Wire ChunkWindowMask (no inputs): produces [P, (L+1)*P] bool, True = in-window.
    let mask_wire_2d = target.wire_node(
        format!("{}.chunk_window_mask", node.name),
        ChunkWindowMask { left_chunks, pulse_size, key_window },
        &[],
    )?[0];

    // ChunkWindowMask is True for in-window positions (normal convention).
    // For the inverted convention the Iff has scores as inputs[2] and fill as
    // inputs[1], so we swap which source input we treat as "true" (score) wire.
    let scores_input = if inverted { node.inputs[2] } else { node.inputs[1] };
    let fill_input = if inverted { node.inputs[1] } else { node.inputs[2] };

    let true_wire = mapping[&scores_input];
    let true_rank = target.outlet_fact(true_wire)?.shape.len();
    let true_dtype = target.outlet_fact(true_wire)?.datum_type;

    // Promote mask from [P, kw] (rank 2) to [1,...,1,P,kw] (rank = true_rank).
    let mask_wire = {
        let mut w = mask_wire_2d;
        for leading in 0..true_rank.saturating_sub(2) {
            w = target.wire_node(
                format!("{}.mask_unsqueeze_{leading}", node.name),
                NonPulsingWrappingOp(Box::new(AxisOp::Add(0))),
                &[w],
            )?[0];
        }
        w
    };

    // Fill branch: extract the scalar fill value from the source model.
    // The pulsed fill-branch wire may have the wrong shape (MultiBroadcastTo([S,S])
    // substitutes S→P in both axes, giving [P,P] instead of [P,(L+1)P]).
    // Instead, create a fresh scalar Const which broadcasts correctly.
    let fill_wire = if let Some(fill_f32) = try_fill_scalar_f32(source, fill_input) {
        let fill_shape = vec![1usize; true_rank];
        let fill_tensor =
            Tensor::from_shape(&fill_shape, &[fill_f32])?.cast_to_dt(true_dtype)?.into_owned();
        target.wire_node(
            format!("{}.fill", node.name),
            NonPulsingWrappingOp(Box::new(Const::new(fill_tensor.into_arc_tensor())?)),
            &[],
        )?[0]
    } else {
        mapping[&fill_input]
    };

    // Wire Iff(ChunkWindowMask=True_in_window, true=scores, false=fill).
    // Streaming axis propagates from the scores (true) branch.
    Ok(Some(target.wire_node(
        &node.name,
        PulseWrappingOp(Box::new(Iff)),
        &[mask_wire, true_wire, fill_wire],
    )?))
}
