/// PulseV2: region/increment-based pulsification.
///
/// Symbols:
///   T — pulse index (0, 1, 2, …)
///   P — pulse size (kept symbolic until runtime)
///
/// Each op declares a RegionTransform: "given my output region, what input
/// regions do I need?" The generic pulsifier compares what each op needs
/// against what the source provides, and inserts PulseV2Buffer where there's
/// a gap (lookback into previous pulses).
///
/// Output shapes are computed by the batch model's output_facts — no separate
/// region propagation needed on the forward path.
use crate::internal::*;

/// Build the 3-parameter canonical pulse-ramp shape:
/// `min(cap, max(0, min((T+1)·P, end) − max(T·P, overlap)))`.
///
/// PulseV2 emits streaming-axis shapes in this form so that two parallel
/// paths arriving at an Add merge with equal `(cap, overlap, end)` produce
/// identical TDim trees — broadcast's `.dedup()` then collapses them
/// trivially, avoiding exponential nesting of `Max` in the simplifier.
pub(crate) fn pulse_ramp_3_dim(
    p_sym: &Symbol,
    t_sym: &Symbol,
    cap: TDim,
    overlap: TDim,
    end: TDim,
) -> TDim {
    let p = TDim::Sym(p_sym.clone());
    let t = TDim::Sym(t_sym.clone());
    let tp = p.clone() * t.clone();
    let tp_plus_p = (t + 1) * p;
    let inner_min = TDim::Min(vec![tp_plus_p, end]);
    let inner_max = TDim::Max(vec![tp, overlap]);
    let diff = inner_min - inner_max;
    TDim::Min(vec![cap, TDim::Max(vec![TDim::Val(0), diff])])
}

/// Extract `(cap, overlap, end)` from a wire's streaming-axis shape.
///
/// If the wire's shape is canonical 3-param, read it directly. Otherwise the
/// only non-canonical op we know how to see through is `PulseV2Buffer`:
/// buffer's output is `input + H_i` (not a pulse-ramp), but *conceptually*
/// buffer extends cap by its lookback without touching overlap or end.
/// We compose that rule into the upstream descriptor as we recurse — so a
/// downstream Conv looking at `Source → Buffer → Conv` extracts the triple
/// that represents "buffer's output as if it were canonical", which is what
/// Conv's own rule composes onto.
pub(crate) fn extract_stream_descriptor(
    typed: &TypedModel,
    outlet: OutletId,
    streaming_axis: usize,
    stream_sym: &Symbol,
    p_sym: &Symbol,
    t_sym: &Symbol,
) -> Option<(TDim, TDim, TDim)> {
    use crate::v2_buffer::PulseV2Buffer;

    let fact = typed.outlet_fact(outlet).ok()?;
    let shape_dim = fact.shape.get(streaming_axis)?;
    if let Some((cap, overlap, end)) = as_pulse_ramp_3(shape_dim, p_sym, t_sym) {
        return Some((cap, overlap, end));
    }
    // Plain `P` (e.g., a Source we haven't rewritten) → seed descriptor.
    // `end = i64::MAX` is the "no-truncation" sentinel: the runner zero-pads
    // past the stream end, so Source's runtime is always `P` (not `min(P, S-T*P)`).
    // The simplifier filters Val(i64::MAX) out of Min terms (tree.rs:785), so the
    // canonical form collapses to a 2-param ramp matching runtime. Slice's rule
    // `end = min(end_in, slice.end)` introduces the real `S` truncation only
    // where it's semantically warranted — slice's runtime *does* truncate.
    let _ = stream_sym; // kept on signature for future use
    if shape_dim == &TDim::Sym(p_sym.clone()) {
        return Some((TDim::Sym(p_sym.clone()), TDim::Val(0), TDim::Val(i64::MAX)));
    }
    // Walk past a PulseV2Buffer, applying buffer's rule (cap += lookback).
    let node = &typed.nodes()[outlet.node];
    if let Some(buffer) = node.op.downcast_ref::<PulseV2Buffer>() {
        let buffer_input = node.inputs.first().copied()?;
        let (cap_in, overlap_in, end_in) = extract_stream_descriptor(
            typed,
            buffer_input,
            streaming_axis,
            stream_sym,
            p_sym,
            t_sym,
        )?;
        let delta = buffer.lookback.get(streaming_axis).copied().unwrap_or(0);
        return Some((cap_in + TDim::Val(delta as i64), overlap_in, end_in));
    }
    None
}

// ── Per-axis region ────────────────────────────────────────────────────

/// Per-axis specification at pulse T.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisRegion {
    /// This axis is not streaming — full extent every pulse.
    Fixed(TDim),
    /// This axis produces [start, end) at pulse T.
    /// start and end are TDim expressions in symbols T and P.
    Streaming { start: TDim, end: TDim },
}

impl AxisRegion {
    pub fn size(&self) -> TDim {
        match self {
            AxisRegion::Fixed(d) => d.clone(),
            AxisRegion::Streaming { start, end } => end.clone() - start.clone(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        matches!(self, AxisRegion::Streaming { .. })
    }
}

/// Region description for a wire at pulse T.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PulseV2Region {
    pub axes: TVec<AxisRegion>,
}

impl PulseV2Region {
    pub fn rank(&self) -> usize {
        self.axes.len()
    }
}

// ── Region transforms (inventory) ──────────────────────────────────────

/// Result of a region transform: either adjust input regions (the generic
/// pulsifier handles buffer insertion and op wiring) or replace the op
/// entirely with pre-wired outlets.
pub enum PulseV2Action {
    /// Use these input regions. None means "not streaming, pass as-is".
    /// overlap: optional per-input, per-axis overlap (un-rounded).
    /// When provided, lookback = region difference (rounded) and
    /// overlap < lookback. The buffer trims (lookback - overlap) from the front.
    InputRegions(TVec<Option<PulseV2Region>>, Option<TVec<TVec<usize>>>),
    /// Skip this op — just forward the mapped inputs as outputs.
    Skip,
    /// Replace this op with a different one (same inputs).
    ReplaceOp(Box<dyn TypedOp>),
    /// Wire the original op normally, then append a post-processing op on its output.
    WireOpThenPostOp(Box<dyn TypedOp>),
    /// Wire a pre-op on the data input (index 0), then wire the replacement op.
    /// Used for decomposing Conv(padded) into PulseV2Pad + Conv(valid).
    WirePreOpThenOp { pre_op: Box<dyn TypedOp>, main_op: Box<dyn TypedOp> },
}

pub type RegionTransformFn = fn(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>>;

/// Inventory entry: maps an op TypeId to its region transform.
pub struct RegionTransform {
    pub type_id: std::any::TypeId,
    pub func: RegionTransformFn,
}

inventory::collect!(RegionTransform);

fn lookup_region_transform(
    op: &dyn TypedOp,
    source_region: &PulseV2Region,
    symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let type_id = op.type_id();
    for rt in inventory::iter::<RegionTransform> {
        if rt.type_id == type_id {
            return (rt.func)(op, source_region, symbols);
        }
    }
    Ok(None)
}

// ── Symbols ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PulseV2Symbols {
    pub stream: Symbol,
    pub pulse: Symbol,
    pub pulse_id: Symbol,
}

// ── PulseV2Model ───────────────────────────────────────────────────────

pub struct PulseV2Model {
    pub typed: TypedModel,
    pub symbols: PulseV2Symbols,
}

impl PulseV2Model {
    /// Rewrite the streaming-axis dim of `new_outlet` into canonical 3-param
    /// pulse-ramp form.
    ///
    /// Pulls `(cap, overlap, end)` from the first streaming input (via
    /// [`extract_stream_descriptor`], which walks past `PulseV2Buffer`
    /// nodes) and applies a per-op rule:
    ///
    /// * [`PulseV2Buffer`]: left alone (its natural shape is
    ///   `input + H_i`, which captures the per-pulse semantics accurately
    ///   including end-of-stream history carryover).
    /// * [`PulseV2Slice`] on the streaming axis: `overlap += slice.start`,
    ///   `end = min(end, slice.end)`.
    /// * `Conv` (stride 1): `cap -= (K-1)·D`, `overlap += (K-1)·D`, `end` unchanged.
    ///   The Buffer+Conv pair is net-zero on cap (Buffer adds lookback,
    ///   Conv subtracts it back); these two terms cancel only because
    ///   `extract_stream_descriptor` walking past `PulseV2Buffer` adds
    ///   `+lookback` to `cap`. Removing one without removing the other
    ///   double-counts the cap shift.
    /// * Anything else: pass-through (default).
    ///
    /// Returns silently when the op isn't on a streaming axis or when no
    /// canonical descriptor can be extracted from any input.
    fn canonicalize_outlet(
        typed: &mut TypedModel,
        new_outlet: OutletId,
        op: &dyn TypedOp,
        input_outlets: &[OutletId],
        wire_regions: &HashMap<OutletId, PulseV2Region>,
        stream_sym: &Symbol,
        p_sym: &Symbol,
        t_sym: &Symbol,
    ) -> TractResult<()> {
        use crate::v2_buffer::PulseV2Buffer;
        use crate::v2_slice::PulseV2Slice;
        use tract_pulse_opl::tract_core::ops::cnn::Conv;

        if op.downcast_ref::<PulseV2Buffer>().is_some() {
            return Ok(());
        }

        let Some(streaming_axis) = wire_regions
            .get(&new_outlet)
            .and_then(|r| r.axes.iter().position(|a| a.is_streaming()))
        else {
            return Ok(());
        };

        let Some(in_outlet) = input_outlets
            .iter()
            .find(|o| wire_regions.get(o).is_some_and(|r| r.axes.iter().any(|a| a.is_streaming())))
        else {
            return Ok(());
        };
        let Some((cap_in, overlap_in, end_in)) =
            extract_stream_descriptor(typed, *in_outlet, streaming_axis, stream_sym, p_sym, t_sym)
        else {
            return Ok(());
        };

        let (cap_out, overlap_out, end_out) = if let Some(slice) = op.downcast_ref::<PulseV2Slice>()
        {
            if slice.axis == streaming_axis {
                let end = TDim::Min(vec![end_in, slice.end.clone()]).simplify();
                (cap_in, (overlap_in + slice.start.clone()).simplify(), end)
            } else {
                (cap_in, overlap_in, end_in)
            }
        } else if let Some(conv) = op.downcast_ref::<Conv>() {
            let h_axis = conv.pool_spec.data_format.h_axis();
            if streaming_axis < h_axis {
                return Ok(());
            }
            let geo_ix = streaming_axis - h_axis;
            let Some(k) = conv.pool_spec.kernel_shape.get(geo_ix).copied() else {
                return Ok(());
            };
            let d =
                conv.pool_spec.dilations.as_ref().and_then(|d| d.get(geo_ix).copied()).unwrap_or(1);
            let s =
                conv.pool_spec.strides.as_ref().and_then(|s| s.get(geo_ix).copied()).unwrap_or(1);
            if s != 1 {
                return Ok(());
            }
            let delta = TDim::Val(((k - 1) * d) as i64);
            ((cap_in - delta.clone()).simplify(), (overlap_in + delta).simplify(), end_in)
        } else {
            // Default: pass-through (elementwise / pointwise / shape-preserving).
            (cap_in, overlap_in, end_in)
        };

        let canonical = pulse_ramp_3_dim(p_sym, t_sym, cap_out, overlap_out, end_out).simplify();
        let node = &mut typed.nodes_mut()[new_outlet.node];
        let mut shape = node.outputs[new_outlet.slot].fact.shape.to_tvec();
        shape[streaming_axis] = canonical;
        node.outputs[new_outlet.slot].fact.shape = shape.into();
        Ok(())
    }

    /// Pulsify a batch model.
    ///
    /// For each Source, substitute S → P. For each op, ask its RegionTransform
    /// what input regions it needs. Where an input needs lookback beyond the
    /// source increment, insert a PulseV2Buffer. Wire the op with (potentially
    /// buffered) inputs. Output shapes are computed by the op's output_facts.
    pub fn new(batch_model: &TypedModel, stream_sym: Symbol) -> TractResult<Self> {
        use crate::v2_buffer::PulseV2Buffer;

        // Pre-process: decompose Conv/MaxPool with non-valid padding on
        // streaming axes into Pad + Conv/MaxPool(valid).
        let batch_model = Self::decompose_streaming_padding(batch_model, &stream_sym)?;
        let batch_model = &batch_model;

        let p_sym = batch_model.symbols.sym("P");
        let t_sym = batch_model.symbols.sym("T");

        // Assert S >= P: the stream must be at least one pulse long.
        // This ensures min(T*P, lookback) is valid — there really are
        // T*P cumulative source samples at pulse T.
        batch_model.symbols.add_assertion(format!("{} >= {}", stream_sym, p_sym)).ok();

        let symbols = PulseV2Symbols {
            stream: stream_sym.clone(),
            pulse: p_sym.clone(),
            pulse_id: t_sym.clone(),
        };

        // The source increment: what one pulse provides on streaming axes.
        // This is the baseline — ops that need more get a buffer.
        let t = TDim::Sym(t_sym.clone());
        let p = TDim::Sym(p_sym.clone());

        let mut typed = TypedModel::default();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        // Track which pulsed wires are streaming and what their source region is.
        let mut wire_regions: HashMap<OutletId, PulseV2Region> = HashMap::new();

        let order = batch_model.eval_order()?;

        for &node_id in &order {
            let node = batch_model.node(node_id);

            // Source: substitute S → P, record the source increment as region.
            if node.op.downcast_ref::<tract_core::ops::source::TypedSource>().is_some() {
                let batch_fact = batch_model.outlet_fact(OutletId::new(node_id, 0))?;
                let mut axes = TVec::new();
                let mut pulse_shape = TVec::new();
                for dim in batch_fact.shape.iter() {
                    if dim.symbols().contains(&stream_sym) && dim == &TDim::Sym(stream_sym.clone())
                    {
                        axes.push(AxisRegion::Streaming {
                            start: t.clone() * p.clone(),
                            end: (t.clone() + 1) * p.clone(),
                        });
                        pulse_shape.push(p.clone());
                    } else {
                        axes.push(AxisRegion::Fixed(dim.clone()));
                        pulse_shape.push(dim.clone());
                    }
                }
                let pulse_fact = batch_fact.datum_type.fact(pulse_shape);
                let new_outlet = typed.add_source(&node.name, pulse_fact)?;
                mapping.insert(OutletId::new(node_id, 0), new_outlet);
                wire_regions.insert(new_outlet, PulseV2Region { axes });
                continue;
            }

            // Find the source region for the first streaming input.
            let source_region = node
                .inputs
                .iter()
                .find_map(|i| mapping.get(i).and_then(|o| wire_regions.get(o)))
                .cloned();

            // Ask the region transform what to do.
            let typed_op: &dyn TypedOp = node.op.as_ref();
            let action = if let Some(src) = &source_region {
                lookup_region_transform(typed_op, src, &symbols)?
            } else {
                None
            };

            // Handle Skip: forward inputs directly as outputs.
            if matches!(action, Some(PulseV2Action::Skip)) {
                for (slot, batch_input) in node.inputs.iter().enumerate() {
                    let pulsed = mapping.get(batch_input).copied().unwrap_or(*batch_input);
                    mapping.insert(OutletId::new(node_id, slot), pulsed);
                }
                if let Some(region) = &source_region {
                    for slot in 0..node.outputs.len() {
                        if let Some(&pulsed) = mapping.get(&OutletId::new(node_id, slot)) {
                            wire_regions.insert(pulsed, region.clone());
                        }
                    }
                }
                continue;
            }

            // Handle ReplaceOp: wire the replacement op instead.
            if let Some(PulseV2Action::ReplaceOp(replacement)) = action {
                let inputs: TVec<OutletId> =
                    node.inputs.iter().map(|i| mapping.get(i).copied().unwrap_or(*i)).collect();
                let replacement_for_canon = replacement.clone();
                let new_outlets = typed.wire_node(&node.name, replacement, &inputs)?;
                for (slot, &new_outlet) in new_outlets.iter().enumerate() {
                    mapping.insert(OutletId::new(node_id, slot), new_outlet);
                }
                if let Some(region) = &source_region {
                    for &new_outlet in &new_outlets {
                        wire_regions.insert(new_outlet, region.clone());
                    }
                }
                for &new_outlet in &new_outlets {
                    Self::canonicalize_outlet(
                        &mut typed,
                        new_outlet,
                        replacement_for_canon.as_ref(),
                        &inputs,
                        &wire_regions,
                        &stream_sym,
                        &p_sym,
                        &t_sym,
                    )?;
                }
                continue;
            }

            // Handle WireOpThenPostOp: wire the original op, then append a post-op.
            if let Some(PulseV2Action::WireOpThenPostOp(post_op)) = action {
                let inputs: TVec<OutletId> =
                    node.inputs.iter().map(|i| mapping.get(i).copied().unwrap_or(*i)).collect();
                let op_outlets = typed.wire_node(&node.name, node.op.clone(), &inputs)?;
                let post_outlets =
                    typed.wire_node(format!("{}.accum", node.name), post_op, &op_outlets)?;
                for (slot, &new_outlet) in post_outlets.iter().enumerate() {
                    mapping.insert(OutletId::new(node_id, slot), new_outlet);
                }
                if let Some(region) = &source_region {
                    for &new_outlet in &post_outlets {
                        wire_regions.insert(new_outlet, region.clone());
                    }
                }
                continue;
            }

            // Handle WirePreOpThenOp: wire pre-op on input 0, then look up
            // the main_op's region transform (for buffer insertion), then wire main_op.
            if let Some(PulseV2Action::WirePreOpThenOp { pre_op, main_op }) = action {
                let mut inputs: TVec<OutletId> =
                    node.inputs.iter().map(|i| mapping.get(i).copied().unwrap_or(*i)).collect();
                // Wire pre-op on the first (data) input.
                let pre_outlets =
                    typed.wire_node(format!("{}.pre", node.name), pre_op, &[inputs[0]])?;
                inputs[0] = pre_outlets[0];
                // Check if the main_op needs a buffer (e.g. Conv(valid) after Pad).
                let main_typed: &dyn TypedOp = main_op.as_ref();
                if let Some(src) = &source_region {
                    if let Some(PulseV2Action::InputRegions(main_regions, _)) =
                        lookup_region_transform(main_typed, src, &symbols)?
                    {
                        // Insert buffer on the data input if needed.
                        if let Some(Some(needed)) = main_regions.first() {
                            let provided = src;
                            let mut lookback = tvec![0usize; needed.rank()];
                            let mut needs_buffer = false;
                            for (ax, (n, p)) in
                                needed.axes.iter().zip(provided.axes.iter()).enumerate()
                            {
                                if let (
                                    AxisRegion::Streaming { start: ns, .. },
                                    AxisRegion::Streaming { start: ps, .. },
                                ) = (n, p)
                                {
                                    let lb = (ps.clone() - ns.clone()).simplify();
                                    if let Ok(v) = lb.to_i64() {
                                        if v > 0 {
                                            lookback[ax] = v as usize;
                                            needs_buffer = true;
                                        }
                                    }
                                }
                            }
                            if needs_buffer {
                                let depth = *lookback.iter().max().unwrap_or(&1);
                                let buffer = PulseV2Buffer {
                                    overlap: lookback.clone(),
                                    lookback,
                                    depth,
                                    pulse_id: t_sym.clone(),
                                    pulse_sym: p_sym.clone(),
                                };
                                let buffered = typed.wire_node(
                                    format!("{}.buffer", node.name),
                                    buffer,
                                    &[inputs[0]],
                                )?;
                                if let Some(src) = &source_region {
                                    wire_regions.insert(buffered[0], src.clone());
                                }
                                inputs[0] = buffered[0];
                            }
                        }
                    }
                }
                let new_outlets = typed.wire_node(&node.name, main_op, &inputs)?;
                for (slot, &new_outlet) in new_outlets.iter().enumerate() {
                    mapping.insert(OutletId::new(node_id, slot), new_outlet);
                }
                if let Some(region) = &source_region {
                    for &new_outlet in &new_outlets {
                        wire_regions.insert(new_outlet, region.clone());
                    }
                }
                continue;
            }

            let (input_regions, overlap_hints): (
                TVec<Option<PulseV2Region>>,
                Option<TVec<TVec<usize>>>,
            ) = match action {
                Some(PulseV2Action::InputRegions(r, o)) => (r, o),
                _ => {
                    let r = if let Some(src) = &source_region {
                        node.inputs.iter().map(|_| Some(src.clone())).collect()
                    } else {
                        tvec![None; node.inputs.len()]
                    };
                    (r, None)
                }
            };

            // Wire inputs, inserting buffers where needed.
            let mut inputs: TVec<OutletId> = TVec::new();
            for (ix, batch_input) in node.inputs.iter().enumerate() {
                let pulsed_input = mapping.get(batch_input).copied().unwrap_or(*batch_input);
                let wire_region = wire_regions.get(&pulsed_input);
                let needed = input_regions.get(ix).and_then(|r| r.as_ref());

                if let (Some(needed), Some(provided)) = (needed, wire_region) {
                    let mut lookback = tvec![0usize; needed.rank()];
                    let mut needs_buffer = false;
                    for (ax, (needed_ax, provided_ax)) in
                        needed.axes.iter().zip(provided.axes.iter()).enumerate()
                    {
                        if let (
                            AxisRegion::Streaming { start: needed_start, .. },
                            AxisRegion::Streaming { start: provided_start, .. },
                        ) = (needed_ax, provided_ax)
                        {
                            let lb = (provided_start.clone() - needed_start.clone()).simplify();
                            if let Ok(v) = lb.to_i64() {
                                if v > 0 {
                                    lookback[ax] = v as usize;
                                    needs_buffer = true;
                                }
                            }
                        }
                    }

                    if needs_buffer {
                        let depth = *lookback.iter().max().unwrap_or(&1);
                        let overlap = overlap_hints
                            .as_ref()
                            .and_then(|h| h.get(ix))
                            .cloned()
                            .unwrap_or_else(|| lookback.clone());
                        let buffer = PulseV2Buffer {
                            overlap,
                            lookback,
                            depth,
                            pulse_id: t_sym.clone(),
                            pulse_sym: p_sym.clone(),
                        };
                        let buffered = typed.wire_node(
                            format!("{}.buffer.{}", node.name, ix),
                            buffer,
                            &[pulsed_input],
                        )?;
                        if let Some(src) = &source_region {
                            wire_regions.insert(buffered[0], src.clone());
                        }
                        inputs.push(buffered[0]);
                    } else {
                        inputs.push(pulsed_input);
                    }
                } else {
                    inputs.push(pulsed_input);
                }
            }

            let new_outlets = typed.wire_node(&node.name, node.op.clone(), &inputs)?;
            for (slot, &new_outlet) in new_outlets.iter().enumerate() {
                mapping.insert(OutletId::new(node_id, slot), new_outlet);
                if let Some(region) = &source_region {
                    wire_regions.insert(new_outlet, region.clone());
                }
            }
            for &new_outlet in &new_outlets {
                Self::canonicalize_outlet(
                    &mut typed,
                    new_outlet,
                    node.op.as_ref(),
                    &inputs,
                    &wire_regions,
                    &stream_sym,
                    &p_sym,
                    &t_sym,
                )?;
            }
        }

        let batch_outputs = batch_model.output_outlets()?.to_vec();
        let pulsed_outputs: TVec<OutletId> = batch_outputs.iter().map(|o| mapping[o]).collect();
        typed.select_output_outlets(&pulsed_outputs)?;
        // Clamp each final pulsed output's streaming dim against going negative
        // during startup pulses (e.g. conv with P < K at T=0). Intermediates
        // were intentionally left flat — nesting `max(0, …)` per node triggers
        // exponential blowup in the TDim simplifier on deep conv chains.
        for &outlet in &pulsed_outputs {
            Self::clamp_negative_streaming_dims(&mut typed, outlet, &t_sym, &p_sym)?;
        }

        Ok(PulseV2Model { typed, symbols })
    }

    /// Clamp any output dimension that depends on T and could evaluate to
    /// negative (e.g. conv output when P < K). Wraps the dim in Max(0, dim).
    fn clamp_negative_streaming_dims(
        model: &mut TypedModel,
        outlet: OutletId,
        pulse_id: &Symbol,
        pulse_sym: &Symbol,
    ) -> TractResult<()> {
        let fact = model.outlet_fact(outlet)?.clone();
        let mut changed = false;
        let mut new_shape = fact.shape.to_tvec();
        for (ax, dim) in fact.shape.iter().enumerate() {
            // Probe multiple worst cases. If ANY evaluates negative, clamp.
            let syms = dim.symbols();
            if !syms.is_empty() {
                // Probe worst cases: various T, P, H values.
                // Any negative result → clamp.
                let probes: &[(i64, i64, i64)] = &[
                    (0, 1, 0),   // T=0, P=1, H=0: startup
                    (0, 100, 0), // T=0, large pulse, H=0
                    (100, 1, 0), // T=large, P=1, H=0: past end, no history
                    (100, 1, 1), // T=large, P=1, H=1: past end, some history
                ];
                let mut needs_clamp = false;
                for &(t_val, p_val, h_val) in probes {
                    let mut sv =
                        SymbolValues::default().with(pulse_id, t_val).with(pulse_sym, p_val);
                    for s in &syms {
                        if format!("{s}").starts_with("H") {
                            sv.set(s, h_val);
                        }
                        if format!("{s}").starts_with("S") {
                            sv.set(s, p_val); // S >= P
                        }
                    }
                    if dim.eval(&sv).to_i64().is_ok_and(|v| v < 0) {
                        needs_clamp = true;
                        break;
                    }
                }
                if needs_clamp {
                    new_shape[ax] = TDim::Max(vec![TDim::Val(0), dim.clone()]);
                    changed = true;
                }
            }
        }
        if changed {
            let node = &mut model.nodes_mut()[outlet.node];
            node.outputs[outlet.slot].fact.shape = new_shape.into();
        }
        Ok(())
    }

    /// Decompose Conv/MaxPool with non-valid padding on streaming axes
    /// into explicit Pad + Conv/MaxPool(valid-on-streaming-axis).
    fn decompose_streaming_padding(
        model: &TypedModel,
        stream_sym: &Symbol,
    ) -> TractResult<TypedModel> {
        use crate::fact::StreamFact;
        use tract_pulse_opl::tract_core::ops::array::{Pad, PadMode};
        use tract_pulse_opl::tract_core::ops::cnn::{Conv, MaxPool, PaddingSpec, PoolSpec};

        let mut new_model = TypedModel::default();
        new_model.symbols = model.symbols.clone();
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();
        let order = model.eval_order()?;
        let mut changed = false;

        for &node_id in &order {
            let node = model.node(node_id);

            // Check if this node is a Conv or MaxPool with non-valid padding
            // on a streaming axis.
            let pool_spec = node
                .op
                .downcast_ref::<Conv>()
                .map(|c| &c.pool_spec)
                .or_else(|| node.op.downcast_ref::<MaxPool>().map(|p| &p.pool_spec));

            if let Some(spec) = pool_spec {
                let input_fact = model.outlet_fact(node.inputs[0])?;
                let stream_axis = input_fact.shape.stream_info(stream_sym).map(|(ax, _)| ax);

                if let Some(stream_ax) = stream_axis {
                    let geo_axis = stream_ax - spec.data_format.h_axis();
                    if geo_axis < spec.kernel_shape.len()
                        && !spec.padding.valid_dim(geo_axis, false)
                    {
                        // Compute padding amounts.
                        let dummy_hw: TVec<usize> =
                            spec.kernel_shape.iter().map(|k| k * 10).collect();
                        let computed = spec.computed_padding(&dummy_hw);

                        // Build Pad op: only pad the streaming axis.
                        let rank = input_fact.rank();
                        let mut pads = vec![(0usize, 0usize); rank];
                        pads[stream_ax] = (
                            computed[geo_axis].pad_before.to_usize().unwrap_or(0),
                            computed[geo_axis].pad_after.to_usize().unwrap_or(0),
                        );

                        // Build new PoolSpec with zero padding on streaming axis.
                        let mut bef = tvec![];
                        let mut aft = tvec![];
                        for ix in 0..spec.kernel_shape.len() {
                            if ix == geo_axis {
                                bef.push(0);
                                aft.push(0);
                            } else {
                                bef.push(computed[ix].pad_before.to_usize().unwrap_or(0));
                                aft.push(computed[ix].pad_after.to_usize().unwrap_or(0));
                            }
                        }
                        let new_padding =
                            if bef.iter().all(|b| *b == 0) && aft.iter().all(|a| *a == 0) {
                                PaddingSpec::Valid
                            } else {
                                PaddingSpec::ExplicitOnnxPool(bef, aft, false)
                            };

                        // Wire: input → Pad → Conv/MaxPool(new_padding)
                        let data_input =
                            mapping.get(&node.inputs[0]).copied().unwrap_or(node.inputs[0]);
                        let pad_wire = new_model.wire_node(
                            format!("{}.pad", node.name),
                            Pad::new(pads, PadMode::Constant(Arc::new(Tensor::from(0.0f32)))),
                            &[data_input],
                        )?;

                        let mut inputs: TVec<OutletId> = node
                            .inputs
                            .iter()
                            .map(|i| mapping.get(i).copied().unwrap_or(*i))
                            .collect();
                        inputs[0] = pad_wire[0];

                        let new_op: Box<dyn TypedOp> =
                            if let Some(conv) = node.op.downcast_ref::<Conv>() {
                                Box::new(Conv {
                                    pool_spec: PoolSpec {
                                        padding: new_padding,
                                        ..conv.pool_spec.clone()
                                    },
                                    ..conv.clone()
                                })
                            } else {
                                let pool = node.op.downcast_ref::<MaxPool>().unwrap();
                                Box::new(MaxPool {
                                    pool_spec: PoolSpec {
                                        padding: new_padding,
                                        ..pool.pool_spec.clone()
                                    },
                                    ..pool.clone()
                                })
                            };

                        let new_outlets = new_model.wire_node(&node.name, new_op, &inputs)?;
                        for (slot, &outlet) in new_outlets.iter().enumerate() {
                            mapping.insert(OutletId::new(node_id, slot), outlet);
                        }
                        changed = true;
                        continue;
                    }
                }
            }

            // Default: copy the node as-is.
            if node
                .op
                .downcast_ref::<tract_pulse_opl::tract_core::ops::source::TypedSource>()
                .is_some()
            {
                let fact = model.outlet_fact(OutletId::new(node_id, 0))?;
                let new_outlet = new_model.add_source(&node.name, fact.clone())?;
                mapping.insert(OutletId::new(node_id, 0), new_outlet);
            } else {
                let inputs: TVec<OutletId> =
                    node.inputs.iter().map(|i| mapping.get(i).copied().unwrap_or(*i)).collect();
                let new_outlets = new_model.wire_node(&node.name, node.op.clone(), &inputs)?;
                for (slot, &outlet) in new_outlets.iter().enumerate() {
                    mapping.insert(OutletId::new(node_id, slot), outlet);
                }
            }
        }

        let outputs = model.output_outlets()?.to_vec();
        let new_outputs: TVec<OutletId> = outputs.iter().map(|o| mapping[o]).collect();
        new_model.select_output_outlets(&new_outputs)?;

        if changed { Ok(new_model) } else { Ok(model.clone()) }
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        Ok(self.typed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_streaming_input() {
        let scope = SymbolScope::default();
        let s = scope.sym("S");
        let p = scope.sym("P");
        let t = scope.sym("T");

        let batch_fact = DatumType::F32.fact(&[1.to_dim(), s.clone().into(), 16.to_dim()]);

        let mut axes = TVec::new();
        for dim in batch_fact.shape.iter() {
            if dim == &TDim::Sym(s.clone()) {
                axes.push(AxisRegion::Streaming {
                    start: TDim::Sym(t.clone()) * TDim::Sym(p.clone()),
                    end: (TDim::Sym(t.clone()) + 1) * TDim::Sym(p.clone()),
                });
            } else {
                axes.push(AxisRegion::Fixed(dim.clone()));
            }
        }
        let region = PulseV2Region { axes };

        assert_eq!(region.rank(), 3);
        assert!(!region.axes[0].is_streaming());
        assert!(region.axes[1].is_streaming());
        assert!(!region.axes[2].is_streaming());
        assert_eq!(region.axes[1].size().simplify(), TDim::Sym(p));
    }
}
