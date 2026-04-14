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
    InputRegions(TVec<Option<PulseV2Region>>),
    /// Skip this op — just forward the mapped inputs as outputs.
    Skip,
    /// Replace this op with a different one (same inputs).
    ReplaceOp(Box<dyn TypedOp>),
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
    /// Pulsify a batch model.
    ///
    /// For each Source, substitute S → P. For each op, ask its RegionTransform
    /// what input regions it needs. Where an input needs lookback beyond the
    /// source increment, insert a PulseV2Buffer. Wire the op with (potentially
    /// buffered) inputs. Output shapes are computed by the op's output_facts.
    pub fn new(batch_model: &TypedModel, stream_sym: Symbol) -> TractResult<Self> {
        use crate::v2_buffer::PulseV2Buffer;

        let p_sym = batch_model.symbols.sym("P");
        let t_sym = batch_model.symbols.sym("T");

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
                let new_outlets = typed.wire_node(&node.name, replacement, &inputs)?;
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

            let input_regions: TVec<Option<PulseV2Region>> = match action {
                Some(PulseV2Action::InputRegions(r)) => r,
                _ => {
                    // Default: every input gets the source region.
                    if let Some(src) = &source_region {
                        node.inputs.iter().map(|_| Some(src.clone())).collect()
                    } else {
                        tvec![None; node.inputs.len()]
                    }
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
                        let buffer = PulseV2Buffer { lookback, depth, pulse_id: t_sym.clone() };
                        let buffered = typed.wire_node(
                            format!("{}.buffer.{}", node.name, ix),
                            buffer,
                            &[pulsed_input],
                        )?;
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
                // Propagate region: the output wire's region is the source region.
                // The shape is computed by output_facts; the region just tracks
                // "this wire is streaming, with these coordinates."
                if let Some(region) = &source_region {
                    wire_regions.insert(new_outlet, region.clone());
                }
            }
        }

        let batch_outputs = batch_model.output_outlets()?.to_vec();
        let pulsed_outputs: TVec<OutletId> = batch_outputs.iter().map(|o| mapping[o]).collect();
        typed.select_output_outlets(&pulsed_outputs)?;

        Ok(PulseV2Model { typed, symbols })
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
