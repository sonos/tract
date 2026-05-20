#![allow(clippy::collapsible_if)]
use std::sync::RwLock;

use crate::fact::StreamInfo;
use crate::{internal::*, ops::sync_inputs};
use tract_core::model::translator::Translate;
use tract_pulse_opl::tract_core::ops::konst::Const;
use tract_pulse_opl::tract_core::ops::source::TypedSource;

pub type PulsedModel = Graph<PulsedFact, Box<dyn PulsedOp>>;
pub type PulsedNode = Node<PulsedFact, Box<dyn PulsedOp>>;

/// Pre-flight check: reject models with wires whose size is superlinear in the
/// streaming symbol but have no `region_of_interest` annotation.
///
/// A wire is superlinear when the streaming symbol appears in more than one
/// shape dimension (e.g. `[T, T]` or `[T, 2T-1]`).  Such wires cannot be
/// pulsified unless ROI narrows the live region to linear size.
fn check_no_unannotated_superlinear_wires(model: &TypedModel, symbol: &Symbol) -> TractResult<()> {
    for node in &model.nodes {
        for (slot, output) in node.outputs.iter().enumerate() {
            let streaming_dims: usize =
                output.fact.shape.iter().filter(|d| d.symbols().contains(symbol)).count();
            if streaming_dims <= 1
                || output.fact.region_of_interest.is_some()
                || output.fact.uniform_tdim.is_some()
                || output.fact.konst.is_some()
            {
                continue;
            }
            // Avoid false positives: if any input to this node already carries
            // an ROI or uniform_tdim that mentions the streaming symbol, the
            // consumer pulsifier can typically derive what it needs from that
            // annotation without one on this wire (e.g. Iff outputs inherit
            // the cond/scores ROI structurally; Softmax output inherits its
            // input's ROI; MultiBroadcastTo fills inherit the broadcast target
            // ROI).  Only ops whose *inputs* are all unannotated are genuine
            // ROI-propagation gaps.
            let any_input_annotated = node.inputs.iter().any(|inp| {
                model
                    .outlet_fact(*inp)
                    .map(|f| f.region_of_interest.is_some() || f.uniform_tdim.is_some())
                    .unwrap_or(false)
            });
            if any_input_annotated {
                continue;
            }
            log::warn!(
                "Wire {}/{} ({:?}) has shape {:?} which is superlinear in streaming \
                 symbol {} ({} dimensions depend on it) but carries no region_of_interest \
                 annotation, and none of its inputs do either. Pulsification may fail.",
                node.name,
                slot,
                OutletId::new(node.id, slot),
                output.fact.shape,
                symbol,
                streaming_dims,
            );
        }
    }
    Ok(())
}

/// LCM of the stream-axis dims across all stream-bearing inputs.
///
/// Used at elementwise pulse meet-points where parallel paths emit
/// different per-pulse sizes (e.g. ConvTranspose with kernel > stride
/// surfacing as `(K_steady, K_initial)` on the two phases of the pulse
/// cycle). Two semantics get conflated otherwise:
///
/// * Tensor shape compatibility (must match at runtime): non-stream
///   axes use NumPy `Broadcast` -- equal or one is 1, anything else
///   fails.
/// * Pulse-divisibility (a scalar constraint on per-pulse cycle): on
///   the stream axis, two paths with steady-state size `K_a` and `K_b`
///   are compatible at any pulse that is a multiple of
///   `LCM(K_a, K_b)`.
///
/// Returns `None` if any stream-axis dim is symbolic; the caller falls
/// back to the unmodified shape `multi_broadcast` produced.
pub fn stream_axis_lcm(inputs: &[&PulsedFact]) -> Option<TDim> {
    let mut dims = inputs.iter().filter_map(|f| f.stream.as_ref().map(|s| &f.shape[s.axis]));
    let first = dims.next()?.clone();
    dims.try_fold(first, |acc, d| acc.lcm(d))
}

/// Pulse-driven path: the pulse value is concrete, so we mint S, substitute
/// `T → pulse_value · S` ourselves, and call blockify just for the section
/// rewrites.  The audio-side multiplier is user-driven — required when
/// there's subsampling between the streaming source and the section's mask
/// wire (e.g. a stride-2 pool: audio chunk = 2 × post-pool chunk).
fn pulse_driven_blockify(
    model: &mut TypedModel,
    symbol: &Symbol,
    pulse_value: i64,
) -> TractResult<(Symbol, TDim)> {
    let chunk_sym = model.symbols.new_with_prefix("S");
    // `S >= 0` is the precondition for the `Div(Add([k·X, …, c]), k) → X`
    // simplification (commit 11b310622).  Without it, post-substitute shapes
    // like `1 + (3 + 56·S)/4` stay unreduced and blockify's chunked Reshape
    // volume check fails comparing them to `14·S`.
    model.symbols.add_assertion(format!("{chunk_sym} >= 0"))?;
    let subs: HashMap<Symbol, TDim> =
        HashMap::from([(symbol.clone(), chunk_sym.to_dim() * pulse_value)]);
    *model = model.substitute_symbols(&subs)?;
    crate::blockify::rewrite_sections(model, &chunk_sym, pulse_value)?;
    model.properties.insert(
        crate::blockify::BLOCKIFY_ORIGINAL_SYMBOL.to_string(),
        tensor1(&[format!("{symbol}")]).into_arc_tensor(),
    );
    // Streaming dim is now `pulse_value · S`, so one pulse covers exactly one S.
    Ok((chunk_sym, 1.to_dim()))
}

#[allow(clippy::new_ret_no_self)]
pub trait PulsedModelExt {
    fn new(source: &TypedModel, symbol: Symbol, pulse: &TDim) -> TractResult<PulsedModel>;

    fn new_with_mapping(
        source: &TypedModel,
        symbol: Symbol,
        pulse: &TDim,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)>;

    fn into_typed(self) -> TractResult<TypedModel>;
}

impl PulsedModelExt for PulsedModel {
    fn new(source: &TypedModel, symbol: Symbol, pulse: &TDim) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, symbol, pulse)?.0)
    }

    fn new_with_mapping(
        source: &TypedModel,
        symbol: Symbol,
        pulse: &TDim,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        check_no_unannotated_superlinear_wires(source, &symbol)?;
        use tract_core::optim::TypedPass;
        let mut blockified = source.clone();
        // Mirror PulseTransform's pre-fold so callers entering through
        // PulsedModel::new (or `--pulse`) get the same treatment.
        crate::ops::diag_gather::detect_diag_gather(&mut blockified)?;
        tract_core::optim::propagate_roi::PropagateRoi.run_direct(&mut blockified)?;
        blockified.declutter()?;
        let (stream_sym, pulse_dim) = match pulse.as_i64() {
            Some(pv) if crate::blockify::has_quadratic_sections(&blockified, &symbol)? => {
                pulse_driven_blockify(&mut blockified, &symbol, pv)?
            }
            _ => (symbol, pulse.clone()),
        };
        let pulsifiers = crate::ops::OpPulsifier::inventory();
        let (mut pulsed, mapping) = Pulsifier(stream_sym, pulse_dim, pulsifiers)
            .translate_model_with_mappings(&blockified)?;
        for key in [
            crate::blockify::BLOCKIFY_CHUNK_SYMBOL,
            crate::blockify::BLOCKIFY_CHUNK_SIZE,
            crate::blockify::BLOCKIFY_ORIGINAL_SYMBOL,
        ] {
            if let Some(v) = blockified.properties.get(key) {
                pulsed.properties.insert(key.to_string(), v.clone());
            }
        }
        Ok((pulsed, mapping))
    }

    fn into_typed(self) -> TractResult<TypedModel> {
        let mut typed = tract_core::model::translator::IntoTranslator.translate_model(&self)?;
        ensure!(
            self.input_outlets()?.iter().any(|o| self.outlet_fact(*o).unwrap().stream.is_some())
        );
        ensure!(
            self.output_outlets()?.iter().any(|o| self.outlet_fact(*o).unwrap().stream.is_some())
        );
        let delays = tensor1(
            &self
                .output_outlets()?
                .iter()
                .map(|oo| {
                    Ok(self.outlet_fact(*oo)?.stream.as_ref().map(|s| s.delay as i64).unwrap_or(0))
                })
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.delay".to_string(), delays.into_arc_tensor());
        let input_axes = tensor1(
            &self
                .input_outlets()?
                .iter()
                .map(|oo| {
                    Ok(self.outlet_fact(*oo)?.stream.as_ref().map(|s| s.axis as i64).unwrap_or(-1))
                })
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.input_axes".to_string(), input_axes.into_arc_tensor());
        let output_axes = tensor1(
            &self
                .output_outlets()?
                .iter()
                .map(|oo| {
                    Ok(self.outlet_fact(*oo)?.stream.as_ref().map(|s| s.axis as i64).unwrap_or(-1))
                })
                .collect::<TractResult<TVec<i64>>>()?,
        );
        typed.properties.insert("pulse.output_axes".to_string(), output_axes.into_arc_tensor());
        // Stash the streaming symbol's name so downstream consumers (CLI run
        // path, etc.) can resolve `op.end_input.eval(...)` and other symbolic
        // dims at runtime.  The symbol lives in TDims like a PulsePad's
        // `end_input = stream.dim + …`; without binding it, those evals hit
        // `usize::MAX` fallbacks and end-of-stream padding silently misfires.
        let stream_syms: Vec<String> = self
            .input_outlets()?
            .iter()
            .filter_map(|oo| self.outlet_fact(*oo).unwrap().stream.as_ref())
            .flat_map(|s| s.dim.symbols())
            .map(|s| s.to_string())
            .collect();
        if let Some(name) = stream_syms.into_iter().next() {
            typed
                .properties
                .insert("pulse.streaming_symbol".to_string(), tensor1(&[name]).into_arc_tensor());
        }
        Ok(typed)
    }
}

impl SpecialOps<PulsedFact, Box<dyn PulsedOp>> for PulsedModel {
    fn is_source(op: &Box<dyn PulsedOp>) -> bool {
        op.as_op().downcast_ref::<crate::ops::source::PulsedSource>().is_some()
    }

    fn create_source(&self, fact: PulsedFact) -> Box<dyn PulsedOp> {
        Box::new(crate::ops::source::PulsedSource(fact))
    }

    fn create_dummy(&self) -> Box<dyn PulsedOp> {
        Box::new(tract_core::ops::dummy::Dummy::new())
    }

    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn PulsedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts = {
            let input_facts =
                inputs.iter().map(|o| self.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            op.pulsed_output_facts(&input_facts)?
        };
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }

    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        for node in &self.nodes {
            if let Some(op) = node.op_as::<Const>() {
                if op.val() == &v {
                    return Ok(node.id.into());
                }
            }
        }
        let op = NonPulsingWrappingOp(Box::new(Const::new(v)?));
        Ok(self.wire_node(name, op, &[])?[0])
    }
}

struct Pulsifier(
    Symbol,
    TDim,
    #[allow(dead_code)] Arc<RwLock<HashMap<TypeId, crate::ops::OpPulsifier>>>,
);

impl std::fmt::Debug for Pulsifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pulsifier({})", self.0)
    }
}

impl
    tract_core::model::translator::Translate<
        TypedFact,
        Box<dyn TypedOp>,
        PulsedFact,
        Box<dyn PulsedOp>,
    > for Pulsifier
{
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(op) = node.op_as::<TypedSource>() {
            return Ok(crate::ops::source::pulsify(
                op, source, node, target, mapping, &self.0, &self.1,
            )?
            .unwrap());
        }
        log::debug!("Pulsifying node {node}");

        if !source
            .node_input_facts(node.id)?
            .iter()
            .any(|f| f.shape.iter().any(|d| d.symbols().contains(&self.0)))
            && !node
                .outputs
                .iter()
                .any(|o| o.fact.shape.iter().any(|d| d.symbols().contains(&self.0)))
        {
            let pulse_op = NonPulsingWrappingOp(node.op.clone());
            let inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
            log::debug!("Pulsified node {node} with NonPulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        if let Some(pulsified) =
            OpPulsifier::pulsify(source, node, target, mapping, &self.0, &self.1)?
        {
            log::debug!("Pulsified node {node} with adhoc pulsifier");
            return Ok(pulsified);
        }

        let pulse_facts: TVec<PulsedFact> =
            node.inputs.iter().map(|i| target.outlet_fact(mapping[i]).unwrap().clone()).collect();
        if pulse_facts.iter().all(|pf| pf.stream.is_none()) {
            let pulse_op = NonPulsingWrappingOp(node.op.clone());
            let inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
            log::debug!("Pulsified node {node} with NonPulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        let (stream_input_ix, pulse_fact) =
            pulse_facts.iter().enumerate().find(|(_ix, pf)| pf.stream.is_some()).unwrap();
        let (input_facts, output_facts) = source.node_facts(node.id)?;
        let axes_mapping = node.op.axes_mapping(&input_facts, &output_facts)?;
        let axis_info = axes_mapping
            .axis((InOut::In(stream_input_ix), pulse_fact.stream.as_ref().unwrap().axis))?;
        if axis_info.outputs[0].len() == 1 {
            let pulse_op = PulseWrappingOp(node.op.clone());
            let inputs = sync_inputs(node, target, mapping)?;
            log::debug!("Pulsified node {node} with PulsingWrappingOp");
            return target.wire_node(&node.name, pulse_op, &inputs);
        }

        bail!("No specific pulse transformation for {}, and could not track pulsing axis.", node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PulseWrappingOp(pub Box<dyn TypedOp>);

impl Op for PulseWrappingOp {
    fn name(&self) -> StaticName {
        format!("PulseWrapping({}", self.0.name()).into()
    }

    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self.0.as_ref())
    }
}

impl EvalOp for PulseWrappingOp {
    fn is_stateless(&self) -> bool {
        self.0.is_stateless()
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.0.eval(inputs)
    }

    fn state(&self, session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        self.0.state(session, node_id)
    }
}

impl PulsedOp for PulseWrappingOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let (pulsing_input, stream) = if let Some((ix, fact)) =
            &inputs.iter().enumerate().find(|(_ix, f)| f.stream.is_some())
        {
            (*ix, fact.stream.as_ref().unwrap())
        } else {
            bail!("PulseWrappingOp used on non streaming input")
        };
        let input_facts =
            inputs.iter().map(|pf| pf.to_typed_fact()).collect::<TractResult<TVec<_>>>()?;
        let input_facts_ref = input_facts.iter().map(|f| f.as_ref()).collect::<TVec<_>>();
        let output_facts = self.0.output_facts(&input_facts_ref)?;
        let output_facts_ref = output_facts.iter().collect::<TVec<_>>();
        let axes_mapping = self.0.axes_mapping(&input_facts_ref, &output_facts_ref)?;
        let axis_info = axes_mapping.axis((InOut::In(pulsing_input), stream.axis))?;
        std::mem::drop(output_facts_ref);
        // When parallel pulse paths converge at an elementwise op, the
        // typed `output_facts` falls through to `multi_broadcast` for shape
        // merging, which produces `Broadcast([K_a, K_b])` on the stream
        // axis when the inputs have different per-pulse sizes (e.g.
        // steady-state `stride` vs initial-state `kernel` of a
        // streaming convtr surfacing on two phases of the cycle).
        // `Broadcast` is the right semantic for non-stream axes (shape
        // compatibility at runtime) but a category error for the stream
        // axis: there the merged constraint is *scalar pulse-divisibility*
        // and the right answer is LCM. Compute it post-hoc and override
        // the offending dim before it propagates downstream.
        let merged_stream_dim = stream_axis_lcm(inputs);
        output_facts
            .into_iter()
            .enumerate()
            .map(|(ix, tf)| {
                if let &[axis] = &*axis_info.outputs[ix] {
                    let mut shape = tf.shape;
                    if let Some(merged) = merged_stream_dim.as_ref() {
                        if matches!(shape[axis], TDim::Broadcast(_)) {
                            shape.set(axis, merged.clone());
                        }
                    }
                    Ok(PulsedFact {
                        shape,
                        datum_type: tf.datum_type,
                        stream: Some(StreamInfo {
                            delay: stream.delay,
                            axis,
                            dim: stream.dim.clone(),
                        }),
                    })
                } else {
                    bail!("Disappearing pulsing axis")
                }
            })
            .collect()
    }

    as_op!();

    fn to_typed(&self) -> Box<dyn TypedOp> {
        self.0.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NonPulsingWrappingOp(pub Box<dyn TypedOp>);

impl Op for NonPulsingWrappingOp {
    fn name(&self) -> StaticName {
        format!("NonePulsingWrapping({}", self.0.name()).into()
    }

    fn as_typed(&self) -> Option<&dyn TypedOp> {
        Some(self.0.as_ref())
    }
}

impl EvalOp for NonPulsingWrappingOp {
    fn is_stateless(&self) -> bool {
        self.0.is_stateless()
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.0.eval(inputs)
    }

    fn state(&self, session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
        self.0.state(session, node_id)
    }
}

impl PulsedOp for NonPulsingWrappingOp {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let input_facts =
            inputs.iter().map(|pf| pf.to_typed_fact()).collect::<TractResult<TVec<_>>>()?;
        let input_facts_ref = input_facts.iter().map(|f| f.as_ref()).collect::<TVec<_>>();
        let output_facts = self.0.output_facts(&input_facts_ref)?;
        let output_facts_ref = output_facts.iter().collect::<TVec<_>>();
        std::mem::drop(output_facts_ref);
        output_facts
            .into_iter()
            .map(|tf| Ok(PulsedFact { shape: tf.shape, datum_type: tf.datum_type, stream: None }))
            .collect()
    }

    as_op!();

    fn to_typed(&self) -> Box<dyn TypedOp> {
        self.0.clone()
    }
}
