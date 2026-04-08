use crate::internal::*;
use tract_core::ops::array::Slice;
use tract_core::ops::logic::classify_chunk_window;

register_all!(Slice: pulsify);

fn pulsify(
    op: &Slice,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();

    // Non-streaming input: the streaming symbol appears only in start/end
    // (e.g. a static PE table sliced to the current frame count, or to a
    // symmetric RPE window centered at MAX with start=MAX-T', end=MAX+T'+1).
    // Substitute S→pulse directly to get the concrete per-pulse bounds.
    // ROI-aware: if this axis is the col_axis of a chunk-window ROI, extend
    // the range by L*P.  The direction depends on whether start grows or shrinks
    // with S: if start decreases (start_sub < start@S=0, e.g. center-S), extend
    // backward (subtract L*P from start); otherwise extend forward (add to end).
    if fact.stream.is_none() {
        let start = op.start.substitute(symbol, pulse)?;
        let end = op.end.substitute(symbol, pulse)?;
        if start.symbols().is_empty() && end.symbols().is_empty() {
            let out_fact = source.outlet_fact(OutletId::new(node.id, 0))?;
            let (start, end) = if let Some(cw) = out_fact
                .region_of_interest
                .as_ref()
                .and_then(|r| classify_chunk_window(&r.clone().simplify()))
            {
                if cw.col_axis == op.axis && cw.left_chunks > 0 {
                    let lp = cw.left_chunks as i64 * pulse.to_i64()?;
                    let start_i64 = start.to_i64()?;
                    let start_base = op.start.substitute(symbol, &TDim::Val(0))?;
                    let (adj_s, adj_e) =
                        if start_base.symbols().is_empty() && start_i64 < start_base.to_i64()? {
                            // start decreases with S (e.g. center-S): extend backward
                            (TDim::Val(start_i64 - lp), end.clone())
                        } else {
                            // start is fixed: extend end forward
                            (start.clone(), TDim::Val(end.to_i64()? + lp))
                        };
                    // Only apply if the adjusted bounds fit the input.
                    let input_len = source.outlet_fact(node.inputs[0])?.shape[op.axis]
                        .substitute(symbol, pulse)?;
                    let fits = adj_s
                        .to_i64()
                        .ok()
                        .zip(adj_e.to_i64().ok())
                        .zip(input_len.to_i64().ok())
                        .map(|((s, e), l)| s >= 0 && e <= l)
                        .unwrap_or(false);
                    if fits { (adj_s, adj_e) } else { (start, end) }
                } else {
                    (start, end)
                }
            } else {
                (start, end)
            };
            use crate::model::NonPulsingWrappingOp;
            let concrete_op = NonPulsingWrappingOp(Box::new(Slice { axis: op.axis, start, end }));
            return target.wire_node(&*node.name, concrete_op, &[input]).map(Some);
        }
        return Ok(None);
    }

    let stream = fact.stream.as_ref().unwrap();
    if op.axis == stream.axis {
        let start = op.start.substitute(symbol, pulse)?;
        let skip = start.to_usize()?;
        let take = node.outputs[0].fact.shape[op.axis].clone();
        let op = PulsedAxisSlice { axis: op.axis, skip, take };
        Ok(Some(target.wire_node(&*node.name, op, &[input])?))
    } else {
        // Slice on a non-streaming axis whose bounds may contain the streaming
        // symbol (e.g. axis 2 sliced to 2*T'-1 after the skew reshape).
        //
        // ROI-aware path: if the source outlet carries a chunk-window ROI whose
        // col_axis matches this slice axis, use W=(L+1)*P instead of substituting S→P.
        // This implements the windowed RPE positional-score truncation.
        let out_fact = source.outlet_fact(OutletId::new(node.id, 0))?;
        if let Some(cw) = out_fact
            .region_of_interest
            .as_ref()
            .and_then(|r: &TDim| classify_chunk_window(&r.clone().simplify()))
        {
            if cw.col_axis == op.axis {
                let pulse_i64 = pulse.to_i64()?;
                let lp = cw.left_chunks as i64 * pulse_i64;
                let start_sub = op.start.substitute(symbol, pulse)?;
                let end_sub = op.end.substitute(symbol, pulse)?;
                if start_sub.symbols().is_empty() && end_sub.symbols().is_empty() {
                    let start_base = op.start.substitute(symbol, &TDim::Val(0))?;
                    let (adj_start, adj_end) = if start_base.symbols().is_empty()
                        && start_sub.to_i64()? < start_base.to_i64()?
                    {
                        (TDim::Val(start_sub.to_i64()? - lp), end_sub)
                    } else {
                        (start_sub, TDim::Val(end_sub.to_i64()? + lp))
                    };
                    // Only apply ROI extension if the adjusted bounds fit within
                    // the input's actual size on this axis.  If the upstream hasn't
                    // been expanded to support the wider range, fall through to the
                    // non-ROI path.
                    let input_len = fact.shape[op.axis].clone();
                    let fits = adj_start
                        .to_i64()
                        .ok()
                        .zip(adj_end.to_i64().ok())
                        .zip(input_len.to_i64().ok())
                        .map(|((s, e), l)| s >= 0 && e <= l)
                        .unwrap_or(false);
                    if fits {
                        use crate::model::PulseWrappingOp;
                        return Ok(Some(target.wire_node(
                            &*node.name,
                            PulseWrappingOp(Box::new(Slice {
                                axis: op.axis,
                                start: adj_start,
                                end: adj_end,
                            })),
                            &[input],
                        )?));
                    }
                }
            }
        }

        // Try full substitution first (S→pulse): correct for bounds like
        // MAX-T' / MAX+T'+1 (RPE symmetric window) where the constant base
        // term must be preserved.
        //
        // If full substitution leaves symbols (e.g. TDim::Broadcast artifacts
        // from shape_of chains), fall back to the boundary-correction delta
        // formula (sub(S,pulse) - sub(S,0)) which cancels those artifacts.
        let start_full = op.start.substitute(symbol, pulse)?;
        let end_full = op.end.substitute(symbol, pulse)?;
        if start_full.symbols().is_empty() && end_full.symbols().is_empty() {
            use crate::model::PulseWrappingOp;
            return Ok(Some(target.wire_node(
                &*node.name,
                PulseWrappingOp(Box::new(Slice {
                    axis: op.axis,
                    start: start_full,
                    end: end_full,
                })),
                &[input],
            )?));
        }
        // Full substitution left symbols; try delta formula to cancel artifacts.
        let start = start_full - op.start.substitute(symbol, &TDim::Val(0))?;
        let end = end_full - op.end.substitute(symbol, &TDim::Val(0))?;
        if start.symbols().is_empty() && end.symbols().is_empty() {
            use crate::model::PulseWrappingOp;
            return Ok(Some(target.wire_node(
                &*node.name,
                PulseWrappingOp(Box::new(Slice { axis: op.axis, start, end })),
                &[input],
            )?));
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}

impl Op for PulsedAxisSlice {
    fn name(&self) -> StaticName {
        "PulsedAxisSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{}, skip:{} take:{}", self.axis, self.skip, self.take)])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedAxisSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl PulsedOp for PulsedAxisSlice {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let stream = fact.stream.as_mut().unwrap();
        stream.delay += self.skip;
        stream.dim = self.take.clone();
        Ok(tvec!(fact))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::<tract_pulse_opl::tract_core::ops::identity::Identity>::default()
    }

    as_op!();
}
