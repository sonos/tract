use crate::internal::*;
use crate::v2::{PulseV2Action, PulseV2Region, PulseV2Symbols, RegionTransform};
use tract_pulse_opl::tract_core::ops::array::{Pad, PadMode};

fn pad_transform(
    op: &dyn TypedOp,
    _source_region: &PulseV2Region,
    symbols: &PulseV2Symbols,
) -> TractResult<Option<PulseV2Action>> {
    let pad = op.downcast_ref::<Pad>().unwrap();
    let PadMode::Constant(value) = &pad.mode else {
        return Ok(None);
    };
    Ok(Some(PulseV2Action::ReplaceOp(Box::new(PulseV2Pad {
        pads: pad.pads.clone(),
        value: value.clone(),
        pulse_id: symbols.pulse_id.clone(),
        stream_sym: symbols.stream.clone(),
    }))))
}

inventory::submit! {
    RegionTransform {
        type_id: std::any::TypeId::of::<Pad>(),
        func: pad_transform,
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2Pad {
    pub pads: Vec<(usize, usize)>,
    pub value: Arc<Tensor>,
    pub pulse_id: Symbol,
    pub stream_sym: Symbol,
}

impl Op for PulseV2Pad {
    fn name(&self) -> StaticName {
        "PulseV2Pad".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.pads)])
    }
    op_as_typed_op!();
}

impl EvalOp for PulseV2Pad {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let t = session.resolved_symbols.get(&self.pulse_id).unwrap_or(0) as usize;
        let s = session.resolved_symbols.get(&self.stream_sym).unwrap_or(0) as usize;

        let mut result = input.into_tensor();
        for (axis, &(before, after)) in self.pads.iter().enumerate() {
            if before == 0 && after == 0 {
                continue;
            }
            let p = result.shape()[axis];
            let real = p.min(s.saturating_sub(t * p));
            let mut parts: Vec<Tensor> = Vec::new();

            // Before-pad: all at T=0.
            if t == 0 && before > 0 {
                let mut shape = result.shape().to_vec();
                shape[axis] = before;
                parts.push(Tensor::broadcast_scalar_to_shape(&self.value, &shape)?);
            }

            // Real data.
            if real > 0 {
                parts.push(result.slice(axis, 0, real)?.into_tensor());
            }

            // After-pad: emitted for positions past S.
            // This pulse covers input-space positions [t*p, (t+1)*p).
            // After-pad occupies positions [s, s+after).
            // Intersection of [(t+1)*p capped to relevant range] with [s, s+after).
            let pulse_end = t * p + p; // (t+1)*p in input space
            if after > 0 && pulse_end > s {
                let after_start = s.max(t * p);
                let after_end = (s + after).min(pulse_end);
                let ap_count = after_end.saturating_sub(after_start);
                if ap_count > 0 {
                    let mut shape = result.shape().to_vec();
                    shape[axis] = ap_count;
                    parts.push(Tensor::broadcast_scalar_to_shape(&self.value, &shape)?);
                }
            }

            result = if parts.is_empty() {
                let mut shape = result.shape().to_vec();
                shape[axis] = 0;
                Tensor::zero_dt(result.datum_type(), &shape)?
            } else {
                Tensor::stack_tensors(axis, &parts)?
            };
        }
        Ok(tvec!(result.into_tvalue()))
    }
}

impl TypedOp for PulseV2Pad {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        let t = TDim::Sym(self.pulse_id.clone());
        let s = TDim::Sym(self.stream_sym.clone());
        for (axis, &(before, after)) in self.pads.iter().enumerate() {
            if before == 0 && after == 0 {
                continue;
            }
            let p = fact.shape[axis].clone();
            // real data = min(P, max(0, S - T*P))
            let real_data = TDim::Min(vec![
                p.clone(),
                TDim::Max(vec![TDim::Val(0), s.clone() - t.clone() * p.clone()]),
            ]);
            // before: all emitted at T=0 → max(0, before * (1 - min(1, T)))
            // simpler: before * Eq(T, 0)... but Eq gives 0/1 and we'd need before * Eq(T,0).
            // Use: min(before, max(0, before - T*P)) which gives `before` at T=0, 0 at T≥1
            // (assuming P ≥ 1 and before < 2*P... actually before - T*P at T=1 is before - P,
            // which could be > 0 if before > P. But we emit ALL before at T=0.)
            //
            // Correct: before if T==0, else 0. Express as: before * Ge(1, T+1)... no.
            // Use: max(0, before - T * max(before, 1)) — at T=0: before, at T≥1: ≤0.
            // Simplest correct: min(before, max(0, 1-T) * before)
            // Actually just: before * (1 - min(1, T)) ... still messy.
            // Pragmatic: before * Eq(T, 0). TDim has Eq!
            let bp =
                TDim::Val(before as i64) * TDim::Eq(Box::new(t.clone()), Box::new(TDim::Val(0)));
            // after: emitted when stream ends. Same approach but harder.
            // Use: after * Eq(real_data + T*P, S) ... no, could span multiple pulses.
            // For now use the incremental formula:
            // How far past the stream end at the start / end of this pulse.
            let cum_before = TDim::Max(vec![TDim::Val(0), t.clone() * p.clone() - s.clone()]);
            let cum_after = TDim::Max(vec![TDim::Val(0), (t.clone() + 1) * p.clone() - s.clone()]);
            let ap = TDim::Min(vec![TDim::Val(after as i64), cum_after])
                - TDim::Min(vec![TDim::Val(after as i64), cum_before]);
            let out_dim = bp + real_data + ap;
            fact.shape.set(axis, out_dim);
        }
        Ok(tvec!(fact))
    }
}
