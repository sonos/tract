use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::change_axes::AxisOp;

register_all!(AxisOp: pulsify_axis_op);

fn pulsify_axis_op(
    op: &AxisOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    let stream = match &fact.stream {
        Some(s) => s.clone(),
        None => return Ok(None),
    };

    let (at, from, to) = match op {
        AxisOp::Reshape(at, from, to) => (at, from, to),
        _ => return Ok(None),
    };

    if stream.axis != *at {
        return Ok(None);
    }

    // Case 1: token-fold [T, ...] → [C, P, ...] where P = pulse.
    // At pulse time: [P, ...] → [1, P, ...] (unsqueeze at `at`).
    if from.len() == 1 && to.len() >= 2 && to.last() == Some(pulse) {
        let p = pulse.to_i64()? as u64;
        let chunk_dim = stream.dim.clone() / p;
        let pulsed_op = PulsedTokenFold { at: *at, chunk_dim };
        return Ok(Some(target.wire_node(&node.name, pulsed_op, &[input])?));
    }

    // Case 2: token-unfold [C, P, ...] → [T, ...] where P = pulse.
    // At pulse time: [1, P, ...] → [P, ...] (squeeze at `at`).
    if from.len() >= 2 && from.last() == Some(pulse) && to.len() == 1 {
        let token_dim = stream.dim.clone() * pulse.clone();
        let pulsed_op = PulsedTokenUnfold { at: *at, token_dim, pulse_size: pulse.clone() };
        return Ok(Some(target.wire_node(&node.name, pulsed_op, &[input])?));
    }

    Ok(None)
}

// ─── Token-fold: [T, ...] → [C, P, ...] ───────────────────────────────────

/// Pulsed form of token-fold reshape.  At pulse time the tensor is `[P, ...]`
/// and the op unsqueezes axis `at` to produce `[1, P, ...]`.
/// Stream info: axis=`at`, dim=chunk_count=T/P, pulse=1.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PulsedTokenFold {
    pub at: usize,
    pub chunk_dim: TDim, // total chunk count = T / P
}

impl Op for PulsedTokenFold {
    fn name(&self) -> StaticName {
        "PulsedTokenFold".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("at:{} chunk_dim:{}", self.at, self.chunk_dim)])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedTokenFold {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // Unsqueeze axis `at`: [P, ...] → [1, P, ...]
        AxisOp::Add(self.at).eval(inputs)
    }
}

impl PulsedOp for PulsedTokenFold {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let stream = inputs[0].stream.as_ref().unwrap();
        let mut out_shape = inputs[0].shape.to_tvec();
        out_shape.insert(self.at, TDim::Val(1));
        Ok(tvec![PulsedFact {
            datum_type: inputs[0].datum_type,
            shape: out_shape.into(),
            stream: Some(StreamInfo {
                axis: self.at,
                dim: self.chunk_dim.clone(),
                delay: stream.delay,
            }),
        }])
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(AxisOp::Add(self.at))
    }

    as_op!();
}

// ─── Token-unfold: [C, P, ...] → [T, ...] ─────────────────────────────────

/// Pulsed form of token-unfold reshape.  At pulse time the tensor is `[1, P, ...]`
/// and the op squeezes axis `at` to produce `[P, ...]`.
/// Stream info: axis=`at`, dim=token_count=C*P, pulse=P.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PulsedTokenUnfold {
    pub at: usize,
    pub token_dim: TDim,  // total token count = C * P
    pub pulse_size: TDim, // P (= pulse size)
}

impl Op for PulsedTokenUnfold {
    fn name(&self) -> StaticName {
        "PulsedTokenUnfold".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("at:{} token_dim:{} pulse:{}", self.at, self.token_dim, self.pulse_size)])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedTokenUnfold {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // Squeeze axis `at`: [1, P, ...] → [P, ...]
        AxisOp::Rm(self.at).eval(inputs)
    }
}

impl PulsedOp for PulsedTokenUnfold {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let stream = inputs[0].stream.as_ref().unwrap();
        let mut out_shape = inputs[0].shape.to_tvec();
        out_shape.remove(self.at);
        out_shape[self.at] = self.pulse_size.clone();
        Ok(tvec![PulsedFact {
            datum_type: inputs[0].datum_type,
            shape: out_shape.into(),
            stream: Some(StreamInfo {
                axis: self.at,
                dim: self.token_dim.clone(),
                delay: stream.delay,
            }),
        }])
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(AxisOp::Rm(self.at))
    }

    as_op!();
}
