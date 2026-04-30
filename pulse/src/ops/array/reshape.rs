use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::change_axes::AxisOp;

register_all!(AxisOp: pulsify);

fn pulsify(
    op: &AxisOp,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_let!(AxisOp::Reshape(at, from, to) = op);
    let input = mapping[&node.inputs[0]];
    let fact = target.outlet_fact(input)?.clone();
    rule_if_some!(stream = &fact.stream);
    rule_if!(stream.axis >= *at && stream.axis < *at + from.len());
    let from_pos = stream.axis - *at;
    rule_if!(from[from_pos].symbols().contains(symbol));
    rule_if!(from.iter().enumerate().all(|(i, d)| i == from_pos || !d.symbols().contains(symbol)));
    let to_streaming: TVec<usize> = to
        .iter()
        .enumerate()
        .filter(|(_, d)| d.symbols().contains(symbol))
        .map(|(i, _)| i)
        .collect();
    rule_if!(to_streaming.len() == 1);
    let to_pos = to_streaming[0];

    let from_pulsed: TVec<TDim> =
        from.iter().map(|d| d.substitute(symbol, pulse)).collect::<TractResult<_>>()?;
    let to_pulsed: TVec<TDim> =
        to.iter().map(|d| d.substitute(symbol, pulse)).collect::<TractResult<_>>()?;

    let pulsed = PulsedReshape {
        op: AxisOp::Reshape(*at, from_pulsed, to_pulsed),
        new_stream_axis: *at + to_pos,
        new_stream_dim: to[to_pos].clone(),
    };
    target.wire_node(&*node.name, pulsed, &[input]).map(Some)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PulsedReshape {
    pub op: AxisOp,
    pub new_stream_axis: usize,
    pub new_stream_dim: TDim,
}

impl Op for PulsedReshape {
    fn name(&self) -> StaticName {
        "PulsedReshape".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "op:{:?} stream_axis:{} stream_dim:{}",
            self.op, self.new_stream_axis, self.new_stream_dim
        )])
    }

    not_a_typed_op!();
}

impl EvalOp for PulsedReshape {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.op.eval(inputs)
    }
}

impl PulsedOp for PulsedReshape {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let input_typed: TypedFact = inputs[0].into();
        let outs = self.op.output_facts(&[&input_typed])?;
        let stream = inputs[0].stream.as_ref().unwrap();
        let out_fact = outs.into_iter().next().context("Reshape produced no output fact")?;
        // `stream.delay` counts elements on the streaming axis.  When the
        // reshape changes the per-pulse size on that axis (e.g. merging
        // `(S, k) → S·k` at a Blockify section boundary), the delay must
        // rescale by `new_per_pulse / old_per_pulse` so the same physical
        // lag is preserved in the new units.
        let AxisOp::Reshape(at, from, to) = &self.op else {
            unreachable!("PulsedReshape only built from AxisOp::Reshape (see pulsify above)");
        };
        let from_pos = stream.axis - at;
        let to_pos = self.new_stream_axis - at;
        let old_per_pulse = from[from_pos].to_usize()?;
        let new_per_pulse = to[to_pos].to_usize()?;
        let scaled = stream.delay * new_per_pulse;
        ensure!(
            scaled % old_per_pulse == 0,
            "PulsedReshape: stream.delay {} can't be rescaled from per-pulse {} \
             to per-pulse {} (would lose precision)",
            stream.delay,
            old_per_pulse,
            new_per_pulse,
        );
        let new_delay = scaled / old_per_pulse;
        Ok(tvec!(PulsedFact {
            datum_type: out_fact.datum_type,
            shape: out_fact.shape,
            stream: Some(StreamInfo {
                axis: self.new_stream_axis,
                dim: self.new_stream_dim.clone(),
                delay: new_delay,
            }),
        }))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(self.op.clone())
    }

    as_op!();
}
