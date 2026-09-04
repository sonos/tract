use std::ops::Range;

use tract_core::ops::array::PadMode;
use tract_nnef::internal::*;
use tract_nnef::ser::tdim;

use crate::lane::lane_runs;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_pulse_pad",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("before"),
            TypeName::Integer.named("after"),
            TypeName::Integer.named("begin_input"),
            TypeName::Integer.named("end_input"),
            TypeName::String.named("border"),
            TypeName::Scalar.named("value"),
            TypeName::Integer.named("overlap"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser,
    );
    registry.register_dumper(ser)
}

fn ser(ast: &mut IntoAst, node: &TypedNode, op: &PulsePad) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let dt = ast.model.outlet_fact(node.inputs[0])?.datum_type;
    let (border, value) = tract_nnef::ops::nnef::ser::pad_mode(&op.mode, dt)?;
    let mut params = vec![
        ("axis", numeric(op.axis)),
        ("before", numeric(op.before)),
        ("begin_input", numeric(op.begin_input)),
        ("overlap", numeric(op.overlap)),
        ("after", tdim(&op.after)),
        ("end_input", tdim(&op.end_input)),
    ];
    params.push(("border", string(border)));
    if let Some(value) = value {
        params.push(("value", value));
    }
    Ok(Some(invocation("tract_pulse_pulse_pad", &[wire], &params)))
}

fn deser(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let before = invocation.named_arg_as(builder, "before")?;
    let begin_input = invocation.named_arg_as(builder, "begin_input")?;
    let overlap = invocation.named_arg_as(builder, "overlap")?;
    let border = invocation.named_arg_as::<String>(builder, "border")?;
    let value: Tensor = tensor0(invocation.named_arg_as::<f32>(builder, "value")?);
    let (after, end_input) = builder.allowing_new_symbols(|builder| {
        TractResult::Ok((
            invocation.named_arg_as(builder, "after")?,
            invocation.named_arg_as(builder, "end_input")?,
        ))
    })?;

    let mode = tract_nnef::ops::nnef::deser::pad_mode(&border, value)?;
    let op = PulsePad { axis, before, after, begin_input, end_input, mode, overlap };
    builder.wire(op, &[wire])
}

/// Repeat one seat's frame at `frame` over `range` along `axis` of that same
/// seat of `data`. The frame and the range it fills sit in the same tensor, so
/// no assign can name both.
fn fill_from_own_frame(
    data: &mut Tensor,
    seat: Option<usize>,
    axis: usize,
    range: Range<usize>,
    frame: usize,
) {
    let dt_size = data.datum_type().size_of();
    let shape: TVec<usize> = data.shape().into();
    let data = data.as_bytes_mut();
    let source = || lane_runs(&shape, dt_size, axis, seat, frame..frame + 1);
    for i in range {
        for (to, from) in lane_runs(&shape, dt_size, axis, seat, i..i + 1).zip(source()) {
            data.copy_within(from, to.start);
        }
    }
}

/// Repeat a lane of `frames` over `range` along `axis` of one seat of `data`.
/// `frames` holds one frame per lane, so its extent along `axis` is 1.
fn fill_from_frame(
    data: &mut Tensor,
    seat: Option<usize>,
    axis: usize,
    range: Range<usize>,
    frames: &Tensor,
    lane: Option<usize>,
) -> TractResult<()> {
    for frame in range {
        data.assign_slice_at_prefix(
            seat.as_slice(),
            frame..frame + 1,
            frames,
            lane.as_slice(),
            0..1,
            axis,
        )?;
    }
    Ok(())
}

/// One padding state per lane: `current_pos` is each lane's position in its own
/// stream and `last_valid_frame` each lane's last frame of valid input, for edge
/// padding. `limits` is per turn rather than per lane: `end_input` and `after`
/// resolve from the turn's symbols, which a turn shares across its seats.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
struct PulsePadOpState {
    current_pos: TVec<usize>,
    last_valid_frame: Option<Tensor>,
    limits: Option<PadLimits>,
    lanes: usize,
}

/// `end_input` and `after` resolved to concrete values, plus the symbol
/// bindings the resolution depended on. Re-evaluating the TDim expressions
/// every pulse costs microseconds; the bindings can only change when the
/// runner binds the stream-length symbol, so the cache revalidates by
/// re-reading just those bindings.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PadLimits {
    deps: Vec<(Symbol, Option<i64>)>,
    end_input: usize,
    after: usize,
}

impl PadLimits {
    fn resolve(cache: &mut Option<PadLimits>, ctx: &EvalContext, op: &PulsePad) -> (usize, usize) {
        if let Some(l) = cache.as_ref() {
            if l.deps.iter().all(|(s, v)| ctx.symbols.get(s) == *v) {
                return (l.end_input, l.after);
            }
        }
        let end_input = op.end_input.eval(ctx.symbols).to_usize().unwrap_or(usize::MAX);
        let after = op.after.eval(ctx.symbols).to_usize().unwrap_or(usize::MAX);
        let deps = op
            .end_input
            .symbols()
            .into_iter()
            .chain(op.after.symbols())
            .map(|s| {
                let v = ctx.symbols.get(&s);
                (s, v)
            })
            .collect();
        *cache = Some(PadLimits { deps, end_input, after });
        (end_input, after)
    }
}

impl OpState for PulsePadOpState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let op = op.downcast_ref::<PulsePad>().ok_or_else(|| format_err!("Wrong Op type"))?;
        self.pad(ctx, op, input).map(|t| tvec!(t))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        if self.lanes == 0 {
            return Ok(());
        }
        ensure!(
            lanes.iter().all(|l| l.0 < self.lanes),
            "PulsePad holds {} lanes, asked to reset {lanes:?}",
            self.lanes
        );
        for lane in lanes {
            self.current_pos[lane.0] = 0;
            if let Some(frames) = self.last_valid_frame.as_mut() {
                let stride = frames.as_bytes().len() / self.lanes;
                frames.as_bytes_mut()[lane.0 * stride..][..stride].fill(0);
            }
        }
        Ok(())
    }
}

impl PulsePadOpState {
    fn save_frame(
        &mut self,
        op: &PulsePad,
        input: &Tensor,
        frame: usize,
        seat: Option<usize>,
        lane: Option<usize>,
    ) -> TractResult<()> {
        let frames = match self.last_valid_frame.as_mut() {
            Some(frames) => frames,
            None => {
                let mut shape: TVec<usize> = input.shape().into();
                shape[op.axis] = 1;
                if lane.is_some() {
                    shape[0] = self.lanes;
                }
                self.last_valid_frame.insert(Tensor::zero_dt(input.datum_type(), &shape)?)
            }
        };
        frames.assign_slice_at_prefix(
            lane.as_slice(),
            0..1,
            input,
            seat.as_slice(),
            frame..frame + 1,
            op.axis,
        )
    }

    fn pad(&mut self, ctx: &EvalContext, op: &PulsePad, input: TValue) -> TractResult<TValue> {
        let dt = input.datum_type();
        ensure!(dt.is_copy(), "PulsePad pads {dt:?}, which is not copy");
        let (end_input, after) = PadLimits::resolve(&mut self.limits, ctx, op);
        let max_lanes = ctx.seating.max_lanes();
        if self.lanes == 0 {
            self.lanes = max_lanes;
            self.current_pos = tvec!(0; max_lanes);
        }
        ensure!(
            self.lanes == max_lanes,
            "PulsePad holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        let pulse = input.shape()[op.axis];
        let occupancy = ctx.seating.occupancy();
        if max_lanes > 1 {
            ensure!(op.axis > 0, "PulsePad on axis 0 leaves no axis 0 for the lanes");
            ensure!(
                input.shape()[0] == occupancy,
                "PulsePad input carries {} streams, this turn seats {occupancy}",
                input.shape()[0]
            );
        }
        // Seats whose pulse is neither entirely valid input nor entirely outside
        // it, with the stream position they start at. Every other seat forwards.
        let mut to_pad: TVec<(Option<usize>, Option<usize>, usize)> = tvec!();
        for ix in 0..occupancy {
            let (seat, lane) = ctx.seating.address(ix);
            let pulse_begin = self.current_pos[lane.unwrap_or(0)];
            let pulse_end = pulse_begin + pulse;
            self.current_pos[lane.unwrap_or(0)] += pulse - op.overlap;
            if let PadMode::Edge = op.mode {
                if after != 0 && pulse_begin < end_input {
                    let latest_valid_frame = (end_input - pulse_begin).min(pulse) - 1;
                    self.save_frame(op, &input, latest_valid_frame, seat, lane)?;
                }
            }
            let valid = pulse_begin >= op.begin_input && pulse_end <= end_input;
            let outside = pulse_end <= op.begin_input - op.before
                || pulse_begin >= end_input.saturating_add(after);
            if !valid && !outside {
                to_pad.push((seat, lane, pulse_begin));
            }
        }
        // Keep the value shared when nothing needs padding: materializing an
        // owned tensor here would copy the whole pulse.
        if to_pad.is_empty() {
            return Ok(input);
        }

        let mut output = input.into_tensor();
        for (seat, lane, pulse_begin) in to_pad {
            if pulse_begin < op.begin_input {
                let fill_up_to = (op.begin_input - pulse_begin).min(pulse);
                match &op.mode {
                    PadMode::Constant(c) => output.fill_slice_at_prefix(
                        seat.as_slice(),
                        0..fill_up_to,
                        &*c.cast_to_dt(dt)?,
                        op.axis,
                    )?,
                    PadMode::Edge => {
                        fill_from_own_frame(&mut output, seat, op.axis, 0..fill_up_to, fill_up_to)
                    }
                    _ => unimplemented!(),
                }
            }
            if pulse_begin + pulse > end_input && after > 0 {
                let fill_from = pulse - (pulse_begin + pulse - end_input).min(pulse);
                match &op.mode {
                    PadMode::Constant(c) => output.fill_slice_at_prefix(
                        seat.as_slice(),
                        fill_from..pulse,
                        &*c.cast_to_dt(dt)?,
                        op.axis,
                    )?,
                    PadMode::Edge => {
                        let frames = self.last_valid_frame.as_ref().unwrap();
                        fill_from_frame(
                            &mut output,
                            seat,
                            op.axis,
                            fill_from..pulse,
                            frames,
                            lane,
                        )?;
                    }
                    _ => unimplemented!(),
                }
            }
        }

        Ok(output.into_tvalue())
    }
}

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct PulsePad {
    pub axis: usize,
    pub before: usize,
    pub after: TDim,
    pub begin_input: usize,
    pub end_input: TDim,
    pub mode: PadMode,
    pub overlap: usize,
}

impl Op for PulsePad {
    fn name(&self) -> StaticName {
        "PulsePad".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "Mode: {:?}, axis: {} before: {} after: {}",
            self.mode, self.axis, self.before, self.after,
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for PulsePad {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsePadOpState>::default()))
    }
}

impl TypedOp for PulsePad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An f16 pulse padded by the f32 constant a model builder leaves in
    /// `PadMode`, on both the leading and the trailing pad.
    #[test]
    fn constant_is_cast_to_the_pulse_datum_type() -> TractResult<()> {
        let mut model = TypedModel::default();
        let source = model.add_source("source", f16::fact([2]))?;
        let pad = model.wire_node(
            "pad",
            PulsePad {
                axis: 0,
                before: 2,
                after: 2.to_dim(),
                begin_input: 2,
                end_input: 6.to_dim(),
                mode: PadMode::Constant(rctensor0(-1f32)),
                overlap: 0,
            },
            &[source],
        )?;
        model.select_output_outlets(&pad)?;

        let mut state = model.into_runnable()?.spawn()?;
        let minus_one = f16::from_f32(-1.0);
        let mut got = vec![];
        for pulse in 0..4 {
            let input = tensor1(&[f16::from_f32(pulse as f32), f16::from_f32(pulse as f32)]);
            let output = state.run(tvec!(input.into_tvalue()))?;
            got.extend_from_slice(output[0].try_as_plain()?.as_slice::<f16>()?);
        }
        assert_eq!(got[0..2], [minus_one; 2]);
        assert_eq!(
            got[2..6],
            [f16::from_f32(1.0), f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(2.0)]
        );
        assert_eq!(got[6..8], [minus_one; 2]);
        Ok(())
    }
}
