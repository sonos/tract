use tract_core::internal::natural_strides;
use tract_core::internal::*;
use tract_gpu::device::get_context;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::fused_output::{chain_shape, unchain_strides};
use tract_gpu::ops::pulse::copy_lane;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_gpu::turn_handler::make_tensor_for_node;
use tract_pulse_opl::ops::Delay;

use crate::kernels::matmul::ring::{RingAxis, dispatch_ring_matvec};
use crate::kernels::matmul::{BasicMatMul, GemmImpl};

/// A pulsed window and the GEMM reading it as its second operand, fused: the
/// window stays a ring of `overlap + 1` slots, held in the layout the GEMM
/// reads (`[.., n, k]` when `transpose_b`, `[.., k, n]` otherwise), and each
/// turn writes its own slot into it and hands the kernel every channel's
/// rotation instead of a contiguous copy of the window.
///
/// `window_axis_ops` is the axis-op chain the window reached its GEMM through,
/// and so the layout the ring is held in. The inputs are the GEMM's activations
/// (`[.., m, k]`, at most `RING_MATVEC_MAX_M` rows) and the turn's slot. Sound
/// only for a window the rotation of a ring can stand for: a `Delay` of no
/// delay and a zeroed pad, a slot per turn, and a layout under which a slot is
/// a run of whole rows of the window or a run of the columns of every row.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalRingGemm {
    pub delay: Delay,
    pub window_axis_ops: TVec<GpuAxisOp>,
    pub transpose_b: bool,
}

impl MetalRingGemm {
    fn gemm(&self) -> GemmImpl<BasicMatMul> {
        GemmImpl::new(false, self.transpose_b)
    }

    /// The number of slots the ring holds, the window being all of them.
    pub fn slots(&self) -> usize {
        self.delay.overlap + 1
    }

    /// The fact of the window as the GEMM sees it, from the fact of one slot.
    fn window_fact(&self, slot: &TypedFact) -> TractResult<TypedFact> {
        let mut fact = self.delay.output_facts(&[slot])?.remove(0);
        for op in &self.window_axis_ops {
            op.inner.change_shape(&mut fact.shape, false)?;
        }
        Ok(fact)
    }
}

impl Op for MetalRingGemm {
    fn name(&self) -> StaticName {
        "MetalRingGemm".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!(
                "axis: {} overlap: {} transpose_b: {}",
                self.delay.axis, self.delay.overlap, self.transpose_b
            ),
            format!(
                "ring of {} slots, held as: {:?}",
                self.slots(),
                self.window_axis_ops.iter().map(|o| &o.inner).collect::<Vec<_>>()
            ),
        ])
    }

    op_as_typed_op!();
}

impl EvalOp for MetalRingGemm {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(MetalRingGemmState {
            ring: None,
            slots_view: None,
            heads: tvec!(),
            lanes: 0,
        })))
    }
}

impl TypedOp for MetalRingGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            let window = self.window_fact(facts[1])?;
            let gemm = self.gemm();
            let shape = gemm.output_shape(&facts[0].shape, &window.shape);
            gemm.output_facts(&shape, facts[0].datum_type, window.datum_type)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |facts| {
            let window = self.window_fact(facts[1])?;
            let k = facts[0].shape.last().unwrap().clone();
            let out = self.gemm().output_shape(&facts[0].shape, &window.shape);
            let mut cost =
                tvec!((Cost::FMA(facts[0].datum_type), out.iter().product::<TDim>() * k));
            cost.extend(self.delay.cost(&[facts[1]])?);
            Ok(cost)
        })
    }

    as_op!();
}

/// The window as a ring, one per lane: `heads` is each lane's slot index of the
/// oldest slot of its window, and `slots_view` addresses the ring in the slot
/// order a turn writes through.
#[derive(Debug)]
pub struct MetalRingGemmState {
    ring: Option<DeviceTensor>,
    slots_view: Option<DeviceTensor>,
    heads: TVec<usize>,
    lanes: usize,
}

impl Clone for MetalRingGemmState {
    fn clone(&self) -> Self {
        let ring = self.ring.as_ref().map(|r| r.deep_clone().expect("cloning a ring"));
        let slots_view = ring.as_ref().zip(self.slots_view.as_ref()).map(|(ring, view)| {
            ring.reshaped(view.shape().into())
                .and_then(|r| r.restrided(view.strides().into()))
                .expect("re-viewing a cloned ring")
        });
        MetalRingGemmState { ring, slots_view, heads: self.heads.clone(), lanes: self.lanes }
    }
}

impl MetalRingGemmState {
    fn init(
        &mut self,
        op: &MetalRingGemm,
        ctx: &EvalContext,
        slot: &DeviceTensor,
    ) -> TractResult<()> {
        let axis = op.delay.axis;
        let max_lanes = ctx.seating.max_lanes();
        let mut slots_shape: TVec<usize> = slot.shape().into();
        slots_shape[axis] = op.slots();
        // The layout the window reaches its GEMM in, for the streams this turn
        // seats.
        let held = chain_shape(&slots_shape, &op.window_axis_ops, ctx.symbols)?;
        let (unchained, mut strides) =
            unchain_strides(&held, &natural_strides(&held), &op.window_axis_ops, ctx.symbols)?;
        ensure!(
            unchained == slots_shape,
            "The ring's layout leads to {unchained:?}, not {slots_shape:?}"
        );
        // A laned ring holds a lane per seat the turn may ever seat, and the
        // chain can fold the lane axis into another, so it holds one stream's
        // worth of that layout per lane rather than being chained at its full
        // width. A ring one stream owns is that layout whole.
        let ring_shape: TVec<usize> = if max_lanes > 1 {
            ensure!(axis > 0, "A ring on axis 0 leaves no axis 0 for the lanes");
            let lane_span = held.iter().product::<usize>() / slots_shape[0];
            ensure!(
                slots_shape[0] == 1 || strides[0] as usize == lane_span,
                "A ring held as {held:?} gives a stream stride of {}, not the {lane_span} a \
                 whole stream of it spans",
                strides[0]
            );
            slots_shape[0] = max_lanes;
            strides[0] = lane_span as isize;
            tvec!(max_lanes, lane_span)
        } else {
            held
        };
        let ring = Tensor::zero_dt(slot.datum_type(), &ring_shape)?.into_device()?;
        self.slots_view = Some(ring.reshaped(slots_shape)?.restrided(strides)?);
        self.ring = Some(ring);
        self.heads = tvec!(0; max_lanes);
        self.lanes = max_lanes;
        Ok(())
    }

    /// Where each channel's window reads: how far it has turned, counted in
    /// whatever unit a slot spans along the axis the ring turns on, and the
    /// element its first row starts at. A seat holds `channels / occupancy`
    /// consecutive channels of the turn, a lane as many of the ring.
    fn table(
        &self,
        ctx: &EvalContext,
        channels: usize,
        window_len: usize,
        slot_unit: usize,
    ) -> TractResult<DeviceTensor> {
        let occupancy = ctx.seating.occupancy();
        ensure!(channels.is_multiple_of(occupancy), "{channels} channels over {occupancy} seats");
        let per_seat = channels / occupancy;
        let mut table: Vec<i32> = vec![0; 2 * channels];
        for ix in 0..occupancy {
            let (seat, lane) = ctx.seating.address(ix);
            let head = self.heads[lane.unwrap_or(0)];
            for c in 0..per_seat {
                let channel = seat.unwrap_or(0) * per_seat + c;
                table[2 * channel] = (head * slot_unit) as i32;
                table[2 * channel + 1] =
                    i32::try_from((lane.unwrap_or(0) * per_seat + c) * window_len)?;
            }
        }
        tensor1(&table).into_device()
    }
}

impl OpState for MetalRingGemmState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (act, slot) = args_2!(inputs);
        let op = op.downcast_ref::<MetalRingGemm>().context("Wrong Op type")?;
        let act = act.as_device_tensor().context("Expected a GPU tensor")?;
        let slot = slot.as_device_tensor().context("Expected a GPU tensor")?;
        let axis = op.delay.axis;
        let slots = op.slots();
        let max_lanes = ctx.seating.max_lanes();
        let occupancy = ctx.seating.occupancy();

        if self.ring.is_none() {
            self.init(op, ctx, slot)?;
        }
        ensure!(
            self.lanes == max_lanes,
            "The ring holds {} lanes, this turn seats {max_lanes} of them",
            self.lanes
        );
        if max_lanes > 1 {
            ensure!(
                slot.shape()[0] == occupancy,
                "The turn's slot carries {} streams, this turn seats {occupancy}",
                slot.shape()[0]
            );
        }

        let mut window_shape: TVec<usize> = slot.shape().into();
        window_shape[axis] = slots;
        let w_shape = chain_shape(&window_shape, &op.window_axis_ops, ctx.symbols)?;
        let rank = w_shape.len();
        // The window held as [rows, cols]: [n, k] under transpose_b, [k, n] otherwise.
        let (rows, cols) = (w_shape[rank - 2], w_shape[rank - 1]);
        let (k, n) = if op.transpose_b { (cols, rows) } else { (rows, cols) };
        let act_shape = act.shape();
        ensure!(
            act_shape.len() == rank && act_shape[..rank - 2] == w_shape[..rank - 2],
            "The ring's activation {act_shape:?} does not match its window {w_shape:?} channel \
             for channel"
        );
        ensure!(act_shape[rank - 1] == k);
        let m = act_shape[rank - 2];
        let out_shape = op.gemm().output_shape(act_shape, &w_shape);
        let out = make_tensor_for_node(ctx, act.datum_type(), &out_shape)?;
        if out.len() == 0 {
            return Ok(tvec!(out.into_tensor().into()));
        }

        let channels: usize = w_shape[..rank - 2].iter().product();
        let slots_view = self.slots_view.clone().unwrap();
        let slot_span = slots_view.strides()[axis] as usize;
        let (turns_rows, slot_unit) = if slot_span.is_multiple_of(cols) {
            ensure!(
                slot_span * slots == rows * cols,
                "A slot spans {slot_span} of a {rows}x{cols} window, which is not a run of \
                 whole rows"
            );
            (true, slot_span / cols)
        } else {
            ensure!(
                slot_span * slots == cols,
                "A slot spans {slot_span} of a {rows}x{cols} window, which is neither a run of \
                 whole rows nor a run of every row's columns"
            );
            (false, slot_span)
        };
        let ring_axis = if turns_rows != op.transpose_b { RingAxis::K } else { RingAxis::N };
        let strides = if op.transpose_b { (1, k) } else { (n, 1) };

        let table = self.table(ctx, channels, rows * cols, slot_unit)?;
        let device = get_context()?;
        for ix in 0..occupancy {
            let (seat, lane) = ctx.seating.address(ix);
            let head = self.heads[lane.unwrap_or(0)];
            copy_lane(
                &*device,
                &slots_view,
                lane,
                (head + slots - 1) % slots,
                slot,
                seat,
                0,
                axis,
                1,
            )?;
            self.heads[lane.unwrap_or(0)] = (head + 1) % slots;
        }

        let ring = self.ring.as_ref().unwrap();
        crate::with_metal_stream(|stream| {
            dispatch_ring_matvec(
                stream,
                act,
                ring,
                &out,
                &table,
                (channels, m, k, n),
                strides,
                ring_axis,
            )
        })?;
        Ok(tvec!(out.into_tensor().into()))
    }

    fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        let Some(ring) = self.ring.as_ref() else { return Ok(()) };
        ensure!(
            lanes.iter().all(|l| l.0 < self.lanes),
            "The ring holds {} lanes, asked to reset {lanes:?}",
            self.lanes
        );
        for lane in lanes {
            self.heads[lane.0] = 0;
        }
        tract_gpu::ops::pulse::zero_lanes(&*get_context()?, ring, lanes, self.lanes > 1)
    }
}
