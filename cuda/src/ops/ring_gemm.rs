use tract_core::internal::natural_strides;
use tract_core::internal::*;
use tract_gpu::device::get_context;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::fused_output::{chain_shape, unchain_strides};
use tract_gpu::ops::pulse::copy_lane;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_gpu::turn_handler::make_tensor_for_node;
use tract_pulse_opl::ops::Delay;

use crate::kernels::matmul::{GgmlGemm, RingAxis};
use crate::ops::CudaGgmlGemm;

/// A pulsed window and the GEMM reading it, fused: the window stays a ring of
/// `overlap + 1` slots, held in the layout the GEMM wants, and each turn writes
/// its own slot into it and hands the kernel every seat's rotation instead of a
/// contiguous copy of the window.
///
/// `window_axis_ops` is the axis-op chain the window reached its GEMM through,
/// and so the layout the ring is held in. The inputs are the GEMM's activations
/// and the turn's slot, in that order, the window being the GEMM's second
/// operand. Sound only for a window the rotation of a ring can stand for: a
/// `Delay` of no delay and a zeroed pad, a slot per turn, and a layout under
/// which a slot is either a run of whole rows of the window or an even run of
/// the columns of every one of its rows -- the two axes a ring can turn on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaRingGemm {
    pub delay: Delay,
    pub window_axis_ops: TVec<GpuAxisOp>,
}

impl CudaRingGemm {
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

impl Op for CudaRingGemm {
    fn name(&self) -> StaticName {
        "CudaRingGemm".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!("axis: {} overlap: {}", self.delay.axis, self.delay.overlap),
            format!(
                "ring of {} slots, held as: {:?}",
                self.slots(),
                self.window_axis_ops.iter().map(|o| &o.inner).collect::<Vec<_>>()
            ),
        ])
    }

    op_as_typed_op!();
}

impl EvalOp for CudaRingGemm {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(CudaRingGemmState {
            ring: None,
            slots_view: None,
            table: None,
            heads: tvec!(),
            lanes: 0,
        })))
    }
}

impl TypedOp for CudaRingGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            let window = self.window_fact(facts[1])?;
            CudaGgmlGemm.resolve_output_facts(&[facts[0], &window])
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |facts| {
            let window = self.window_fact(facts[1])?;
            let mut cost = CudaGgmlGemm.cost(&[facts[0], &window])?;
            cost.extend(self.delay.cost(&[facts[1]])?);
            Ok(cost)
        })
    }

    as_op!();
}

/// The window as a ring, one per lane: `heads` is each lane's slot index of the
/// oldest slot of its window, `slots_view` addresses the ring in the slot order
/// a turn writes through, and `table` is the kernel's copy of where each
/// activation channel reads, rebuilt each turn.
#[derive(Debug, Clone)]
pub struct CudaRingGemmState {
    ring: Option<DeviceTensor>,
    slots_view: Option<DeviceTensor>,
    table: Option<DeviceTensor>,
    heads: TVec<usize>,
    lanes: usize,
}

impl CudaRingGemmState {
    fn init(
        &mut self,
        op: &CudaRingGemm,
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

    /// Where each activation channel reads: how far its window has turned,
    /// counted in whatever unit a slot spans along the axis the ring turns on,
    /// and the row its lane starts at. A seat holds `channels / occupancy`
    /// consecutive channels of the turn, a lane as many of the ring.
    fn fill_table(
        &mut self,
        ctx: &EvalContext,
        channels: usize,
        n: usize,
        slot_unit: usize,
    ) -> TractResult<()> {
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
                table[2 * channel + 1] = ((lane.unwrap_or(0) * per_seat + c) * n) as i32;
            }
        }
        self.table = Some(tensor1(&table).into_device()?);
        Ok(())
    }
}

impl OpState for CudaRingGemmState {
    fn eval(
        &mut self,
        ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (act, slot) = args_2!(inputs);
        let op = op.downcast_ref::<CudaRingGemm>().context("Wrong Op type")?;
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
        let out_shape = GgmlGemm.output_shape(act.shape(), &w_shape);
        let out = make_tensor_for_node(ctx, act.datum_type(), &out_shape)?;
        if out.len() == 0 {
            return Ok(tvec!(out.into_tensor().into()));
        }

        let (n, k) = (w_shape[w_shape.len() - 2], w_shape[w_shape.len() - 1]);
        let channels: usize = w_shape[..w_shape.len() - 2].iter().product();
        let slots_view = self.slots_view.clone().unwrap();
        let slot_span = slots_view.strides()[axis] as usize;
        let (ring_axis, slot_unit) = if slot_span.is_multiple_of(k) {
            ensure!(
                slot_span * slots == n * k,
                "A slot spans {slot_span} of a {n}x{k} window, which is not a run of whole rows"
            );
            (RingAxis::Rows, slot_span / k)
        } else {
            ensure!(
                slot_span * slots == k && slot_span.is_multiple_of(2),
                "A slot spans {slot_span} of a {n}x{k} window, which is neither a run of \
                 whole rows nor an even run of every row's columns"
            );
            (RingAxis::Contracted, slot_span / 2)
        };

        self.fill_table(ctx, channels, n, slot_unit)?;
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
        let table = self.table.as_ref().unwrap();
        crate::with_cuda_stream(|stream| {
            GgmlGemm.dispatch_matvec_ring(stream, act, ring, &out, &w_shape, table, ring_axis)
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
