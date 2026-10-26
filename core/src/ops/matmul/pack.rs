use crate::axes::Axis;
use crate::internal::*;
use ndarray::*;
use tract_linalg::block_quant::{
    BlockQuantStorage, PackedBlockQuantFact, PackedBlockQuantFormat, block_quant_slice,
};
use tract_linalg::mmm::{EagerPackedInput, MMMInputFormat, MMMInputValue, PackedMatrixStorage};
use tract_linalg::pack::PackedFormat;

use super::ModePicker;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptMatMulPack {
    pub(crate) packers: Vec<Box<dyn MMMInputFormat>>,
    pub(crate) mode_picker: ModePicker,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
}

impl Op for OptMatMulPack {
    fn name(&self) -> StaticName {
        "OptMatMulPack".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}. k axis: {}, mn axis: {}", self.packers, self.k_axis, self.mn_axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for OptMatMulPack {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs.remove(0);
        if ctx.session == SessionId::NONE {
            return self.do_eval(ctx, input);
        }
        KEPT_PACKS.with_borrow_mut(|kept| {
            kept.reap_ended();
            let session = kept.session(ctx.session);
            if let Kept::Packed(previous) = session.slot(ctx.node_id)
                && let Some(repacked) = self.repack(previous, &input)?
            {
                return Ok(tvec!(repacked));
            }
            let keep = !matches!(session.slot(ctx.node_id), Kept::Cold)
                && self.packed_format(input.shape()).is_some();
            let output = self.do_eval(ctx, input)?;
            session.keep(ctx.node_id, output[0].as_arc_tensor().filter(|_| keep));
            Ok(output)
        })
    }

    fn drop_session(&self, session: SessionId, _node_id: usize) {
        KEPT_PACKS.with_borrow_mut(|kept| kept.drop_session(session));
    }
}

crate::declare_knob!(
    TRACT_KEPT_PACK_BYTES,
    usize,
    64 << 20,
    "Bytes of packed matmul operands a session may keep between turns to repack in place."
);

/// What a pack node keeps between the turns of a session: nothing before its
/// first turn, so a session run once frees its packs as soon as they are
/// consumed, then its last packed output, refilled in place by the next turn
/// when no one else holds it, as long as the session's packs fit its budget.
#[derive(Clone, Debug, Default)]
enum Kept {
    #[default]
    Cold,
    Warm,
    Packed(Arc<Tensor>),
}

/// The pack nodes' [`Kept`] slots of each session evaluated on this thread,
/// indexed by node id. A cache: it carries nothing a stream depends on, and a
/// session moved to another thread just starts cold there. A session's slots
/// go when the plan calls [`EvalOp::drop_session`] on this thread, their
/// allocation kept for the thread's next session, or else at this thread's
/// next pack eval after the session ends elsewhere.
struct KeptPacks {
    sessions: Vec<KeptSession>,
    spare: Vec<Kept>,
    reaper: SessionReaper,
}

/// One session's slots, with the bytes its kept packs hold and the
/// [`TRACT_KEPT_PACK_BYTES`] budget read when the session first ran here.
struct KeptSession {
    id: SessionId,
    slots: Vec<Kept>,
    bytes: usize,
    budget: usize,
}

impl KeptPacks {
    fn session(&mut self, id: SessionId) -> &mut KeptSession {
        let ix = match self.sessions.iter().position(|s| s.id == id) {
            Some(ix) => ix,
            None => {
                self.sessions.push(KeptSession {
                    id,
                    slots: std::mem::take(&mut self.spare),
                    bytes: 0,
                    budget: TRACT_KEPT_PACK_BYTES.get(),
                });
                self.sessions.len() - 1
            }
        };
        &mut self.sessions[ix]
    }

    fn reap_ended(&mut self) {
        let sessions = &mut self.sessions;
        self.reaper.reap(|live| sessions.retain(|s| live(s.id)));
    }

    fn drop_session(&mut self, id: SessionId) {
        if let Some(ix) = self.sessions.iter().position(|s| s.id == id) {
            let mut slots = self.sessions.swap_remove(ix).slots;
            slots.clear();
            self.spare = slots;
        }
    }
}

impl KeptSession {
    fn slot(&mut self, node_id: usize) -> &mut Kept {
        if self.slots.len() <= node_id {
            self.slots.resize(node_id + 1, Kept::Cold);
        }
        &mut self.slots[node_id]
    }

    /// Makes `packed` the node's kept pack, or leaves the node warm when there
    /// is none or it does not fit what the session's budget has left.
    fn keep(&mut self, node_id: usize, packed: Option<&Arc<Tensor>>) {
        if let Kept::Packed(previous) = self.slot(node_id) {
            self.bytes -= packed_len(previous);
        }
        let len = packed.map_or(0, |packed| packed_len(packed));
        self.slots[node_id] = match packed {
            Some(packed) if self.bytes + len <= self.budget => {
                self.bytes += len;
                Kept::Packed(packed.clone())
            }
            _ => Kept::Warm,
        };
    }
}

/// The bytes of the packed blobs a plain pack output holds.
fn packed_len(packed: &Tensor) -> usize {
    packed.storage_as::<PackedMatrixStorage>().map_or(0, |storage| {
        storage
            .values()
            .iter()
            .filter_map(|value| value.downcast_ref::<EagerPackedInput>())
            .map(|value| value.packed.len())
            .sum()
    })
}

thread_local! {
    static KEPT_PACKS: std::cell::RefCell<KeptPacks> =
        const { std::cell::RefCell::new(KeptPacks {
            sessions: Vec::new(),
            spare: Vec::new(),
            reaper: SessionReaper::new(),
        }) };
}

impl TypedOp for OptMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        match self.mode_picker {
            ModePicker::Single => ensure!(self.packers.len() == 1),
            ModePicker::VecVsMat => ensure!(self.packers.len() == 2),
        }
        let k = inputs[0].shape[self.k_axis].clone();
        let mn = inputs[0].shape[self.mn_axis].clone();
        let exotic_fact = DynPackedExoticFact { k, mn, packers: self.packers.clone() };
        Ok(tvec!(
            inputs[0]
                .datum_type
                .fact(self.output_shape(&inputs[0].shape))
                .with_exotic_fact(exotic_fact)
        ))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes: Vec<Axis> = (0..inputs[0].rank())
            .filter(|&ix| ix != self.k_axis && ix != self.mn_axis)
            .enumerate()
            .zip('a'..)
            .map(|((o, i), repr)| Axis::new(repr, 1, 1).input(0, i).output(0, o))
            .collect();
        axes.push(Axis::new('K', 1, 1).input(0, self.k_axis));
        axes.push(Axis::new('M', 1, 1).input(0, self.mn_axis));
        axes.push(Axis::new('P', 1, 1).output(0, outputs[0].rank()));
        AxesMapping::new(1, 1, axes)
    }

    as_op!();
}

impl OptMatMulPack {
    fn do_eval(&self, _ctx: &EvalContext, input: TValue) -> TractResult<TVec<TValue>> {
        // The packers read the bytes unchecked: a weight whose bytes live elsewhere is fetched
        // here, where failing to is an error.
        input.try_as_plain_ram()?;
        let mode = self.mode_picker.pick(input.shape()[self.mn_axis])?;
        let packer = &self.packers[mode];
        let output_shape: TVec<usize> = self.output_shape(input.shape());
        if let Some(strided) = packer.downcast_ref::<tract_linalg::strided_panel::StridedKMajor>() {
            // Share the activation allocation. A copy here is the pack this format exists to skip.
            let arc = input.as_arc_tensor().unwrap().clone();
            let values = if output_shape.iter().all(|d| *d == 1) {
                tvec![strided.panel_from_arc(arc, 0, self.k_axis, self.mn_axis)?]
            } else {
                let mut bc_shape: TVec<usize> = input.shape().into();
                bc_shape[self.k_axis] = 1;
                bc_shape[self.mn_axis] = 1;
                let mut values: TVec<Box<dyn MMMInputValue>> =
                    TVec::with_capacity(output_shape.iter().product());
                for coord in indices(&*bc_shape) {
                    let offset = coord
                        .as_array_view()
                        .iter()
                        .zip(input.strides())
                        .map(|(x, s)| *x as isize * s)
                        .sum::<isize>()
                        * input.datum_type().size_of() as isize;
                    values.push(strided.panel_from_arc(
                        arc.clone(),
                        offset,
                        self.k_axis,
                        self.mn_axis,
                    )?);
                }
                values
            };
            let stores = PackedMatrixStorage::new_batched(&output_shape, values)
                .into_tensor(input.datum_type());
            return Ok(tvec!(stores.into_tvalue()));
        }
        unsafe {
            let stores = if output_shape.iter().all(|d| *d == 1) {
                let packed = packer.prepare_one_view(&input.view(), self.k_axis, self.mn_axis)?;
                PackedMatrixStorage::new_batched(&output_shape, tvec![packed])
                    .into_tensor(input.datum_type())
            } else {
                let mut bc_shape: TVec<usize> = input.shape().into();
                bc_shape[self.k_axis] = 1;
                bc_shape[self.mn_axis] = 1;

                let mut values: TVec<Box<dyn MMMInputValue>> =
                    TVec::with_capacity(output_shape.iter().product());
                for coord in indices(&*bc_shape) {
                    let offset = coord
                        .as_array_view()
                        .iter()
                        .zip(input.strides())
                        .map(|(x, s)| *x as isize * s)
                        .sum::<isize>()
                        * input.datum_type().size_of() as isize;
                    let view =
                        TensorView::from_bytes(&input, offset, input.shape(), input.strides());
                    values.push(packer.prepare_one_view(&view, self.k_axis, self.mn_axis)?);
                }
                PackedMatrixStorage::new_batched(&output_shape, values)
                    .into_tensor(input.datum_type())
            };
            Ok(tvec!(stores.into_tvalue()))
        }
    }

    /// The plain packed format `input` of this shape is packed with, if any:
    /// the only one whose output [`Self::repack`] can refill.
    fn packed_format(&self, input_shape: &[usize]) -> Option<&PackedFormat> {
        let mode = self.mode_picker.pick(input_shape[self.mn_axis]).ok()?;
        self.packers[mode].downcast_ref::<PackedFormat>()
    }

    /// Refill `previous`, this node's last output, with `input` in place. None
    /// when another holder still reads it or its layout no longer fits.
    fn repack(&self, previous: &mut Arc<Tensor>, input: &Tensor) -> TractResult<Option<TValue>> {
        let Some(format) = self.packed_format(input.shape()) else { return Ok(None) };
        let output_shape: TVec<usize> = self.output_shape(input.shape());
        let k = input.shape()[self.k_axis];
        let mn = input.shape()[self.mn_axis];
        let Some(tensor) = Arc::get_mut(previous) else { return Ok(None) };
        if tensor.shape() != &*output_shape {
            return Ok(None);
        }
        let Some(storage) = tensor.storage_as_mut::<PackedMatrixStorage>() else {
            return Ok(None);
        };
        let fits = |value: &mut Box<dyn MMMInputValue>| {
            value.downcast_mut::<EagerPackedInput>().is_some_and(|packed| {
                packed.fact.k == k
                    && packed.mn == mn
                    && packed.fact.format.dyn_eq(format)
                    && Arc::get_mut(&mut packed.packed).is_some()
            })
        };
        if !storage.values_mut().iter_mut().all(fits) {
            return Ok(None);
        }
        let mut bc_shape: TVec<usize> = input.shape().into();
        bc_shape[self.k_axis] = 1;
        bc_shape[self.mn_axis] = 1;
        for (value, coord) in storage.values_mut().iter_mut().zip(indices(&*bc_shape)) {
            let view = TensorView::offsetting(input, coord.slice())?;
            let packed = value.downcast_mut::<EagerPackedInput>().unwrap();
            format.repack_tensor_view(packed, &view, self.k_axis, self.mn_axis)?;
        }
        Ok(Some(previous.clone().into_tvalue()))
    }

    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut packed_shape: TVec<D> = input.into();
        packed_shape.remove(self.mn_axis.max(self.k_axis));
        packed_shape.remove(self.mn_axis.min(self.k_axis));
        packed_shape
    }
}

#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub struct DynPackedExoticFact {
    pub k: TDim,
    pub mn: TDim,
    pub packers: Vec<Box<dyn MMMInputFormat>>,
}

impl ExoticFact for DynPackedExoticFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(self.packers[0].mem_size(self.k.clone(), self.mn.clone()))
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct OptSimpleMatMulPack {
    pub(crate) packed_format: PackedBlockQuantFormat,
    pub(crate) k: usize,
    pub(crate) m: usize,
}

impl Op for OptSimpleMatMulPack {
    fn name(&self) -> StaticName {
        "OptSimpleMatMulPack".into()
    }
    op_as_typed_op!();
}

impl EvalOp for OptSimpleMatMulPack {
    op_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let bqs = input.try_storage_as::<BlockQuantStorage>()?;
        // Leading dims before the last 2 (M, K) are batch/group dims
        let num_groups: usize = input.shape()[..input.rank().saturating_sub(2)].iter().product();
        let m_per_group = input.shape()[input.rank() - 2];
        let k = *input.shape().last().unwrap();
        let values = (0..num_groups)
            .map(|g| {
                let slice = block_quant_slice(bqs.value(), bqs.format(), m_per_group, k, g);
                let iv: Box<dyn MMMInputValue> = Box::new(self.packed_format.pack(slice, k)?);
                Ok(iv)
            })
            .collect::<TractResult<TVec<_>>>()?;
        let leading_shape = &input.shape()[..input.rank().saturating_sub(2)];
        let output =
            PackedMatrixStorage::new_batched(leading_shape, values).into_tensor(input.datum_type());
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for OptSimpleMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        // Input shape is [G, M, K] — output removes M and K, keeping leading dims
        let output_shape: TVec<TDim> = if input.rank() > 2 {
            input.shape[..input.rank() - 2].to_vec().into()
        } else {
            tvec!()
        };
        let fact =
            inputs[0].datum_type.fact(&*output_shape).with_exotic_fact(PackedBlockQuantFact {
                format: self.packed_format.clone(),
                shape: tvec!(self.m, self.k),
            });
        Ok(tvec!(fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::einsum::EinSum;

    const G: usize = 4;
    const K: usize = 8;
    const N: usize = 6;

    /// A grouped vector-matrix product against constant weights, cut at the
    /// pack of its activation: one packed value per group, every turn.
    fn activation_pack() -> TractResult<(Arc<TypedSimplePlan>, OptMatMulPack)> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([G, 1, K]))?;
        let w: Vec<f32> = (0..G * K * N).map(|i| (i % 7) as f32 - 3.0).collect();
        let w = model.add_const("w", Tensor::from_shape(&[G, K, N], &w)?)?;
        let y = model.wire_node(
            "gemm",
            EinSum::new("gmk,gkn->gmn".parse()?, f32::datum_type()),
            &[x, w],
        )?;
        model.select_output_outlets(&y)?;
        let mut model = model.into_optimized()?;
        let pack = model
            .nodes()
            .iter()
            .find(|n| {
                n.op_is::<OptMatMulPack>()
                    && model.outlet_fact(n.inputs[0]).unwrap().konst.is_none()
            })
            .context("no pack of a dynamic operand")?;
        let op = pack.op_as::<OptMatMulPack>().unwrap().clone();
        model.select_output_outlets(&[pack.id.into()])?;
        Ok((model.into_runnable()?, op))
    }

    fn input(turn: usize) -> TractResult<TValue> {
        let x: Vec<f32> = (0..G * K).map(|i| ((i * (turn + 3)) % 5) as f32 - 2.0).collect();
        Ok(Tensor::from_shape(&[G, 1, K], &x)?.into_tvalue())
    }

    fn packed_bytes(t: &Tensor) -> Vec<Vec<u8>> {
        let storage = t.try_storage_as::<PackedMatrixStorage>().unwrap();
        storage
            .values()
            .iter()
            .map(|v| {
                let packed = v.downcast_ref::<EagerPackedInput>().unwrap();
                packed.packed.as_bytes()[..packed.k() * 4].to_vec()
            })
            .collect()
    }

    fn kept_packs() -> usize {
        KEPT_PACKS.with_borrow(|kept| {
            kept.sessions
                .iter()
                .flat_map(|session| &session.slots)
                .filter(|slot| matches!(slot, Kept::Packed(_)))
                .count()
        })
    }

    #[test]
    fn pack_is_kept_from_second_turn_until_session_drop() -> TractResult<()> {
        let (plan, _) = activation_pack()?;
        let mut state = plan.spawn()?;
        state.run(tvec!(input(0)?))?;
        assert_eq!(kept_packs(), 0);
        state.run(tvec!(input(1)?))?;
        assert_eq!(kept_packs(), 1);
        drop(state);
        assert_eq!(kept_packs(), 0);
        assert!(KEPT_PACKS.with_borrow(|kept| kept.sessions.is_empty()));
        Ok(())
    }

    #[test]
    fn pack_past_the_session_budget_is_not_kept() -> TractResult<()> {
        let (plan, _) = activation_pack()?;
        let mut state = plan.spawn()?;
        state.run(tvec!(input(0)?))?;
        KEPT_PACKS.with_borrow_mut(|kept| kept.sessions[0].budget = 0);
        state.run(tvec!(input(1)?))?;
        state.run(tvec!(input(2)?))?;
        assert_eq!(kept_packs(), 0);
        assert_eq!(KEPT_PACKS.with_borrow(|kept| kept.sessions[0].bytes), 0);
        Ok(())
    }

    #[test]
    fn pack_of_session_dropped_on_another_thread_is_reaped() -> TractResult<()> {
        let (plan, _) = activation_pack()?;
        let mut moved = plan.spawn()?;
        moved.run(tvec!(input(0)?))?;
        moved.run(tvec!(input(1)?))?;
        assert_eq!(kept_packs(), 1);
        std::thread::spawn(move || drop(moved)).join().unwrap();
        assert_eq!(kept_packs(), 1);
        plan.spawn()?.run(tvec!(input(0)?))?;
        assert_eq!(kept_packs(), 0);
        assert!(KEPT_PACKS.with_borrow(|kept| kept.sessions.is_empty()));
        Ok(())
    }

    #[test]
    fn dynamic_pack_is_refilled_in_place_when_released() -> TractResult<()> {
        let (plan, op) = activation_pack()?;
        let mut state = plan.spawn()?;
        state.run(tvec!(input(0)?))?;
        let kept_out = state.run(tvec!(input(1)?))?.remove(0);
        let kept_ptr = Arc::as_ptr(kept_out.as_arc_tensor().unwrap());
        drop(kept_out);
        for turn in 2..5 {
            let out = state.run(tvec!(input(turn)?))?.remove(0);
            assert_eq!(Arc::as_ptr(out.as_arc_tensor().unwrap()), kept_ptr);
            let fresh = op.eval(&EvalContext::out_of_plan(), tvec!(input(turn)?))?.remove(0);
            assert_eq!(packed_bytes(&out), packed_bytes(&fresh));
        }
        Ok(())
    }

    #[test]
    fn held_pack_is_not_overwritten() -> TractResult<()> {
        let (plan, _) = activation_pack()?;
        let mut state = plan.spawn()?;
        state.run(tvec!(input(0)?))?;
        let held = state.run(tvec!(input(1)?))?.remove(0);
        let before = packed_bytes(&held);
        let next = state.run(tvec!(input(2)?))?.remove(0);
        assert_ne!(
            Arc::as_ptr(next.as_arc_tensor().unwrap()),
            Arc::as_ptr(held.as_arc_tensor().unwrap())
        );
        assert_eq!(packed_bytes(&held), before);
        assert_ne!(packed_bytes(&next), before);
        Ok(())
    }

    #[test]
    fn sessions_keep_their_own_packs() -> TractResult<()> {
        let (plan, _) = activation_pack()?;
        let (mut a, mut b) = (plan.spawn()?, plan.spawn()?);
        for turn in 0..3 {
            let out_a = a.run(tvec!(input(turn)?))?.remove(0);
            let out_b = b.run(tvec!(input(turn + 1)?))?.remove(0);
            assert_ne!(packed_bytes(&out_a), packed_bytes(&out_b));
        }
        assert_eq!(kept_packs(), 2);
        Ok(())
    }
}
