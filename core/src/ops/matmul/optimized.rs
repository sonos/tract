use crate::internal::*;
use crate::ops::cast::{Cast, cast};
use crate::ops::change_axes::wire_with_rank_broadcast;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::nn::LeakyRelu;
use crate::ops::{FrozenOpState, OpStateFreeze};
use ndarray::*;
use tract_itertools::Itertools;

use tract_linalg::mmm::{
    AsInputValue, EagerPackedInput, FusedSpec, MMMInputValue, MatMatMul, OutputStore,
    OutputStoreSpec, PackedMatrixStorage, PanelExtractInput, PanelExtractor,
};
use tract_linalg::pack::PackedFormat;
use tract_linalg::{BinOp, Scaler};
use tract_smallvec::ToSmallVec;

use super::ModePicker;

/// If `new` is `old` with only size-1 axes dropped (non-unit axes untouched, in
/// order, nothing added or merged), return the removed axis indices; otherwise
/// `None`. Such a reshape is a pure metadata squeeze the matmul store can absorb.
fn pure_squeeze_removed(old: &[usize], new: &[usize]) -> Option<TVec<usize>> {
    let mut removed: TVec<usize> = tvec!();
    let mut j = 0;
    for (i, &d) in old.iter().enumerate() {
        if j < new.len() && d == new[j] {
            j += 1;
        } else if d == 1 {
            removed.push(i);
        } else {
            return None;
        }
    }
    (j == new.len() && !removed.is_empty()).then_some(removed)
}

/// A matmul operand: either an index into the runtime inputs, or a constant
/// packed value baked in at `fuse()` time (its source outlet was `konst`), so
/// it is never re-resolved per call. Only non-batched operands are baked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatMulOperand {
    Input(usize),
    Const(Box<dyn MMMInputValue>),
}

impl MatMulOperand {
    /// Base packed value for the trivial (non-batched) path.
    #[inline]
    unsafe fn trivial_value<'t>(&'t self, inputs: &'t [TValue]) -> &'t dyn MMMInputValue {
        match self {
            MatMulOperand::Input(i) => unsafe {
                inputs
                    .get_unchecked(*i)
                    .try_storage_as::<PackedMatrixStorage>()
                    .unwrap_unchecked()
                    .value()
            },
            MatMulOperand::Const(v) => &**v,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtoFusedSpec {
    AddMatMul {
        geo: AddMatMulGeometry,
        a: MatMulOperand,
        b: MatMulOperand,
        packings: Vec<(usize, Option<PanelExtractor>)>,
    },
    BinScalar(usize, BinOp),
    LeakyRelu(usize),
    BinPerRow(usize, BinOp, MapOutputAxisToInput),
    BinPerCol(usize, BinOp, MapOutputAxisToInput),
    AddRowColProducts(usize, usize),
    AddUnicast(OutputStoreSpec, usize, MapOutputAxisToInput),
    Scaler(Scaler),
    Store(Vec<OutputStoreSpec>),
}

impl ProtoFusedSpec {
    pub fn format(&self, mmm: &dyn MatMatMul, mode: usize) -> String {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul { geo, packings: packing, .. } => {
                let (a, b) = &mmm.packings()[packing[mode].0];
                format!("matmul(k={}, {a:?}•{b:?})", geo.k)
            }
            BinScalar(_, op) => format!("scalar{op:?}"),
            LeakyRelu(alpha) => format!("leaky_relu({alpha:?})"),
            BinPerRow(_, op, _) => format!("row{op:?}"),
            BinPerCol(_, op, _) => format!("col{op:?}"),
            AddRowColProducts(_, _) => "add_row_col_product".to_string(),
            AddUnicast(_, _, _) => "add_to_matrix".to_string(),
            Scaler(s) => format!("scale({})", 1f32 * *s),
            Store(_oss) => "store".to_string(),
        }
    }

    pub fn resolve<'t>(
        &'t self,
        inputs: &'t [TValue],
        output_coords: &[usize],
        output: &Tensor,
        mmm: &dyn MatMatMul,
        mode: usize,
    ) -> FusedSpec<'t> {
        #[allow(clippy::let_and_return)]
        let fs = match self {
            ProtoFusedSpec::AddMatMul { geo, a, b, packings } => {
                let resolve =
                    |operand: &'t MatMulOperand, mapping: &MapOutputAxisToInput| match operand {
                        MatMulOperand::Input(i) => {
                            let storage =
                                inputs[*i].try_storage_as::<PackedMatrixStorage>().unwrap();
                            let idx = mapping.flat_index(output_coords, storage.batch_strides());
                            storage.value_at_flat(idx)
                        }
                        MatMulOperand::Const(v) => &**v,
                    };
                let a = resolve(a, &geo.c_to_a_axis_mapping);
                let b = resolve(b, &geo.c_to_b_axis_mapping);

                let (_a_packing, b_packing) = &mmm.packings()[packings[mode].0];
                let pa = if let Some(extractor) = &packings[mode].1 {
                    let data = a.downcast_ref::<EagerPackedInput>().unwrap();
                    AsInputValue::Owned(Box::new(PanelExtractInput {
                        format: extractor.clone(),
                        data: data.clone(),
                    }))
                } else {
                    AsInputValue::Borrowed(a)
                };
                assert!(
                    b_packing.dyn_eq(b.format())
                        || (b_packing.is::<PackedFormat>() && b_packing.r() == b.format().r())
                );
                debug_assert!(pa.k().to_dim().compatible_with(&geo.k.to_dim()));
                debug_assert!(b.k().to_dim().compatible_with(&geo.k.to_dim()));
                FusedSpec::AddMatMul {
                    a: pa,
                    b: AsInputValue::Borrowed(b),
                    packing: packings[mode].0,
                }
            }
            ProtoFusedSpec::BinScalar(v, op) => FusedSpec::BinScalar(&inputs[*v], *op),
            ProtoFusedSpec::LeakyRelu(v) => FusedSpec::LeakyRelu(&inputs[*v]),
            ProtoFusedSpec::BinPerRow(v, op, map) => {
                let mut v = inputs[*v].view();
                unsafe { map.translate_view(output_coords, &mut v) }
                FusedSpec::BinPerRow(v, *op)
            }
            ProtoFusedSpec::BinPerCol(v, op, map) => {
                let mut v = inputs[*v].view();
                unsafe { map.translate_view(output_coords, &mut v) }
                FusedSpec::BinPerCol(v, *op)
            }
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(&inputs[*row], &inputs[*col])
            }
            ProtoFusedSpec::AddUnicast(store, v, map) => unsafe {
                let mut view = inputs[*v].view();
                map.translate_view(output_coords, &mut view);
                FusedSpec::AddUnicast(store.wrap(&view))
            },
            ProtoFusedSpec::Scaler(scaler) => scaler.as_fused_spec(),
            ProtoFusedSpec::Store(oss) => unsafe {
                let view = output.view_offsetting_unchecked(output_coords);
                FusedSpec::Store(oss[mode].wrap(&view))
            },
        };
        fs
    }

    pub fn is_trivial(&self) -> bool {
        match self {
            ProtoFusedSpec::AddMatMul { geo, .. } => geo.k.as_i64().is_some(),
            _ => true,
        }
    }

    pub fn resolve_trivial<'t>(
        &'t self,
        inputs: &'t [TValue],
        output: &mut Tensor,
        _mmm: &dyn MatMatMul,
        mode: usize,
    ) -> FusedSpec<'t> {
        #[allow(clippy::let_and_return)]
        let fs = match self {
            ProtoFusedSpec::AddMatMul { a, b, packings, .. } => unsafe {
                let a = a.trivial_value(inputs);
                let b = b.trivial_value(inputs);
                debug_assert!(packings.len() == 1);
                debug_assert!(packings[0].1.is_none()); // no panel extraction
                #[cfg(debug_assertions)]
                {
                    let (a_packing, b_packing) = &_mmm.packings()[packings[mode].0];
                    debug_assert!(
                        a_packing.dyn_eq(a.format())
                            || (a_packing.is::<PackedFormat>() && a_packing.r() == a.format().r())
                    );
                    debug_assert!(
                        b_packing.dyn_eq(b.format())
                            || (b_packing.is::<PackedFormat>() && b_packing.r() == b.format().r())
                    );
                }
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(a),
                    b: AsInputValue::Borrowed(b),
                    packing: packings[mode].0,
                }
            },
            ProtoFusedSpec::BinScalar(v, op) => FusedSpec::BinScalar(&inputs[*v], *op),
            ProtoFusedSpec::LeakyRelu(v) => FusedSpec::LeakyRelu(&inputs[*v]),
            ProtoFusedSpec::BinPerRow(v, op, _) => {
                let v = inputs[*v].view();
                FusedSpec::BinPerRow(v, *op)
            }
            ProtoFusedSpec::BinPerCol(v, op, _) => {
                let v = inputs[*v].view();
                FusedSpec::BinPerCol(v, *op)
            }
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(&inputs[*row], &inputs[*col])
            }
            ProtoFusedSpec::AddUnicast(store, v, _) => unsafe {
                let view = inputs[*v].view();
                FusedSpec::AddUnicast(store.wrap(&view))
            },
            ProtoFusedSpec::Scaler(scaler) => scaler.as_fused_spec(),
            ProtoFusedSpec::Store(oss) => unsafe {
                FusedSpec::Store(oss[mode].wrap(&output.view_mut()))
            },
        };
        fs
    }

    /// Like [`resolve_trivial`], but a `Store` reuses a cached [`OutputStore`]
    /// whose strides/layout are fixed for the output shape, refreshing only its
    /// base pointer from `output`. Everything else defers to [`resolve_trivial`]
    /// (constant operands are already baked into the op, so no per-call work).
    fn resolve_trivial_cached<'t>(
        &'t self,
        inputs: &'t [TValue],
        output: &mut Tensor,
        mmm: &dyn MatMatMul,
        mode: usize,
        store: Option<OutputStore>,
    ) -> FusedSpec<'t> {
        match self {
            ProtoFusedSpec::Store(oss) => unsafe {
                FusedSpec::Store(match store {
                    Some(cached) => cached.with_tensor(&output.view()),
                    None => oss[mode].wrap(&output.view_mut()),
                })
            },
            _ => self.resolve_trivial(inputs, output, mmm, mode),
        }
    }

    fn check_inputs(&self, inputs: &[&TypedFact]) -> TractResult<()> {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul { a, b, .. } => {
                for operand in [a, b] {
                    if let MatMulOperand::Input(ix) = operand {
                        ensure!(inputs[*ix].is_exotic());
                    }
                }
            }
            BinScalar(v, _)
            | LeakyRelu(v)
            | BinPerCol(v, _, _)
            | BinPerRow(v, _, _)
            | AddUnicast(_, v, _) => {
                ensure!(inputs[*v].datum_type.is_number());
            }
            AddRowColProducts(row, col) => {
                ensure!(inputs[*row].datum_type.is_number());
                ensure!(inputs[*col].datum_type.is_number());
            }
            _ => (),
        };
        Ok(())
    }

    fn cost(&self, m: &TDim, n: &TDim, idt: DatumType) -> TVec<(Cost, TDim)> {
        match self {
            ProtoFusedSpec::AddMatMul { geo, .. } => {
                tvec!((Cost::FMA(idt), m.clone() * n * &geo.k))
            }
            _ => tvec!(), /* FIXME maybe */
        }
    }

    /// Collect the C axes this op reads through an output→input mapping — i.e.
    /// the matmul batch axes. `rm_c_axis` only shifts indices past a removed
    /// axis; it assumes none of these is the one being removed, so a fusion that
    /// folds a C axis away must first check it is absent here.
    fn push_mapped_c_axes(&self, out: &mut TVec<usize>) {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul { geo, .. } => {
                out.extend(geo.c_to_a_axis_mapping.0.iter().map(|(c, _)| *c));
                out.extend(geo.c_to_b_axis_mapping.0.iter().map(|(c, _)| *c));
            }
            BinPerRow(_, _, map) | BinPerCol(_, _, map) | AddUnicast(_, _, map) => {
                out.extend(map.0.iter().map(|(c, _)| *c));
            }
            BinScalar(..) | Scaler(..) | AddRowColProducts(_, _) | LeakyRelu(_) | Store(..) => {}
        }
    }

    fn rm_c_axis(&mut self, axis: usize) {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul { geo, .. } => {
                geo.c_to_a_axis_mapping.rm_c_axis(axis);
                geo.c_to_b_axis_mapping.rm_c_axis(axis);
            }
            BinScalar(..) | Scaler(..) | AddRowColProducts(_, _) | LeakyRelu(_) => {}
            BinPerRow(_, _, map) | BinPerCol(_, _, map) => map.rm_c_axis(axis),
            AddUnicast(_, _, map) => {
                map.rm_c_axis(axis);
            }
            Store(oss, ..) => {
                for oss in oss {
                    match oss {
                        OutputStoreSpec::View { m_axis, n_axis, .. } => {
                            if let Some(m) = m_axis {
                                *m -= (*m > axis) as usize
                            };
                            if let Some(n) = n_axis {
                                *n -= (*n > axis) as usize
                            }
                        }
                        OutputStoreSpec::Strides { .. } => {}
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapOutputAxisToInput(pub TVec<(usize, usize)>);

impl MapOutputAxisToInput {
    #[inline]
    unsafe fn translate_view(&self, output_coords: &[usize], v: &mut TensorView) {
        for &(out_axis, in_axis) in &self.0 {
            unsafe { v.offset_axis(in_axis, output_coords[out_axis] as isize) }
        }
    }

    #[inline]
    fn rm_c_axis(&mut self, axis: usize) {
        for (c, _) in &mut self.0 {
            *c -= (*c > axis) as usize;
        }
    }

    /// Compute a flat index into a PackedMatrixStorage from output coordinates and batch strides.
    #[inline]
    pub fn flat_index(&self, output_coords: &[usize], batch_strides: &[isize]) -> usize {
        self.0
            .iter()
            .map(|&(out_axis, in_axis)| output_coords[out_axis] * batch_strides[in_axis] as usize)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddMatMulGeometry {
    pub k: TDim,
    pub c_to_a_axis_mapping: MapOutputAxisToInput,
    pub c_to_b_axis_mapping: MapOutputAxisToInput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptMatMul {
    pub c_fact: TypedFact,
    pub micro_ops: Vec<ProtoFusedSpec>,
    pub mmm: Vec<Box<dyn MatMatMul>>,
    pub mode_picker: ModePicker,
    pub c_m_axis: Option<usize>,
    pub c_n_axis: Option<usize>,
    pub trivial_packing: bool,
    pub trivial_path: bool,
}

impl Op for OptMatMul {
    fn name(&self) -> StaticName {
        "OptMatMul".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let m = self.c_m_axis.map(|ix| &self.c_fact.shape[ix]).unwrap_or(&TDim::Val(1));
        let n = self.c_n_axis.map(|ix| &self.c_fact.shape[ix]).unwrap_or(&TDim::Val(1));
        let mut infos = vec![format!(
            "c_shape:{:?}, c_m_axis:{:?} c_n_axis:{:?} m:{} n:{}",
            self.c_fact, self.c_m_axis, self.c_n_axis, m, n,
        )];
        if let Some(k) = self.guess_k() {
            infos.push(format!("Mult: m:{} k:{} n:{} with {:?}", m, k, n, self.mmm));
        } else {
            infos.push(format!("Mult: {:?}", self.mmm));
        }
        for (mode, mmm) in self.mmm.iter().enumerate() {
            infos.push(format!(
                "Ops: {}",
                self.micro_ops.iter().map(|o| o.format(&**mmm, mode)).join(" >>> ")
            ));
        }
        Ok(infos)
    }

    op_as_typed_op!();
}

/// Per-execution state for [`OptMatMul`]: on the trivial path it caches each
/// micro-op's `Store` [`OutputStore`] layout (strides fixed for the output
/// shape), so only the base pointer is refreshed per call. Constant operands
/// are baked into the op at `fuse()`, so they need no per-call state here. Pure
/// memoization: dropped on freeze and rebuilt lazily on next eval.
#[derive(Clone, Debug, Default)]
pub struct OptMatMulState {
    trivial_stores: Option<TVec<Option<OutputStore>>>,
}

impl EvalOp for OptMatMul {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<OptMatMulState>::default()))
    }
}

impl OpState for OptMatMulState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<OptMatMul>().context("OptMatMulState on non-OptMatMul op")?;
        op.eval_with_state(session, inputs, self)
    }
}

#[derive(Clone, Debug)]
struct FrozenOptMatMulState;

impl FrozenOpState for FrozenOptMatMulState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::<OptMatMulState>::default()
    }
}

impl OpStateFreeze for OptMatMulState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenOptMatMulState)
    }
}

impl OptMatMul {
    fn eval_with_state(
        &self,
        session: &TurnState,
        inputs: TVec<TValue>,
        state: &mut OptMatMulState,
    ) -> TractResult<TVec<TValue>> {
        unsafe {
            let c_shape = self.c_fact.shape.eval_to_usize(&session.resolved_symbols)?;
            let mut c = Tensor::uninitialized_dt(self.c_fact.datum_type, &c_shape)?;
            let m = self.c_m_axis.map(|c_m| c.shape()[c_m]).unwrap_or(1);
            let n = self.c_n_axis.map(|c_n| c.shape()[c_n]).unwrap_or(1);
            let mode = self.mode_picker.pick(n)?;
            let mmm = &*self.mmm[mode];
            let mut cell = session.cached_mmm_scratch_space.borrow_mut();
            if !cell.as_ref().is_some_and(|scratch| mmm.can_use_scratch_space(&**scratch)) {
                *cell = None
            }
            let scratch = cell.get_or_insert_with(|| mmm.allocate_scratch_space());
            if self.trivial_path {
                let stores = state.trivial_stores.get_or_insert_with(|| {
                    self.micro_ops
                        .iter()
                        .map(|o| match o {
                            ProtoFusedSpec::Store(oss) => Some(oss[mode].wrap(&c.view())),
                            _ => None,
                        })
                        .collect()
                });
                let uops: TVec<FusedSpec> = self
                    .micro_ops
                    .iter()
                    .zip(stores.iter())
                    .map(|(o, store)| o.resolve_trivial_cached(&inputs, &mut c, mmm, mode, *store))
                    .collect();
                mmm.run_with_scratch_space(m, n, scratch.as_mut(), &uops)?;
                Ok(tvec!(c.into_tvalue()))
            } else {
                let mut uops = vec![FusedSpec::ShiftLeft(0); self.micro_ops.len()];
                let mut looping_shape: TVec<usize> = c_shape.to_smallvec();
                if let Some(ax) = self.c_m_axis {
                    looping_shape[ax] = 1;
                }
                if let Some(ax) = self.c_n_axis {
                    looping_shape[ax] = 1;
                }
                for c_coords in indices(&*looping_shape) {
                    for ix in 0..self.micro_ops.len() {
                        *uops.get_unchecked_mut(ix) = self.micro_ops.get_unchecked(ix).resolve(
                            &inputs,
                            c_coords.slice(),
                            &c,
                            mmm,
                            mode,
                        );
                    }
                    mmm.run_with_scratch_space(m, n, scratch.as_mut(), &uops)
                        .context("In mmm.run_with_scratch_space")?;
                }
                Ok(tvec!(c.into_tvalue()))
            }
        }
    }
}

impl TypedOp for OptMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.c_m_axis.map(|ax| ax < self.c_fact.rank()).unwrap_or(true));
        ensure!(self.c_n_axis.map(|ax| ax < self.c_fact.rank()).unwrap_or(true));
        ensure!(self.trivial_path == self.can_use_trivial_path());
        ensure!(self.mmm.iter().map(|mmm| mmm.internal_type()).all_equal());
        for op in &self.micro_ops {
            op.check_inputs(inputs)?;
        }
        Ok(tvec!(self.c_fact.clone()))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mut sums = HashMap::new();
        for op in &self.micro_ops {
            for (cost, count) in op.cost(self.m(), self.n(), self.mmm[0].internal_type()) {
                *sums.entry(cost).or_default() += count;
            }
        }
        let loops = self
            .c_fact
            .shape
            .iter()
            .enumerate()
            .map(|(ix, d)| {
                if Some(ix) == self.c_m_axis || Some(ix) == self.c_n_axis {
                    1.to_dim()
                } else {
                    d.clone()
                }
            })
            .product::<TDim>();
        for s in &mut sums.values_mut() {
            *s *= &loops;
        }
        Ok(sums.into_iter().collect())
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if let Some(patch) = self.bake_const_operands(model, node)? {
            return Ok(Some(patch));
        }
        rule_if!(node.outputs.len() == 1);
        rule_if!(node.outputs[0].successors.len() == 1);
        rule_if!(!model.output_outlets()?.contains(&node.id.into()));
        let succ = model.node(node.outputs[0].successors[0].node);
        let mut patch = TypedModelPatch::new(format!("fusing {succ}"));

        if let Some(op) = succ.op_as::<ops::binary::TypedBinOp>() {
            rule_if_some!(mut binop = op.0.as_linalg_binop());
            let flipped = succ.inputs[0].node == node.id;
            if flipped {
                binop = binop.flip();
            }
            let other_outlet = succ.inputs[flipped as usize];
            return self.fuse_binary(model, node, patch, other_outlet, binop);
        }
        if let Some(op) = succ.op_as::<ops::binary::OptBinByScalar>() {
            rule_if_some!(mut binop = op.binop.as_linalg_binop());
            let flipped = succ.inputs[0].node == node.id;
            if flipped {
                binop = binop.flip();
            }
            let other_outlet = succ.inputs[flipped as usize];
            return self.fuse_binary(model, node, patch, other_outlet, binop);
        }

        if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>().map(|ew| ew.0.as_ref()) {
            if let Some(op) = op.downcast_ref::<ops::math::QScale>() {
                return self.fuse_op(
                    model,
                    node,
                    patch,
                    vec![ProtoFusedSpec::Scaler(op.scaler)],
                    &[],
                );
            }
            if let Some(op) = op.downcast_ref::<LeakyRelu>() {
                rule_if!(
                    self.mmm
                        .iter()
                        .all(|mmm| mmm.can_fuse(&FusedSpec::LeakyRelu(&tensor0(op.alpha))))
                );
                let alpha = patch.add_const(
                    node.name.to_string() + ".alpha",
                    tensor0(op.alpha).cast_to_dt(self.mmm[0].internal_type())?.into_owned(),
                )?;
                return self.fuse_op(
                    model,
                    node,
                    patch,
                    vec![ProtoFusedSpec::LeakyRelu(node.inputs.len())],
                    &[alpha],
                );
            }
        }
        if let Some(cast_to) = succ.op_as::<ops::cast::Cast>().map(|cast| cast.to)
            && (((cast_to.unquantized() == i8::datum_type()
                || cast_to.unquantized() == u8::datum_type())
                && self.c_fact.datum_type == i32::datum_type())
                || self.mmm.iter().all(|m| m.stores().contains(&cast_to)))
            && let Some(ProtoFusedSpec::Store(stores)) = self.micro_ops.last()
        {
            rule_if!(stores.iter().all(|s| !matches!(s, OutputStoreSpec::Strides { .. })));
            let c_fact = cast_to.fact(self.c_fact.shape.clone());
            let mut patch =
                TypedModelPatch::fuse_with_next(model, node, Self { c_fact, ..self.clone() })?;
            patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
            return Ok(Some(patch));
        }
        if let Some(AxisOp::Rm(axis)) = succ.op_as::<ops::AxisOp>() {
            rule_if!(Some(*axis) != self.c_m_axis);
            rule_if!(Some(*axis) != self.c_n_axis);
            let mut new_op = self.clone();
            new_op.c_fact.shape.remove_axis(*axis)?;
            if let Some(c_m_axis) = &mut new_op.c_m_axis {
                *c_m_axis -= (*c_m_axis > *axis) as usize;
            }
            if let Some(c_n_axis) = &mut new_op.c_n_axis {
                *c_n_axis -= (*c_n_axis > *axis) as usize;
            }
            for uop in &mut new_op.micro_ops {
                uop.rm_c_axis(*axis);
            }
            let mut patch = TypedModelPatch::fuse_with_next(model, node, new_op)?;
            patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
            return Ok(Some(patch));
        }
        if let Some(into) = succ.op_as::<IntoShape>()
            && let Some(new_op) = self.absorb_squeeze(into)
        {
            let mut patch = TypedModelPatch::fuse_with_next(model, node, new_op)?;
            patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
            return Ok(Some(patch));
        }
        // Reach over a shape-agnostic elementwise (Tanh/Sigmoid/Cast/…) to absorb
        // a squeeze reshape into the store: matmul → ew → squeeze becomes
        // matmul(squeezed) → ew, unchanged since the op is per-element. With the
        // direct arm above, squeeze reshapes fuse whether before or after the ew.
        if (succ.op_is::<ElementWiseOp>() || succ.op_is::<Cast>())
            && succ.outputs.len() == 1
            && let &[next] = &*succ.outputs[0].successors
        {
            let into_node = model.node(next.node);
            if let Some(into) = into_node.op_as::<IntoShape>()
                && let Some(new_op) = self.absorb_squeeze(into)
            {
                let mut patch = TypedModelPatch::default();
                let inputs = node
                    .inputs
                    .iter()
                    .map(|i| patch.tap_model(model, *i))
                    .collect::<TractResult<TVec<_>>>()?;
                let mm = patch.wire_node(&node.name, new_op, &inputs)?[0];
                let ew = patch.wire_node(&succ.name, succ.op.clone(), &[mm])?[0];
                patch.shunt_outside(model, into_node.id.into(), ew)?;
                patch.dont_apply_twice = Some(format!("Reach {into_node} into {node}"));
                return Ok(Some(patch));
            }
        }
        if (succ.op_is::<AxisOp>() || succ.op_is::<IntoShape>())
            && let &[next] = &*succ.outputs[0].successors
        {
            let next_node = model.node(next.node);
            if let Some(cast) = next_node.op_as::<Cast>() {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.id.into())?;
                wire = patch.wire_node(&next_node.name, cast.clone(), &[wire])?[0];
                wire = patch.wire_node(&succ.name, succ.op.clone(), &[wire])?[0];
                patch.shunt_outside(model, next_node.id.into(), wire)?;
                return Ok(Some(patch));
            } else {
                // matmul -> reshape -> elementwise(broadcast operand): reorder into
                // matmul -> elementwise -> reshape so the bias/activation can fuse
                // into the matmul epilogue on a later pass. Match the generic
                // TypedBinOp as well as its already-codegen'd OptBin* forms.
                let rewire_op = next_node
                    .op_as::<ops::binary::TypedBinOp>()
                    .and_then(|op| op.0.as_linalg_binop().map(|_| op.clone()))
                    .or_else(|| {
                        next_node.op_as::<ops::binary::OptBinByScalar>().and_then(|op| {
                            op.binop
                                .as_linalg_binop()
                                .map(|_| ops::binary::TypedBinOp(op.binop.clone(), None))
                        })
                    })
                    .or_else(|| {
                        next_node.op_as::<ops::binary::OptBinUnicast>().and_then(|op| {
                            op.binop
                                .as_linalg_binop()
                                .map(|_| ops::binary::TypedBinOp(op.binop.clone(), None))
                        })
                    });
                if let Some(rewire_op) = rewire_op {
                    // The op has two inputs: the reshaped matmul output (data) and a
                    // broadcast operand. Move the operand in front of the reshape.
                    let data_slot = (next_node.inputs[1].node == succ.id) as usize;
                    let other_outlet = next_node.inputs[1 - data_slot];
                    let other_fact = model.outlet_fact(other_outlet)?;
                    let mut patch = TypedModelPatch::default();
                    let operand: Option<OutletId> = if let Some(uni) = &other_fact.uniform {
                        // Uniform value: rank-broadcast a scalar const (fuses as BinScalar).
                        Some(patch.add_const(&model.node(other_outlet.node).name, uni.clone())?)
                    } else if let (Some(konst), Some(nd_shape), Some(into)) = (
                        other_fact.konst.clone(),
                        other_fact.shape.as_concrete(),
                        succ.op_as::<IntoShape>(),
                    ) {
                        // Per-channel constant: track its single non-unary axis back
                        // through the reshape to the matmul output axis, and only
                        // reorder when that lands on m or n (where fuse_binary can
                        // turn it into a BinPerRow / BinPerCol epilogue).
                        use crate::ops::change_axes::InOut;
                        let nonunary: TVec<usize> = nd_shape
                            .iter()
                            .enumerate()
                            .filter(|(_, d)| **d != 1)
                            .map(|(i, _)| i)
                            .collect();
                        let mm_ax = if nonunary.len() == 1 {
                            into.mapping.track_axis((InOut::Out(0), nonunary[0]), InOut::In(0))?
                        } else {
                            None
                        };
                        match mm_ax {
                            Some(ax) if Some(ax) == self.c_m_axis || Some(ax) == self.c_n_axis => {
                                let mut target = tvec![1usize; self.c_fact.rank()];
                                target[ax] = nd_shape[nonunary[0]];
                                let reshaped = konst.as_ref().clone().into_shape(&target)?;
                                Some(patch.add_const(format!("{}.fused-operand", node.name), reshaped)?)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some(operand) = operand {
                        let output = patch.tap_model(model, node.id.into())?;
                        let wire = wire_with_rank_broadcast(
                            &next_node.name,
                            &mut patch,
                            rewire_op,
                            &if data_slot == 0 { [output, operand] } else { [operand, output] },
                        )?;
                        let wire = patch.wire_node(&succ.name, succ.op.clone(), &wire)?[0];
                        patch.shunt_outside(model, next_node.id.into(), wire)?;
                        return Ok(Some(patch));
                    }
                }
            }
        }
        if let Some(op) = succ.op_as::<ops::binary::OptBinUnicast>() {
            let in_1_fact = model.outlet_fact(succ.inputs[0])?;
            let in_2_fact = model.outlet_fact(succ.inputs[1])?;
            if op.binop.is::<ops::math::Add>()
                && self.mmm.len() == 1
                && in_1_fact.without_value() == in_2_fact.without_value()
            {
                let other_slot = 1 - node.outputs[0].successors[0].slot;
                let other_input = succ.inputs[other_slot];
                let other_input = patch.tap_model(model, other_input)?;
                let other_fact = patch.outlet_fact(other_input)?;

                if other_fact.shape == self.c_fact.shape {
                    let other_storage = unsafe { self.mmm[0].c_view(self.c_m_axis, self.c_n_axis) };
                    let mapping =
                        MapOutputAxisToInput((0..other_fact.rank()).map(|x| (x, x)).collect());
                    return self.fuse_op(
                        model,
                        node,
                        patch,
                        vec![ProtoFusedSpec::AddUnicast(other_storage, node.inputs.len(), mapping)],
                        &[other_input],
                    );
                }
            } else {
                rule_if_some!(mut binop = op.binop.as_linalg_binop());
                let flipped = succ.inputs[0].node == node.id;
                if flipped {
                    binop = binop.flip();
                }
                let other_outlet = succ.inputs[flipped as usize];
                return self.fuse_binary(model, node, patch, other_outlet, binop);
            }
        };
        Ok(None)
    }

    as_op!();
}

impl OptMatMul {
    pub fn new(
        mmm: Vec<Box<dyn MatMatMul>>,
        mode_picker: ModePicker,
        c_fact: TypedFact,
        c_m_axis: Option<usize>,
        c_n_axis: Option<usize>,
        micro_ops: Vec<ProtoFusedSpec>,
        trivial_packing: bool,
    ) -> TractResult<Self> {
        if let Some(m) = c_m_axis {
            ensure!(m < c_fact.rank());
        }
        if let Some(n) = c_n_axis {
            ensure!(n < c_fact.rank());
        }
        let mut it = OptMatMul {
            mmm,
            mode_picker,
            c_fact,
            c_m_axis,
            c_n_axis,
            micro_ops,
            trivial_path: false,
            trivial_packing,
        };
        it.update_trivial_path();
        Ok(it)
    }

    // for auditing only (may return None if no AddMatMul is found)
    pub fn guess_k(&self) -> Option<TDim> {
        self.micro_ops
            .iter()
            .find_map(
                |o| {
                    if let ProtoFusedSpec::AddMatMul { geo, .. } = o { Some(geo) } else { None }
                },
            )
            .map(|geo| geo.k.clone())
    }

    #[inline]
    pub fn m(&self) -> &TDim {
        self.c_m_axis.map(|ax| &self.c_fact.shape[ax]).unwrap_or(&TDim::Val(1))
    }

    #[inline]
    pub fn n(&self) -> &TDim {
        self.c_n_axis.map(|ax| &self.c_fact.shape[ax]).unwrap_or(&TDim::Val(1))
    }

    fn update_trivial_path(&mut self) {
        self.trivial_path = self.can_use_trivial_path();
    }

    /// If `into` is a pure unit-axis squeeze of this op's (concrete) output that
    /// leaves the m/n axes intact, return a clone whose store produces the
    /// squeezed shape directly. `None` when the reshape can't be absorbed.
    fn absorb_squeeze(&self, into: &IntoShape) -> Option<Self> {
        if into.strides != Tensor::natural_strides(&into.dims) {
            return None;
        }
        let old = self.c_fact.shape.as_concrete()?;
        let removed = pure_squeeze_removed(old, &into.dims)?;
        if removed.iter().any(|ax| Some(*ax) == self.c_m_axis || Some(*ax) == self.c_n_axis) {
            return None;
        }
        // A non-unit matmul batch axis (e.g. grouped conv) makes the packed
        // inputs per-batch; folding any axis then desyncs that batch indexing.
        // Only fuse when every batch axis is trivial (size 1).
        let mut batch_axes: TVec<usize> = tvec!();
        self.micro_ops.iter().for_each(|uop| uop.push_mapped_c_axes(&mut batch_axes));
        if batch_axes.iter().any(|ax| old.get(*ax).copied().unwrap_or(1) > 1) {
            return None;
        }
        let mut new_op = self.clone();
        for axis in removed.iter().rev() {
            new_op.c_fact.shape.remove_axis(*axis).ok()?;
            if let Some(c_m_axis) = &mut new_op.c_m_axis {
                *c_m_axis -= (*c_m_axis > *axis) as usize;
            }
            if let Some(c_n_axis) = &mut new_op.c_n_axis {
                *c_n_axis -= (*c_n_axis > *axis) as usize;
            }
            for uop in &mut new_op.micro_ops {
                uop.rm_c_axis(*axis);
            }
        }
        Some(new_op)
    }

    fn can_use_trivial_path(&self) -> bool {
        self.c_fact.shape.is_concrete()
            && self.c_fact.shape.iter().enumerate().all(|(ax, dim)| {
                Some(ax) == self.c_m_axis || Some(ax) == self.c_n_axis || dim.is_one()
            })
            && self.trivial_packing
            && self.micro_ops.iter().all(|o| o.is_trivial())
    }

    /// Bake matmul operands that are fed by a constant, non-batched input into
    /// the op as [`MatMulOperand::Const`], dropping the corresponding graph
    /// input. Runs once the packing has const-folded so the operand's outlet
    /// carries a packed `konst`. Returns a patch that rewires the node with the
    /// remaining inputs, or `None` if nothing is bakeable.
    fn bake_const_operands(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let bakeable = |operand: &MatMulOperand, mapping: &MapOutputAxisToInput| -> bool {
            if let MatMulOperand::Input(i) = operand {
                mapping.0.is_empty()
                    && model.outlet_fact(node.inputs[*i]).is_ok_and(|f| {
                        f.konst
                            .as_ref()
                            .and_then(|k| k.try_storage_as::<PackedMatrixStorage>().ok())
                            .is_some()
                    })
            } else {
                false
            }
        };
        let mut baked: TVec<usize> = tvec!();
        for op in &self.micro_ops {
            if let ProtoFusedSpec::AddMatMul { geo, a, b, .. } = op {
                if bakeable(a, &geo.c_to_a_axis_mapping) {
                    let MatMulOperand::Input(i) = a else { unreachable!() };
                    baked.push(*i);
                }
                if bakeable(b, &geo.c_to_b_axis_mapping) {
                    let MatMulOperand::Input(i) = b else { unreachable!() };
                    baked.push(*i);
                }
            }
        }
        if baked.is_empty() {
            return Ok(None);
        }
        baked.sort();
        baked.dedup();
        let remap: Vec<Option<usize>> = {
            let mut ni = 0;
            (0..node.inputs.len())
                .map(|i| {
                    (!baked.contains(&i)).then(|| {
                        let cur = ni;
                        ni += 1;
                        cur
                    })
                })
                .collect()
        };
        let const_value = |i: usize| -> TractResult<Box<dyn MMMInputValue>> {
            let konst = model.outlet_fact(node.inputs[i])?.konst.clone().unwrap();
            Ok(dyn_clone::clone_box(konst.try_storage_as::<PackedMatrixStorage>()?.value()))
        };
        let map_operand = |operand: &MatMulOperand| -> TractResult<MatMulOperand> {
            Ok(match operand {
                MatMulOperand::Input(i) if baked.contains(i) => {
                    MatMulOperand::Const(const_value(*i)?)
                }
                MatMulOperand::Input(i) => MatMulOperand::Input(remap[*i].unwrap()),
                MatMulOperand::Const(v) => MatMulOperand::Const(v.clone()),
            })
        };
        let micro_ops = self
            .micro_ops
            .iter()
            .map(|op| -> TractResult<ProtoFusedSpec> {
                use ProtoFusedSpec::*;
                Ok(match op {
                    AddMatMul { geo, a, b, packings } => AddMatMul {
                        geo: geo.clone(),
                        a: map_operand(a)?,
                        b: map_operand(b)?,
                        packings: packings.clone(),
                    },
                    BinScalar(v, op) => BinScalar(remap[*v].unwrap(), *op),
                    LeakyRelu(v) => LeakyRelu(remap[*v].unwrap()),
                    BinPerRow(v, op, m) => BinPerRow(remap[*v].unwrap(), *op, m.clone()),
                    BinPerCol(v, op, m) => BinPerCol(remap[*v].unwrap(), *op, m.clone()),
                    AddRowColProducts(r, c) => {
                        AddRowColProducts(remap[*r].unwrap(), remap[*c].unwrap())
                    }
                    AddUnicast(s, v, m) => AddUnicast(*s, remap[*v].unwrap(), m.clone()),
                    Scaler(s) => Scaler(*s),
                    Store(o) => Store(o.clone()),
                })
            })
            .collect::<TractResult<Vec<_>>>()?;
        let new_op = OptMatMul { micro_ops, ..self.clone() };
        let kept: TVec<OutletId> =
            (0..node.inputs.len()).filter(|i| !baked.contains(i)).map(|i| node.inputs[i]).collect();
        let mut patch = TypedModelPatch::new(format!("bake const operands into {}", node.name));
        let taps = patch.taps(model, &kept)?;
        let output = patch.wire_node(&node.name, new_op, &taps)?;
        patch.shunt_outside(model, node.id.into(), output[0])?;
        Ok(Some(patch))
    }

    fn fuse_op(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        mut patch: TypedModelPatch,
        fused_micro_op: Vec<ProtoFusedSpec>,
        additional_inputs: &[OutletId],
    ) -> TractResult<Option<TypedModelPatch>> {
        let succ = model.node(node.outputs[0].successors[0].node);
        let mut new_op = self.clone();
        let before_last = new_op.micro_ops.len() - 1..new_op.micro_ops.len() - 1;
        new_op.micro_ops.splice(before_last, fused_micro_op);
        new_op.c_fact = succ.outputs[0].fact.clone();
        new_op.update_trivial_path();
        let mut inputs = patch.taps(model, &node.inputs)?;
        inputs.extend(additional_inputs.iter().cloned());
        let output = patch.wire_node(&succ.name, new_op, &inputs)?;
        patch.shunt_outside(model, succ.id.into(), output[0])?;
        Ok(Some(patch))
    }

    fn fuse_binary(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        mut patch: TypedModelPatch,
        value: OutletId,
        binop: BinOp,
    ) -> TractResult<Option<TypedModelPatch>> {
        let fact = model.outlet_fact(value)?;
        let mut v = patch.tap_model(model, value)?;
        if fact.datum_type != self.mmm[0].internal_type() {
            v = patch.wire_node(
                format!("{}.cast-input-{}", node.name, node.inputs.len()),
                cast(self.mmm[0].internal_type()),
                &[v],
            )?[0];
        }
        let value = node.inputs.len();
        let additional_input = tvec!(v);
        if fact.shape.volume() == 1.to_dim() {
            return self.fuse_op(
                model,
                node,
                patch,
                vec![ProtoFusedSpec::BinScalar(value, binop)],
                &additional_input,
            );
        }
        let other_shape = fact.shape.to_owned();
        if self.c_m_axis.is_some_and(|ax| {
            other_shape[ax] == self.c_fact.shape[ax] && other_shape[ax] == other_shape.volume()
        }) {
            return self.fuse_op(
                model,
                node,
                patch,
                vec![ProtoFusedSpec::BinPerRow(
                    value,
                    binop,
                    MapOutputAxisToInput(tvec!((self.c_m_axis.unwrap(), self.c_m_axis.unwrap()))),
                )],
                &additional_input,
            );
        }
        if self.c_n_axis.is_some_and(|ax| {
            other_shape[ax] == self.c_fact.shape[ax] && other_shape[ax] == other_shape.volume()
        }) {
            return self.fuse_op(
                model,
                node,
                patch,
                vec![ProtoFusedSpec::BinPerCol(
                    value,
                    binop,
                    MapOutputAxisToInput(tvec!((self.c_n_axis.unwrap(), self.c_n_axis.unwrap()))),
                )],
                &additional_input,
            );
        }
        Ok(None)
    }
}
