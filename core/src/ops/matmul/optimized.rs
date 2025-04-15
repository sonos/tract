use crate::internal::*;
use crate::ops::cast::cast;
use crate::ops::change_axes::wire_with_rank_broadcast;
use crate::ops::nn::LeakyRelu;
use ndarray::*;
use tract_itertools::Itertools;

use tract_linalg::mmm::{
    AsInputValue, EagerPackedInput, FusedSpec, MMMInputValue, MatMatMul, OutputStoreSpec,
    PanelExtractInput, PanelExtractor,
};
use tract_linalg::pack::PackedFormat;
use tract_linalg::{BinOp, Scaler};
use tract_smallvec::ToSmallVec;

use super::ModePicker;

#[derive(Clone, Debug)]
pub enum ProtoFusedSpec {
    AddMatMul {
        geo: AddMatMulGeometry,
        a: usize,
        b: usize,
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
        let fs = match self {
            ProtoFusedSpec::AddMatMul { geo, a, b, packings } => {
                let mut a = inputs[*a].view();
                let mut b = inputs[*b].view();
                unsafe {
                    geo.c_to_a_axis_mapping.translate_view(output_coords, &mut a);
                }
                let a = a.as_slice::<Opaque>().unwrap()[0]
                    .downcast_ref::<Box<dyn MMMInputValue>>()
                    .unwrap();
                unsafe {
                    geo.c_to_b_axis_mapping.translate_view(output_coords, &mut b);
                }
                let b = b.as_slice::<Opaque>().unwrap()[0]
                    .downcast_ref::<Box<dyn MMMInputValue>>()
                    .unwrap();
                let (_a_packing, b_packing) = &mmm.packings()[packings[mode].0];
                let pa = if let Some(extractor) = &packings[mode].1 {
                    let data = a.downcast_ref::<EagerPackedInput>().unwrap();
                    AsInputValue::Owned(Box::new(PanelExtractInput {
                        format: extractor.clone(),
                        data: data.clone(),
                    }))
                } else {
                    AsInputValue::Borrowed(&**a)
                };
                assert!(
                    b_packing.same_as(b.format())
                        || (b_packing.is::<PackedFormat>() && b_packing.r() == b.format().r())
                );
                debug_assert!(pa.k().to_dim().compatible_with(&geo.k.to_dim()));
                debug_assert!(b.k().to_dim().compatible_with(&geo.k.to_dim()));
                FusedSpec::AddMatMul {
                    a: pa,
                    b: AsInputValue::Borrowed(&**b),
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
        let fs = match self {
            ProtoFusedSpec::AddMatMul { a, b, packings, .. } => unsafe {
                debug_assert!(inputs.get(*a).is_some());
                debug_assert!(inputs.get(*b).is_some());
                let a = inputs.get_unchecked(*a);
                let b = inputs.get_unchecked(*b);
                debug_assert!(a.datum_type().is_opaque());
                debug_assert!(a.len() == 1);
                debug_assert!(b.datum_type().is_opaque());
                debug_assert!(b.len() == 1);
                let a = a.as_slice_unchecked::<Opaque>().get_unchecked(0);
                let b = b.as_slice_unchecked::<Opaque>().get_unchecked(0);
                debug_assert!(a.is::<Box<dyn MMMInputValue>>());
                debug_assert!(b.is::<Box<dyn MMMInputValue>>());
                let a = a.downcast_ref::<Box<dyn MMMInputValue>>().unwrap_unchecked();
                let b = b.downcast_ref::<Box<dyn MMMInputValue>>().unwrap_unchecked();
                debug_assert!(packings.len() == 1);
                debug_assert!(packings[0].1.is_none()); // no panel extraction
                #[cfg(debug_assertions)]
                {
                    let (a_packing, b_packing) = &_mmm.packings()[packings[mode].0];
                    debug_assert!(
                        a_packing.same_as(a.format())
                            || (a_packing.is::<PackedFormat>() && a_packing.r() == a.format().r())
                    );
                    debug_assert!(
                        b_packing.same_as(b.format())
                            || (b_packing.is::<PackedFormat>() && b_packing.r() == b.format().r())
                    );
                }
                FusedSpec::AddMatMul {
                    a: AsInputValue::Borrowed(&**a),
                    b: AsInputValue::Borrowed(&**b),
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

    fn check_inputs(&self, inputs: &[&TypedFact]) -> TractResult<()> {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul { a, b, .. } => {
                ensure!(inputs[*a].datum_type == Opaque::datum_type());
                ensure!(inputs[*b].datum_type == Opaque::datum_type());
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

#[derive(Clone, Debug)]
pub struct MapOutputAxisToInput(pub TVec<(usize, usize)>);

impl MapOutputAxisToInput {
    #[inline]
    unsafe fn translate_view(&self, output_coords: &[usize], v: &mut TensorView) {
        for &(out_axis, in_axis) in &self.0 {
            v.offset_axis(in_axis, output_coords[out_axis] as isize)
        }
    }

    #[inline]
    fn rm_c_axis(&mut self, axis: usize) {
        for (c, _) in &mut self.0 {
            *c -= (*c > axis) as usize;
        }
    }
}

#[derive(Clone, Debug)]
pub struct AddMatMulGeometry {
    pub k: TDim,
    pub c_to_a_axis_mapping: MapOutputAxisToInput,
    pub c_to_b_axis_mapping: MapOutputAxisToInput,
}

#[derive(Clone, Debug)]
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
    fn name(&self) -> Cow<str> {
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

impl EvalOp for OptMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
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
                let uops: Vec<FusedSpec> = self
                    .micro_ops
                    .iter()
                    .map(|o| o.resolve_trivial(&inputs, &mut c, mmm, mode))
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
        if node.outputs.len() != 1
            || node.outputs[0].successors.len() != 1
            || model.output_outlets()?.contains(&node.id.into())
        {
            return Ok(None);
        }
        let succ = model.node(node.outputs[0].successors[0].node);
        let mut patch = TypedModelPatch::new(format!("fusing {succ}"));

        if let Some(op) = succ.op_as::<ops::binary::TypedBinOp>() {
            let mut binop = if let Some(op) = op.0.as_linalg_binop() {
                op
            } else {
                return Ok(None);
            };
            let flipped = succ.inputs[0].node == node.id;
            if flipped {
                binop = binop.flip();
            }
            let other_outlet = succ.inputs[flipped as usize];
            return self.fuse_binary(model, node, patch, other_outlet, binop);
        }
        if let Some(op) = succ.op_as::<ops::binary::OptBinByScalar>() {
            let mut binop = if let Some(op) = op.binop.as_linalg_binop() {
                op
            } else {
                return Ok(None);
            };
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
                if !self
                    .mmm
                    .iter()
                    .all(|mmm| mmm.can_fuse(&FusedSpec::LeakyRelu(&tensor0(op.alpha))))
                {
                    return Ok(None);
                }
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
        if let Some(cast_to) = succ.op_as::<ops::cast::Cast>().map(|cast| cast.to) {
            if (cast_to.unquantized() == i8::datum_type()
                || cast_to.unquantized() == u8::datum_type())
                && self.c_fact.datum_type == i32::datum_type()
            {
                if let Some(ProtoFusedSpec::Store(stores)) = self.micro_ops.last() {
                    if stores.iter().any(|s| matches!(s, OutputStoreSpec::Strides { .. })) {
                        return Ok(None);
                    }
                    let c_fact = cast_to.fact(self.c_fact.shape.clone());
                    let mut patch = TypedModelPatch::fuse_with_next(
                        model,
                        node,
                        Self { c_fact, ..self.clone() },
                    )?;
                    patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
                    return Ok(Some(patch));
                }
            }
        }
        if let Some(AxisOp::Rm(axis)) = succ.op_as::<ops::AxisOp>() {
            if Some(*axis) == self.c_m_axis || Some(*axis) == self.c_n_axis {
                return Ok(None);
            }
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
        if succ.op_is::<AxisOp>() {
            if let &[next] = &*succ.outputs[0].successors {
                let bin = model.node(next.node);
                if let Some(op) = bin.op_as::<ops::binary::TypedBinOp>() {
                    if op.0.as_linalg_binop().is_none() {
                        return Ok(None);
                    };
                    let flipped = succ.inputs[0].node == node.id;
                    let other_outlet = bin.inputs[flipped as usize];
                    if let Some(uni) = &model.outlet_fact(other_outlet)?.uniform {
                        let mut patch = TypedModelPatch::default();
                        let cst =
                            patch.add_const(&model.node(other_outlet.node).name, uni.clone())?;
                        let output = patch.tap_model(model, node.id.into())?;
                        let wire = wire_with_rank_broadcast(
                            &bin.name,
                            &mut patch,
                            op.clone(),
                            &if flipped { [output, cst] } else { [cst, output] },
                        )?;
                        let wire = patch.wire_node(&succ.name, succ.op.clone(), &wire)?[0];
                        patch.shunt_outside(model, bin.id.into(), wire)?;
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
                let mut binop = if let Some(op) = op.binop.as_linalg_binop() {
                    op
                } else {
                    return Ok(None);
                };
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
                    if let ProtoFusedSpec::AddMatMul { geo, .. } = o {
                        Some(geo)
                    } else {
                        None
                    }
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

    fn can_use_trivial_path(&self) -> bool {
        self.c_fact.shape.is_concrete()
            && self.c_fact.shape.iter().enumerate().all(|(ax, dim)| {
                Some(ax) == self.c_m_axis || Some(ax) == self.c_n_axis || dim.is_one()
            })
            && self.trivial_packing
            && self.micro_ops.iter().all(|o| o.is_trivial())
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
