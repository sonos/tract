use crate::internal::*;
use crate::ops::binary::wire_with_rank_broadcast;
use crate::ops::cast::cast;
use crate::ops::nn::LeakyRelu;
use ndarray::*;
use tract_itertools::Itertools;

use tract_linalg::mmm::{
    BinOp, FusedSpec, InputStoreSpec, MatMatMul, OutputStoreSpec, ScratchSpace, VirtualInputSpec,
};
use tract_linalg::Scaler;
use tract_smallvec::ToSmallVec;

#[derive(Clone, Debug)]
pub enum ProtoInputStoreSpec {
    Packed { item_size: usize },
    Virtual { func: Box<dyn VirtualInputSpec> },
}

#[derive(Clone, Debug)]
pub enum ProtoFusedSpec {
    AddMatMul(AddMatMulGeometry, usize, usize),
    BinScalar(usize, BinOp),
    LeakyRelu(usize),
    BinPerRow(usize, BinOp, MapOutputAxisToInput),
    BinPerCol(usize, BinOp, MapOutputAxisToInput),
    AddRowColProducts(usize, usize),
    AddUnicast(OutputStoreSpec, usize, MapOutputAxisToInput),
    Scaler(Scaler),
    Store(OutputStoreSpec),
}

impl ProtoFusedSpec {
    pub fn name(&self) -> String {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul(geo, _, _) => format!("matmul(k={})", geo.k),
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
        symbols: &SymbolValues,
        output: &Tensor,
    ) -> FusedSpec<'t> {
        let fs = match self {
            ProtoFusedSpec::AddMatMul(geo, a, b) => {
                let mut a = inputs[*a].view();
                unsafe {
                    geo.c_to_a_axis_mapping.translate_view(output_coords, &mut a);
                }
                let mut b = inputs[*b].view();
                unsafe {
                    geo.c_to_b_axis_mapping.translate_view(output_coords, &mut b);
                }
                let k = geo.k.eval(symbols).to_usize().unwrap();
                unsafe {
                    // careful here. this work because a_packed() return a packer from which
                    // nothing is borrowed
                    let a = if let Some(sto) = &geo.a_storage {
                        sto.wrap(&a)
                    } else {
                        geo.mmm.a_packed(a.datum_type().size_of(), k).wrap(&a)
                    };
                    let b = if let Some(sto) = &geo.b_storage {
                        sto.wrap(&b)
                    } else {
                        geo.mmm.b_packed(b.datum_type().size_of(), k).wrap(&b)
                    };
                    FusedSpec::AddMatMul { k, a, b }
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
                FusedSpec::Store(oss.wrap(&view))
            },
        };
        fs
    }

    pub fn has_symbols(&self) -> bool {
        match self {
            ProtoFusedSpec::AddMatMul(geo, _, _) => geo.k.as_i64().is_none(),
            _ => false,
        }
    }

    pub fn resolve_trivial<'t>(
        &'t self,
        inputs: &'t [TValue],
        output: &mut Tensor,
    ) -> FusedSpec<'t> {
        let fs = match self {
            ProtoFusedSpec::AddMatMul(geo, a, b) => {
                let a = inputs[*a].view();
                let b = inputs[*b].view();
                unsafe {
                    let k = geo.k.as_i64().unwrap_unchecked() as usize;
                    // careful here. this work because a_packed() return a packer from which
                    // nothing is borrowed
                    let a = if let Some(sto) = &geo.a_storage {
                        sto.wrap(&a)
                    } else {
                        geo.mmm.a_packed(a.datum_type().size_of(), k).wrap(&a)
                    };
                    let b = if let Some(sto) = &geo.b_storage {
                        sto.wrap(&b)
                    } else {
                        geo.mmm.b_packed(b.datum_type().size_of(), k).wrap(&b)
                    };
                    FusedSpec::AddMatMul { k, a, b }
                }
            }
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
            ProtoFusedSpec::Store(oss) => unsafe { FusedSpec::Store(oss.wrap(&output.view_mut())) },
        };
        fs
    }

    fn cost(&self, m: &TDim, n: &TDim, idt: DatumType) -> TVec<(Cost, TDim)> {
        match self {
            ProtoFusedSpec::AddMatMul(geo, _, _) => {
                tvec!((Cost::FMA(idt), m.clone() * n * &geo.k))
            }
            _ => tvec!(), /* FIXME maybe */
        }
    }

    fn rm_c_axis(&mut self, axis: usize) {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul(geo, _a, _b) => {
                geo.c_to_a_axis_mapping.rm_c_axis(axis);
                geo.c_to_b_axis_mapping.rm_c_axis(axis);
            }
            BinScalar(..) | Scaler(..) | AddRowColProducts(_, _) | LeakyRelu(_) => {}
            BinPerRow(_, _, map) | BinPerCol(_, _, map) => map.rm_c_axis(axis),
            AddUnicast(_, _, map) => {
                map.rm_c_axis(axis);
            }
            Store(oss, ..) => match oss {
                OutputStoreSpec::View { m_axis, n_axis, .. } => {
                    *m_axis -= (*m_axis > axis) as usize;
                    *n_axis -= (*n_axis > axis) as usize;
                }
                OutputStoreSpec::Strides { .. } => {}
            },
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
    pub a_storage: Option<InputStoreSpec>,
    pub b_storage: Option<InputStoreSpec>,
    pub mmm: Box<dyn MatMatMul>,
    pub c_to_a_axis_mapping: MapOutputAxisToInput,
    pub c_to_b_axis_mapping: MapOutputAxisToInput,
}

#[derive(Clone, Debug, Hash)]
pub struct ConcreteMatrixGeometry {
    pub m: usize,
    pub n: usize,
}

#[derive(Clone, Debug, Hash)]
pub struct SymbolicMatrixGeometry {
    pub m: TDim,
    pub n: TDim,
    pub mmm: Box<dyn MatMatMul>,
}

impl ResolveTo<ConcreteMatrixGeometry> for SymbolicMatrixGeometry {
    type Param = SymbolValues;
    fn resolve(&self, param: &Self::Param) -> TractResult<ConcreteMatrixGeometry> {
        let m = self.m.eval(param).to_usize()?;
        let n = self.n.eval(param).to_usize()?;
        //        let b_storage = unsafe { self.mmm.b_packed(self.b_datum_type.size_of(), k) };
        Ok(ConcreteMatrixGeometry { m, n })
    }
}

pub type MatrixGeometry = GeometryBound<SymbolicMatrixGeometry, ConcreteMatrixGeometry>;

#[derive(Clone, Debug)]
pub struct LirMatMulUnary {
    pub c_fact: TypedFact,
    pub micro_ops: Vec<ProtoFusedSpec>,
    pub geometry: MatrixGeometry,
    pub mmm: Box<dyn MatMatMul>,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
    pub trivial_path: bool,
}

impl Op for LirMatMulUnary {
    fn name(&self) -> Cow<str> {
        "LirMatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut infos = vec![format!(
            "c_shape:{:?}, c_m_axis:{} c_n_axis:{} geometry:{:?}",
            self.c_fact, self.c_m_axis, self.c_n_axis, self.geometry,
        )];
        let (m, n) = self.m_n();
        if let Some(k) = self.guess_k() {
            infos.push(format!("Mult: m:{} k:{} n:{} with {}", m, k, n, self.mmm));
        } else {
            infos.push(format!("Mult: {}", self.mmm));
        }
        infos.push(format!("Ops: {}", self.micro_ops.iter().map(|o| o.name()).join(" >>> ")));
        Ok(infos)
    }

    op_as_typed_op!();
}

#[derive(Clone, Debug)]
struct State;
trivial_op_state_freeeze!(State);

impl OpState for State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<LirMatMulUnary>().unwrap();
        unsafe {
            if session
                .cached_mmm_scratch_space
                .as_deref()
                .map(|scratch| op.mmm.can_use_scratch_space(scratch))
                == Some(false)
            {
                session.cached_mmm_scratch_space = None
            }
            let scratch = session
                .cached_mmm_scratch_space
                .get_or_insert_with(|| op.mmm.allocate_scratch_space());
            eval(op, &session.resolved_symbols, scratch.as_mut(), &inputs)
        }
    }
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
        self.geometry.is_concrete()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State)))
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut scratch = unsafe { self.mmm.allocate_scratch_space() };
        eval(self, &Default::default(), scratch.as_mut(), &inputs)
    }
}

#[allow(clippy::too_many_arguments)]
fn eval(
    op: &LirMatMulUnary,
    symbols: &SymbolValues,
    scratch: &mut dyn ScratchSpace,
    inputs: &[TValue],
) -> TractResult<TVec<TValue>> {
    unsafe {
        if op.trivial_path {
            let c_shape = op.c_fact.shape.as_concrete().unwrap_unchecked();
            let geometry = op.geometry.as_concrete().unwrap_unchecked();
            let mut c = Tensor::uninitialized_dt(op.c_fact.datum_type, c_shape)?;
            let uops: Vec<FusedSpec> =
                op.micro_ops.iter().map(|o| o.resolve_trivial(inputs, &mut c)).collect();
            op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &uops)?;
            Ok(tvec!(c.into_tvalue()))
        } else {
            let geometry = op.geometry.to_concrete(symbols)?;
            let c_shape = op.c_fact.shape.eval_to_usize(symbols)?;
            let c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
            let mut uops = vec![FusedSpec::ShiftLeft(0); op.micro_ops.len()];
            let mut looping_shape: TVec<usize> = c_shape.to_smallvec();
            looping_shape[op.c_m_axis] = 1;
            looping_shape[op.c_n_axis] = 1;
            for c_coords in indices(&*looping_shape) {
                for ix in 0..op.micro_ops.len() {
                    *uops.get_unchecked_mut(ix) = op.micro_ops.get_unchecked(ix).resolve(
                        inputs,
                        c_coords.slice(),
                        symbols,
                        &c,
                    );
                }
                op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &uops)?;
            }
            Ok(tvec!(c.into_tvalue()))
        }
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.c_m_axis < self.c_fact.rank());
        ensure!(self.c_n_axis < self.c_fact.rank());
        ensure!(self.trivial_path == self.can_use_trivial_path());
        Ok(tvec!(self.c_fact.clone()))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mut sums = HashMap::new();
        let (m, n) = self.m_n();
        for op in &self.micro_ops {
            for (cost, count) in op.cost(&m, &n, self.mmm.internal_type()) {
                *sums.entry(cost).or_default() += count;
            }
        }
        let loops = self
            .c_fact
            .shape
            .iter()
            .enumerate()
            .map(|(ix, d)| if ix == self.c_m_axis || ix == self.c_n_axis { 1.to_dim() } else { d })
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
            let mut binop =
                if let Some(op) = op.0.as_linalg_binop() { op } else { return Ok(None) };
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
                if !self.mmm.can_fuse(&FusedSpec::LeakyRelu(&tensor0(op.alpha))) {
                    return Ok(None)
                }
                let alpha = patch.add_const(
                    node.name.to_string() + ".alpha",
                    tensor0(op.alpha).cast_to_dt(self.mmm.internal_type())?.into_owned(),
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
                if let Some(ProtoFusedSpec::Store(OutputStoreSpec::View { .. })) =
                    self.micro_ops.last()
                {
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
            if *axis == self.c_m_axis || *axis == self.c_n_axis {
                return Ok(None);
            }
            let mut new_op = self.clone();
            new_op.c_fact.shape.remove_axis(*axis)?;
            new_op.c_m_axis -= (new_op.c_m_axis > *axis) as usize;
            new_op.c_n_axis -= (new_op.c_n_axis > *axis) as usize;
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
        if let Some(op) = succ.op_as::<ops::binary::MergeOpUnicast>() {
            if op.0.is::<ops::math::Add>() {
                let other_slot = 1 - node.outputs[0].successors[0].slot;
                let other_input = succ.inputs[other_slot];
                let other_input = patch.tap_model(model, other_input)?;
                let other_fact = patch.outlet_fact(other_input)?;

                if other_fact.shape == self.c_fact.shape {
                    let other_storage = unsafe { self.mmm.c_view(self.c_m_axis, self.c_n_axis) };
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
            }
        };
        Ok(None)
    }

    as_op!();
}

impl LirMatMulUnary {
    pub fn new(
        mmm: Box<dyn MatMatMul>,
        c_fact: TypedFact,
        c_m_axis: usize,
        c_n_axis: usize,
        micro_ops: Vec<ProtoFusedSpec>,
    ) -> TractResult<Self> {
        ensure!(c_m_axis < c_fact.rank());
        ensure!(c_n_axis < c_fact.rank());
        let geometry = MatrixGeometry::from(SymbolicMatrixGeometry {
            mmm: mmm.clone(),
            m: c_fact.shape[c_m_axis].clone(),
            n: c_fact.shape[c_n_axis].clone(),
        });
        let geometry = geometry.clone().optimize_if(Some(&Default::default())).unwrap_or(geometry);
        let mut it = LirMatMulUnary {
            mmm,
            c_fact,
            geometry,
            c_m_axis,
            c_n_axis,
            micro_ops,
            trivial_path: false,
        };
        it.update_trivial_path();
        Ok(it)
    }
    // for cost and info
    fn guess_k(&self) -> Option<TDim> {
        self.micro_ops
            .iter()
            .find_map(
                |o| {
                    if let ProtoFusedSpec::AddMatMul(geo, _, _) = o {
                        Some(geo)
                    } else {
                        None
                    }
                },
            )
            .map(|geo| geo.k.clone())
    }

    fn m_n(&self) -> (TDim, TDim) {
        match &self.geometry {
            MatrixGeometry::Concrete(ConcreteMatrixGeometry { m, n }) => (m.to_dim(), n.to_dim()),
            MatrixGeometry::Symbolic(SymbolicMatrixGeometry { m, n, .. }) => (m.clone(), n.clone()),
        }
    }

    fn update_trivial_path(&mut self) {
        self.trivial_path = self.can_use_trivial_path();
    }

    fn can_use_trivial_path(&self) -> bool {
        self.c_fact.shape.is_concrete()
            && self.geometry.is_concrete()
            && self
                .c_fact
                .shape
                .iter()
                .enumerate()
                .all(|(ax, dim)| ax == self.c_m_axis || ax == self.c_n_axis || dim == TDim::Val(1))
            && self.micro_ops.iter().all(|o| !o.has_symbols())
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
        if fact.datum_type != self.mmm.internal_type() {
            v = patch.wire_node(
                format!("{}.cast-input-{}", node.name, node.inputs.len()),
                cast(self.mmm.internal_type()),
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
        if other_shape[self.c_m_axis] == self.c_fact.shape[self.c_m_axis]
            && other_shape[self.c_m_axis] == other_shape.volume()
        {
            return self.fuse_op(
                model,
                node,
                patch,
                vec![ProtoFusedSpec::BinPerRow(
                    value,
                    binop,
                    MapOutputAxisToInput(tvec!((self.c_m_axis, self.c_m_axis))),
                )],
                &additional_input,
            );
        }
        if other_shape[self.c_n_axis] == self.c_fact.shape[self.c_n_axis]
            && other_shape[self.c_n_axis] == other_shape.volume()
        {
            return self.fuse_op(
                model,
                node,
                patch,
                vec![ProtoFusedSpec::BinPerCol(
                    value,
                    binop,
                    MapOutputAxisToInput(tvec!((self.c_n_axis, self.c_n_axis))),
                )],
                &additional_input,
            );
        }
        Ok(None)
    }
}
