use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{
    BinOp, FusedSpec, InputStoreSpec, MatMatMul, OutputStore, OutputStoreSpec, RoundingPolicy,
    ScratchSpace,
};

#[derive(PartialEq, Clone, Hash, Debug)]
pub enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(AttrOrInput),
    QScale(usize, RoundingPolicy, i32),
    Store,
}

impl ProtoFusedSpec {
    pub fn resolve<'t>(
        &'t self,
        inputs: &'t [Arc<Tensor>],
        output_spec: &'t OutputStoreSpec,
        output: OutputStore,
    ) -> FusedSpec<'t> {
        match self {
            ProtoFusedSpec::BinScalar(v, op) => FusedSpec::BinScalar(v.tensor(inputs), *op),
            ProtoFusedSpec::BinPerRow(v, op) => FusedSpec::BinPerRow(v.tensor(inputs), *op),
            ProtoFusedSpec::BinPerCol(v, op) => FusedSpec::BinPerCol(v.tensor(inputs), *op),
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(row.tensor(inputs), col.tensor(inputs))
            }
            ProtoFusedSpec::AddUnicast(v) => unsafe {
                FusedSpec::AddUnicast(output_spec.wrap(&v.tensor(inputs).view()))
            },
            ProtoFusedSpec::QScale(s, rp, m) => FusedSpec::QScale(*s, *rp, *m),
            ProtoFusedSpec::Store => FusedSpec::Store(output),
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct ConcreteMatMulGeometry {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub b_storage: InputStoreSpec,
}

#[derive(Clone, Debug, Hash)]
pub struct SymbolicMatMulGeometry {
    pub m: TDim,
    pub k: TDim,
    pub n: TDim,
    pub mmm: Box<dyn MatMatMul>,
    pub b_datum_type: DatumType,
}

impl ResolveTo<ConcreteMatMulGeometry> for SymbolicMatMulGeometry {
    type Param = SymbolValues;
    fn resolve(&self, param: &Self::Param) -> TractResult<ConcreteMatMulGeometry> {
        let m = self.m.eval(param).to_usize()?;
        let k = self.k.eval(param).to_usize()?;
        let n = self.n.eval(param).to_usize()?;
        let b_storage = unsafe { self.mmm.b_packed(self.b_datum_type.size_of(), k) };
        Ok(ConcreteMatMulGeometry { m, k, n, b_storage })
    }
}

pub type MatMulGeometry = GeometryBound<SymbolicMatMulGeometry, ConcreteMatMulGeometry>;

impl MatMulGeometry {
    fn k(&self) -> Cow<TDim> {
        match self {
            Self::Symbolic(it) => Cow::Borrowed(&it.k),
            Self::Concrete(it) => Cow::Owned(it.k.to_dim()),
        }
    }
}

#[derive(Clone, Educe, Debug)]
#[educe(Hash)]
pub struct LirMatMulUnary {
    pub c_fact: TypedFact,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
    pub micro_ops: ArrayD<(Arc<Tensor>, Vec<ProtoFusedSpec>)>,
    pub c_final_shape: ShapeFact,
    pub geometry: MatMulGeometry,
    #[educe(Hash(method = "hash_mmm"))]
    pub mmm: Box<dyn MatMatMul>,
    pub reshape_post: Vec<AxisOp>,
}

fn hash_mmm<H: std::hash::Hasher>(mmm: &Box<dyn MatMatMul>, state: &mut H) {
    mmm.type_id().hash(state)
}

impl DynHash for LirMatMulUnary {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Op for LirMatMulUnary {
    fn name(&self) -> Cow<str> {
        "LirMatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut infos = vec![format!(
            "c_shape:{:?}, c_m_axis:{} c_n_axis:{} b_storage:{:?}",
            self.c_fact, self.c_m_axis, self.c_n_axis, self.geometry,
        )];
        infos.push(format!("Mult: {}", self.mmm));
        infos.push(format!("Ops: {:?}", self.micro_ops));
        Ok(infos)
    }

    op_core_lir!();
    op_as_typed_op!();
}

#[derive(Clone, Debug)]
struct State;
impl OpState for State {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op = op.downcast_ref::<LirMatMulUnary>().unwrap();
        let shape = op.c_fact.shape.eval_to_usize(&session.resolved_symbols)?;
        let final_shape = op.c_final_shape.eval_to_usize(&session.resolved_symbols)?;
        unsafe {
            let geometry = op.geometry.to_concrete(&session.resolved_symbols)?;
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
            eval(
                op,
                &geometry,
                scratch.as_mut(),
                &inputs,
                &shape,
                op.c_m_axis,
                op.c_n_axis,
                &*final_shape,
            )
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

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let geometry = self.geometry.to_concrete(&SymbolValues::default())?;
        let mut scratch = unsafe { self.mmm.allocate_scratch_space() };
        eval(
            self,
            &geometry,
            scratch.as_mut(),
            &*inputs,
            self.c_fact.shape.as_concrete().unwrap(),
            self.c_m_axis,
            self.c_n_axis,
            self.c_final_shape.as_concrete().unwrap(),
        )
    }
}

fn eval(
    op: &LirMatMulUnary,
    geometry: &ConcreteMatMulGeometry,
    scratch: &mut dyn ScratchSpace,
    inputs: &[Arc<Tensor>],
    c_shape: &[usize],
    c_m_axis: usize,
    c_n_axis: usize,
    c_final_shape: &[usize],
) -> TractResult<TVec<Arc<Tensor>>> {
    unsafe {
        debug_assert!(op.micro_ops.len() > 0);
        let size_of_a = (&*op.micro_ops.as_ptr()).0.datum_type().size_of();
        let mut c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
        let c_storage = op.mmm.c_view_with_axis(c_m_axis, c_n_axis);
        if op
            .c_fact
            .shape
            .iter()
            .enumerate()
            .any(|(ix, d)| ix != c_m_axis && ix != c_n_axis && d != 1.to_dim())
        {
            let mut looping_shape: TVec<usize> = c_shape.into();
            looping_shape[c_m_axis] = 1;
            looping_shape[c_n_axis] = 1;
            for prefix in indices(&*looping_shape) {
                let mut ops = op.micro_ops.view();
                let mut b_prefix = tvec!();
                let mut c_view = c.view();
                for (ix, &dim) in prefix.slice().iter().enumerate() {
                    if ix != c_m_axis && ix != c_n_axis {
                        ops.index_axis_inplace(Axis(0), dim.min(ops.shape()[0] - 1));
                        b_prefix.push(dim);
                    }
                    c_view.offset_axis_unchecked(ix, dim as isize);
                }
                let (pa, fused) = ops.iter().next().unwrap();
                let c_store = c_storage.wrap(&c_view);
                let mut f = tvec!(FusedSpec::AddMatMul {
                    k: geometry.k,
                    a: op.mmm.a_packed(size_of_a, geometry.k).wrap(&pa.view()),
                    b: geometry
                        .b_storage
                        .wrap(&TensorView::at_prefix_unchecked(&inputs[0], &*b_prefix)),
                });
                f.extend(fused.iter().map(|f| f.resolve(inputs, &c_storage, c_store)));
                op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &f)?;
            }
        } else {
            let (pa, fused) = &*op.micro_ops.as_ptr();
            let c_store = c_storage.wrap(&c.view_mut());
            let mut f = Vec::with_capacity(fused.len() + 1);
            f.push(FusedSpec::AddMatMul {
                k: geometry.k,
                a: op.mmm.a_packed(size_of_a, geometry.k).wrap(&pa.view()),
                b: geometry.b_storage.wrap(&inputs[0].view()),
            });
            for ix in 0..fused.len() {
                f.push(fused.get_unchecked(ix).resolve(inputs, &c_storage, c_store));
            }
            op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &f)?;
        }
        c.set_shape_unchecked(c_final_shape);
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let c_prefix_len = self.c_fact.rank() - 2;
        if self.micro_ops.ndim() != c_prefix_len {
            bail!(
                "Constant A table and c_prefix should have the same len. (resp {} and {})",
                self.micro_ops.ndim(),
                c_prefix_len
            );
        }
        let mut fact = self.c_fact.clone();
        fact.shape = self.c_final_shape.clone();
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let sums: TDim = self.c_fact.shape.iter().product();
        Ok(tvec!(
            (Cost::FMA(self.mmm.internal_type()), sums * self.geometry.k().as_ref()),
            (
                Cost::Params(self.micro_ops.as_slice().unwrap()[0].0.datum_type().unquantized()),
                self.micro_ops.iter().fold(0.to_dim(), |sum, a| sum + a.0.len())
            )
        ))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if node.outputs.len() != 1
            || node.outputs[0].successors.len() != 1
            || model.output_outlets()?.iter().any(|outlet| outlet.node == node.id)
        {
            return Ok(None);
        }
        let succ = model.node(node.outputs[0].successors[0].node);
        if let Some(op) = succ.op_as::<ops::AxisOp>() {
            if op.only_shape() {
                let mut reshape_post = self.reshape_post.clone();
                reshape_post.push(op.clone());
                let mut patch = TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self {
                        c_final_shape: succ.outputs[0].fact.shape.clone(),
                        reshape_post,
                        ..self.clone()
                    },
                )?;
                patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ, node));
                return Ok(Some(patch));
            }
        }

        if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>().map(|ew| ew.0.as_ref()) {
            if let Some(cast) = op.downcast_ref::<ops::cast::Cast>().map(|cast| cast.to) {
                if (cast.unquantized() == i8::datum_type()
                    || cast.unquantized() == u8::datum_type())
                    && self.c_fact.datum_type == i32::datum_type()
                {
                    let at = self.micro_ops.iter().nth(0).unwrap().0.datum_type();
                    let bt = model.outlet_fact(node.inputs[0])?.datum_type;
                    let mmm = tract_linalg::ops()
                        .mmm(
                            at,
                            bt,
                            i8::datum_type(),
                            self.c_fact.shape[self.c_m_axis].to_usize().ok(),
                            None,
                            self.c_fact.shape[self.c_n_axis].to_usize().ok(),
                        )
                        .unwrap();

                    let c_fact = TypedFact::dt_shape(cast, self.c_fact.shape.clone());
                    let mut patch = TypedModelPatch::fuse_with_next(
                        model,
                        &node,
                        Self { mmm, c_fact, ..self.clone() },
                    )?;
                    patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ, node));
                    return Ok(Some(patch));
                }
            } else if let Some(op) = op.downcast_ref::<ops::math::QScale>() {
                return self.fuse_op_with_broadcast(
                    model,
                    node,
                    &[ProtoFusedSpec::QScale(op.shift, op.policy, op.mult)],
                    &[],
                );
            }
        } else if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
            let binop =
                if let Some(op) = op.mini_op.as_linalg_binop() { op } else { return Ok(None) };
            let shape = op.a.shape().into();
            return self.fuse_binary(model, node, &shape, op.a.clone().into(), binop, &[]);
        } else if let Some(op) = succ.op_as::<ops::binary::TypedBinOp>() {
            let mut binop =
                if let Some(op) = op.0.as_linalg_binop() { op } else { return Ok(None) };
            let flipped = succ.inputs[0].node == node.id;
            if flipped {
                binop = binop.flip();
            }
            let other_outlet = succ.inputs[flipped as usize];
            let other_fact = model.outlet_fact(other_outlet)?;
            let value = node.inputs.len().into();
            return self.fuse_binary(model, node, &other_fact.shape, value, binop, &[other_outlet]);
        } else if let Some(op) = succ.op_as::<ops::binary::MergeOpUnicast>() {
            if self.c_n_axis == self.c_final_shape.rank() - 2
                && self.c_m_axis == self.c_final_shape.rank() - 1
                && self.micro_ops.len() == 1
            {
                let other_slot = 1 - node.outputs[0].successors[0].slot;
                let other_input = succ.inputs[other_slot];

                if op.0.is::<ops::math::Add>() {
                    return self.fuse_op_with_broadcast(
                        model,
                        node,
                        &[ProtoFusedSpec::AddUnicast(node.inputs.len().into())],
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
    fn fuse_op(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        fused_micro_op: &ArrayD<Vec<ProtoFusedSpec>>,
        additional_inputs: &[OutletId],
    ) -> TractResult<Option<TypedModelPatch>> {
        let succ = model.node(node.outputs[0].successors[0].node);
        let mut new_op = self.clone();
        new_op.micro_ops.zip_mut_with(fused_micro_op, |lhs, rhs| {
            lhs.1.pop();
            lhs.1.extend(rhs.iter().cloned());
            lhs.1.push(ProtoFusedSpec::Store);
        });
        let mut patch = TypedModelPatch::new(format!("fusing {}", succ));
        patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ.name, node.name));
        let inputs = node
            .inputs
            .iter()
            .chain(additional_inputs.iter())
            .map(|i| patch.tap_model(model, *i))
            .collect::<TractResult<TVec<OutletId>>>()?;
        let output = patch.wire_node(&node.name, new_op, &inputs)?;
        patch.shunt_outside(model, succ.id.into(), output[0])?;
        Ok(Some(patch))
    }

    fn fuse_op_with_broadcast(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        fused_micro_op: &[ProtoFusedSpec],
        additional_inputs: &[OutletId],
    ) -> TractResult<Option<TypedModelPatch>> {
        let array = arr0(fused_micro_op.to_vec()).into_dyn();
        self.fuse_op(model, node, &array, additional_inputs)
    }

    fn fuse_binary(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        shape: &ShapeFact,
        value: AttrOrInput,
        binop: BinOp,
        additional_inputs: &[OutletId],
    ) -> TractResult<Option<TypedModelPatch>> {
        if shape.volume() == 1.to_dim() {
            return self.fuse_op_with_broadcast(
                model,
                node,
                &[ProtoFusedSpec::BinScalar(value, binop)],
                additional_inputs,
            );
        }
        let mut other_shape = shape.to_owned();
        for axis_change in self.reshape_post.iter().rev() {
            if axis_change.recip().change_shape(&mut other_shape, true).is_err() {
                return Ok(None)
            }
        }
        if other_shape[self.c_m_axis] == self.c_fact.shape[self.c_m_axis]
            && other_shape[self.c_m_axis] == other_shape.volume()
        {
            return self.fuse_op_with_broadcast(
                model,
                node,
                &[ProtoFusedSpec::BinPerRow(value, binop)],
                additional_inputs,
            );
        }
        if other_shape[self.c_n_axis] == self.c_fact.shape[self.c_n_axis]
            && other_shape[self.c_n_axis] == other_shape.volume()
        {
            return self.fuse_op_with_broadcast(
                model,
                node,
                &[ProtoFusedSpec::BinPerCol(value, binop)],
                additional_inputs,
            );
        }
        Ok(None)
    }
}
