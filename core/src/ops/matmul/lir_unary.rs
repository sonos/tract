use crate::internal::*;
use crate::ops::cast;
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
    AddMatMul(AddMatMulGeometry, AttrOrInput, AttrOrInput),
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp, MapOutputAxisToInput),
    BinPerCol(AttrOrInput, BinOp, MapOutputAxisToInput),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Scaler(Scaler),
    Store(OutputStoreSpec),
}

impl ProtoFusedSpec {
    pub fn name(&self) -> String {
        use ProtoFusedSpec::*;
        match self {
            AddMatMul(_, _, _) => format!("matmul"),
            BinScalar(_, op) => format!("scalar {op:?}"),
            BinPerRow(_, op, _) => format!("row {op:?}"),
            BinPerCol(_, op, _) => format!("col {op:?}"),
            AddRowColProducts(_, _) => "add row*col product".to_string(),
            AddUnicast(_, _) => "add to matrix".to_string(),
            Scaler(s) => format!("scale by {}", 1f32 * *s),
            Store(_oss) => "Store".to_string(),
        }
    }

    pub fn resolve<'t>(
        &'t self,
        inputs: &'t [TValue],
        output_coords: &[usize],
        symbols: &SymbolValues,
        output: &mut Tensor,
    ) -> TractResult<FusedSpec<'t>> {
        let fs = match self {
            ProtoFusedSpec::AddMatMul(geo, a, b) => {
                let mut a = a.tensor(inputs).view();
                unsafe {
                    geo.c_to_a_axis_mapping.translate_view(output_coords, &mut a);
                }
                let mut b = b.tensor(inputs).view();
                unsafe {
                    geo.c_to_b_axis_mapping.translate_view(output_coords, &mut b);
                }
                FusedSpec::AddMatMul {
                    k: geo.k.eval(&symbols).to_usize()?,
                    a: unsafe { geo.a_storage.wrap(&a)? },
                    b: unsafe { geo.b_storage.wrap(&b)? },
                }
            }
            ProtoFusedSpec::BinScalar(v, op) => FusedSpec::BinScalar(v.tensor(inputs), *op),
            ProtoFusedSpec::BinPerRow(v, op, map) => {
                let mut v = v.tensor(inputs).view();
                unsafe { map.translate_view(output_coords, &mut v) }
                FusedSpec::BinPerRow(v, *op)
            }
            ProtoFusedSpec::BinPerCol(v, op, map) => {
                let mut v = v.tensor(inputs).view();
                unsafe { map.translate_view(output_coords, &mut v) }
                FusedSpec::BinPerCol(v, *op)
            }
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(row.tensor(inputs), col.tensor(inputs))
            }
            ProtoFusedSpec::AddUnicast(store, v) => unsafe {
                let view = v.tensor(inputs).view_offsetting(&output_coords)?;
                FusedSpec::AddUnicast(store.wrap(&view))
            },
            ProtoFusedSpec::Scaler(scaler) => scaler.as_fused_spec(),
            ProtoFusedSpec::Store(oss) => {
                let view = output.view_offsetting_mut(&output_coords)?;
                FusedSpec::Store(unsafe { oss.wrap(&view) })
            }
        };
        Ok(fs)
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
}

#[derive(Clone, Debug)]
pub struct AddMatMulGeometry {
    pub k: TDim,
    pub a_storage: InputStoreSpec,
    pub b_storage: InputStoreSpec,
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
    pub c_final_shape: ShapeFact,
    pub geometry: MatrixGeometry,
    pub mmm: Box<dyn MatMatMul>,
    pub reshape_post: Vec<AxisOp>,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
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
        if let Some(geo) = self.geometry.as_concrete() {
            infos.push(format!(
                "Mult: m:{} k:{} n:{} with {}",
                geo.m, /*geo.k, */ 0, geo.n, self.mmm
            ));
        } else {
            infos.push(format!("Mult: {}", self.mmm));
        }
        infos.push(format!("Ops: {:?}", self.micro_ops.iter().map(|o| o.name()).join(">")));
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
        let geometry = op.geometry.to_concrete(symbols)?;
        let c_shape = op.c_fact.shape.eval_to_usize(symbols)?;
        let c_final_shape = op.c_final_shape.eval_to_usize(symbols)?;
        dbg!(&c_shape);
        let mut c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
        let mut looping_shape: TVec<usize> = c_shape.to_smallvec();
        looping_shape[op.c_m_axis] = 1;
        looping_shape[op.c_n_axis] = 1;
        dbg!(&op.micro_ops);
        //        if looping_shape.iter().any(|d| *d > 1) {
        for c_coords in indices(&*looping_shape) {
            let ops = op
                .micro_ops
                .iter()
                .map(|f| f.resolve(inputs, c_coords.slice(), symbols, &mut c))
                .collect::<TractResult<TVec<_>>>()?;
            dbg!(&ops);
            op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &ops)?;
        }
        /*
        } else {
        dbg!("fast path");
        let ops = op
        .micro_ops
        .iter()
        .map(|f| f.resolve(inputs, &looping_shape, symbols, &mut c))
        .collect::<TractResult<TVec<_>>>()?;
        dbg!("done prep");
        op.mmm.run_with_scratch_space(geometry.m, geometry.n, scratch, &ops)?;
        }
        */
        c.set_shape_unchecked(&c_final_shape);
        Ok(tvec!(c.into_tvalue()))
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        /*
        ensure!(
        self.micro_ops.ndim() == self.c_fact.rank(),
        "Constant A array rank and C rank should be the same. (resp {} and {})",
        self.micro_ops.ndim(),
        self.c_fact.rank()
        );
        */
        let mut fact = self.c_fact.clone();
        fact.shape = self.c_final_shape.clone();
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        todo!();
        /*
        let sums: TDim = self.c_fact.shape.iter().product();
        Ok(tvec!(
        (Cost::FMA(self.mmm.internal_type()), sums * self.geometry.k().as_ref()),
        (
        Cost::Params(self.micro_ops.as_slice().unwrap()[0].0.datum_type().unquantized()),
        self.micro_ops.iter().fold(0.to_dim(), |sum, a| sum + a.0.len())
        )
        ))
        */
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        return Ok(None);
        /*
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
                   node,
                   Self {
                   c_final_shape: succ.outputs[0].fact.shape.clone(),
                   reshape_post,
                   ..self.clone()
                   },
                   )?;
                   patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
                   return Ok(Some(patch));
                   }
                   }

                /*
                if let Some(cast) = succ.op_as::<ops::cast::Cast>().map(|cast| cast.to) {
                if (cast.unquantized() == i8::datum_type() || cast.unquantized() == u8::datum_type())
                && self.c_fact.datum_type == i32::datum_type()
                {
                let at = self.micro_ops.iter().next().unwrap().datum_type();
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

                let c_fact = cast.fact(self.c_fact.shape.clone());
                let mut patch = TypedModelPatch::fuse_with_next(
                model,
                node,
                Self { mmm, c_fact, ..self.clone() },
                )?;
                patch.dont_apply_twice = Some(format!("Fuse {succ} into {node}"));
                return Ok(Some(patch));
                }
                }
                */

                let mut patch = TypedModelPatch::new(format!("fusing {succ}"));
                patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ.name, node.name));
                if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>().map(|ew| ew.0.as_ref()) {
                    if let Some(op) = op.downcast_ref::<ops::math::QScale>() {
                        return self.fuse_op_with_broadcast(
                            model,
                            node,
                            patch,
                            &[ProtoFusedSpec::Scaler(op.scaler)],
                            &[],
                            );
                    }
        <<<<<<< HEAD
        =======
                    /* TODO
                       } else if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                       let binop =
                       if let Some(op) = op.mini_op.as_linalg_binop() { op } else { return Ok(None) };
                       let shape = op.a.shape().into();
                       if op.a.datum_type() != self.mmm.internal_type() {
                       return Ok(None);
                       }
                       return self.fuse_binary(model, node, &shape, op.a.clone().into(), binop, &[]);
                       */
        >>>>>>> ff2e4973 (f32 conv ok)
                } else if let Some(op) = succ.op_as::<ops::binary::TypedBinOp>() {
                    let mut binop =
                        if let Some(op) = op.0.as_linalg_binop() { op } else { return Ok(None) };
                    let flipped = succ.inputs[0].node == node.id;
                    if flipped {
                        binop = binop.flip();
                    }
                    let other_outlet = succ.inputs[flipped as usize];
                    return self.fuse_binary(model, node, patch, other_outlet, binop);
                } else if let Some(op) = succ.op_as::<ops::binary::MergeOpUnicast>() {
                    if self.micro_ops.len() == 1 && op.0.is::<ops::math::Add>() {
                        let other_slot = 1 - node.outputs[0].successors[0].slot;
                        let other_input = succ.inputs[other_slot];
                        let other_input = patch.tap_model(model, other_input)?;

                        if model.outlet_fact(other_input)?.shape == self.c_fact.shape {
                            let other_storage = unsafe { self.mmm.c_view(self.c_m_axis, self.c_n_axis) };
                            return self.fuse_op_with_broadcast(
                                model,
                                node,
                                patch,
                                &[ProtoFusedSpec::AddUnicast(other_storage, node.inputs.len().into())],
                                &[other_input],
                                );
                        }
                    }
                };
                Ok(None)
                    */
    }

    as_op!();
}

impl LirMatMulUnary {
    fn fuse_op(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        mut patch: TypedModelPatch,
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
        let mut inputs: TVec<OutletId> =
            node.inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<_>>()?;
        inputs.extend(additional_inputs.iter().cloned());
        let output = patch.wire_node(&node.name, new_op, &inputs)?;
        patch.shunt_outside(model, succ.id.into(), output[0])?;
        Ok(Some(patch))
    }

    fn fuse_op_with_broadcast(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: TypedModelPatch,
        fused_micro_op: &[ProtoFusedSpec],
        additional_inputs: &[OutletId],
    ) -> TractResult<Option<TypedModelPatch>> {
        let array = arr0(fused_micro_op.to_vec()).into_dyn();
        self.fuse_op(model, node, patch, &array, additional_inputs)
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
        let (value, additional_inputs): (AttrOrInput, TVec<OutletId>) =
            if let Some(konst) = &fact.konst {
                let v = konst.cast_to_dt(self.mmm.internal_type())?.into_owned().into_arc_tensor();
                (v.into(), tvec!())
            } else {
                let mut v = patch.tap_model(model, value)?;
                if fact.datum_type != self.mmm.internal_type() {
                    v = patch.wire_node(
                        format!("{}.cast-input-{}", node.name, node.inputs.len()),
                        cast::cast(self.mmm.internal_type()),
                        &[v],
                    )?[0];
                }
                (AttrOrInput::Input(node.inputs.len()), tvec!(v))
            };
        if fact.shape.volume() == 1.to_dim() {
            return self.fuse_op_with_broadcast(
                model,
                node,
                patch,
                &[ProtoFusedSpec::BinScalar(value, binop)],
                &additional_inputs,
            );
        }
        let mut other_shape = fact.shape.to_owned();
        for axis_change in self.reshape_post.iter().rev() {
            if axis_change.recip().change_shape(&mut other_shape, true).is_err() {
                return Ok(None);
            }
        }
        if other_shape[self.c_m_axis] == self.c_fact.shape[self.c_m_axis]
            && other_shape[self.c_m_axis] == other_shape.volume()
        {
            return self.fuse_op_with_broadcast(
                model,
                node,
                patch,
                &[ProtoFusedSpec::BinPerRow(value, binop)],
                &additional_inputs,
            );
        }
        if other_shape[self.c_n_axis] == self.c_fact.shape[self.c_n_axis]
            && other_shape[self.c_n_axis] == other_shape.volume()
        {
            return self.fuse_op_with_broadcast(
                model,
                node,
                patch,
                &[ProtoFusedSpec::BinPerCol(value, binop)],
                &additional_inputs,
            );
        }
        return Ok(None);
    }
}
