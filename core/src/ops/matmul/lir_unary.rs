use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul, MatrixStoreSpec, ScratchSpace};

#[derive(PartialEq, Clone, Hash, Debug)]
pub enum ProtoFusedSpec {
    Min(AttrOrInput),
    Max(AttrOrInput),
    PerRowMul(AttrOrInput),
    PerRowAdd(AttrOrInput),
    PerColMul(AttrOrInput),
    PerColAdd(AttrOrInput),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    ScalarMul(AttrOrInput),
    ScalarAdd(AttrOrInput),
    QAway(AttrOrInput, usize),
    AddUnicast(AttrOrInput),
}

impl ProtoFusedSpec {
    pub fn resolve<'t>(&'t self, inputs: &'t [Arc<Tensor>]) -> FusedSpec<'t> {
        match self {
            ProtoFusedSpec::Min(v) => FusedSpec::Min(v.tensor(inputs)),
            ProtoFusedSpec::Max(v) => FusedSpec::Max(v.tensor(inputs)),
            ProtoFusedSpec::PerColAdd(v) => FusedSpec::PerColAdd(v.tensor(inputs)),
            ProtoFusedSpec::PerRowAdd(v) => FusedSpec::PerRowAdd(v.tensor(inputs)),
            ProtoFusedSpec::PerColMul(v) => FusedSpec::PerColMul(v.tensor(inputs)),
            ProtoFusedSpec::PerRowMul(v) => FusedSpec::PerRowMul(v.tensor(inputs)),
            ProtoFusedSpec::ScalarMul(v) => FusedSpec::ScalarMul(v.tensor(inputs)),
            ProtoFusedSpec::ScalarAdd(v) => FusedSpec::ScalarAdd(v.tensor(inputs)),
            ProtoFusedSpec::QAway(v, n) => FusedSpec::QAway(v.tensor(inputs), *n),
            ProtoFusedSpec::AddRowColProducts(row, col) => {
                FusedSpec::AddRowColProducts(row.tensor(inputs), col.tensor(inputs))
            }
            ProtoFusedSpec::AddUnicast(v) => FusedSpec::AddUnicast(v.tensor(inputs).view()),
        }
    }
}

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct LirMatMulUnary {
    pub b_storage: MatrixStoreSpec,
    pub c_fact: TypedFact,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
    pub micro_ops: ArrayD<(Arc<Tensor>, Vec<ProtoFusedSpec>)>,
    #[educe(Hash(method = "hash_mmm"))]
    pub mmm: Box<dyn MatMatMul>,
    pub m: usize,
    pub k: usize,
    pub c_final_shape: ShapeFact,
}

impl LirMatMulUnary {
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> &TDim {
        &self.c_fact.shape[self.c_n_axis]
    }
}

fn hash_mmm<H: std::hash::Hasher>(mmm: &Box<dyn MatMatMul>, state: &mut H) {
    // FIXME: this is buggy, but it should not matter too much
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
            "c_shape: {:?} m:{} k:{} n:{}",
            self.c_fact,
            self.m(),
            self.k(),
            self.n(),
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
            eval(op, scratch.as_mut(), &inputs, &shape, op.c_m_axis, op.c_n_axis, &final_shape)
        }
    }
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
        self.c_fact.shape.is_concrete()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State)))
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut scratch = unsafe { self.mmm.allocate_scratch_space() };
        eval(
            self,
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
    scratch: &mut dyn ScratchSpace,
    inputs: &[Arc<Tensor>],
    c_shape: &[usize],
    c_m_axis: usize,
    c_n_axis: usize,
    c_final_shape: &[usize],
) -> TractResult<TVec<Arc<Tensor>>> {
    unsafe {
        let a_dt = op.micro_ops.iter().next().unwrap().0.datum_type();
        let mut c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
        let c_storage = if c_shape[c_n_axis] == 1 {
            op.mmm.c_vec_from_data()
        } else {
            op.mmm.c_view_with_axis(c_m_axis, c_n_axis)
        };
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
                let f: Vec<FusedSpec> = fused.iter().map(|f| f.resolve(inputs)).collect::<Vec<_>>();
                op.mmm.run_with_scratch_space(
                    scratch,
                    &op.mmm.a_packed(a_dt).wrap(&pa.view()),
                    &op.b_storage.wrap(&TensorView::at_prefix_unchecked(&inputs[0], &*b_prefix)),
                    &mut c_storage.wrap(&c_view),
                    &f,
                )?;
            }
        } else {
            let (pa, fused) = op.micro_ops.iter().next().unwrap();
            let f: Vec<FusedSpec> = fused.iter().map(|f| f.resolve(inputs)).collect::<Vec<_>>();
            op.mmm.run_with_scratch_space(
                scratch,
                &op.mmm.a_packed(a_dt).wrap(&pa.view()),
                &op.b_storage.wrap(&inputs[0].view()),
                &mut c_storage.wrap(&c.view_mut()),
                &f,
            )?;
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
        let sums = self.c_fact.shape.iter().maybe_product()?;
        Ok(tvec!(
            (Cost::FMA(self.mmm.internal_type()), sums.maybe_mul(&self.k().to_dim())?),
            (
                Cost::Params(self.micro_ops.as_slice().unwrap()[0].0.datum_type()),
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
                let mut patch = TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self { c_final_shape: succ.outputs[0].fact.shape.clone(), ..self.clone() },
                )?;
                patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ, node));
                return Ok(Some(patch));
            }
        }

        let merge = |fused_micro_op: &[ProtoFusedSpec],
                     additional_inputs: &[OutletId]|
         -> TractResult<Option<TypedModelPatch>> {
            let mut new_op = self.clone();
            new_op
                .micro_ops
                .iter_mut()
                .for_each(|ops| ops.1.extend(fused_micro_op.iter().cloned()));
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
        };

        if let Some(op) = succ.op_as::<ops::element_wise::ElementWiseOp>().map(|ew| ew.0.as_ref()) {
            if let Some(cast) = op.downcast_ref::<ops::cast::Cast>().map(|cast| cast.to) {
                if cast == i8::datum_type() && self.c_fact.datum_type == i32::datum_type() {
                    let mmm = tract_linalg::ops()
                        .mmm(
                            self.micro_ops.iter().next().unwrap().0.datum_type(),
                            model.outlet_fact(node.inputs[0])?.datum_type,
                            i8::datum_type(),
                            self.m(),
                            self.k(),
                            self.n().to_usize()?,
                        )
                        .context("MMM instantiation")?;
                    let c_fact = TypedFact::dt_shape(i8::datum_type(), self.c_fact.shape.clone());
                    let mut patch = TypedModelPatch::fuse_with_next(
                        model,
                        &node,
                        Self { mmm, c_fact, ..self.clone() },
                    )?;
                    patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ, node));
                    return Ok(Some(patch));
                }
            }
        } else if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
            if op.a.len() == 1 {
                if op.mini_op.is::<ops::quant::Scale>()
                    && self.c_fact.datum_type == i32::datum_type()
                {
                    // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/util/gemmlowp_common.h#L16
                    let factor = op.a.cast_to_scalar::<f32>()?;
                    if factor <= 0.0 || factor >= 0.5 {
                        return Ok(None);
                    }
                    let factor_bits = factor.to_bits();
                    let current_exponent = factor_bits >> 23;
                    let bumped_multi = f32::from_bits(factor_bits & 0x007fffff | 0x3f000000);
                    let int_multi = (bumped_multi * (1i64 << 31) as f32).round() as u32;
                    let shift = 126usize - current_exponent as usize;
                    return merge(&[ProtoFusedSpec::QAway(tensor0(int_multi).into(), shift)], &[]);
                } else if op.mini_op.is::<ops::math::Max>() {
                    return merge(&[ProtoFusedSpec::Max((&op.a).into())], &[]);
                } else if op.mini_op.is::<ops::math::Min>() {
                    return merge(&[ProtoFusedSpec::Min((&op.a).into())], &[]);
                } else if op.mini_op.is::<ops::math::Mul>() {
                    return merge(&[ProtoFusedSpec::ScalarMul((&op.a).into())], &[]);
                }
            } else if op.a.shape()[op.a.rank() - 2] == 1
                && op.a.shape()[op.a.rank() - 1].to_dim() == self.c_fact.shape[self.c_m_axis]
            {
                if op.mini_op.is::<ops::math::Mul>() {
                    return merge(&[ProtoFusedSpec::PerRowMul((&op.a).into())], &[]);
                } else if op.mini_op.is::<ops::math::Add>() {
                    return merge(&[ProtoFusedSpec::PerRowAdd((&op.a).into())], &[]);
                }
            } else if op.a.shape()[op.a.rank() - 1] == 1
                && op.a.shape()[op.a.rank() - 2].to_dim()
                    == self.c_fact.shape[self.c_fact.rank() - 2]
            {
                let arg = &op.a;
                if op.mini_op.is::<ops::math::Mul>() {
                    return merge(&[ProtoFusedSpec::PerRowMul(arg.into())], &[]);
                } else if op.mini_op.is::<ops::math::Add>() {
                    return merge(&[ProtoFusedSpec::PerRowAdd(arg.into())], &[]);
                }
            }
            /*
        } else if let Some(op) = succ.op_as::<ops::binary::MergeOpUnicast>() {
            let other_slot = 1 - node.outputs[0].successors[0].slot;
            let other_input = succ.inputs[other_slot];
            if op.0.is::<ops::math::Add>() {
                return merge(
                    &[ProtoFusedSpec::AddUnicast(node.inputs.len().into())],
                    &[other_input],
                );
            }
            */
        };
        Ok(None)
    }

    as_op!();
}
