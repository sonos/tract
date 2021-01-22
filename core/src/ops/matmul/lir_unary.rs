use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul, MatrixStoreSpec};

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct LirMatMulUnary {
    pub b_storage: MatrixStoreSpec,
    pub c_fact: TypedFact,
    pub c_m_axis: usize,
    pub c_n_axis: usize,
    pub packed_as: ArrayD<Arc<Tensor>>,
    pub fused_ops: Option<ArrayD<Vec<FusedSpec>>>,
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
        if let Some(f) = &self.fused_ops {
            infos.push(format!("{:?}", f));
        }
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
        eval(op, &inputs[0], &shape, op.c_m_axis, op.c_n_axis, &final_shape)
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
        Ok(if self.is_stateless() { None } else { Some(Box::new(State)) })
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        eval(
            self,
            &inputs[0],
            self.c_fact.shape.as_concrete().unwrap(),
            self.c_m_axis,
            self.c_n_axis,
            self.c_final_shape.as_concrete().unwrap(),
        )
    }
}

fn eval(
    op: &LirMatMulUnary,
    input: &Tensor,
    c_shape: &[usize],
    c_m_axis: usize,
    c_n_axis: usize,
    c_final_shape: &[usize],
) -> TractResult<TVec<Arc<Tensor>>> {
    unsafe {
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
                let mut a = op.packed_as.view();
                let mut b_prefix = tvec!();
                let mut c_view = c.view();
                let mut fused = op.fused_ops.as_ref().map(|f| f.view());
                for (ix, &dim) in prefix.slice().iter().enumerate() {
                    if ix != c_m_axis && ix != c_n_axis {
                        a.index_axis_inplace(Axis(0), dim.min(a.shape()[0] - 1));
                        b_prefix.push(dim);
                        if let Some(f) = fused.as_mut() {
                            let d = dim.min(f.shape()[0] - 1);
                            f.index_axis_inplace(Axis(0), d);
                        }
                    }
                    c_view.offset_axis_unchecked(ix, dim as isize);
                }
                let pa: &Tensor = a.iter().next().unwrap();
                if let Some(fused) = fused {
                    op.mmm.run(
                        &op.mmm.a_packed().wrap(&pa.view()),
                        &op.b_storage.wrap(&TensorView::at_prefix_unchecked(&input, &*b_prefix)),
                        &mut c_storage.wrap(&c_view),
                        &fused.as_slice().unwrap()[0],
                    )?;
                } else {
                    op.mmm.run(
                        &op.mmm.a_packed().wrap(&pa.view()),
                        &op.b_storage.wrap(&TensorView::at_prefix_unchecked(&input, &*b_prefix)),
                        &mut c_storage.wrap(&c_view),
                        &[],
                    )?;
                }
            }
        } else {
            if let Some(fused) = &op.fused_ops {
                op.mmm.run(
                    &op.mmm.a_packed().wrap(&op.packed_as.as_ptr().as_ref().unwrap().view()),
                    &op.b_storage.wrap(&input.view()),
                    &mut c_storage.wrap(&c.view_mut()),
                    &fused.as_ptr().as_ref().unwrap(),
                )?;
            } else {
                op.mmm.run(
                    &op.mmm.a_packed().wrap(&op.packed_as.as_ptr().as_ref().unwrap().view()),
                    &op.b_storage.wrap(&input.view()),
                    &mut c_storage.wrap(&c.view_mut()),
                    &[],
                )?;
            }
        }
        c.set_shape_unchecked(c_final_shape);
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let c_prefix_len = self.c_fact.rank() - 2;
        if self.packed_as.ndim() != c_prefix_len {
            bail!(
                "Constant A table and c_prefix should have the same len. (resp {} and {})",
                self.packed_as.ndim(),
                c_prefix_len
            );
        }
        if let Some(f) = &self.fused_ops {
            if f.ndim() != self.c_fact.rank() - 2 {
                bail!(
                    "Fused op prefix and c_prefix should be of rank two less than output. fused: {:?} output: {:?}",
                    self.fused_ops,
                    self.c_fact
                    );
            }
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
                Cost::Params(self.packed_as.as_slice().unwrap()[0].datum_type()),
                self.packed_as.iter().fold(0.to_dim(), |sum, a| sum + a.len())
            )
        ))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if let Some(succ) = model.single_succ(node.id)? {
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
            let fused_micro_op = if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                if op.a.len() == 1 {
                    if op.mini_op.is::<ops::math::Max>() {
                        Some(tvec!(FusedSpec::Max(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Min>() {
                        Some(tvec!(FusedSpec::Min(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::ScalarMul(op.a.clone().into_tensor())))
                    } else {
                        None
                    }
                } else if op.a.shape()[op.a.rank() - 2] == 1
                    && op.a.shape()[op.a.rank() - 1].to_dim() == self.c_fact.shape[self.c_m_axis]
                {
                    if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::PerRowMul(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Add>() {
                        Some(tvec!(FusedSpec::PerRowAdd(op.a.clone().into_tensor())))
                    } else {
                        None
                    }
                } else if op.a.shape()[op.a.rank() - 1] == 1
                    && op.a.shape()[op.a.rank() - 2].to_dim()
                        == self.c_fact.shape[self.c_fact.rank() - 2]
                {
                    let arg = op.a.clone().into_tensor();
                    if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::PerRowMul(arg)))
                    } else if op.mini_op.is::<ops::math::Add>() {
                        Some(tvec!(FusedSpec::PerRowAdd(arg)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(op) = fused_micro_op {
                let mut new_op = self.clone();
                new_op
                    .fused_ops
                    .get_or_insert_with(|| {
                        let shape = vec![1; self.c_fact.rank() - 2];
                        ArrayD::from_shape_fn(shape, |_| vec![])
                    })
                    .map_inplace(|v| v.extend(op.iter().cloned()));
                let mut patch = TypedModelPatch::fuse_with_next(model, &node, new_op)?;
                patch.dont_apply_twice = Some(format!("Fuse {} into {}", succ, node));
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    as_op!();
}
