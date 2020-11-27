use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul, MatrixStoreSpec};

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct LirMatMulUnary {
    pub b_storage: MatrixStoreSpec,
    pub c_fact: TypedFact,
    pub c_shape_override: Option<(Dims, Dims)>,
    pub packed_as: ArrayD<Arc<Tensor>>,
    pub fused_ops: Option<ArrayD<Vec<FusedSpec>>>,
    #[educe(Hash(method = "hash_mmm"))]
    pub mmm: Box<dyn MatMatMul>,
    pub m: usize,
    pub k: usize,
}

impl LirMatMulUnary {
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> &TDim {
        if let Some(over) = &self.c_shape_override {
            over.0.last().unwrap()
        } else {
            self.c_fact.shape.last().unwrap()
        }
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
            "c_shape_orerride: {:?} m:{} k:{} n:{}",
            self.c_shape_override,
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
        if let Some(prefix) = &op.c_shape_override {
            let shape = prefix.0.eval_to_usize(&session.resolved_symbols)?;
            let strides = prefix.1.eval_to_isize(&session.resolved_symbols)?;
            eval(
                op,
                &inputs[0],
                &op.c_fact.shape.eval_to_usize(&session.resolved_symbols)?,
                Some((&shape, &strides)),
            )
        } else {
            eval(op, &inputs[0], &op.c_fact.shape.eval_to_usize(&session.resolved_symbols)?, None)
        }
    }
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
        self.c_fact.shape.is_concrete()
            && self
                .c_shape_override
                .as_ref()
                .map(|p| p.0.is_concrete() && p.1.is_concrete())
                .unwrap_or(true)
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(if self.is_stateless() { None } else { Some(Box::new(State)) })
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if let Some(p) = &self.c_shape_override {
            let shape = p.0.as_concrete().unwrap();
            let strides = p.1.as_concrete().unwrap();
            eval(
                self,
                &inputs[0],
                self.c_fact.shape.as_concrete().unwrap(),
                Some((shape, unsafe { std::mem::transmute(strides) })),
            )
        } else {
            eval(self, &inputs[0], self.c_fact.shape.as_concrete().unwrap(), None)
        }
    }
}

fn eval(
    op: &LirMatMulUnary,
    input: &Tensor,
    c_shape: &[usize],
    c_shape_override: Option<(&[usize], &[isize])>,
) -> TractResult<TVec<Arc<Tensor>>> {
    unsafe {
        let c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
        let c_view = if let Some((dims, strides)) = c_shape_override {
            TensorView::from_bytes(&c, 0, &dims, &strides)
        } else {
            c.view()
        };
        let c_storage = op.mmm.c_view();
        if op.packed_as.ndim() > 0 {
            for prefix in indices(&c_view.shape()[..c_view.rank() - 2]).into_iter() {
                let mut a = op.packed_as.view();
                let mut b_prefix = tvec!();
                let mut c_view = c_view.clone();
                for (ix, &dim) in prefix.slice().iter().enumerate() {
                    a.index_axis_inplace(Axis(0), dim.min(a.shape()[0] - 1));
                    b_prefix.push(dim.min(input.shape()[ix] - 1));
                    c_view.offset_axis_unchecked(ix, dim as isize);
                }
                let pa: &Tensor = a.iter().next().unwrap();
                if let Some(fused) = &op.fused_ops {
                    let mut fused = fused.view();
                    for &dim in prefix.slice() {
                        let d = dim.min(fused.shape()[0] - 1);
                        fused.index_axis_inplace(Axis(0), d);
                    }
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
                    &mut c_storage.wrap(&c_view),
                    &fused.as_ptr().as_ref().unwrap(),
                )?;
            } else {
                op.mmm.run(
                    &op.mmm.a_packed().wrap(&op.packed_as.as_ptr().as_ref().unwrap().view()),
                    &op.b_storage.wrap(&input.view()),
                    &mut c_storage.wrap(&c_view),
                    &[],
                )?;
            }
        }
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let c_prefix_len =
            self.c_shape_override.as_ref().map(|prefix| prefix.0.len() - 2).unwrap_or(0);
        if self.packed_as.ndim() != c_prefix_len {
            bail!(
                "Constant table and c_prefix should have the same len. (resp {} and {})",
                self.packed_as.ndim(),
                c_prefix_len
            );
        }
        if let Some(f) = &self.fused_ops {
            if f.ndim() != c_prefix_len {
                bail!(
                    "Fused op prefix and c_prefix should have the same len. (resp {} and {})",
                    f.ndim(),
                    c_prefix_len
                );
            }
        }
        Ok(tvec!(self.c_fact.clone()))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let sums = self
            .c_shape_override
            .as_ref()
            .map(|c| c.0.iter().maybe_product().unwrap())
            .unwrap_or(self.n().clone() * self.m());
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
                    return Ok(Some(TypedModelPatch::fuse_with_next(
                        model,
                        &node,
                        Self { c_fact: succ.outputs[0].fact.clone(), ..self.clone() },
                    )?));
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
                    /*
                } else if op.a.shape()[op.a.rank() - 1] == 1 {
                    if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::PerRowMul(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Add>() {
                        Some(tvec!(FusedSpec::PerRowAdd(op.a.clone().into_tensor())))
                    } else {
                        None
                    }
                    */
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
                        let shape =
                            vec![
                                1;
                                self.c_shape_override.as_ref().map(|c| c.0.len() - 2).unwrap_or(0)
                            ];
                        ArrayD::from_shape_fn(shape, |_| vec![])
                    })
                    .map_inplace(|v| v.extend(op.iter().cloned()));
                return Ok(Some(TypedModelPatch::fuse_with_next(model, &node, new_op)?));
            }
        }
        Ok(None)
    }

    as_op!();
}
