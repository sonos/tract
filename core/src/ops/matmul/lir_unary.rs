use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul};

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct LirMatMulUnary {
    pub c_trans: bool,
    pub bc_c_shape: TVec<usize>,
    pub c_fact: TypedFact,
    pub c_prefix_dim_and_stride: Option<(TVec<usize>, TVec<isize>)>,
    pub packed_as: ArrayD<Arc<Tensor>>,
    pub fused_ops: Option<ArrayD<Vec<FusedSpec>>>,
    #[educe(Hash(method = "hash_mmm"))]
    pub mmm: Box<dyn MatMatMul>,
    pub k: usize,
}

impl LirMatMulUnary {
    pub fn m(&self) -> usize {
        self.c_fact.shape[self.c_fact.rank() - 2 + self.c_trans as usize].to_usize().unwrap()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> &TDim {
        &self.c_fact.shape[self.c_fact.rank() - 2 + !self.c_trans as usize]
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
            "c_prefix: {:?} m:{} k:{} n:{} c_trans:{:?}",
            self.c_prefix_dim_and_stride, 0, 0, 0, self.c_trans
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
        let c_shape: TVec<usize> = op
            .c_fact
            .shape
            .iter()
            .map(|d| d.eval(&session.resolved_symbols).to_usize())
            .collect::<TractResult<_>>()?;
        eval(op, &inputs[0], &c_shape)
    }
}

impl EvalOp for LirMatMulUnary {
    fn is_stateless(&self) -> bool {
        self.c_fact.shape.as_finite().is_some()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(State)))
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        eval(self, &inputs[0], self.c_fact.shape.as_finite().unwrap())
    }
}

fn eval(op: &LirMatMulUnary, input: &Tensor, c_shape: &[usize]) -> TractResult<TVec<Arc<Tensor>>> {
    unsafe {
        let mut c = Tensor::uninitialized_dt(op.c_fact.datum_type, &c_shape)?;
        if let Some((prefix_dim, prefix_strides)) = &op.c_prefix_dim_and_stride {
            let mut tmp_shape: TVec<usize> = prefix_dim.iter().copied().collect();
            tmp_shape.push(c_shape[c_shape.len() - 2 + op.c_trans as usize]);
            tmp_shape.push(c_shape[c_shape.len() - 2 + op.c_trans as usize]);
            let mut tmp_strides: TVec<isize> = prefix_strides.iter().copied().collect();
            tmp_strides.push(0);
            tmp_strides.push(0);
            for prefix in indices(&**prefix_dim).into_iter() {
                let mut c = TensorView::from_bytes(&c, 0, &tmp_shape, &tmp_strides);
                let mut a = op.packed_as.view();
                let mut b_prefix = tvec!();
                for (ix, &dim) in prefix.slice().iter().enumerate() {
                    a.index_axis_inplace(Axis(0), dim.min(a.shape()[0] - 1));
                    b_prefix.push(dim.min(input.shape()[ix] - 1));
                    c.offset_axis_unchecked(ix, dim as isize);
                }
                let pa: &Tensor = a.iter().next().unwrap();
                if let Some(fused) = &op.fused_ops {
                    let mut fused = fused.view();
                    for &dim in prefix.slice() {
                        let d = dim.min(fused.shape()[0] - 1);
                        fused.index_axis_inplace(Axis(0), d);
                    }
                    op.mmm.run(
                        &pa.view(),
                        &TensorView::at_prefix_unchecked(&input, &*b_prefix),
                        &mut c,
                        &fused.as_slice().unwrap()[0],
                    )?;
                } else {
                    op.mmm.run(
                        &pa.view(),
                        &TensorView::at_prefix_unchecked(&input, &*b_prefix),
                        &mut c,
                        &[],
                    )?;
                }
            }
        } else {
            if let Some(fused) = &op.fused_ops {
                op.mmm.run(
                    &op.packed_as.as_ptr().as_ref().unwrap().view(),
                    &input.view(),
                    &mut c.view_mut(),
                    &fused.as_ptr().as_ref().unwrap(),
                )?;
            } else {
                op.mmm.run(
                    &op.packed_as.as_ptr().as_ref().unwrap().view(),
                    &input.view(),
                    &mut c.view_mut(),
                    &[],
                )?;
            }
        }
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl TypedOp for LirMatMulUnary {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if let Some(f) = &self.fused_ops {
            let c_prefix_len =
                self.c_prefix_dim_and_stride.as_ref().map(|prefix| prefix.0.len()).unwrap_or(0);
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
        let mul = self
            .c_prefix_dim_and_stride
            .as_ref()
            .map(|c| c.0.iter().product())
            .unwrap_or(1)
            .to_dim();
        Ok(tvec!(
            (Cost::FMA(self.mmm.internal_type()), mul.maybe_mul(self.n())? * self.m() * self.k()),
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
                let m = self.m();
                if op.a.len() == m
                    && op.a.shape()[op.a.rank() - 1 - ((!self.c_trans) as usize)] == m
                {
                    if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::PerRowMul(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Add>() {
                        Some(tvec!(FusedSpec::PerRowAdd(op.a.clone().into_tensor())))
                    } else {
                        None
                    }
                } else if op.a.len() == 1 {
                    if op.mini_op.is::<ops::math::Max>() {
                        Some(tvec!(FusedSpec::Max(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Min>() {
                        Some(tvec!(FusedSpec::Min(op.a.clone().into_tensor())))
                    } else if op.mini_op.is::<ops::math::Mul>() {
                        Some(tvec!(FusedSpec::ScalarMul(op.a.clone().into_tensor())))
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
                        let shape = vec![
                            1;
                            self.c_prefix_dim_and_stride
                                .as_ref()
                                .map(|c| c.0.len())
                                .unwrap_or(0)
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
