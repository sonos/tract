use dyn_clone::clone_box;
use tract_itertools::Itertools;
use tract_linalg::block_quant::BlockQuantValue;

use crate::internal::*;
use crate::ops::array::Gather;
use crate::ops::einsum::EinSum;

use super::einsum::optimize::EinSumAnnotatedAsLinear;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Const(Arc<Tensor>, Option<Box<dyn OpaqueFact>>);

impl Const {
    pub fn new(tensor: Arc<Tensor>) -> TractResult<Const> {
        Self::new_with_opt_opaque_fact(tensor, None)
    }

    pub fn new_with_opaque_fact(
        tensor: Arc<Tensor>,
        fact: Box<dyn OpaqueFact>,
    ) -> TractResult<Const> {
        Self::new_with_opt_opaque_fact(tensor, Some(fact))
    }

    pub fn new_with_opt_opaque_fact(
        tensor: Arc<Tensor>,
        fact: Option<Box<dyn OpaqueFact>>,
    ) -> TractResult<Const> {
        ensure!(fact.is_some() == tensor.datum_type().is_opaque());
        Ok(Const(tensor, fact))
    }

    pub fn val(&self) -> &Arc<Tensor> {
        &self.0
    }

    pub fn opaque_fact(&self) -> Option<&dyn OpaqueFact> {
        self.1.as_deref()
    }
}

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for Const {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec![Arc::clone(&self.0).into_tvalue()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let fact = TypedFact::from(&self.0);
        if let Some(opaque) = &self.1 {
            Ok(tvec!(fact.with_opaque_fact(opaque.clone())))
        } else {
            Ok(tvec!(fact))
        }
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        anyhow::ensure!(io == InOut::Out(0));
        let mut new_tensor = self.0.clone().into_tensor();
        if change.change_tensor(&mut new_tensor, false).is_ok() {
            Ok(Some(AxisChangeConsequence {
                substitute_op: Some(Box::new(Const(new_tensor.into_arc_tensor(), self.1.clone()))),
                wire_changes: tvec!((io, change.clone())),
            }))
        } else {
            Ok(None)
        }
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Params(self.0.datum_type().unquantized()), self.0.len().into())))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = if self.0.datum_type() == TDim::datum_type() {
            let mut tensor = self.0.clone().into_tensor();
            for d in tensor.as_slice_mut::<TDim>()? {
                *d = d.eval(values);
            }
            Const(tensor.into_arc_tensor(), self.1.clone())
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[])
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let looks_like_weights = (self.0.datum_type().is_number() && self.0.rank() == 2)
            || (self.0.to_scalar::<Opaque>().is_ok_and(|opaque| opaque.is::<BlockQuantValue>()));
        if !looks_like_weights {
            return Ok(None);
        }
        let mut matmuls = vec![];
        for succ in &node.outputs[0].successors {
            let snode = model.node(succ.node);
            if let Some(gather) = snode.op_as::<Gather>() {
                if succ.slot != 0 || gather.axis != 0 {
                    return Ok(None);
                }
            } else if let Some(einsum) = snode.op_as::<EinSum>() {
                if let Some(linear) = EinSumAnnotatedAsLinear::from(model, snode, einsum)? {
                    matmuls.push(linear);
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }
        if matmuls.len() == 0 {
            return Ok(None);
        }

        ensure!(matmuls.iter().map(|linear| linear.m_axis.inputs[0][0]).all_equal());
        ensure!(matmuls.iter().map(|linear| linear.k_axis.inputs[0][0]).all_equal());

        let m_axis = matmuls[0].m_axis.inputs[0][0];
        let k_axis = matmuls[0].k_axis.inputs[0][0];
        let must_swap = m_axis == 1;

        let ops = tract_linalg::ops();
        let (choice,) = matmuls
            .iter()
            .map(|mm| mm.preferred_packing())
            .dedup_by(|a, b| a.same_as(&**b))
            .collect_tuple::<(_,)>()
            .unwrap_or_else(|| {
                let it = ops
                    .all_possible_packing(matmuls[0].weight_type.clone())
                    .min_by_key(|format| {
                        matmuls
                            .iter()
                            .map(|linear| linear.cost_for_weights(&**format))
                            .max()
                            .unwrap()
                    })
                    .unwrap();
                (clone_box(it),)
            });

        let packed = choice.prepare_tensor(&self.0, k_axis, m_axis).context("in prepare_tensor")?;
        let fact = clone_box(packed.opaque_fact());
        let opaque = Opaque(Arc::new(packed));
        let konst = Const(rctensor0(opaque), Some(fact));
        let mut patch = TypedModelPatch::new(format!("Packing {node} as {choice:?}"));
        let konst = patch.wire_node(&node.name, konst, &[])?;
        for succ in &node.outputs[0].successors {
            let succ_node = model.node(succ.node);
            let mut taps = patch.taps(model, &succ_node.inputs)?;
            taps[succ.slot] = konst[0];
            let new_op: Box<dyn TypedOp> = if let Some(gather) = succ_node.op_as::<Gather>() {
                let output_type = succ_node.outputs[0].fact.datum_type;
                Box::new(Gather { axis: gather.axis, output_type: Some(output_type) })
            } else if let Some(linear) = succ_node.op_as::<EinSum>() {
                let mut op = linear.clone();
                if must_swap {
                    op.axes
                        .iter_all_axes_mut()
                        .for_each(|axes| axes.inputs[0].iter_mut().for_each(|pos| *pos = 1 - *pos));
                }
                Box::new(op)
            } else {
                bail!("Unexpected op")
            };
            let replacement = patch.wire_node(&succ_node.name, new_op, &taps)?;
            patch.shunt_outside(model, succ.node.into(), replacement[0])?;
        }
        Ok(Some(patch))
    }
}
