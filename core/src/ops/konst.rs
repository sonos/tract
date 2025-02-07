use dyn_clone::clone_box;
use tract_linalg::block_quant::BlockQuantValue;
use tract_linalg::kit::WeightType;

use crate::internal::*;
use crate::ops::array::Gather;
use crate::ops::einsum::EinSum;

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
                substitute_op: Some(Box::new(Const(
                    new_tensor.into_arc_tensor(),
                    self.1.clone(),
                ))),
                wire_changes: tvec!((io, change.clone())),
            }))
        } else {
            Ok(None)
        }
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((
            Cost::Params(self.0.datum_type().unquantized()),
            self.0.len().into()
        )))
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
            || (self
                .0
                .to_scalar::<Opaque>()
                .is_ok_and(|opaque| opaque.is::<BlockQuantValue>()));
        if !looks_like_weights {
            return Ok(None);
        }
        let mut have_abstract_einsum = false;
        for succ in &node.outputs[0].successors {
            let snode = model.node(succ.node);
            if let Some(gather) = snode.op_as::<Gather>() {
                if succ.slot != 0 || gather.axis != 0 {
                    return Ok(None);
                }
            } else if let Some(einsum) = snode.op_as::<EinSum>() {
                if succ.slot != 0 || snode.inputs.len() != 2 {
                    return Ok(None);
                }
                let m_axis = einsum.axes.axis((InOut::In(0), 0))?;
                if m_axis.inputs[0].len() != 1
                    || m_axis.inputs[1].len() != 0
                    || m_axis.outputs[0].len() != 1
                {
                    return Ok(None);
                }
                let k_axis = einsum.axes.axis((InOut::In(0), 1))?;
                if k_axis.inputs[0].len() != 1
                    || k_axis.inputs[1].len() != 1
                    || k_axis.outputs[0].len() != 0
                {
                    return Ok(None);
                }
                for axis in einsum.axes.iter_all_axes() {
                    if axis != k_axis
                        && axis != m_axis
                        && axis.inputs[0].len() == 0
                        && axis.inputs[1].len() == 1
                        && axis.outputs[0].len() == 1
                        && snode.outputs[0].fact.shape[axis.outputs[0][0]]
                            .as_i64()
                            .is_none()
                    {
                        have_abstract_einsum = true;
                    }
                }
            } else {
                return Ok(None);
            }
        }
        if node.outputs[0].successors.len() > 1 || have_abstract_einsum {
            let weight = self
                .0
                .to_scalar::<Opaque>()
                .ok()
                .and_then(|a| a.downcast_ref::<BlockQuantValue>());
            let weight_type = if let Some(a_payload) = weight {
                WeightType::BlockQuant(a_payload.fact.format.clone())
            } else {
                WeightType::Plain(self.0.datum_type())
            };
            let format = tract_linalg::ops().kit_input_format(weight_type);
            let packed = format.prepare_tensor(&self.0, 1, 0)?;
            let fact = clone_box(packed.opaque_fact());
            let opaque = Opaque(Arc::new(packed));
            let konst = Const(rctensor0(opaque), Some(fact));
            let mut patch = TypedModelPatch::new(format!("Versatile packing {node}"));
            let konst = patch.wire_node(&node.name, konst, &[])?;
            for succ in &node.outputs[0].successors {
                let succ_node = model.node(succ.node);
                let mut taps = patch.taps(model, &succ_node.inputs)?;
                taps[succ.slot] = konst[0];
                let replacement = patch.wire_node(&succ_node.name, succ_node.op.clone(), &taps)?;
                patch.shunt_outside(model, succ.node.into(), replacement[0])?;
            }
            return Ok(Some(patch));
        }
        Ok(None)
    }
}
