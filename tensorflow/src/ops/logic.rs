use crate::model::ParsingContext;
use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;
use std::collections::HashSet;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ops as tractops;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("Equal", |_, _| Ok(Box::new(tractops::logic::equals::bin())));
    reg.insert("Greater", |_, _| Ok(Box::new(tractops::logic::greater::bin())));
    reg.insert("GreaterEqual", |_, _| Ok(Box::new(tractops::logic::greater_equal::bin())));
    reg.insert("Less", |_, _| Ok(Box::new(tractops::logic::lesser::bin())));
    reg.insert("LessEqual", |_, _| Ok(Box::new(tractops::logic::lesser_equal::bin())));
    reg.insert("LogicalAnd", |_, _| Ok(Box::new(tractops::logic::and::bin())));
    reg.insert("LogicalOr", |_, _| Ok(Box::new(tractops::logic::or::bin())));
    reg.insert("Merge", merge);
    reg.insert("Switch", |_, _| Ok(Box::new(Switch)));
}

#[derive(Debug, Clone, new)]
pub struct Switch;

impl Op for Switch {
    fn name(&self) -> Cow<str> {
        "tf.Switch".into()
    }

    not_a_typed_op!();
}

impl StatefullOp for Switch {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }
}

impl InferenceRulesOp for Switch {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 2)?;
        s.equals(&inputs[1].datum_type, DatumType::Bool)?;
        s.equals(&inputs[1].shape, shapefactoid!())?;
        for i in 0..outputs.len() {
            s.equals(&inputs[0].datum_type, &outputs[i].datum_type)?;
            s.equals(&inputs[0].shape, &outputs[i].shape)?;
        }
        Ok(())
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        let pred = model.outlet_fact(node.inputs[1])?;
        if let Some(pred) = pred.concretize() {
            let pred = *pred.to_scalar::<bool>()?;
            let mut dead_to_visit = HashSet::new();
            let mut dead_done = HashSet::new();
            let mut patch = InferenceModelPatch::default();
            dead_to_visit.insert(OutletId::new(node.id, !pred as usize));
            while let Some(dead_outlet) = dead_to_visit.iter().cloned().next() {
                dead_to_visit.remove(&dead_outlet);
                dead_done.insert(dead_outlet);
                for succ in model.outlet_successors(dead_outlet) {
                    if model.node(succ.node).op_is::<Merge>() {
                        let outlet = model.node(succ.node).inputs[(succ.slot == 0) as usize];
                        let tap = patch.tap_model(model, outlet)?;
                        patch.shunt_outside(succ.node.into(), tap)?;
                    } else {
                        for slot in 0..model.node(succ.node).outputs.len() {
                            let new = OutletId::new(succ.node, slot);
                            if !dead_done.contains(&new) {
                                dead_to_visit.insert(new);
                            }
                        }
                    }
                }
            }
            let tap = patch.tap_model(model, node.inputs[0])?;
            patch.shunt_outside(OutletId::new(node.id, 0) , tap)?;
            patch.shunt_outside(OutletId::new(node.id, 1) , tap)?;
            return Ok(Some(patch))
        }
        Ok(None)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }

    as_op!();
}


fn merge(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let inputs = pb.get_attr_int::<i32>("N")?;
    Ok(Box::new(Merge::new(inputs as usize)))
}

#[derive(Debug, Clone, new)]
pub struct Merge {
    n: usize,
}

impl Op for Merge {
    fn name(&self) -> Cow<str> {
        "tf.Merge".into()
    }

    op_as_typed_op!();
}

impl StatefullOp for Merge {
    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }
}

impl InferenceRulesOp for Merge {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, self.n)?;
        check_output_arity(&outputs, 1)?;
        for i in 1..self.n {
            s.equals(&inputs[0].datum_type, &inputs[i].datum_type)?;
            s.equals(&inputs[0].shape, &inputs[i].shape)?;
        }
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Merge {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(
            TypedFact::dt_shape(f32::datum_type(), inputs[0].shape.clone())?,
            TypedFact::dt_shape(i32::datum_type(), ())?
        ))
    }
}
