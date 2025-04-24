use tract_data::TooEarly;

use crate::internal::*;
use crate::ops::array::Slice;
use crate::ops::dummy::Dummy;
use crate::ops::konst::Const;
use crate::ops::source::TypedSource;
use crate::optim::OptimizerSession;

#[derive(Clone, Debug, Default)]
pub struct PropConst(usize);

impl super::TypedPass for PropConst {
    fn reset(&mut self) -> TractResult<()> {
        self.0 = 0;
        Ok(())
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        for node in &model.nodes[self.0..] {
            if node.op_is::<Const>() && node.outputs[0].fact.konst.is_none() {
                self.0 = node.id;
                let mut patch = TypedModelPatch::default();
                let wire =
                    patch.add_const(&node.name, node.op_as::<Const>().unwrap().val().clone())?;
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
            let inputs = model.node_input_facts(node.id)?;
            if !node.op_is::<Const>()
                && !node.op_is::<Dummy>()
                && !node.op_is::<TypedSource>()
                && node.op.is_stateless()
                && inputs.iter().zip(&node.inputs).all(|(fact, outlet)| {
                    fact.konst.is_some()
                        && (model.node(outlet.node).outputs[outlet.slot].successors.len() == 1
                            || node.op_is::<Slice>()
                            || (fact.datum_type.is_number()
                                && fact.shape.volume().as_i64().is_some_and(|d| d < 1024)))
                })
            {
                let inputs =
                    inputs.iter().map(|f| f.konst.clone().unwrap().into_tvalue()).collect();
                match node.op.eval_with_session(&SessionState::default(), inputs) {
                    Ok(mut res) => {
                        self.0 = node.id;
                        let mut node = node;
                        loop {
                            let Some(succ) = model.single_succ(node.id)? else {
                                break;
                            };
                            if succ.inputs.len() > 1 || !succ.op.is_stateless() {
                                break;
                            }
                            let Ok(succ_res) =
                                succ.op.eval_with_session(&SessionState::default(), res.clone())
                            else {
                                break;
                            };
                            res = succ_res;
                            node = succ;
                        }
                        let mut patch = TypedModelPatch::default();
                        for (ix, output) in res.into_iter().enumerate() {
                            let opaque_fact =
                                model.outlet_fact(OutletId::new(node.id, ix))?.opaque_fact.clone();

                            let name = if ix > 0 {
                                format!("{}.{ix}", node.name)
                            } else {
                                node.name.clone()
                            };
                            let wire = patch.wire_node(
                                name,
                                Const::new_with_opt_opaque_fact(
                                    output.into_arc_tensor(),
                                    opaque_fact,
                                )?,
                                &[],
                            )?[0];
                            patch.shunt_outside(model, (node.id, ix).into(), wire)?;
                        }
                        self.0 = node.id;
                        return Ok(Some(patch));
                    }
                    Err(e) => {
                        if !e.root_cause().is::<TooEarly>() {
                            Err(e).with_context(|| {
                                format!("Eager eval {node} during optimisation")
                            })?;
                        }
                    }
                }
            }
        }
        Ok(None)
    }
}
