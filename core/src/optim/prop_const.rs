use tract_data::UndeterminedSymbol;

use crate::internal::*;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;

#[derive(Clone, Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        for n in model.eval_order()? {
            let node = model.node(n);
            // bit of a hack here. pulse erase const from fact, so this little dance puts it back
            if node.op_is::<Const>() && node.outputs[0].fact.konst.is_none() {
                let k = node.op_as::<Const>().unwrap().0.clone();
                let wire = patch.add_const(&node.name, k)?;
                patch.shunt_outside(model, node.id.into(), wire)?;
            }
            if node.op.is_stateless() && !node.op_is::<Const>() {
                if let Some(inputs) = model
                    .node_input_facts(n)?
                    .iter()
                    .map(|f| f.konst.clone().map(|t| t.into_tvalue()))
                    .collect()
                {
                    match node.op.eval_with_session(&SessionState::default(), inputs) {
                        Ok(res) => {
                            for (ix, output) in res.into_iter().enumerate() {
                                let mut name = node.name.clone();
                                if ix > 0 {
                                    name = format!("{name}.{ix}");
                                }
                                let wire = patch.add_const(name, output.into_arc_tensor())?;
                                patch.shunt_outside(model, (n, ix).into(), wire)?;
                            }
                        }
                        Err(e) => {
                            if !e.root_cause().is::<UndeterminedSymbol>() {
                                Err(e).with_context(|| {
                                    format!("Eager eval {} during optimisation", model.node(n))
                                })?;
                            }
                        }
                    }
                }
            }
        }
        Ok(Some(patch).filter(|p| p.nodes.len() > 0))
    }
}
