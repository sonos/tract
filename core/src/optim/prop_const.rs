use tract_data::UndeterminedSymbol;

use crate::internal::*;
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
        for n in model.eval_order()? {
            let node = model.node(n);
            let (inputs, outputs) = model.node_facts(n)?;
            if node.op.is_stateless()
                && inputs.iter().all(|i| i.konst.is_some())
                && outputs.iter().any(|o| o.konst.is_none())
            {
                let inputs =
                    inputs.iter().map(|f| f.konst.clone().unwrap().into_tvalue()).collect();
                match node.op.eval_with_session(&SessionState::default(), inputs) {
                    Ok(res) => {
                        let mut patch = TypedModelPatch::default();
                        for (ix, output) in res.into_iter().enumerate() {
                            let mut name = node.name.clone();
                            if ix > 0 {
                                name = format!("{name}.{ix}");
                            }
                            let wire = patch.add_const(name, output.into_arc_tensor())?;
                            patch.shunt_outside(model, (n, ix).into(), wire)?;
                        }
                        return Ok(Some(patch))
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
        Ok(None)
    }
}
