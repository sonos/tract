use tract_data::UndeterminedSymbol;

use crate::internal::*;
use crate::ops::konst::Const;
use crate::ops::source::TypedSource;
use crate::optim::OptimizerSession;

#[derive(Clone, Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn reset(&mut self) -> TractResult<()> {
        return Ok(());
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        for node in model.eval_order()? {
            if model.node(node).op.is_stateless()
                && !model.node(node).op_is::<Const>()
                && !model.node(node).op_is::<TypedSource>()
            {
                if let Some(inputs) =
                    model.node_input_facts(node)?.iter().map(|f| f.konst.clone()).collect()
                {
                    match model.node(node).op.eval(inputs) {
                        Ok(res) => {
                            for (ix, output) in res.into_iter().enumerate() {
                                let wire = patch.add_const(
                                    format!("{}.{}", model.node(node).name, ix),
                                    output,
                                )?;
                                patch.shunt_outside(model, (node, ix).into(), wire)?;
                            }
                            return Ok(Some(patch));
                        }
                        Err(e) => {
                            if !e.root_cause().is::<UndeterminedSymbol>() {
                                Err(e).with_context(|| {
                                    format!("Eager eval {} during optimisation", model.node(node))
                                })?;
                            }
                        }
                    }
                }
            }
        }
        for node in model.eval_order()? {
            trace!("Cleanup inputs for {}", model.node(node));
            for i in 0..model.node(node).inputs.len() {
                if let Some(k) = model.node_input_facts(node)?[i].konst.clone() {
                    let outlet = model.node(node).inputs[i];
                    let knode = model.node(outlet.node);
                    if knode.op_is::<Const>() {
                        continue;
                    }
                    let k = patch.add_const(knode.name.to_string() + ".konst", k)?;
                    patch.shunt_outside(model, outlet, k)?;
                }
            }
        }
        Ok(Some(patch).filter(|p| p.nodes.len() > 0))
    }
}
