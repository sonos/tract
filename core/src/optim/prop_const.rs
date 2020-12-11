use crate::internal::*;
use crate::ops::konst::Const;
use crate::TractResult;

#[derive(Clone, Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn reset(&mut self) -> TractResult<()> {
        return Ok(());
    }
    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        for node in model.eval_order()? {
            if model.node(node).op.is_stateless() && !model.node(node).op_is::<Const>() {
                if let Some(inputs) =
                    model.node_input_facts(node)?.iter().map(|f| f.konst.clone()).collect()
                {
                    let outputs = model
                        .node(node)
                        .op
                        .eval(inputs)
                        .context("Eager eval during optimisation")?;
                    for (ix, output) in outputs.into_iter().enumerate() {
                        let wire =
                            patch.add_const(format!("{}.{}", model.node(node).name, ix), output)?;
                        patch.shunt_outside(model, (node, ix).into(), wire)?;
                    }
                    return Ok(Some(patch));
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
                    let k = patch.add_const(&*knode.name, k)?;
                    patch.shunt_outside(model, outlet, k)?;
                }
            }
        }
        Ok(Some(patch).filter(|p| p.nodes.len() > 0))
    }
}
