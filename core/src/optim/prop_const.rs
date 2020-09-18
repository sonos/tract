use crate::model::*;
use crate::TractResult;

#[derive(Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn reset(&mut self) -> TractResult<()> {
        return Ok(());
    }
    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        for node in model.eval_order()? {
            trace!("Cleanup inputs for {}", model.node(node));
            for i in 0..model.node(node).inputs.len() {
                if let Some(k) = model.node_input_facts(node)?[i].konst.clone() {
                    let outlet = model.node(node).inputs[i];
                    let knode = model.node(outlet.node);
                    if knode.op_is::<crate::ops::konst::Const>() {
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
