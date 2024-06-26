use tract_data::UndeterminedSymbol;

use crate::internal::*;
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
            let inputs = model.node_input_facts(node.id)?;
            if node.op_is::<Const>() && node.outputs[0].fact.konst.is_none() {
                self.0 = node.id;
                let mut patch = TypedModelPatch::default();
                let wire = patch.add_const(&node.name, node.op_as::<Const>().unwrap().0.clone())?;
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch))
            }
            if !node.op_is::<Const>()
                && !node.op_is::<Dummy>()
                && !node.op_is::<TypedSource>()
                && node.op.is_stateless()
                && inputs.iter().all(|i| i.konst.is_some())
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
                            patch.shunt_outside(model, (node.id, ix).into(), wire)?;
                        }
                        self.0 = node.id;
                        return Ok(Some(patch));
                    }
                    Err(e) => {
                        if !e.root_cause().is::<UndeterminedSymbol>() {
                            Err(e).with_context(|| {
                                format!("Eager eval {} during optimisation", node)
                            })?;
                        }
                    }
                }
            }
        }
        Ok(None)
    }
}
