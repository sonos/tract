use crate::internal::*;

#[derive(Debug, Clone, Default)]
pub struct IfThenElse {
    pub then_body: TypedModel,
    pub then_input_mapping: Vec<usize>,
    pub else_body: TypedModel,
    pub else_input_mapping: Vec<usize>,
}

impl Op for IfThenElse {
    fn name(&self) -> Cow<str> {
        "IfThenElse".into()
    }

    op_as_typed_op!();
}

impl TypedOp for IfThenElse {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs[0].datum_type == bool::datum_type());
        ensure!(inputs[0].shape.volume() == 1.to_dim());
        ensure!(self.then_body.inputs.len() == self.then_input_mapping.len());
        ensure!(self.else_body.inputs.len() == self.else_input_mapping.len());
        let mut facts = tvec!();
        for i in 0..self.then_body.outputs.len() {
            ensure!(
                self.then_body.output_fact(i)?.without_value()
                    == self.else_body.output_fact(i)?.without_value()
            );
            facts.push(self.then_body.output_fact(i)?.clone());
        }
        Ok(facts)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(cond) = &model.outlet_fact(node.inputs[0])?.konst {
            let cond = cond.cast_to_scalar::<bool>()?;
            let (body, input_mapping) = if cond {
                (&self.then_body, &self.then_input_mapping)
            } else {
                (&self.else_body, &self.else_input_mapping)
            };
            let mut inner_mapping: HashMap<OutletId, OutletId> = HashMap::default();
            let mut patch = TypedModelPatch::default();
            for (input_ix, outlet) in tract_itertools::izip!(input_mapping, body.input_outlets()?) {
                let tap = patch.tap_model(model, node.inputs[*input_ix])?;
                inner_mapping.insert(*outlet, tap);
            }
            for node in body.eval_order()? {
                if Graph::is_source(&body.node(node).op) {
                    continue;
                }
                let node_inputs =
                    body.node(node).inputs.iter().map(|o| inner_mapping[o]).collect::<TVec<_>>();
                let node_outputs =
                    patch.wire_node(&body.node(node).name, &body.node(node).op, &node_inputs)?;
                for (slot_ix, outlet) in node_outputs.iter().enumerate() {
                    inner_mapping.insert((node, slot_ix).into(), *outlet);
                }
            }
            for (ix, output) in body.outputs.iter().enumerate() {
                patch.shunt_outside(model, OutletId::new(node.id, ix), inner_mapping[output])?;
            }
            Ok(Some(patch))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl EvalOp for IfThenElse {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let cond = inputs[0].cast_to_scalar::<bool>()?;
        let (input_mapping, body) = if cond {
            (&self.then_input_mapping, &self.then_body)
        } else {
            (&self.else_input_mapping, &self.else_body)
        };
        let inputs: TVec<TValue> = input_mapping.iter().map(|&ix| inputs[ix].clone()).collect();
        body.clone().into_runnable()?.run(inputs)
    }
}
