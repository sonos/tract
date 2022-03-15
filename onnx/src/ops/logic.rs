use crate::model::OnnxOpRegister;
use crate::model::ParseResult;
use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_core::ops;
use tract_itertools::Itertools;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(ops::logic::not()), vec![])));
    reg.insert("And", |_, _| Ok((ops::logic::And.into_hir(), vec![])));
    reg.insert("Or", |_, _| Ok((ops::logic::Or.into_hir(), vec![])));
    reg.insert("Xor", |_, _| Ok((ops::logic::Xor.into_hir(), vec![])));

    reg.insert("Equal", |_, _| Ok((ops::logic::Equals.into_hir(), vec![])));
    reg.insert("Greater", |_, _| Ok((ops::logic::Greater.into_hir(), vec![])));
    reg.insert("Less", |_, _| Ok((ops::logic::Less.into_hir(), vec![])));
    reg.insert("LessOrEqual", |_, _| Ok((ops::logic::LessEqual.into_hir(), vec![])));
    reg.insert("GreaterOrEqual", |_, _| Ok((ops::logic::GreaterEqual.into_hir(), vec![])));

    reg.insert("Where", |_, _| Ok((expand(tract_hir::ops::logic::Iff), vec![])));

    reg.insert("If", _if)
}

pub fn _if(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let graph_then = node.get_attr("then_branch")?;
    let graph_else = node.get_attr("else_branch")?;
    let ParseResult { model: then_body, unresolved_inputs: unresolved_inputs_then, .. } =
        ctx.parse_graph(graph_then)?;
    let ParseResult { model: else_body, unresolved_inputs: unresolved_inputs_else, .. } =
        ctx.parse_graph(graph_else)?;
    let unresolved_inputs: Vec<String> = unresolved_inputs_then
        .iter()
        .chain(unresolved_inputs_else.iter())
        .sorted()
        .unique()
        .cloned()
        .collect();
    let then_input_mapping = unresolved_inputs_then
        .iter()
        .map(|i| unresolved_inputs.iter().position(|s| s == i).unwrap() + 1)
        .collect();
    let else_input_mapping = unresolved_inputs_else
        .iter()
        .map(|i| unresolved_inputs.iter().position(|s| s == i).unwrap() + 1)
        .collect();
    Ok((
        Box::new(If { then_body, then_input_mapping, else_body, else_input_mapping }),
        unresolved_inputs,
    ))
}

#[derive(Debug, Clone, new, Hash)]
struct If {
    then_body: InferenceModel,
    then_input_mapping: Vec<usize>,
    else_body: InferenceModel,
    else_input_mapping: Vec<usize>,
}

impl_dyn_hash!(If);

impl Op for If {
    fn name(&self) -> Cow<str> {
        "If".into()
    }

    op_onnx!();
    not_a_typed_op!();
}

impl EvalOp for If {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let cond = inputs[0].cast_to_scalar::<bool>()?;
        let (input_mapping, body) = if cond {
            (&self.then_input_mapping, &self.then_body)
        } else {
            (&self.else_input_mapping, &self.else_body)
        };
        let inputs: TVec<Tensor> =
            input_mapping.iter().map(|&ix| inputs[ix].clone().into_tensor()).collect();
        body.clone().into_runnable()?.run(inputs)
    }
}

impl InferenceOp for If {
    fn infer_facts(
        &mut self,
        inputs: TVec<&InferenceFact>,
        outputs: TVec<&InferenceFact>,
        observed: TVec<&InferenceFact>,
    ) -> TractResult<(TVec<InferenceFact>, TVec<InferenceFact>, TVec<InferenceFact>)> {
        let mut inputs: TVec<InferenceFact> = inputs.into_iter().cloned().collect();
        let mut outputs: TVec<InferenceFact> = outputs.into_iter().cloned().collect();
        loop {
            let mut changed = false;
            changed = changed || inputs[0].datum_type.unify_with(&bool::datum_type().into())?;
            for (body_ix, outer_ix) in self.then_input_mapping.iter().enumerate() {
                changed = changed
                    || self
                        .then_body
                        .input_fact_mut(body_ix)?
                        .unify_with_mut(&mut inputs[*outer_ix])?;
            }
            for (body_ix, outer_ix) in self.else_input_mapping.iter().enumerate() {
                changed = changed
                    || self
                        .else_body
                        .input_fact_mut(body_ix)?
                        .unify_with_mut(&mut inputs[*outer_ix])?;
            }
            if let Some(a) = inputs[0].value.concretize() {
                let a = a.cast_to_scalar()?;
                let body = if a { &mut self.then_body } else { &mut self.else_body };
                for oix in 0..body.output_outlets()?.len() {
                    changed =
                        changed || body.output_fact_mut(oix)?.unify_with_mut(&mut outputs[oix])?;
                }
            }
            changed = changed || self.then_body.analyse(false)?;
            changed = changed || self.else_body.analyse(false)?;
            if !changed {
                return Ok((inputs, outputs, observed.into_iter().cloned().collect()));
            }
        }
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(cond) = &target.outlet_fact(mapping[&node.inputs[0]])?.konst {
            let cond = cond.cast_to_scalar::<bool>()?;
            let (body, input_mapping) = if cond {
                (&self.then_body, &self.then_input_mapping)
            } else {
                (&self.else_body, &self.else_input_mapping)
            };
            let mut inner_mapping: HashMap<OutletId, OutletId> = HashMap::default();
            let body = body.clone().into_typed()?;
            for (input_ix, outlet) in tract_itertools::izip!(input_mapping, body.input_outlets()?) {
                inner_mapping.insert(*outlet, mapping[&node.inputs[*input_ix]]);
            }
            for node in body.eval_order()? {
                if Graph::is_source(&body.node(node).op) {
                    continue;
                }
                let node_inputs =
                    body.node(node).inputs.iter().map(|o| inner_mapping[o]).collect::<TVec<_>>();
                let node_outputs =
                    target.wire_node(&body.node(node).name, &body.node(node).op, &node_inputs)?;
                for (slot_ix, outlet) in node_outputs.iter().enumerate() {
                    inner_mapping.insert((node, slot_ix).into(), *outlet);
                }
            }
            return Ok(body.output_outlets()?.iter().map(|o| inner_mapping[o]).collect());
        }
        bail!("Can only deal with constant conditions in If translation")
    }

    as_op!();
}
