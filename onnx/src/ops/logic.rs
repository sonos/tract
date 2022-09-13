use crate::model::OnnxOpRegister;
use crate::model::ParseResult;
use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops;
use tract_hir::internal::*;
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
    then_input_mapping: TVec<usize>,
    else_body: InferenceModel,
    else_input_mapping: TVec<usize>,
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

            Ok(body.output_outlets()?.iter().map(|o| inner_mapping[o]).collect())
        } else {
            target.wire_node(
                &node.name,
                IfMir {
                    then_body: self.then_body.clone().into_typed()?,
                    then_input_mapping: self.then_input_mapping.clone(),
                    else_body: self.else_body.clone().into_typed()?,
                    else_input_mapping: self.else_input_mapping.clone(),
                },
                &node.inputs,
            )
        }
    }

    as_op!();
}

/// Returns the output fact that is the result of the If control flow.
/// This could be thought of as the output fact of the Phi node of the Then and Else subgraphs,
/// (but it's arguably not as fancy as that.)
pub fn phi_result(then: &TypedFact, elze: &TypedFact) -> TractResult<TypedFact> {
    if then.konst.is_some() && elze.konst.is_some() && then.konst == elze.konst {
        return Ok(then.clone());
    }

    if then.datum_type != elze.datum_type {
        bail!(
            "If operator branches has incompatible datum types (then: {:?}; else: {:?})",
            then.datum_type,
            elze.datum_type
        )
    }

    if then.shape.rank() != elze.shape.rank() {
        bail!(
            "If operator branches has incompatible ranks (then: {:?}; else: {:?})",
            then.shape.rank(),
            elze.shape.rank()
        )
    }

    // [4, 'n', 18] . [4, 'k', 3] => [4, '?', '?']
    let shape: TVec<_> = then
        .shape
        .iter()
        .zip(elze.shape.iter())
        .map(|(then_dim, else_dim)| {
            let then_dim = then_dim.eval(&SymbolValues::default());
            if then_dim == else_dim.eval(&SymbolValues::default()) {
                then_dim
            } else {
                Symbol::new('h').to_dim()
            }
        })
        .collect();

    Ok(TypedFact::dt_shape(then.datum_type, shape))
}

#[derive(Debug, Clone, new, Hash)]
struct IfMir {
    then_body: TypedModel,
    then_input_mapping: TVec<usize>,
    else_body: TypedModel,
    else_input_mapping: TVec<usize>,
}

impl_dyn_hash!(IfMir);

impl Op for IfMir {
    fn name(&self) -> Cow<str> {
        "If".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EvalOp for IfMir {
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

impl TypedOp for IfMir {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let then_outputs =
            self.then_body.outputs.iter().copied().map(|outlet| self.then_body.outlet_fact(outlet));
        let else_outputs =
            self.else_body.outputs.iter().copied().map(|outlet| self.else_body.outlet_fact(outlet));

        let facts = then_outputs
            .zip(else_outputs)
            .map(|(tfact, efact)| {
                let (tfact, efact) = (tfact?.without_value(), efact?.without_value());
                phi_result(&tfact, &efact)
            })
            .collect();

        facts
    }
}
