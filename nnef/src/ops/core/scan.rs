use crate::ast;
use crate::deser::Value;
use crate::internal::*;
use crate::ser::*;
use tract_core::ops;
use tract_core::ops::scan::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::scan::Scan>(), ser_scan);
    registry.register_primitive(
        "tract_core_scan",
        &[
            TypeName::String.named("body"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),   // body param name
                TypeName::Scalar.tensor(), // input
                TypeName::Integer.spec(),  // axis
                TypeName::Integer.spec(),  // step
            ])
            .array()
            .named("scan"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),   // body param name
                TypeName::Scalar.tensor(), // input
            ])
            .array()
            .named("full"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),   // body param name
                TypeName::Scalar.tensor(), // initializer
                TypeName::String.spec(),   // body result name
            ])
            .array()
            .named("state"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),  // body param name
                TypeName::String.spec(),  // "all" or "last"
                TypeName::Integer.spec(), // axis
            ])
            .array()
            .named("output"),
            TypeName::Integer.spec()    // if present, assumes B is first axis in all inputs
                    .named("seq_length")
                    .default(-1),
            TypeName::Integer.spec().named("skip").default(0), // needed for pulse
        ],
        de_scan,
    );
}

fn ser_scan(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<Scan>().unwrap();
    let (mut body, body_tensors) = crate::ser::to_fragment_def(ast, &op.body)?;
    body.decl.id = format!("scan_body_{}", ast.fragments.len());
    let mut scan = vec![];
    let mut state = vec![];
    let mut full = vec![];
    let mut outputs = vec![];
    for (ix, input) in op.input_mapping.iter().enumerate() {
        let name = string(body.decl.parameters[ix].id.to_string());
        match input {
            InputMapping::Scan { slot, axis, chunk } => {
                scan.push(tuple_4(
                    name,
                    ast.mapping[&node.inputs[*slot]].as_ref().clone(),
                    numeric(axis),
                    numeric(chunk),
                ));
            }
            InputMapping::State { initializer } => {
                let initializer = match &initializer {
                    &StateInitializer::Value(v) => {
                        ast.konst_variable(format!("{}_state_init_{}", node.name, state.len()), v)
                    }
                    &StateInitializer::FromInput(slot) => ast.mapping[&node.inputs[*slot]].clone(),
                }
                .as_ref()
                .clone();
                let output: usize = op
                    .output_mapping
                    .iter()
                    .enumerate()
                    .filter(|(_ix, o)| o.state)
                    .nth(state.len())
                    .unwrap()
                    .0;
                state.push(tuple_3(
                    name,
                    initializer,
                    string(body.decl.results[output].id.clone()),
                ));
            }
            InputMapping::Full { slot } => {
                full.push(tuple_2(name, ast.mapping[&node.inputs[*slot]].as_ref().clone()))
            }
        }
    }
    for tensor in &body_tensors {
        let t = ast.konst_variable(&tensor.label, &tensor.value);
        full.push(tuple_2(string(&tensor.parameter_id), t.as_ref().clone()));
    }
    for slot in 0..node.outputs.len() {
        if let Some((r_ix, om)) =
            op.output_mapping.iter().enumerate().find(|(_ix, om)| om.full_slot == Some(slot))
        {
            outputs.push(tuple_3(
                string(body.decl.results[r_ix].id.clone()),
                string("full"),
                numeric(om.axis),
            ));
        } else if let Some((r_ix, om)) =
            op.output_mapping.iter().enumerate().find(|(_ix, om)| om.last_value_slot == Some(slot))
        {
            outputs.push(tuple_3(
                string(body.decl.results[r_ix].id.clone()),
                string("last"),
                numeric(om.axis),
            ));
        } else {
            bail!("output {} is unbound", slot);
        };
    }
    let invoke = invocation(
        "tract_core_scan",
        &[],
        &[
            ("body", string(&body.decl.id)),
            ("scan", array(scan)),
            ("full", array(full)),
            ("state", array(state)),
            ("output", array(outputs)),
            ("skip", numeric(op.skip)),
        ],
    );
    ast.fragments.insert(body.decl.id.clone(), body);
    Ok(Some(invoke))
}

fn de_scan(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let fragment_name: String = invocation.named_arg_as(builder, "body")?;
    let fragment = builder
        .proto_model
        .doc
        .fragments
        .iter()
        .find(|n| n.decl.id == fragment_name)
        .ok_or(format!("Cound not find fragment `{}'", fragment_name))?;
    let mut body = ModelBuilder::new(builder.framework, builder.proto_model);
    body.scopes.push(HashMap::new());
    let mut outer_inputs: TVec<OutletId> = tvec!();
    let mut input_mapping = vec![];
    let scan: TVec<(String, OutletId, usize, isize)> = invocation.named_arg_as(builder, "scan")?;
    let full: TVec<(String, OutletId)> = invocation.named_arg_as(builder, "full")?;
    let state: TVec<(String, OutletId, String)> = invocation.named_arg_as(builder, "state")?;
    for par in &fragment.decl.parameters {
        let (outer_input_wire, inner_fact) =
            if let Some((_, wire, axis, chunk)) = scan.iter().find(|s| s.0 == par.id) {
                input_mapping.push(InputMapping::Scan {
                    slot: outer_inputs.len(),
                    axis: *axis,
                    chunk: *chunk,
                });
                let mut fact = builder.model.outlet_fact(*wire)?.clone();
                fact.shape.set_dim(*axis, chunk.abs().to_dim())?;
                (*wire, fact)
            } else if let Some((_, wire)) = full.iter().find(|s| s.0 == par.id) {
                input_mapping.push(InputMapping::Full { slot: outer_inputs.len() });
                let fact = builder.model.outlet_fact(*wire)?.clone();
                (*wire, fact)
            } else if let Some((_, wire, _out)) = state.iter().find(|s| s.0 == par.id) {
                input_mapping.push(InputMapping::State {
                    initializer: StateInitializer::FromInput(outer_inputs.len()),
                });
                let fact = builder.model.outlet_fact(*wire)?.clone();
                (*wire, TypedFact::dt_shape(fact.datum_type, fact.shape)?)
            } else {
                bail!("Unbound body input parameter {}", par.id);
            };
        outer_inputs.push(outer_input_wire);
        body.scopes.last_mut().unwrap().insert(
            par.id.clone(),
            Value::Wire(body.model.add_source(par.id.to_string(), inner_fact)?),
        );
    }
    body.wire_body(fragment.body.as_deref().unwrap())?;
    let body_outputs: Vec<OutletId> = fragment
        .decl
        .results
        .iter()
        .map(|r| body.scopes.last().unwrap()[&r.id].to::<OutletId>(builder))
        .collect::<TractResult<Vec<OutletId>>>()?;
    body.model.set_output_outlets(&body_outputs)?;
    let outputs: TVec<(String, String, usize)> = invocation.named_arg_as(builder, "output")?;
    let mut output_mapping = vec![];
    for output_name in fragment.decl.results.iter().map(|o| &*o.id) {
        output_mapping.push(OutputMapping {
            chunk: 1, // FIXME
            full_dim_hint: None,
            full_slot: outputs
                .iter()
                .enumerate()
                .find(|(_, om)| om.0 == output_name && om.1 == "full")
                .map(|(ix, _om)| ix),
            last_value_slot: outputs
                .iter()
                .enumerate()
                .find(|(_, om)| om.0 == output_name && om.1 == "last")
                .map(|(ix, _om)| ix),
            axis: outputs
                .iter()
                .enumerate()
                .find(|(_, om)| om.0 == output_name && om.1 == "full")
                .map(|(_ix, om)| om.2)
                .unwrap_or(0),
            state: state.iter().any(|state| state.2 == output_name),
        });
    }
    let skip:usize = invocation.named_arg_as(builder, "skip")?;
    let op = Scan::new(body.model, input_mapping, output_mapping, None, skip)?;
    builder.wire(op, &*outer_inputs)
}
