use crate::ast;
use crate::ast::Identifier;
use crate::deser::Value;
use crate::internal::*;
use crate::ser::*;
use tract_core::ops::scan::*;
use tract_itertools::Itertools;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_scan);
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
                TypeName::String.spec(),  // "full" or "last"
                TypeName::Integer.spec(), // axis (ignored for last)
                TypeName::Integer.spec(), // step (ignored for last)
            ])
            .array()
            .named("output"),
            TypeName::Integer.spec().named("skip").default(0), // needed for pulse
            TypeName::Integer.spec().named("reset_every_turn").default(0), // needed for pulse
        ],
        &[("outputs", TypeName::Scalar.tensor().array())],
        de_scan,
    );
}

fn ser_scan(ast: &mut IntoAst, node: &TypedNode, op: &Scan) -> TractResult<Option<Arc<RValue>>> {
    let (mut body, body_tensors) = crate::ser::to_fragment_def(ast, &op.body)?;
    body.decl.id = Identifier(format!("scan_body_{}", ast.fragments.len()));
    let mut scan = vec![];
    let mut state = vec![];
    let mut full = vec![];
    let mut outputs = vec![];
    for (slot, input) in op.input_mapping.iter().enumerate() {
        let name = string(&body.decl.parameters[slot].id.0);
        match input {
            InputMapping::Scan(info) => {
                scan.push(tuple_4(
                    name,
                    ast.mapping[&node.inputs[slot]].as_ref().clone(),
                    numeric(info.axis),
                    numeric(info.chunk),
                ));
            }
            InputMapping::State => {
                let initializer = (*ast.mapping[&node.inputs[slot]]).clone();
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
            InputMapping::Full => {
                full.push(tuple_2(name, ast.mapping[&node.inputs[slot]].as_ref().clone()))
            }
        }
    }
    for tensor in body_tensors.iter().sorted_by_key(|t| &t.label) {
        let t = ast.konst_variable(&tensor.label, &tensor.value)?;
        full.push(tuple_2(string(&tensor.parameter_id), t.as_ref().clone()));
    }
    for slot in 0..node.outputs.len() {
        if let Some((r_ix, om)) = op
            .output_mapping
            .iter()
            .enumerate()
            .find(|(_ix, om)| om.scan.map(|s| s.0) == Some(slot))
        {
            outputs.push(tuple_4(
                string(body.decl.results[r_ix].id.clone()),
                string("full"),
                numeric(om.scan.unwrap().1.axis),
                numeric(om.scan.unwrap().1.chunk),
            ));
        } else if let Some((r_ix, _om)) =
            op.output_mapping.iter().enumerate().find(|(_ix, om)| om.last_value_slot == Some(slot))
        {
            outputs.push(tuple_4(
                string(body.decl.results[r_ix].id.clone()),
                string("last"),
                numeric(0),
                numeric(1),
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
            ("reset_every_turn", numeric(op.reset_every_turn)),
        ],
    );
    ast.fragments.insert(body.decl.id.clone(), body);
    Ok(Some(invoke))
}

fn de_scan(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let fragment_name: String = invocation.named_arg_as(builder, "body")?;
    let fragment = builder
        .proto_model
        .doc
        .fragments
        .iter()
        .find(|n| n.decl.id.0 == fragment_name)
        .ok_or_else(|| format_err!("Cound not find fragment `{}'", fragment_name))?;
    let template = TypedModel { symbols: builder.model.symbols.clone(), ..TypedModel::default() };
    let mut body = ModelBuilder::new(builder.framework, builder.proto_model, template);
    body.scopes.push(HashMap::new());
    body.naming_scopes.clone_from(&builder.naming_scopes);
    body.registries.clone_from(&builder.registries);
    let mut outer_inputs: TVec<OutletId> = tvec!();
    let mut input_mapping = vec![];
    let scan: TVec<(String, OutletId, usize, isize)> = invocation.named_arg_as(builder, "scan")?;
    let full: TVec<(String, OutletId)> = invocation.named_arg_as(builder, "full")?;
    let state: TVec<(String, OutletId, String)> = invocation.named_arg_as(builder, "state")?;
    for par in &fragment.decl.parameters {
        let (outer_input_wire, inner_fact) = if let Some((_, wire, axis, chunk)) =
            scan.iter().find(|s| s.0 == par.id.0 || escape(&s.0) == par.id.0)
        {
            input_mapping.push(InputMapping::Scan(ScanInfo { axis: *axis, chunk: *chunk }));
            let mut fact = builder.model.outlet_fact(*wire)?.clone();
            fact.shape.set(*axis, chunk.abs().to_dim());
            (*wire, fact)
        } else if let Some((_, wire)) =
            full.iter().find(|s| s.0 == par.id.0 || escape(&s.0) == par.id.0)
        {
            input_mapping.push(InputMapping::Full);
            let fact = builder.model.outlet_fact(*wire)?.clone();
            (*wire, fact)
        } else if let Some((_, wire, _out)) =
            state.iter().find(|s| s.0 == par.id.0 || escape(&s.0) == par.id.0)
        {
            let fact = builder.model.outlet_fact(*wire)?.clone();
            input_mapping.push(InputMapping::State);
            (*wire, fact.datum_type.fact(fact.shape))
        } else {
            bail!("Unbound body input parameter {}", par.id.0);
        };
        outer_inputs.push(outer_input_wire);
        body.scopes.last_mut().unwrap().insert(
            par.id.clone(),
            Value::Wire(body.model.add_source(par.id.0.to_string(), inner_fact)?),
        );
    }
    body.wire_body(fragment.body.as_deref().unwrap()).context("wiring scan body")?;
    let body_outputs = fragment
        .decl
        .results
        .iter()
        .map(|r| {
            body.scopes.last().unwrap().get(&r.id).with_context(|| {
                format!("Could not find variable for scan output named `{}'", r.id.0)
            })
        })
        .collect::<TractResult<Vec<&Value>>>()
        .context("Finding output in body")?;

    let body_outputs: Vec<OutletId> = body_outputs
        .iter()
        .map(|v| v.to::<OutletId>(builder))
        .collect::<TractResult<Vec<OutletId>>>()
        .context("Coercing outputs to wires")?;
    body.model.set_output_outlets(&body_outputs)?;
    // preferred form for output is 4 arguments, but early models had 3 arguments output,
    // breaking support for bidirectional
    // this awkward dance somewhat maintains compatibility
    let outputs: TVec<(String, String, usize, isize)> = if let Ok(tuples_4) =
        invocation.named_arg_as(builder, "output")
    {
        tuples_4
    } else {
        let outputs: TVec<(String, String, usize)> = invocation.named_arg_as(builder, "output")?;
        outputs.into_iter().map(|(a, b, c)| (a, b, c, 1)).collect()
    };
    for output in &outputs {
        if output.1 != "full" && output.1 != "last" {
            bail!(
                "output named `{}' must specify type \"full\" or \"last\", found `{}'",
                output.0,
                output.1
            )
        }
    }
    let mut output_mapping = vec![];
    for output_name in fragment.decl.results.iter().map(|o| &*o.id.0) {
        output_mapping.push(OutputMapping {
            full_dim_hint: None,
            scan: outputs
                .iter()
                .enumerate()
                .find(|(_, om)| {
                    (om.0 == output_name || escape(&om.0) == output_name) && om.1 == "full"
                })
                .map(|(ix, om)| (ix, ScanInfo { axis: om.2, chunk: om.3 })),
            last_value_slot: outputs
                .iter()
                .enumerate()
                .find(|(_, om)| {
                    (om.0 == output_name || escape(&om.0) == output_name) && om.1 == "last"
                })
                .map(|(ix, _om)| ix),
            state: state
                .iter()
                .any(|state| state.2 == output_name || escape(&state.2) == output_name),
        });
    }
    let skip: usize = invocation.named_arg_as(builder, "skip")?;
    let mut op = Scan::new(body.model, input_mapping, output_mapping, skip)?;
    op.reset_every_turn = invocation.named_arg_as(builder, "reset_every_turn")?;
    builder.wire(op, &outer_inputs)
}

fn escape(it: &str) -> String {
    let mut escaped = String::new();
    let first = it.chars().next().unwrap();
    if !(first.is_alphabetic() || first == '_') {
        escaped.push('_');
    }
    escaped.extend(it.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }));
    escaped
}
