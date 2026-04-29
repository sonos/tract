use crate::ast::parse::parse_assignments;
use crate::ast::*;
use crate::deser::{ModelBuilder, Value};
use tract_core::internal::*;
use tract_core::model::TypedModelPatch;
use tract_core::transform::{ModelTransform, ModelTransformFactory};

#[derive(Debug, serde::Deserialize)]
struct PatchConfig {
    body: String,
    #[serde(default)]
    new_outputs: Vec<String>,
}

#[derive(Debug)]
struct PatchTransform(PatchConfig);

impl ModelTransform for PatchTransform {
    fn name(&self) -> StaticName {
        "patch".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let assignments = parse_assignments(&self.0.body)
            .with_context(|| format!("Parsing wire statement: {:?}", self.0.body))?;

        let lhs_names: Vec<String> = assignments
            .iter()
            .filter_map(|a| match &a.left {
                LValue::Identifier(id) => Some(id.0.clone()),
                _ => None,
            })
            .collect();

        // Run the builder in a block so it (and its borrow of model) is dropped before we mutate model
        let (patch_model, taps, scope) = {
            let framework = crate::nnef().with_tract_core();

            let doc = Document {
                version: "1.0".into(),
                extension: vec![],
                fragments: vec![],
                graph_def: GraphDef {
                    id: Identifier("patch".into()),
                    parameters: vec![],
                    results: vec![],
                    body: vec![],
                },
            };
            let proto_model = ProtoModel {
                doc,
                tensors: Default::default(),
                quantization: None,
                resources: Default::default(),
            };

            let model_ref: &TypedModel = &*model;
            let mut taps = HashMap::<OutletId, OutletId>::default();

            let template = TypedModel { symbols: model.symbols.clone(), ..TypedModel::default() };
            let mut builder = ModelBuilder::new(&framework, &proto_model, template);
            builder.registries.push("tract_core".into());
            builder.scopes.push(HashMap::default());

            let taps_ref = &mut taps;
            builder.wire_resolver =
                Some(Box::new(move |name: &str, patch_model: &mut TypedModel| {
                    rule_if_some!(
                        node_id = model_ref.nodes.iter().find(|n| n.name == name).map(|n| n.id)
                    );
                    let original_outlet = OutletId::new(node_id, 0);
                    let fact = model_ref.outlet_fact(original_outlet)?.clone();
                    let patch_outlet = patch_model.add_source(name, fact)?;
                    taps_ref.insert(patch_outlet, original_outlet);
                    Ok(Some(patch_outlet))
                }));

            builder.wire_body(&assignments).with_context(|| "Executing wire statements")?;

            let scope = builder.scopes.last().unwrap().clone();
            builder.wire_resolver.take();
            let patch_model = std::mem::take(&mut builder.model);
            drop(builder);
            (patch_model, taps, scope)
        };

        // Build the TypedModelPatch
        let mut patch = TypedModelPatch { model: patch_model, taps, ..TypedModelPatch::default() };

        let mut inputs_to_remove = vec![];
        let mut new_output_names = vec![];

        for (i, lhs_name) in lhs_names.iter().enumerate() {
            let patch_outlet = match scope.get(&Identifier(lhs_name.clone())) {
                Some(Value::Wire(o)) => *o,
                _ => continue,
            };

            let is_model_input =
                model.inputs.iter().find(|&&inp| model.node(inp.node).name == *lhs_name).copied();

            if let Some(input_outlet) = is_model_input {
                let expected_fact = model.outlet_fact(input_outlet)?;
                let mut wire = patch_outlet;
                let patch_fact = patch.model.outlet_fact(wire)?.clone();
                // Cast dtype if needed (e.g. TDim → I64)
                if patch_fact.datum_type != expected_fact.datum_type {
                    wire = patch.model.wire_node(
                        format!("{}_cast", lhs_name),
                        tract_core::ops::cast::cast(expected_fact.datum_type),
                        &[wire],
                    )?[0];
                }
                // Reshape if needed (e.g. scalar → [1])
                let wire_fact = patch.model.outlet_fact(wire)?.clone();
                if wire_fact.shape != expected_fact.shape {
                    wire = patch.model.wire_node(
                        format!("{}_reshape", lhs_name),
                        tract_core::ops::change_axes::AxisOp::Reshape(
                            0,
                            wire_fact.shape.to_tvec(),
                            expected_fact.shape.to_tvec(),
                        ),
                        &[wire],
                    )?[0];
                }
                patch.shunt_outside(model, input_outlet, wire)?;
                inputs_to_remove.push(input_outlet);
            } else if self.0.new_outputs.contains(lhs_name) {
                new_output_names.push(lhs_name.clone());
            } else {
                let is_intermediate = i < lhs_names.len() - 1;
                if !is_intermediate {
                    bail!(
                        "Wire '{}' is not a model input and not declared in new_outputs",
                        lhs_name
                    );
                }
            }
        }

        patch.apply(model)?;

        for inp in &inputs_to_remove {
            model.inputs.retain(|o| o != inp);
        }
        for name in &new_output_names {
            let node_id = model.node_id_by_name(name)?;
            model.outputs.push(OutletId::new(node_id, 0));
        }

        model.refresh_output_facts()?;
        Ok(())
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "patch",
        build_default: || {
            bail!("patch transform requires a 'body' parameter, e.g. patch(body: \"x = some_op(y);\")")
        },
        build: |de| {
            let config: PatchConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow!("deserializing patch config: {e}"))?;
            Ok(Box::new(PatchTransform(config)))
        },
    }
}
