use crate::params::SomeGraphDef;
use crate::tensor::run_params_from_subcommand;
use crate::Parameters;
use ansi_term::Style;
use tract_hir::internal::*;
use tract_libcli::annotations::*;
use tract_libcli::display_params::*;
use tract_libcli::model::Model;
use tract_libcli::profile::BenchLimits;
use tract_libcli::terminal;

#[allow(unused_variables)]
pub fn annotate_with_graph_def(
    annotations: &mut Annotations,
    model: &dyn Model,
    graph_def: &SomeGraphDef,
) -> TractResult<()> {
    match graph_def {
        SomeGraphDef::NoGraphDef => Ok(()),
        #[cfg(feature = "kaldi")]
        SomeGraphDef::Kaldi(kaldi) => annotate_with_kaldi(annotations, model, kaldi),
        SomeGraphDef::Nnef(_) => todo!(),
        #[cfg(feature = "onnx")]
        SomeGraphDef::Onnx(onnx, _) => annotate_with_onnx_model(annotations, model, onnx),
        #[cfg(feature = "tf")]
        SomeGraphDef::Tf(tf) => annotate_with_tf_graph_def(annotations, model, tf),
    }
}

#[cfg(feature = "kaldi")]
fn annotate_with_kaldi(
    annotations: &mut Annotations,
    model: &dyn Model,
    proto_model: &tract_kaldi::KaldiProtoModel,
) -> TractResult<()> {
    use tract_kaldi::model::NodeLine;
    let bold = Style::new().bold();
    for (name, proto_node) in &proto_model.config_lines.nodes {
        if let Ok(node_id) = model.node_id_by_name(name) {
            let mut vs = vec![];
            if let NodeLine::Component(compo) = proto_node {
                let comp = &proto_model.components[&compo.component];
                for (k, v) in &comp.attributes {
                    let value = format!("{:?}", v);
                    vs.push(format!("Attr {}: {:.240}", bold.paint(k), value));
                }
            }
            annotations.node_mut(node_id.into()).sections.push(vs)
        }
    }
    Ok(())
}

#[cfg(feature = "tf")]
fn annotate_with_tf_graph_def(
    annotations: &mut Annotations,
    model: &dyn Model,
    graph_def: &tract_tensorflow::tfpb::tensorflow::GraphDef,
) -> TractResult<()> {
    let bold = Style::new().bold();
    for gnode in graph_def.node.iter() {
        if let Ok(node_id) = model.node_id_by_name(&gnode.name) {
            let mut v = vec![];
            for a in gnode.attr.iter() {
                let value =
                    if let Some(tract_tensorflow::tfpb::tensorflow::attr_value::Value::Tensor(r)) =
                        &a.1.value
                    {
                        format!("{:?}", r)
                    } else {
                        format!("{:?}", a.1)
                    };
                v.push(format!("Attr {}: {:.240}", bold.paint(a.0), value));
            }
            annotations.node_mut(node_id.into()).sections.push(v);
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn annotate_with_onnx_model(
    annotations: &mut Annotations,
    model: &dyn Model,
    model_proto: &tract_onnx::pb::ModelProto,
) -> TractResult<()> {
    let bold = Style::new().bold();
    for gnode in model_proto.graph.as_ref().unwrap().node.iter() {
        let mut node_name = &gnode.name;
        if !node_name.is_empty() && gnode.output.len() > 0 {
            node_name = &gnode.output[0];
        }
        if let Ok(id) = model.node_id_by_name(node_name) {
            let mut v = vec![];
            for a in gnode.attribute.iter() {
                let value = if let Some(t) = &a.t {
                    format!("{:?}", Tensor::try_from(t)?)
                } else {
                    format!("{:?}", a)
                };
                v.push(format!("Attr {}: {:.240}", bold.paint(&a.name), value));
            }
            annotations.node_mut(id.into()).sections.push(v);
        }
    }
    Ok(())
}

pub fn handle(
    params: &Parameters,
    options: &DisplayParams,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
    bench_limits: &BenchLimits,
    _inner: Vec<String>,
) -> TractResult<()> {
    let model = &*params.tract_model;
    let mut annotations = Annotations::from_model(model)?;
    annotate_with_graph_def(&mut annotations, model, &params.graph)?;
    if options.cost {
        annotations.extract_costs(model)?;
    }
    if options.profile {
        let run_params = run_params_from_subcommand(params, sub_matches)?;
        let model = params
            .tract_model
            .downcast_ref::<TypedModel>()
            .context("Can only profile typed models")?;
        tract_libcli::profile::profile(model, bench_limits, &mut annotations, &run_params)?;
    }

    if let Some(asserts) = &params.assertions.assert_output_facts {
        let outputs_facts: Vec<InferenceFact> = model
            .output_outlets()
            .iter()
            .map(|o| Ok(InferenceFact::from(&model.outlet_typedfact(*o)?)))
            .collect::<TractResult<Vec<InferenceFact>>>()?;
        crate::utils::check_inferred(&*outputs_facts, asserts)?;
    }
    if let Some(asserts) = &params.assertions.assert_op_count {
        for (name, expected) in asserts {
            let count = crate::utils::count_op(model, name)?;
            if count != *expected {
                bail!("Wrong number of {} operators: expected {}, got {}", name, expected, count);
            }
        }
    }

    if let Some(path) = sub_matches.value_of("nnef") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let file = std::fs::File::create(path)?;
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            nnef.write_to_tar(&typed, encoder).context("Writting model to tar")?;
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    if let Some(path) = sub_matches.value_of("nnef-tar") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let file = std::fs::File::create(path)?;
            nnef.write_to_tar(&typed, file).context("Writting model to tar")?;
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    if let Some(path) = sub_matches.value_of("nnef-dir") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            if let Some(renamed) = sub_matches.values_of("nnef-override-output-name") {
                for (ix, name) in renamed.into_iter().enumerate() {
                    let output = typed.wire_node(
                        name,
                        tract_core::ops::identity::Identity,
                        &[typed.output_outlets()?[ix]],
                    )?;
                    typed.outputs[ix] = output[0];
                }
            }
            nnef.write_to_dir(&typed, path)?
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    if let Some(path) = sub_matches.value_of("nnef-graph") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let proto = tract_nnef::ser::to_proto_model(&nnef, &typed)?;
            if path == "-" {
                tract_nnef::ast::dump::Dumper::new(&mut std::io::stdout()).document(&proto.doc)?;
            } else {
                let mut file = std::fs::File::create(path)?;
                tract_nnef::ast::dump::Dumper::new(&mut file).document(&proto.doc)?;
            }
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    if options.cost {
        let total = annotations.tags.values().sum::<NodeTags>();
        let assert =
            sub_matches.value_of("assert-cost").map(crate::cost::parse_costs).transpose()?;
        if let Some(assert) = assert {
            let assert: HashMap<Cost, TDim> =
                assert.iter().map(|(c, n)| (*c, n.to_dim())).collect();
            let total = total.cost.iter().cloned().collect::<HashMap<_, _>>();
            if assert != total {
                bail!("Cost assertion not met: expected {:?} got {:?}", assert, total);
            }
        }
    }

    if options.json {
        let export = crate::export::GraphPerfInfo::from(model, &annotations);
        serde_json::to_writer(std::io::stdout(), &export)?;
    } else {
        terminal::render(model, &annotations, options)?;
        terminal::render_summaries(model, &annotations, options)?;
    }

    Ok(())
}

fn rename_outputs(typed: &mut TypedModel, sub_matches: &clap::ArgMatches) -> TractResult<()> {
    if let Some(renamed) = sub_matches.values_of("nnef-override-output-name") {
        for (ix, name) in renamed.into_iter().enumerate() {
            let output = typed.wire_node(
                name,
                tract_core::ops::identity::Identity,
                &[typed.output_outlets()?[ix]],
            )?;
            typed.outputs[ix] = output[0];
        }
    }
    Ok(())
}
