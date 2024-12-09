use crate::params::SomeGraphDef;
use crate::plan_options::plan_options_from_subcommand;
use crate::tensor::run_params_from_subcommand;
use crate::Parameters;
use fs_err as fs;
#[allow(unused_imports)]
use nu_ansi_term::Style;
use tract_hir::internal::*;
use tract_libcli::annotations::*;
use tract_libcli::display_params::*;
use tract_libcli::model::Model;
use tract_libcli::profile::BenchLimits;
use tract_libcli::tensor::retrieve_or_make_inputs;
use tract_libcli::terminal;

#[allow(unused_variables)]
pub fn annotate_with_graph_def(
    annotations: &mut Annotations,
    model: &dyn Model,
    graph_def: &SomeGraphDef,
) -> TractResult<()> {
    match graph_def {
        SomeGraphDef::NoGraphDef => Ok(()),
        SomeGraphDef::Nnef(_) => todo!(),
        #[cfg(feature = "onnx")]
        SomeGraphDef::Onnx(onnx, _) => annotate_with_onnx_model(annotations, model, onnx),
        #[cfg(feature = "tf")]
        SomeGraphDef::Tf(tf) => annotate_with_tf_graph_def(annotations, model, tf),
        #[cfg(feature = "tflite")]
        SomeGraphDef::Tflite(tflite) => annotate_with_tflite_graph_def(annotations, model, tflite),
    }
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
                        format!("{r:?}")
                    } else {
                        format!("{:?}", a.1)
                    };
                v.push(format!("Attr {}: {:.300}", bold.paint(a.0), value));
            }
            annotations.node_mut(node_id.into()).sections.push(v);
        }
    }
    Ok(())
}

#[cfg(feature = "tflite")]
fn annotate_with_tflite_graph_def(
    _annotations: &mut Annotations,
    _model: &dyn Model,
    _graph_def: &tract_tflite::internal::TfliteProtoModel,
) -> TractResult<()> {
    Ok(())
}

#[cfg(feature = "onnx")]
fn annotate_with_onnx_model(
    annotations: &mut Annotations,
    model: &dyn Model,
    model_proto: &tract_onnx::pb::ModelProto,
) -> TractResult<()> {
    use tract_onnx::data_resolver::FopenDataResolver;
    use tract_onnx::tensor::load_tensor;

    let bold = Style::new().bold();
    for gnode in model_proto.graph.as_ref().unwrap().node.iter() {
        if let Some(id) = model
            .node_id_by_name(&gnode.name)
            .ok()
            .or_else(|| gnode.output.first().and_then(|n| model.node_id_by_name(n).ok()))
        {
            let mut v = vec![];
            for a in gnode.attribute.iter() {
                let value = if let Some(t) = &a.t {
                    format!("{:?}", load_tensor(&FopenDataResolver, t, None)?)
                } else {
                    format!("{a:?}")
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
    let run_params = run_params_from_subcommand(params, sub_matches)?;
    if options.cost {
        tract_libcli::profile::extract_costs(&mut annotations, model, &run_params.symbols)?;
    }
    if options.profile {
        let run_params = run_params_from_subcommand(params, sub_matches)?;
        let plan_options = plan_options_from_subcommand(sub_matches)?;
        let model = params
            .tract_model
            .downcast_ref::<TypedModel>()
            .context("Can only profile typed models")?;
        let inputs = retrieve_or_make_inputs(model, &run_params)?;

        if matches.is_present("metal") {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                tract_libcli::profile::profile_metal(
                    model,
                    bench_limits,
                    &mut annotations,
                    &plan_options,
                    &inputs[0],
                )?;
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                bail!("Metal profiling called on non-Metal device");
            }
        } else {
            tract_libcli::profile::profile(
                model,
                bench_limits,
                &mut annotations,
                &plan_options,
                &inputs[0],
                None,
                options.folded,
            )?;
        }
    }

    if sub_matches.is_present("axes") || sub_matches.is_present("axes-names") {
        let mut hints = HashMap::default();
        if let Some(params) = sub_matches.values_of("axes-names") {
            for param in params {
                let (node, names) = if let Some((node, axes)) = param.split_once('=') {
                    (model.node_id_by_name(node)?, axes)
                } else {
                    (model.input_outlets()[0].node, param)
                };
                let names: TVec<String> = names.split(',').map(|s| s.to_string()).collect();
                hints.insert(OutletId::new(node, 0), names);
            }
        }
        annotations.track_axes(model, &hints)?;
    }

    if sub_matches.is_present("tmp_mem_usage") {
        let plan_options = plan_options_from_subcommand(sub_matches)?;
        annotations.track_tmp_memory_usage(
            model,
            |n| !(n.op_is::<tract_core::ops::konst::Const>()),
            plan_options.skip_order_opt_ram,
        )?;
    }

    if let Some(asserts) = &params.assertions.assert_output_facts {
        let outputs_facts: Vec<InferenceFact> = model
            .output_outlets()
            .iter()
            .map(|o| Ok(InferenceFact::from(&model.outlet_typedfact(*o)?)))
            .collect::<TractResult<Vec<InferenceFact>>>()?;
        crate::utils::check_inferred(&outputs_facts, asserts)?;
    }
    if let Some(asserts) = &params.assertions.assert_op_count {
        for (name, expected) in asserts {
            let count = crate::utils::count_op(model, name)?;
            if count != *expected {
                bail!("Wrong number of {} operators: expected {}, got {}", name, expected, count);
            }
        }
    }

    let compress_submodels = sub_matches.is_present("compress-submodels");
    let deterministic = sub_matches.is_present("nnef-deterministic");
    if let Some(path) = sub_matches.value_of("nnef") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let file = fs::File::create(path)?;
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            nnef.write_to_tar_with_config(&typed, encoder, compress_submodels, deterministic)
                .context("Writing model to tgz")?;
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    if let Some(path) = sub_matches.value_of("nnef-tar") {
        let nnef = super::nnef(matches);
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let file = fs::File::create(path)?;
            nnef.write_to_tar_with_config(&typed, file, compress_submodels, deterministic)
                .context("Writing model to tar")?;
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
                tract_nnef::ast::dump::Dumper::new(&nnef, &mut std::io::stdout())
                    .document(&proto.doc)?;
            } else {
                let mut file = fs::File::create(path)?;
                tract_nnef::ast::dump::Dumper::new(&nnef, &mut file).document(&proto.doc)?;
            }
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    #[cfg(feature = "tflite")]
    if let Some(path) = sub_matches.value_of("tflite") {
        let tflite = tract_tflite::tflite();
        if let Some(mut typed) = model.downcast_ref::<TypedModel>().cloned() {
            rename_outputs(&mut typed, sub_matches)?;
            let file = fs::File::create(path)?;
            tflite.write(&typed, file).context("Writing model to tflite")?;
        } else {
            bail!("Only typed model can be dumped")
        }
    }

    #[cfg(not(feature = "tflite"))]
    if sub_matches.value_of("tflite").is_some() {
        bail!("This is a tract build without support for tflite.")
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
        let export = tract_libcli::export::GraphPerfInfo::from(model, &annotations);
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
