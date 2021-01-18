use std::str::FromStr;
#[allow(unused_imports)]
use tract_itertools::Itertools;

use tract_core::internal::*;
use tract_core::model::TypedModel;
use tract_hir::internal::*;
#[cfg(feature = "pulse")]
use tract_pulse::internal::*;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;

use crate::display_params::DisplayParams;
use crate::CliResult;

use readings_probe::*;

use super::display_params;
use super::{info_usage, tensor};

use super::model::Model;

use std::convert::*;

#[derive(Debug)]
pub enum SomeGraphDef {
    NoGraphDef,
    #[cfg(feature = "kaldi")]
    Kaldi(tract_kaldi::KaldiProtoModel),
    Nnef(tract_nnef::ProtoModel),
    #[cfg(feature = "onnx")]
    Onnx(tract_onnx::pb::ModelProto, tract_onnx::model::ParseResult),
    #[cfg(feature = "tf")]
    Tf(GraphDef),
}

#[derive(Debug)]
pub struct ModelBuildingError(pub Box<dyn Model>, pub Box<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for ModelBuildingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&*self.1, f)
    }
}

impl std::error::Error for ModelBuildingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.1)
    }
}

#[cfg(not(feature = "pulse"))]
type PulsedModel = ();

/// Structure holding the parsed parameters.
pub struct Parameters {
    pub graph: SomeGraphDef,

    pub pulsed_model: Option<Arc<PulsedModel>>,

    pub tract_model: Arc<dyn Model>,
    pub reference_model: Option<Arc<dyn Model>>,

    #[cfg(feature = "conform")]
    pub tf_model: Option<tract_tensorflow::conform::tf::Tensorflow>,

    #[cfg(not(feature = "conform"))]
    #[allow(dead_code)]
    pub tf_model: (),

    pub input_values: HashMap<String, Arc<Tensor>>,

    pub assertions: Assertions,

    pub machine_friendly: bool,
}

#[cfg(feature = "tf")]
type TfExt = tract_tensorflow::model::TfModelExtensions;
#[cfg(not(feature = "tf"))]
type TfExt = ();

impl Parameters {
    fn disco_model(matches: &clap::ArgMatches) -> CliResult<(std::path::PathBuf, bool)> {
        let filename = matches.value_of("model").context("Model argument required")?;
        let filename = std::path::PathBuf::from(filename);
        let (filename, onnx_tc) = if !filename.exists() {
            bail!("model not found: {:?}", filename)
        } else if std::fs::metadata(&filename)?.is_dir() && filename.join("graph.nnef").exists() {
            (filename, false)
        } else if std::fs::metadata(&filename)?.is_dir() && filename.join("model.onnx").exists() {
            (filename.join("model.onnx"), true)
        } else {
            (filename, false)
        };
        Ok((filename, onnx_tc))
    }

    fn load_model(
        matches: &clap::ArgMatches,
        probe: Option<&Probe>,
        filename: &std::path::Path,
    ) -> CliResult<(SomeGraphDef, Box<dyn Model>, Option<TfExt>)> {
        let need_graph =
            matches.is_present("proto") || matches.subcommand_name() == Some("compare-pbdir");

        let format = matches.value_of("format").unwrap_or(
            if filename.extension().map(|s| s == "onnx").unwrap_or(false) {
                "onnx"
            } else if filename.extension().map(|s| s == "raw" || s == "txt").unwrap_or(false) {
                "kaldi"
            } else if filename.is_dir()
                || filename.to_string_lossy().ends_with(".tar")
                || filename.to_string_lossy().ends_with(".tar.gz")
                || filename.extension().map(|s| s == "tgz").unwrap_or(false)
            {
                "nnef"
            } else {
                "tf"
            },
        );
        let triplet: (SomeGraphDef, Box<dyn Model>, Option<TfExt>) = match format {
            #[cfg(feature = "kaldi")]
            "kaldi" => {
                let kaldi = tract_kaldi::kaldi();
                info_usage("loaded framework (kaldi)", probe);
                let mut graph = kaldi.proto_model_for_path(&filename)?;
                info_usage("proto model loaded", probe);
                if let Some(i) = matches.value_of("kaldi_adjust_final_offset") {
                    graph.adjust_final_offset = i.parse()?;
                }
                let parsed = kaldi.model_for_proto_model(&graph)?;
                if need_graph {
                    (SomeGraphDef::Kaldi(graph), Box::new(parsed), Option::<TfExt>::None)
                } else {
                    (SomeGraphDef::NoGraphDef, Box::new(parsed), Option::<TfExt>::None)
                }
            }
            "nnef" => {
                let nnef = super::nnef(&matches);
                let mut file = std::fs::File::open(&filename)?;
                let proto_model = if filename.to_string_lossy().ends_with("gz") {
                    nnef.proto_model_for_read(&mut flate2::read::GzDecoder::new(file))?
                } else {
                    nnef.proto_model_for_read(&mut file)?
                };
                info_usage("proto model loaded", probe);
                if need_graph {
                    (
                        SomeGraphDef::Nnef(proto_model.clone()),
                        Box::new(
                            nnef.translate(&proto_model)
                                .map_err(|(g, e)| ModelBuildingError(Box::new(g), e.into()))?,
                        ),
                        Option::<TfExt>::None,
                    )
                } else {
                    (
                        SomeGraphDef::NoGraphDef,
                        Box::new(
                            nnef.translate(&proto_model)
                                .map_err(|(g, e)| ModelBuildingError(Box::new(g), e.into()))?,
                        ),
                        Option::<TfExt>::None,
                    )
                }
            }
            #[cfg(feature = "onnx")]
            "onnx" => {
                let onnx = tract_onnx::onnx();
                info_usage("loaded framework (onnx)", probe);
                let graph = onnx.proto_model_for_path(&filename)?;
                info_usage("proto model loaded", probe);
                let parsed = onnx.parse(&graph)?;
                if need_graph {
                    (
                        SomeGraphDef::Onnx(graph, parsed.clone()),
                        Box::new(parsed.model),
                        Option::<TfExt>::None,
                    )
                } else {
                    (SomeGraphDef::NoGraphDef, Box::new(parsed.model), Option::<TfExt>::None)
                }
            }
            #[cfg(feature = "tf")]
            "tf" => {
                let tf = tract_tensorflow::tensorflow();
                info_usage("loaded framework (tf)", probe);
                let mut graph = tf.proto_model_for_path(&filename)?;
                info_usage("proto model loaded", probe);
                if matches.is_present("determinize") {
                    tract_tensorflow::Tensorflow::determinize(&mut graph)?;
                }
                let mut model_and_ext = tf.parse_graph(&graph)?;
                model_and_ext.1.initializing_nodes = matches
                    .values_of("tf_initializer_output_node")
                    .map(|values| {
                        values
                            .map(|name| model_and_ext.0.node_id_by_name(name))
                            .collect::<TractResult<Vec<usize>>>()
                    })
                    .transpose()?
                    .unwrap_or(vec![]);
                if need_graph {
                    (SomeGraphDef::Tf(graph), Box::new(model_and_ext.0), Some(model_and_ext.1))
                } else {
                    (SomeGraphDef::NoGraphDef, Box::new(model_and_ext.0), Some(model_and_ext.1))
                }
            }
            _ => bail!(
                "Format {} not supported. You may need to recompile tract with the right features.",
                format
            ),
        };
        Ok(triplet)
    }

    fn kaldi_downsample<F, O>(raw_model: &mut Graph<F, O>, period: isize) -> CliResult<()>
    where
        F: std::fmt::Debug + Clone + Hash + Fact,
        O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + Hash,
        Graph<F, O>: SpecialOps<F, O>,
        tract_core::ops::Downsample: Into<O>,
    {
        if period != 1 {
            let mut outputs = raw_model.output_outlets()?.to_vec();
            let output_name = raw_model.node(outputs[0].node).name.clone();
            raw_model.node_mut(outputs[0].node).name = format!("{}-old", output_name);
            let id = raw_model.wire_node(
                output_name,
                tract_core::ops::Downsample::new(0, period as _, 0),
                &outputs[0..1],
            )?[0];
            outputs[0] = id;
            raw_model.set_output_outlets(&*outputs)?;
        }
        Ok(())
    }

    fn kaldi_context<F, O>(raw_model: &mut Graph<F, O>, left: usize, right: usize) -> CliResult<()>
    where
        F: std::fmt::Debug + Clone + Hash + Fact,
        O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + Hash,
        Graph<F, O>: SpecialOps<F, O>,
        tract_hir::ops::array::Pad: Into<O>,
    {
        let op = tract_core::ops::array::Pad::new(
            vec![(left, right), (0, 0)],
            tract_core::ops::array::PadMode::Edge,
        );
        let mut patch = ModelPatch::default();
        for input in raw_model.input_outlets()? {
            let tap = patch.tap_model(raw_model, *input)?;
            let pad = patch.wire_node(
                format!("{}-pad", raw_model.node(input.node).name),
                op.clone(),
                &[tap],
            )?[0];
            patch.shunt_outside(&raw_model, *input, pad)?;
        }
        patch.apply(raw_model)?;
        Ok(())
    }

    fn use_onnx_test_case_data_set<F, O, E>(
        raw_model: &mut Graph<F, O>,
        input_values: &mut HashMap<String, Arc<Tensor>>,
        assertions: &mut Assertions,
        inputs_dir: &std::path::Path,
    ) -> CliResult<()>
    where
        F: std::fmt::Debug + Clone + Hash + Fact + for<'a> TryFrom<&'a InferenceFact, Error = E>,
        O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + Hash,
        Graph<F, O>: SpecialOps<F, O>,
        E: std::fmt::Debug,
    {
        let files = inputs_dir
            .read_dir()?
            .map(|file| {
                let file = file?;
                let filename = file
                    .file_name()
                    .into_string()
                    .map_err(|s| format_err!("Can't convert OSString to String ({:?})", s))?;
                if filename.starts_with("input_") || filename.starts_with("output_") {
                    let ix = filename
                        .split("_")
                        .nth(1)
                        .unwrap()
                        .split(".")
                        .nth(0)
                        .unwrap()
                        .parse::<usize>()?;
                    let (name, tensor) = tensor::for_data(file.path().to_str().unwrap())?;
                    Ok(Some((ix, filename.starts_with("input_"), filename, name.unwrap(), tensor)))
                } else {
                    Ok(None)
                }
            })
            .collect::<CliResult<Vec<Option<_>>>>()?;
        let files = files.into_iter().filter_map(|x| x).collect::<Vec<_>>();
        let (inputs, outputs) = files.iter().partition::<Vec<_>, _>(|f| f.1);
        let inputs = inputs.into_iter().sorted_by_key(|f| f.0).collect::<Vec<_>>();
        let outputs = outputs.into_iter().sorted_by_key(|f| f.0).collect::<Vec<_>>();
        let input_names = inputs.iter().map(|i| &*i.3).collect::<Vec<_>>();
        let output_names = outputs.iter().map(|i| &*i.3).collect::<Vec<_>>();
        debug!("input_names from files: {:?}", input_names);
        debug!("output_names from files: {:?}", output_names);
        if input_names.iter().all(|n| n.len() > 0) {
            raw_model.set_input_names(input_names)?;
        }
        if output_names.iter().all(|n| n.len() > 0) {
            raw_model.set_output_names(output_names)?;
        }
        for (ix, _, filename, name, tensor) in inputs.into_iter() {
            debug!("Using {} as input {} ({}): {:?}", filename, ix, name, tensor);
            if let Some(v) = tensor.value.concretize() {
                input_values.insert(name.to_string(), v);
            }
            raw_model.set_input_fact(*ix, (&tensor.clone().without_value()).try_into().unwrap())?;
        }
        let outputs = outputs
            .into_iter()
            .inspect(|(ix, _, filename, name, tensor)| {
                debug!("Using {} as output {} ({}): {:?}", filename, ix, name, tensor);
            })
            .map(|(_, _, _, _, tensor)| tensor.concretize())
            .collect();
        assertions.assert_outputs = outputs;
        Ok(())
    }

    fn inputs<F, O, E>(
        raw_model: &mut Graph<F, O>,
        assertions: &mut Assertions,
        matches: &clap::ArgMatches,
        filename: &std::path::Path,
        onnx_tc: bool,
    ) -> CliResult<HashMap<String, Arc<Tensor>>>
    where
        F: std::fmt::Debug + Clone + Hash + Fact + for<'a> TryFrom<&'a InferenceFact, Error = E>,
        O: std::fmt::Debug
            + std::fmt::Display
            + AsRef<dyn Op>
            + AsMut<dyn Op>
            + Clone
            + Hash
            + Send
            + Sync,
        Graph<F, O>: SpecialOps<F, O> + Send,
        tract_core::ops::konst::Const: Into<O>,
        E: std::fmt::Debug,
    {
        let mut input_values = HashMap::new();

        if let Some(inputs) = matches.values_of("input") {
            for (ix, v) in inputs.enumerate() {
                let (name, t) = tensor::for_string(v)?;
                let fact = t.clone().without_value();
                let fact: F = (&fact).try_into().unwrap();
                let outlet = if let Some(name) = name.filter(|s| s.len() > 0) {
                    let node = raw_model.node_by_name(&*name)?;
                    OutletId::new(node.id, 0)
                } else {
                    raw_model.input_outlets()?[ix]
                };
                if let Some(v) = t.value.concretize() {
                    input_values
                        .insert(raw_model.node(raw_model.inputs[ix].node).name.to_string(), v);
                }
                if !raw_model.inputs.contains(&outlet) {
                    // shed edges from parents to us
                    for input in raw_model.node(outlet.node).inputs.clone() {
                        raw_model.node_mut(input.node).outputs[input.slot]
                            .successors
                            .retain(|s| s.node != outlet.node);
                    }
                    // clear our inputs and change ourselves to a source
                    raw_model.node_mut(outlet.node).inputs.clear();
                    raw_model.node_mut(outlet.node).op = raw_model.create_source(fact.clone())
                }
                info!("Input #{}: {:?}", ix, t);
                raw_model.set_outlet_fact(outlet, fact)?;
            }
        }

        if let Some(bundle) = matches.values_of("input_bundle") {
            for input in bundle {
                let mut npz = ndarray_npy::NpzReader::new(
                    std::fs::File::open(input).with_context(|| format!("opening {:?}", input))?,
                )?;
                for name in npz.names()? {
                    match tensor::for_npz(&mut npz, &*name) {
                        Ok(t) => debug!("{} contains {}: {:?}", input, name, t),
                        Err(r) => warn!("Could not read {} from {} ({})", name, input, r),
                    }
                }
                let input_outlets = raw_model.input_outlets()?.to_vec();
                for (ix, input) in input_outlets.iter().enumerate() {
                    let name = raw_model.node(input.node).name.clone();
                    let npy_name = format!("{}.npy", name);
                    if let Ok(t) = tensor::for_npz(&mut npz, &npy_name) {
                        let shape = t.shape().to_vec();
                        let fact = InferenceFact::dt_shape(t.datum_type(), shape);
                        raw_model.set_input_fact(ix, (&fact).try_into().unwrap())?;
                        input_values.insert(name, t.into_arc_tensor());
                    }
                }
            }
        }

        if onnx_tc {
            Self::use_onnx_test_case_data_set(
                raw_model,
                &mut input_values,
                assertions,
                filename.parent().unwrap().join("test_data_set_0").as_path(),
            )?
        }

        if let Some(tc) = matches.value_of("onnx_test_data_set") {
            Self::use_onnx_test_case_data_set(
                raw_model,
                &mut input_values,
                assertions,
                &std::path::Path::new(tc),
            )?
        }

        let const_inputs = matches.values_of("const_input").map(|c| c.collect()).unwrap_or(vec![]);
        for i in (0..raw_model.inputs.len()).rev() {
            let input = raw_model.inputs[i];
            let name = raw_model.node_name(input.node);
            if const_inputs.contains(&raw_model.node_name(input.node)) {
                if let Some(v) = input_values.remove(name) {
                    raw_model.node_mut(input.node).op =
                        tract_core::ops::konst::Const::new(v.clone()).into();
                    raw_model.node_mut(input.node).outputs[0].fact =
                        F::try_from(&InferenceFact::from(v.into_tensor())).unwrap();
                } else {
                    bail!(
                        "Don't have value for input {}, can't make it const",
                        raw_model.node_name(input.node)
                    );
                }
                raw_model.inputs.remove(i);
            }
        }
        Ok(input_values)
    }

    #[allow(unused_variables)]
    fn pipeline(
        matches: &clap::ArgMatches,
        probe: Option<&readings_probe::Probe>,
        raw_model: Box<dyn Model>,
        tf_model_extensions: Option<TfExt>,
        reference_stage: Option<&str>,
    ) -> CliResult<(Arc<dyn Model>, Option<Arc<PulsedModel>>, Option<Arc<dyn Model>>)> {
        let keep_last = matches.is_present("verbose");
        #[cfg(feature = "pulse")]
        let pulse: Option<usize> =
            matches.value_of("pulse").map(|s| s.parse::<usize>()).transpose()?;
        #[cfg(feature = "pulse")]
        let concretize_stream_dim: Option<usize> =
            matches.value_of("concretize_stream_dim").map(|s| s.parse()).transpose()?;

        let stop_at = matches.value_of("pass").unwrap_or(if matches.is_present("optimize") {
            "optimize"
        } else {
            "before-optimize"
        });

        let nnef_cycle = matches.is_present("nnef_cycle");

        info!("Will stop at {}", stop_at);

        if stop_at == "load" {
            return Ok((raw_model.into(), None, None));
        }

        let mut inference_model: Option<Arc<InferenceModel>> = None;
        let mut typed_model: Option<Arc<TypedModel>> = None;
        #[allow(unused_mut)]
        let mut pulsed_model: Option<Arc<PulsedModel>> = None;
        let mut reference_model: Option<Arc<dyn Model>> = None;

        if raw_model.is::<InferenceModel>() {
            inference_model = Some(raw_model.downcast::<InferenceModel>().unwrap().into());
        } else if raw_model.is::<TypedModel>() {
            typed_model = Some(raw_model.downcast::<TypedModel>().unwrap().into());
        }

        macro_rules! stage {
            ($name:expr, $from:ident -> $to:ident, $block:expr) => {
                if let Some(from) = $from.take() {
                    info!(concat!("Running '", $name, "'"));
                    let mut last_model: Option<Box<dyn Model>> =
                        if keep_last { Some(Box::new(from.as_ref().clone())) } else { None };
                    let block: &dyn Fn(_) -> CliResult<_> = &$block;
                    let owned_model =
                        Arc::try_unwrap(from).unwrap_or_else(|from| from.as_ref().clone());
                    match block(owned_model) {
                        Ok(it) => {
                            $to = Some(Arc::new(it));
                        }
                        Err(e) => {
                            if let Some(last_model) = last_model.take() {
                                return Err(ModelBuildingError(last_model, e.into()))?;
                            } else {
                                return Err(e);
                            }
                        }
                    }
                    info_usage(concat!("after ", $name), probe);
                    if reference_stage.as_deref() == Some($name) {
                        reference_model = Some($to.as_ref().unwrap().clone());
                    }
                    if stop_at == $name {
                        return Ok((
                            $to.take().expect("returnable model"),
                            pulsed_model,
                            reference_model,
                        ));
                    }
                }
            };
        };

        stage!("analyse", inference_model -> inference_model,
        |mut m:InferenceModel| -> TractResult<_> {
            let result = m.analyse(matches.is_present("analyse_fail_fast"));
            match result {
                Ok(_) => Ok(m),
                Err(e) => Err(ModelBuildingError(Box::new(m), e.into()).into())
            }});
        if let Some(ext) = tf_model_extensions {
            #[cfg(feature = "tf")]
            stage!("tf-preproc", inference_model -> inference_model, |m:InferenceModel| Ok(ext.preproc(m)?));
        }
        stage!("incorporate", inference_model -> inference_model, |m:InferenceModel| { Ok(m.incorporate()?)});
        stage!("type", inference_model -> typed_model, |m:InferenceModel| Ok(m.into_typed()?));
        stage!("declutter", typed_model -> typed_model, |m:TypedModel| { let mut dec = tract_core::optim::Optimizer::declutter();
            if let Some(steps) = matches.value_of("declutter_step") {
                dec = dec.stopping_at(steps.parse()?);
            }
            dec.optimize(&m)
        });
        #[cfg(feature = "pulse")]
        {
            if let Some(dim) = concretize_stream_dim {
                stage!("concretize-stream-dim", typed_model -> typed_model, |m:TypedModel| Ok(m.concretize_dims(&SymbolValues::default().with(stream_symbol(), dim as _))?));
                stage!("concretize-stream-dim-declutter", typed_model -> typed_model, |m:TypedModel| Ok(m.declutter()?));
            } else if let Some(pulse) = pulse {
                stage!("pulse", typed_model -> pulsed_model, |m:TypedModel| Ok(PulsedModel::new(&m, pulse)?));
                stage!("pulse-to-type", pulsed_model -> typed_model, |m:PulsedModel| Ok(m.into_typed()?));
                stage!("pulse-declutter", typed_model -> typed_model, |m:TypedModel| Ok(m.declutter()?));
            }
        }
        if nnef_cycle {
            stage!("nnef-cycle", typed_model -> typed_model, |m:TypedModel| {
                let nnef = super::nnef(&matches);
                let mut vec = vec!();
                nnef.write(&m, &mut vec)?;
                Ok(nnef.model_for_read(&mut &*vec)?)
            });
            stage!("nnef-declutter", typed_model -> typed_model, |m:TypedModel| Ok(m.declutter()?));
        }
        stage!("before-optimize", typed_model -> typed_model, |m:TypedModel| Ok(m));
        stage!("optimize", typed_model -> typed_model, |m:TypedModel| Ok(m.optimize()?));
        Ok((typed_model.clone().unwrap(), pulsed_model, reference_model))
    }

    #[allow(unused_variables)]
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches, probe: Option<&Probe>) -> CliResult<Parameters> {
        let (filename, onnx_tc) = Self::disco_model(matches)?;
        let (mut graph, mut raw_model, tf_model_extensions) =
            Self::load_model(matches, probe, &filename)?;

        info!("Model {:?} loaded", filename);
        info_usage("model loaded", probe);

        let (need_tensorflow_model, need_reference_model) = match matches.subcommand() {
            ("compare", Some(sm)) => {
                if let Some(with) = sm.value_of("with") {
                    (false, Some(with))
                } else {
                    (true, None)
                }
            }
            ("optimize-check", _) => (false, Some("declutter")),
            _ => (false, None),
        };

        #[cfg(not(feature = "conform"))]
        let tf_model = ();
        #[cfg(feature = "conform")]
        let tf_model = if need_tensorflow_model {
            info!("Tensorflow version: {}", tract_tensorflow::conform::tf::version());
            if matches.is_present("determinize") {
                if let SomeGraphDef::Tf(ref graph) = graph {
                    let graph = graph.write_to_bytes().unwrap();
                    Some(tract_tensorflow::conform::tf::for_slice(&graph)?)
                } else {
                    unreachable!()
                }
            } else {
                Some(tract_tensorflow::conform::tf::for_path(&filename)?)
            }
        } else {
            None
        };

        if !matches.is_present("proto") && matches.subcommand_name() != Some("compare-pbdir") {
            graph = SomeGraphDef::NoGraphDef;
        }

        if let Some(inputs) = matches.values_of("input") {
            let names = inputs
                .map(|t| Ok(tensor::for_string(t)?.0))
                .collect::<CliResult<Vec<Option<String>>>>()?;
            if names.iter().all(|s| s.is_some() && s.as_ref().unwrap().len() > 0) {
                let names: Vec<&str> = names.iter().map(|s| &**s.as_ref().unwrap()).collect();
                raw_model.set_input_names(&*names)?;
            }
        }

        if let Some(inputs) = matches.values_of("input_node") {
            let inputs: Vec<&str> = inputs.map(|s| s).collect();
            raw_model.set_input_names(&inputs)?;
        };

        if let Some(outputs) = matches.values_of("output_node") {
            let outputs: Vec<&str> = outputs.map(|s| s).collect();
            raw_model.set_output_names(&outputs)?;
        };

        if let Some(override_facts) = matches.values_of("override_fact") {
            for fact in override_facts {
                let (name, fact) = tensor::for_string(fact)?;
                let node = raw_model.node_id_by_name(&name.unwrap())?;
                if let Some(inf) = raw_model.downcast_mut::<InferenceModel>() {
                    inf.set_outlet_fact(OutletId::new(node, 0), fact)?;
                } else if let Some(typ) = raw_model.downcast_mut::<TypedModel>() {
                    typ.set_outlet_fact(OutletId::new(node, 0), (&fact).try_into()?)?;
                }
            }
        };

        let output_names_and_labels: Vec<Vec<String>> = raw_model
            .output_outlets()
            .iter()
            .map(|o| {
                let mut v = vec![format!("{}:{}", raw_model.node_name(o.node), o.slot)];
                if o.slot == 0 {
                    v.push(raw_model.node_name(o.node).to_string());
                }
                if let Some(l) = raw_model.outlet_label(*o) {
                    v.push(l.to_string());
                }
                v
            })
            .collect();

        let mut assertions = Assertions::from_clap(matches, &*output_names_and_labels)?;

        if let Some(sub) = matches.value_of("kaldi_downsample") {
            dispatch_model_mut_no_pulse!(raw_model, |m| Self::kaldi_downsample(m, sub.parse()?))?;
        }

        if matches.value_of("kaldi_left_context").is_some()
            || matches.value_of("kaldi_right_context").is_some()
        {
            let left = matches.value_of("kaldi_left_context").unwrap_or("0").parse()?;
            let right = matches.value_of("kaldi_right_context").unwrap_or("0").parse()?;
            dispatch_model_mut_no_pulse!(raw_model, |m| Self::kaldi_context(m, left, right))?;
        }

        let input_values = dispatch_model_mut_no_pulse!(raw_model, |m| Self::inputs(
            m,
            &mut assertions,
            matches,
            &filename,
            onnx_tc
        ))?;

        if matches.is_present("partial") {
            if let Some(m) = raw_model.downcast_ref::<InferenceModel>() {
                raw_model = Box::new(m.compact()?);
            } else if let Some(m) = raw_model.downcast_ref::<TypedModel>() {
                raw_model = Box::new(m.compact()?);
            }
        }

        Self::pipeline(
            matches,
            probe,
            raw_model,
            tf_model_extensions,
            need_reference_model.as_deref(),
        )
        .map(|(tract_model, pulsed_model, reference_model)| {
            info!("Model ready");
            info_usage("model ready", probe);
            Parameters {
                graph,
                pulsed_model,
                tract_model,
                reference_model,
                tf_model,
                input_values,
                assertions,
                machine_friendly: matches.is_present("machine_friendly"),
            }
        })
    }
}

pub struct BenchLimits {
    pub max_iters: usize,
    pub max_time: std::time::Duration,
}

impl BenchLimits {
    pub fn from_clap(matches: &clap::ArgMatches) -> CliResult<BenchLimits> {
        let max_iters =
            matches.value_of("max_iters").map(usize::from_str).transpose()?.unwrap_or(100_000);
        let max_time = matches
            .value_of("max-time")
            .map(u64::from_str)
            .transpose()?
            .map(std::time::Duration::from_millis)
            .unwrap_or(std::time::Duration::from_secs(5));
        Ok(BenchLimits { max_iters, max_time })
    }
}

pub fn display_params_from_clap(
    root_matches: &clap::ArgMatches,
    matches: &clap::ArgMatches,
) -> CliResult<DisplayParams> {
    Ok(DisplayParams {
        konst: matches.is_present("const"),
        cost: matches.is_present("cost"),
        profile: matches.is_present("profile"),
        left_column_width: 0,
        invariants: matches.is_present("invariants"),
        quiet: matches.is_present("quiet"),
        natural_order: matches.is_present("natural-order"),
        debug_op: matches.is_present("debug-op"),
        node_ids: matches.values_of("node_id").map(|values| {
            values.map(|id| tvec!((id.parse::<usize>().unwrap(), "".to_string()))).collect()
        }),
        node_name: matches.value_of("node_name").map(String::from),
        op_name: matches.value_of("op_name").map(String::from),
        //        successors: matches.value_of("successors").map(|id| id.parse().unwrap()),
        expect_core: root_matches.value_of("pass").unwrap_or("declutter") == "declutter"
            && !root_matches.is_present("optimize"),
        outlet_labels: matches.is_present("outlet-labels"),
        io: if matches.is_present("io-long") {
            display_params::Io::Long
        } else if matches.is_present("io-none") {
            display_params::Io::None
        } else {
            display_params::Io::Short
        },
        info: matches.is_present("info"),
        json: matches.is_present("json"),
    })
}

#[derive(Debug)]
pub struct Assertions {
    pub assert_outputs: Vec<Option<Arc<Tensor>>>,
    pub assert_output_facts: Option<Vec<InferenceFact>>,
}

impl Assertions {
    fn from_clap(
        matches: &clap::ArgMatches,
        output_names: &[Vec<String>],
    ) -> CliResult<Assertions> {
        if let Some(sub) = matches.subcommand.as_ref().map(|sub| &sub.matches) {
            let mut assert_outputs: Vec<Option<Arc<Tensor>>> = vec![None; output_names.len()];
            if let Some(values) = sub.values_of("assert-output") {
                for (ix, o) in values.enumerate() {
                    assert_outputs[ix] = tensor::for_string(o).unwrap().1.value.concretize();
                }
            }

            if let Some(bundles) = sub.values_of("assert-output-bundle") {
                for bundle in bundles {
                    let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(bundle)?)?;
                    for (ix, labels) in output_names.iter().enumerate() {
                        for label in labels {
                            if assert_outputs[ix].is_some() {
                                continue;
                            }
                            let npy_name = format!("{}.npy", label);
                            if let Ok(t) = tensor::for_npz(&mut npz, &npy_name) {
                                assert_outputs[ix] = Some(t.into_arc_tensor())
                            }
                        }
                    }
                }
            }

            if sub.values_of("assert_output").is_some()
                || sub.values_of("assert-output-bundle").is_some()
            {
                if assert_outputs.contains(&None) {
                    bail!("Could not find assertions for all outputs: names and aliases are {:?}, found {:?}", output_names, assert_outputs);
                }
            }

            let assert_output_facts: Option<Vec<InferenceFact>> = matches
                .values_of("assert-output-fact")
                .map(|vs| vs.map(|v| tensor::for_string(v).unwrap().1).collect());

            Ok(Assertions { assert_outputs, assert_output_facts })
        } else {
            Ok(Assertions {
                assert_outputs: vec![None; output_names.len()],
                assert_output_facts: None,
            })
        }
    }
}
