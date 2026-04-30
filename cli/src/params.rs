use fs_err as fs;
use reqwest::Url;
use scan_fmt::scan_fmt;
use std::io::Cursor;
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;
use tract_core::internal::*;
use tract_core::model::TypedModel;
use tract_core::ops::konst::Const;
#[allow(unused_imports)]
use tract_core::transform::ModelTransform;
use tract_hir::internal::*;
#[allow(unused_imports)]
use tract_itertools::Itertools;
use tract_libcli::profile::BenchLimits;
use tract_nnef::tensors::read_tensor;
#[cfg(feature = "pulse")]
use tract_pulse::internal::*;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;
#[cfg(feature = "tflite")]
use tract_tflite::internal::TfliteProtoModel;

use tract_nnef::ast::dump::Dumper;

use crate::TractResult;
use tract_libcli::display_params;
use tract_libcli::display_params::DisplayParams;
use tract_libcli::model::Model;
use tract_libcli::tensor;
use tract_libcli::tensor::{TensorValues, TensorsValues};

use readings_probe::*;

use super::info_usage;

use std::convert::*;

#[derive(Debug)]
enum Location {
    Fs(PathBuf),
    Http(Url),
}

impl Location {
    fn path(&self) -> Cow<'_, std::path::Path> {
        match self {
            Location::Fs(p) => p.into(),
            Location::Http(u) => std::path::Path::new(u.path()).into(),
        }
    }

    fn is_dir(&self) -> bool {
        if let &Location::Fs(p) = &self { p.is_dir() } else { false }
    }

    fn read(&self) -> TractResult<Box<dyn Read>> {
        match self {
            Location::Fs(p) => Ok(Box::new(fs::File::open(p)?)),
            Location::Http(u) => Ok(Box::new(http_client()?.get(u.clone()).send()?)),
        }
    }

    fn bytes(&self) -> TractResult<Vec<u8>> {
        let mut vec = vec![];
        self.read()?.read_to_end(&mut vec)?;
        Ok(vec)
    }

    fn find(s: impl AsRef<str>) -> TractResult<Self> {
        let s = s.as_ref();
        let path = std::path::PathBuf::from(s);
        if s.starts_with("http://") || s.starts_with("https://") {
            return Ok(Location::Http(s.parse()?));
        } else if path.exists() {
            return Ok(Location::Fs(path));
        } else if path.is_relative()
            && cfg!(any(
                target_os = "ios",
                target_os = "watchos",
                target_os = "tvos",
                target_os = "android"
            ))
        {
            if let Ok(pwd) = std::env::current_exe() {
                let absolute = pwd.parent().unwrap().join(&path);
                if absolute.exists() {
                    return Ok(Location::Fs(absolute));
                }
            }
        }
        bail!("File not found {}", s)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant, dead_code)]
pub enum SomeGraphDef {
    NoGraphDef,
    Nnef(tract_nnef::ProtoModel),
    #[cfg(feature = "onnx")]
    Onnx(tract_onnx::pb::ModelProto, tract_onnx::model::ParseResult),
    #[cfg(feature = "tf")]
    Tf(GraphDef),
    #[cfg(feature = "tflite")]
    Tflite(TfliteProtoModel),
}

#[derive(Debug)]
pub struct ModelBuildingError(pub Box<dyn Model>, pub Box<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for ModelBuildingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelBuildingError")
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
#[derive(Clone, Debug)]
pub struct Parameters {
    pub graph: SomeGraphDef,

    pub runnable: Option<Arc<dyn Runnable>>,
    pub tract_model: Arc<dyn Model>,
    pub reference_model: Option<Arc<dyn Model>>,

    #[cfg(feature = "conform")]
    pub tf_model: Option<tract_tensorflow::conform::tf::Tensorflow>,

    #[cfg(not(feature = "conform"))]
    #[allow(dead_code)]
    pub tf_model: (),

    pub tensors_values: TensorsValues,
    pub assertions: Assertions,

    pub machine_friendly: bool,
    pub allow_random_input: bool,
    pub allow_float_casts: bool,
}

#[cfg(feature = "tf")]
type TfExt = tract_tensorflow::model::TfModelExtensions;
#[cfg(not(feature = "tf"))]
type TfExt = ();

impl Parameters {
    fn disco_model(matches: &clap::ArgMatches) -> TractResult<(Location, bool)> {
        let model = matches.get_one::<String>("model").with_context(|| {
            format!(
                "Model argument required for subcommand {}",
                matches.subcommand_name().unwrap_or("")
            )
        })?;
        let location = Location::find(model)?;
        if location.is_dir() && location.path().join("model.onnx").exists() {
            Ok((Location::Fs(location.path().join("model.onnx")), false))
        } else {
            Ok((location, false))
        }
    }

    fn load_model(
        matches: &clap::ArgMatches,
        probe: Option<&Probe>,
        location: &Location,
        tensors_values: &TensorsValues,
        symbols: SymbolScope,
    ) -> TractResult<(SomeGraphDef, Box<dyn Model>, Option<TfExt>)> {
        let need_graph =
            matches.get_flag("proto") || matches.subcommand_name() == Some("compare-pbdir");

        let format = matches.get_one::<String>("format").map(String::as_str).unwrap_or(
            if location.path().extension().map(|s| s == "onnx").unwrap_or(false) {
                "onnx"
            } else if location.path().extension().map(|s| s == "tflite").unwrap_or(false) {
                "tflite"
            } else if location.is_dir()
                || location.path().to_string_lossy().ends_with(".tar")
                || location.path().to_string_lossy().ends_with(".tar.gz")
                || location.path().extension().map(|s| s == "tgz").unwrap_or(false)
            {
                "nnef"
            } else {
                "tf"
            },
        );
        let triplet: (SomeGraphDef, Box<dyn Model>, Option<TfExt>) = match format {
            "nnef" => {
                let nnef = super::nnef(matches);
                let mut proto_model = if location.is_dir() {
                    if let Location::Fs(dir) = location {
                        nnef.proto_model_for_path(dir)?
                    } else {
                        unreachable!();
                    }
                } else if location
                    .path()
                    .extension()
                    .map(|e| e.to_string_lossy().ends_with("gz"))
                    .unwrap_or(false)
                {
                    nnef.proto_model_for_read(&mut flate2::read::GzDecoder::new(
                        &mut *location.read()?,
                    ))?
                } else {
                    nnef.proto_model_for_read(&mut *location.read()?)?
                };
                for (ix, name) in proto_model.doc.graph_def.parameters.iter().enumerate() {
                    #[allow(unused_imports)]
                    use tract_nnef::ast::{LValue, RValue};
                    if let Some(over) = tensors_values
                        .input_by_name(&name.0)
                        .or_else(|| tensors_values.by_input_ix(ix))
                        .and_then(|tv| tv.fact.as_ref())
                    {
                        let assignment_id = proto_model
                            .doc
                            .graph_def
                            .body
                            .iter()
                            .position(|a| a.left == LValue::Identifier(name.clone()))
                            .context("Could not find input assignement in nnef body")?;
                        let mut formatted = vec![];
                        let ass = &mut proto_model.doc.graph_def.body[assignment_id];
                        let inv = if let RValue::Invocation(inv) = &mut ass.right {
                            inv
                        } else {
                            unreachable!();
                        };
                        assert!(
                            inv.id.0 == "external" || inv.id.0 == "tract_core_external",
                            "invalid id: expected 'external' or 'tract_core_external' but found {:?}",
                            inv.id
                        );
                        assert!(
                            inv.arguments.len() <= 2,
                            "expected 1 argument but found {:?} for inv.arguments={:?}",
                            inv.arguments.len(),
                            inv.arguments
                        );
                        assert_eq!(inv.arguments[0].id.as_ref().map(|i| &*i.0), Some("shape"));
                        Dumper::new(&nnef, &mut formatted).rvalue(&inv.arguments[0].rvalue)?;
                        let shape = over
                            .shape
                            .concretize()
                            .context("Can only use concrete shapes in override")?;
                        info!(
                            "Overriding model input shape named \"{}\". Replacing {} by {:?}.",
                            name.0,
                            String::from_utf8_lossy(&formatted),
                            &shape
                        );
                        inv.arguments[0].rvalue = tract_nnef::ser::tdims(&shape);
                    }
                }
                info_usage("proto model loaded", probe);
                let template = TypedModel { symbols, ..TypedModel::default() };
                let graph_def = if need_graph {
                    SomeGraphDef::Nnef(proto_model.clone())
                } else {
                    SomeGraphDef::NoGraphDef
                };
                (
                    graph_def,
                    Box::new(
                        nnef.translate(&proto_model, template)
                            .map_err(|(g, e)| ModelBuildingError(Box::new(g), e.into()))?,
                    ),
                    Option::<TfExt>::None,
                )
            }
            #[cfg(feature = "tflite")]
            "tflite" => {
                let tflite = tract_tflite::tflite();
                info_usage("loaded framework (tflite)", probe);
                let proto = tflite.proto_model_for_read(&mut *location.read()?)?;
                info_usage("proto model loaded", probe);
                let template = TypedModel { symbols, ..TypedModel::default() };
                let model = tflite.model_for_proto_model_with_model_template(&proto, template)?;
                info_usage("proto model translated", probe);
                (SomeGraphDef::Tflite(proto), Box::new(model), Option::<TfExt>::None)
            }
            #[cfg(feature = "onnx")]
            "onnx" => {
                let onnx = tract_onnx::onnx()
                    .with_ignore_output_shapes(matches.get_flag("onnx-ignore-output-shapes"))
                    .with_ignore_output_types(matches.get_flag("onnx-ignore-output-types"))
                    .with_ignore_value_info(matches.get_flag("onnx-ignore-value-info"));
                info_usage("loaded framework (onnx)", probe);
                let graph = onnx.proto_model_for_read(&mut *location.read()?)?;
                info_usage("proto model loaded", probe);
                let path = &location.path().clone();
                let template = InferenceModel { symbols, ..InferenceModel::default() };
                let mut parsed = onnx.parse_with_template(
                    &graph,
                    path.parent().and_then(|it| it.to_str()),
                    template,
                )?;

                if matches.get_flag("determinize") {
                    tract_onnx::Onnx::determinize(&mut parsed.model)?;
                }

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
                let mut graph = tf.proto_model_for_read(&mut *location.read()?)?;
                info_usage("proto model loaded", probe);
                if matches.get_flag("determinize") {
                    tract_tensorflow::Tensorflow::determinize(&mut graph)?;
                }
                let template = InferenceModel { symbols, ..InferenceModel::default() };
                let model_and_ext = tf.parse_graph_with_template(&graph, template)?;
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

    fn use_onnx_test_case_data_set(
        symbol_table: &SymbolScope,
        inputs_dir: &std::path::Path,
    ) -> TractResult<Vec<TensorValues>> {
        let mut result = vec![];
        for file in inputs_dir.read_dir()? {
            let file = file?;
            let filename = file
                .file_name()
                .into_string()
                .map_err(|s| format_err!("Can't convert OSString to String ({:?})", s))?;
            let is_input = filename.starts_with("input_");
            let is_output = filename.starts_with("output_");
            if is_input || is_output {
                let ix = filename
                    .split('_')
                    .nth(1)
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .parse::<usize>()?;
                let fd = fs::File::open(file.path())?;
                let (name, tensor) =
                    tensor::for_data(symbol_table, file.path().to_str().unwrap(), fd)?;
                result.push(TensorValues {
                    input_index: Some(ix).filter(|_| is_input),
                    output_index: Some(ix).filter(|_| is_output),
                    name,
                    values: tensor.value.concretize().map(|t| vec![t.into_tensor().into()]),
                    fact: Some(tensor.without_value()),
                    random_range: None,
                    only_input: is_input,
                    only_output: is_output,
                })
            }
        }
        Ok(result)
    }

    fn tensor_values_from_iter(
        iter: impl Iterator<Item = (String, usize, Tensor)>,
        get_values: bool,
        get_facts: bool,
    ) -> Vec<TensorValues> {
        let mut result = vec![];
        for (name, vals) in iter.chunk_by(|triple| triple.0.clone()).into_iter() {
            let vals: Vec<_> = vals
                .into_iter()
                .sorted_by_key(|(_, turn, _)| *turn)
                .map(|(_, _, tensor)| tensor.into_tvalue())
                .collect();
            result.push(TensorValues {
                name: Some(name),
                fact: if get_facts {
                    Some(vals[0].datum_type().fact(vals[0].shape()).into())
                } else {
                    None
                },
                values: if get_values { Some(vals) } else { None },
                ..TensorValues::default()
            })
        }
        result
    }

    pub fn parse_nnef_tensors(
        input: &str,
        get_values: bool,
        get_facts: bool,
    ) -> TractResult<Vec<TensorValues>> {
        let files = fs::read_dir(input)?;
        let vector = files
            .map(|n| {
                let file_path = n?.path();
                let tensor_file = fs::read(&file_path)?;
                let file_name = file_path.as_os_str().to_str().unwrap();
                if let Ok((turn, name)) =
                    scan_fmt::scan_fmt!(file_name, "turn_{d}/{}.dat", usize, String)
                {
                    Ok((name, turn, read_tensor(tensor_file.as_slice())?))
                } else {
                    let name = file_path.file_stem().unwrap().to_os_string().into_string().unwrap();
                    Ok((name, 0, read_tensor(tensor_file.as_slice())?))
                }
            })
            .collect::<TractResult<Vec<_>>>()?;
        Ok(Self::tensor_values_from_iter(vector.into_iter(), get_values, get_facts))
    }

    pub fn parse_set_and_hint(
        typed_model: &TypedModel,
        set: impl Iterator<Item = impl AsRef<str>>,
    ) -> TractResult<SymbolValues> {
        let mut values = SymbolValues::default();
        for set in set {
            let set = set.as_ref();
            let (key, value) = set.split_once('=').with_context(|| {
                format!("--set and --hint must be in the X=value form, got {set}")
            })?;
            let value: i64 = value
                .parse()
                .with_context(|| format!("value expected to be an integer, got {value}"))?;
            let key = typed_model.get_or_intern_symbol(key);
            values.set(&key, value);
        }
        Ok(values)
    }

    pub fn parse_npz(
        input: &str,
        get_values: bool,
        get_facts: bool,
    ) -> TractResult<Vec<TensorValues>> {
        let loc = Location::find(input)?;
        let mut npz = ndarray_npy::NpzReader::new(Cursor::new(loc.bytes()?))?;
        let triples = npz
            .names()?
            .iter()
            .map(|n| {
                if let Ok((turn, name)) = scan_fmt::scan_fmt!(n, "turn_{d}/{}.npy", usize, String) {
                    Ok((name, turn, tensor::for_npz(&mut npz, n)?))
                } else {
                    let name = n.trim_end_matches(".npy").to_string();
                    Ok((name, 0, tensor::for_npz(&mut npz, n)?))
                }
            })
            .collect::<TractResult<Vec<_>>>()?;
        Ok(Self::tensor_values_from_iter(triples.into_iter(), get_values, get_facts))
    }

    fn parse_tensors(
        matches: &clap::ArgMatches,
        location: &Location,
        onnx_tc: bool,
        symbol_table: &SymbolScope,
    ) -> TractResult<TensorsValues> {
        let mut result = TensorsValues::default();

        if let Some(inputs) = matches.get_many::<String>("input") {
            for (ix, v) in inputs.enumerate() {
                let v = v.as_str();
                let (name, fact) = tensor::for_string(symbol_table, v)?;
                let input_index = if name.is_some() { None } else { Some(ix) };
                result.add(TensorValues {
                    input_index,
                    name,
                    values: fact.value.concretize().map(|t| vec![t.into_tensor().into()]),
                    fact: Some(fact.without_value()),
                    only_input: true,
                    ..TensorValues::default()
                });
            }
        }

        if let Some(bundle) = matches.get_many::<String>("input-bundle") {
            warn!(
                "Argument --input-bundle is deprecated and may be removed in a future release. Use --input-facts-from-bundle and/or --input-from-bundle instead."
            );
            for input in bundle {
                let input = input.as_str();
                for tv in Self::parse_npz(input, true, true)? {
                    result.add(tv);
                }
            }
        }

        if let Some(bundle) = matches.get_many::<String>("input-facts-from-bundle") {
            for input in bundle {
                let input = input.as_str();
                for tv in Self::parse_npz(input, false, true)? {
                    result.add(tv);
                }
            }
        }

        if let Some((_, sub)) = matches.subcommand() {
            if let Some(values) = sub.get_many::<String>("assert-output") {
                for (ix, o) in values.enumerate() {
                    let o = o.as_str();
                    let (name, fact) = tensor::for_string(symbol_table, o)?;
                    info!(
                        "Output assertion #{}: (named: {}) {:?}",
                        ix,
                        name.as_deref().unwrap_or(""),
                        fact
                    );
                    result.add(TensorValues {
                        output_index: Some(ix),
                        name,
                        values: fact.value.concretize().map(|t| vec![t.into_tensor().into()]),
                        fact: Some(fact.without_value()),
                        only_output: true,
                        ..TensorValues::default()
                    });
                }
            }

            if let Some(bundles) = sub.get_many::<String>("assert-output-bundle") {
                for bundle in bundles {
                    let bundle = bundle.as_str();
                    for mut tv in Self::parse_npz(bundle, true, false)? {
                        tv.only_output = true;
                        result.add(tv);
                    }
                }
            }
        }

        if onnx_tc {
            let data_set_name = matches
                .get_one::<String>("onnx-test-data-set")
                .map(String::as_str)
                .unwrap_or("test_data_set_0");

            for tv in Self::use_onnx_test_case_data_set(
                symbol_table,
                location.path().parent().unwrap().join(data_set_name).as_path(),
            )? {
                result.add(tv)
            }
        }

        if let Some((_, sub)) = matches.subcommand() {
            if let Some(ranges) = sub.get_many::<String>("random-range") {
                for (ix, spec) in ranges.enumerate() {
                    let spec = spec.as_str();
                    let (name, from, to) = if let Ok((name, from, to)) =
                        scan_fmt!(spec, "{}={f}..{f}", String, f32, f32)
                    {
                        (Some(name), from, to)
                    } else if let Ok((from, to)) = scan_fmt!(spec, "{f}..{f}", f32, f32) {
                        (None, from, to)
                    } else {
                        bail!("Can't parse random-range parameter {}", spec)
                    };
                    let tv = if let Some(name) = name {
                        result.by_name_mut_with_default(&name)
                    } else {
                        result.by_input_ix_mut_with_default(ix)
                    };
                    tv.random_range = Some(from..to);
                }
            }
        }

        Ok(result)
    }

    #[allow(unused_variables)]
    #[allow(clippy::type_complexity)]
    fn load_and_declutter(
        matches: &clap::ArgMatches,
        probe: Option<&readings_probe::Probe>,
        raw_model: Box<dyn Model>,
        tf_model_extensions: Option<TfExt>,
        reference_stage: Option<&str>,
        keep_last: bool,
    ) -> TractResult<(Arc<dyn Model>, Option<Arc<dyn Model>>)> {
        let stop_at = matches
            .get_one::<String>("pass")
            .map(String::as_str)
            .unwrap_or(if matches.get_flag("optimize") { "optimize" } else { "before-optimize" });

        info!("Will stop at {stop_at}");

        if stop_at == "load" {
            return Ok((raw_model.into(), None));
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
                    info!("Running {:?}", $name);
                    let mut last_model: Option<Box<dyn Model>> =
                        if keep_last { Some(Box::new(from.as_ref().clone())) } else { None };
                    let block: &dyn Fn(_) -> TractResult<_> = &$block;
                    let owned_model =
                        Arc::try_unwrap(from).unwrap_or_else(|from| from.as_ref().clone());
                    match block(owned_model).with_context(|| format!("Error at stage {:?}", $name)) {
                        Ok(it) => {
                            $to = Some(Arc::new(it));
                        }
                        Err(e) => {
                            if e.is::<ModelBuildingError>() {
                                return Err(e)?;
                            } else if let Some(last_model) = last_model.take() {
                                return Err(ModelBuildingError(last_model, e.into()))?;
                            } else {
                                return Err(e)?;
                            }
                        }
                    }
                    info_usage(&format!("after {:?}", $name), probe);
                    if reference_stage.as_deref() == Some($name) {
                        reference_model = Some($to.as_ref().unwrap().clone());
                    }
                    if stop_at == $name {
                        return Ok((
                                $to.take().expect("returnable model"),
                                reference_model,
                                ));
                    }
                } else {
                    debug!("Skip stage {}", $name);
                    if stop_at == $name {
                        bail!("Stage {} is skipped, it can not be used as stop with these input format or parameters.", $name);
                    }
                }
            };
        }

        stage!("analyse", inference_model -> inference_model,
        |mut m:InferenceModel| -> TractResult<_> {
            m.analyse(!matches.get_flag("analyse-fail-fast")).map_err(|e|
                                                                        ModelBuildingError(Box::new(m.clone()), e.into())
                                                                       )?;
            if let Some(fail) = m.missing_type_shape()?.first() {
                bail!(ModelBuildingError(Box::new(m.clone()), format!("{} has incomplete typing", m.node(fail.node)).into()))
            }
            Ok(m)
        });
        if let Some(ext) = tf_model_extensions {
            #[cfg(feature = "tf")]
            stage!("tf-preproc", inference_model -> inference_model, |m:InferenceModel| ext.preproc(m));
        }
        stage!("incorporate", inference_model -> inference_model, |m:InferenceModel| m.incorporate());
        stage!("type", inference_model -> typed_model, |m:InferenceModel| { let mut m = m.into_typed()?; m.compact()?; Ok(m) });
        stage!("declutter", typed_model -> typed_model, |mut m:TypedModel| {
            if matches.get_flag("label-wires") {
                for node in 0..m.nodes().len() {
                    if m.outlet_label(node.into()).is_none() {
                        m.set_outlet_label(node.into(), m.node(node).name.to_string())?;
                    }
                }
            }
            let mut dec = tract_core::optim::Optimizer::declutter();
            if let Some(steps) = matches.get_one::<String>("declutter-step") {
                dec = dec.stopping_at(steps.parse()?);
            }
            dec.optimize(&mut m)?;
            Ok(m)
        });
        #[cfg(not(feature = "pulse"))]
        {
            if matches.get_one::<String>("pulse").is_some() {
                bail!("This build of tract has pulse disabled.")
            }
        }
        #[cfg(feature = "pulse")]
        {
            if let Some(spec) = matches.get_one::<String>("pulse") {
                stage!("pulse", typed_model -> pulsed_model, |m:TypedModel| {
                    let (sym, pulse) = if let Ok((s,p)) = scan_fmt!(spec, "{}={}", String, String) {
                        (s, parse_tdim(&m.symbols, &p)?)
                    } else if let Ok(i) = parse_tdim(&m.symbols, spec) {
                        ("S".to_owned(), i)
                    } else {
                        bail!("Can not parse pulse specification {}", spec)
                    };
                    let sym = m.symbols.sym(&sym);
                    PulsedModel::new(&m, sym, &pulse)
                });
                stage!("pulse-to-type", pulsed_model -> typed_model, |m:PulsedModel| m.into_typed());
                stage!("pulse-declutter", typed_model -> typed_model, |m:TypedModel| m.into_decluttered());
            }
            if let Some(spec) = matches.get_one::<String>("pulse-v2") {
                stage!("pulse-v2", typed_model -> typed_model, |m:TypedModel| {
                    use tract_pulse::v2::PulseV2Model;
                    let stream_sym = m.symbols.sym(spec);
                    let pv2 = PulseV2Model::new(&m, stream_sym)?;
                    pv2.into_typed()
                });
                stage!("pulse-v2-to-type", typed_model -> typed_model, |m:TypedModel| Ok(m));
                stage!("pulse-v2-declutter", typed_model -> typed_model, |m:TypedModel| m.into_decluttered());
            }
        }
        let mut transforms: Vec<&str> = matches
            .get_many::<String>("transform")
            .map(|values| values.map(String::as_str).collect())
            .unwrap_or_default();
        if matches.get_flag("llm") {
            transforms.insert(0, "transformers_detect_all");
        }
        if transforms.len() > 0 {
            for spec in transforms {
                let (name, params_str) = tract_core::transform::split_spec(spec);
                let transform = if params_str.is_empty() {
                    tract_core::transform::get_transform(&name)?
                } else {
                    let mut de = ron::Deserializer::from_str(params_str)
                        .with_context(|| format!("Parsing RON params for transform {name}"))?;
                    tract_core::transform::get_transform_with_params(
                        &name,
                        &mut <dyn erased_serde::Deserializer>::erase(&mut de),
                    )?
                }
                .with_context(|| format!("Could not find transform named {name}"))?;
                stage!(&transform.name(), typed_model -> typed_model, |m:TypedModel| {
                    transform.transform_into(m)
                });
                stage!(&format!("{}_declutter", transform.name()), typed_model -> typed_model, |m:TypedModel| m.into_decluttered());
            }
        }

        if let Some(set) = matches.get_many::<String>("set") {
            let values = Self::parse_set_and_hint(typed_model.as_ref().unwrap(), set)?;
            stage!("set", typed_model -> typed_model, |mut m: TypedModel| {
                for node in m.eval_order()? {
                    let node = m.node_mut(node);
                    if let Some(op) = node.op_as_mut::<Const>() {
                        if op.val().datum_type() == DatumType::TDim { {
                            // get inner value to Arc<Tensor>
                            let mut constant:Tensor = (**op.val()).clone();
                            // Generally a shape or hyperparam
                            constant
                                .try_as_plain_mut()?
                                .as_slice_mut::<TDim>()?
                                .iter_mut()
                                .for_each(|x| *x = x.eval(&values));

                            *op = Const::new(constant.into_arc_tensor())?;
                        }
                        }
                    }
                }
                m.substitute_symbols(&values.to_dim_map())
            });
            stage!("set-declutter", typed_model -> typed_model, |mut m| {
                let mut dec = tract_core::optim::Optimizer::declutter();
                if let Some(steps) = matches.get_one::<String>("declutter-set-step") {
                    dec = dec.stopping_at(steps.parse()?);
                }
                dec.optimize(&mut m)?;
                Ok(m)
            })
        }
        if matches.get_flag("nnef-cycle") {
            stage!("nnef-cycle", typed_model -> typed_model, |m:TypedModel| {
                let nnef = super::nnef(matches);
                let mut vec = vec!();
                nnef.write(&m, &mut vec).context("Serializing to nnef")?;
                info!("Dumped, now reloading...");
                nnef.model_for_read(&mut &*vec).context("Deserializing from nnef intermediary")
            });
            stage!("nnef-declutter", typed_model -> typed_model, |m:TypedModel| m.into_decluttered());
        }
        #[cfg(feature = "tflite")]
        if matches.get_flag("tflite-cycle") {
            stage!("tflite-cycle-predump", typed_model -> typed_model, |mut m:TypedModel| {
                tract_tflite::rewriter::rewrite_for_tflite(&mut m)?;
                Ok(m)
            });
            stage!("tflite-cycle", typed_model -> typed_model, |m:TypedModel| {
                let tflite = tract_tflite::tflite();
                let mut vec = vec!();
                tflite.write(&m, &mut vec).context("Serializing to tflite")?;
                info!("Dumped, now reloading...");
                tflite.model_for_read(&mut &*vec).context("Deserializing from tflite intermediary")
            });
            stage!("tflite-declutter", typed_model -> typed_model, |m:TypedModel| m.into_decluttered());
        }
        #[cfg(not(feature = "tflite"))]
        if matches.get_flag("tflite-cycle") {
            bail!("This tract build did not include tflite features.");
        }
        if let Some(sub) = matches.get_one::<String>("extract-decluttered-sub") {
            stage!("extract", typed_model -> typed_model, |m:TypedModel| {
                let node = m.node_id_by_name(sub)?;
                Ok(m.nested_models(node)[0].1.downcast_ref::<TypedModel>().unwrap().clone())
            });
        }
        stage!("before-optimize", typed_model -> typed_model, Ok);
        Ok((typed_model.clone().unwrap(), reference_model))
    }

    #[allow(unused_variables)]
    #[allow(clippy::let_unit_value)]
    /// Parses the command-line arguments.
    pub fn from_clap(matches: &clap::ArgMatches, probe: Option<&Probe>) -> TractResult<Parameters> {
        let symbols = SymbolScope::default();
        for scenario in matches.get_many::<String>("scenario").unwrap_or_default() {
            symbols.add_scenario(scenario)?;
        }
        for rule in matches.get_many::<String>("assert").unwrap_or_default() {
            if let Some((scenario, assertion)) = rule.split_once(':') {
                symbols.add_scenario_assertion(scenario, assertion)?;
            } else {
                symbols.add_assertion(rule)?;
            }
        }
        let (filename, onnx_tc) = Self::disco_model(matches)?;
        let tensors_values = Self::parse_tensors(matches, &filename, onnx_tc, &symbols)?;
        let (mut graph, mut raw_model, tf_model_extensions) =
            Self::load_model(matches, probe, &filename, &tensors_values, symbols.clone())?;

        info!("Model {filename:?} loaded");
        info_usage("model loaded", probe);

        let (need_tensorflow_model, need_reference_model) = match matches.subcommand() {
            Some(("compare", sm)) => {
                if let Some(with) = sm.get_one::<String>("stage").map(String::as_str) {
                    (false, Some(with))
                } else if sm.get_flag("stream") {
                    (false, Some("declutter"))
                } else {
                    (true, None)
                }
            }
            _ => (false, None),
        };

        #[cfg(not(feature = "conform"))]
        let tf_model = ();
        #[cfg(feature = "conform")]
        let tf_model = if need_tensorflow_model {
            info!("Tensorflow version: {}", tract_tensorflow::conform::tf::version());
            if matches.get_flag("determinize") {
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

        let need_proto = matches.get_flag("proto")
            || (matches.subcommand_matches("compare").map(|sc| sc.contains_id("pbdir")))
                .unwrap_or(false);

        if !need_proto {
            graph = SomeGraphDef::NoGraphDef;
        }

        if let Some(inputs) = matches.get_many::<String>("input-node") {
            let inputs: Vec<&str> = inputs.map(String::as_str).collect();
            raw_model.set_input_names(&inputs)?;
        };

        if let Some(outputs) = matches.get_many::<String>("output-node") {
            let outputs: Vec<&str> = outputs.map(String::as_str).collect();
            raw_model.select_outputs_by_name(&outputs)?;
        };

        if let Some(override_facts) = matches.get_many::<String>("override-fact") {
            for fact in override_facts {
                let fact = fact.as_str();
                let (name, fact) = tensor::for_string(&symbols, fact)?;
                let node = raw_model.node_id_by_name(name.as_ref().unwrap())?;
                if let Some(inf) = raw_model.downcast_mut::<InferenceModel>() {
                    inf.set_outlet_fact(OutletId::new(node, 0), fact)?;
                } else if let Some(typ) = raw_model.downcast_mut::<TypedModel>() {
                    typ.set_outlet_fact(OutletId::new(node, 0), (&fact).try_into()?)?;
                }
            }
        };

        if let Some(consts) = matches.get_many::<String>("constantize") {
            for konst in consts {
                let konst = konst.as_str();
                if let Some(value) = tensors_values
                    .by_name(konst)
                    .and_then(|tv| tv.values.as_ref())
                    .and_then(|v| v.first())
                {
                    let value = value.clone().into_arc_tensor();
                    let id = raw_model.node_id_by_name(konst)?;
                    info!(
                        "Commuting {}, fact:{:?} into a const of {:?}",
                        raw_model.node_display(id),
                        raw_model.outlet_typedfact(id.into())?,
                        value
                    );
                    let op = Box::new(Const::new(value.clone().into_arc_tensor())?);
                    if let Some(inf) = raw_model.downcast_mut::<InferenceModel>() {
                        inf.inputs.retain(|i| i.node != id);
                        inf.nodes[id].op = op;
                        inf.nodes[id].outputs[0].fact = Default::default();
                    } else if let Some(typ) = raw_model.downcast_mut::<TypedModel>() {
                        typ.inputs.retain(|i| i.node != id);
                        typ.nodes[id].op = op;
                        typ.nodes[id].outputs[0].fact = TypedFact::try_from(value.clone())?;
                    }
                }
            }
        }

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

        let assertions = match matches.subcommand() {
            Some(("dump" | "run" | "compare", sm)) => Assertions::from_clap(sm, &symbols)?,
            _ => Assertions::default(),
        };

        if let Some(infer) = raw_model.downcast_mut::<InferenceModel>() {
            for (ix, node_id) in infer.inputs.iter().enumerate() {
                let tv = tensors_values
                    .input_by_name(&infer.node(node_id.node).name)
                    .or_else(|| tensors_values.by_input_ix(ix));
                if let Some(tv) = tv {
                    if let Some(fact) = &tv.fact {
                        infer.nodes[node_id.node].outputs[0].fact = fact.clone();
                    }
                }
            }
        }

        if matches.get_flag("partial") {
            if let Some(m) = raw_model.downcast_ref::<InferenceModel>() {
                raw_model = Box::new(m.clone().into_compact()?);
            } else if let Some(m) = raw_model.downcast_ref::<TypedModel>() {
                raw_model = Box::new(m.clone().into_compact()?);
            }
        }

        let (allow_random_input, allow_float_casts) = match matches.subcommand() {
            None => (false, false),
            Some((_, m)) => (m.get_flag("allow-random-input"), m.get_flag("allow-float-casts")),
        };

        let keep_last = matches.get_flag("keep-last");
        let (tract_model, reference_model) = Self::load_and_declutter(
            matches,
            probe,
            raw_model,
            tf_model_extensions,
            need_reference_model,
            keep_last,
        )?;

        info!("Model fully loaded");
        info_usage("model fully loaded", probe);

        let runtime = if let Some(rt) = matches.get_one::<String>("runtime").map(String::as_str) {
            rt
        } else if matches.get_flag("cuda") {
            "cuda"
        } else if matches.get_flag("metal") {
            "metal"
        } else if matches.get_flag("optimize") {
            "default"
        } else {
            "unoptimized"
        };

        let runtime =
            runtime_for_name(runtime)?.with_context(|| format!("Runtime `{runtime}' not found"))?;
        let (tract_model, runnable): (Arc<dyn Model>, Option<Arc<dyn Runnable>>) =
            if tract_model.downcast_ref::<TypedModel>().is_some() {
                let tract_model: Arc<TypedModel> = Arc::downcast(tract_model).unwrap();
                let typed_model = Arc::try_unwrap(tract_model).unwrap();
                let hints = if let Some(hints) = matches.get_many::<String>("hint") {
                    Some(Self::parse_set_and_hint(&typed_model, hints)?)
                } else if matches.get_flag("llm") || matches.get_flag("causal-llm-hints") {
                    #[cfg(feature = "transformers")]
                    {
                        Some(tract_transformers::memory_arena_hints_for_causal_llm(&typed_model)?)
                    }
                    #[cfg(not(feature = "transformers"))]
                    {
                        bail!("transformers feature is required for llms")
                    }
                } else {
                    None
                };

                let options = RunOptions { memory_sizing_hints: hints, ..Default::default() };
                let runnable = runtime.prepare_with_options(typed_model, &options)?;
                // we assume the runnable will be a typed_model() (it is the case for all current runtimes)
                // so we consume tract_model knowning the runnable will give us a new one later.
                // we should hold on the old model in the general case, but this leads to dup models weights in memory
                let typed_model = runnable.typed_model().unwrap();
                (typed_model.clone(), Some(runnable.into()))
            } else {
                (tract_model, None)
            };

        info!("Model ready");
        info_usage("model ready", probe);
        Ok(Parameters {
            graph,
            runnable,
            tract_model,
            reference_model,
            tf_model,
            tensors_values,
            assertions,
            machine_friendly: matches.get_flag("machine-friendly"),
            allow_random_input,
            allow_float_casts,
        })
    }

    pub(crate) fn typed_model(&self) -> Option<Arc<TypedModel>> {
        self.runnable
            .clone()
            .and_then(|runnable| runnable.typed_model().cloned())
            .or_else(|| Arc::downcast(self.tract_model.clone()).ok())
    }

    pub(crate) fn req_typed_model(&self) -> Arc<TypedModel> {
        self.typed_model().expect("Not a TypedModel")
    }

    #[allow(dead_code)]
    pub(crate) fn req_runnable(&self) -> TractResult<Arc<dyn Runnable>> {
        self.runnable.clone().context("Requires a runnable")
    }
}

pub fn bench_limits_from_clap(matches: &clap::ArgMatches) -> TractResult<BenchLimits> {
    let max_loops = matches
        .get_one::<String>("max-loops")
        .map(|s| usize::from_str(s))
        .transpose()?
        .unwrap_or(100_000);
    let warmup_loops = matches
        .get_one::<String>("warmup-loops")
        .map(|s| usize::from_str(s))
        .transpose()?
        .unwrap_or(0);
    let max_time = matches
        .get_one::<String>("max-time")
        .map(|s| u64::from_str(s))
        .transpose()?
        .map(std::time::Duration::from_millis)
        .unwrap_or(std::time::Duration::from_secs(5));
    let warmup_time = matches
        .get_one::<String>("warmup-time")
        .map(|s| u64::from_str(s))
        .transpose()?
        .map(std::time::Duration::from_millis)
        .unwrap_or(std::time::Duration::from_secs(0));
    Ok(BenchLimits { max_loops, max_time, warmup_time, warmup_loops })
}

pub fn display_params_from_clap(
    root_matches: &clap::ArgMatches,
    matches: &clap::ArgMatches,
) -> TractResult<DisplayParams> {
    Ok(DisplayParams {
        konst: matches.get_flag("const"),
        cost: matches.get_flag("cost"),
        tmp_mem_usage: matches.get_flag("tmp_mem_usage"),
        profile: matches.get_flag("profile"),
        folded: matches.get_flag("folded"),
        left_column_width: 0,
        invariants: matches.get_flag("invariants"),
        quiet: matches.get_flag("quiet"),
        natural_order: matches.get_flag("natural-order"),
        opt_ram_order: matches.get_flag("opt-ram-order"),
        debug_op: matches.get_flag("debug-op"),
        node_ids: matches.get_many::<String>("node-id").map(|values| {
            values.map(|id| tvec!((id.parse::<usize>().unwrap(), "".to_string()))).collect()
        }),
        node_name: matches.get_one::<String>("node-name").cloned(),
        op_name: matches.get_one::<String>("op-name").cloned(),
        //        successors: matches.get_one::<String>("successors").map(|id| id.parse().unwrap()),
        expect_core: root_matches
            .get_one::<String>("pass")
            .map(String::as_str)
            .unwrap_or("declutter")
            == "declutter"
            && !root_matches.get_flag("optimize"),
        outlet_labels: matches.get_flag("outlet-labels"),
        io: if matches.get_flag("io-long") {
            display_params::Io::Long
        } else if matches.get_flag("io-none") {
            display_params::Io::None
        } else {
            display_params::Io::Short
        },
        info: matches.get_flag("info"),
        json: matches.get_flag("json"),
        mm: matches.get_flag("mm"),
        summary: matches.try_get_one::<bool>("summary").ok().flatten().copied().unwrap_or(false),
        audit_json: matches
            .try_get_one::<bool>("audit-json")
            .ok()
            .flatten()
            .copied()
            .unwrap_or(false),
    })
}

#[derive(Debug, Default, Clone)]
pub struct Assertions {
    pub assert_outputs: bool,
    pub assert_output_facts: Option<Vec<InferenceFact>>,
    pub assert_op_count: Option<Vec<(String, usize)>>,
    pub approximation: Approximation,
    pub allow_missing_outputs: bool,
    pub assert_llm_rbo: Option<f64>,
    pub assert_llm_rbo_p: f64,
    pub assert_llm_rbo_depth: usize,
    pub assert_op_only: Option<Vec<String>>,
}

impl Assertions {
    fn from_clap(sub: &clap::ArgMatches, symbol_table: &SymbolScope) -> TractResult<Assertions> {
        let assert_outputs =
            sub.contains_id("assert-output") || sub.contains_id("assert-output-bundle");
        let assert_output_facts: Option<Vec<InferenceFact>> = sub
            .get_many::<String>("assert-output-fact")
            .map(|vs| vs.map(|v| tensor::for_string(symbol_table, v).unwrap().1).collect());
        let assert_op_count: Option<Vec<(String, usize)>> =
            sub.get_many::<String>("assert-op-count").and_then(|vs| {
                vs.map(String::as_str)
                    .chunks(2)
                    .into_iter()
                    .map(|mut args| Some((args.next()?.to_string(), args.next()?.parse().ok()?)))
                    .collect()
            });
        let allow_missing_outputs = sub.get_flag("allow-missing-outputs");
        let approximation = if let Some(custom) = sub.get_one::<String>("approx-custom") {
            let Some((atol, rtol, approx)) = custom.split(",").collect_tuple() else {
                bail!("Can't parse approx custom. It should look like 0.001,0.002,0.003")
            };
            Approximation::Custom(atol.parse()?, rtol.parse()?, approx.parse()?)
        } else {
            match sub.get_one::<String>("approx").map(String::as_str).unwrap() {
                "exact" => Approximation::Exact,
                "close" => Approximation::Close,
                "approx" | "approximate" => Approximation::Approximate,
                "very" => Approximation::VeryApproximate,
                "super" => Approximation::SuperApproximate,
                "ultra" => Approximation::UltraApproximate,
                _ => panic!(),
            }
        };
        let assert_llm_rbo =
            sub.get_one::<String>("assert-llm-rbo").map(|v| v.parse()).transpose()?;
        let assert_llm_rbo_p: f64 = sub
            .get_one::<String>("assert-llm-rbo-p")
            .map(String::as_str)
            .unwrap_or("0.9")
            .parse()?;
        let assert_llm_rbo_depth: usize = sub
            .get_one::<String>("assert-llm-rbo-depth")
            .map(String::as_str)
            .unwrap_or("100")
            .parse()?;
        let assert_op_only = sub
            .get_one::<String>("assert-op-only")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
        Ok(Assertions {
            assert_outputs,
            assert_output_facts,
            assert_op_count,
            approximation,
            allow_missing_outputs,
            assert_llm_rbo,
            assert_llm_rbo_p,
            assert_llm_rbo_depth,
            assert_op_only,
        })
    }
}

fn http_client() -> TractResult<reqwest::blocking::Client> {
    use rustls::{ClientConfig, RootCertStore};
    use std::sync::Arc;

    let mut root_store = RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()?
            .with_root_certificates(root_store)
            .with_no_client_auth();

    Ok(reqwest::blocking::Client::builder().use_preconfigured_tls(config).build()?)
}
