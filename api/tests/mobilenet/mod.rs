use std::sync::Once;

fn grace_hopper() -> Value {
    let data = std::fs::read("../tests/grace_hopper_3_224_224.f32.raw").unwrap();
    let data: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as _, 3 * 224 * 224) };
    Value::from_slice(&[1, 3, 224, 224], data).unwrap()
}

fn ensure_models() -> anyhow::Result<()> {
    static START: Once = Once::new();
    START.call_once(|| {
        for (url, file) in [
            (
                "https://s3.amazonaws.com/tract-ci-builds/tests/mobilenetv2-7.onnx",
                "mobilenetv2-7.onnx",
            ),
            (
                "https://s3.amazonaws.com/tract-ci-builds/tests/mobilenet_v2_1.0.onnx.nnef.tgz",
                "mobilenet_v2_1.0.onnx.nnef.tgz",
            ),
        ] {
            if std::fs::metadata(file).is_err() {
                let client = reqwest::blocking::Client::new();
                let model = client.get(url).send().unwrap();
                std::fs::write(file, model.bytes().unwrap()).unwrap();
            }
        }
    });
    Ok(())
}

#[test]
fn test_onnx() -> anyhow::Result<()> {
    ensure_models()?;
    let model = onnx()?.load("mobilenetv2-7.onnx")?.into_tract()?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    let result = result[0].view::<f32>()?;
    let best = result
        .as_slice()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best.0, 652);
    Ok(())
}

#[test]
fn test_state() -> anyhow::Result<()> {
    ensure_models()?;
    let model = onnx()?.load("mobilenetv2-7.onnx")?.into_tract()?.into_runnable()?;
    let mut state = model.spawn_state()?;
    let hopper = grace_hopper();
    let result = state.run([hopper])?;
    let result = result[0].view::<f32>()?;
    let best = result
        .as_slice()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best.0, 652);
    Ok(())
}

#[test]
fn test_nnef() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    assert_eq!(result[0].datum_type()?, f32::datum_type());
    let result = result[0].view::<f32>()?;
    let best = result
        .as_slice()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best.0, 652);
    Ok(())
}

#[test]
fn test_inference_model() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    assert_eq!(model.input_count().unwrap(), 1);
    assert_eq!(model.output_count().unwrap(), 1);
    assert_eq!(model.input_name(0).unwrap(), "data");
    assert_eq!(model.output_name(0).unwrap(), "mobilenetv20_output_flatten0_reshape0");
    assert_eq!(model.input_fact(0).unwrap().to_string(), "1,3,224,224,F32");
    model.set_input_fact(0, "1,3,224,224,F32")?;
    let model = model.into_tract()?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    let view = result[0].view::<f32>()?;
    let best = view
        .as_slice()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best.0, 652);
    Ok(())
}

#[test]
fn test_set_output_names_on_inference_model() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.set_output_fact(0, None)?;
    model.analyse()?;
    model.set_output_names(["mobilenetv20_output_pred_fwd"])?;
    assert_eq!(model.output_fact(0).unwrap().to_string(), "B,1000,1,1,F32");
    Ok(())
}

#[test]
fn test_typed_model() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    assert_eq!(model.input_count()?, 1);
    assert_eq!(model.output_count()?, 1);
    assert_eq!(model.input_name(0)?, "data");
    assert_eq!(model.output_name(0)?, "conv_53");
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F32");
    Ok(())
}

#[test]
fn test_runtime() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    let rt = runtime_for_name("default")?;
    let runnable = rt.prepare(model)?;
    let hopper = grace_hopper();
    let _result = runnable.run([hopper])?;
    Ok(())
}

#[test]
fn test_set_output_names() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    model.set_output_names(["mean_reduce_mean_reduce_output"])?;
    assert_eq!(model.output_fact(0)?.to_string(), "1280,1,1,F32");
    Ok(())
}

#[test]
fn test_concretize() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_tract()?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F32");
    typed.concretize_symbols([("B", 1)])?;
    assert_eq!(typed.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(typed.output_fact(0)?.to_string(), "1,1000,F32");
    Ok(())
}

#[test]
fn test_pulse() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_tract()?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F32");
    typed.pulse("B", "5")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "5,3,224,224,F32");
    assert_eq!(typed.output_fact(0)?.to_string(), "5,1000,F32");
    let mut properties = typed.property_keys()?;
    properties.sort();
    assert_eq!(&properties, &["pulse.delay", "pulse.input_axes", "pulse.output_axes"]);
    assert_eq!(typed.property("pulse.delay")?.view::<i64>()?, ndarray::arr1(&[0i64]).into_dyn());
    Ok(())
}

#[test]
fn test_runtime_fact() -> anyhow::Result<()> {
    ensure_models()?;
    let runnable = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    assert_eq!(runnable.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(runnable.output_fact(0)?.to_string(), "1,1000,F32");
    Ok(())
}

#[test]
fn test_runtime_properties() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_tract()?;
    typed.pulse("B", "5")?;
    let runnable = typed.into_runnable()?;
    let mut properties = runnable.property_keys()?;
    properties.sort();
    assert_eq!(&properties, &["pulse.delay", "pulse.input_axes", "pulse.output_axes"]);
    assert_eq!(runnable.property("pulse.delay")?.view::<i64>()?, ndarray::arr1(&[0i64]).into_dyn());
    Ok(())
}

#[test]
fn test_f32_to_f16() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_tract()?;
    typed.transform("f32-to-f16")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F16");
    Ok(())
}

#[test]
fn test_f16_to_f32() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_tract()?;

    // Convert model to half
    typed.transform("f32-to-f16")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F16");

    // Convert back to f32
    typed.transform("f16-to-f32")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F32");
    Ok(())
}

#[test]
fn test_typed_model_to_nnef_and_back() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let typed = model.into_tract()?;
    let dir = tempfile::tempdir()?;
    let nnef = nnef()?.with_tract_core()?;

    let path = dir.path().join("nnef-dir");
    nnef.write_model_to_dir(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");

    let path = dir.path().join("nnef.tar");
    nnef.write_model_to_tar(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");

    let path = dir.path().join("nnef.tar.gz");
    nnef.write_model_to_tar_gz(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");
    Ok(())
}

#[test]
fn test_cost() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    let profile = model.cost_json()?;
    let json: serde_json::Value = serde_json::from_str(&profile)?;
    let nodes = json.get("nodes").unwrap().as_array().unwrap();
    assert!(nodes.len() > 10);
    let node = nodes[0].as_object().unwrap();
    assert!(node["node_name"].as_str().unwrap() != "");
    assert!(node["op_name"].as_str().unwrap() != "");
    assert!(
        nodes
            .iter()
            .find_map(|n| n.get("cost").and_then(|c| c.as_object().unwrap().get("FMA(F32)")))
            .is_some()
    );
    Ok(())
}

#[test]
fn test_profile() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    let data = ndarray::ArrayD::<f32>::zeros(vec![1, 3, 224, 224]);
    let states: Option<Vec<Value>> = None;
    let profile = model.profile_json(Some([data]), states)?;
    let profile: serde_json::Value = serde_json::from_str(&profile)?;
    let profiling_info = profile["profiling_info"].as_object().unwrap();
    assert!(profiling_info["iterations"].as_i64().unwrap() >= 1);
    let nodes = profile.get("nodes").unwrap().as_array().unwrap();
    assert!(nodes.iter().find_map(|n| n.get("secs_per_iter").and_then(|c| c.as_f64())).is_some());
    Ok(())
}

#[test]
fn test_transform_registry() -> anyhow::Result<()> {
    ensure_models()?;

    let nnef = nnef()?.with_tract_core()?;
    let mut model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;

    // Convert model to half
    model.transform("f32-to-f16")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F16");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F16");

    // Convert back to f32
    model.transform("f16-to-f32")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F32");
    Ok(())
}

#[test]
fn test_fact_and_dims() -> anyhow::Result<()> {
    ensure_models()?;
    let nnef = nnef()?.with_tract_core()?;
    let model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    let fact = model.parse_fact("B,S+P,64,f32")?;
    assert_eq!(fact.datum_type()?, f32::datum_type());
    assert_eq!(fact.rank()?, 3);
    assert_eq!(fact.dim(1)?.to_string(), "S+P");
    let s_plus_p = fact.dim(1)?;
    let s_plus_twelve = s_plus_p.eval([("P", 12)])?;
    assert_eq!(s_plus_twelve.to_string(), "S+12");
    let fourteen = s_plus_twelve.eval([("S", 2)])?;
    assert_eq!(fourteen.to_int64()?, 14);
    Ok(())
}

#[test]
fn test_fact_and_dims_iterators() -> anyhow::Result<()> {
    ensure_models()?;
    let nnef = nnef()?.with_tract_core()?;
    let model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    let fact = model.input_facts()?.collect::<Vec<_>>();
    assert!(fact.len() == 1);
    let dims = fact[0].dims()?.collect::<Vec<_>>();
    assert_eq!(dims.len(), 4);
    assert_eq!(dims[0].to_string(), "1");
    assert_eq!(dims[1].to_string(), "3");
    assert_eq!(dims[2].to_string(), "224");
    assert_eq!(dims[3].to_string(), "224");
    Ok(())
}

#[test]
fn test_runtime_fact_iterator() -> anyhow::Result<()> {
    ensure_models()?;
    let nnef = nnef()?.with_tract_core()?;
    let runnable = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    let inputs = runnable.input_facts()?.collect::<Vec<_>>();
    assert!(inputs.len() == 1);
    assert_eq!(inputs[0].to_string(), "1,3,224,224,F32");
    let outputs = runnable.output_facts()?.collect::<Vec<_>>();
    assert!(outputs.len() == 1);
    assert_eq!(outputs[0].to_string(), "1,1000,F32");
    Ok(())
}

#[test]
fn test_value_methods() -> anyhow::Result<()> {
    let floats: Value = ndarray::prelude::arr1(&[-1f32, -0.3, 0., 0.25, 0.75, 1.2]).try_into()?;
    assert!(floats.datum_type()?.is_float());
    let ints = floats.convert_to(i8::datum_type())?;
    assert!(ints.datum_type()?.is_signed());
    assert_eq!(ints.view::<i8>()?.as_slice().unwrap(), &[-1, 0, 0, 0, 0, 1]);
    let same: Value = ndarray::prelude::arr1(&[-1f32, -0.3, 0., 0.25, 0.75, 1.2]).try_into()?;
    assert_eq!(floats, same);
    Ok(())
}

fn state_init_from_facts(
    facts: Vec<Fact>,
    default_symbol_value: usize,
) -> Vec<ndarray::ArrayD<f32>> {
    let mut state_initializers = vec![];
    for fact in facts {
        let fact = fact.to_string();
        let mut parsed = fact.split(',').collect::<Vec<_>>();

        let dt = parsed.pop().unwrap();
        let dims = parsed
            .iter()
            .map(|p| {
                //Set symbols to 4
                p.parse::<usize>().unwrap_or(default_symbol_value)
            })
            .collect::<Vec<usize>>();

        assert_eq!(dt, "F32");
        let tensor = ndarray::ArrayD::<f32>::zeros(dims);
        state_initializers.push(tensor);
    }
    state_initializers
}

#[test]
#[ignore = "Model need to be downloaded locally (use .travis/test-llm.sh)"]
fn test_state_init() -> anyhow::Result<()> {
    let nnef = nnef()?.with_tract_core()?.with_tract_transformers()?;
    let mut model = nnef.load("TinyLlama--TinyLlama_v1.1-q40ef32.nnef.tgz")?;

    // Do KV Cache optim
    model.transform("detect-kv-cache")?;
    assert_eq!(model.input_count()?, 1);

    let mut state = model.into_runnable()?.spawn_state()?;

    let state_initializers = state_init_from_facts(state.get_states_facts()?, 4);
    state.set_states(state_initializers.clone())?;

    let mut out_states = state.get_states()?;
    for v in state_initializers {
        let s = out_states.remove(0);
        assert_eq!(s.view::<f32>()?, v);
    }
    Ok(())
}

#[test]
#[ignore = "Model need to be downloaded locally (use .travis/test-llm.sh)"]
fn test_profile_with_state_init() -> anyhow::Result<()> {
    let nnef = nnef()?.with_tract_core()?.with_tract_transformers()?;
    let mut model = nnef.load("TinyLlama--TinyLlama_v1.1-q40ef32.nnef.tgz")?;

    let input = ndarray::ArrayD::<i64>::zeros(vec![1, 1]);
    let state_initializers: Vec<ndarray::ArrayD<f32>> = (1..model.input_count()?)
        .map(|_| ndarray::ArrayD::<f32>::zeros(vec![1, 4, 4, 64]).into())
        .collect();

    // Do KV Cache optim
    model.transform("detect-kv-cache")?;
    assert_eq!(model.input_count()?, 1);

    let model = model.into_runnable()?;
    model.profile_json(Some([input]), Some(state_initializers))?;

    Ok(())
}
