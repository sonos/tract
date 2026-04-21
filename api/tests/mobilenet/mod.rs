use std::sync::Once;

fn grace_hopper() -> Tensor {
    let data = std::fs::read("../tests/grace_hopper_3_224_224.f32.raw").unwrap();
    let data: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as _, 3 * 224 * 224) };
    Tensor::from_slice(&[1, 3, 224, 224], data).unwrap()
}

fn ensure_models() -> anyhow::Result<()> {
    static START: Once = Once::new();
    START.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
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

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn test_onnx() -> anyhow::Result<()> {
    ensure_models()?;
    let model = onnx()?.load("mobilenetv2-7.onnx")?.into_model()?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    assert_eq!(argmax(result[0].as_slice::<f32>()?), 652);
    Ok(())
}

#[test]
fn test_state() -> anyhow::Result<()> {
    ensure_models()?;
    let model = onnx()?.load("mobilenetv2-7.onnx")?.into_model()?.into_runnable()?;
    let mut state = model.spawn_state()?;
    let hopper = grace_hopper();
    let result = state.run([hopper])?;
    assert_eq!(argmax(result[0].as_slice::<f32>()?), 652);
    Ok(())
}

#[test]
fn test_nnef() -> anyhow::Result<()> {
    ensure_models()?;
    let model = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    assert_eq!(result[0].datum_type()?, f32::datum_type());
    assert_eq!(argmax(result[0].as_slice::<f32>()?), 652);
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
    assert_eq!(model.input_fact(0).unwrap().to_string(), "1,3,224,224,f32");
    model.set_input_fact(0, "1,3,224,224,f32")?;
    let model = model.into_model()?.into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    assert_eq!(argmax(result[0].as_slice::<f32>()?), 652);
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
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,f32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,f32");
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
fn test_concretize() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f32");
    typed.transform(ConcretizeSymbols::new().value("B", 1))?;
    assert_eq!(typed.input_fact(0)?.to_string(), "1,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "1,1000,f32");
    Ok(())
}

#[test]
fn test_concretize_raw_string() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform(r#"{"name":"concretize_symbols","values":{"B":1}}"#)?;
    assert_eq!(typed.input_fact(0)?.to_string(), "1,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "1,1000,f32");
    Ok(())
}

#[test]
fn test_pulse() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f32");
    typed.transform(Pulse::new("5").symbol("B"))?;
    assert_eq!(typed.input_fact(0)?.to_string(), "5,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "5,1000,f32");
    let mut properties = typed.property_keys()?;
    properties.sort();
    assert_eq!(&properties, &["pulse.delay", "pulse.input_axes", "pulse.output_axes"]);
    assert_eq!(typed.property("pulse.delay")?.as_slice::<i64>()?, &[0i64]);
    Ok(())
}

#[test]
fn test_pulse_raw_string() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform(r#"{"name":"pulse","symbol":"B","pulse":"5"}"#)?;
    assert_eq!(typed.input_fact(0)?.to_string(), "5,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "5,1000,f32");
    Ok(())
}

#[test]
fn test_runtime_fact() -> anyhow::Result<()> {
    ensure_models()?;
    let runnable = nnef()?.load("mobilenet_v2_1.0.onnx.nnef.tgz")?.into_runnable()?;
    assert_eq!(runnable.input_fact(0)?.to_string(), "1,3,224,224,f32");
    assert_eq!(runnable.output_fact(0)?.to_string(), "1,1000,f32");
    Ok(())
}

#[test]
fn test_runtime_properties() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform(r#"{"name":"pulse","symbol":"B","pulse":"5"}"#)?;
    let runnable = typed.into_runnable()?;
    let mut properties = runnable.property_keys()?;
    properties.sort();
    assert_eq!(&properties, &["pulse.delay", "pulse.input_axes", "pulse.output_axes"]);
    assert_eq!(runnable.property("pulse.delay")?.as_slice::<i64>()?, &[0i64]);
    Ok(())
}

#[test]
fn test_f32_to_f16() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform(FloatPrecision::new(
        DatumType::F32,
        DatumType::F16,
    ))?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f16");
    Ok(())
}

#[test]
fn test_f32_to_f16_raw_string() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform("f32_to_f16")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f16");
    Ok(())
}

#[test]
fn test_f16_to_f32() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;

    // Convert model to half
    typed.transform(FloatPrecision::new(
        DatumType::F32,
        DatumType::F16,
    ))?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f16");

    // Convert back to f32
    typed.transform(FloatPrecision::new(
        DatumType::F16,
        DatumType::F32,
    ))?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f32");
    Ok(())
}

#[test]
fn test_f16_to_f32_raw_string() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_model()?;
    typed.transform("f32_to_f16")?;
    typed.transform("f16_to_f32")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,f32");
    Ok(())
}

#[test]
fn test_typed_model_to_nnef_and_back() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.load("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let typed = model.into_model()?;
    let dir = tempfile::tempdir()?;
    let nnef = nnef()?.with_tract_core()?;

    let path = dir.path().join("nnef-dir");
    nnef.write_model_to_dir(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,f32");

    let path = dir.path().join("nnef.tar");
    nnef.write_model_to_tar(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,f32");

    let path = dir.path().join("nnef.tar.gz");
    nnef.write_model_to_tar_gz(&path, &typed)?;
    let reloaded = nnef.load(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,f32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,f32");
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
    let data = Tensor::from_slice::<f32>(&[1, 3, 224, 224], &vec![0f32; 1 * 3 * 224 * 224])?;
    let profile = model.profile_json(Some([data]))?;
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
    model.transform("f32_to_f16")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,f16");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,f16");

    // Convert back to f32
    model.transform("f16_to_f32")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,f32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,f32");
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
    assert_eq!(inputs[0].to_string(), "1,3,224,224,f32");
    let outputs = runnable.output_facts()?.collect::<Vec<_>>();
    assert!(outputs.len() == 1);
    assert_eq!(outputs[0].to_string(), "1,1000,f32");
    Ok(())
}

#[test]
fn test_tensor_methods() -> anyhow::Result<()> {
    let floats = Tensor::from_slice::<f32>(&[6], &[-1f32, -0.3, 0., 0.25, 0.75, 1.2])?;
    assert!(floats.datum_type()?.is_float());
    let ints = floats.convert_to(i8::datum_type())?;
    assert!(ints.datum_type()?.is_signed());
    assert_eq!(ints.as_slice::<i8>()?, &[-1, 0, 0, 0, 0, 1]);
    let same = Tensor::from_slice::<f32>(&[6], &[-1f32, -0.3, 0., 0.25, 0.75, 1.2])?;
    assert_eq!(floats, same);
    Ok(())
}
