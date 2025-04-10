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
    let model = onnx()?.model_for_path("mobilenetv2-7.onnx")?.into_optimized()?.into_runnable()?;
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
    let model = onnx()?.model_for_path("mobilenetv2-7.onnx")?.into_optimized()?.into_runnable()?;
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
    let model = nnef()?
        .model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?
        .into_optimized()?
        .into_runnable()?;
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
fn test_inference_model() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    assert_eq!(model.input_count().unwrap(), 1);
    assert_eq!(model.output_count().unwrap(), 1);
    assert_eq!(model.input_name(0).unwrap(), "data");
    assert_eq!(model.output_name(0).unwrap(), "mobilenetv20_output_flatten0_reshape0");
    assert_eq!(model.input_fact(0).unwrap().to_string(), "1,3,224,224,F32");
    model.set_input_fact(0, "1,3,224,224,F32")?;
    let model = model.into_optimized()?.into_runnable()?;
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
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
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
    let mut model = nnef()?.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    assert_eq!(model.input_count()?, 1);
    assert_eq!(model.output_count()?, 1);
    assert_eq!(model.input_name(0)?, "data");
    assert_eq!(model.output_name(0)?, "mobilenetv20_output_flatten0_reshape0");
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F32");
    model.declutter()?;
    Ok(())
}

#[test]
fn test_set_output_names() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = nnef()?.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    model.set_output_names(["conv_53"])?;
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,1,1,F32");
    Ok(())
}

#[test]
fn test_concretize() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_typed()?.into_decluttered()?;
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
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_typed()?.into_decluttered()?;
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
fn test_f32_to_f16() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_typed()?.into_decluttered()?;
    typed.transform("f32-to-f16")?;
    assert_eq!(typed.input_fact(0)?.to_string(), "B,3,224,224,F16");
    assert_eq!(typed.output_fact(0)?.to_string(), "B,1000,F16");
    Ok(())
}

#[test]
fn test_f16_to_f32() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let mut typed = model.into_typed()?.into_decluttered()?;

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
    let mut model = onnx()?.model_for_path("mobilenetv2-7.onnx")?;
    model.set_input_fact(0, "B,3,224,224,f32")?;
    model.analyse()?;
    let typed = model.into_typed()?;
    let dir = tempfile::tempdir()?;
    let nnef = nnef()?.with_tract_core()?;

    let path = dir.path().join("nnef-dir");
    nnef.write_model_to_dir(&path, &typed)?;
    let reloaded = nnef.model_for_path(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");

    let path = dir.path().join("nnef.tar");
    nnef.write_model_to_tar(&path, &typed)?;
    let reloaded = nnef.model_for_path(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");

    let path = dir.path().join("nnef.tar.gz");
    nnef.write_model_to_tar_gz(&path, &typed)?;
    let reloaded = nnef.model_for_path(path)?;
    assert_eq!(reloaded.input_fact(0)?.to_string(), "B,3,224,224,F32");
    assert_eq!(reloaded.output_fact(0)?.to_string(), "B,1000,F32");
    Ok(())
}

#[test]
fn test_cost() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = nnef()?.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    model.declutter()?;
    model.optimize()?;
    let profile = model.cost_json()?;
    let json: serde_json::Value = serde_json::from_str(&profile)?;
    let nodes = json.get("nodes").unwrap().as_array().unwrap();
    assert!(nodes.len() > 10);
    let node = nodes[0].as_object().unwrap();
    assert!(node["node_name"].as_str().unwrap() != "");
    assert!(node["op_name"].as_str().unwrap() != "");
    assert!(nodes
        .iter()
        .find_map(|n| n.get("cost").and_then(|c| c.as_object().unwrap().get("FMA(F32)")))
        .is_some());
    Ok(())
}

#[test]
fn test_profile() -> anyhow::Result<()> {
    ensure_models()?;
    let mut model = nnef()?.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    model.declutter()?;
    model.optimize()?;
    let data = ndarray::ArrayD::<f32>::zeros(vec![1, 3, 224, 224]);
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
    let mut model = nnef.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")?;
    model.declutter()?;

    // Convert model to half
    nnef.transform_model(&mut model, "f32-to-f16")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F16");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F16");

    // Convert back to f32
    nnef.transform_model(&mut model, "f16-to-f32")?;
    assert_eq!(model.input_fact(0)?.to_string(), "1,3,224,224,F32");
    assert_eq!(model.output_fact(0)?.to_string(), "1,1000,F32");
    Ok(())
}
