fn grace_hopper() -> tract_rs::Value {
    let data = std::fs::read("tests/grace_hopper_3_224_224.f32.raw").unwrap();
    let data: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as _, 3 * 224 * 224) };
    tract_rs::Value::from_shape_and_slice(&[1, 3, 224, 224], data).unwrap()
}

#[test]
fn test_nnef() -> anyhow::Result<()> {
    let model = tract_rs::nnef()?
        .model_for_path("tests/mobilenet_v2_1.0.onnx.nnef.tgz")?
        .into_optimized()?
        .into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    let view = ndarray::ArrayView::try_from(&result[0])?;
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
fn test_inference_nnef() -> anyhow::Result<()> {
    let model = tract_rs::onnx()?.model_for_path("tests/mobilenetv2-7.onnx")?;
    assert_eq!(model.input_count().unwrap(), 1);
    assert_eq!(model.output_count().unwrap(), 1);
    assert_eq!(model.input_name(0).unwrap(), "data");
    assert_eq!(model.output_name(0).unwrap(), "mobilenetv20_output_flatten0_reshape0");
    assert_eq!(model.input_fact(0).unwrap().to_string(), "1,3,224,224,F32");
    model.set_input_fact(0, "1,3,224,224,F32");
    /*
        .into_optimized()?
        .into_runnable()?;
    let hopper = grace_hopper();
    let result = model.run([hopper])?;
    let view = ndarray::ArrayView::try_from(&result[0])?;
    let best = view
        .as_slice()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(best.0, 652);
    */
    Ok(())
}
