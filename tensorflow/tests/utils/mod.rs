use tract_core::prelude::*;

pub fn compare<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, Tensor)>,
    output: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    compare_optim(graph, &inputs, output, false)?;
    compare_optim(graph, &inputs, output, true)?;
    Ok(())
}

pub fn compare_optim<S: AsRef<str>>(
    graph: &[u8],
    inputs: &Vec<(S, Tensor)>,
    output: &str,
    optim: bool,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    // Run TFD
    let mut model = tract_tensorflow::tensorflow().model_for_read(&mut &*graph)?;
    model.set_input_names(&inputs.iter().map(|pair| pair.0.as_ref()).collect::<Vec<&str>>())?;
    model.set_output_names(&[output])?;
    for (ix, (_, tf)) in inputs.iter().enumerate() {
        model.set_input_fact(ix, TensorFact::from(tf.clone()))?;
    }
    let mut model = model.into_typed()?;
    if optim {
        model = model.into_optimized()?;
    }
    let plan = SimplePlan::new(&model)?;
    let mut state = SimpleState::new(&plan)?;
    for (ix, (_, t)) in inputs.iter().enumerate() {
        state.set_input(ix, t.clone()).unwrap();
    }
    let output = model.node_by_name(output)?;
    info!("Checking {} behaviour against tensorflow", output.name);
    state.compute_one(output.id)?;
    let found = &state.values[output.id].as_ref().unwrap();

    // Run SharedTensor
    let tf_inputs: Vec<(&str, Tensor)> =
        inputs.iter().map(|(s, m)| (s.as_ref(), m.clone())).collect();
    let expected =
        tract_tensorflow::conform::tf::for_slice(&graph)?.run(tf_inputs.clone(), &output.name)?;

    prop_assert!(
        expected[0].shape() == found[0].shape() && expected[0].close_enough(&found[0], true),
        "expected: {:?} found: {:?}",
        expected[0].to_array_view::<f32>().unwrap(),
        found[0].to_array_view::<f32>().unwrap(),
    );
    Ok(())
}

#[allow(dead_code)]
pub fn infer<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, Tensor)>,
    output: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    // Run TFD
    let mut model = tract_tensorflow::tensorflow().model_for_read(&mut &*graph)?;
    model.set_input_names(&inputs.iter().map(|pair| pair.0.as_ref()).collect::<Vec<&str>>())?;
    model.set_output_names(&[output])?;
    let plan = SimplePlan::new(&model)?;
    let mut state = SimpleState::new(&plan)?;
    for (ix, (_, t)) in inputs.iter().enumerate() {
        state.set_input(ix, t.clone()).unwrap();
    }
    let output = model.node_by_name(output)?;
    info!("Checking {} behaviour against tensorflow", output.name);
    state.compute_one(output.id)?;
    let _found = &state.values[output.id].as_ref().unwrap();

    info!("Checking inference consistency on {}", output.name);
    let input_vectors: TVec<TensorFact> = output
        .inputs
        .iter()
        .map(|outlet| {
            state.values[outlet.node].as_ref().unwrap()[outlet.slot].as_tensor().clone().into()
        })
        .collect();
    let output_vectors: TVec<TensorFact> =
        tvec![state.values[output.id].as_ref().unwrap()[0].as_tensor().clone().into(),];

    let input_facts = input_vectors.iter().collect();
    let output_facts = output_vectors.iter().collect();

    let e = output.op.infer_facts(input_facts, output_facts);
    prop_assert!(e.is_ok(), "{:?}", e);

    Ok(())
}
