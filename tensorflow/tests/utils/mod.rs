use tract_core::prelude::*;
use tract_core::infer::*;

fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Mode {
    Infer,
    Type,
    Declutter,
    Opt,
}

pub fn compare<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, Tensor)>,
    output: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    for mode in &[Mode::Infer, Mode::Type, Mode::Declutter, Mode::Opt] {
        compare_optim(graph, &inputs, output, *mode)?;
    }
    Ok(())
}

pub fn run_tract<S: AsRef<str>>(
    graph: &[u8],
    inputs: &Vec<(S, Tensor)>,
    output: &str,
    mode: Mode,
) -> TractResult<TVec<Arc<Tensor>>> {
    let mut model = tract_tensorflow::tensorflow().model_for_read(&mut &*graph)?;
    model.set_input_names(&inputs.iter().map(|pair| pair.0.as_ref()).collect::<Vec<&str>>())?;
    model.set_output_names(&[output])?;
    for (ix, (_, tf)) in inputs.iter().enumerate() {
        model.set_input_fact(ix, InferenceFact::dt_shape(tf.datum_type(), tf.shape()))?;
    }
    debug!("analysed");
    let inputs = inputs.iter().map(|pair| pair.1.clone()).collect();
    if mode == Mode::Infer {
        let plan = SimplePlan::new(&model)?;
        plan.run(inputs)
    } else {
        let mut model = model.into_typed()?;
        debug!("typed");
        if mode == Mode::Declutter {
            model = model.declutter()?;
            debug!("decluttered");
        } else if mode == Mode::Opt {
            model = model.into_optimized()?;
            debug!("optimized");
        };
        trace!("{:#?}", model);
        let plan = SimplePlan::new(&model)?;
        plan.run(inputs)
    }
}

pub fn compare_optim<S: AsRef<str>>(
    graph: &[u8],
    inputs: &Vec<(S, Tensor)>,
    output: &str,
    mode: Mode,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    setup_test_logger();
    let tf_inputs: Vec<(&str, Tensor)> =
        inputs.iter().map(|(s, m)| (s.as_ref(), m.clone())).collect();
    let expected =
        tract_tensorflow::conform::tf::for_slice(&graph)?.run(tf_inputs.clone(), &output)?;
    info!("Mode: {:?} starting", mode);
    info!("Tensorflow says: {:?}", expected);

    let found = match run_tract(graph, inputs, output, mode) {
        Err(e) => {
            use crate::tract_core::error_chain::ChainedError;
            error!("{}", e.display_chain());
            return Err(e.into());
        }
        Ok(t) => t,
    };

    if let Err(e) = expected[0].close_enough(&found[0], true) {
        error!("{:?} (mode: {:?})", e, mode);
        error!("Tensorflow says: {:?}", expected);
        error!("Tract says     : {:?}", found);
        Err(e)?
    } else {
        info!("Mode: {:?} passed", mode);
        Ok(())
    }
}

#[allow(dead_code)]
pub fn infer<S: AsRef<str>>(
    graph: &[u8],
    inputs: Vec<(S, Tensor)>,
    output_str: &str,
) -> std::result::Result<(), ::proptest::test_runner::TestCaseError> {
    setup_test_logger();
    let mut model = tract_tensorflow::tensorflow().model_for_read(&mut &*graph)?;
    model.set_input_names(&inputs.iter().map(|pair| pair.0.as_ref()).collect::<Vec<&str>>())?;
    model.set_output_names(&[output_str])?;
    for (ix, (_, tf)) in inputs.iter().enumerate() {
        model.set_input_fact(ix, InferenceFact::dt_shape(tf.datum_type(), tf.shape()))?;
    }
    let plan = SimplePlan::new(&model)?;
    let mut state = SimpleState::new(&plan)?;
    for (ix, (_, t)) in inputs.iter().enumerate() {
        state.set_input(ix, t.clone()).unwrap();
    }
    let output = model.node_by_name(output_str)?;
    info!("Checking {} behaviour against tensorflow", output.name);
    state.compute_recursively(output.id)?;
    let _found = &state.values[output.id].as_ref().unwrap();

    info!("Checking inference consistency on {}", output.name);
    let input_vectors: TVec<InferenceFact> = output
        .inputs
        .iter()
        .map(|outlet| {
            state.values[outlet.node].as_ref().unwrap()[outlet.slot]
                .clone()
                .into_tensor()
                .clone()
                .into()
        })
        .collect();
    let output_vectors: TVec<InferenceFact> =
        tvec![state.values[output.id].as_ref().unwrap()[0].clone().into_tensor().clone().into(),];

    let input_facts = input_vectors.iter().collect();
    let output_facts = output_vectors.iter().collect();

    let output = model.node_by_name_mut(output_str)?;
    let e = output.op.infer_facts(input_facts, output_facts, tvec!());
    prop_assert!(e.is_ok(), "{:?}", e);

    Ok(())
}
