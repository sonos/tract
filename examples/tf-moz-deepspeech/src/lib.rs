#![allow(unused_imports,dead_code)]
use std::fs;
use std::path;
use std::str::FromStr;
use std::rc::Rc;

use ndarray::*;

use tract_core::datum::Datum;
use tract_core::model::OutletId;
use tract_core::*;

fn download() {
    use std::sync::{Once, ONCE_INIT};
    static START: Once = ONCE_INIT;

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = ::std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        Err("Failed to download inception model files")?
    }
    Ok(())
}

fn parse_tensor<T: Datum + FromStr>(s: &str) -> TractResult<Tensor> {
    let mut tokens = s.split(" ");
    let _name = tokens.next().unwrap();
    let shape = tokens.next().unwrap();
    let shape = &shape[1..shape.len() - 1];
    let shape: Vec<usize> = shape.split(",").map(|s| s.parse().unwrap()).collect();
    Ok(ndarray::Array1::from_iter(tokens.filter_map(|s| s.parse::<T>().ok()))
        .into_shape(shape)?
        .into())
}

fn parse_scalar<T: Datum + FromStr>(s: &str) -> TractResult<Tensor> {
    let mut tokens = s.split(" ");
    let _name = tokens.next().unwrap();
    let value: T = tokens.next().unwrap().parse().map_err(|_| "foo")?;
    Ok(arr0(value).into())
}

fn cachedir() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or("../../.cached".to_string()).into()
}

#[test]
fn deepspeech() -> TractResult<()> {
    download();
    let tf = tract_tensorflow::tensorflow();
//    let mut model = tf.model_for_path("deepspeech-0.4.1-models/output_graph.pb")?;
    let mut model = tf.model_for_path(cachedir().join("deepspeech-0.4.1.pb"))?;
    model.set_inputs(&["input_node", "input_lengths"])?;
    model.set_input_fact(0, TensorFact::dt_shape(f32::datum_type(), tvec!(1, 16, 19, 26)))?;
    model.set_input_fact(1, TensorFact::dt_shape(i32::datum_type(), tvec!(1)))?;
    let model = model.into_typed()?;
    let model = Rc::new(model);

    let init_node = model.node_by_name("initialize_state")?.id;
    let init_plan =
        tract_core::SimplePlan::new_for_output(model.clone(), OutletId::new(init_node, 0))?;
    //    let logit_node = model.node_by_name("logits")?.id;
    let logit_node = model.node_by_name("logits")?.id;
    let lstm_node = model.node_by_name("lstm_fused_cell/BlockLSTM")?.id;
    let logit_plan = tract_core::SimplePlan::new_for_outputs(
        model.clone(),
        &[OutletId::new(logit_node, 0), OutletId::new(lstm_node, 1), OutletId::new(lstm_node, 6)],
    )?;

    let mut state = tract_core::SimpleState::new_multiplan(vec![init_plan, logit_plan])?;

    // initialize_state
    state.run_plan(tvec!(), 0)?;
    let mut inputs = tvec!(Tensor::from(arr0(0)), Tensor::from(arr0(1)));
    let mut h = None;
    let mut cs = None;
    let mut logits = None;

    for line in fs::read_to_string(cachedir().join("deepspeech-0.4.1-smoketest.txt"))?.split("\n") {
        if line.starts_with("INPUT_NODE") {
            let tensor = parse_tensor::<f32>(line)?;
            inputs[0] = tensor;
        }
        if line.starts_with("INPUT_LENGTH") {
            let length = parse_scalar::<i32>(line)?;
            inputs[1] = arr1(&[*length.to_scalar::<i32>()?]).into();
        }
        if line.starts_with("H:") {
            h = Some(parse_tensor::<f32>(line)?);
        }
        if line.starts_with("CS:") {
            cs = Some(parse_tensor::<f32>(line)?);
        }
        if line.starts_with("LOGITS") {
            logits = Some(parse_tensor::<f32>(line)?);
        }
        if h.is_some() && cs.is_some() && logits.is_some() {
            let mut outputs = state.run_plan(inputs.clone(), 1)?;
            let (logits_, cs_, h_) = args_3!(outputs);
            assert!(h.take().unwrap().close_enough(&h_, true));
            assert!(cs.take().unwrap().close_enough(&cs_, true));
            assert!(logits.take().unwrap().close_enough(&logits_, true));
            println!("chunk ok");
        }
    }

    Ok(())
}
