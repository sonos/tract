#![allow(unused_imports, dead_code)]

#[macro_use]
extern crate log;

use std::fs;
use std::path;
use std::rc::Rc;
use std::str::FromStr;

use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray;

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

fn download() {
    use std::sync::Once;
    static START: Once = Once::new();

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
    Ok(ndarray::Array1::from(tokens.filter_map(|s| s.parse::<T>().ok()).collect::<Vec<_>>())
        .into_shape(shape)?
        .into())
}

fn parse_scalar<T: Datum + FromStr>(s: &str) -> TractResult<Tensor> {
    let mut tokens = s.split(" ");
    let _name = tokens.next().unwrap();
    let value: T = tokens.next().unwrap().parse().map_err(|_| "foo")?;
    Ok(tensor0(value))
}

fn cachedir() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or("../../.cached".to_string()).into()
}

#[ignore]
#[test]
fn deepspeech() -> TractResult<()> {
    setup_test_logger();
    download();
    let tf = tract_tensorflow::tensorflow();
    //    let mut model = tf.model_for_path("deepspeech-0.4.1-models/output_graph.pb")?;
    let mut model = tf.model_for_path(cachedir().join("deepspeech-0.4.1.pb"))?;
    model.set_input_names(&["input_node", "input_lengths"])?;
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 16, 19, 26)))?;
    model.set_input_fact(1, InferenceFact::dt_shape(i32::datum_type(), tvec!(1)))?;
    model.set_output_names(&["initialize_state", "logits"])?;
    trace!("{:#?}", &model);
    let model = model.into_typed()?;
    let model = Rc::new(model);
    trace!("{:#?}", &model);

    let init_node = model.node_by_name("initialize_state")?.id;
    let init_plan = SimplePlan::new_for_output(model.clone(), OutletId::new(init_node, 0))?;
    //    let logit_node = model.node_by_name("logits")?.id;
    let logit_node = model.node_by_name("logits")?.id;
    let lstm_node = model.node_by_name("lstm_fused_cell/BlockLSTM")?.id;
    let logit_plan = SimplePlan::new_for_outputs(
        model.clone(),
        &[OutletId::new(logit_node, 0), OutletId::new(lstm_node, 1), OutletId::new(lstm_node, 6)],
    )?;

    let mut state = SimpleState::new_multiplan(vec![init_plan, logit_plan])?;

    // initialize_state
    state.run_plan(tvec!(), 0)?;
    let mut inputs = tvec!(tensor0(0), tensor0(1));
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
            inputs[1] = tensor1(&[*length.to_scalar::<i32>()?]);
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
            h.take().unwrap().close_enough(&h_, true).unwrap();
            cs.take().unwrap().close_enough(&cs_, true).unwrap();
            logits.take().unwrap().close_enough(&logits_, true).unwrap();
            println!("chunk ok");
        }
    }

    Ok(())
}
