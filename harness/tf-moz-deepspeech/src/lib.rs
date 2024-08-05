#![allow(unused_imports, dead_code)]

#[macro_use]
extern crate log;

use std::fs;
use std::path;
use std::rc::Rc;
use std::str::FromStr;

use tract_tensorflow::model::TfModelAndExtensions;
use tract_tensorflow::prelude::*;
use tract_tensorflow::tract_core::internal::*;

#[cfg(test)]
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
        bail!("Failed to download inception model files")
    }
    Ok(())
}

fn parse_tensor<T: Datum + FromStr>(s: &str) -> TractResult<Tensor> {
    let mut tokens = s.split(' ');
    let _name = tokens.next().unwrap();
    let shape = tokens.next().unwrap();
    let shape = &shape[1..shape.len() - 1];
    let shape: Vec<usize> = shape.split(',').map(|s| s.parse().unwrap()).collect();
    Ok(tract_ndarray::Array1::from(tokens.filter_map(|s| s.parse::<T>().ok()).collect::<Vec<_>>())
        .into_shape_with_order(shape)?
        .into())
}

fn parse_scalar<T: Datum + FromStr>(s: &str) -> TractResult<Tensor> {
    let mut tokens = s.split(' ');
    let _name = tokens.next().unwrap();
    let value: T = tokens.next().unwrap().parse().map_err(|_| format_err!("parsing"))?;
    Ok(tensor0(value))
}

fn cachedir() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or_else(|| "../../.cached".to_string()).into()
}

fn initialized_model() -> TractResult<TypedModel> {
    download();
    let tf = tract_tensorflow::tensorflow();
    let graph =
        tf.read_frozen_model(&mut std::fs::File::open(cachedir().join("deepspeech-0.4.1.pb"))?)?;
    let TfModelAndExtensions(mut model, mut extensions) = tf.parse_graph(&graph)?;
    extensions.initializing_nodes = vec![model.node_id_by_name("initialize_state")?];
    model.set_input_names(["input_node", "input_lengths"])?;
    model.set_input_fact(0, f32::fact([1, 16, 19, 26]).into())?;
    model.set_input_fact(1, i32::fact([1]).into())?;
    model.set_output_names(["logits", "Assign_2", "Assign_3"])?;

    extensions.preproc(model)?.into_typed()
}

#[test]
#[ignore]
fn deepspeech_raw() -> TractResult<()> {
    setup_test_logger();
    let model = initialized_model()?;

    let logits = model.node_id_by_name("logits")?;
    let lstm = model.node_id_by_name("lstm_fused_cell/BlockLSTM")?;
    let assign_2 = model.node_id_by_name("Assign_2")?;
    let assign_3 = model.node_id_by_name("Assign_3")?;

    let plan = SimplePlan::build(
        model,
        &[logits.into(), (lstm, 1).into(), (lstm, 6).into(), assign_2.into(), assign_3.into()],
        &[],
        &PlanOptions::default(),
    )?;

    let mut state = SimpleState::new(plan)?;

    let mut inputs = tvec!(tensor0(0).into_tvalue(), tensor0(1).into());
    let mut h = None;
    let mut cs = None;
    let mut logits = None;

    for line in fs::read_to_string(cachedir().join("deepspeech-0.4.1-smoketest.txt"))?.split('\n') {
        if line.starts_with("INPUT_NODE") {
            let tensor = parse_tensor::<f32>(line)?;
            inputs[0] = tensor.into();
        }
        if line.starts_with("INPUT_LENGTH") {
            let length = parse_scalar::<i32>(line)?;
            inputs[1] = tensor1(&[*length.to_scalar::<i32>()?]).into();
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
            let mut outputs = state.run(inputs.clone())?.into_iter();
            let (logits_, cs_, h_) =
                (outputs.next().unwrap(), outputs.next().unwrap(), outputs.next().unwrap());
            h.take().unwrap().close_enough(&h_, true).unwrap();
            cs.take().unwrap().close_enough(&cs_, true).unwrap();
            logits.take().unwrap().close_enough(&logits_, true).unwrap();
            println!("chunk ok");
        }
    }

    Ok(())
}

#[test]
#[ignore]
fn deepspeech_decluttered() -> TractResult<()> {
    deepspeech_run(false)
}

#[test]
#[ignore]
fn deepspeech_optimized() -> TractResult<()> {
    deepspeech_run(true)
}

#[cfg(test)]
fn deepspeech_run(opt: bool) -> TractResult<()> {
    setup_test_logger();
    let mut model = initialized_model()?.into_decluttered()?;
    if opt {
        model.optimize()?;
    }
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;

    let mut inputs = tvec!(tensor0(0).into(), tensor0(1).into());
    let mut logits = None;

    for line in fs::read_to_string(cachedir().join("deepspeech-0.4.1-smoketest.txt"))?.split('\n') {
        if line.starts_with("INPUT_NODE") {
            let tensor = parse_tensor::<f32>(line)?;
            inputs[0] = tensor.into();
        }
        if line.starts_with("INPUT_LENGTH") {
            let length = parse_scalar::<i32>(line)?;
            inputs[1] = tensor1(&[*length.to_scalar::<i32>()?]).into();
        }
        if line.starts_with("LOGITS") {
            logits = Some(parse_tensor::<f32>(line)?);
        }
        if logits.is_some() {
            let mut outputs = state.run(inputs.clone())?.into_iter();
            let logits_ = outputs.next().unwrap();
            logits.take().unwrap().close_enough(&logits_, true).unwrap();
            println!("chunk ok");
        }
    }

    Ok(())
}
