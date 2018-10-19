use std::borrow::Borrow;
use std::sync::Arc;
use std::thread;

use log::Level::Info;

use display_graph::DisplayOptions;
use errors::*;
use format::*;
use ndarray::Axis;
use profile::ProfileData;
use rusage::{Duration, Instant};
use tfdeploy::prelude::op::*;
use tfdeploy::{Model, Tensor};
use {Parameters, ProfilingMode};

fn build_streaming_plan(params: &Parameters) -> CliResult<(StreamingPlan<&Model>, Tensor)> {
    let start = Instant::now();
    let model = &params.tfd_model;
    let input = model.inputs()?[0];
    let input = model.fact(input)?;

    let plan = StreamingPlan::new(&params.tfd_model)?;

    let measure = Duration::since(&start, 1);
    info!(
        "Initialized the StreamingModelState in: {}",
        dur_avg_oneline(measure)
    );

    let chunk_shape = input
        .shape
        .dims
        .iter()
        .map(|d| {
            d.concretize()
                .and_then(|d| d.to_integer().ok())
                .unwrap_or(1) as usize
        }).collect::<Vec<_>>();
    let chunk = ::tensor::random(chunk_shape, input.datum_type.concretize().unwrap());

    Ok((plan, chunk))
}

// feed the network until it outputs something
fn bufferize<P: Borrow<StreamingPlan<M>>, M: Borrow<Model>>(
    state: &mut StreamingModelState<M, P>,
    chunk: &Tensor,
) -> CliResult<()> {
    let buffering = Instant::now();
    info!("Buffering...");
    let mut buffered = 0;
    loop {
        let result = state.step(chunk.clone())?;
        if result.len() != 0 {
            break;
        }
        buffered += 1;
    }
    info!(
        "Buffered {} chunks in {}",
        buffered,
        dur_avg_oneline(Duration::since(&buffering, 1))
    );
    Ok(())
}

pub fn handle_bench(
    params: Parameters,
    profiling: ProfilingMode,
    _display_options: DisplayOptions,
) -> CliResult<()> {
    let (max_iters, max_time) = if let ProfilingMode::StreamBenching {
        max_iters,
        max_time,
    } = profiling
    {
        (max_iters, max_time)
    } else {
        bail!("Expecting bench profile mode")
    };
    let (plan, chunk) = build_streaming_plan(&params)?;
    let mut state = StreamingModelState::new(plan)?;
    bufferize(&mut state, &chunk)?;

    info!("Starting bench itself {} {}", max_time, max_iters);
    let start = Instant::now();
    let mut fed = 0;
    let mut read = 0;
    while fed < max_iters && start.elapsed_real() < (max_time as f64 * 1e-3) {
        let result = state.step(chunk.clone())?;
        read += result.len();
        fed += 1;
    }

    println!(
        "Fed {} chunks, obtained {}. {}",
        fed,
        read,
        dur_avg_oneline(Duration::since(&start, fed))
    );
    Ok(())
}

pub fn handle_cruise(params: Parameters, display_options: DisplayOptions) -> CliResult<()> {
    let (plan, chunk) = build_streaming_plan(&params)?;
    let plan = Arc::new(plan);
    let mut state = StreamingModelState::new(Arc::clone(&plan))?;
    bufferize(&mut state, &chunk)?;

    let mut profile = ProfileData::new(state.model());
    for _ in 0..100 {
        let _result = state.step_wrapping_ops(chunk.clone(), |node, input, buffer| {
            let start = Instant::now();
            let r = node.op.step(input, buffer)?;
            profile.add(node, Duration::since(&start, 1))?;
            Ok(r)
        });
    }

    profile.print_most_consuming_nodes(plan.model(), &params.graph, display_options)?;
    println!();

    profile.print_most_consuming_ops(plan.model())?;

    Ok(())
}

/// Handles the `profile` subcommand when there are streaming dimensions.
pub fn handle_buffering(params: Parameters, display_options: DisplayOptions) -> CliResult<()> {
    let start = Instant::now();
    info!("Initializing the StreamingModelState.");
    let (plan, _chunk) = build_streaming_plan(&params)?;
    let measure = Duration::since(&start, 1);
    info!(
        "Initialized the StreamingModelState in: {}",
        dur_avg_oneline(measure)
    );

    let input = params.tfd_model.input_fact()?;

    let info = input.stream_info()?.expect("No streaming dim");

    let plan = Arc::new(plan);

    let mut states = (0..100)
        .map(|_| StreamingModelState::new(Arc::clone(&plan)).unwrap())
        .collect::<Vec<_>>();

    if log_enabled!(Info) {
        println!();
        print_header(format!("Streaming profiling for {}:", params.name), "white");
    }

    let shape = input
        .shape
        .dims
        .iter()
        .map(|d| {
            d.concretize()
                .map(|d| d.to_integer().unwrap())
                .unwrap_or(20) as usize
        }).collect::<Vec<_>>();
    let data = input
        .concretize()
        .unwrap_or_else(|| ::tensor::random(shape, input.datum_type.concretize().unwrap()));

    // Split the input data into chunks along the streaming axis.
    macro_rules! split_inner {
        ($constr:path, $array:expr) => {{
            $array
                .axis_iter(Axis(info.axis))
                .map(|v| $constr(v.insert_axis(Axis(info.axis)).to_owned()))
                .collect::<Vec<_>>()
        }};
    }

    let chunks = match data {
        Tensor::Bool(m) => split_inner!(Tensor::Bool, m),
        Tensor::F32(m) => split_inner!(Tensor::F32, m),
        Tensor::F64(m) => split_inner!(Tensor::F64, m),
        Tensor::I8(m) => split_inner!(Tensor::I8, m),
        Tensor::I16(m) => split_inner!(Tensor::I16, m),
        Tensor::I32(m) => split_inner!(Tensor::I32, m),
        Tensor::I64(m) => split_inner!(Tensor::I64, m),
        Tensor::U8(m) => split_inner!(Tensor::U8, m),
        Tensor::U16(m) => split_inner!(Tensor::U16, m),
        Tensor::TDim(m) => split_inner!(Tensor::TDim, m),
        Tensor::String(m) => split_inner!(Tensor::String, m),
    };

    let mut profile = ProfileData::new(&plan.model());

    for (step, chunk) in chunks.into_iter().enumerate() {
        trace!("Starting step {:?} with input {:?}.", step, chunk);

        let mut input_chunks = vec![Some(chunk.clone()); 100];
        let mut outputs = Vec::with_capacity(100);
        let start = Instant::now();

        for i in 0..100 {
            outputs.push(states[i].step_wrapping_ops(
                input_chunks[i].take().unwrap(),
                |node, input, buffer| {
                    let start = Instant::now();
                    let r = node.op.step(input, buffer)?;
                    profile.add(node, Duration::since(&start, 1))?;
                    Ok(r)
                },
            ));
        }

        let measure = Duration::since(&start, 100);
        println!(
            "Completed step {:2} with output {:?} in: {}",
            step,
            outputs[0],
            dur_avg_oneline(measure)
        );
        thread::sleep(::std::time::Duration::from_secs(1));
    }

    println!();
    print_header(format!("Summary for {}:", params.name), "white");

    profile.print_most_consuming_nodes(&plan.model(), &params.graph, display_options)?;
    println!();

    profile.print_most_consuming_ops(&plan.model())?;

    Ok(())
}
