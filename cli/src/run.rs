use errors::*;
use tfdeploy::ops::prelude::*;
use tfdeploy::SimplePlan;
use Parameters;

pub fn handle(params: Parameters, assert_outputs: Option<Vec<TensorFact>>) -> CliResult<()> {
    let outputs = if params.pulsified_model.is_some() {
        run_pulse(params)?
    } else {
        run_regular(params)?
    };

    for (ix, output) in outputs.iter().enumerate() {
        println!("output #{}\n{}\n", ix, output.dump(true)?);
    }

    if let Some(asserts) = assert_outputs {
        ::utils::check_outputs(&*outputs, &asserts)?;
    }

    Ok(())
}

fn run_regular(params: Parameters) -> CliResult<Vec<Tensor>> {
    let tfd = params.tfd_model;
    let plan = SimplePlan::new(&tfd)?;
    Ok(plan.run(
        params
            .inputs
            .unwrap()
            .iter()
            .map(|t| t.clone().unwrap())
            .collect(),
    )?)
}

fn run_pulse(params: Parameters) -> CliResult<Vec<Tensor>> {
    let pulsified_model = params.pulsified_model.as_ref().unwrap();
    let input_fact = pulsified_model.input_fact()?;
//    println!("input_fact: {:?}", input_fact);
    let output_fact = pulsified_model.output_fact()?;
    let output_pulse = output_fact.pulse();
//    println!("output_fact: {:?}", output_fact);
    let axis = input_fact.axis;
    let input: &Tensor = &params.inputs.as_ref().unwrap()[0].as_ref().unwrap();
//    println!("input_shape: {:?}", input.shape());
    let input_dim = input.shape()[axis];
//    println!("output_fact: {:?}", output_fact);
    let output_dim = output_fact.dim.eval(input_dim as i64).unwrap() as i64;
    let mut output_shape = output_fact.shape.to_vec();
    output_shape[output_fact.axis] = output_dim as usize + output_fact.delay + 4*output_fact.pulse();
    let plan = SimplePlan::new(&pulsified_model.model)?;
    let mut state = ::tfdeploy::plan::SimpleState::new(&plan)?;
//    println!("output_shape: {:?}", output_shape);
    let pulse = input_fact.pulse();
    let mut result = ::ndarray::ArrayD::<f32>::default(output_shape);
    for (ix, chunk) in input.axis_chunks(axis, pulse)?.into_iter().enumerate() {
        state.reset()?;
        if chunk.shape()[input_fact.axis] < pulse {
            let mut chunk_shape = chunk.shape().to_vec();
            chunk_shape[input_fact.axis] = pulse;
            let mut padded_chunk = ::ndarray::ArrayD::<f32>::default(chunk_shape);
            padded_chunk.slice_axis_mut(
                ::ndarray::Axis(input_fact.axis),
                (..chunk.shape()[input_fact.axis]).into()
            ).assign(&chunk.to_array_view::<f32>()?);
            state.set_input(0, padded_chunk.into())?;
        } else {
            state.set_input(0, chunk)?;
        }
        state.eval_all_in_order()?;
        let result_chunk = state.take_outputs()?.remove(0).into_array::<f32>()?;
        result.slice_axis_mut(
            ::ndarray::Axis(output_fact.axis),
            ((output_pulse * ix)..(output_pulse * (ix + 1))).into(),
        ).assign(&result_chunk);
    }
//    println!("must slice: {:?}", output_fact.delay..(output_fact.delay+output_dim as usize));
    result.slice_axis_inplace(::ndarray::Axis(output_fact.axis), (output_fact.delay..).into());
    result.slice_axis_inplace(::ndarray::Axis(output_fact.axis), (..output_dim as usize).into());
    Ok(vec!(result.into()))
}
