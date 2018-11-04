use errors::*;
use tract_core::ops::prelude::*;
use tract_core::SimplePlan;
use Parameters;

pub fn handle(params: Parameters, assert_outputs: Option<Vec<TensorFact>>) -> CliResult<()> {
    let outputs = if params.pulse_facts.is_some() {
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

fn run_regular(params: Parameters) -> CliResult<TVec<Tensor>> {
    let tract = params.tract_model;
    let plan = SimplePlan::new(&tract)?;
    let mut inputs: TVec<DtArray> = tvec!();
    for (ix, input) in tract.inputs()?.iter().enumerate() {
        if let Some(input) = params
            .inputs
            .as_ref()
            .and_then(|v| v.get(ix))
            .and_then(|t| t.as_ref())
        {
            inputs.push(input.as_tensor().to_owned());
        } else {
            let fact = tract.fact(*input)?;
            inputs.push(::tensor::tensor_for_fact(fact, None)?);
        }
    }
    info!("Running");
    Ok(plan.run(inputs)?)
}

fn run_pulse(params: Parameters) -> CliResult<TVec<Tensor>> {
    let (input_fact, output_fact) = params.pulse_facts.unwrap();
    let output_pulse = output_fact.pulse();
    //    println!("output_fact: {:?}", output_fact);
    let axis = input_fact.axis;
    let input: &DtArray = &params.inputs.as_ref().unwrap()[0].as_ref().unwrap();
    //    println!("input_shape: {:?}", input.shape());
    let input_dim = input.shape()[axis];
    //    println!("output_fact: {:?}", output_fact);
    let output_dim = output_fact.dim.eval(input_dim as i32).unwrap() as i32;
    let mut output_shape = output_fact.shape.to_vec();
    output_shape[output_fact.axis] =
        output_dim as usize + output_fact.delay + 4 * output_fact.pulse();
    let plan = SimplePlan::new(&params.tract_model)?;
    let mut state = ::tract_core::plan::SimpleState::new(&plan)?;
    //    println!("output_shape: {:?}", output_shape);
    let pulse = input_fact.pulse();
    let mut result = ::ndarray::ArrayD::<f32>::default(output_shape);
    for (ix, chunk) in input.axis_chunks(axis, pulse)?.into_iter().enumerate() {
        let input = if chunk.shape()[input_fact.axis] < pulse {
            let mut chunk_shape = chunk.shape().to_vec();
            chunk_shape[input_fact.axis] = pulse;
            let mut padded_chunk = ::ndarray::ArrayD::<f32>::default(chunk_shape);
            padded_chunk
                .slice_axis_mut(
                    ::ndarray::Axis(input_fact.axis),
                    (..chunk.shape()[input_fact.axis]).into(),
                ).assign(&chunk.to_array_view::<f32>()?);
            padded_chunk.into()
        } else {
            chunk
        };
        let mut outputs = state.run(tvec!(input))?;
        let result_chunk = outputs[0].to_array_view::<f32>()?;
        result
            .slice_axis_mut(
                ::ndarray::Axis(output_fact.axis),
                ((output_pulse * ix)..(output_pulse * (ix + 1))).into(),
            ).assign(&result_chunk);
    }
    result.slice_axis_inplace(
        ::ndarray::Axis(output_fact.axis),
        (output_fact.delay..).into(),
    );
    result.slice_axis_inplace(
        ::ndarray::Axis(output_fact.axis),
        (..output_dim as usize).into(),
    );
    Ok(tvec!(result.into()))
}
