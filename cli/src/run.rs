use crate::errors::*;
use crate::Parameters;
use tract_core::ops::prelude::*;
use tract_core::SimplePlan;

pub fn handle(params: Parameters) -> CliResult<()> {
    let outputs = if params.pulse_facts.is_some() {
        run_pulse(&params)?
    } else {
        run_regular(&params)?
    };

    for (ix, output) in outputs.iter().enumerate() {
        println!("output #{}\n{}\n", ix, output.dump(true)?);
    }

    if let Some(asserts) = &params.assertions {
        if let Some(asserts) = &asserts.assert_outputs {
            crate::utils::check_outputs(&*outputs, &asserts)?;
        }
        if let Some(facts) = &asserts.assert_output_facts {
            let outputs: Vec<TensorFact> = outputs
                .iter()
                .map(|t| TensorFact::dt_shape(t.datum_type(), t.shape()))
                .collect();
            crate::utils::check_inferred(&*outputs, &*facts)?;
        }
    }

    Ok(())
}

fn run_regular(params: &Parameters) -> CliResult<TVec<SharedTensor>> {
    let tract = &params.tract_model;
    let plan = SimplePlan::new(tract)?;
    let mut inputs: TVec<Tensor> = tvec!();
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
            inputs.push(crate::tensor::tensor_for_fact(fact, None)?);
        }
    }
    info!("Running");
    Ok(plan.run(inputs)?)
}

fn run_pulse(params: &Parameters) -> CliResult<TVec<SharedTensor>> {
    let (input_fact, output_fact) = params.pulse_facts.clone().unwrap();
    let output_pulse = output_fact.pulse();
    //    println!("output_fact: {:?}", output_fact);
    let axis = input_fact.axis;
    let input: &Tensor = &params.inputs.as_ref().unwrap()[0].as_ref().unwrap();
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
    let input = input.to_array_view::<f32>()?;
    for ix in 0..input_dim.div_ceil(pulse) {
        let chunk = input.slice_axis(ndarray::Axis(axis), (ix * pulse..(ix + 1) * pulse).into());
        let input = if chunk.shape()[input_fact.axis] < pulse {
            let mut chunk_shape = chunk.shape().to_vec();
            chunk_shape[input_fact.axis] = pulse;
            let mut padded_chunk = ::ndarray::ArrayD::<f32>::default(chunk_shape);
            padded_chunk
                .slice_axis_mut(
                    ::ndarray::Axis(input_fact.axis),
                    (..chunk.shape()[input_fact.axis]).into(),
                )
                .assign(&chunk);
            padded_chunk
        } else {
            chunk.to_owned()
        };
        let outputs = state.run(tvec!(input.into()))?;
        let result_chunk = outputs[0].to_array_view::<f32>()?;
        result
            .slice_axis_mut(
                ::ndarray::Axis(output_fact.axis),
                ((output_pulse * ix)..(output_pulse * (ix + 1))).into(),
            )
            .assign(&result_chunk);
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
