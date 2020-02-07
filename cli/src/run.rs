use crate::errors::*;
use crate::{Model, Parameters};
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray;

pub fn handle(params: &Parameters, dump: bool) -> CliResult<()> {
    let outputs = if let Some(pulse) = params.tract_model.downcast_ref::<PulsedModel>() {
        run_pulse_t(pulse, &params)?
    } else {
        dispatch_model!(params.tract_model, |m| run_regular(m, &params))?
    };

    if dump {
        for (ix, output) in outputs.iter().enumerate() {
            println!("output #{}\n{}\n", ix, output.dump(true)?);
        }
    }

    if let Some(asserts) = &params.assertions {
        if let Some(asserts) = &asserts.assert_outputs {
            crate::utils::check_outputs(&*outputs, &asserts)?;
        }
        if let Some(facts) = &asserts.assert_output_facts {
            let outputs: Vec<InferenceFact> = outputs
                .iter()
                .map(|t| InferenceFact::dt_shape(t.datum_type(), t.shape()))
                .collect();
            crate::utils::check_inferred(&*outputs, &*facts)?;
        }
    }

    Ok(())
}

fn run_regular(tract: &dyn Model, params: &Parameters) -> CliResult<TVec<Arc<Tensor>>> {
    let mut inputs: TVec<Tensor> = tvec!();
    for (ix, input) in tract.input_outlets().iter().enumerate() {
        if let Some(input) = params.input_values.get(ix).and_then(|x| x.as_ref()) {
            inputs.push(input.clone().into_tensor())
        } else {
            let fact = tract.outlet_typedfact(*input)?;
            inputs.push(crate::tensor::tensor_for_fact(&fact, None)?);
        }
    }
    Ok(dispatch_model!(tract, |m| SimplePlan::new(m)?.run(inputs))?)
}

fn run_pulse_t(model: &PulsedModel, params: &Parameters) -> CliResult<TVec<Arc<Tensor>>> {
    let input_fact = model.input_fact(0)?;
    let output_fact = model.output_fact(0)?;

    let output_pulse = output_fact.pulse();
    //    println!("output_fact: {:?}", output_fact);
    let axis = input_fact.axis;
    let input: &Tensor = &params.input_values[0].as_ref().unwrap();
    //    println!("input_shape: {:?}", input.shape());
    let input_dim = input.shape()[axis];
    //    println!("output_fact: {:?}", output_fact);
    let output_dim = output_fact.dim.eval(input_dim as i32).unwrap() as i32;
    let mut output_shape = output_fact.shape.to_vec();
    output_shape[output_fact.axis] =
        output_dim as usize + output_fact.delay + 4 * output_fact.pulse();
    let plan = SimplePlan::new(model)?;
    let mut state = ::tract_core::plan::SimpleState::new(&plan)?;
    //    println!("output_shape: {:?}", output_shape);
    let pulse = input_fact.pulse();
    let mut result = ndarray::ArrayD::<f32>::default(output_shape);
    let input = input.to_array_view::<f32>()?;
    for ix in 0..input_dim.div_ceil(pulse) {
        let chunk = input.slice_axis(ndarray::Axis(axis), (ix * pulse..(ix + 1) * pulse).into());
        let input = if chunk.shape()[input_fact.axis] < pulse {
            let mut chunk_shape = chunk.shape().to_vec();
            chunk_shape[input_fact.axis] = pulse;
            let mut padded_chunk = ndarray::ArrayD::<f32>::default(chunk_shape);
            padded_chunk
                .slice_axis_mut(
                    ndarray::Axis(input_fact.axis),
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
                ndarray::Axis(output_fact.axis),
                ((output_pulse * ix)..(output_pulse * (ix + 1))).into(),
            )
            .assign(&result_chunk);
    }
    result.slice_axis_inplace(ndarray::Axis(output_fact.axis), (output_fact.delay..).into());
    result.slice_axis_inplace(ndarray::Axis(output_fact.axis), (..output_dim as usize).into());
    Ok(tvec!(result.into_arc_tensor()))
}
