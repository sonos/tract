use crate::errors::*;
use crate::{Model, Parameters};
use tract_hir::internal::*;

pub fn handle(params: &Parameters, options: &clap::ArgMatches) -> CliResult<()> {
    let dump = options.is_present("dump");
    let outputs = if let Some(pulse) = params.tract_model.downcast_ref::<PulsedModel>() {
        run_pulse_t(pulse, &params)?
    } else {
        dispatch_model!(params.tract_model, |m| run_regular(m, &params, options))?
    };

    if dump {
        for (ix, output) in outputs.iter().enumerate() {
            println!("output #{}\n{}\n", ix, output.dump(true)?);
        }
    }

    if let Some(asserts) = &params.assertions.assert_outputs {
        crate::utils::check_outputs(&*outputs, &asserts)?;
    }
    if let Some(facts) = &params.assertions.assert_output_facts {
        let outputs: Vec<InferenceFact> =
            outputs.iter().map(|t| InferenceFact::dt_shape(t.datum_type(), t.shape())).collect();
        crate::utils::check_inferred(&*outputs, &*facts)?;
    }

    Ok(())
}

fn run_regular(
    tract: &dyn Model,
    params: &Parameters,
    options: &clap::ArgMatches
) -> CliResult<TVec<Arc<Tensor>>> {
    let steps = options.is_present("steps");
    let assert_sane_floats = options.is_present("assert-sane-floats");
    let mut inputs: TVec<Tensor> = tvec!();
    for (ix, input) in tract.input_outlets().iter().enumerate() {
        if let Some(input) = params.input_values.get(ix).and_then(|x| x.as_ref()) {
            inputs.push(input.clone().into_tensor())
        } else {
            let fact = tract.outlet_typedfact(*input)?;
            inputs.push(crate::tensor::tensor_for_fact(&fact, None)?);
        }
    }
    dispatch_model!(tract, |m| {
        let plan = SimplePlan::new(m)?;
        let mut state = SimpleState::new(plan)?;
        Ok(state.run_plan_with_eval(inputs, |session_state, state, node, input| {
            if steps {
                eprintln!("{}: <{:?}", node, input);
            }
            let r = tract_core::plan::eval(session_state, state, node, input);
            if steps {
                eprintln!("{}: >{:?}", node, r);
            }
            let r = r?;
            if assert_sane_floats {
                for (ix, o) in r.iter().enumerate() {
                    if let Ok(floats) = o.as_slice::<f32>() {
                        if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                            eprintln!("{:?}", floats);
                            bail!("Found {} in output {} of {}", floats[pos], ix, node);
                        }
                    }
                }
            }
            Ok(r)
        })?)
    })
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
    let output_dim = output_fact.dim.eval(input_dim as i64).unwrap() as i64;
    let mut output_shape = output_fact.shape.to_vec();
    output_shape[output_fact.axis] =
        output_dim as usize + output_fact.delay + 4 * output_fact.pulse();
    let plan = SimplePlan::new(model)?;
    let mut state = ::tract_core::plan::SimpleState::new(&plan)?;
    //    println!("output_shape: {:?}", output_shape);
    let pulse = input_fact.pulse();
    let mut result = tract_ndarray::ArrayD::<f32>::default(output_shape);
    let input = input.to_array_view::<f32>()?;
    for ix in 0..input_dim.div_ceil(pulse) {
        let chunk =
            input.slice_axis(tract_ndarray::Axis(axis), (ix * pulse..(ix + 1) * pulse).into());
        let input = if chunk.shape()[input_fact.axis] < pulse {
            let mut chunk_shape = chunk.shape().to_vec();
            chunk_shape[input_fact.axis] = pulse;
            let mut padded_chunk = tract_ndarray::ArrayD::<f32>::default(chunk_shape);
            padded_chunk
                .slice_axis_mut(
                    tract_ndarray::Axis(input_fact.axis),
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
                tract_ndarray::Axis(output_fact.axis),
                ((output_pulse * ix)..(output_pulse * (ix + 1))).into(),
            )
            .assign(&result_chunk);
    }
    result.slice_axis_inplace(tract_ndarray::Axis(output_fact.axis), (output_fact.delay..).into());
    result
        .slice_axis_inplace(tract_ndarray::Axis(output_fact.axis), (..output_dim as usize).into());
    Ok(tvec!(result.into_arc_tensor()))
}
