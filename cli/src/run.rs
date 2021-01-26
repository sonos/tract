use crate::CliResult;
use crate::{Model, Parameters};
use ansi_term::Color::*;
use tract_hir::internal::*;
#[cfg(feature = "pulse")]
use tract_pulse::internal::*;

pub fn handle(params: &Parameters, options: &clap::ArgMatches) -> CliResult<()> {
    let dump = options.is_present("dump");
    #[cfg(feature = "pulse")]
    let outputs = if let Some(pulse) = params.tract_model.downcast_ref::<PulsedModel>() {
        run_pulse_t(pulse, &params)?
    } else {
        dispatch_model!(&*params.tract_model, |m| run_regular(m, &params, options))?
    };

    #[cfg(not(feature = "pulse"))]
    let outputs = dispatch_model!(&*params.tract_model, |m| run_regular(m, &params, options))?;

    if dump {
        for (ix, output) in outputs.iter().enumerate() {
            println!("output #{}\n{}\n", ix, output.dump(true)?);
        }
    }

    if let Some(count) = options.value_of("assert-output-count") {
        let count = count.parse::<usize>()?;
        if count != outputs.len() {
            bail!(
                "Wrong number of outputs, command line expected {}, found {:?}",
                count,
                outputs.len()
            );
        }
    }

    crate::utils::check_outputs(&*outputs, &params.assertions.assert_outputs)?;

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
    options: &clap::ArgMatches,
) -> CliResult<TVec<Arc<Tensor>>> {
    let steps = options.is_present("steps");
    let assert_sane_floats = options.is_present("assert-sane-floats");
    let mut inputs: TVec<Tensor> = tvec!();
    for input in tract.input_outlets() {
        let name = tract.node_name(input.node);
        if let Some(input) = params.input_values.get(name) {
            info!("Using fixed input for input called {:?}: {:?}", name, input);
            inputs.push(input.clone().into_tensor())
        } else {
            let fact = tract.outlet_typedfact(*input)?;
            info!("Using random input for input called {:?}: {:?}", name, fact);
            inputs.push(crate::tensor::tensor_for_fact(&fact, None)?);
        }
    }
    dispatch_model!(tract, |m| {
        let plan = SimplePlan::new(m)?;
        let mut state = SimpleState::new(plan)?;
        Ok(state.run_plan_with_eval(inputs, |session_state, state, node, input| {
            if steps {
                for i in &input {
                    eprintln!(
                        "{}{}{:?}",
                        White.bold().paint(node.to_string()),
                        Blue.bold().paint(" << "),
                        i
                    );
                }
            }
            let r = tract_core::plan::eval(session_state, state, node, input)?;
            if steps {
                for o in &r {
                    eprintln!(
                        "{}{}{:?}",
                        White.bold().paint(node.to_string()),
                        Blue.bold().paint(" >> "),
                        o
                    );
                }
            }
            if assert_sane_floats {
                for (ix, o) in r.iter().enumerate() {
                    if let Ok(floats) = o.as_slice::<f32>() {
                        if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                            eprintln!("{:?}", floats);
                            tract_core::anyhow::bail!(
                                "Found {} in output {} of {}",
                                floats[pos],
                                ix,
                                node
                            );
                        }
                    }
                }
            }
            Ok(r)
        })?)
    })
}

#[cfg(feature = "pulse")]
fn run_pulse_t(model: &PulsedModel, params: &Parameters) -> CliResult<TVec<Arc<Tensor>>> {
    let input_fact = model.input_fact(0)?;
    let output_fact = model.output_fact(0)?;

    let output_pulse = output_fact.pulse();
    //    println!("output_fact: {:?}", output_fact);
    let axis = input_fact.axis;
    let name = model.node_name(model.input_outlets()?[0].node);
    let input: &Tensor = &params.input_values.get(name).unwrap();
    //    println!("input_shape: {:?}", input.shape());
    let input_dim = input.shape()[axis];
    //    println!("output_fact: {:?}", output_fact);
    let output_dim = output_fact
        .dim
        .eval(&SymbolValues::default().with(stream_symbol(), input_dim as i64))
        .to_usize()?;
    let mut output_shape = output_fact.shape.to_vec();
    output_shape[output_fact.axis] =
        (output_dim as usize + output_fact.delay + 4 * output_fact.pulse()).to_dim();
    let output_shape: TVec<usize> = output_shape.iter().map(|d| d.to_usize().unwrap()).collect();
    let plan = SimplePlan::new(model)?;
    let mut state = ::tract_core::plan::SimpleState::new(&plan)?;
    //    println!("output_shape: {:?}", output_shape);
    let pulse = input_fact.pulse();
    let mut result = tract_ndarray::ArrayD::<f32>::default(&*output_shape);
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
