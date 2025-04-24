use fs::File;
use std::path::PathBuf;
use std::str::FromStr;

use crate::TractResult;
use crate::{Model, Parameters};
use fs_err as fs;
use ndarray_npy::NpzWriter;
use nu_ansi_term::Color::*;
use tract_core::ops::cnn::conv::Im2Col;
use tract_core::ops::matmul::pack::OptMatMulPack;
use tract_core::tract_data::itertools::izip;
use tract_hir::internal::*;
use tract_libcli::tensor::RunParams;
use tract_nnef::tensors::write_tensor;
#[cfg(feature = "pulse")]
use tract_pulse::internal::*;

/// Add a tensor entry into a npz file.
fn npz_add_tensor(npz: &mut NpzWriter<File>, name: String, tensor: &Tensor) -> TractResult<()> {
    match tensor.datum_type() {
        DatumType::F16 => npz.add_array(name, &tensor.cast_to::<f32>()?.to_array_view::<f32>()?)?,
        DatumType::Bool => npz.add_array(name, &tensor.to_array_view::<bool>()?)?,
        DatumType::U8 => npz.add_array(name, &tensor.to_array_view::<u8>()?)?,
        DatumType::U16 => npz.add_array(name, &tensor.to_array_view::<u16>()?)?,
        DatumType::U32 => npz.add_array(name, &tensor.to_array_view::<u32>()?)?,
        DatumType::U64 => npz.add_array(name, &tensor.to_array_view::<u64>()?)?,
        DatumType::I8 => npz.add_array(name, &tensor.to_array_view::<i8>()?)?,
        DatumType::I16 => npz.add_array(name, &tensor.to_array_view::<i16>()?)?,
        DatumType::I32 => npz.add_array(name, &tensor.to_array_view::<i32>()?)?,
        DatumType::I64 => npz.add_array(name, &tensor.to_array_view::<i64>()?)?,
        DatumType::F32 => npz.add_array(name, &tensor.to_array_view::<f32>()?)?,
        DatumType::F64 => npz.add_array(name, &tensor.to_array_view::<f64>()?)?,
        DatumType::QI8(_) => npz.add_array(name, &tensor.to_array_view::<i8>()?)?,
        DatumType::QU8(_) => npz.add_array(name, &tensor.to_array_view::<u8>()?)?,
        DatumType::QI32(_) => npz.add_array(name, &tensor.to_array_view::<i32>()?)?,
        _ => warn!("Not writing {name}, {tensor:?}, unsupported type"),
    }

    Ok(())
}

pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<()> {
    let run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;

    let dump = sub_matches.is_present("dump");
    let outputs = dispatch_model!(&*params.tract_model, |m| run_regular(
        m,
        &run_params,
        matches,
        sub_matches
    ))?;

    if dump {
        for (ix, output) in outputs.iter().enumerate() {
            for (turn, output) in output.iter().enumerate() {
                println!("output #{}, turn #{}\n{}\n", ix, turn, output.dump(true)?);
            }
        }
    }

    if let Some(file_path) = sub_matches.value_of("save-outputs-nnef") {
        fs::create_dir_all(file_path).with_context(|| format!("Creating {file_path} directory"))?;
        for (ix, outputs) in outputs.iter().enumerate() {
            let name = params
                .tract_model
                .outlet_label(params.tract_model.output_outlets()[ix])
                .map(|name| format!("{name}.dat"))
                .unwrap_or_else(|| format!("output_{ix}.dat"));

            if outputs.len() == 1 {
                let mut f = fs::File::create(PathBuf::from_str(file_path)?.join(&name))?;
                write_tensor(&mut f, &outputs[0])?;
            } else {
                for (turn, output) in outputs.iter().enumerate() {
                    let name = format!("turn_{turn}/{name}");
                    let mut f = fs::File::open(PathBuf::from_str(file_path)?.join(name))?;
                    write_tensor(&mut f, output)?;
                }
            }
        }
    }

    if let Some(file_path) = sub_matches.value_of("save-outputs-npz") {
        let file = fs::File::create(file_path).with_context(|| format!("Creating {file_path}"))?;
        let mut npz = ndarray_npy::NpzWriter::new_compressed(file);

        for (ix, outputs) in outputs.iter().enumerate() {
            let name = params
                .tract_model
                .outlet_label(params.tract_model.output_outlets()[ix])
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("output_{ix}"));
            if outputs.len() == 1 {
                npz_add_tensor(&mut npz, name, &outputs[0])?;
            } else {
                for (turn, output) in outputs.iter().enumerate() {
                    let name = format!("turn_{turn}/{name}");
                    npz_add_tensor(&mut npz, name, output)?;
                }
            }
        }
    }

    if let Some(count) = sub_matches.value_of("assert-output-count") {
        let count = count.parse::<usize>()?;
        if count != outputs.len() {
            bail!(
                "Wrong number of outputs, command line expected {}, found {:?}",
                count,
                outputs.len()
            );
        }
    }

    if params.assertions.assert_outputs {
        crate::utils::check_outputs(&outputs, params)?;
    }

    if let Some(facts) = &params.assertions.assert_output_facts {
        let outputs: Vec<InferenceFact> =
            outputs.iter().map(|t| t[0].datum_type().fact(t[0].shape()).into()).collect();
        crate::utils::check_inferred(&outputs, facts)?;
    }

    if let Some(asserts) = &params.assertions.assert_op_count {
        for (name, expected) in asserts {
            let count = crate::utils::count_op(&*params.tract_model, name)?;
            if count != *expected {
                bail!("Wrong number of {} operators: expected {}, got {}", name, expected, count);
            }
        }
    }

    Ok(())
}

fn run_regular(
    tract: &dyn Model,
    run_params: &RunParams,
    _matches: &clap::ArgMatches,
    sub_matches: &clap::ArgMatches,
) -> TractResult<TVec<Vec<TValue>>> {
    let plan_options = crate::plan_options::plan_options_from_subcommand(sub_matches)?;
    let steps = sub_matches.is_present("steps");
    let check_f16_overflow = sub_matches.is_present("check-f16-overflow");
    let assert_sane_floats = sub_matches.is_present("assert-sane-floats");
    let mut npz = if let Some(npz) = sub_matches.value_of("save-steps") {
        let npz = fs::File::create(npz).with_context(|| format!("Creating {npz}"))?;
        Some(ndarray_npy::NpzWriter::new_compressed(npz))
    } else {
        None
    };
    dispatch_model!(tract, |m| {
        let plan = SimplePlan::new_with_options(m, &plan_options)?;
        let mut state = SimpleState::new(plan)?;
        let inputs = tract_libcli::tensor::retrieve_or_make_inputs(tract, run_params)?;
        let mut results = tvec!(vec!(); state.model().outputs.len());
        let multiturn = inputs.len() > 1;
        for (turn, inputs) in inputs.into_iter().enumerate() {
            let turn_results =
                state.run_plan_with_eval(inputs, |session_state, state, node, input| {
                    if steps {
                        for (ix, i) in input.iter().enumerate() {
                            eprintln!(
                                "{} {}{}{:?}",
                                White.bold().paint(node.to_string()),
                                ix,
                                Blue.bold().paint("<< "),
                                i
                            );
                        }
                    }
                    let r = tract_core::plan::eval(session_state, state, node, input)?;
                    let clarified_r = crate::utils::clarify_tvalues(&r)?;

                    if steps {
                        for (ix, o) in clarified_r.iter().enumerate() {
                            eprintln!(
                                "{} {}{}{:?}",
                                White.bold().paint(node.to_string()),
                                ix,
                                Yellow.bold().paint(">> "),
                                o
                            );
                        }
                    }
                    if let Some(npz) = npz.as_mut() {
                        for (ix, t) in clarified_r.iter().enumerate() {
                            let mut name = if ix == 0 {
                                node.name.to_string()
                            } else {
                                format!("{}:{}", node.name, ix)
                            };
                            if multiturn {
                                name = format!("turn_{turn}/{name}");
                            }
                            npz_add_tensor(npz, name, t)?;
                        }
                    }
                    if check_f16_overflow {
                        for (ix, o) in clarified_r.iter().enumerate() {
                            if let Ok(f32s) = o.as_slice::<f32>() {
                                if f32s.iter().any(|f| f.abs() > f16::MAX.to_f32()) {
                                    warn!("{node}, output {ix} overflows f16");
                                }
                            }
                        }
                    }
                    if assert_sane_floats {
                        for (ix, o) in clarified_r.iter().enumerate() {
                            if node.op_is::<Im2Col>() || node.op_is::<OptMatMulPack>() {
                                continue;
                            }
                            if let Ok(floats) = o.as_slice::<f32>() {
                                if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                                    eprintln!("{floats:?}");
                                    bail!("Found {} in output {} of {}", floats[pos], ix, node);
                                }
                            } else if let Ok(floats) = o.as_slice::<f16>() {
                                if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                                    eprintln!("{floats:?}");
                                    bail!("Found {} in output {} of {}", floats[pos], ix, node);
                                }
                            }
                        }
                    }
                    Ok(r)
                })?;
            izip!(&mut results, turn_results).for_each(|(r, tr)| r.push(tr));
        }
        Ok(results)
    })
}

/*
#[cfg(feature = "pulse")]
fn run_pulse_t(model: &PulsedModel, params: &Parameters) -> TractResult<TVec<TValue>> {
let input_fact = model.input_fact(0)?;
let output_fact = model.output_fact(0)?;

let output_pulse = output_fact.pulse();
//    println!("output_fact: {:?}", output_fact);
let axis = input_fact.axis;
let name = model.node_name(model.input_outlets()?[0].node);
let input: &Tensor = &params.tensors_values.by_name(name).unwrap().values.as_ref().unwrap()[0];
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
for ix in 0..input_dim.divceil(pulse) {
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
let outputs = state.run(tvec!(input.into_tensor().into()))?;
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
Ok(tvec!(result.into_tvalue()))
}
*/
