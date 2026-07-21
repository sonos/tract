use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

use tract_core::runtime::RunOptions;
use tract_core::transform::ModelTransform;
use tract_cuda::CudaTransform;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};
use tract_nnef::internal::*;

fn on_device(value: &TValue) -> TractResult<DeviceTensor> {
    match value.to_device_tensor() {
        Ok(tensor) => Ok(tensor.clone()),
        Err(_) => value.clone().into_tensor().into_device(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NnefGatedDeltaNetRecurrent;

impl Op for NnefGatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        "NnefGatedDeltaNetRecurrent".into()
    }
    op_as_typed_op!();
}

impl EvalOp for NnefGatedDeltaNetRecurrent {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        _session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6);
        tract_cuda::with_cuda_stream(|stream| {
            let tensors = inputs.iter().map(on_device).collect::<TractResult<TVec<_>>>()?;
            stream.synchronize()?;
            let (output, state) = tract_cuda::kernels::gdn_recurrent::CudaGdnRecurrent.eval(
                stream,
                &tensors[0],
                &tensors[1],
                &tensors[2],
                &tensors[3],
                &tensors[4],
                &tensors[5],
            )?;
            Ok(tvec![
                output.to_host()?.into_tensor().into_tvalue(),
                state.to_host()?.into_tensor().into_tvalue(),
            ])
        })
    }
}

impl TypedOp for NnefGatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 6);
        Ok(tvec![inputs[0].without_value(), inputs[5].without_value()])
    }
    as_op!();
}

fn qwen35_registry() -> Registry {
    fn deserialize(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<Value> {
        let inputs = ["query", "key", "value", "log_decay", "beta", "initial_state"]
            .map(|name| invocation.named_arg_as(builder, name))
            .into_iter()
            .collect::<TractResult<TVec<_>>>()?;
        builder.wire(NnefGatedDeltaNetRecurrent, &inputs)
    }
    let mut registry = Registry::new("tract_qwen35_cuda");
    registry.register_primitive(
        "tract_qwen35_gdn_recurrent",
        &[
            TypeName::Scalar.tensor().named("query"),
            TypeName::Scalar.tensor().named("key"),
            TypeName::Scalar.tensor().named("value"),
            TypeName::Scalar.tensor().named("log_decay"),
            TypeName::Scalar.tensor().named("beta"),
            TypeName::Scalar.tensor().named("initial_state"),
        ],
        &[("output", TypeName::Scalar.tensor()), ("final_state", TypeName::Scalar.tensor())],
        deserialize,
    );
    registry
}

fn load_npz(path: &str) -> TractResult<HashMap<String, Tensor>> {
    let mut npz = ndarray_npy::NpzReader::new(File::open(path)?)?;
    let mut tensors = HashMap::new();
    for entry in npz.names()? {
        let name = entry.trim_end_matches(".npy").to_string();
        tensors.insert(name, tract_libcli::tensor::for_npz(&mut npz, &entry)?);
    }
    Ok(tensors)
}

fn repeated_suffix(tokens: &[i64], repeats: usize, max_period: usize) -> Option<usize> {
    for period in 1..=max_period.min(tokens.len() / repeats) {
        let width = period * repeats;
        let suffix = &tokens[tokens.len() - width..];
        if suffix.chunks_exact(period).all(|chunk| chunk == &suffix[..period]) {
            return Some(period);
        }
    }
    None
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context("model directory required")?;
    let fixture_path = args.next().context("input npz required")?;
    let max_tokens: usize = args.next().unwrap_or_else(|| "256".into()).parse()?;
    let eos_token: i64 = args.next().unwrap_or_else(|| "2".into()).parse()?;
    let eos_margin: f32 = args.next().unwrap_or_else(|| "0.1".into()).parse()?;

    let load_start = Instant::now();
    let mut model = tract_nnef::nnef()
        .with_registry(qwen35_registry())
        .model_for_path(model_path)?
        .into_decluttered()?;
    let mut scan_names = vec![];
    if std::env::var_os("QWEN35_SCAN_FINITE").is_some() {
        let mut outlets = model.output_outlets()?.to_vec();
        for layer in 0..24 {
            for suffix in ["xAdd2", "xAdd0"] {
                let needle = format!("model_model__{layer}_{suffix}");
                if let Some(node) = model.nodes().iter().find(|node| node.name.contains(&needle)) {
                    outlets.push(node.id.into());
                    scan_names.push(needle);
                }
            }
        }
        model.select_output_outlets(&outlets)?;
        eprintln!("finite scan: {} layer boundaries", scan_names.len());
    }
    CudaTransform.transform(&mut model)?;
    model.optimize()?;

    let input_names =
        model.input_outlets()?.iter().map(|o| model.node(o.node).name.clone()).collect::<Vec<_>>();
    let input_dtypes = (0..input_names.len())
        .map(|ix| model.input_fact(ix).map(|f| f.datum_type))
        .collect::<TractResult<Vec<_>>>()?;
    let plan = TypedSimplePlan::new_with_options(
        model,
        &RunOptions { skip_order_opt_ram: true, ..Default::default() },
    )?;
    eprintln!("loaded in {:.3}s; {} inputs", load_start.elapsed().as_secs_f64(), input_names.len());

    let mut fixture = load_npz(&fixture_path)?;
    let mut position = unsafe {
        fixture
            .get("position_ids")
            .context("position_ids missing from fixture")?
            .as_slice_unchecked::<i64>()[0]
    };
    let mut inputs = input_names
        .iter()
        .zip(input_dtypes)
        .map(|(name, datum_type)| {
            Ok(fixture
                .remove(name)
                .with_context(|| format!("fixture lacks input {name}"))?
                .cast_to_dt(datum_type)?
                .into_owned()
                .into_tvalue())
        })
        .collect::<TractResult<TVec<_>>>()?;
    ensure!(fixture.is_empty(), "unused fixture inputs: {:?}", fixture.keys());

    let decode_start = Instant::now();
    let mut timings = Vec::with_capacity(max_tokens);
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut stop_reason = "token_budget";

    for _ in 0..max_tokens {
        let step_start = Instant::now();
        let outputs = plan.run(inputs)?;
        timings.push(step_start.elapsed().as_secs_f64() * 1e3);

        let logits = outputs[0].clone().into_tensor();
        let values = unsafe { logits.as_slice_unchecked::<f32>() };
        if !scan_names.is_empty() {
            for (name, value) in scan_names.iter().zip(outputs.iter().skip(49)) {
                let tensor = value.clone().into_tensor();
                let bad = match tensor.datum_type() {
                    DatumType::F16 => unsafe { tensor.as_slice_unchecked::<f16>() }
                        .iter()
                        .filter(|v| !v.is_finite())
                        .count(),
                    DatumType::F32 => unsafe { tensor.as_slice_unchecked::<f32>() }
                        .iter()
                        .filter(|v| !v.is_finite())
                        .count(),
                    _ => 0,
                };
                eprintln!("finite-scan {name}: {bad}/{}", tensor.len());
            }
        }
        let non_finite = values.iter().filter(|v| !v.is_finite()).count();
        ensure!(
            non_finite == 0,
            "decoder produced {non_finite}/{} non-finite logits",
            values.len()
        );
        let mut token = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i64)
            .context("empty logits")?;
        let eos_gap = values[token as usize] - values[eos_token as usize];
        if token != eos_token && eos_gap <= eos_margin {
            token = eos_token;
        }
        if std::env::var_os("QWEN35_EOS_GAPS").is_some() {
            let best = values[token as usize];
            let eos = values[eos_token as usize];
            eprintln!(
                "eos-gap step={} token={} gap={:.6} best={:.6} eos={:.6}",
                tokens.len(),
                token,
                best - eos,
                best,
                eos
            );
            let mut ranked = values.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("top5 step={} {:?}", tokens.len(), &ranked[..5]);
        }
        tokens.push(token);

        position += 1;
        let token_value = Tensor::from_shape(&[1, 1], &[token])?.into_tvalue();
        let position_value = Tensor::from_shape(&[1, 1], &[position])?.into_tvalue();
        inputs = tvec![token_value, position_value];
        inputs.extend(outputs.into_iter().skip(1));

        if token == eos_token {
            stop_reason = "eos";
            break;
        }
        if repeated_suffix(&tokens, 3, 64).is_some() {
            stop_reason = "repetition";
            break;
        }
    }

    let elapsed = decode_start.elapsed().as_secs_f64();
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    println!(
        "{{\"tokens\":{:?},\"count\":{},\"stop_reason\":\"{}\",\"seconds\":{:.6},\"ms_per_token\":{:.3},\"median_ms\":{:.3},\"tokens_per_second\":{:.2}}}",
        tokens,
        tokens.len(),
        stop_reason,
        elapsed,
        elapsed * 1000.0 / tokens.len() as f64,
        median,
        tokens.len() as f64 / elapsed,
    );
    Ok(())
}
