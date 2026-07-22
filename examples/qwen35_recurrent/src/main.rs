use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use ndarray::{IxDyn, OwnedRepr};
use tract::prelude::*;

fn load_npz(path: &str) -> Result<HashMap<String, Tensor>> {
    let mut npz = ndarray_npy::NpzReader::new(File::open(path)?)?;
    let mut tensors = HashMap::new();
    for entry in npz.names()? {
        let name = entry.trim_end_matches(".npy").to_string();
        macro_rules! read {
            ($ty:ty) => {
                if let Ok(array) = npz.by_name::<OwnedRepr<$ty>, IxDyn>(&entry) {
                    let shape = array.shape().to_vec();
                    let values = array.into_raw_vec_and_offset().0;
                    tensors.insert(name, Tensor::from_slice(&shape, &values)?);
                    continue;
                }
            };
        }
        read!(f32);
        read!(f64);
        read!(i8);
        read!(i16);
        read!(i32);
        read!(i64);
        read!(u8);
        read!(u16);
        read!(u32);
        read!(u64);
        read!(bool);
        anyhow::bail!("cannot extract tensor {entry}");
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

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context("model directory required")?;
    let fixture_path = args.next().context("input npz required")?;
    let max_tokens: usize = args.next().unwrap_or_else(|| "256".into()).parse()?;
    let eos_token: i64 = args.next().unwrap_or_else(|| "2".into()).parse()?;
    let eos_margin: f32 = args.next().unwrap_or_else(|| "0.1".into()).parse()?;

    let load_start = Instant::now();
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let model = nnef.load(model_path)?;
    let input_names =
        (0..model.input_count()?).map(|ix| model.input_name(ix)).collect::<Result<Vec<_>>>()?;
    let input_dtypes = (0..input_names.len())
        .map(|ix| model.input_fact(ix)?.datum_type())
        .collect::<Result<Vec<_>>>()?;
    let runtime = ["cuda", "metal", "default"]
        .iter()
        .find_map(|name| tract::runtime_for_name(name).ok())
        .context("no Tract runtime is available")?;
    let plan = runtime.prepare(model)?;
    eprintln!(
        "loaded with {} in {:.3}s; {} inputs",
        runtime.name()?,
        load_start.elapsed().as_secs_f64(),
        input_names.len()
    );

    let mut fixture = load_npz(&fixture_path)?;
    let mut position = fixture
        .get("position_ids")
        .context("position_ids missing from fixture")?
        .convert_to(DatumType::I64)?
        .as_slice::<i64>()?[0];
    let mut inputs = input_names
        .iter()
        .zip(input_dtypes)
        .map(|(name, datum_type)| {
            fixture
                .remove(name)
                .with_context(|| format!("fixture lacks input {name}"))?
                .convert_to(datum_type)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(fixture.is_empty(), "unused fixture inputs: {:?}", fixture.keys());

    let decode_start = Instant::now();
    let mut timings = Vec::with_capacity(max_tokens);
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut stop_reason = "token_budget";
    for _ in 0..max_tokens {
        let step_start = Instant::now();
        let outputs = plan.run(inputs)?;
        timings.push(step_start.elapsed().as_secs_f64() * 1e3);
        let logits = outputs[0].convert_to(DatumType::F32)?;
        let values = logits.as_slice::<f32>()?;
        let non_finite = values.iter().filter(|value| !value.is_finite()).count();
        ensure!(
            non_finite == 0,
            "decoder produced {non_finite}/{} non-finite logits",
            values.len()
        );
        let mut token = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(ix, _)| ix as i64)
            .context("empty logits")?;
        let eos_gap = values[token as usize] - values[eos_token as usize];
        if token != eos_token && eos_gap <= eos_margin {
            token = eos_token;
        }
        tokens.push(token);
        position += 1;
        inputs =
            vec![Tensor::from_slice(&[1, 1], &[token])?, Tensor::from_slice(&[1, 1], &[position])?];
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
