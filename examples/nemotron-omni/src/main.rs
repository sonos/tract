use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::*;
use clap::Parser;
use tokenizers::Tokenizer;
use tract::prelude::*;

/// Asks Nemotron 3 Nano Omni about an image, an audio clip, or both, and prints its answer as
/// it is generated (greedy decoding).
#[derive(Parser, Debug)]
struct Args {
    /// Directory written by export.py.
    #[arg(long, default_value = "assets")]
    assets: String,
    /// An image (any format the image crate reads).
    #[arg(long)]
    image: Option<String>,
    /// A WAV clip; the first channel is used, at any sample rate.
    #[arg(long)]
    audio: Option<String>,
    #[arg(long, default_value = "cuda")]
    runtime: String,
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Let the model reason in a <think> block before it answers.
    #[arg(long)]
    think: bool,
    prompt: String,
}

const PATCH: usize = 16;
const MERGE: usize = 2;
const MIN_PATCHES: usize = 1024;
const MAX_PATCHES: usize = 13312;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const SAMPLE_RATE: u32 = 16000;
const VOCAB: usize = 131072;
const EOS: [u32; 2] = [2, 11];

/// The patch grid NVIDIA's image processor resizes an image to: its own grid, scaled into
/// [MIN_PATCHES, MAX_PATCHES] patches, both sides rounded to whole merge cells.
fn patch_grid(height: usize, width: usize) -> (usize, usize) {
    let (mut h, mut w) = (height.div_ceil(PATCH), width.div_ceil(PATCH));
    let shrink = (MAX_PATCHES as f64 / (h * w) as f64).sqrt().min(1.0);
    (h, w) = ((shrink * h as f64).floor() as usize, (shrink * w as f64).floor() as usize);
    if h * w < MIN_PATCHES {
        let grow = (MIN_PATCHES as f64 / (h * w) as f64).sqrt();
        (h, w) = ((grow * h as f64).ceil() as usize, (grow * w as f64).ceil() as usize);
    }
    if h % MERGE != 0 {
        h = if (h + 1) * w <= MAX_PATCHES { h + 1 } else { (h - 1).max(MERGE) };
    }
    if w % MERGE != 0 {
        w = if h * (w + 1) <= MAX_PATCHES { w + 1 } else { (w - 1).max(MERGE) };
    }
    (h, w)
}

/// Antialiased bicubic (a = -0.5) resampling of one axis, as torch and PIL compute it: the
/// filter widens by the downscaling factor.
fn resize_axis(input: &[f32], len: usize, stride: usize, count: usize, out_len: usize) -> Vec<f32> {
    let cubic = |x: f32| {
        let (a, x) = (-0.5f32, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = len as f32 / out_len as f32;
    let widen = scale.max(1.0);
    let taps: Vec<(usize, Vec<f32>)> = (0..out_len)
        .map(|o| {
            let center = (o as f32 + 0.5) * scale;
            let lo = ((center - 2.0 * widen + 0.5) as isize).max(0) as usize;
            let hi = ((center + 2.0 * widen + 0.5) as usize).min(len);
            let w: Vec<f32> = (lo..hi).map(|i| cubic((i as f32 - center + 0.5) / widen)).collect();
            let sum: f32 = w.iter().sum();
            (lo, w.into_iter().map(|w| w / sum).collect())
        })
        .collect();
    let mut out = vec![0f32; out_len * count];
    for c in 0..count {
        for (o, (lo, w)) in taps.iter().enumerate() {
            out[o * count + c] =
                w.iter().enumerate().map(|(k, w)| w * input[(lo + k) * stride + c]).sum();
        }
    }
    out
}

/// The vision graph's input: the image resized to its patch grid and normalized, as 16x16
/// channel-major patches grouped by 2x2 merge cell, [h / 2, w / 2, 2, 2, 768].
fn image_patches(path: &str) -> Result<Tensor> {
    let img = image::open(path)?.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let (gh, gw) = patch_grid(height, width);
    let (oh, ow) = (gh * PATCH, gw * PATCH);
    let pixels: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
    let rows: Vec<f32> =
        pixels.chunks(width * 3).flat_map(|row| resize_axis(row, width, 3, 3, ow)).collect();
    let resized = resize_axis(&rows, height, ow * 3, ow * 3, oh);
    let dim = 3 * PATCH * PATCH;
    let mut patches = vec![0f32; gh * gw * dim];
    for (y, x, c) in (0..oh).flat_map(|y| (0..ow).flat_map(move |x| (0..3).map(move |c| (y, x, c))))
    {
        let (py, px) = (y / PATCH, x / PATCH);
        let cell = (py / MERGE) * (gw / MERGE) + px / MERGE;
        let patch = cell * MERGE * MERGE + (py % MERGE) * MERGE + px % MERGE;
        let offset = c * PATCH * PATCH + (y % PATCH) * PATCH + x % PATCH;
        let value = resized[(y * ow + x) * 3 + c] / 255.0;
        patches[patch * dim + offset] = (value - MEAN[c]) / STD[c];
    }
    Tensor::from_slice(&[gh / MERGE, gw / MERGE, MERGE, MERGE, dim], &patches)?
        .convert_to(DatumType::F16)
}

fn read_audio(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let pcm: Vec<f32> = samples.into_iter().step_by(spec.channels as usize).collect();
    let n = (pcm.len() as u64 * SAMPLE_RATE as u64 / spec.sample_rate as u64) as usize;
    Ok((0..n)
        .map(|i| {
            let x = i as f64 * spec.sample_rate as f64 / SAMPLE_RATE as f64;
            let (j, f) = (x as usize, (x - x.floor()) as f32);
            match (pcm.get(j), pcm.get(j + 1)) {
                (Some(a), Some(b)) => a * (1.0 - f) + b * f,
                (Some(a), None) => *a,
                _ => 0.0,
            }
        })
        .collect())
}

/// One user turn in the model's chat template, each placeholder expanded to one token per
/// embedding row.
fn chat(prompt: &str, image_rows: usize, audio_rows: usize, think: bool) -> String {
    let mut user = String::new();
    if image_rows > 0 {
        user += &format!("<img>{}</img>\n", "<image>".repeat(image_rows));
    }
    if audio_rows > 0 {
        user += &format!("<so_start>{}<so_end>\n", "<so_embedding>".repeat(audio_rows));
    }
    let think = if think { "<think>\n" } else { "<think></think>" };
    format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user}{prompt}<|im_end|>\n<|im_start|>assistant\n{think}"
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let assets = Path::new(&args.assets);
    let model = |name: &str| assets.join("model").join(name).join("model.nnef.tgz");
    let runtime = tract::runtime_for_name(&args.runtime)?;
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let tokenizer = Tokenizer::from_file(assets.join("tokenizer.json")).map_err(Error::msg)?;

    let t0 = Instant::now();
    let empty = || Tensor::from_slice(&[0, 2688], &[] as &[f32])?.convert_to(DatumType::F16);
    let image = match &args.image {
        Some(path) => {
            let vision = runtime.prepare(nnef.load(model("vision"))?)?;
            vision.run([image_patches(path)?])?.remove(0)
        }
        None => empty()?,
    };
    let audio = match &args.audio {
        Some(path) => {
            let pcm = read_audio(path)?;
            let mel = runtime.prepare(nnef.load(assets.join("mel.nnef.tgz"))?)?;
            let mel = mel.run([Tensor::from_slice(&[1, pcm.len()], &pcm)?])?.remove(0);
            let encoder = runtime.prepare(nnef.load(model("audio"))?)?;
            encoder.run([mel.convert_to(DatumType::F16)?])?.remove(0)
        }
        None => empty()?,
    };
    let (image_rows, audio_rows) = (image.shape()?[0], audio.shape()?[0]);
    if image_rows + audio_rows > 0 {
        eprintln!(
            "encoded {image_rows} image and {audio_rows} audio embeddings in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }

    let t0 = Instant::now();
    let mut decoder = nnef.load(model("decoder"))?;
    decoder.transform("transformers_detect_all")?;
    let input_names: Vec<String> =
        (0..decoder.input_count()?).map(|i| decoder.input_name(i)).collect::<Result<_>>()?;
    // states are zero before the first token: [1, 6144, 4] conv and [1, 64, 64, 128] f32 SSM
    // states per Mamba-2 layer, empty [1, 2, 0, 128] caches per attention layer
    let mut states: HashMap<String, Tensor> = HashMap::new();
    for (ix, name) in input_names.iter().enumerate() {
        if name.starts_with("in_cache_") {
            let fact = decoder.input_fact(ix)?;
            let shape: Vec<usize> =
                fact.dims()?.map(|d| d.to_int64().map(|d| d as usize).unwrap_or(0)).collect();
            let zeros = Tensor::from_slice(&shape, &vec![0f32; shape.iter().product()])?;
            states.insert(name.clone(), zeros.convert_to(fact.datum_type()?)?);
        }
    }
    let decoder = runtime.prepare(decoder)?;
    let mut decoder_state = decoder.spawn_state()?;
    eprintln!("loaded the decoder on {} in {:.1} s", runtime.name()?, t0.elapsed().as_secs_f64());

    let text = chat(&args.prompt, image_rows, audio_rows, args.think);
    let ids: Vec<i64> = tokenizer
        .encode(text, false)
        .map_err(Error::msg)?
        .get_ids()
        .iter()
        .map(|&id| id as i64)
        .collect();
    let mut step = |tokens: &[i64], image: Tensor, audio: Tensor| -> Result<u32> {
        let inputs = input_names
            .iter()
            .map(|name| match name.as_str() {
                "input_ids" => Tensor::from_slice(&[1, tokens.len()], tokens),
                "in_image_embeddings" => Ok(image.clone()),
                "in_audio_embeddings" => Ok(audio.clone()),
                state => states.remove(state).with_context(|| format!("no value for {state}")),
            })
            .collect::<Result<Vec<_>>>()?;
        // outputs: logits, then the updated states in the order of their inputs
        let mut outputs = decoder_state.run(inputs)?.into_iter();
        let logits = outputs.next().context("no logits")?.convert_to(DatumType::F32)?;
        for (name, value) in input_names.iter().filter(|n| n.starts_with("in_cache_")).zip(outputs)
        {
            states.insert(name.clone(), value);
        }
        let logits = logits.as_slice::<f32>()?;
        let last = &logits[logits.len() - VOCAB..];
        Ok((0..VOCAB).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap() as u32)
    };

    let t0 = Instant::now();
    let mut token = step(&ids, image, audio)?;
    let prefill = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut generated = vec![];
    let mut printed = 0;
    while !EOS.contains(&token) && generated.len() < args.max_tokens {
        generated.push(token);
        let text = tokenizer.decode(&generated, false).map_err(Error::msg)?;
        // hold back a partial UTF-8 sequence until its last token arrives
        if !text.ends_with('\u{fffd}') {
            let start = printed.max(text.len() - text.trim_start().len());
            print!("{}", &text[start..]);
            std::io::stdout().flush()?;
            printed = text.len();
        }
        token = step(&[token as i64], empty()?, empty()?)?;
    }
    println!();
    eprintln!(
        "prompt: {} tokens in {:.2} s, answer: {} tokens at {:.1} tokens/s",
        ids.len(),
        prefill,
        generated.len(),
        generated.len() as f64 / t0.elapsed().as_secs_f64()
    );
    Ok(())
}
