use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use ndarray::prelude::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

const MEL_BINS: usize = 128;
const CHUNK_FRAMES: usize = 100;
const TOKENS_PER_CHUNK: usize = 13;
const CHUNKS_PER_WINDOW: usize = 8;
const AUDIO_PAD: i64 = 151676;
const EOS: [usize; 2] = [151643, 151645];
const MAX_TOKENS: usize = 256;

fn chunk_tokens(frames: usize) -> usize {
    (0..3).fold(frames, |l, _| (l - 1) / 2 + 1)
}

/// Runs the encoder on each 8 s window, the last chunk of a window zero-padded and its padding
/// tokens masked out of attention, then dropped.
fn encode(encoder: &Runnable, features: ArrayView2<f32>) -> Result<Array2<f32>> {
    let frames = features.shape()[1];
    let mut embeddings = vec![];
    for window in (0..frames).step_by(CHUNK_FRAMES * CHUNKS_PER_WINDOW) {
        let window =
            features.slice(s![.., window..frames.min(window + CHUNK_FRAMES * CHUNKS_PER_WINDOW)]);
        let len = window.shape()[1];
        let n_chunks = len.div_ceil(CHUNK_FRAMES);
        let mut chunks = Array3::<f32>::zeros((n_chunks, MEL_BINS, CHUNK_FRAMES));
        for (c, mut chunk) in chunks.outer_iter_mut().enumerate() {
            let src = window.slice(s![.., c * CHUNK_FRAMES..len.min((c + 1) * CHUNK_FRAMES)]);
            chunk.slice_mut(s![.., ..src.shape()[1]]).assign(&src);
        }
        let valid =
            (n_chunks - 1) * TOKENS_PER_CHUNK + chunk_tokens(len - (n_chunks - 1) * CHUNK_FRAMES);
        let mut key_bias = Array2::<f32>::zeros((n_chunks, TOKENS_PER_CHUNK));
        key_bias.iter_mut().skip(valid).for_each(|b| *b = f32::MIN);
        let [out] = encoder.run([chunks.tract()?, key_bias.tract()?])?.try_into().unwrap();
        embeddings.push(out.ndarray2::<f32>()?.slice(s![..valid, ..]).to_owned());
    }
    Ok(ndarray::concatenate(Axis(0), &embeddings.iter().map(|e| e.view()).collect_vec())?)
}

fn prompt(tokenizer: &tokenizers::Tokenizer, n_audio: usize) -> Result<Vec<i64>> {
    let encode = |s: &str| -> Result<Vec<i64>> {
        let encoding = tokenizer.encode(s, false).map_err(|e| anyhow!(e))?;
        Ok(encoding.get_ids().iter().map(|&i| i as i64).collect())
    };
    let mut ids = encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
    ids.extend(std::iter::repeat_n(AUDIO_PAD, n_audio));
    ids.extend(encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?);
    Ok(ids)
}

fn argmax(logits: &Tensor) -> Result<usize> {
    Ok(logits.as_slice::<f32>()?.iter().position_max_by_key(|x| FloatOrd(**x)).unwrap())
}

fn main() -> anyhow::Result<()> {
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let gpu = tract::runtime_for_name("gpu-or-cpu")?;
    eprintln!("runtime: {}", gpu.name()?);

    let preprocessor = nnef.load("assets/model/preprocessor.nnef.tgz")?.into_runnable()?;
    let encoder = gpu.prepare(nnef.load("assets/model/encoder.nnef.tgz")?)?;
    let mut decoder = nnef.load("assets/model/decoder/model/model.nnef.tgz")?;
    decoder.transform("transformers_detect_all")?;
    let decoder = gpu.prepare(decoder)?;
    let tokenizer =
        tokenizers::Tokenizer::from_file("assets/model/tokenizer.json").map_err(|e| anyhow!(e))?;

    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let [features] =
        preprocessor.run([Tensor::from_slice(&[1, wav.len()], &wav)?])?.try_into().unwrap();
    let audio = encode(&encoder, features.ndarray2::<f32>()?)?;

    let ids = prompt(&tokenizer, audio.shape()[0])?;
    let mut state = decoder.spawn_state()?;
    let input = Array2::from_shape_vec((1, ids.len()), ids)?.tract()?;
    let [logits, _] = state.run([input, audio.view().tract()?])?.try_into().unwrap();
    let mut token = argmax(&logits)?;
    let no_audio = Array2::<f32>::zeros((0, audio.shape()[1])).tract()?;
    let mut hyp = vec![];
    while !EOS.contains(&token) && hyp.len() < MAX_TOKENS {
        hyp.push(token as u32);
        let input = Tensor::from_slice(&[1, 1], &[token as i64])?;
        let [logits, _] = state.run([input, no_audio.clone()])?.try_into().unwrap();
        token = argmax(&logits)?;
    }

    let output = tokenizer.decode(&hyp, true).map_err(|e| anyhow!(e))?;
    let (language, transcript) = output.split_once("<asr_text>").context("no <asr_text> tag")?;
    println!("{language}: {transcript}");
    assert_eq!(language, "language English");
    assert_eq!(
        transcript,
        r#"Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait.""#
    );
    Ok(())
}
