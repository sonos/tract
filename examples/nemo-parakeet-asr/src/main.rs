use std::fs::File;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use ndarray::prelude::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.iter().position_max_by_key(|x| FloatOrd(**x))
}

/// One prednet (decoder) step, tolerant of the split-RNNT export shape. t2n>=0.24
/// adds a `target_length` input and a `prednet_lengths` output; when the latter is
/// a pass-through tract prunes both back to the older 3-in/3-out shape. States are
/// always the last two outputs.
fn run_decoder(
    decoder: &Runnable,
    wants_length: bool,
    tokens: Tensor,
    n_tokens: i32,
    state_0: Tensor,
    state_1: Tensor,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let mut inputs = vec![tokens];
    if wants_length {
        inputs.push(Tensor::from_slice(&[1], &[n_tokens])?);
    }
    inputs.push(state_0);
    inputs.push(state_1);
    let out = decoder.run(inputs)?;
    Ok((out[0].clone(), out[out.len() - 2].clone(), out[out.len() - 1].clone()))
}

fn main() -> anyhow::Result<()> {
    let config: serde_json::Value =
        serde_json::from_reader(File::open("assets/model/model_config.json")?)?;
    let blank_id = config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
    let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
    let vocab: Vec<&str> = vocab.iter().map(|v| v.as_str().unwrap()).collect();

    let nnef = tract::nnef()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();

    let preprocessor = nnef.load("assets/model/preprocessor.nnef.tgz")?.into_runnable()?;

    let mut encoder = nnef.load("assets/model/encoder.nnef.tgz")?;
    encoder.transform("transformers_detect_all")?;
    let encoder = gpu.prepare(encoder)?;

    let decoder = nnef.load("assets/model/decoder.nnef.tgz")?;
    let decoder = gpu.prepare(decoder)?;
    let dec_wants_length = decoder.input_count()? == 4;

    let joint = nnef.load("assets/model/joint.nnef.tgz")?;
    let joint = gpu.prepare(joint)?;

    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32)
        .collect();
    let samples = Tensor::from_slice(&[1, wav.len()], &wav)?;
    let len = arr1(&[wav.len() as i64]).tract()?;

    let [features, feat_len] = preprocessor.run([samples, len])?.try_into().unwrap();
    let [encoded, _lens] = encoder.run([features, feat_len])?.try_into().unwrap();

    let encoded: ArrayD<f32> = encoded.ndarray()?.into_owned();

    let max_frames = encoded.shape()[2];
    let max_len = max_frames * 6 + 10;

    let mut hyp = vec![];
    let mut frame_ix = 0;
    let mut token = Tensor::from_slice(&[1, 1], &[0i32])?;
    let mut state_0 = Array3::<f32>::zeros([2, 1, 640]).tract()?;
    let mut state_1 = Array3::<f32>::zeros([2, 1, 640]).tract()?;

    (token, state_0, state_1) =
        run_decoder(&decoder, dec_wants_length, token, 1, state_0, state_1)?;
    while hyp.len() < max_len && frame_ix < max_frames {
        let frame = encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()).tract()?;
        let [logits] = joint.run([frame, token.clone()])?.try_into().unwrap();
        let logits = logits.as_slice::<f32>()?;
        let token_id = argmax(&logits[0..blank_id + 1]).unwrap();
        if token_id == blank_id {
            frame_ix += argmax(&logits[blank_id + 1..]).unwrap_or(0).max(1);
        } else {
            hyp.push(token_id);
            let next = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
            (token, state_0, state_1) =
                run_decoder(&decoder, dec_wants_length, next, 1, state_0, state_1)?;
        }
    }

    let transcript = hyp.into_iter().map(|t| vocab[t]).join("");
    println!("Transcript: {transcript}");
    assert_eq!(
        transcript,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    Ok(())
}
