use std::fs::File;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract::prelude::tract_ndarray::prelude::*;
use tract::prelude::*;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

fn concretize_batch(mut model: Model) -> anyhow::Result<Model> {
    model.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
    Ok(model)
}

fn main() -> anyhow::Result<()> {
    let config: serde_json::Value =
        serde_json::from_reader(File::open("assets/model/model_config.json")?)?;
    let blank_id = config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
    let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
    let vocab: Vec<&str> = vocab.iter().map(|v| v.as_str().unwrap()).collect();

    let nnef = tract::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();

    let preprocessor =
        concretize_batch(nnef.load("assets/model/preprocessor.nnef.tgz")?)?.into_runnable()?;

    let mut encoder = nnef.load("assets/model/encoder.nnef.tgz")?;
    encoder.transform("transformers_detect_all")?;
    let encoder = gpu.prepare(concretize_batch(encoder)?)?;

    let decoder = gpu.prepare(concretize_batch(nnef.load("assets/model/decoder.nnef.tgz")?)?)?;
    let joint = gpu.prepare(concretize_batch(nnef.load("assets/model/joint.nnef.tgz")?)?)?;

    // soundfile (Python) normalizes i16 PCM to [-1, 1]; match that here
    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let samples = Tensor::from_slice(&[1, wav.len()], &wav)?;
    let len = tensor(arr1(&[wav.len() as i64]))?;

    let [features, feat_len] = preprocessor.run([samples, len])?.try_into().unwrap();
    let [encoded, _lens] = encoder.run([features, feat_len])?.try_into().unwrap();

    let encoded: ArrayD<f32> = encoded.view()?.into_owned();

    let max_frames = encoded.shape()[2];
    let max_len = max_frames * 6 + 10;

    let mut hyp = vec![];
    let mut frame_ix = 0;

    // Warm-up: NeMo's predict(add_sos=True, y=None) prepends a zero vector then processes
    // one zero embedding — 2 steps of zero input total. Token blank_id has zero embedding
    // (padding_idx=blank_id), so pass [blank_id, blank_id] and take the last output step.
    let warmup_tokens = Tensor::from_slice(&[1, 2], &[blank_id as i32, blank_id as i32])?;
    let state_0 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    let state_1 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    let [warmup_out, mut state_0, mut state_1] =
        decoder.run([warmup_tokens, state_0, state_1])?.try_into().unwrap();
    // warmup_out shape: [1, 640, 2] — take last timestep → [1, 640, 1]
    let warmup_out: ArrayD<f32> = warmup_out.view()?.into_owned();
    let mut token: Tensor = tensor(warmup_out.slice_axis(Axis(2), (1..2).into()).to_owned())?;

    while hyp.len() < max_len && frame_ix < max_frames {
        let frame: Tensor = tensor(encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()))?;
        let [logits] = joint.run([frame, token.clone()])?.try_into().unwrap();
        let logits = logits.view::<f32>()?;
        let logits = logits.as_slice().unwrap();
        let token_id = argmax(logits).unwrap();
        if token_id == blank_id {
            frame_ix += 1;
        } else {
            hyp.push(token_id);
            token = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
            [token, state_0, state_1] = decoder.run([token, state_0, state_1])?.try_into().unwrap();
        }
    }

    let transcript = hyp.into_iter().map(|t| vocab[t]).join("");
    println!("Transcript: {transcript}");
    assert_eq!(
        transcript,
        "▁well▁I▁don't▁wish▁to▁see▁it▁any▁more▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait"
    );
    Ok(())
}
