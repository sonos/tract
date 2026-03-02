use std::fs::File;
use std::path::Path;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::prelude::*;
use tract_rs::Nnef;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

struct TdtModel {
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    joint: Runnable,
    vocab: Vec<String>,
    blank_id: usize,
}

impl TdtModel {
    fn load(model_dir: impl AsRef<Path>, nnef: &Nnef, gpu: &Runtime) -> Result<TdtModel> {
        let model_dir = model_dir.as_ref();
        let config: serde_json::Value =
            serde_json::from_reader(File::open(model_dir.join("model_config.json"))?)?;
        let blank_id =
            config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
        let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
        let vocab: Vec<String> = vocab.iter().map(|v| v.as_str().unwrap().to_owned()).collect();

        let preprocessor =
            nnef.load(model_dir.join("preprocessor.nnef.tgz"))?.into_runnable()?;

        let mut encoder = nnef.load(model_dir.join("encoder.nnef.tgz"))?;
        encoder.transform("transformers-detect-all")?;
        let encoder = gpu.prepare(encoder)?;

        let decoder = nnef.load(model_dir.join("decoder.nnef.tgz"))?;
        let decoder = gpu.prepare(decoder)?;

        let joint = nnef.load(model_dir.join("joint.nnef.tgz"))?;
        let joint = gpu.prepare(joint)?;

        Ok(TdtModel { preprocessor, encoder, decoder, joint, vocab, blank_id })
    }

    fn transcribe(&self, wav: &[f32]) -> Result<String> {
        let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
        let len: Value = arr1(&[wav.len() as i64]).try_into()?;

        let [features, feat_len] =
            self.preprocessor.run([samples, len])?.try_into().unwrap();
        let [encoded, _lens] =
            self.encoder.run([features, feat_len])?.try_into().unwrap();

        let encoded: ArrayD<f32> = encoded.view()?.into_owned();

        let max_frames = encoded.shape()[2];
        let max_len = max_frames * 6 + 10;

        let mut hyp = vec![];
        let mut frame_ix = 0;
        let mut token = Value::from_slice(&[1, 1], &[0i32])?;
        let mut state_0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
        let mut state_1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;

        [token, state_0, state_1] =
            self.decoder.run([token, state_0, state_1])?.try_into().unwrap();
        while hyp.len() < max_len && frame_ix < max_frames {
            let frame: Value =
                encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()).try_into()?;
            let [logits] = self.joint.run([frame, token.clone()])?.try_into().unwrap();
            let logits = logits.view::<f32>()?;
            let logits = logits.as_slice().unwrap();
            let token_id = argmax(&logits[0..self.blank_id + 1]).unwrap();
            if token_id == self.blank_id {
                frame_ix += argmax(&logits[self.blank_id + 1..]).unwrap_or(0).max(1);
            } else {
                hyp.push(token_id);
                token = Value::from_slice(&[1, 1], &[token_id as i32])?;
                [token, state_0, state_1] =
                    self.decoder.run([token, state_0, state_1])?.try_into().unwrap();
            }
        }

        Ok(hyp.into_iter().map(|t| self.vocab[t].as_str()).join(""))
    }
}

fn main() -> anyhow::Result<()> {
    let nnef = tract_rs::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract_rs::runtime_for_name(rt).ok())
        .unwrap();

    let model = TdtModel::load(Path::new("assets/model"), &nnef, &gpu)?;

    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32)
        .collect();

    let transcript = model.transcribe(&wav)?;
    println!("Transcript: {transcript}");
    assert_eq!(
        transcript,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    Ok(())
}
