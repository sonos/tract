//! Streaming speech recognition with `nvidia/nemotron-3.5-asr-streaming-0.6b`: the pulsed
//! preprocessor and the RNNT decoder and joint on the CPU, the pulsed encoder on tract's WebGPU
//! runtime, all their weights read from Hugging Face's own model.safetensors.

use anyhow::{Context, Result, ensure};
use tract::ReadAt;
use tract::prelude::*;

#[cfg(target_arch = "wasm32")]
mod web;

/// The sample rate the model takes.
pub const SAMPLE_RATE: usize = 16000;
/// Audio samples per preprocessor pulse.
const PREPROC_PULSE: usize = 1600;
/// Mel frames per encoder pulse: the model's attention chunk, 320 ms.
const ENCODER_PULSE: usize = 32;
const DECODER_HIDDEN: usize = 640;
const MAX_SYMBOLS_PER_FRAME: usize = 10;
/// The `lang_id` that has the model detect the language.
pub const AUTO_LANGUAGE: i64 = 101;

/// The four NNEF graphs, as `export.py` writes them.
pub struct Graphs<'a> {
    pub preprocessor: &'a [u8],
    pub encoder: &'a [u8],
    pub decoder: &'a [u8],
    pub joint: &'a [u8],
}

/// The loaded model, ready to start [`Transcriber`]s.
#[derive(Clone)]
pub struct Models {
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    joint: Runnable,
    vocab: Vec<String>,
    blank: usize,
    pp_delay: usize,
    pp_out_pulse: usize,
    mels: usize,
    enc_delay: usize,
    enc_out_axis: usize,
    enc_out_pulse: usize,
}

fn delay_and_axis(model: &Model) -> Result<(usize, usize)> {
    let delay = model.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
    let axis = model.property("pulse.output_axes")?.as_slice::<i64>()?[0] as usize;
    Ok((delay, axis))
}

impl Models {
    /// Loads the graphs, the vocabulary of Hugging Face's `tokenizer.json`, and the weights from
    /// the model.safetensors that each reader `weights` makes reads.
    pub fn load<R: ReadAt + 'static>(
        graphs: Graphs,
        tokenizer_json: &str,
        weights: impl Fn() -> Result<R>,
    ) -> Result<Models> {
        let nnef = tract::nnef()?.with_tract_transformers()?;

        let mut pp = nnef.load_buffer(graphs.preprocessor)?;
        pp.transform(SetSymbols::new().value("BATCH", 1))?;
        pp.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(input_signal)[1];"}"#,
        )?;
        pp.transform(r#"{"name":"select_inputs","inputs":["input_signal"]}"#)?;
        pp.transform(r#"{"name":"select_outputs","outputs":["processed_signal"]}"#)?;
        pp.transform(Pulse::new(PREPROC_PULSE.to_string()).symbol("INPUT_SIGNAL__TIME"))?;
        let (pp_delay, pp_out_axis) = delay_and_axis(&pp)?;
        let pp_out = pp.output_fact(0)?;
        ensure!(pp_out.rank()? == 3 && pp_out_axis == 2, "preprocessor output: {pp_out}");
        let mels = pp_out.dim(1)?.to_int64()? as usize;
        let pp_out_pulse = pp_out.dim(2)?.to_int64()? as usize;

        let mut enc = nnef.load_buffer(graphs.encoder)?;
        enc.link_safetensors(weights()?)?;
        enc.transform(SetSymbols::new().value("BATCH", 1))?;
        enc.transform("transformers_detect_all")?;
        enc.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#,
        )?;
        enc.transform(r#"{"name":"select_inputs","inputs":["audio_signal","lang_id"]}"#)?;
        enc.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
        enc.transform(Pulse::new(ENCODER_PULSE.to_string()).symbol("AUDIO_SIGNAL__TIME"))?;
        let (enc_delay, enc_out_axis) = delay_and_axis(&enc)?;
        let enc_out_pulse = enc.output_fact(0)?.dim(enc_out_axis)?.to_int64()? as usize;

        let mut dec = nnef.load_buffer(graphs.decoder)?;
        dec.link_safetensors(weights()?)?;
        dec.transform(SetSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;

        let mut joint = nnef.load_buffer(graphs.joint)?;
        joint.link_safetensors(weights()?)?;
        joint.transform(
            SetSymbols::new()
                .value("BATCH", 1)
                .value("ENCODER_OUTPUTS__TIME", 1)
                .value("DECODER_OUTPUTS__TIME", 1),
        )?;

        let vocab = vocabulary(tokenizer_json)?;
        Ok(Models {
            preprocessor: pp.into_runnable()?,
            encoder: tract::runtime_for_name("wgpu")?.prepare(enc)?,
            decoder: dec.into_runnable()?,
            joint: joint.into_runnable()?,
            blank: vocab.len(),
            vocab,
            pp_delay,
            pp_out_pulse,
            mels,
            enc_delay,
            enc_out_axis,
            enc_out_pulse,
        })
    }

    /// The text of these tokens, the language tags (`<en-US>`) the model emits included.
    pub fn text(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&t| self.vocab[t].as_str()).collect::<String>().replace('\u{2581}', " ")
    }

    /// One prednet step: the embedding of `token` and the state after it.
    fn decoder_step(&self, token: usize, state: [Tensor; 2]) -> Result<(Tensor, [Tensor; 2])> {
        let token = Tensor::from_slice(&[1, 1], &[token as i32])?;
        let target_length = Tensor::from_slice(&[1], &[1i32])?;
        let [s0, s1] = state;
        let [emb, _lengths, s0, s1] = self
            .decoder
            .run([token, target_length, s0, s1])?
            .try_into()
            .map_err(|_| anyhow::anyhow!("decoder outputs"))?;
        Ok((emb, [s0, s1]))
    }
}

/// The token strings by id, from a Hugging Face `tokenizer.json`.
fn vocabulary(tokenizer_json: &str) -> Result<Vec<String>> {
    let tokenizer: serde_json::Value = serde_json::from_str(tokenizer_json)?;
    let vocab = tokenizer["model"]["vocab"].as_object().context("tokenizer.json has no vocab")?;
    let mut tokens = vec![String::new(); vocab.len()];
    for (token, id) in vocab {
        let id = id.as_u64().context("token id")? as usize;
        *tokens.get_mut(id).context("sparse token ids")? = token.clone();
    }
    Ok(tokens)
}

/// One stream of 16 kHz audio in [-1, 1] being transcribed, greedily, as it comes.
pub struct Transcriber {
    models: Models,
    preprocessor: State,
    encoder: State,
    lang: Tensor,
    dec_emb: Tensor,
    dec_state: [Tensor; 2],
    audio: Vec<f32>,
    /// Mel frames not yet encoded, frame after frame.
    frames: Vec<f32>,
    pp_skip: usize,
    enc_skip: usize,
    tokens: Vec<usize>,
}

impl Transcriber {
    /// A stream in the language of that `lang_id` (`AUTO_LANGUAGE` to have it detected).
    pub fn new(models: &Models, lang_id: i64) -> Result<Transcriber> {
        let zeros = || Tensor::from_slice(&[2, 1, DECODER_HIDDEN], &[0f32; 2 * DECODER_HIDDEN]);
        // As NeMo's predict(add_sos=True): two blank steps through the prednet.
        let (_, state) = models.decoder_step(models.blank, [zeros()?, zeros()?])?;
        let (dec_emb, dec_state) = models.decoder_step(models.blank, state)?;
        Ok(Transcriber {
            preprocessor: models.preprocessor.spawn_state()?,
            encoder: models.encoder.spawn_state()?,
            lang: Tensor::from_slice(&[1], &[lang_id])?,
            dec_emb,
            dec_state,
            audio: vec![],
            frames: vec![],
            pp_skip: models.pp_delay,
            enc_skip: models.enc_delay,
            tokens: vec![],
            models: models.clone(),
        })
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn text(&self) -> String {
        self.models.text(&self.tokens)
    }

    /// Feeds more audio, decoding what it completes.
    pub async fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.audio.extend_from_slice(samples);
        let pulses = self.audio.len() / PREPROC_PULSE;
        for pulse in 0..pulses {
            let chunk = &self.audio[pulse * PREPROC_PULSE..][..PREPROC_PULSE];
            let chunk = Tensor::from_slice(&[1, PREPROC_PULSE], chunk)?;
            self.preprocess(chunk).await?;
        }
        self.audio.drain(..pulses * PREPROC_PULSE);
        Ok(())
    }

    /// Ends the stream: what is left goes through, padded with silence.
    pub async fn finish(&mut self) -> Result<()> {
        if !self.audio.is_empty() {
            self.audio.resize(PREPROC_PULSE, 0.0);
            let chunk = Tensor::from_slice(&[1, PREPROC_PULSE], &self.audio)?;
            self.audio.clear();
            self.preprocess(chunk).await?;
        }
        if !self.frames.is_empty() {
            self.frames.resize(ENCODER_PULSE * self.models.mels, 0.0);
            self.encode().await?;
        }
        Ok(())
    }

    async fn preprocess(&mut self, chunk: Tensor) -> Result<()> {
        let [features] = self
            .preprocessor
            .run([chunk])?
            .try_into()
            .map_err(|_| anyhow::anyhow!("preprocessor outputs"))?;
        let features = features.as_slice::<f32>()?;
        let (mels, pulse) = (self.models.mels, self.models.pp_out_pulse);
        let skip = self.pp_skip.min(pulse);
        self.pp_skip -= skip;
        for frame in skip..pulse {
            self.frames.extend((0..mels).map(|mel| features[mel * pulse + frame]));
            if self.frames.len() == ENCODER_PULSE * mels {
                self.encode().await?;
            }
        }
        Ok(())
    }

    async fn encode(&mut self) -> Result<()> {
        let mels = self.models.mels;
        let input: Vec<f32> = (0..mels)
            .flat_map(|mel| (0..ENCODER_PULSE).map(move |frame| (mel, frame)))
            .map(|(mel, frame)| self.frames[frame * mels + mel])
            .collect();
        self.frames.clear();
        let input = Tensor::from_slice(&[1, mels, ENCODER_PULSE], &input)?;
        let [encoded] = self
            .encoder
            .run([input, self.lang.clone()])?
            .try_into()
            .map_err(|_| anyhow::anyhow!("encoder outputs"))?;
        let encoded = encoded.to_host_async().await?;
        for frame in 0..self.models.enc_out_pulse {
            if self.enc_skip > 0 {
                self.enc_skip -= 1;
                continue;
            }
            self.decode(encoded.sliced(self.models.enc_out_axis, frame, frame + 1)?)?;
        }
        Ok(())
    }

    fn decode(&mut self, frame: Tensor) -> Result<()> {
        for _ in 0..MAX_SYMBOLS_PER_FRAME {
            let [logits] = self
                .models
                .joint
                .run([frame.clone(), self.dec_emb.clone()])?
                .try_into()
                .map_err(|_| anyhow::anyhow!("joint outputs"))?;
            let logits = logits.as_slice::<f32>()?;
            let token = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
            if token == self.models.blank {
                break;
            }
            self.tokens.push(token);
            let state = self.dec_state.clone();
            (self.dec_emb, self.dec_state) = self.models.decoder_step(token, state)?;
        }
        Ok(())
    }
}

/// Brings audio at a multiple of [`SAMPLE_RATE`] down to it: a windowed-sinc low-pass at
/// 7.2 kHz, kept every `factor` samples.
pub struct Decimator {
    factor: usize,
    taps: Vec<f32>,
    history: Vec<f32>,
}

impl Decimator {
    pub fn new(input_rate: usize) -> Result<Decimator> {
        ensure!(
            input_rate.is_multiple_of(SAMPLE_RATE),
            "{input_rate} Hz audio, expected a multiple of {SAMPLE_RATE} Hz"
        );
        let factor = input_rate / SAMPLE_RATE;
        let taps = if factor == 1 { vec![1.0] } else { low_pass(127, 7200.0 / input_rate as f64) };
        Ok(Decimator { factor, history: vec![0.0; taps.len() - 1], taps })
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.history.extend_from_slice(samples);
        let n = self.taps.len();
        let out: Vec<f32> = (0..)
            .map(|i| i * self.factor)
            .take_while(|start| start + n <= self.history.len())
            .map(|start| {
                self.history[start..][..n].iter().zip(&self.taps).map(|(a, b)| a * b).sum()
            })
            .collect();
        self.history.drain(..out.len() * self.factor);
        out
    }
}

/// A Blackman-windowed sinc low-pass of unit gain, its cutoff as a fraction of the sample rate.
fn low_pass(taps: usize, cutoff: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let m = (taps - 1) as f64;
    let raw: Vec<f64> = (0..taps)
        .map(|i| {
            let t = i as f64 - m / 2.0;
            let x = 2.0 * cutoff * t;
            let sinc = if t == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) };
            let window = 0.42 - 0.5 * (2.0 * PI * i as f64 / m).cos()
                + 0.08 * (4.0 * PI * i as f64 / m).cos();
            sinc * window
        })
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|t| (t / sum) as f32).collect()
}
