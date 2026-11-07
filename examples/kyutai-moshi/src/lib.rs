//! Moshi on tract: Mimi in and out, Helium and the depformer on Kyutai's delayed token
//! schedule, one 80 ms frame per step.

use anyhow::*;
use tract::prelude::*;

pub const FRAME: usize = 1920;
pub const SAMPLE_RATE: usize = 24000;
const DELAYS: [usize; 17] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
const RING: usize = 3;
const TEXT_INITIAL: i64 = 32000;
const AUDIO_INITIAL: i64 = 2048;

/// The loaded graphs, shared by every session.
pub struct Models {
    encoder: Runnable,
    decoder: Runnable,
    helium: Runnable,
    depformer: Vec<Runnable>,
    vocab: Vec<String>,
}

fn pulsed(
    nnef: &Nnef,
    runtime: &Runtime,
    path: String,
    symbol: &str,
    pulse: usize,
) -> Result<Runnable> {
    let mut model = nnef.load(path)?;
    model.transform(Pulse::new(pulse.to_string()).symbol(symbol))?;
    ensure!(model.property("pulse.delay")?.as_slice::<i64>()?[0] == 0);
    runtime.prepare(model)
}

impl Models {
    /// Loads the graphs export.py wrote to `dir`, folding Helium's K/V into a stateful cache
    /// and pulsing Mimi to one frame.
    pub fn load(dir: &str, runtime: &Runtime) -> Result<Models> {
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let encoder = pulsed(&nnef, runtime, format!("{dir}/mimi-encoder.nnef.tgz"), "T", FRAME)?;
        let decoder = pulsed(&nnef, runtime, format!("{dir}/mimi-decoder.nnef.tgz"), "F", 1)?;
        let mut helium = nnef.load(format!("{dir}/helium.nnef.tgz"))?;
        helium.transform("transformers_detect_all")?;
        let helium = runtime.prepare(helium)?;
        let depformer = (0..8)
            .map(|k| runtime.prepare(nnef.load(format!("{dir}/depformer-step{k}.nnef.tgz"))?))
            .collect::<Result<Vec<_>>>()?;
        let vocab = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/vocab.json"))?)?;
        Ok(Models { encoder, decoder, helium, depformer, vocab })
    }

    /// A conversation from its first frame.
    pub fn session(&self, seed: u64, greedy: bool) -> Result<Session<'_>> {
        Ok(Session {
            models: self,
            encoder: self.encoder.spawn_state()?,
            decoder: self.decoder.spawn_state()?,
            helium: self.helium.spawn_state()?,
            cache: [[-2; RING]; 17],
            offset: 0,
            sampler: Sampler { state: seed.max(1), greedy },
        })
    }
}

/// What one step produced: nothing for the first frame, then Moshi's text piece (if it said
/// a word), its 9 tokens and 1920 samples of its voice.
pub struct Frame {
    pub text: Option<String>,
    pub tokens: Option<Vec<i64>>,
    pub audio: Vec<f32>,
}

pub struct Session<'m> {
    models: &'m Models,
    encoder: State,
    decoder: State,
    helium: State,
    cache: [[i64; RING]; 17],
    offset: usize,
    sampler: Sampler,
}

fn to_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.convert_to(DatumType::F32)?.as_slice::<f32>()?.to_vec())
}

impl Session<'_> {
    /// One frame of the user's audio, 1920 samples at 24 kHz.
    pub fn step_audio(&mut self, pcm: &[f32]) -> Result<Frame> {
        ensure!(pcm.len() == FRAME);
        let out = self.encoder.run([Tensor::from_slice(&[1, 1, FRAME], pcm)?])?;
        let codes: [i64; 8] = out[0].as_slice::<i64>()?.try_into()?;
        self.step_codes(&codes)
    }

    /// One frame of the user's Mimi codes.
    pub fn step_codes(&mut self, user: &[i64; 8]) -> Result<Frame> {
        let offset = self.offset;
        for (c, &code) in user.iter().enumerate() {
            self.cache[9 + c][(offset + DELAYS[9 + c]) % RING] = code;
        }
        let tokens: Vec<i64> = (0..17)
            .map(|c| {
                if offset <= DELAYS[c] {
                    if c == 0 { TEXT_INITIAL } else { AUDIO_INITIAL }
                } else {
                    self.cache[c][offset % RING]
                }
            })
            .collect();
        ensure!(tokens.iter().all(|&t| t >= 0), "ungenerated token at frame {offset}: {tokens:?}");
        let mut out = self.helium.run([Tensor::from_slice(&[1, 17, 1], &tokens)?])?;
        let text_logits = to_f32(&out[1])?;
        let hidden = out.remove(0);
        let text_token = self.sampler.sample(&text_logits, 0.7, 25);

        let mut codes = [0i64; 8];
        let mut prev = text_token;
        let mut past: Vec<Tensor> = vec![];
        for (k, step) in self.models.depformer.iter().enumerate() {
            let mut inputs = vec![Tensor::from_slice(&[1, 1], &[prev])?, hidden.clone()];
            inputs.append(&mut past);
            let mut out = step.run(inputs)?;
            let logits = to_f32(&out.remove(0))?;
            past = out;
            codes[k] = self.sampler.sample(&logits, 0.8, 250);
            prev = codes[k];
        }

        let next = offset + 1;
        self.offset = next;
        self.cache[0][next % RING] = text_token;
        for (k, &code) in codes.iter().enumerate() {
            self.cache[1 + k][next % RING] = code;
        }
        if next <= 1 {
            return Ok(Frame { text: None, tokens: None, audio: vec![] });
        }
        let emitted: Vec<i64> =
            (0..9).map(|c| self.cache[c][(next - 1 + DELAYS[c]) % RING]).collect();
        let text = (emitted[0] != 0 && emitted[0] != 3)
            .then(|| self.models.vocab[emitted[0] as usize].replace('\u{2581}', " "));
        let audio = self.decoder.run([Tensor::from_slice(&[1, 8, 1], &emitted[1..])?])?;
        let audio = audio[0].as_slice::<f32>()?.to_vec();
        Ok(Frame { text, tokens: Some(emitted), audio })
    }
}

struct Sampler {
    state: u64,
    greedy: bool,
}

impl Sampler {
    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Kyutai's sample_token: softmax at `temp` over the `top_k` largest logits.
    fn sample(&mut self, logits: &[f32], temp: f32, top_k: usize) -> i64 {
        let mut ix: Vec<usize> = (0..logits.len()).collect();
        ix.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.greedy {
            return ix[0] as i64;
        }
        ix.truncate(top_k);
        let max = logits[ix[0]];
        let weights: Vec<f64> =
            ix.iter().map(|&i| (((logits[i] - max) / temp) as f64).exp()).collect();
        let mut r = self.uniform() * weights.iter().sum::<f64>();
        for (w, &i) in weights.iter().zip(&ix) {
            r -= w;
            if r <= 0.0 {
                return i as i64;
            }
        }
        ix[ix.len() - 1] as i64
    }
}
