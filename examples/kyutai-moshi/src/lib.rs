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
    depformer: Runnable,
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
    ///
    /// With `q4`, Helium's and the depformer's weights go to Q4_0 blocks: about a quarter of
    /// the memory and of the bandwidth a frame takes, for GPUs short of either.
    pub fn load(dir: &str, runtime: &Runtime, q4: bool) -> Result<Models> {
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let encoder = pulsed(&nnef, runtime, format!("{dir}/mimi-encoder.nnef.tgz"), "T", FRAME)?;
        let decoder = pulsed(&nnef, runtime, format!("{dir}/mimi-decoder.nnef.tgz"), "F", 1)?;
        let mut helium = nnef.load(format!("{dir}/helium.nnef.tgz"))?;
        helium.transform("transformers_detect_all")?;
        if q4 {
            helium.transform("block_quant")?;
        }
        let helium = runtime.prepare(helium)?;
        let mut depformer = nnef.load(format!("{dir}/depformer.nnef.tgz"))?;
        if q4 {
            depformer.transform("block_quant")?;
        }
        let depformer = runtime.prepare(depformer)?;
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
        let text_logits = out.remove(1);
        let hidden = out.remove(0);
        let noise = Tensor::from_slice(&[9, 250], &self.sampler.gumbel(9 * 250))?;
        let sampled = self.models.depformer.run([text_logits, hidden, noise])?;
        let sampled = sampled[0].as_slice::<i64>()?;

        let next = offset + 1;
        self.offset = next;
        for (c, &token) in sampled.iter().enumerate() {
            self.cache[c][next % RING] = token;
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
    /// Standard Gumbel noise for the depformer's draws, or zeros for greedy decoding.
    fn gumbel(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                if self.greedy {
                    return 0.0;
                }
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                let u = ((self.state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                -(-u.ln()).ln() as f32
            })
            .collect()
    }
}

/// Wire format between the server and its client, both ways little-endian: the client sends
/// frames of 1920 f32 samples; the server answers each with a u32 text length, the text,
/// a u32 sample count and the samples (0 for the first frame).
pub mod wire {
    use std::io::{Read, Write};

    pub fn write_frame(w: &mut impl Write, pcm: &[f32]) -> std::io::Result<()> {
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        w.write_all(&bytes)
    }

    pub fn read_frame(r: &mut impl Read) -> std::io::Result<Vec<f32>> {
        let mut bytes = vec![0u8; super::FRAME * 4];
        r.read_exact(&mut bytes)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    pub fn write_reply(w: &mut impl Write, text: &str, audio: &[f32]) -> std::io::Result<()> {
        let mut out = (text.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(text.as_bytes());
        out.extend_from_slice(&(audio.len() as u32).to_le_bytes());
        out.extend(audio.iter().flat_map(|s| s.to_le_bytes()));
        w.write_all(&out)
    }

    pub fn read_reply(r: &mut impl Read) -> std::io::Result<(String, Vec<f32>)> {
        let mut len = [0u8; 4];
        r.read_exact(&mut len)?;
        let mut text = vec![0u8; u32::from_le_bytes(len) as usize];
        r.read_exact(&mut text)?;
        r.read_exact(&mut len)?;
        let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize * 4];
        r.read_exact(&mut bytes)?;
        let audio = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok((String::from_utf8_lossy(&text).into_owned(), audio))
    }
}

/// Linear-interpolation resampler for a mono stream, stateful across chunks.
pub struct Resampler {
    step: f64,
    pos: f64,
    pending: Vec<f32>,
}

impl Resampler {
    pub fn new(from_rate: f64, to_rate: f64) -> Resampler {
        Resampler { step: from_rate / to_rate, pos: 0.0, pending: vec![] }
    }

    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        while self.pos + 1.0 < self.pending.len() as f64 {
            let i = self.pos as usize;
            let frac = (self.pos - i as f64) as f32;
            out.push(self.pending[i] * (1.0 - frac) + self.pending[i + 1] * frac);
            self.pos += self.step;
        }
        let consumed = (self.pos as usize).min(self.pending.len());
        self.pending.drain(..consumed);
        self.pos -= consumed as f64;
    }
}
