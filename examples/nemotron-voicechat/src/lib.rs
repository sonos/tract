//! NVIDIA's NemotronLabs VoiceChat on tract: a full-duplex voice assistant streamed one 80 ms
//! frame at a time. Each frame of microphone audio goes through a pulsed mel spectrogram and
//! speech encoder; the encoder frame drives one step of the Nemotron-H language model (text
//! out), one step of the EAR-TTS speech generator (31 codec codes out) and one frame of the
//! pulsed codec decoder (1764 samples at 22.05 kHz out). An RNNT head on the encoder transcribes
//! the user and drives NVIDIA's turn-taking rules.

use std::time::Instant;

use anyhow::*;
use tract::prelude::*;

/// Microphone sample rate.
pub const INPUT_RATE: u32 = 16000;
/// Voice sample rate.
pub const OUTPUT_RATE: u32 = 22050;
/// Seconds per frame.
pub const FRAME_SECS: f64 = 0.08;
pub const PAD: i64 = 12;
pub const BOS: i64 = 1;
pub const EOS: i64 = 2;

const CONV: [usize; 3] = [1, 12288, 4];
const SSM: [usize; 4] = [1, 128, 80, 128];
const HIDDEN: usize = 1152;
const CODEBOOKS: usize = 31;
const MASKED_CODE: i64 = 1024;
const CHUNK: usize = 1280;
const HALF_FFT: usize = 256;
const PREEMPH: f32 = 0.97;
const OLA_CROP: usize = 6;
const RNNT_BLANK: usize = 1024;
const RNNT_MAX_SYMBOLS: usize = 10;
const PUNCT: [(&str, f32); 4] = [(".", 1.5), (",", 1.0), ("?", 1.5), ("!", 1.0)];
const PUNCT_MIN_BLANKS: u32 = 5;
const LOOKBACK: usize = 40;
const EOU_FRAMES: u32 = 40;
const BOU_FRAMES: u32 = 40;
const MIN_SPEECH_FIRST_TURN: u32 = 2;
const MIN_SPEECH: u32 = 3;
const LOW_DENSITY: f32 = 0.15;
const MIN_SPEECH_LOW_DENSITY: u32 = 6;
const NOISE_RESET_FRAMES: u32 = 10;
const FIRST_TURN_FALLBACK: usize = 50;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0
}

fn json(path: &str) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn ids(v: &serde_json::Value) -> Result<Vec<i64>> {
    v.as_array()
        .context("expected an array")?
        .iter()
        .map(|x| x.as_i64().context("expected an integer"))
        .collect()
}

fn pulsed(nnef: &Nnef, path: String, symbol: &str, pulse: usize) -> Result<(Model, usize)> {
    let mut model = nnef.load(path)?;
    model.transform(Pulse::new(pulse.to_string()).symbol(symbol))?;
    let delay = model.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
    Ok((model, delay))
}

/// The loaded graphs and tables, shared by every session.
pub struct Models {
    mel: Runnable,
    mel_delay: usize,
    mel_pulse: usize,
    encoder: Runnable,
    encoder_delay: usize,
    llm: Runnable,
    tts: Runnable,
    codec: Runnable,
    codec_delay: usize,
    rnnt_decoder: Runnable,
    rnnt_decoder_wants_length: bool,
    rnnt_joint: Runnable,
    rnnt_vocab: Vec<String>,
    punct: Vec<(usize, f32)>,
    prompt: Vec<i64>,
    silence: Vec<i64>,
    vocab: Vec<String>,
    warmup: Vec<f32>,
    warmup_len: usize,
}

impl Models {
    /// Loads the graphs export.py wrote to `dir`: pulses the mel, encoder and codec graphs to
    /// one frame, folds the language model's and the TTS's K/V into stateful caches.
    ///
    /// With `q4`, the language model's weights go to Q4_0 blocks: about a quarter of its 18 GB.
    pub fn load(dir: &str, runtime: &Runtime, q4: bool) -> Result<Models> {
        let path = |name: &str| format!("{dir}/{name}");
        let meta = json(&path("meta.json"))?;
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let (mel, mel_delay) = pulsed(&nnef, path("mel.nnef.tgz"), "N", CHUNK)?;
        let mel_pulse = mel.output_fact(0)?.dim(2)?.to_int64()? as usize;
        let mut encoder = nnef.load(path("encoder.nnef.tgz"))?;
        encoder.transform(SetSymbols::new().value("BATCH", 1))?;
        encoder.transform("transformers_detect_all")?;
        encoder.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#,
        )?;
        encoder.transform(r#"{"name":"select_inputs","inputs":["audio_signal"]}"#)?;
        encoder.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
        encoder.transform(Pulse::new("8").symbol("AUDIO_SIGNAL__TIME"))?;
        let encoder_delay = encoder.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
        let mut llm = nnef.load(path("llm.nnef.tgz"))?;
        llm.transform("transformers_detect_all")?;
        if q4 {
            llm.transform("block_quant")?;
        }
        let mut tts = nnef.load(path("tts.nnef.tgz"))?;
        tts.transform("transformers_detect_all")?;
        let (codec, codec_delay) = pulsed(&nnef, path("codec.nnef.tgz"), "T", 1)?;
        let mut rnnt_decoder = nnef.load(path("rnnt-decoder.nnef.tgz"))?;
        rnnt_decoder.transform(SetSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;
        let mut rnnt_joint = nnef.load(path("rnnt-joint.nnef.tgz"))?;
        rnnt_joint.transform(
            SetSymbols::new()
                .value("ENCODER_OUTPUTS__BATCH", 1)
                .value("ENCODER_OUTPUTS__TIME", 1)
                .value("DECODER_OUTPUTS__BATCH", 1)
                .value("DECODER_OUTPUTS__TIME", 1),
        )?;
        let rnnt_vocab: Vec<String> = serde_json::from_value(json(&path("rnnt-vocab.json"))?)?;
        let punct = PUNCT
            .iter()
            .filter_map(|(p, inc)| rnnt_vocab.iter().position(|v| v == p).map(|i| (i, *inc)))
            .collect();
        let warmup = std::fs::read(path("warmup_f32.bin"))?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(Models {
            mel: runtime.prepare(mel)?,
            mel_delay,
            mel_pulse,
            encoder: runtime.prepare(encoder)?,
            encoder_delay,
            llm: runtime.prepare(llm)?,
            tts: runtime.prepare(tts)?,
            codec: runtime.prepare(codec)?,
            codec_delay,
            rnnt_decoder_wants_length: rnnt_decoder.input_count()? == 4,
            rnnt_decoder: runtime.prepare(rnnt_decoder)?,
            rnnt_joint: runtime.prepare(rnnt_joint)?,
            rnnt_vocab,
            punct,
            prompt: ids(&meta["prompt"])?,
            silence: ids(&meta["silence"])?,
            vocab: serde_json::from_value(json(&path("vocab.json"))?)?,
            warmup,
            warmup_len: meta["warm_len"].as_u64().context("warm_len")? as usize,
        })
    }

    /// The text of a language model token; empty for PAD, a newline for EOS.
    pub fn piece(&self, token: i64) -> &str {
        match token {
            PAD | BOS => "",
            EOS => "\n",
            _ => &self.vocab[token as usize],
        }
    }
}

/// Xorshift draws for the TTS sampler.
struct Draws(u64);

impl Draws {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64) as f32
    }

    /// The five sampler passes' uniforms [5, 1024] and standard normals [5, 512].
    fn step(&mut self) -> (Vec<f32>, Vec<f32>) {
        let u = (0..5120).map(|_| self.uniform()).collect();
        let n = (0..2560)
            .map(|_| {
                let (a, b) = (self.uniform() as f64, self.uniform() as f64);
                ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
            })
            .collect();
        (u, n)
    }
}

/// The RNNT head, greedy over the encoder frames as NVIDIA's streaming wrapper runs it,
/// punctuation bias included: after words, silent frames grow a boost on punctuation tokens.
/// The bias also advances the predictor, so it changes what follows, not only the display.
struct Rnnt {
    decoded: Tensor,
    state_0: Tensor,
    state_1: Tensor,
    punct_bias: f32,
    words_since_punct: bool,
}

impl Rnnt {
    fn new(models: &Models) -> Result<Rnnt> {
        let zeros = || Tensor::from_slice(&[2, 1, 640], &[0f32; 1280]);
        let mut r = Rnnt {
            decoded: Tensor::from_slice(&[1], &[0f32])?,
            state_0: zeros()?,
            state_1: zeros()?,
            punct_bias: 0.0,
            words_since_punct: false,
        };
        // The start-of-sequence prediction: two zero embeddings, which is how blank embeds.
        r.advance(models, RNNT_BLANK)?;
        r.advance(models, RNNT_BLANK)?;
        Ok(r)
    }

    fn advance(&mut self, models: &Models, token: usize) -> Result<()> {
        let mut inputs = vec![Tensor::from_slice(&[1, 1], &[token as i32])?];
        if models.rnnt_decoder_wants_length {
            inputs.push(Tensor::from_slice(&[1], &[1i32])?);
        }
        inputs.push(self.state_0.clone());
        inputs.push(self.state_1.clone());
        let mut out = models.rnnt_decoder.run(inputs)?;
        self.state_1 = out.pop().context("decoder state")?;
        self.state_0 = out.pop().context("decoder state")?;
        self.decoded = out.swap_remove(0);
        Ok(())
    }

    fn logits(&self, models: &Models, frame: &Tensor) -> Result<Vec<f32>> {
        let out = models.rnnt_joint.run([frame.clone(), self.decoded.clone()])?.remove(0);
        Ok(out.convert_to(DatumType::F32)?.as_slice::<f32>()?.to_vec())
    }

    /// One encoder frame: whether its first symbol is blank, and the tokens it emits.
    fn step(
        &mut self,
        models: &Models,
        frame: &Tensor,
        blank_count: u32,
    ) -> Result<(bool, Vec<usize>)> {
        let logits = self.logits(models, frame)?;
        let mut token = argmax(&logits);
        let is_blank = token == RNNT_BLANK;
        let is_punct = |t: usize| models.punct.iter().any(|&(id, _)| id == t);
        let mut emitted = vec![];
        if is_blank && self.punct_bias > 0.0 {
            let mut biased = logits;
            for &(id, inc) in &models.punct {
                biased[id] += self.punct_bias * inc;
            }
            let p = argmax(&biased);
            if is_punct(p) {
                emitted.push(p);
                self.advance(models, p)?;
            }
        }
        while token != RNNT_BLANK && emitted.len() < RNNT_MAX_SYMBOLS {
            emitted.push(token);
            self.advance(models, token)?;
            token = argmax(&self.logits(models, frame)?);
        }
        if emitted.iter().any(|&t| is_punct(t)) {
            self.words_since_punct = false;
            self.punct_bias = 0.0;
        } else if !emitted.is_empty() {
            self.words_since_punct = true;
            self.punct_bias = 0.0;
        } else if self.words_since_punct && blank_count + 1 >= PUNCT_MIN_BLANKS {
            self.punct_bias += 1.0;
        }
        Ok((is_blank, emitted))
    }
}

/// NVIDIA's RNNT turn-taking rules with the defaults of its streaming configuration: the
/// agent's BOS is forced after 3.2 s of user silence following speech, its EOS after 3.2 s of
/// user speech over it, and a BOS on the first turn once the user has spoken after 4 s.
/// The rules that configuration leaves off (maximum response length, the TTS ratio cap,
/// post-EOS fallbacks, self-play suppression) are not implemented.
struct TurnTaking {
    blank_count: u32,
    nonblank_consec: u32,
    nonblank_total: u32,
    speech_confirmed: bool,
    agent_speaking: bool,
    first_turn: bool,
    user_first_turn: bool,
    density: f32,
    history: Vec<i64>,
}

impl TurnTaking {
    fn new() -> TurnTaking {
        TurnTaking {
            blank_count: 0,
            nonblank_consec: 0,
            nonblank_total: 0,
            speech_confirmed: false,
            agent_speaking: false,
            first_turn: true,
            user_first_turn: true,
            density: 0.0,
            history: vec![],
        }
    }

    fn observe(&mut self, is_blank: bool) {
        if !self.agent_speaking {
            self.density = 0.1 * (!is_blank as u8 as f32) + 0.9 * self.density;
        }
        if is_blank {
            self.blank_count += 1;
            self.nonblank_consec = 0;
        } else {
            self.blank_count = 0;
            self.nonblank_consec += 1;
            self.nonblank_total += 1;
        }
    }

    fn agent_starts(&mut self) {
        self.speech_confirmed = false;
        self.nonblank_total = 0;
        self.agent_speaking = true;
        self.first_turn = false;
        self.user_first_turn = false;
    }

    fn decide(&mut self, token: i64) -> (i64, Option<&'static str>) {
        let t = self.history.len();
        let window = &self.history[t.saturating_sub(LOOKBACK)..];
        let (bos_in_window, eos_in_window) = (window.contains(&BOS), window.contains(&EOS));
        let mut min_speech = if self.user_first_turn { MIN_SPEECH_FIRST_TURN } else { MIN_SPEECH };
        if !self.first_turn && self.density > 0.0 && self.density < LOW_DENSITY {
            min_speech = MIN_SPEECH_LOW_DENSITY;
        }
        if bos_in_window || token == BOS {
            self.agent_speaking = true;
            self.first_turn = false;
            if token == BOS {
                self.speech_confirmed = false;
                self.nonblank_total = 0;
            }
        }
        if eos_in_window || token == EOS {
            self.agent_speaking = false;
            self.speech_confirmed = false;
            self.nonblank_total = 0;
        }
        if self.blank_count >= NOISE_RESET_FRAMES && !self.speech_confirmed && !self.agent_speaking
        {
            self.nonblank_total = 0;
        }
        if (self.nonblank_consec >= min_speech || self.nonblank_total >= min_speech)
            && !self.agent_speaking
        {
            self.speech_confirmed = true;
        }
        if self.first_turn && t >= FIRST_TURN_FALLBACK && self.speech_confirmed && !bos_in_window {
            self.agent_starts();
            return (BOS, Some("first-turn fallback: agent starts"));
        }
        if self.blank_count >= EOU_FRAMES && self.speech_confirmed && !self.agent_speaking {
            if !bos_in_window && token != BOS {
                self.agent_starts();
                return (BOS, Some("end of user turn: agent starts"));
            }
            return (token, None);
        }
        if self.nonblank_consec >= BOU_FRAMES
            && self.agent_speaking
            && !eos_in_window
            && token != EOS
        {
            self.nonblank_consec = 0;
            self.nonblank_total = 0;
            self.agent_speaking = false;
            return (EOS, Some("barge-in: agent stops"));
        }
        (token, None)
    }

    /// The agent token for this frame, possibly forced, and the rule that forced it.
    fn apply(&mut self, is_blank: bool, token: i64) -> (i64, Option<&'static str>) {
        self.observe(is_blank);
        let decided = self.decide(token);
        self.history.push(decided.0);
        decided
    }
}

/// What happened in one 80 ms frame.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The RNNT heard speech: the frame's first symbol is not blank.
    pub user_speech: bool,
    /// The RNNT transcript of the user emitted on this frame.
    pub user: String,
    /// The agent's text token, after turn-taking.
    pub agent: i64,
    /// The language model's own token, when turn-taking replaced it.
    pub model_said: Option<i64>,
    /// The turn-taking rule that fired on this frame.
    pub rule: Option<&'static str>,
    /// Compute time of the frame, from encoder to codec.
    pub millis: f64,
}

/// One conversation. Push 16 kHz samples in any amount; every complete 80 ms of input yields a
/// frame and 1764 samples of the agent's voice.
pub struct Session<'m> {
    models: &'m Models,
    mel: State,
    encoder: State,
    llm: State,
    tts: State,
    codec: State,
    llm_states: Vec<Tensor>,
    prev_text: i64,
    prev_func: i64,
    code: Vec<i64>,
    draws: Draws,
    last_sample: Option<f32>,
    pending: Vec<f32>,
    started: bool,
    mel_frames: Vec<Vec<f32>>,
    mel_seen: usize,
    encoder_seen: usize,
    voice_out: usize,
    rnnt: Rnnt,
    turns: Option<TurnTaking>,
    frames: Vec<Frame>,
}

impl<'m> Session<'m> {
    /// A new conversation: runs the voice warm-up and the system prompt.
    pub fn new(models: &'m Models, seed: u64, turn_taking: bool) -> Result<Session<'m>> {
        let llm_states = (0..models.llm.input_count()? - 5)
            .map(|i| {
                if i % 2 == 0 {
                    Tensor::from_slice(&CONV, &vec![0f32; CONV.iter().product()])?
                        .convert_to(DatumType::F16)
                } else {
                    Tensor::from_slice(&SSM, &vec![0f32; SSM.iter().product()])
                }
            })
            .collect::<Result<_>>()?;
        let mut s = Session {
            models,
            mel: models.mel.spawn_state()?,
            encoder: models.encoder.spawn_state()?,
            llm: models.llm.spawn_state()?,
            tts: models.tts.spawn_state()?,
            codec: models.codec.spawn_state()?,
            llm_states,
            prev_text: PAD,
            prev_func: PAD,
            code: vec![MASKED_CODE; CODEBOOKS],
            draws: Draws(seed.max(1)),
            last_sample: None,
            pending: vec![],
            started: false,
            mel_frames: vec![],
            mel_seen: 0,
            encoder_seen: 0,
            voice_out: 0,
            rnnt: Rnnt::new(models)?,
            turns: turn_taking.then(TurnTaking::new),
            frames: vec![],
        };
        let (w, n) = (&models.warmup, models.warmup_len);
        for i in 0..n {
            let mut cond_uncond = w[i * HIDDEN..][..HIDDEN].to_vec();
            cond_uncond.extend_from_slice(&w[(n + i) * HIDDEN..][..HIDDEN]);
            s.tts_step(
                &[MASKED_CODE; CODEBOOKS],
                0,
                &cond_uncond,
                1.0,
                (vec![0.5; 5120], vec![0.0; 2560]),
            )?;
        }
        for (t, &p) in models.prompt.iter().enumerate() {
            s.llm_step(Some(p), &[0f32; 1024])?;
            if t >= 1 {
                let draws = s.draws.step();
                s.code = s.tts_step(&s.code.clone(), PAD, &[0f32; 2 * HIDDEN], 0.0, draws)?;
            }
        }
        Ok(s)
    }

    /// Every frame so far.
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    fn tts_step(
        &mut self,
        code: &[i64],
        subword: i64,
        over: &[f32],
        mask: f32,
        draws: (Vec<f32>, Vec<f32>),
    ) -> Result<Vec<i64>> {
        let out = self.tts.run([
            Tensor::from_slice(&[1, 1, CODEBOOKS], code)?,
            Tensor::from_slice(&[1, 1], &[subword])?,
            Tensor::from_slice(&[2, 1, HIDDEN], over)?,
            Tensor::from_slice(&[1, 1, 1], &[mask])?,
            Tensor::from_slice(&[1, 5, 1024], &draws.0)?,
            Tensor::from_slice(&[1, 5, 512], &draws.1)?,
        ])?;
        Ok(out[0].as_slice::<i64>()?.to_vec())
    }

    /// One language model step on a system prompt token or an encoder frame.
    fn llm_step(&mut self, prompt: Option<i64>, audio: &[f32]) -> Result<(i64, i64)> {
        let mut inputs = vec![
            Tensor::from_slice(&[1, 1], &[self.prev_text])?,
            Tensor::from_slice(&[1, 1], &[self.prev_func])?,
            Tensor::from_slice(&[1, 1], &[prompt.unwrap_or(PAD)])?,
            Tensor::from_slice(&[1, 1, 1], &[if prompt.is_some() { 1f32 } else { 0f32 }])?,
            Tensor::from_slice(&[1, 1, 1024], audio)?,
        ];
        inputs.append(&mut self.llm_states);
        let mut out = self.llm.run(inputs)?;
        let tokens = match prompt {
            Some(_) => (PAD, PAD),
            None => (out[0].as_slice::<i64>()?[0], out[1].as_slice::<i64>()?[0]),
        };
        self.llm_states = out.split_off(2);
        Ok(tokens)
    }

    /// Feeds 16 kHz samples. Returns the agent's voice (22.05 kHz) and the frames completed.
    pub fn push(&mut self, pcm: &[f32]) -> Result<(Vec<f32>, &[Frame])> {
        let before = self.frames.len();
        for &s in pcm {
            self.pending.push(match self.last_sample {
                None => s,
                Some(prev) => s - PREEMPH * prev,
            });
            self.last_sample = Some(s);
        }
        let mut voice = vec![];
        if !self.started {
            // The STFT is centered: the stream starts with the reflection of its first samples.
            if self.pending.len() <= HALF_FFT {
                return Ok((voice, &[]));
            }
            let mut padded: Vec<f32> = (1..=HALF_FFT).rev().map(|i| self.pending[i]).collect();
            padded.append(&mut self.pending);
            self.pending = padded;
            self.started = true;
        }
        while self.pending.len() >= CHUNK {
            let chunk: Vec<f32> = self.pending.drain(..CHUNK).collect();
            self.chunk(&chunk, &mut voice)?;
        }
        Ok((voice, &self.frames[before..]))
    }

    fn chunk(&mut self, chunk: &[f32], voice: &mut Vec<f32>) -> Result<()> {
        let m = self.models;
        let mel = self.mel.run([Tensor::from_slice(&[1, CHUNK], chunk)?])?.remove(0);
        let mel = mel.as_slice::<f32>()?;
        for f in 0..m.mel_pulse {
            if self.mel_seen >= m.mel_delay {
                self.mel_frames.push((0..128).map(|c| mel[c * m.mel_pulse + f]).collect());
            }
            self.mel_seen += 1;
        }
        while self.mel_frames.len() >= 8 {
            let block: Vec<f32> = (0..128)
                .flat_map(|c| (0..8).map(move |f| (c, f)))
                .map(|(c, f)| self.mel_frames[f][c])
                .collect();
            self.mel_frames.drain(..8);
            let start = Instant::now();
            let encoded = self.encoder.run([Tensor::from_slice(&[1, 128, 8], &block)?])?.remove(0);
            self.encoder_seen += 1;
            if self.encoder_seen > m.encoder_delay {
                let encoded = encoded.convert_to(DatumType::F32)?;
                self.frame(&encoded, start, voice)?;
            }
        }
        Ok(())
    }

    fn frame(&mut self, encoded: &Tensor, start: Instant, voice: &mut Vec<f32>) -> Result<()> {
        let m = self.models;
        let (token, func) = self.llm_step(None, encoded.as_slice::<f32>()?)?;
        let blank_count = self.turns.as_ref().map_or(0, |t| t.blank_count);
        let (is_blank, emitted) = self.rnnt.step(m, encoded, blank_count)?;
        let (agent, rule) = match &mut self.turns {
            Some(turns) => turns.apply(is_blank, token),
            None => (token, None),
        };
        (self.prev_text, self.prev_func) = (agent, func);
        if agent == EOS {
            self.code = m.silence.clone();
        }
        let draws = self.draws.step();
        self.code = self.tts_step(&self.code.clone(), agent, &[0f32; 2 * HIDDEN], 0.0, draws)?;
        let wav = self.codec.run([Tensor::from_slice(&[1, 1, CODEBOOKS], &self.code)?])?.remove(0);
        let wav = wav.convert_to(DatumType::F32)?;
        let wav = wav.as_slice::<f32>()?;
        let skip = (m.codec_delay + OLA_CROP).saturating_sub(self.voice_out).min(wav.len());
        voice.extend_from_slice(&wav[skip..]);
        self.voice_out += wav.len();
        self.frames.push(Frame {
            user_speech: !is_blank,
            user: emitted.iter().map(|&u| m.rnnt_vocab[u].replace('\u{2581}', " ")).collect(),
            agent,
            model_said: (agent != token).then_some(token),
            rule,
            millis: start.elapsed().as_secs_f64() * 1000.0,
        });
        Ok(())
    }
}

/// Median, p99 and max frame compute time.
pub fn frame_stats(frames: &[Frame]) -> String {
    if frames.is_empty() {
        return "no frames".into();
    }
    let mut ms: Vec<f64> = frames.iter().map(|f| f.millis).collect();
    ms.sort_by(f64::total_cmp);
    format!(
        "{} frames of 80 ms: median {:.1} ms, p99 {:.1} ms, max {:.1} ms",
        ms.len(),
        ms[ms.len() / 2],
        ms[ms.len() * 99 / 100],
        ms[ms.len() - 1]
    )
}

/// Linear resampling of mono audio.
pub fn resample(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return pcm.to_vec();
    }
    let n = (pcm.len() as u64 * to as u64 / from as u64) as usize;
    (0..n)
        .map(|i| {
            let x = i as f64 * from as f64 / to as f64;
            let (j, f) = (x as usize, (x - x.floor()) as f32);
            match (pcm.get(j), pcm.get(j + 1)) {
                (Some(a), Some(b)) => a * (1.0 - f) + b * f,
                (Some(a), None) => *a,
                _ => 0.0,
            }
        })
        .collect()
}
