//! Streaming ASR demo for the multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b`.
//!
//! Pulsified preprocessor + encoder with RNNT greedy decoding, fed a WAV file in
//! fixed audio chunks (deterministic; no real-time pacing). The encoder is the
//! prompt-fused variant (`--fuse-prompt-into-encoder`), so it carries a `lang_id`
//! input selecting the language. The bundled clip is English, so the default is
//! en-US (0); pass a second arg for another language, or 101 for "auto"
//! language-ID detection. Language tags (`<xx-YY>`) the model emits are kept in
//! the transcript verbatim.
//!
//! The encoder pulse must equal the model's attention chunk (32 mel frames = 4
//! encoder frames = 320 ms); other pulses fail chunk-alignment during
//! pulsification.
//!
//! A fourth argument asks for several concurrent sessions. The encoder then
//! keeps its batch axis symbolic and is served by an autobatch runtime, which
//! batches whatever turns arrive together; every session is checked against the
//! same audio transcribed a session at a time, through the same graph. Nothing
//! makes a box queue two sessions in one turn, so `TRACT_TURN_LINGER_US` is
//! what widens them; that the turns really were wide is asserted by the CLI
//! case in `harness/nemotron-3.5-asr-streaming-0.6b`, not here.

use std::fs::File;
use std::path::PathBuf;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use ndarray::prelude::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

const PREPROC_PULSE: usize = 1600; // audio samples per preprocessor pulse (~100 ms)
const ENCODER_PULSE: usize = 32; // mel frames per encoder pulse (one attention chunk)
const DEC_STATE_HIDDEN: usize = 640;
const MAX_SYMBOLS_PER_FRAME: usize = 10;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.iter().position_max_by_key(|x| FloatOrd(**x))
}

fn fact_shape(f: &Fact) -> anyhow::Result<Vec<usize>> {
    (0..f.rank()?).map(|a| f.dim(a).and_then(|d| d.to_int64()).map(|v| v as usize)).collect()
}

/// The shape one session feeds. A symbolic dim is the encoder's batch axis, and
/// a session feeds one row of it per turn whatever the turn ends up carrying.
fn fact_shape_one_row(f: &Fact) -> anyhow::Result<Vec<usize>> {
    (0..f.rank()?).map(|a| Ok(f.dim(a)?.to_int64().map(|v| v as usize).unwrap_or(1))).collect()
}

/// How the encoder's batch axis is set up, and what serves it.
#[derive(Clone, Copy)]
enum EncoderForm {
    /// Pinned to one row: the single-session demo.
    Pinned,
    /// Symbolic, served a session at a time -- the reference an autobatched run
    /// has to match, on the same graph.
    Solo,
    /// Symbolic, served by an autobatch runtime holding that many sessions.
    Autobatch(usize),
}

struct NemotronModels {
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    joint: Runnable,
    vocab: Vec<String>,
    blank_id: usize,
    lang_id: i64,
    pp_delay: usize,
    pp_out_axis: usize,
    pp_out_pulse: usize,
    pp_input_shape: Vec<usize>,
    enc_delay: usize,
    enc_output_axis: usize,
    enc_output_pulse: usize,
    enc_input_shape: Vec<usize>,
}

impl NemotronModels {
    fn load(assets: &str, lang_id: i64, form: EncoderForm) -> anyhow::Result<Self> {
        let model_config: serde_json::Value =
            serde_json::from_reader(File::open(format!("{assets}/model/model_config.json"))?)?;
        // t2n>=0.24 config passthrough: vocabulary is `labels`, RNNT blank is the
        // index just past it.
        let vocab: Vec<String> = model_config["labels"]
            .as_array()
            .context("model_config.json has no `labels`")?
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let blank_id = vocab.len();

        let nnef = tract::nnef()?.with_tract_transformers()?;
        let runtime = tract::runtime_for_name("gpu-or-cpu")?;
        eprintln!("runtime: {}", runtime.name()?);

        // Preprocessor: pulsify over the raw-audio time axis. Drop the `length`
        // input (recomputed from the pulsed signal shape) so streaming feeds a
        // single tensor per pulse.
        let mut pp = nnef.load(format!("{assets}/model/preprocessor.nnef.tgz"))?;
        pp.transform(SetSymbols::new().value("BATCH", 1))?;
        pp.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(input_signal)[1];"}"#,
        )?;
        pp.transform(r#"{"name":"select_inputs","inputs":["input_signal"]}"#)?;
        pp.transform(r#"{"name":"select_outputs","outputs":["processed_signal"]}"#)?;
        pp.transform(Pulse::new(PREPROC_PULSE.to_string()).symbol("INPUT_SIGNAL__TIME"))?;
        let pp_delay = pp.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
        let pp_out_axis = pp.property("pulse.output_axes")?.as_slice::<i64>()?[0] as usize;
        let pp_out_pulse = pp.output_fact(0)?.dim(pp_out_axis)?.to_int64()? as usize;
        let pp_input_shape = fact_shape(&pp.input_fact(0)?)?;
        let preprocessor = runtime.prepare(pp)?;

        // Encoder: prompt-fused, so it keeps `audio_signal` + `lang_id`. Pulsify
        // over the mel time axis; `lang_id` (no time axis) rides along as a
        // constant per-pulse input. Drop the `length` output-side input.
        //
        // The batch axis is what a session's turns are seated on, so the two
        // batched forms keep BATCH symbolic. `length` as a scalar then reshapes
        // to [BATCH] and only typechecks at one row, hence the shape-generic
        // body; and `batchify_data_free` gives the batch axis to the mask wires
        // no input feeds, which pulsification would otherwise turn into buffers
        // every session shares.
        let mut enc = nnef.load(format!("{assets}/model/encoder.nnef.tgz"))?;
        if matches!(form, EncoderForm::Pinned) {
            enc.transform(SetSymbols::new().value("BATCH", 1))?;
        }
        enc.transform("transformers_detect_all")?;
        if matches!(form, EncoderForm::Pinned) {
            enc.transform(
                r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#,
            )?;
        } else {
            enc.transform(
                r#"{"name":"patch","body":"length = tract_core_cast(squeeze(sum_reduce(audio_signal, axes=[1,2]), axes=[1,2]) * 0.0, to = \"i64\") + tract_core_cast(tract_core_shape_of(audio_signal)[2], to = \"i64\");"}"#,
            )?;
        }
        enc.transform(r#"{"name":"select_inputs","inputs":["audio_signal","lang_id"]}"#)?;
        enc.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
        if !matches!(form, EncoderForm::Pinned) {
            enc.transform(r#"{"name":"batchify_data_free","symbol":"BATCH"}"#)?;
        }
        enc.transform(Pulse::new(ENCODER_PULSE.to_string()).symbol("AUDIO_SIGNAL__TIME"))?;
        let enc_delay = enc.property("pulse.delay")?.as_slice::<i64>()?[0] as usize;
        let enc_output_axis = enc.property("pulse.output_axes")?.as_slice::<i64>()?[0] as usize;
        let enc_output_pulse = enc.output_fact(0)?.dim(enc_output_axis)?.to_int64()? as usize;
        let enc_input_shape = fact_shape_one_row(&enc.input_fact(0)?)?;
        let encoder = match form {
            EncoderForm::Autobatch(sessions) => runtime.prepare(enc)?.autobatch(sessions)?,
            _ => runtime.prepare(enc)?,
        };

        // Decoder (prednet) and joint stay non-streaming, one step at a time.
        let mut dec = nnef.load(format!("{assets}/model/decoder.nnef.tgz"))?;
        dec.transform(SetSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;
        let decoder = runtime.prepare(dec)?;

        let mut jnt = nnef.load(format!("{assets}/model/joint.nnef.tgz"))?;
        jnt.transform(
            SetSymbols::new()
                .value("BATCH", 1)
                .value("ENCODER_OUTPUTS__TIME", 1)
                .value("DECODER_OUTPUTS__TIME", 1),
        )?;
        let joint = runtime.prepare(jnt)?;

        eprintln!(
            "pulses: preproc={PREPROC_PULSE} (delay {pp_delay}, out {pp_out_pulse}), \
             encoder={ENCODER_PULSE} (delay {enc_delay}, out {enc_output_pulse}), lang_id={lang_id}"
        );

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            joint,
            vocab,
            blank_id,
            lang_id,
            pp_delay,
            pp_out_axis,
            pp_out_pulse,
            pp_input_shape,
            enc_delay,
            enc_output_axis,
            enc_output_pulse,
            enc_input_shape,
        })
    }

    /// One decoder (prednet) step: feed the last token + carried state, get the
    /// prednet embedding and the next state. Hides the `target_length` input and
    /// the interleaved `prednet_lengths` output of the split RNNT decoder.
    fn decoder_step(
        &self,
        token: Tensor,
        state_0: Tensor,
        state_1: Tensor,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let target_length = Tensor::from_slice(&[1], &[1i32])?;
        let [emb, _prednet_lengths, s0, s1] =
            self.decoder.run([token, target_length, state_0, state_1])?.try_into().unwrap();
        Ok((emb, s0, s1))
    }
}

struct StreamState<'a> {
    models: &'a NemotronModels,
    preproc: State,
    encoder: State,
    lang: Tensor,
    dec_emb: Tensor,
    dec_state_0: Tensor,
    dec_state_1: Tensor,
    audio_buf: Vec<f32>,
    audio_consumed: usize,
    feat_buf: Vec<ArrayD<f32>>,
    feat_buf_frames: usize,
    pp_delay_remaining: usize,
    enc_delay_remaining: usize,
    hyp: Vec<usize>,
}

impl<'a> StreamState<'a> {
    fn new(models: &'a NemotronModels) -> anyhow::Result<Self> {
        let preproc = models.preprocessor.spawn_state()?;
        let encoder = models.encoder.spawn_state()?;
        let lang = Tensor::from_slice(&[1], &[models.lang_id])?;

        // Warm-up mirrors NeMo predict(add_sos=True, y=None): two zero-input steps
        // through the prednet (blank has a zero embedding, padding_idx=blank).
        let blank = Tensor::from_slice(&[1, 1], &[models.blank_id as i32])?;
        let s0 = Array3::<f32>::zeros([2, 1, DEC_STATE_HIDDEN]).tract()?;
        let s1 = Array3::<f32>::zeros([2, 1, DEC_STATE_HIDDEN]).tract()?;
        let (_e, s0, s1) = models.decoder_step(blank.clone(), s0, s1)?;
        let (dec_emb, dec_state_0, dec_state_1) = models.decoder_step(blank, s0, s1)?;

        Ok(Self {
            pp_delay_remaining: models.pp_delay,
            enc_delay_remaining: models.enc_delay,
            models,
            preproc,
            encoder,
            lang,
            dec_emb,
            dec_state_0,
            dec_state_1,
            audio_buf: Vec::new(),
            audio_consumed: 0,
            feat_buf: Vec::new(),
            feat_buf_frames: 0,
            hyp: Vec::new(),
        })
    }

    fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        self.audio_buf.extend_from_slice(samples);
        while self.audio_consumed + PREPROC_PULSE <= self.audio_buf.len() {
            let start = self.audio_consumed;
            let end = start + PREPROC_PULSE;
            let pp_input =
                Tensor::from_slice(&self.models.pp_input_shape, &self.audio_buf[start..end])?;
            self.audio_consumed = end;
            self.run_preproc(pp_input)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        let remaining = self.audio_buf.len() - self.audio_consumed;
        if remaining > 0 {
            let mut data = vec![0.0f32; self.models.pp_input_shape.iter().product()];
            data[..remaining].copy_from_slice(&self.audio_buf[self.audio_consumed..]);
            let pp_input = Tensor::from_slice(&self.models.pp_input_shape, &data)?;
            self.run_preproc(pp_input)?;
        }
        if self.feat_buf_frames > 0 {
            let refs: Vec<_> = self.feat_buf.iter().map(|a| a.view()).collect();
            let leftover = ndarray::concatenate(Axis(self.models.pp_out_axis), &refs)?;
            let s = &self.models.enc_input_shape;
            let mut enc_input = Array3::<f32>::zeros((s[0], s[1], s[2]));
            let n = leftover.shape()[self.models.pp_out_axis].min(s[2]);
            enc_input.slice_mut(s![.., .., ..n]).assign(&leftover.slice(s![.., .., ..n]));
            self.run_encoder_pulse(enc_input.into_dyn())?;
        }
        Ok(())
    }

    fn run_preproc(&mut self, input: Tensor) -> anyhow::Result<()> {
        let results = self.preproc.run([input])?;
        let features: ArrayD<f32> = results[0].ndarray()?.into_owned();
        self.feed_features(features)
    }

    fn feed_features(&mut self, features: ArrayD<f32>) -> anyhow::Result<()> {
        let pp_out_pulse = self.models.pp_out_pulse;
        let pp_out_axis = self.models.pp_out_axis;
        let usable_start = self.pp_delay_remaining.min(pp_out_pulse);
        self.pp_delay_remaining = self.pp_delay_remaining.saturating_sub(pp_out_pulse);
        if usable_start >= pp_out_pulse {
            return Ok(());
        }
        let usable = features.slice_axis(Axis(pp_out_axis), (usable_start..pp_out_pulse).into());
        self.feat_buf_frames += usable.shape()[pp_out_axis];
        self.feat_buf.push(usable.to_owned());

        while self.feat_buf_frames >= ENCODER_PULSE {
            let refs: Vec<_> = self.feat_buf.iter().map(|a| a.view()).collect();
            let all = ndarray::concatenate(Axis(pp_out_axis), &refs)?;
            let enc_feat = all.slice_axis(Axis(pp_out_axis), (..ENCODER_PULSE).into()).to_owned();
            self.run_encoder_pulse(enc_feat)?;
            let leftover = all.slice_axis(Axis(pp_out_axis), (ENCODER_PULSE..).into());
            self.feat_buf_frames -= ENCODER_PULSE;
            self.feat_buf.clear();
            if self.feat_buf_frames > 0 {
                self.feat_buf.push(leftover.to_owned());
            }
        }
        Ok(())
    }

    fn run_encoder_pulse(&mut self, features: ArrayD<f32>) -> anyhow::Result<()> {
        let pulse_tensor: Tensor = features.tract()?;
        let results = self.encoder.run([pulse_tensor, self.lang.clone()])?;
        let enc_out: ArrayD<f32> = results[0].ndarray()?.into_owned();
        for f in 0..self.models.enc_output_pulse {
            if self.enc_delay_remaining > 0 {
                self.enc_delay_remaining -= 1;
                continue;
            }
            let frame: Tensor =
                enc_out.slice_axis(Axis(self.models.enc_output_axis), (f..f + 1).into()).tract()?;
            self.decode_frame(frame)?;
        }
        Ok(())
    }

    fn decode_frame(&mut self, frame: Tensor) -> anyhow::Result<()> {
        let mut tokens_this_frame = 0usize;
        loop {
            let [logits] =
                self.models.joint.run([frame.clone(), self.dec_emb.clone()])?.try_into().unwrap();
            let token_id = argmax(logits.as_slice::<f32>()?).unwrap();
            if token_id == self.models.blank_id {
                break;
            }
            self.hyp.push(token_id);
            tokens_this_frame += 1;
            let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
            let (emb, s0, s1) = self.models.decoder_step(
                tok,
                self.dec_state_0.clone(),
                self.dec_state_1.clone(),
            )?;
            self.dec_emb = emb;
            self.dec_state_0 = s0;
            self.dec_state_1 = s1;
            if tokens_this_frame >= MAX_SYMBOLS_PER_FRAME {
                break;
            }
        }
        Ok(())
    }
}

/// Feed one session the whole clip in small chunks, deterministically, and
/// return the tokens it decoded.
fn transcribe(models: &NemotronModels, audio: &[f32]) -> anyhow::Result<Vec<usize>> {
    let mut state = StreamState::new(models)?;
    for chunk in audio.chunks(80) {
        state.push_audio(chunk)?;
    }
    state.flush()?;
    Ok(state.hyp)
}

fn text(models: &NemotronModels, hyp: &[usize]) -> String {
    hyp.iter()
        .map(|&t| models.vocab[t].as_str())
        .join("")
        .replace('\u{2581}', " ")
        .trim()
        .to_string()
}

/// The transcript of the bundled English clip at lang_id=0 (en-US). The
/// `<en-US>` language tag is part of the model's output and is kept verbatim.
const EXPECTED: &str = "well I don't wish to see it any more, observed Phoebe, turning away her eyes. <en-US> It is certainly very like the old portrait";

/// Several sessions at once must transcribe exactly as each of them does alone,
/// and again on a lane a departed session left.
///
/// Session `k` arrives `k` pulses late, so no two of them sit at the same
/// position and a buffer two of them share cannot stay invisible. The same
/// staggered audio is transcribed a session at a time on the same graph, the
/// autobatch runtime being all that differs, and that solo run is what the
/// batched transcripts are held against -- leading silence can move the greedy
/// path, so only session 0 transcribes the clip as the clip.
fn check_sessions_batch_as_they_run_alone(
    assets: &str,
    lang_id: i64,
    wav: &[f32],
    streams: usize,
) -> anyhow::Result<()> {
    let arrivals: Vec<Vec<f32>> = (0..streams)
        .map(|k| {
            let mut audio = vec![0.0f32; k * PREPROC_PULSE];
            audio.extend_from_slice(wav);
            audio
        })
        .collect();

    // One at a time: the solo encoder and the autobatched one never hold their
    // weights together.
    let alone: Vec<Vec<usize>> = {
        let models = NemotronModels::load(assets, lang_id, EncoderForm::Solo)?;
        arrivals.iter().map(|audio| transcribe(&models, audio)).collect::<anyhow::Result<_>>()?
    };

    let models = NemotronModels::load(assets, lang_id, EncoderForm::Autobatch(streams))?;
    let batched = |arrivals: &[Vec<f32>]| -> anyhow::Result<Vec<Vec<usize>>> {
        std::thread::scope(|scope| {
            let running: Vec<_> =
                arrivals.iter().map(|audio| scope.spawn(|| transcribe(&models, audio))).collect();
            running.into_iter().map(|session| session.join().unwrap()).collect::<Vec<_>>()
        })
        .into_iter()
        .collect()
    };

    let served = batched(&arrivals)?;
    for (k, (hyp, alone)) in served.iter().zip(&alone).enumerate() {
        let hyp = text(&models, hyp);
        ensure!(
            hyp == text(&models, alone),
            "session {k} batched transcribes `{hyp}', alone `{}'",
            text(&models, alone)
        );
        println!("session {k}: {hyp}");
    }
    ensure!(
        lang_id != 0 || text(&models, &served[0]) == EXPECTED,
        "session 0 feeds the clip from its start, so it transcribes it"
    );

    // The same sessions again, so every lane is one a departed session left:
    // what a lane carries of its last session has to be gone, and the whole
    // encoder is 11.4 MiB of it.
    ensure!(
        batched(&arrivals)? == served,
        "a session admitted to a lane a previous one left transcribes differently"
    );

    println!("{streams} sessions batch as they run alone, and again on the lanes they left.");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let assets = std::env::args().nth(1).unwrap_or_else(|| "assets".to_string());
    let lang_id: i64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let wav_path: PathBuf =
        std::env::args().nth(3).unwrap_or_else(|| format!("{assets}/2086-149220-0033.wav")).into();
    let streams: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    let wav: Vec<f32> = hound::WavReader::open(&wav_path)?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();

    if streams > 1 {
        return check_sessions_batch_as_they_run_alone(&assets, lang_id, &wav, streams);
    }

    let models = NemotronModels::load(&assets, lang_id, EncoderForm::Pinned)?;
    let transcript = text(&models, &transcribe(&models, &wav)?);
    println!("Transcript: {transcript}");

    if lang_id == 0 && wav_path.ends_with("2086-149220-0033.wav") {
        assert_eq!(transcript, EXPECTED);
    }
    Ok(())
}
