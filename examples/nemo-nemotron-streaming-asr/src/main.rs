use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::*;
use clap::Parser;
#[cfg(feature = "live")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use float_ord::FloatOrd;
use itertools::Itertools;
use tract::prelude::tract_ndarray::prelude::*;
use tract::prelude::*;

/// Streaming ASR demo using nvidia/nemotron-speech-streaming-en-0.6b.
///
/// Transcribes audio incrementally using pulsified preprocessor + encoder
/// on GPU with RNNT greedy decoding.
///
/// By default, reads from a WAV file with simulated real-time playback.
/// Use --live to capture from the system microphone.
#[derive(Parser)]
struct Config {
    /// Path to the model assets directory.
    #[arg(long, default_value = "assets")]
    assets: PathBuf,

    /// WAV file to transcribe (16kHz mono PCM). Ignored if --live is set.
    #[arg(long, default_value = "assets/2086-149220-0033.wav")]
    wav: PathBuf,

    /// Capture from the system microphone instead of a WAV file.
    /// Requires the `live` feature: cargo run --features live -- --live
    #[arg(long)]
    live: bool,

    /// Preprocessor pulse in audio samples. ~100ms = 1600 samples.
    #[arg(long, default_value_t = 1600)]
    preproc_pulse: usize,

    /// Encoder pulse in feature frames. 14 token chunks * 8x subsampling = 112.
    #[arg(long, default_value_t = 112)]
    encoder_pulse: usize,

    /// Do not simulate real-time playback (WAV mode only: process as fast as possible).
    #[arg(long)]
    no_realtime: bool,
}

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

fn fact_shape(f: &Fact) -> anyhow::Result<Vec<usize>> {
    (0..f.rank()?).map(|a| f.dim(a).and_then(|d| d.to_int64()).map(|v| v as usize)).collect()
}

// ─── Shared read-only model context ─────────────────────────────────────────

struct NemotronModels {
    config: Config,
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    joint: Runnable,
    vocab: Vec<String>,
    blank_id: usize,
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
    fn load(config: Config) -> anyhow::Result<(Arc<Self>, Duration)> {
        let t0 = Instant::now();
        let assets = config.assets.display();

        let model_config: serde_json::Value =
            serde_json::from_reader(File::open(format!("{assets}/model/model_config.json"))?)?;
        let blank_id =
            model_config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
        let vocab: Vec<String> = model_config
            .pointer("/joint/vocabulary")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();

        let nnef = tract::nnef()?.with_tract_core()?.with_tract_transformers()?;
        let runtime = ["cuda", "metal", "default"]
            .iter()
            .find_map(|rt| tract::runtime_for_name(rt).ok())
            .unwrap();

        eprint!("Loading preprocessor to {}...", runtime.name()?);
        let mut pp = nnef.load(format!("{assets}/model/preprocessor.nnef.tgz"))?;
        pp.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
        pp.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(input_signal)[1];"}"#,
        )?;
        pp.transform(r#"{"name":"select_outputs","outputs":["processed_signal"]}"#)?;
        pp.transform(Pulse::new(config.preproc_pulse.to_string()).symbol("INPUT_SIGNAL__TIME"))?;
        let pp_delay = pp.property("pulse.delay")?.view::<i64>()?[0].to_owned() as usize;
        let pp_out_axis = pp.property("pulse.output_axes")?.view::<i64>()?[0].to_owned() as usize;
        let pp_out_pulse = pp.output_fact(0)?.dim(pp_out_axis)?.to_int64()? as usize;
        let pp_input_shape = fact_shape(&pp.input_fact(0)?)?;
        let preprocessor = runtime.prepare(pp)?;
        eprintln!(" done.");

        eprint!("Loading encoder to {}...", runtime.name()?);
        let mut enc = nnef.load(format!("{assets}/model/encoder.p1.nnef.tgz"))?;
        enc.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
        enc.transform("transformers_detect_all")?;
        enc.transform(
            r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#,
        )?;
        enc.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
        enc.transform(Pulse::new(config.encoder_pulse.to_string()).symbol("AUDIO_SIGNAL__TIME"))?;
        let enc_delay = enc.property("pulse.delay")?.view::<i64>()?[0].to_owned() as usize;
        let enc_output_axis =
            enc.property("pulse.output_axes")?.view::<i64>()?[0].to_owned() as usize;
        let enc_output_pulse = enc.output_fact(0)?.dim(enc_output_axis)?.to_int64()? as usize;
        let enc_input_shape = fact_shape(&enc.input_fact(0)?)?;
        let encoder = runtime.prepare(enc)?;
        eprintln!(" done.");

        eprint!("Loading decoder to {}...", runtime.name()?);
        let mut dec = nnef.load(format!("{assets}/model/decoder.nnef.tgz"))?;
        dec.transform(ConcretizeSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;
        let decoder = runtime.prepare(dec)?;
        eprintln!(" done.");

        eprint!("Loading joint to {}...", runtime.name()?);
        let mut jnt = nnef.load(format!("{assets}/model/joint.nnef.tgz"))?;
        jnt.transform(
            ConcretizeSymbols::new()
                .value("BATCH", 1)
                .value("ENCODER_OUTPUTS__TIME", 1)
                .value("DECODER_OUTPUTS__TIME", 1),
        )?;
        let joint = runtime.prepare(jnt)?;
        eprintln!(" done.");

        let load_time = t0.elapsed();
        eprintln!("Ready ({:.1}s)", load_time.as_secs_f64());

        Ok((
            Arc::new(Self {
                config,
                preprocessor,
                encoder,
                decoder,
                joint,
                vocab,
                blank_id,
                pp_delay,
                pp_out_axis,
                pp_out_pulse,
                pp_input_shape,
                enc_delay,
                enc_output_axis,
                enc_output_pulse,
                enc_input_shape,
            }),
            load_time,
        ))
    }

    fn spawn(self: &Arc<Self>) -> anyhow::Result<StreamState> {
        StreamState::new(Arc::clone(self))
    }
}

// ─── Mutable streaming state ────────────────────────────────────────────────

struct StreamState {
    models: Arc<NemotronModels>,
    preproc: State,
    encoder: State,
    dec_token: Tensor,
    dec_state_0: Tensor,
    dec_state_1: Tensor,
    audio_buf: Vec<f32>,
    audio_consumed: usize,
    feat_buf: Vec<ArrayD<f32>>,
    feat_buf_frames: usize,
    pp_delay_remaining: usize,
    enc_delay_remaining: usize,
    hyp: Vec<usize>,
    pulse_count: usize,
    total_preproc: Duration,
    total_encoder: Duration,
    total_joint: Duration,
    total_decoder: Duration,
    n_preproc: usize,
    n_encoder: usize,
    n_joint: usize,
    n_decoder: usize,
}

impl StreamState {
    fn new(models: Arc<NemotronModels>) -> anyhow::Result<Self> {
        let preproc = models.preprocessor.spawn_state()?;
        let encoder = models.encoder.spawn_state()?;

        let blank_tok = Tensor::from_slice(&[1, 1], &[models.blank_id as i32])?;
        let s0 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
        let s1 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
        let [_out, s0, s1] = models.decoder.run([blank_tok.clone(), s0, s1])?.try_into().unwrap();
        let [dec_token, dec_state_0, dec_state_1] =
            models.decoder.run([blank_tok, s0, s1])?.try_into().unwrap();

        Ok(Self {
            pp_delay_remaining: models.pp_delay,
            enc_delay_remaining: models.enc_delay,
            models,
            preproc,
            encoder,
            dec_token,
            dec_state_0,
            dec_state_1,
            audio_buf: Vec::new(),
            audio_consumed: 0,
            feat_buf: Vec::new(),
            feat_buf_frames: 0,
            hyp: Vec::new(),
            pulse_count: 0,
            total_preproc: Duration::ZERO,
            total_encoder: Duration::ZERO,
            total_joint: Duration::ZERO,
            total_decoder: Duration::ZERO,
            n_preproc: 0,
            n_encoder: 0,
            n_joint: 0,
            n_decoder: 0,
        })
    }

    fn show(&self, label: &str) {
        let vocab: Vec<&str> = self.models.vocab.iter().map(|s| s.as_str()).collect();
        let text: String = self.hyp.iter().map(|&t| vocab[t]).join("");
        let display = text.replace('▁', " ");
        eprint!("\r{} {label}          ", display.trim_start());
    }

    fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        self.audio_buf.extend_from_slice(samples);
        self.show("");
        let preproc_pulse = self.models.config.preproc_pulse;
        while self.audio_consumed + preproc_pulse <= self.audio_buf.len() {
            let start = self.audio_consumed;
            let end = start + preproc_pulse;
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
            let leftover =
                tract::prelude::tract_ndarray::concatenate(Axis(self.models.pp_out_axis), &refs)?;
            let s = &self.models.enc_input_shape;
            let mut enc_input = Array3::<f32>::zeros((s[0], s[1], s[2]));
            let n = leftover.shape()[self.models.pp_out_axis].min(s[2]);
            enc_input.slice_mut(s![.., .., ..n]).assign(&leftover.slice(s![.., .., ..n]));
            self.run_encoder_pulse(enc_input.into_dyn())?;
        }
        Ok(())
    }

    fn transcript(&self) -> String {
        let vocab: Vec<&str> = self.models.vocab.iter().map(|s| s.as_str()).collect();
        self.hyp.iter().map(|&t| vocab[t]).join("")
    }

    fn run_preproc(&mut self, input: Tensor) -> anyhow::Result<()> {
        self.show("[pre]");
        let t = Instant::now();
        let results = self.preproc.run([input])?;
        self.total_preproc += t.elapsed();
        self.n_preproc += 1;
        let features: ArrayD<f32> = results[0].view()?.into_owned();
        self.feed_features(features)
    }

    fn feed_features(&mut self, features: ArrayD<f32>) -> anyhow::Result<()> {
        let pp_out_pulse = self.models.pp_out_pulse;
        let pp_out_axis = self.models.pp_out_axis;
        let encoder_pulse = self.models.config.encoder_pulse;
        let usable_start = self.pp_delay_remaining.min(pp_out_pulse);
        self.pp_delay_remaining = self.pp_delay_remaining.saturating_sub(pp_out_pulse);
        if usable_start >= pp_out_pulse {
            return Ok(());
        }
        let usable = features.slice_axis(Axis(pp_out_axis), (usable_start..pp_out_pulse).into());
        self.feat_buf_frames += usable.shape()[pp_out_axis];
        self.feat_buf.push(usable.to_owned());

        while self.feat_buf_frames >= encoder_pulse {
            let refs: Vec<_> = self.feat_buf.iter().map(|a| a.view()).collect();
            let all = tract::prelude::tract_ndarray::concatenate(Axis(pp_out_axis), &refs)?;
            let enc_feat = all.slice_axis(Axis(pp_out_axis), (..encoder_pulse).into()).to_owned();
            self.run_encoder_pulse(enc_feat)?;
            let leftover = all.slice_axis(Axis(pp_out_axis), (encoder_pulse..).into());
            self.feat_buf_frames -= encoder_pulse;
            self.feat_buf.clear();
            if self.feat_buf_frames > 0 {
                self.feat_buf.push(leftover.to_owned());
            }
        }
        Ok(())
    }

    fn run_encoder_pulse(&mut self, features: ArrayD<f32>) -> anyhow::Result<()> {
        self.show("[enc]");
        let t = Instant::now();
        let pulse_tensor: Tensor = tensor(features)?;
        let results = self.encoder.run([pulse_tensor])?;
        self.total_encoder += t.elapsed();
        self.n_encoder += 1;
        let enc_out: ArrayD<f32> = results[0].view()?.into_owned();
        self.pulse_count += 1;
        for f in 0..self.models.enc_output_pulse {
            if self.enc_delay_remaining > 0 {
                self.enc_delay_remaining -= 1;
                continue;
            }
            let frame: Tensor =
                tensor(enc_out.slice_axis(Axis(self.models.enc_output_axis), (f..f + 1).into()))?;
            self.decode_frame(frame)?;
        }
        Ok(())
    }

    fn decode_frame(&mut self, frame: Tensor) -> anyhow::Result<()> {
        let mut tokens_this_frame = 0usize;
        loop {
            self.show("[jnt]");
            let t = Instant::now();
            let [logits] =
                self.models.joint.run([frame.clone(), self.dec_token.clone()])?.try_into().unwrap();
            self.total_joint += t.elapsed();
            self.n_joint += 1;
            let logits_view = logits.view::<f32>()?;
            let token_id = argmax(logits_view.as_slice().unwrap()).unwrap();
            if token_id == self.models.blank_id {
                break;
            }
            self.hyp.push(token_id);
            tokens_this_frame += 1;
            self.show("[dec]");
            let t = Instant::now();
            let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
            [self.dec_token, self.dec_state_0, self.dec_state_1] = self
                .models
                .decoder
                .run([tok, self.dec_state_0.clone(), self.dec_state_1.clone()])?
                .try_into()
                .unwrap();
            self.total_decoder += t.elapsed();
            self.n_decoder += 1;
            if tokens_this_frame >= 10 {
                break;
            }
        }
        Ok(())
    }

    fn print_stats(&self, load_time: Duration, stream_time: Duration, audio_duration: f64) {
        eprintln!();
        eprintln!("--- stats ---");
        eprintln!("model load:    {:.1}s", load_time.as_secs_f64());
        eprintln!("audio:         {:.2}s", audio_duration);
        eprintln!(
            "stream wall:   {:.2}s ({:.1}x real-time)",
            stream_time.as_secs_f64(),
            audio_duration / stream_time.as_secs_f64()
        );
        eprintln!("pulses:        {}", self.pulse_count);
        let stats = [
            ("preprocessor", self.total_preproc, self.n_preproc),
            ("encoder", self.total_encoder, self.n_encoder),
            ("joint", self.total_joint, self.n_joint),
            ("decoder", self.total_decoder, self.n_decoder),
        ];
        for (name, total, n) in &stats {
            if *n > 0 {
                eprintln!(
                    "{name:14} {:.1}ms total, {:.2}ms/call ({n} calls)",
                    total.as_secs_f64() * 1000.0,
                    total.as_secs_f64() * 1000.0 / *n as f64,
                );
            }
        }
        let compute =
            self.total_preproc + self.total_encoder + self.total_joint + self.total_decoder;
        eprintln!(
            "compute total: {:.1}ms ({:.1}x real-time)",
            compute.as_secs_f64() * 1000.0,
            audio_duration / compute.as_secs_f64()
        );
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

/// Start a WAV file audio source: reads samples in chunks with optional real-time pacing.
fn start_wav_source(
    path: &PathBuf,
    realtime: bool,
) -> anyhow::Result<(mpsc::Receiver<Vec<f32>>, std::thread::JoinHandle<()>, f64)> {
    let wav: Vec<f32> = hound::WavReader::open(path)?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let audio_duration = wav.len() as f64 / 16000.0;
    let total_samples = wav.len();
    let audio_chunk = 80; // 5ms chunks

    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let handle = std::thread::Builder::new().name("wav".into()).spawn(move || {
        let mut offset = 0;
        while offset < total_samples {
            let end = (offset + audio_chunk).min(total_samples);
            if tx.send(wav[offset..end].to_vec()).is_err() {
                break;
            }
            offset = end;
            if realtime {
                std::thread::sleep(Duration::from_secs_f64(audio_chunk as f64 / 16000.0));
            }
        }
    })?;
    Ok((rx, handle, audio_duration))
}

/// Start a live microphone audio source via cpal.
/// Tries 16kHz mono first; falls back to the device's default config
/// and resamples to 16kHz mono on the fly.
#[cfg(feature = "live")]
fn start_live_source() -> anyhow::Result<(mpsc::Receiver<Vec<f32>>, cpal::Stream)> {
    let host = cpal::default_host();
    let device = host.default_input_device().context("no input device")?;
    eprintln!("Microphone: {}", device.name()?);

    let target_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(8);

    // Try native 16kHz mono first, fall back to device default with resampling.
    let stream = if let Ok(stream) = device.build_input_stream(
        &target_config,
        {
            let tx = tx.clone();
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let _ = tx.send(data.to_vec());
            }
        },
        |err| eprintln!("audio error: {err}"),
        None,
    ) {
        eprintln!("Audio: 16kHz mono (native)");
        stream
    } else {
        let default_config = device.default_input_config()?;
        let channels = default_config.channels() as usize;
        let sample_rate = default_config.sample_rate().0 as f64;
        eprintln!("Audio: {}Hz {}ch (resampling to 16kHz mono)", sample_rate as u32, channels);
        let ratio = 16000.0 / sample_rate;
        let mut resample_pos = 0.0f64;

        device.build_input_stream(
            &default_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut out = Vec::new();
                while (resample_pos as usize) < data.len() / channels {
                    let idx = resample_pos as usize * channels;
                    // Average channels to mono
                    let sample: f32 =
                        data[idx..idx + channels].iter().sum::<f32>() / channels as f32;
                    out.push(sample);
                    resample_pos += 1.0 / ratio;
                }
                resample_pos -= (data.len() / channels) as f64;
                if !out.is_empty() {
                    let _ = tx.send(out);
                }
            },
            |err| eprintln!("audio error: {err}"),
            None,
        )?
    };
    stream.play()?;
    Ok((rx, stream))
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let live = config.live;
    let wav_path = config.wav.clone();
    let no_realtime = config.no_realtime;

    let (models, load_time) = NemotronModels::load(config)?;
    let mut state = models.spawn()?;

    if live {
        #[cfg(not(feature = "live"))]
        anyhow::bail!("--live requires the `live` feature: cargo run --features live -- --live");

        #[cfg(feature = "live")]
        {
            let (audio_rx, _stream) = start_live_source()?;
            eprintln!("Listening... (press Ctrl-C to stop)\n");

            let stream_start = Instant::now();
            for chunk in audio_rx {
                state.push_audio(&chunk)?;
            }
            state.flush()?;
            let stream_time = stream_start.elapsed();
            let audio_duration = stream_time.as_secs_f64();

            let transcript = state.transcript();
            let display = transcript.replace('▁', " ");
            eprint!("\r{}                    \n", display.trim_start());
            state.print_stats(load_time, stream_time, audio_duration);
        }
    } else {
        // ── WAV file mode ───────────────────────────────────────────
        let (audio_rx, mic_handle, audio_duration) = start_wav_source(&wav_path, !no_realtime)?;

        let stream_start = Instant::now();
        for chunk in audio_rx {
            state.push_audio(&chunk)?;
        }
        state.flush()?;
        let stream_time = stream_start.elapsed();

        mic_handle.join().unwrap();

        let transcript = state.transcript();
        let display = transcript.replace('▁', " ");
        eprint!("\r{}                    \n", display.trim_start());

        state.print_stats(load_time, stream_time, audio_duration);

        let expected = "▁well▁I▁don't▁wish▁to▁see▁it▁any▁more▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait";
        if transcript != expected {
            eprintln!("\nNOTE: streaming transcript differs slightly from batch reference");
        }
    }
    Ok(())
}
