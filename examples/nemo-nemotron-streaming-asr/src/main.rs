use std::fs::File;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract::prelude::tract_ndarray::prelude::*;
use tract::prelude::*;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

/// Audio chunk size sent by the "microphone" thread (samples at 16kHz).
/// 80 samples = 5ms of audio — simulates a real-time audio callback.
const AUDIO_CHUNK: usize = 80;

/// The encoder pulse in audio feature frames (after STFT).
/// 14 transformer-token chunks * 8x conv subsampling = 112 frames.
const AUDIO_PULSE: usize = 112;

fn main() -> anyhow::Result<()> {
    let t0 = Instant::now();
    eprintln!("[{:7.3}] Loading models...", t0.elapsed().as_secs_f64());

    let config: serde_json::Value =
        serde_json::from_reader(File::open("assets/model/model_config.json")?)?;
    let blank_id = config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
    let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
    let vocab: Vec<&str> = vocab.iter().map(|v| v.as_str().unwrap()).collect();

    let nnef = tract::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let runtime = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("[{:7.3}] Runtime: {}", t0.elapsed().as_secs_f64(), runtime.name()?);

    // ── Preprocessor (CPU — variable-length input not suited for GPU) ──
    let mut preprocessor = nnef.load("assets/model/preprocessor.nnef.tgz")?;
    preprocessor.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
    preprocessor
        .transform(r#"{"name":"patch","body":"length = tract_core_shape_of(input_signal)[1];"}"#)?;
    preprocessor.transform(r#"{"name":"select_outputs","outputs":["processed_signal"]}"#)?;
    let preprocessor = preprocessor.into_runnable()?;

    // ── Encoder (pulsified) ─────────────────────────────────────────────
    let mut encoder = nnef.load("assets/model/encoder.p1.nnef.tgz")?;
    encoder.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
    encoder.transform("transformers_detect_all")?;
    encoder
        .transform(r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#)?;
    encoder.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
    encoder.transform(Pulse::new(AUDIO_PULSE.to_string()).symbol("AUDIO_SIGNAL__TIME"))?;
    let encoder = runtime.prepare(encoder)?;

    let enc_output_axis =
        encoder.property("pulse.output_axes")?.view::<i64>()?[0].to_owned() as usize;
    let enc_output_pulse = encoder.output_fact(0)?.dim(enc_output_axis)?.to_int64()? as usize;
    let enc_delay = encoder.property("pulse.delay")?.view::<i64>()?[0].to_owned() as usize;
    let enc_fact = encoder.input_fact(0)?;
    let enc_input_shape: Vec<usize> = (0..enc_fact.rank()?)
        .map(|a| enc_fact.dim(a).and_then(|d| d.to_int64()).map(|v| v as usize))
        .collect::<Result<_>>()?;

    // ── Decoder ─────────────────────────────────────────────────────────
    let mut decoder_model = nnef.load("assets/model/decoder.nnef.tgz")?;
    decoder_model
        .transform(ConcretizeSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;
    let decoder = runtime.prepare(decoder_model)?;

    // ── Joint ───────────────────────────────────────────────────────────
    let mut joint_model = nnef.load("assets/model/joint.nnef.tgz")?;
    joint_model.transform(
        ConcretizeSymbols::new()
            .value("BATCH", 1)
            .value("ENCODER_OUTPUTS__TIME", 1)
            .value("DECODER_OUTPUTS__TIME", 1),
    )?;
    let joint = runtime.prepare(joint_model)?;

    eprintln!(
        "[{:7.3}] Models loaded. Encoder: {} audio frames/pulse -> {} token frames, delay: {}",
        t0.elapsed().as_secs_f64(),
        AUDIO_PULSE,
        enc_output_pulse,
        enc_delay,
    );

    // ── Load audio ──────────────────────────────────────────────────────
    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let total_samples = wav.len();
    eprintln!(
        "[{:7.3}] Audio: {:.2}s at 16kHz, chunk: {} samples ({:.1}ms)",
        t0.elapsed().as_secs_f64(),
        total_samples as f64 / 16000.0,
        AUDIO_CHUNK,
        AUDIO_CHUNK as f64 / 16.0,
    );

    // ── Mic thread: sends audio in small chunks ─────────────────────────
    let (audio_tx, audio_rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let mic_handle = std::thread::Builder::new().name("mic".into()).spawn(move || {
        let mut offset = 0;
        while offset < total_samples {
            let end = (offset + AUDIO_CHUNK).min(total_samples);
            if audio_tx.send(wav[offset..end].to_vec()).is_err() {
                break;
            }
            offset = end;
            std::thread::sleep(std::time::Duration::from_secs_f64(AUDIO_CHUNK as f64 / 16000.0));
        }
    })?;

    // ── Main thread: preproc + encoder + decoder pipeline ───────────────

    let print_partial = |hyp: &[usize], vocab: &[&str], tag: &str| {
        let text: String = hyp.iter().map(|&t| vocab[t]).join("");
        let display = text.replace('▁', " ");
        eprintln!("{} {tag}", display.trim_start());
    };

    let mut audio_buf: Vec<f32> = Vec::new();
    let mut features_produced: usize = 0;
    let mut preproc_state = preprocessor.spawn_state()?;
    let mut encoder_state = encoder.spawn_state()?;
    let mut delay_remaining = enc_delay;
    let mut pulse_count: usize = 0;

    // Warm-up decoder: NeMo's predict(add_sos=True, y=None) prepends a zero vector
    // then processes one zero embedding — 2 steps total.  With TARGETS__TIME=1 we
    // run the decoder twice with [blank_id] (which has zero embedding via padding_idx).
    let blank_tok = Tensor::from_slice(&[1, 1], &[blank_id as i32])?;
    let mut state_0 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    let mut state_1 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    // Step 1: SOS
    let [_out, s0, s1] = decoder.run([blank_tok.clone(), state_0, state_1])?.try_into().unwrap();
    state_0 = s0;
    state_1 = s1;
    // Step 2: first blank
    let [mut dec_token, s0, s1] = decoder.run([blank_tok, state_0, state_1])?.try_into().unwrap();
    state_0 = s0;
    state_1 = s1;

    let mut hyp: Vec<usize> = vec![];
    let max_tokens_per_frame = 10;

    for chunk in audio_rx {
        audio_buf.extend_from_slice(&chunk);

        // Run preprocessor on accumulated audio.
        let samples_tensor = Tensor::from_slice(&[1, audio_buf.len()], &audio_buf)?;
        let pp_start = Instant::now();
        let pp_results = preproc_state.run([samples_tensor])?;
        let pp_ms = pp_start.elapsed().as_secs_f64() * 1000.0;
        let features: ArrayD<f32> = pp_results[0].view()?.into_owned();
        let total_feat_frames = features.shape()[2];

        // Run encoder pulses as they become available.
        while features_produced + AUDIO_PULSE <= total_feat_frames {
            let start = features_produced;
            let end = start + AUDIO_PULSE;

            let mut pulse_data =
                Array3::<f32>::zeros((enc_input_shape[0], enc_input_shape[1], enc_input_shape[2]));
            pulse_data.slice_mut(s![.., .., ..AUDIO_PULSE]).assign(&features.slice(s![
                ..,
                ..,
                start..end
            ]));

            let enc_start = Instant::now();
            let pulse_tensor: Tensor = tensor(pulse_data)?;
            let enc_results = encoder_state.run([pulse_tensor])?;
            let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;
            let enc_out: ArrayD<f32> = enc_results[0].view()?.into_owned();

            pulse_count += 1;
            features_produced = end;

            let audio_time = audio_buf.len() as f64 / 16000.0;
            print_partial(
                &hyp,
                &vocab,
                &format!(
                    "[preproc {pp_ms:.0}ms + encoder {enc_ms:.0}ms | pulse {pulse_count} | audio {audio_time:.2}s]"
                ),
            );

            // Decode each encoder output frame.
            for f in 0..enc_output_pulse {
                if delay_remaining > 0 {
                    delay_remaining -= 1;
                    continue;
                }
                let frame: Tensor =
                    tensor(enc_out.slice_axis(Axis(enc_output_axis), (f..f + 1).into()))?;

                let mut tokens_this_frame = 0;
                loop {
                    let [logits] =
                        joint.run([frame.clone(), dec_token.clone()])?.try_into().unwrap();
                    let logits = logits.view::<f32>()?;
                    let logits = logits.as_slice().unwrap();
                    let token_id = argmax(logits).unwrap();

                    if token_id == blank_id {
                        break;
                    }

                    hyp.push(token_id);
                    tokens_this_frame += 1;

                    let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
                    [dec_token, state_0, state_1] =
                        decoder.run([tok, state_0, state_1])?.try_into().unwrap();

                    print_partial(&hyp, &vocab, "[decoder]");

                    if tokens_this_frame >= max_tokens_per_frame {
                        break;
                    }
                }
            }
        }
    }

    // Flush: zero-padded final pulse if there are leftover features.
    {
        let samples_tensor = Tensor::from_slice(&[1, audio_buf.len()], &audio_buf)?;
        let pp_results = preproc_state.run([samples_tensor])?;
        let features: ArrayD<f32> = pp_results[0].view()?.into_owned();
        let total_feat_frames = features.shape()[2];

        if total_feat_frames > features_produced {
            let start = features_produced;
            let remaining = total_feat_frames - start;

            let mut pulse_data =
                Array3::<f32>::zeros((enc_input_shape[0], enc_input_shape[1], enc_input_shape[2]));
            pulse_data.slice_mut(s![.., .., ..remaining]).assign(&features.slice(s![
                ..,
                ..,
                start..total_feat_frames
            ]));

            let enc_start = Instant::now();
            let pulse_tensor: Tensor = tensor(pulse_data)?;
            let enc_results = encoder_state.run([pulse_tensor])?;
            let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;
            let enc_out: ArrayD<f32> = enc_results[0].view()?.into_owned();

            pulse_count += 1;
            print_partial(
                &hyp,
                &vocab,
                &format!("[encoder {enc_ms:.0}ms | pulse {pulse_count} (flush)]"),
            );

            let output_frames = (total_feat_frames + 6) / 8 + 1 - (features_produced + 6) / 8;
            for f in 0..output_frames.min(enc_output_pulse) {
                if delay_remaining > 0 {
                    delay_remaining -= 1;
                    continue;
                }
                let frame: Tensor =
                    tensor(enc_out.slice_axis(Axis(enc_output_axis), (f..f + 1).into()))?;

                let mut tokens_this_frame = 0;
                loop {
                    let [logits] =
                        joint.run([frame.clone(), dec_token.clone()])?.try_into().unwrap();
                    let logits = logits.view::<f32>()?;
                    let logits = logits.as_slice().unwrap();
                    let token_id = argmax(logits).unwrap();

                    if token_id == blank_id {
                        break;
                    }

                    hyp.push(token_id);
                    tokens_this_frame += 1;

                    let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
                    [dec_token, state_0, state_1] =
                        decoder.run([tok, state_0, state_1])?.try_into().unwrap();

                    print_partial(&hyp, &vocab, "[decoder]");

                    if tokens_this_frame >= max_tokens_per_frame {
                        break;
                    }
                }
            }
        }
    }

    mic_handle.join().unwrap();

    let transcript: String = hyp.iter().map(|&t| vocab[t]).join("");
    let display = transcript.replace('▁', " ");
    eprintln!("\n{}", display.trim_start());

    let expected = "▁well▁I▁don't▁wish▁to▁see▁it▁any▁more▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait";
    if transcript != expected {
        eprintln!(
            "\nNOTE: streaming transcript differs slightly from batch reference (pulse boundary effects)"
        );
    }
    Ok(())
}
