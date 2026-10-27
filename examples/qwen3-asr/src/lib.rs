use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use ndarray::prelude::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const MEL_BINS: usize = 128;
pub const CHUNK_FRAMES: usize = 100;
pub const TOKENS_PER_CHUNK: usize = 13;
pub const WINDOW_FRAMES: usize = 8 * CHUNK_FRAMES;
pub const HIDDEN: usize = 1024;
pub const AUDIO_PAD: i64 = 151676;
const EOS: [i64; 2] = [151643, 151645];
const MAX_TOKENS: usize = 256;

pub const SAMPLE_WAV: &str = "assets/2086-149220-0033.wav";
pub const SAMPLE_TRANSCRIPT: &str = r#"language English<asr_text>Well, I don't wish to see it any more," observed Phoebe, turning away her eyes. "It is certainly very like the old portrait.""#;

pub fn load_wav(path: &str) -> Result<Vec<f32>> {
    Ok(hound::WavReader::open(path)?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect())
}

pub struct Models {
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    tokenizer: tokenizers::Tokenizer,
    pub prompt_head: Vec<i64>,
    prompt_tail: Vec<i64>,
}

impl Models {
    pub fn load() -> Result<Models> {
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let gpu = tract::runtime_for_name("gpu-or-cpu")?;
        eprintln!("runtime: {}", gpu.name()?);
        let preprocessor = nnef.load("assets/model/preprocessor.nnef.tgz")?.into_runnable()?;
        let encoder = gpu.prepare(nnef.load("assets/model/encoder.nnef.tgz")?)?;
        let mut decoder = nnef.load("assets/model/decoder/model/model.nnef.tgz")?;
        decoder.transform("transformers_detect_all")?;
        decoder.transform("unfold-kv-cache")?;
        let decoder = gpu.prepare(decoder)?;
        let tokenizer = tokenizers::Tokenizer::from_file("assets/model/tokenizer.json")
            .map_err(|e| anyhow!(e))?;
        let encode = |s: &str| -> Result<Vec<i64>> {
            let encoding = tokenizer.encode(s, false).map_err(|e| anyhow!(e))?;
            Ok(encoding.get_ids().iter().map(|&i| i as i64).collect())
        };
        let prompt_head =
            encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
        let prompt_tail = encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?;
        Ok(Models { preprocessor, encoder, decoder, tokenizer, prompt_head, prompt_tail })
    }

    /// log10 mel of every frame whose 400-sample window fits in `padded`.
    pub fn log_mel(&self, padded: &[f32]) -> Result<Array2<f32>> {
        if padded.len() < N_FFT {
            return Ok(Array2::zeros((MEL_BINS, 0)));
        }
        let signal = Tensor::from_slice(&[1, padded.len()], padded)?;
        let [mel] = self.preprocessor.run([signal])?.try_into().unwrap();
        Ok(mel.ndarray2::<f32>()?.to_owned())
    }

    /// Audio embeddings of one attention window of at most 8 s of log mel, floored at `max` - 8
    /// as Whisper features are. The last chunk is zero-padded, its padding tokens masked out of
    /// attention, then dropped.
    pub fn encode_window(&self, log_mel: ArrayView2<f32>, max: f32) -> Result<Array2<f32>> {
        let features = log_mel.mapv(|x| (x.max(max - 8.0) + 4.0) / 4.0);
        let len = features.shape()[1];
        let n_chunks = len.div_ceil(CHUNK_FRAMES);
        let mut chunks = Array3::<f32>::zeros((n_chunks, MEL_BINS, CHUNK_FRAMES));
        for (c, mut chunk) in chunks.outer_iter_mut().enumerate() {
            let src = features.slice(s![.., c * CHUNK_FRAMES..len.min((c + 1) * CHUNK_FRAMES)]);
            chunk.slice_mut(s![.., ..src.shape()[1]]).assign(&src);
        }
        let tail = len - (n_chunks - 1) * CHUNK_FRAMES;
        let valid = (n_chunks - 1) * TOKENS_PER_CHUNK + (0..3).fold(tail, |l, _| (l - 1) / 2 + 1);
        let mut key_bias = Array2::<f32>::zeros((n_chunks, TOKENS_PER_CHUNK));
        key_bias.iter_mut().skip(valid).for_each(|b| *b = f32::MIN);
        let [out] = self.encoder.run([chunks.tract()?, key_bias.tract()?])?.try_into().unwrap();
        Ok(out.ndarray2::<f32>()?.slice(s![..valid, ..]).to_owned())
    }

    pub fn empty_kv(&self) -> Result<Vec<Tensor>> {
        (2..self.decoder.input_count()?)
            .map(|_| Ok(Array4::<f32>::zeros((1, 8, 0, 128)).tract()?))
            .collect()
    }

    /// Feeds `ids` after `kv`, the audio embeddings going to its AUDIO_PAD positions. Returns
    /// the greedy next token and the extended KV cache.
    pub fn forward(
        &self,
        ids: &[i64],
        audio: ArrayView2<f32>,
        kv: &[Tensor],
    ) -> Result<(usize, Vec<Tensor>)> {
        let mut inputs = vec![Tensor::from_slice(&[1, ids.len()], ids)?, audio.tract()?];
        inputs.extend(kv.iter().cloned());
        let mut outputs = self.decoder.run(inputs)?;
        let kv = outputs.split_off(2);
        let logits = outputs[0].as_slice::<f32>()?;
        Ok((logits.iter().position_max_by_key(|x| FloatOrd(**x)).unwrap(), kv))
    }

    /// Prefills `ids` after `kv`, then decodes greedily until end of turn.
    pub fn generate(&self, ids: &[i64], audio: ArrayView2<f32>, kv: &[Tensor]) -> Result<Vec<i64>> {
        let no_audio = Array2::<f32>::zeros((0, HIDDEN));
        let (mut token, mut kv) = self.forward(ids, audio, kv)?;
        let mut generated = vec![];
        while !EOS.contains(&(token as i64)) && generated.len() < MAX_TOKENS {
            generated.push(token as i64);
            (token, kv) = self.forward(&[token as i64], no_audio.view(), &kv)?;
        }
        Ok(generated)
    }

    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let ids = ids.iter().map(|&i| i as u32).collect_vec();
        self.tokenizer.decode(&ids, true).map_err(|e| anyhow!(e))
    }

    /// The prompt after `<|audio_start|>`: `n_audio` audio placeholders, the assistant turn
    /// header, then `text` as the start of the answer.
    pub fn audio_prompt(&self, n_audio: usize, text: &[i64]) -> Vec<i64> {
        let mut ids = vec![AUDIO_PAD; n_audio];
        ids.extend(&self.prompt_tail);
        ids.extend(text);
        ids
    }
}

/// The reflected samples Whisper's centered STFT pads `signal` with, on its left or right.
pub fn reflect(signal: &[f32], left: bool) -> impl Iterator<Item = f32> + '_ {
    let n = signal.len();
    (1..=N_FFT / 2).map(move |i| if left { signal[N_FFT / 2 + 1 - i] } else { signal[n - 1 - i] })
}
