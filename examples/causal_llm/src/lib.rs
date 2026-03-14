use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, ensure};
use float_ord::FloatOrd;
use log::{info, trace};
use tokenizers::Tokenizer;
use tract::prelude::*;

#[derive(Clone, Debug)]
struct KvCacheInfo {
    axis: usize,
    dt: DatumType,
    /// Concrete shape for an empty cache (seq_len=0 on the cache axis)
    empty_shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct CausalLlmModelConfig {
    pub force_cpu: bool,
}

#[derive(Clone, Debug)]
pub struct CausalLlmModel {
    pub tokenizer: Tokenizer,
    pub nn: Runnable,
    pub conf: CausalLlmModelConfig,
    kv_caches: Vec<KvCacheInfo>,
    n_regular_outputs: usize,
}

impl CausalLlmModel {
    fn from_tokenizer_and_model(
        tokenizer: tokenizers::Tokenizer,
        nn: Model,
        conf: CausalLlmModelConfig,
    ) -> anyhow::Result<Arc<CausalLlmModel>> {
        let mut nn = nn;

        // Detect transformer patterns (creates DynKeyValueCache ops)
        nn.transform("transformers_detect_all")?;

        let n_regular_outputs = nn.output_count()?;

        // Unfold KV caches into explicit model I/O
        nn.transform("unfold-kv-cache")?;

        // Detect KV cache inputs by comparing pre/post unfold input counts.
        // Token input (idx 0) is integer, KV cache inputs are float.
        let token_fact = nn.input_fact(0)?;
        let mut token_symbols = HashSet::new();
        for axis in 0..token_fact.rank()? {
            let dim = token_fact.dim(axis)?;
            if dim.to_int64().is_err() {
                token_symbols.insert(format!("{dim}"));
            }
        }

        let n_inputs = nn.input_count()?;
        let mut kv_infos = Vec::new();
        for i in 1..n_inputs {
            let fact = nn.input_fact(i)?;
            let dt = fact.datum_type()?;
            let mut cache_axis = None;
            let mut empty_shape = vec![];
            for axis in 0..fact.rank()? {
                let dim = fact.dim(axis)?;
                match dim.to_int64() {
                    Ok(v) => empty_shape.push(v as usize),
                    Err(_) => {
                        let sym_name = format!("{dim}");
                        if token_symbols.contains(&sym_name) {
                            // Batch dim — default to 1
                            empty_shape.push(1);
                        } else {
                            // Cache axis (past sequence) — start at 0
                            cache_axis = Some(axis);
                            empty_shape.push(0);
                        }
                    }
                }
            }
            kv_infos.push(KvCacheInfo {
                axis: cache_axis.context("no symbolic cache axis found in KV cache input")?,
                dt,
                empty_shape,
            });
        }

        // Try runtimes in order of preference
        let runtimes =
            if conf.force_cpu { vec!["default"] } else { vec!["cuda", "metal", "default"] };
        let mut runnable = None;
        for rt_name in &runtimes {
            match runtime_for_name(rt_name) {
                Ok(rt) => match rt.prepare(nn.clone()) {
                    Ok(r) => {
                        info!("Using runtime {rt_name}");
                        runnable = Some(r);
                        break;
                    }
                    Err(e) => info!("{rt_name} {e:?}"),
                },
                Err(_) => continue,
            }
        }
        let nn = runnable.context("No suitable runtime")?;
        Ok(Arc::new(CausalLlmModel { tokenizer, nn, conf, kv_caches: kv_infos, n_regular_outputs }))
    }

    pub fn from_paths_and_conf(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
        conf: CausalLlmModelConfig,
    ) -> anyhow::Result<Arc<CausalLlmModel>> {
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!(e))?;
        let nn = nnef.load(nn)?;
        CausalLlmModel::from_tokenizer_and_model(tokenizer, nn, conf)
    }

    pub fn from_paths(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
    ) -> anyhow::Result<Arc<CausalLlmModel>> {
        Self::from_paths_and_conf(tokenizer, nn, Default::default())
    }

    pub fn from_bytes_and_conf(
        tokenizer_bytes: impl AsRef<[u8]>,
        llm_model_bytes: impl AsRef<[u8]>,
        conf: CausalLlmModelConfig,
    ) -> anyhow::Result<Arc<CausalLlmModel>> {
        let nnef = tract::nnef()?.with_tract_transformers()?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow::anyhow!(e))?;
        let nn = nnef.load_buffer(llm_model_bytes.as_ref())?;
        CausalLlmModel::from_tokenizer_and_model(tokenizer, nn, conf)
    }

    pub fn from_bytes(
        tokenizer_bytes: impl AsRef<[u8]>,
        llm_model_bytes: impl AsRef<[u8]>,
    ) -> anyhow::Result<Arc<CausalLlmModel>> {
        Self::from_bytes_and_conf(tokenizer_bytes, llm_model_bytes, Default::default())
    }

    fn make_empty_kv_caches(&self) -> anyhow::Result<Vec<Tensor>> {
        self.kv_caches
            .iter()
            .map(|info| {
                let n_bytes = info.empty_shape.iter().product::<usize>() * info.dt.size_of();
                Tensor::from_bytes(info.dt, &info.empty_shape, &vec![0u8; n_bytes])
            })
            .collect()
    }

    pub fn spawn(self: &Arc<Self>) -> anyhow::Result<CausalLlmState> {
        let kv_caches = self.make_empty_kv_caches()?;
        Ok(CausalLlmState {
            model: self.clone(),
            kv_caches,
            seq: vec![],
            processed_tokens: 0,
            config: CausalLlmStateConfig::default(),
        })
    }

    pub fn spawn_with_config(
        self: &Arc<Self>,
        config: CausalLlmStateConfig,
    ) -> anyhow::Result<CausalLlmState> {
        let kv_caches = self.make_empty_kv_caches()?;
        Ok(CausalLlmState {
            model: self.clone(),
            kv_caches,
            seq: vec![],
            processed_tokens: 0,
            config,
        })
    }
}

/// Config of the state containing specific behavior requested
/// prompt chunking, sampling strategy, ...
#[derive(Debug, Clone, PartialEq)]
pub struct CausalLlmStateConfig {
    /// split long prompt into chunks of fixed number of tokens,
    /// reducing RAM usage
    pub prompt_chunk_size: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}
impl Default for CausalLlmStateConfig {
    fn default() -> Self {
        Self { prompt_chunk_size: Some(512), repeat_penalty: 1.0, repeat_last_n: 64 }
    }
}

impl CausalLlmStateConfig {
    pub fn new(
        prompt_chunk_size: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> anyhow::Result<Self> {
        let conf = Self { prompt_chunk_size, repeat_penalty, repeat_last_n };
        conf.validate()?;
        Ok(conf)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(chunk_size) = self.prompt_chunk_size {
            anyhow::ensure!(
                chunk_size > 0,
                format!(
                    "prompt_chunk_size cannot be set to '{chunk_size}', set None to deactivate it."
                )
            )
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CausalLlmState {
    pub model: Arc<CausalLlmModel>,
    kv_caches: Vec<Tensor>,
    pub seq: Vec<u32>,
    pub processed_tokens: usize,
    pub config: CausalLlmStateConfig,
}

impl CausalLlmState {
    pub fn append_text(&mut self, prompt: &str) -> anyhow::Result<()> {
        self.seq.extend(self.encode(prompt, true)?);
        Ok(())
    }

    pub fn generate_next_token(&mut self) -> anyhow::Result<()> {
        let tokens = &self.seq[self.processed_tokens..];
        ensure!(tokens.len() > 0);
        let chunk_size = self.config.prompt_chunk_size.unwrap_or(usize::MAX);
        let output = tokens
            .chunks(chunk_size)
            .map(|chunk| -> anyhow::Result<Tensor> {
                let start = Instant::now();
                let token_data: Vec<i64> = chunk.iter().map(|t| *t as i64).collect();
                let input: Tensor =
                    tract_ndarray::Array2::from_shape_vec((1, chunk.len()), token_data)?
                        .try_into()?;
                // Build inputs: token_ids + current kv caches
                let mut inputs: Vec<Tensor> = vec![input];
                inputs.extend(self.kv_caches.iter().cloned());
                let mut results = self.model.nn.run(inputs)?;
                // Extract updated KV caches from outputs
                // outputs layout: [regular_outputs..., kv_cache_0, kv_cache_1, ...]
                let n_regular = self.model.n_regular_outputs;
                for (i, kv) in results.drain(n_regular..).enumerate() {
                    self.kv_caches[i] = kv;
                }
                trace!("Processed {} tokens in {:?}", chunk.len(), start.elapsed());
                Ok(results.remove(0))
            })
            .last()
            .unwrap()?;

        let start_at = self.seq.len().saturating_sub(self.config.repeat_last_n);

        let output_f32 = output.convert_to(DatumType::F32)?;
        let mut last_token_logits: Vec<f32> = output_f32.as_slice::<f32>()?.to_vec();

        apply_repeat_penalty(
            last_token_logits.as_mut_slice(),
            self.config.repeat_penalty,
            &self.seq[start_at..],
        );

        let next_tok = last_token_logits
            .iter()
            .enumerate()
            .max_by_key(|(_ix, v)| FloatOrd(**v))
            .context("no tokens in output ?")?
            .0 as u32;

        self.processed_tokens += tokens.len();
        self.seq.push(next_tok);
        Ok(())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.model.tokenizer
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.tokenizer().decode(tokens, skip_special_tokens).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .tokenizer()
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    pub fn freeze(self) -> FrozenCausalLlmState {
        FrozenCausalLlmState {
            model: self.model,
            kv_caches: self.kv_caches,
            seq: self.seq,
            config: self.config,
        }
    }

    pub fn truncate(&mut self, len: usize) -> anyhow::Result<()> {
        anyhow::ensure!(len > 0, "cannot truncate to 0");
        self.seq.truncate(len);
        // Slice KV caches to len-1: the last token will be re-processed by
        // the next generate_next_token call to produce output logits.
        let cache_len = len - 1;
        self.processed_tokens = self.processed_tokens.min(cache_len);
        for (kv, info) in self.kv_caches.iter_mut().zip(&self.model.kv_caches) {
            *kv = slice_value(kv, info.axis, 0, cache_len)?;
        }
        Ok(())
    }
}

/// Slice a Value along an axis, extracting elements [start..end].
fn slice_value(value: &Tensor, axis: usize, start: usize, end: usize) -> anyhow::Result<Tensor> {
    let (dt, shape, data) = value.as_bytes()?;
    let elem_size = dt.size_of();
    let new_len = end - start;
    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_len;

    let inner_size: usize = shape[axis + 1..].iter().product::<usize>() * elem_size;
    let axis_stride = shape[axis] * inner_size;
    let outer_count: usize = shape[..axis].iter().product::<usize>().max(1);

    let new_data_len = outer_count * new_len * inner_size;
    let mut new_data = vec![0u8; new_data_len];

    for outer in 0..outer_count {
        let src_offset = outer * axis_stride + start * inner_size;
        let dst_offset = outer * new_len * inner_size;
        new_data[dst_offset..dst_offset + new_len * inner_size]
            .copy_from_slice(&data[src_offset..src_offset + new_len * inner_size]);
    }

    Tensor::from_bytes(dt, &new_shape, &new_data)
}

#[derive(Clone, Debug)]
pub struct FrozenCausalLlmState {
    pub model: Arc<CausalLlmModel>,
    kv_caches: Vec<Tensor>,
    pub seq: Vec<u32>,
    pub config: CausalLlmStateConfig,
}

impl FrozenCausalLlmState {
    pub fn unfreeze(self) -> anyhow::Result<CausalLlmState> {
        Ok(CausalLlmState {
            model: self.model,
            kv_caches: self.kv_caches,
            seq: self.seq,
            processed_tokens: 0,
            config: self.config,
        })
    }
}

/// Cheap way to avoid repeating model (mostly usefull for tiny models)
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    if penalty == 1.0 {
        return;
    }
    let context: std::collections::HashSet<_> = context.iter().collect();
    for (token_id, logit) in logits.iter_mut().enumerate() {
        if context.contains(&(token_id as u32)) {
            if *logit >= 0. { *logit /= penalty } else { *logit *= penalty }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::FrozenCausalLlmState;
    use tract::prelude::*;

    fn is_send<T: Send>() {}

    #[test]
    fn frozen_state_is_send() {
        is_send::<FrozenCausalLlmState>();
    }

    #[test]
    fn truncate_prefix_cache_hit() -> anyhow::Result<()> {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.cached/llm/541/meta-llama--Llama-3.2-1B-Instruct-q40ef16");
        if !model_dir.exists() {
            eprintln!("Skipping: model not found at {model_dir:?}");
            return Ok(());
        }
        let llm = crate::CausalLlmModel::from_paths(
            model_dir.join("tokenizer.json"),
            model_dir.join("meta-llama--Llama-3.2-1B-Instruct-q40ef16.nnef.tgz"),
        )?;

        // Generate from "Hello world" for 5 tokens
        let mut state = llm.spawn()?;
        state.append_text("Hello world")?;
        for _ in 0..5 {
            state.generate_next_token()?;
        }
        let full_seq = state.seq.clone();
        let prompt_len = state.encode("Hello world", true)?.len();

        // Truncate back to just the prompt (simulating prefix cache hit)
        // KV cache is sliced to prompt_len-1, last token will be re-processed
        state.truncate(prompt_len)?;
        assert_eq!(state.seq.len(), prompt_len);
        assert_eq!(state.processed_tokens, prompt_len - 1);

        // Generate one token and verify KV cache grew by exactly 1
        // (proving the truncated prefix was preserved, not recomputed from scratch)
        state.generate_next_token()?;
        {
            let shape = state.kv_caches[0].shape()?;
            let cache_axis = state.model.kv_caches[0].axis;
            assert_eq!(
                shape[cache_axis], prompt_len,
                "KV cache should have prompt_len entries after truncate + 1 generate"
            );
        }

        // Generate 4 more tokens
        for _ in 0..4 {
            state.generate_next_token()?;
        }
        let regenerated_seq = state.seq.clone();

        // A fresh state from the same prompt should produce identical output
        let mut fresh = llm.spawn()?;
        fresh.append_text("Hello world")?;
        for _ in 0..5 {
            fresh.generate_next_token()?;
        }

        // The truncated-and-regenerated sequence should match the fresh sequence
        assert_eq!(
            regenerated_seq, fresh.seq,
            "truncate + regenerate should match fresh generation"
        );
        // And should also match the original since generation is deterministic
        assert_eq!(full_seq, fresh.seq, "generation should be deterministic");

        Ok(())
    }
}
