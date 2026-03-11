use std::path::Path;
use std::time::Instant;

use anyhow::ensure;
use float_ord::FloatOrd;
use log::{info, trace};
use tokenizers::Tokenizer;
use tract_nnef::internal::DimLike;
use tract_nnef::internal::{TractErrorContext, anyhow};
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_transformers::WithTractTransformers;

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
    pub nn: Arc<dyn Runnable>,
    pub conf: CausalLlmModelConfig,
    kv_caches: Vec<KvCacheInfo>,
    n_regular_outputs: usize,
}

impl CausalLlmModel {
    fn from_tokenizer_and_typed_model(
        tokenizer: tokenizers::Tokenizer,
        nn: TypedModel,
        conf: CausalLlmModelConfig,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let transform = tract_core::transform::get_transform("transformers_detect_all")?
            .context("transformers_detect_all not found")?;
        let mut nn = nn.into_decluttered()?;
        nn.transform(&*transform)?;

        // Collect KV cache metadata before unfolding
        let (b, _s, p) = tract_transformers::figure_out_causal_llm_b_s_p(&nn)?;
        let mut resolve = SymbolValues::default();
        if let Some(p) = &p {
            resolve = resolve.with(p, 0);
        }
        if let Some(b) = &b {
            resolve = resolve.with(b, 1);
        }

        let n_regular_outputs = nn.outputs.len();

        let mut kv_infos = Vec::new();
        for node in nn.nodes() {
            if let Some(op) =
                node.op_as::<tract_transformers::ops::dyn_kv_cache::DynKeyValueCache>()
            {
                let empty_shape: Vec<usize> = op
                    .past_sequence_fact
                    .shape
                    .iter()
                    .map(|d| d.eval(&resolve).to_usize())
                    .collect::<TractResult<_>>()?;
                kv_infos.push(KvCacheInfo {
                    axis: op.axis,
                    dt: op.past_sequence_fact.datum_type,
                    empty_shape,
                });
            }
        }

        // Unfold KV caches into explicit model I/O
        let unfold = tract_core::transform::get_transform("unfold-kv-cache")?
            .context("unfold-kv-cache not found")?;
        nn.transform(&*unfold)?;

        let options = RunOptions {
            memory_sizing_hints: Some(tract_transformers::memory_arena_hints_for_causal_llm(&nn)?),
            ..Default::default()
        };

        let runtimes =
            if conf.force_cpu { vec!["default"] } else { vec!["cuda", "metal", "default"] };
        let mut runnable = None;
        for rt in runtimes {
            if let Some(rt) = runtime_for_name(rt) {
                match rt.prepare_with_options(nn.clone(), &options) {
                    Ok(it) => {
                        info!("Using runtime {rt:?}");
                        runnable = Some(it);
                        break;
                    }
                    Err(e) => info!("{rt:?} {e:?}"),
                }
            }
        }
        let nn = runnable.context("No suitable runtime")?.into();
        Ok(Arc::new(CausalLlmModel { tokenizer, nn, conf, kv_caches: kv_infos, n_regular_outputs }))
    }

    pub fn from_paths_and_conf(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
        conf: CausalLlmModelConfig,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow!(e))?;
        let nn = nnef.model_for_path(nn.as_ref())?;
        CausalLlmModel::from_tokenizer_and_typed_model(tokenizer, nn, conf)
    }

    pub fn from_paths(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        Self::from_paths_and_conf(tokenizer, nn, Default::default())
    }

    pub fn from_bytes_and_conf(
        tokenizer_bytes: impl AsRef<[u8]>,
        llm_model_bytes: impl AsRef<[u8]>,
        conf: CausalLlmModelConfig,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!(e))?;
        let mut llm_read = std::io::Cursor::new(llm_model_bytes);
        let nn = nnef.model_for_read(&mut llm_read)?;
        CausalLlmModel::from_tokenizer_and_typed_model(tokenizer, nn, conf)
    }

    pub fn from_bytes(
        tokenizer_bytes: impl AsRef<[u8]>,
        llm_model_bytes: impl AsRef<[u8]>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        Self::from_bytes_and_conf(tokenizer_bytes, llm_model_bytes, Default::default())
    }

    fn make_empty_kv_caches(&self) -> TractResult<Vec<TValue>> {
        self.kv_caches
            .iter()
            .map(|info| Tensor::zero_dt(info.dt, &info.empty_shape).map(|t| t.into_tvalue()))
            .collect()
    }

    pub fn spawn(self: &Arc<Self>) -> TractResult<CausalLlmState> {
        let nn_state = self.nn.spawn()?;
        let kv_caches = self.make_empty_kv_caches()?;
        Ok(CausalLlmState {
            model: self.clone(),
            nn_state,
            kv_caches,
            seq: vec![],
            processed_tokens: 0,
            config: CausalLlmStateConfig::default(),
        })
    }

    pub fn spawn_with_config(
        self: &Arc<Self>,
        config: CausalLlmStateConfig,
    ) -> TractResult<CausalLlmState> {
        let nn_state = self.nn.spawn()?;
        let kv_caches = self.make_empty_kv_caches()?;
        Ok(CausalLlmState {
            model: self.clone(),
            nn_state,
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
    ) -> TractResult<Self> {
        let conf = Self { prompt_chunk_size, repeat_penalty, repeat_last_n };
        conf.validate()?;
        Ok(conf)
    }

    pub fn validate(&self) -> TractResult<()> {
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
    nn_state: Box<dyn State>,
    kv_caches: Vec<TValue>,
    pub seq: Vec<u32>,
    pub processed_tokens: usize,
    pub config: CausalLlmStateConfig,
}

impl CausalLlmState {
    pub fn append_text(&mut self, prompt: &str) -> TractResult<()> {
        let _: () = self.seq.extend(self.encode(prompt, true)?);
        Ok(())
    }

    pub fn generate_next_token(&mut self) -> TractResult<()> {
        let tokens = &self.seq[self.processed_tokens..];
        ensure!(tokens.len() > 0);
        let chunk_size = self.config.prompt_chunk_size.unwrap_or(usize::MAX);
        let output = tokens
            .chunks(chunk_size)
            .map(|chunk| -> TractResult<Tensor> {
                let start = Instant::now();
                let mut input = tensor1(&chunk.iter().map(|t| *t as i64).collect_vec());
                input.insert_axis(0)?;
                // Build inputs: token_ids + current kv caches
                let mut inputs: TVec<TValue> = tvec![input.into_tvalue()];
                inputs.extend(self.kv_caches.iter().cloned());
                let mut results = self.nn_state.run(inputs)?;
                // Extract updated KV caches from outputs
                // outputs layout: [regular_outputs..., kv_cache_0, kv_cache_1, ...]
                let n_regular = self.model.n_regular_outputs;
                for (i, kv) in results.drain(n_regular..).enumerate() {
                    self.kv_caches[i] = kv;
                }
                trace!("Processed {} tokens in {:?}", chunk.len(), start.elapsed());
                Ok(results.remove(0).into_tensor())
            })
            .last()
            .unwrap()?;

        let start_at = self.seq.len().saturating_sub(self.config.repeat_last_n);

        let mut last_token_logits = output
            .cast_to::<f32>()?
            .try_as_dense()?
            .as_slice::<f32>()?
            .iter()
            .copied()
            .collect_vec();

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

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> TractResult<String> {
        self.tokenizer().decode(tokens, skip_special_tokens).map_err(|e| anyhow!(e))
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TractResult<Vec<u32>> {
        Ok(self
            .tokenizer()
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    pub fn freeze(self) -> FrozenCausalLlmState {
        FrozenCausalLlmState {
            model: self.model,
            kv_caches: self.kv_caches.into_iter().map(|t| t.into_tensor()).collect(),
            seq: self.seq,
            config: self.config,
        }
    }

    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        self.seq.truncate(len);
        for (kv, info) in self.kv_caches.iter_mut().zip(&self.model.kv_caches) {
            *kv = kv.slice(info.axis, 0, len)?.into_tvalue();
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FrozenCausalLlmState {
    pub model: Arc<CausalLlmModel>,
    kv_caches: Vec<Tensor>,
    pub seq: Vec<u32>,
    pub config: CausalLlmStateConfig,
}

impl FrozenCausalLlmState {
    pub fn unfreeze(self) -> TractResult<CausalLlmState> {
        let nn_state = self.model.nn.spawn()?;
        Ok(CausalLlmState {
            model: self.model,
            nn_state,
            kv_caches: self.kv_caches.into_iter().map(|t| t.into_tvalue()).collect(),
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

    fn is_send<T: Send>() {}

    #[test]
    fn frozen_state_is_send() {
        is_send::<FrozenCausalLlmState>();
    }
}
