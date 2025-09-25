use std::path::Path;
use std::time::Instant;

use anyhow::ensure;
use float_ord::FloatOrd;
use log::trace;
use tokenizers::Tokenizer;
use tract_nnef::internal::{TractErrorContext, anyhow};
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_nnef::tract_core::ops::source::SourceState;
use tract_transformers::WithTractTransformers;
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCacheState;

#[derive(Clone, Debug, Default)]
pub struct CausalLlmModelConfig {
    pub force_cpu: bool,
}

#[derive(Clone, Debug)]
pub struct CausalLlmModel {
    pub tokenizer: Tokenizer,
    pub nn: Arc<TypedRunnableModel<TypedModel>>,
    pub conf: CausalLlmModelConfig,
}

fn plan_with_session_handler(model: TypedModel) -> TractResult<TypedSimplePlan<TypedModel>> {
    if model.properties.contains_key("GPU") {
        let plan_options = PlanOptions { skip_order_opt_ram: true, ..Default::default() };
        let plan = model.into_runnable_with_options(&plan_options)?;
        let model = plan.model();
        let mut symbol_values = SymbolValues::default();
        let sequence_length = model.symbols.get("S").context("Could not find symbol S in model")?;
        let past_sequence_length =
            model.symbols.get("P").context("Could not find symbol P in model")?;

        symbol_values.set(&sequence_length, 1024);
        symbol_values.set(&past_sequence_length, 0);
        let session_handler =
            tract_gpu::session_handler::DeviceSessionHandler::from_plan(&plan, &symbol_values)?;
        Ok(plan.with_session_handler(session_handler))
    } else {
        model.into_runnable()
    }
}

impl CausalLlmModel {
    fn from_tokenizer_and_typed_model(
        tokenizer: tokenizers::Tokenizer,
        nn: TypedModel,
        conf: CausalLlmModelConfig,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;
        let mut nn = nn.into_decluttered()?;
        nn.transform(&*transform)?;
        if !conf.force_cpu {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                use crate::tract_core::transform::ModelTransform;
                use std::str::FromStr;
                nn.properties.insert("GPU".into(), rctensor0(true));
                tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                use tract_core::transform::ModelTransform;
                if tract_cuda::utils::is_culib_present() {
                    nn.properties.insert("GPU".into(), rctensor0(true));
                    tract_cuda::CudaTransform.transform(&mut nn)?;
                }
            }
        }
        nn.optimize()?;
        let nn = plan_with_session_handler(nn)?;
        Ok(Arc::new(CausalLlmModel { tokenizer, nn: Arc::new(nn), conf }))
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

    pub fn spawn(self: &Arc<Self>) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState {
            model: self.clone(),
            nn_state,
            seq: vec![],
            processed_tokens: 0,
            config: CausalLlmStateConfig::default(),
        })
    }

    pub fn spawn_with_config(
        self: &Arc<Self>,
        config: CausalLlmStateConfig,
    ) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState {
            model: self.clone(),
            nn_state,
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

#[derive(Debug, Clone)]
pub struct CausalLlmState {
    pub model: Arc<CausalLlmModel>,
    pub nn_state: TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
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
                // let input_node = self.nn_state.model().input_outlets()?[0].node;
                // let input_name = &self.nn_state.model().nodes[input_node].name;
                // ndarray_npy::NpzWriter::new_compressed(std::fs::File::create("io.npz").unwrap())
                //     .add_array(input_name, &input.to_array_view::<i64>()?)?;
                let mut result = self.nn_state.run(tvec![input.into_tvalue()])?;
                trace!("Processed {} tokens in {:?}", chunk.len(), start.elapsed());
                Ok(result.remove(0).into_tensor())
            })
            .last()
            .unwrap()?;

        let start_at = self.seq.len().saturating_sub(self.config.repeat_last_n);

        let mut last_token_logits =
            output.cast_to::<f32>()?.as_slice::<f32>()?.iter().copied().collect_vec();

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
            nn_state: self.nn_state.freeze(),
            seq: self.seq,
            config: self.config,
        }
    }

    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        self.seq.truncate(len);
        for s in &mut self.nn_state.states {
            let Some(s) = s else { continue };
            if let Some(s) = s.downcast_mut::<DynKeyValueCacheState>() {
                s.truncate(len)?;
                continue;
            } else if s.is::<SourceState>() {
                continue;
            }
            if let Some(s) = s.downcast_mut::<tract_cuda::ops::CudaDynKVCacheState>() {
                s.truncate(len)?;
                continue;
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if let Some(s) = s.downcast_mut::<tract_metal::ops::MetalDynKVCacheState>() {
                s.truncate(len)?;
                continue;
            }
            anyhow::bail!("Can not truncate context with state {s:?}");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FrozenCausalLlmState {
    pub model: Arc<CausalLlmModel>,
    pub nn_state: TypedFrozenSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
    pub seq: Vec<u32>,
    pub config: CausalLlmStateConfig,
}

impl FrozenCausalLlmState {
    pub fn unfreeze(self) -> CausalLlmState {
        CausalLlmState {
            model: self.model,
            nn_state: self.nn_state.unfreeze(),
            seq: self.seq,
            processed_tokens: 0,
            config: self.config,
        }
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
