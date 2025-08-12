use std::path::Path;

use float_ord::FloatOrd;
use tokenizers::Tokenizer;
use tract_nnef::internal::{TractErrorContext, anyhow};
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_nnef::tract_core::ops::source::SourceState;
use tract_transformers::WithTractTransformers;
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCacheState;

#[derive(Clone, Debug)]
pub struct CausalLlmModel {
    pub tokenizer: Tokenizer,
    pub nn: Arc<TypedRunnableModel<TypedModel>>,
}
fn plan_with_session_handler(model: TypedModel) -> TractResult<TypedSimplePlan<TypedModel>> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
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
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        model.into_runnable()
    }
}

impl CausalLlmModel {
    fn from_tokenizer_and_typed_model(
        tokenizer: tokenizers::Tokenizer,
        nn: TypedModel,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;
        let mut nn = nn.into_decluttered()?;
        nn.transform(&*transform)?;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            use crate::tract_core::transform::ModelTransform;
            use std::str::FromStr;
            tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
        }
        nn.optimize()?;
        let nn = plan_with_session_handler(nn)?;
        Ok(Arc::new(CausalLlmModel { tokenizer, nn: Arc::new(nn) }))
    }

    fn from_paths(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow!(e))?;
        let nn = nnef.model_for_path(nn.as_ref())?;
        CausalLlmModel::from_tokenizer_and_typed_model(tokenizer, nn)
    }

    pub fn from_bytes(
        tokenizer_bytes: impl AsRef<[u8]>,
        llm_model_bytes: impl AsRef<[u8]>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!(e))?;
        let mut llm_read = std::io::Cursor::new(llm_model_bytes);
        let nn = nnef.model_for_read(&mut llm_read)?;
        CausalLlmModel::from_tokenizer_and_typed_model(tokenizer, nn)
    }

    pub fn spawn(self: &Arc<Self>) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState {
            model: self.clone(),
            nn_state,
            seq: vec![],
            config: CausalLlmStateConfig::default(),
        })
    }

    pub fn spawn_with_config(
        self: &Arc<Self>,
        config: CausalLlmStateConfig,
    ) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState { model: self.clone(), nn_state, seq: vec![], config })
    }
}

/// Config of the state containing specific behavior requested
/// prompt chunking, sampling strategy, ...
#[derive(Debug, Clone, PartialEq)]
pub struct CausalLlmStateConfig {
    /// split long prompt into chunks of fixed number of tokens,
    /// reducing RAM usage
    prompt_chunk_size: Option<usize>,
}
impl Default for CausalLlmStateConfig {
    fn default() -> Self {
        Self { prompt_chunk_size: Some(512) }
    }
}

impl CausalLlmStateConfig {
    pub fn new(prompt_chunk_size: Option<usize>) -> TractResult<Self> {
        let conf = Self { prompt_chunk_size };
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
    pub config: CausalLlmStateConfig,
}

impl CausalLlmState {
    pub fn process_text(&mut self, prompt: &str) -> TractResult<u32> {
        self.process_tokens(&self.encode(prompt, true)?)
    }

    pub fn process_tokens(&mut self, tokens: &[u32]) -> TractResult<u32> {
        self.seq.extend(tokens);
        let mut input = tensor1(&tokens.iter().map(|t| *t as i64).collect_vec());
        input.insert_axis(0)?;
        let output = if let Some(s) = self.config.prompt_chunk_size {
            let mut pos = 0;
            while pos + s < tokens.len() {
                self.nn_state.run(tvec![input.slice(1, pos, pos + s)?.into_tvalue()])?;
                pos += s;
            }
            if pos > 0 {
                input = input.slice(1, pos, tokens.len())?;
            }
            self.nn_state.run(tvec![input.into_tvalue()])?
        } else {
            self.nn_state.run(tvec![input.into_tvalue()])?
        };
        let next_tok = output[0]
            .as_slice::<f32>()?
            .iter()
            .enumerate()
            .max_by_key(|(_ix, v)| FloatOrd(**v))
            .context("no tokens in output ?")?
            .0 as u32;
        Ok(next_tok)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.model.tokenizer
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> TractResult<String> {
        self.tokenizer().decode(&tokens, skip_special_tokens).map_err(|e| anyhow!(e))
    }

    pub fn encode(&self, text: &str, skip_special_tokens: bool) -> TractResult<Vec<u32>> {
        Ok(self
            .tokenizer()
            .encode(text, skip_special_tokens)
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
            config: self.config,
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
