use std::path::Path;

use float_ord::FloatOrd;
use tokenizers::Tokenizer;
use tract_nnef::internal::{anyhow, TractErrorContext};
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_transformers::ops::dyn_kv_cache::DynKeyValueCacheState;
use tract_transformers::WithTractTransformers;

#[derive(Clone, Debug)]
pub struct CausalLlmModel {
    pub tokenizer: Tokenizer,
    pub nn: Arc<TypedRunnableModel<TypedModel>>,
}

impl CausalLlmModel {
    pub fn from_paths(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow!(e))?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut nn = nnef.model_for_path(nn.as_ref())?.into_decluttered()?;

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;
        nn.transform(&*transform)?;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            use crate::tract_core::transform::ModelTransform;
            use std::str::FromStr;
            tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
        }
        nn.optimize()?;
        let nn = nn.into_runnable()?;
        Ok(Arc::new(CausalLlmModel { tokenizer, nn: Arc::new(nn) }))
    }

    pub fn spawn(self: &Arc<Self>) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState { model: self.clone(), nn_state, seq: vec![] })
    }
}

#[derive(Debug)]
pub struct CausalLlmState {
    pub model: Arc<CausalLlmModel>,
    pub nn_state: TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
    pub seq: Vec<u32>,
}

impl CausalLlmState {
    pub fn process_text(&mut self, prompt: &str) -> TractResult<u32> {
        self.process_tokens(&self.encode(prompt, true)?)
    }

    pub fn process_tokens(&mut self, tokens: &[u32]) -> TractResult<u32> {
        self.seq.extend(tokens);
        let mut input = tensor1(&tokens.iter().map(|t| *t as i64).collect_vec());
        input.insert_axis(0)?;
        let output = self.nn_state.run(tvec![input.into_tvalue()])?;
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
        FrozenCausalLlmState { model: self.model, nn_state: self.nn_state.freeze(), seq: self.seq }
    }

    pub fn truncate(&mut self, len: usize) -> TractResult<()> {
        self.seq.truncate(len);
        for s in &mut self.nn_state.states {
            if let Some(s) = s.as_mut().and_then(|s| s.downcast_mut::<DynKeyValueCacheState>()) {
                s.truncate(len)?;
            }
        }
        Ok(())
    }
}

pub struct FrozenCausalLlmState {
    pub model: Arc<CausalLlmModel>,
    pub nn_state: TypedFrozenSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
    pub seq: Vec<u32>,
}

impl FrozenCausalLlmState {
    pub fn unfreeze(self) -> CausalLlmState {
        CausalLlmState { model: self.model, nn_state: self.nn_state.unfreeze(), seq: self.seq }
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
