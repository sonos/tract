use std::path::Path;
use std::str::FromStr;

use crate::tract_core::transform::ModelTransform;
use float_ord::FloatOrd;
use tokenizers::Tokenizer;
use tract_nnef::internal::{anyhow, TractErrorContext};
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
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
        let seq = self.model.tokenizer.encode(prompt, false).map_err(|e| anyhow!(e))?;
        self.process_tokens(&seq.get_ids())
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

    pub fn to_words(&self, tokens: &[u32]) -> TractResult<String> {
        self.model.tokenizer.decode(&tokens, true).map_err(|e| anyhow!(e))
    }
}
