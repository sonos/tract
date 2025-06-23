use std::error::Error;
use std::path::Path;
use std::str::FromStr;

use crate::tract_core::transform::ModelTransform;
use float_ord::FloatOrd;
use tokenizers::Tokenizer;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::prelude::*;
use tract_transformers::WithTractTransformers;

#[derive(Clone, Debug)]
pub struct CausalLlmModel {
    pub tokenizer: Tokenizer,
    pub nn: Arc<TypedRunnableModel<TypedModel>>,
}

impl CausalLlmModel {
    fn from_paths(
        tokenizer: impl AsRef<Path>,
        nn: impl AsRef<Path>,
    ) -> TractResult<Arc<CausalLlmModel>> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).unwrap();

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut nn = nnef.model_for_path(nn.as_ref())?.into_decluttered()?;

        let transform = nnef.get_transform("transformers-detect-all").unwrap().unwrap();
        nn.transform(&*transform)?;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            tract_metal::MetalTransform::from_str("")?.transform(&mut nn)?;
        }
        nn.optimize()?;
        let nn = nn.into_runnable()?;
        Ok(Arc::new(CausalLlmModel { tokenizer, nn: Arc::new(nn) }))
    }

    fn spawn(self: &Arc<Self>) -> TractResult<CausalLlmState> {
        let nn_state = SimpleState::new(self.nn.clone())?;
        Ok(CausalLlmState { model: self.clone(), nn_state, seq: vec![] })
    }
}

#[derive(Debug)]
struct CausalLlmState {
    pub model: Arc<CausalLlmModel>,
    pub nn_state: TypedSimpleState<TypedModel, Arc<TypedRunnableModel<TypedModel>>>,
    pub seq: Vec<u32>,
}

impl CausalLlmState {
    pub fn process_text(&mut self, prompt: &str) -> TractResult<u32> {
        let seq = self.model.tokenizer.encode(prompt, false).unwrap();
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
            .unwrap()
            .0 as u32;
        Ok(next_tok)
    }

    pub fn to_words(&self, tokens: &[u32]) -> TractResult<String> {
        Ok(self.model.tokenizer.decode(&tokens, true).unwrap())
    }
}

fn main() -> TractResult<()> {
    let tokenizer_path = "OhanaTract-llama3_2_1B_f32/tokenizer/tokenizer.json";
    let nn_path = "OhanaTract-llama3_2_1B_f32/model/model.nnef.tgz";
    let model = CausalLlmModel::from_paths(tokenizer_path, nn_path)?;
    let mut state = model.spawn()?;

    // prompt processing
    let mut current_token = state.process_text("Paul McCartney is")?;

    // text gen loop
    loop {
        current_token = state.process_tokens(&[current_token])?;
        if state.seq.len() == 128 {
            break;
        }
    }
    println!("{}", state.to_words(&state.seq)?);

    Ok(())
}
