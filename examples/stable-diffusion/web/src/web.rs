//! The worker-side API: models loaded from weights cached in the origin private file system,
//! images generated on the browser's WebGPU.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;
use tract::ReadAt;
use tract::prelude::*;
use wasm_bindgen::prelude::*;

use crate::pipeline::{Pipeline, Progress, Request, load};
use crate::scheduler::Sampler;
use crate::tokenizer::{PROMPT_TOKENS, Tokenizer};

#[wasm_bindgen]
extern "C" {
    /// Fills `dest` from byte `offset` of weight file `file`, synchronously: the worker reads
    /// through an OPFS sync access handle.
    #[wasm_bindgen(js_name = sd15ReadInto)]
    fn read_into(file: u32, offset: f64, dest: &mut [u8]) -> bool;
}

/// A weight file the worker holds open.
struct OpfsFile(u32);

impl ReadAt for OpfsFile {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        if read_into(self.0, offset as f64, buf) {
            Ok(())
        } else {
            Err(std::io::ErrorKind::UnexpectedEof.into())
        }
    }
}

#[derive(Default)]
struct State {
    tokenizer: Option<Tokenizer>,
    models: HashMap<String, Runnable>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn js(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Brings up the WebGPU device. Await it once before loading anything.
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tract::wgpu::init().await.map_err(js)
}

#[wasm_bindgen]
pub fn set_tokenizer(vocab_json: &str, merges_txt: &str) -> Result<(), JsValue> {
    let tokenizer = Tokenizer::new(vocab_json, merges_txt).map_err(js)?;
    STATE.with_borrow_mut(|s| s.tokenizer = Some(tokenizer));
    Ok(())
}

/// How many prompt tokens `text` makes, against [`PROMPT_TOKENS`] the encoder keeps; -1 before
/// the tokenizer is set.
#[wasm_bindgen]
pub fn count_tokens(text: &str) -> Result<i32, JsValue> {
    STATE.with_borrow(|s| match &s.tokenizer {
        Some(t) => Ok(t.tokens(text).map_err(js)?.len() as i32),
        None => Ok(-1),
    })
}

/// Prompt tokens the text encoder sees; the rest is cut.
#[wasm_bindgen]
pub fn prompt_token_limit() -> u32 {
    PROMPT_TOKENS as u32
}

/// Loads `component` from its NNEF tar and weight file `file`, onto the GPU.
#[wasm_bindgen]
pub fn load_component(component: &str, graph_tar: &[u8], file: u32) -> Result<(), JsValue> {
    let runnable = load(graph_tar, OpfsFile(file)).map_err(js)?;
    STATE.with_borrow_mut(|s| s.models.insert(component.to_string(), runnable));
    Ok(())
}

/// A 512x512 RGBA image. `on_progress(stage, step, steps)` is called as it goes.
#[wasm_bindgen]
pub async fn generate(
    prompt: String,
    negative: String,
    steps: usize,
    seed: f64,
    guidance: f32,
    sampler: String,
    on_progress: js_sys::Function,
) -> Result<Vec<u8>, JsValue> {
    let sampler =
        Sampler::parse(&sampler).ok_or_else(|| js(format!("unknown sampler {sampler}")))?;
    let pipeline = STATE.with_borrow(|s| -> Result<Pipeline> {
        let model =
            |name: &str| s.models.get(name).cloned().ok_or_else(|| anyhow::anyhow!("no {name}"));
        Ok(Pipeline {
            tokenizer: s.tokenizer.clone().ok_or_else(|| anyhow::anyhow!("no tokenizer"))?,
            text_encoder: model("text_encoder")?,
            unet: model("unet")?,
            vae: model("vae_decoder")?,
        })
    });
    let pipeline = pipeline.map_err(js)?;
    let request = Request {
        prompt: &prompt,
        negative: &negative,
        steps,
        seed: seed as u64,
        guidance,
        sampler,
    };
    let mut report = |p: Progress| {
        let (stage, step) = match p {
            Progress::Encoding => ("encoding", 0),
            Progress::Step(i, _) => ("denoising", i),
            Progress::Decoding => ("decoding", steps),
        };
        let _ = on_progress.call3(
            &JsValue::NULL,
            &JsValue::from_str(stage),
            &JsValue::from(step as u32),
            &JsValue::from(steps as u32),
        );
    };
    pipeline.generate(&request, &mut report).await.map_err(js)
}
