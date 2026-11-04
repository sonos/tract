//! The worker-side API: stages read from tars cached in the origin private file system, images
//! generated on the browser's WebGPU.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;

use anyhow::Result;
use tract::prelude::*;
use wasm_bindgen::prelude::*;

use crate::pipeline::{Progress, TEXT_LEN, generate as run_pipeline, load};

#[wasm_bindgen]
extern "C" {
    /// Fills `dest` from byte `offset` of cached file `file`, synchronously: the worker reads
    /// through an OPFS sync access handle. Returns the bytes read.
    #[wasm_bindgen(js_name = bonsaiRead)]
    fn read_into(file: &str, offset: f64, dest: &mut [u8]) -> u32;
}

struct OpfsFile {
    file: String,
    offset: u64,
}

impl OpfsFile {
    fn new(file: String) -> OpfsFile {
        OpfsFile { file, offset: 0 }
    }
}

impl Read for OpfsFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = read_into(&self.file, self.offset as f64, buf) as usize;
        self.offset += n as u64;
        Ok(n)
    }
}

thread_local! {
    static TOKENIZER: RefCell<Option<tokenizers::Tokenizer>> = const { RefCell::new(None) };
    /// The models ready to run: each stage's, for the last size it was asked for.
    static STAGES: RefCell<HashMap<&'static str, (String, Runnable)>> = RefCell::new(HashMap::new());
    /// The last prompt's embeddings, reused on a Remix of the same prompt.
    static LAST_EMBEDDINGS: RefCell<Option<(String, Tensor)>> = const { RefCell::new(None) };
}

fn js(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Brings up the WebGPU device. Await it once before anything else.
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tract::wgpu::init().await.map_err(js)
}

#[wasm_bindgen]
pub fn set_tokenizer(tokenizer_json: &[u8]) -> Result<(), JsValue> {
    let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json).map_err(js)?;
    TOKENIZER.with_borrow_mut(|t| *t = Some(tokenizer));
    Ok(())
}

/// How many tokens `text` makes; -1 before the tokenizer is set.
#[wasm_bindgen]
pub fn count_tokens(text: &str) -> Result<i32, JsValue> {
    TOKENIZER.with_borrow(|t| match t {
        Some(t) => Ok(t.encode(text, false).map_err(js)?.get_ids().len() as i32),
        None => Ok(-1),
    })
}

/// Tokens a prompt can use before the end is cut: the transformer's, less the chat template's 12.
#[wasm_bindgen]
pub fn prompt_token_limit() -> u32 {
    TEXT_LEN as u32 - 12
}

/// A square RGBA image. `on_progress(stage, step, steps)` is called as it goes.
#[wasm_bindgen]
pub async fn generate(
    prompt: String,
    steps: usize,
    seed: f64,
    width: usize,
    height: usize,
    on_progress: js_sys::Function,
    on_preview: js_sys::Function,
) -> Result<Vec<u8>, JsValue> {
    let tokenizer =
        TOKENIZER.with_borrow(|t| t.clone()).ok_or_else(|| js("the tokenizer is not loaded"))?;
    let size = format!("{height}x{width}");
    let mut stage = |name: &'static str| -> Result<Runnable> {
        let wanted = if name == "text_encoder" { "any" } else { &size };
        let ready = STAGES.with_borrow_mut(|s| match s.get(name) {
            Some((of, model)) if of == wanted => Some(model.clone()),
            _ => {
                s.remove(name);
                None
            }
        });
        if let Some(model) = ready {
            return Ok(model);
        }
        let model = if name == "text_encoder" {
            load(&mut OpfsFile::new("text_encoder.nnef.tar".to_string()))?
        } else {
            load(
                &mut OpfsFile::new(format!("{name}-{size}.tar"))
                    .chain(OpfsFile::new(format!("{name}.weights.tar"))),
            )?
        };
        STAGES.with_borrow_mut(|s| s.insert(name, (wanted.to_string(), model.clone())));
        Ok(model)
    };
    let mut report = |p: Progress| {
        if let Progress::Preview(step, rgba) = p {
            let rgba = js_sys::Uint8Array::from(&rgba[..]);
            let _ = on_preview.call3(
                &JsValue::NULL,
                &rgba,
                &JsValue::from(step as u32),
                &JsValue::from((width / 16) as u32),
            );
            return;
        }
        let (stage, step) = match p {
            Progress::Preview(..) => unreachable!(),
            Progress::Loading(what) => (format!("loading {what}"), 0),
            Progress::Encoding => ("encoding".to_string(), 0),
            Progress::Step(i) => ("denoising".to_string(), i),
            Progress::Decoding => ("decoding".to_string(), steps),
        };
        let _ = on_progress.call3(
            &JsValue::NULL,
            &JsValue::from_str(&stage),
            &JsValue::from(step as u32),
            &JsValue::from(steps as u32),
        );
    };
    let mut cached = LAST_EMBEDDINGS.with_borrow_mut(Option::take);
    let result = run_pipeline(
        &tokenizer,
        &prompt,
        steps,
        seed as u64,
        width,
        height,
        &mut stage,
        &mut report,
        &mut cached,
    )
    .await;
    LAST_EMBEDDINGS.with_borrow_mut(|c| *c = cached);
    result.map_err(js)
}
