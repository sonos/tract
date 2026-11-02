//! The worker-side API: the model loaded from weights cached in the origin private file system,
//! one stream transcribed at a time on the browser's WebGPU.

use std::cell::RefCell;

use anyhow::Result;
use tract::ReadAt;
use wasm_bindgen::prelude::*;

use crate::{Decimator, Graphs, Models, Transcriber};

#[wasm_bindgen]
extern "C" {
    /// Fills `dest` from byte `offset` of the weight file, synchronously: the worker reads
    /// through an OPFS sync access handle.
    #[wasm_bindgen(js_name = nemotronReadInto)]
    fn read_into(offset: f64, dest: &mut [u8]) -> bool;
}

/// The weight file the worker holds open.
struct OpfsFile;

impl ReadAt for OpfsFile {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        if read_into(offset as f64, buf) {
            Ok(())
        } else {
            Err(std::io::ErrorKind::UnexpectedEof.into())
        }
    }
}

struct Stream {
    decimator: Decimator,
    transcriber: Transcriber,
}

thread_local! {
    static MODELS: RefCell<Option<Models>> = const { RefCell::new(None) };
    static STREAM: RefCell<Option<Stream>> = const { RefCell::new(None) };
}

fn js(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Brings up the WebGPU device. Await it once before loading.
#[wasm_bindgen]
pub async fn init() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    tract::wgpu::init().await.map_err(js)
}

/// Loads the four graphs, the vocabulary and, as the runtimes need them, the weights.
#[wasm_bindgen]
pub fn load(
    preprocessor: &[u8],
    encoder: &[u8],
    decoder: &[u8],
    joint: &[u8],
    tokenizer_json: &str,
) -> Result<(), JsValue> {
    let graphs = Graphs { preprocessor, encoder, decoder, joint };
    let models = Models::load(graphs, tokenizer_json, || Ok(OpfsFile)).map_err(js)?;
    MODELS.with_borrow_mut(|m| *m = Some(models));
    Ok(())
}

/// Starts a new stream, of audio at `sample_rate`, in the language of `lang_id`.
#[wasm_bindgen]
pub fn start(lang_id: i32, sample_rate: u32) -> Result<(), JsValue> {
    let stream = MODELS.with_borrow(|m| -> Result<Stream> {
        let models = m.as_ref().ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        Ok(Stream {
            decimator: Decimator::new(sample_rate as usize)?,
            transcriber: Transcriber::new(models, lang_id as i64)?,
        })
    });
    let stream = stream.map_err(js)?;
    STREAM.with_borrow_mut(|s| *s = Some(stream));
    Ok(())
}

/// Feeds the stream (or, with `last`, ends it) and returns its transcript so far.
#[wasm_bindgen]
pub async fn push(samples: Vec<f32>, last: bool) -> Result<String, JsValue> {
    let mut stream = STREAM.with_borrow_mut(Option::take).ok_or_else(|| js("no stream started"))?;
    let resampled = stream.decimator.push(&samples);
    let fed = async {
        stream.transcriber.push(&resampled).await?;
        if last {
            stream.transcriber.finish().await?;
        }
        anyhow::Ok(())
    }
    .await;
    let text = stream.transcriber.text();
    STREAM.with_borrow_mut(|s| {
        if s.is_none() {
            *s = Some(stream)
        }
    });
    fed.map_err(js)?;
    Ok(text)
}
