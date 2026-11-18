use wasm_bindgen::prelude::*;

use crate::engine::Graphs;
use crate::engine::Request;
use crate::engine::Tts;
use crate::engine::Voice;
use crate::text;

fn js_error(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// The page's handle on one Supertonic release prepared on one runtime.
#[wasm_bindgen]
pub struct Engine {
    tts: Tts,
}

#[wasm_bindgen]
impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub async fn create(
        runtime: String,
        duration_predictor: Vec<u8>,
        text_encoder: Vec<u8>,
        vector_estimator: Vec<u8>,
        vocoder: Vec<u8>,
        tts_json: Vec<u8>,
        unicode_indexer_json: Vec<u8>,
    ) -> Result<Engine, JsValue> {
        console_error_panic_hook::set_once();
        let graphs = Graphs {
            duration_predictor,
            text_encoder,
            vector_estimator,
            vocoder,
            tts_json,
            unicode_indexer_json,
        };
        let tts = Tts::create(&runtime, &graphs).await.map_err(js_error)?;
        Ok(Engine { tts })
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.tts.sample_rate() as u32
    }

    /// Speaks one chunk of text with a voice style (the JSON file Supertone publishes).
    pub async fn speak(
        &self,
        voice: Vec<u8>,
        text: String,
        lang: String,
        steps: u32,
        speed: f32,
        seed: u32,
    ) -> Result<Vec<f32>, JsValue> {
        let voice = Voice::from_json(&voice).map_err(js_error)?;
        let request =
            Request { text: &text, lang: &lang, steps: steps as usize, speed, seed: seed as u64 };
        self.tts.speak(&voice, &request).await.map_err(js_error)
    }
}

/// Splits text into the chunks [`Engine::speak`] reads well.
#[wasm_bindgen]
pub fn chunks(text: String, lang: String) -> Vec<String> {
    text::chunks(&text, &lang)
}
