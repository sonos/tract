use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use tract::prelude::*;

/// The sample rate the model takes.
pub const SAMPLE_RATE: u32 = 48000;

/// The wake word detector: the pulsed graph of resampler, featurizer and wavenet, fed audio in
/// [-1, 1] at [`SAMPLE_RATE`].
///
/// [`Detector::push`] returns one score per 10 ms frame once the model has a full window: the
/// keyword logit minus the background one, for the window ending at that frame.
pub struct Detector {
    state: tract::State,
    pulse: usize,
    pending: Vec<f32>,
    warm_up: usize,
}

impl Detector {
    pub fn new(model: &[u8]) -> Result<Detector> {
        let mut nnef = tract::nnef()?;
        nnef.enable_pulse()?;
        let model = nnef.load_buffer(model)?;
        let input = model.input_fact(0)?.to_string();
        let pulse = input
            .strip_suffix(",f32")
            .and_then(|pulse| pulse.parse().ok())
            .with_context(|| format!("Expected a pulsed 1D f32 input, got {input}"))?;
        let delay = model.property("pulse.delay")?;
        let warm_up = *delay.as_slice::<i64>()?.first().context("Empty pulse.delay")? as usize;
        let state = model.into_runnable()?.spawn_state()?;
        Ok(Detector { state, pulse, pending: vec![], warm_up })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        self.pending.extend_from_slice(samples);
        let mut scores = vec![];
        while self.pending.len() >= self.pulse {
            let chunk = tract::Tensor::from_slice(&[self.pulse], &self.pending[..self.pulse])?;
            self.pending.drain(..self.pulse);
            let outputs = self.state.run([chunk])?;
            let logits = outputs[0].as_slice::<f32>()?;
            ensure!(logits.len() % 2 == 0, "Expected two logits per frame");
            scores.extend(logits.chunks(2).map(|frame| frame[1] - frame[0]));
        }
        let skip = self.warm_up.min(scores.len());
        self.warm_up -= skip;
        Ok(scores.split_off(skip))
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;

    fn js_error(error: anyhow::Error) -> JsValue {
        JsValue::from_str(&format!("{error:#}"))
    }

    /// [`super::Detector`] for the page's worker.
    #[wasm_bindgen]
    pub struct Detector(RefCell<super::Detector>);

    #[wasm_bindgen]
    impl Detector {
        #[wasm_bindgen(constructor)]
        pub fn new(model: &[u8]) -> Result<Detector, JsValue> {
            console_error_panic_hook::set_once();
            Ok(Detector(RefCell::new(super::Detector::new(model).map_err(js_error)?)))
        }

        /// Scores for the 10 ms frames these samples complete.
        pub fn push(&self, samples: &[f32]) -> Result<Vec<f32>, JsValue> {
            self.0.borrow_mut().push(samples).map_err(js_error)
        }
    }
}
