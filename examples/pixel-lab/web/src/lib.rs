#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use tract::prelude::*;
use wasm_bindgen::prelude::*;

fn js_error(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// A person segmenter whose mask stays on the GPU for the page's compositor.
#[wasm_bindgen]
pub struct Segmenter {
    state: RefCell<tract::State>,
    mask: tract::WgpuMask,
    graph: String,
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

#[wasm_bindgen]
impl Segmenter {
    /// Prepare a single-input RGB NCHW ONNX segmenter at `width` x `height`.
    /// Each input channel is fed as `(value - mean) / std` from RGB in [0, 1];
    /// `mean` and `std` hold one value per channel.
    pub async fn create(
        model: Vec<u8>,
        width: u32,
        height: u32,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Segmenter, JsValue> {
        if width == 0 || height == 0 || width > 2048 || height > 2048 {
            return Err(JsValue::from_str("Invalid segmentation resolution"));
        }
        let (Ok(mean), Ok(std)) = (<[f32; 3]>::try_from(mean), <[f32; 3]>::try_from(std)) else {
            return Err(JsValue::from_str("Expected three means and three deviations"));
        };
        console_error_panic_hook::set_once();
        tract::wgpu::init().await.map_err(js_error)?;
        let options = OnnxOptions::new().ignore_value_info(true);
        let mut inference = tract::onnx()
            .map_err(js_error)?
            .load_buffer_with_options(&model, options)
            .map_err(js_error)?;
        inference
            .set_input_fact(0, format!("1,3,{height},{width},f32").as_str())
            .map_err(js_error)?;
        inference.set_output_fact(0, ()).map_err(js_error)?;
        let model = inference.into_model().map_err(js_error)?;
        if model.input_count().map_err(js_error)? != 1
            || model.output_count().map_err(js_error)? != 1
        {
            return Err(JsValue::from_str(
                "Expected a single RGB input and single alpha-mask output",
            ));
        }
        let output = model.output_fact(0).map_err(js_error)?.to_string();
        if output != format!("1,1,{height},{width},f32") {
            return Err(JsValue::from_str(&format!("Unexpected mask shape {output}")));
        }
        let runnable =
            tract::runtime_for_name("wgpu").map_err(js_error)?.prepare(model).map_err(js_error)?;
        let graph = runnable.cost_json().map_err(js_error)?;
        let state = runnable.spawn_state().map_err(js_error)?;
        let mask = tract::WgpuMask::new(width, height).map_err(js_error)?;
        Ok(Self { state: RefCell::new(state), mask, graph, width, height, mean, std })
    }

    /// The shared WebGPU device for capture and compositing.
    pub fn device(&self) -> Result<JsValue, JsValue> {
        tract::wgpu_device().map_err(js_error)
    }

    /// The GPU-only output mask texture.
    pub fn mask_texture(&self) -> JsValue {
        self.mask.texture()
    }

    /// Prepared model nodes for the page's GPU coverage audit.
    pub fn graph(&self) -> String {
        self.graph.clone()
    }

    /// Submit GPU canvas ingest, inference and mask export, with no tensor readback.
    pub fn run(&self, canvas: web_sys::OffscreenCanvas) -> Result<(), JsValue> {
        if canvas.width() != self.width || canvas.height() != self.height {
            return Err(JsValue::from_str("Canvas does not match the model resolution"));
        }
        let input =
            tract::Tensor::from_canvas_normalized(canvas, self.mean, self.std).map_err(js_error)?;
        let outputs = self.state.borrow_mut().run([input]).map_err(js_error)?;
        self.mask.write(&outputs[0]).map_err(js_error)
    }
}
