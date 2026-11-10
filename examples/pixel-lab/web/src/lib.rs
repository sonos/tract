#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};

use tract::prelude::*;
use wasm_bindgen::prelude::*;

fn js_error(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

/// A person segmenter whose mask stays on the GPU for the page's compositor.
#[wasm_bindgen]
pub struct Segmenter {
    state: RefCell<tract::State>,
    /// RVM's four recurrent states, kept on the device between frames. Empty for a still segmenter.
    recurrent: RefCell<Vec<tract::Tensor>>,
    /// Which output is the alpha mask. RVM's foreground comes first, so alpha is output 1.
    alpha_index: usize,
    mask: tract::WgpuMask,
    graph: RefCell<String>,
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    /// Official RVM bytes, kept so a new factor can be built without the page sending them again.
    rvm_model: Option<Vec<u8>>,
    ratio: Cell<Option<f32>>,
    /// Networks already built for a factor. Switching back does not build again.
    prepared: RefCell<Vec<RvmReady>>,
}

/// One RVM network, specialized to a downsample factor. The plan is shared; each use gets new state.
#[derive(Clone)]
struct RvmReady {
    ratio: f32,
    runnable: tract::Runnable,
    shapes: [[usize; 4]; 4],
    graph: String,
}

fn same_ratio(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits() || (left - right).abs() < 1e-5
}

/// MobileNetV3 RVM keeps four encoder memories. `ratio` is the downsample applied to the
/// camera frame before that encoder, so the memories are sized from the downsampled frame.
fn rvm_states(width: u32, height: u32, ratio: f32) -> Result<[[usize; 4]; 4], JsValue> {
    if !(ratio > 0.0 && ratio <= 1.0) {
        return Err(JsValue::from_str("RVM scale must be in (0, 1]"));
    }
    let w = (width as f32 * ratio).round() as usize;
    let h = (height as f32 * ratio).round() as usize;
    if w == 0 || h == 0 {
        return Err(JsValue::from_str("RVM scale collapses the camera frame"));
    }
    let spec = [(16usize, 2usize), (20, 4), (40, 8), (64, 16)];
    Ok(spec.map(|(channels, stride)| [1, channels, h.div_ceil(stride), w.div_ceil(stride)]))
}

fn rvm_memory(shapes: &[[usize; 4]; 4]) -> Result<Vec<tract::Tensor>, JsValue> {
    shapes
        .iter()
        .map(|shape| {
            tract::Tensor::from_slice(shape, &vec![0f32; shape.iter().product()]).map_err(js_error)
        })
        .collect()
}

/// Build the RVM network for one downsample factor. Shapes follow that factor, so a different
/// factor is a different network. The bytes stay available for the next factor.
fn build_rvm(model: &[u8], width: u32, height: u32, ratio: f32) -> Result<RvmReady, JsValue> {
    let shapes = rvm_states(width, height, ratio)?;
    let options = OnnxOptions::new().ignore_value_info(true).constant_input(
        "downsample_ratio",
        DatumType::F32,
        &[1],
        &[f64::from(ratio)],
    );
    let mut inference = tract::onnx()
        .map_err(js_error)?
        .load_buffer_with_options(model, options)
        .map_err(js_error)?;
    let expected_in = ["src", "r1i", "r2i", "r3i", "r4i"];
    let input_count = inference.input_count().map_err(js_error)?;
    if input_count != expected_in.len() {
        return Err(JsValue::from_str(&format!(
            "RVM should have {} inputs after fixing its scale, found {input_count}",
            expected_in.len()
        )));
    }
    for (i, name) in expected_in.iter().enumerate() {
        let actual = inference.input_name(i).map_err(js_error)?;
        if actual != *name {
            return Err(JsValue::from_str(&format!("RVM input {i} is {actual}, expected {name}")));
        }
    }
    let declared = inference.input_fact(0).map_err(js_error)?.to_string();
    if !declared.ends_with("f32") {
        return Err(JsValue::from_str(&format!("RVM src is {declared}, expected f32")));
    }
    inference.set_input_fact(0, format!("1,3,{height},{width},f32").as_str()).map_err(js_error)?;
    for (i, shape) in shapes.iter().enumerate() {
        inference
            .set_input_fact(
                i + 1,
                format!("{},{},{},{},f32", shape[0], shape[1], shape[2], shape[3]).as_str(),
            )
            .map_err(js_error)?;
    }
    // Tract names an output after the node that produces it, not the ONNX name,
    // so the check is the shape. Order is the file's: foreground, alpha, then the
    // four recurrent states.
    let output_count = inference.output_count().map_err(js_error)?;
    if output_count != 6 {
        return Err(JsValue::from_str(&format!("RVM should have 6 outputs, found {output_count}")));
    }
    for i in 0..output_count {
        inference.set_output_fact(i, ()).map_err(js_error)?;
    }
    let built = inference.into_model().map_err(js_error)?;
    let mut expected =
        vec![format!("1,3,{height},{width},f32"), format!("1,1,{height},{width},f32")];
    for shape in &shapes {
        expected.push(format!("{},{},{},{},f32", shape[0], shape[1], shape[2], shape[3]));
    }
    for (i, shape) in expected.iter().enumerate() {
        let actual = built.output_fact(i).map_err(js_error)?.to_string();
        if actual != *shape {
            return Err(JsValue::from_str(&format!(
                "RVM output {i} is {actual}, expected {shape}"
            )));
        }
    }
    let runnable =
        tract::runtime_for_name("wgpu").map_err(js_error)?.prepare(built).map_err(js_error)?;
    let graph = runnable.cost_json().map_err(js_error)?;
    Ok(RvmReady { ratio, runnable, shapes, graph })
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
        Ok(Self {
            state: RefCell::new(state),
            recurrent: RefCell::new(Vec::new()),
            alpha_index: 0,
            mask,
            graph: RefCell::new(graph),
            width,
            height,
            mean,
            std,
            rvm_model: None,
            ratio: Cell::new(None),
            prepared: RefCell::new(Vec::new()),
        })
    }

    /// Robust Video Matting, MobileNetV3, the official fp32 or fp16 ONNX. `ratio` is the
    /// downsample factor for this run, in (0, 1]. Pass it at init and change it later with
    /// [`Segmenter::set_ratio`]. The four recurrent states stay on the GPU from frame to frame.
    /// An fp16 file keeps an fp16 body; the canvas and the mask stay fp32.
    pub async fn create_rvm(
        model: Vec<u8>,
        width: u32,
        height: u32,
        ratio: f32,
    ) -> Result<Segmenter, JsValue> {
        if width == 0 || height == 0 || width > 2048 || height > 2048 {
            return Err(JsValue::from_str("Invalid segmentation resolution"));
        }
        console_error_panic_hook::set_once();
        tract::wgpu::init().await.map_err(js_error)?;
        let ready = build_rvm(&model, width, height, ratio)?;
        let state = ready.runnable.spawn_state().map_err(js_error)?;
        let recurrent = rvm_memory(&ready.shapes)?;
        let mask = tract::WgpuMask::new(width, height).map_err(js_error)?;
        let graph = ready.graph.clone();
        Ok(Self {
            state: RefCell::new(state),
            recurrent: RefCell::new(recurrent),
            alpha_index: 1,
            mask,
            graph: RefCell::new(graph),
            width,
            height,
            mean: [0.0; 3],
            std: [1.0; 3],
            rvm_model: Some(model),
            ratio: Cell::new(Some(ratio)),
            prepared: RefCell::new(vec![ready]),
        })
    }

    /// Use a different downsample factor without restarting the camera.
    ///
    /// The first time a factor is used, its network is built. A factor already built in this
    /// session is swapped in. Recurrent memory is cleared, because its spatial size follows the
    /// factor. The mask texture stays the camera size. The same factor is left as it is.
    pub fn set_ratio(&self, ratio: f32) -> Result<f32, JsValue> {
        let Some(bytes) = self.rvm_model.as_deref() else {
            return Err(JsValue::from_str("R factor is only for RVM"));
        };
        if !(ratio > 0.0 && ratio <= 1.0) {
            return Err(JsValue::from_str("RVM scale must be in (0, 1]"));
        }
        if self.ratio.get().is_some_and(|current| same_ratio(current, ratio)) {
            return Ok(ratio);
        }
        let cached =
            self.prepared.borrow().iter().find(|item| same_ratio(item.ratio, ratio)).cloned();
        let (ready, store) = match cached {
            Some(ready) => (ready, false),
            None => (build_rvm(bytes, self.width, self.height, ratio)?, true),
        };
        *self.state.borrow_mut() = ready.runnable.spawn_state().map_err(js_error)?;
        *self.recurrent.borrow_mut() = rvm_memory(&ready.shapes)?;
        *self.graph.borrow_mut() = ready.graph.clone();
        self.ratio.set(Some(ready.ratio));
        let applied = ready.ratio;
        if store {
            self.prepared.borrow_mut().push(ready);
        }
        Ok(applied)
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
        self.graph.borrow().clone()
    }

    /// Submit GPU canvas ingest, inference and mask export, with no tensor readback.
    pub fn run(&self, canvas: web_sys::OffscreenCanvas) -> Result<(), JsValue> {
        if canvas.width() != self.width || canvas.height() != self.height {
            return Err(JsValue::from_str("Canvas does not match the model resolution"));
        }
        let mut inputs = vec![
            tract::Tensor::from_canvas_normalized(canvas, self.mean, self.std).map_err(js_error)?,
        ];
        inputs.extend(self.recurrent.borrow().iter().cloned());
        let outputs = self.state.borrow_mut().run(inputs).map_err(js_error)?;
        if !self.recurrent.borrow().is_empty() {
            *self.recurrent.borrow_mut() = outputs[self.alpha_index + 1..].to_vec();
        }
        self.mask.write(&outputs[self.alpha_index]).map_err(js_error)
    }
}
