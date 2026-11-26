use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use std::cell::RefCell;

use tract_core::prelude::*;
use tract_core::runtime::State;
use tract_core::runtime::runtime_for_name;
use tract_gpu::tensor::DeviceTensorExt;

/// Side of the palm detector's square input, in pixels.
pub const DETECTOR_SIZE: usize = 192;
/// Side of the landmark model's square input, in pixels.
pub const LANDMARKS_SIZE: usize = 224;
/// Anchors the palm detector scores, each with a box and seven keypoints.
pub const ANCHORS: usize = 2016;
/// Values per anchor in the detector's regression output: box (4) and 7 keypoints (2 each).
pub const COORDS: usize = 18;

/// The two files a MediaPipe `hand_landmarker.task` bundle holds. The bundle is a zip whose
/// entries are stored, not deflated, so they are read in place.
pub fn task_entries(task: &[u8]) -> Result<(&[u8], &[u8])> {
    let mut detector = None;
    let mut landmarks = None;
    // Entries are padded for alignment, so each local header is looked for a few bytes on.
    let signature = [0x50, 0x4b, 0x03, 0x04];
    let next_header = |from: usize| {
        task.get(from..)
            .and_then(|rest| rest.windows(4).take(64).position(|w| w == signature))
            .map(|skip| from + skip)
    };
    let u16_at = |at: usize| u16::from_le_bytes([task[at], task[at + 1]]) as usize;
    let u32_at = |at: usize| u32::from_le_bytes(task[at..at + 4].try_into().unwrap()) as usize;
    let mut cursor = next_header(0).context("The hand model bundle is not a zip")?;
    while let Some(at) = next_header(cursor).filter(|at| at + 30 <= task.len()) {
        let method = u16_at(at + 8);
        let size = u32_at(at + 18);
        let name_len = u16_at(at + 26);
        let extra_len = u16_at(at + 28);
        let name_at = at + 30;
        let data_at = name_at + name_len + extra_len;
        ensure!(data_at + size <= task.len(), "The hand model bundle is truncated");
        let name = std::str::from_utf8(&task[name_at..name_at + name_len])?;
        if method != 0 {
            bail!("{name} is compressed in the hand model bundle; expected a stored entry");
        }
        let data = &task[data_at..data_at + size];
        match name {
            "hand_detector.tflite" => detector = Some(data),
            "hand_landmarks_detector.tflite" => landmarks = Some(data),
            _ => (),
        }
        cursor = data_at + size;
    }
    Ok((
        detector.context("No hand_detector.tflite in the hand model bundle")?,
        landmarks.context("No hand_landmarks_detector.tflite in the hand model bundle")?,
    ))
}

/// Where tract runs the two models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// tract's CPU kernels: simd128 in a browser.
    Cpu,
    /// tract's wgpu runtime: WebGPU in a browser, Metal, Vulkan or DX12 natively.
    Wgpu,
}

impl std::str::FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(name: &str) -> Result<Backend> {
        match name {
            "cpu" => Ok(Backend::Cpu),
            "wgpu" | "gpu" => Ok(Backend::Wgpu),
            _ => bail!("Unknown backend {name}, expected cpu or wgpu"),
        }
    }
}

enum Engine {
    Cpu(Arc<TypedSimplePlan>),
    Wgpu(RefCell<Box<dyn State>>),
}

fn load(bytes: &[u8], side: usize, backend: Backend) -> Result<Engine> {
    let model = tract_tflite::tflite().model_for_read(&mut &*bytes)?;
    let shape = model.input_fact(0)?.shape.as_concrete().map(|s| s.to_vec());
    ensure!(
        shape.as_deref() == Some(&[1, side, side, 3][..]),
        "Expected a 1x{side}x{side}x3 input, got {shape:?}"
    );
    Ok(match backend {
        Backend::Cpu => Engine::Cpu(model.into_optimized()?.into_runnable()?),
        Backend::Wgpu => {
            let runtime = runtime_for_name("wgpu")?.context("tract was built without wgpu")?;
            let runnable = runtime.prepare(model.into_decluttered()?)?;
            Engine::Wgpu(RefCell::new(runnable.spawn()?))
        }
    })
}

/// Runs the model and brings every output back to host memory. On wasm32 the wgpu runtime's
/// outputs stay on the GPU until downloaded, which a browser can only do asynchronously.
async fn run(engine: &Engine, side: usize, rgb: &[f32]) -> Result<Vec<Vec<f32>>> {
    ensure!(rgb.len() == side * side * 3, "Expected {side}x{side} RGB, got {} values", rgb.len());
    let input: TValue = Tensor::from_shape(&[1, side, side, 3], rgb)?.into();
    let outputs = match engine {
        Engine::Cpu(plan) => plan.run(tvec!(input))?,
        Engine::Wgpu(state) => state.borrow_mut().run(tvec!(input))?,
    };
    let mut host = Vec::with_capacity(outputs.len());
    for output in &outputs {
        let values = match output.as_device_tensor() {
            Some(device) => tract_wgpu::to_host_async(device).await?.into_arc_tensor(),
            None => output.clone().into_arc_tensor(),
        };
        host.push(values.to_plain_array_view::<f32>()?.iter().copied().collect());
    }
    Ok(host)
}

/// MediaPipe's palm detector and hand landmark model, run by tract.
///
/// Both take RGB in [0, 1], row-major, interleaved. The anchors, the crop and the projection back
/// to the camera picture are the page's job. With [`Backend::Wgpu`] in a browser, the WebGPU
/// device must be up first: see [`init_wgpu`].
pub struct Hands {
    detector: Engine,
    landmarks: Engine,
}

/// Brings up the WebGPU device tract's wgpu runtime uses. A browser can only request it
/// asynchronously, so await this once before building [`Hands`] on [`Backend::Wgpu`].
pub async fn init_wgpu() -> Result<()> {
    tract_wgpu::wgpu_context_async().await?;
    Ok(())
}

impl Hands {
    pub fn new(detector: &[u8], landmarks: &[u8], backend: Backend) -> Result<Hands> {
        Ok(Hands {
            detector: load(detector, DETECTOR_SIZE, backend)
                .context("Loading the palm detector")?,
            landmarks: load(landmarks, LANDMARKS_SIZE, backend)
                .context("Loading the hand landmark model")?,
        })
    }

    pub fn from_task(task: &[u8], backend: Backend) -> Result<Hands> {
        let (detector, landmarks) = task_entries(task)?;
        Hands::new(detector, landmarks, backend)
    }

    /// Palm detection on a 192x192 picture: [`ANCHORS`] x [`COORDS`] box and keypoint offsets in
    /// input pixels, then [`ANCHORS`] score logits.
    pub async fn detect(&self, rgb: &[f32]) -> Result<Vec<f32>> {
        let outputs = run(&self.detector, DETECTOR_SIZE, rgb).await?;
        ensure!(outputs.len() == 2, "Expected 2 detector outputs, got {}", outputs.len());
        ensure!(outputs[0].len() == ANCHORS * COORDS, "Unexpected box output length");
        ensure!(outputs[1].len() == ANCHORS, "Unexpected score output length");
        Ok(outputs.concat())
    }

    /// Landmarks on a 224x224 crop: 21 x (x, y, z) in crop pixels, then the hand presence score
    /// and the handedness score, both already in [0, 1].
    pub async fn landmarks(&self, rgb: &[f32]) -> Result<Vec<f32>> {
        let outputs = run(&self.landmarks, LANDMARKS_SIZE, rgb).await?;
        ensure!(outputs.len() == 4, "Expected 4 landmark outputs, got {}", outputs.len());
        ensure!(outputs[0].len() == 63, "Unexpected landmark output length");
        Ok(outputs[..3].concat())
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::rc::Rc;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::future_to_promise;

    fn js_error(error: anyhow::Error) -> JsValue {
        JsValue::from_str(&format!("{error:#}"))
    }

    /// [`super::Hands`] for the page's worker. Every call returns a promise: on WebGPU the
    /// outputs are downloaded asynchronously.
    #[wasm_bindgen]
    pub struct Hands(Rc<super::Hands>);

    #[wasm_bindgen]
    impl Hands {
        /// Loads both models from the bytes of a `hand_landmarker.task` bundle, on `"cpu"` or
        /// `"wgpu"`.
        pub async fn create(task: Vec<u8>, backend: String) -> Result<Hands, JsValue> {
            console_error_panic_hook::set_once();
            let backend: super::Backend = backend.parse().map_err(js_error)?;
            if backend == super::Backend::Wgpu {
                super::init_wgpu().await.map_err(js_error)?;
            }
            Ok(Hands(Rc::new(super::Hands::from_task(&task, backend).map_err(js_error)?)))
        }

        pub fn detect(&self, rgb: Vec<f32>) -> js_sys::Promise {
            let hands = self.0.clone();
            future_to_promise(async move {
                let out = hands.detect(&rgb).await.map_err(js_error)?;
                Ok(js_sys::Float32Array::from(&out[..]).into())
            })
        }

        pub fn landmarks(&self, rgb: Vec<f32>) -> js_sys::Promise {
            let hands = self.0.clone();
            future_to_promise(async move {
                let out = hands.landmarks(&rgb).await.map_err(js_error)?;
                Ok(js_sys::Float32Array::from(&out[..]).into())
            })
        }
    }
}
