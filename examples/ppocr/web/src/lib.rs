#![cfg(target_arch = "wasm32")]

mod engine;

use engine::Engine as Core;
use wasm_bindgen::prelude::*;

fn js_error(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}

#[wasm_bindgen]
pub struct Map {
    data: Vec<f32>,
    height: u32,
    width: u32,
}

#[wasm_bindgen]
impl Map {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
}

#[wasm_bindgen]
pub struct Sequence {
    data: Vec<f32>,
    time: u32,
    classes: u32,
}

#[wasm_bindgen]
impl Sequence {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> u32 {
        self.time
    }

    #[wasm_bindgen(getter)]
    pub fn classes(&self) -> u32 {
        self.classes
    }
}

#[wasm_bindgen]
pub struct Engine {
    inner: Core,
}

#[wasm_bindgen]
impl Engine {
    pub async fn create(runtime: String, det: Vec<u8>, rec: Vec<u8>) -> Result<Engine, JsValue> {
        console_error_panic_hook::set_once();
        let inner = Core::create(&runtime, det, rec).await.map_err(js_error)?;
        Ok(Engine { inner })
    }

    pub async fn detect(
        &mut self,
        data: Vec<f32>,
        height: u32,
        width: u32,
    ) -> Result<Map, JsValue> {
        let out =
            self.inner.detect(&data, height as usize, width as usize).await.map_err(js_error)?;
        let (height, width) = map_hw(&out.dims).map_err(js_error)?;
        Ok(Map { data: out.data, height, width })
    }

    pub async fn recognize(&mut self, data: Vec<f32>, width: u32) -> Result<Sequence, JsValue> {
        let out = self.inner.recognize(&data, width as usize).await.map_err(js_error)?;
        let (time, classes) = sequence_tc(&out.dims).map_err(js_error)?;
        Ok(Sequence { data: out.data, time, classes })
    }
}

fn map_hw(dims: &[usize]) -> anyhow::Result<(u32, u32)> {
    anyhow::ensure!(dims.len() == 4, "detection map has rank {}", dims.len());
    Ok((dims[2] as u32, dims[3] as u32))
}

fn sequence_tc(dims: &[usize]) -> anyhow::Result<(u32, u32)> {
    anyhow::ensure!(dims.len() == 3, "recognition output has rank {}", dims.len());
    Ok((dims[1] as u32, dims[2] as u32))
}
