use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use tract::prelude::*;

pub struct TensorOut {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
}

/// Detection and recognition for one PP-OCRv6 tier, on one tract runtime.
///
/// Each concrete input size is prepared once. Paddle's declared shapes are
/// discarded: they do not match the graph, which is also how the benches load
/// these models.
pub struct Engine {
    runtime: String,
    det_bytes: Vec<u8>,
    rec_bytes: Vec<u8>,
    det: HashMap<(usize, usize), Runnable>,
    rec: HashMap<usize, Runnable>,
}

impl Engine {
    pub async fn create(runtime: &str, det: Vec<u8>, rec: Vec<u8>) -> Result<Self> {
        ensure!(runtime == "cpu" || runtime == "wgpu", "Unknown runtime {runtime}");
        // On wasm, looking up the wgpu runtime checks for a device, and that
        // check cannot create one. The device has to exist first.
        if runtime == "wgpu" {
            tract::wgpu::init().await?;
        }
        tract::runtime_for_name(runtime)?;
        Ok(Self {
            runtime: runtime.to_string(),
            det_bytes: det,
            rec_bytes: rec,
            det: HashMap::new(),
            rec: HashMap::new(),
        })
    }

    pub async fn detect(&mut self, data: &[f32], height: usize, width: usize) -> Result<TensorOut> {
        ensure!(
            height > 0 && width > 0 && height <= 2048 && width <= 2048,
            "detection size {width}x{height}"
        );
        ensure!(
            data.len() == 3 * height * width,
            "detection input has {} values, expected {}",
            data.len(),
            3 * height * width
        );
        if !self.det.contains_key(&(height, width)) {
            let spec = format!("1,3,{height},{width},f32");
            let runnable = prepare(&self.det_bytes, &self.runtime, &spec)?;
            self.det.insert((height, width), runnable);
        }
        let input = Tensor::from_slice(&[1, 3, height, width], data)?;
        let outputs = self.det[&(height, width)].run([input])?;
        host_f32(&outputs[0]).await
    }

    pub async fn recognize(&mut self, data: &[f32], width: usize) -> Result<TensorOut> {
        ensure!(width > 0 && width <= 2048, "recognition width {width}");
        ensure!(
            data.len() == 3 * 48 * width,
            "recognition input has {} values, expected {}",
            data.len(),
            3 * 48 * width
        );
        if !self.rec.contains_key(&width) {
            let spec = format!("1,3,48,{width},f32");
            let runnable = prepare(&self.rec_bytes, &self.runtime, &spec)?;
            self.rec.insert(width, runnable);
        }
        let input = Tensor::from_slice(&[1, 3, 48, width], data)?;
        let outputs = self.rec[&width].run([input])?;
        host_f32(&outputs[0]).await
    }
}

fn prepare(bytes: &[u8], runtime: &str, input: &str) -> Result<Runnable> {
    let mut model = tract::onnx()?
        .load_buffer_with_options(bytes, OnnxOptions::new().ignore_value_info(true))
        .context("Loading the ONNX graph")?;
    model.set_input_fact(0, input)?;
    model.set_output_fact(0, ())?;
    let typed = model.into_model()?;
    tract::runtime_for_name(runtime)?.prepare(typed)
}

async fn host_f32(tensor: &Tensor) -> Result<TensorOut> {
    let host = tensor.to_host_async().await?;
    let dims = host.shape()?.to_vec();
    let data = host.as_slice::<f32>()?.to_vec();
    Ok(TensorOut { data, dims })
}
