use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use wasm_bindgen::JsValue;

use crate::Tensor;

/// The browser GPU device shared by WGPU inference and a page compositor.
/// Call `tract::wgpu::init` before requesting it.
pub fn wgpu_device() -> Result<JsValue> {
    tract_wgpu::webgpu_device().context("No browser WebGPU device; initialize WGPU first")
}

impl Tensor {
    /// Copy an OffscreenCanvas into RGB NCHW f32 on the GPU, without CPU pixel readback.
    /// The whole canvas is used at its current positive width and height.
    pub fn from_canvas(canvas: web_sys::OffscreenCanvas) -> Result<Self> {
        let (width, height) = (canvas.width(), canvas.height());
        let tensor = tract_wgpu::tensor_from_external_image(
            wgpu::ExternalImageSource::OffscreenCanvas(canvas),
            width,
            height,
        )?;
        Ok(Self(std::sync::Arc::new(tensor.into_tensor())))
    }

    /// [`Tensor::from_canvas`], with each RGB channel mapped from [0, 1] to
    /// `(value - mean) / std` in the same GPU pass. `std` must be finite and nonzero.
    pub fn from_canvas_normalized(
        canvas: web_sys::OffscreenCanvas,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Self> {
        let (width, height) = (canvas.width(), canvas.height());
        let tensor = tract_wgpu::tensor_from_external_image_normalized(
            wgpu::ExternalImageSource::OffscreenCanvas(canvas),
            width,
            height,
            tract_wgpu::Normalization { mean, std },
        )?;
        Ok(Self(std::sync::Arc::new(tensor.into_tensor())))
    }
}

/// An RGBA8 mask texture shared with JavaScript on the inference device.
/// Serialize writes with inference; the mask is quantized to eight bits, never read back.
pub struct WgpuMask {
    texture: wgpu::Texture,
    handle: JsValue,
    width: u32,
    height: u32,
}

impl WgpuMask {
    /// Allocate a positive-sized mask after `tract::wgpu::init`.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        ensure!(width > 0 && height > 0, "Mask size must be positive");
        let limit = tract_wgpu::wgpu_context().device().limits().max_texture_dimension_2d;
        ensure!(width <= limit && height <= limit, "Mask exceeds the device texture limit");
        ensure!(u64::from(width) * u64::from(height) <= u64::from(u32::MAX), "Mask is too large");
        let (texture, handle) = tract_wgpu::mask_texture(width, height)?;
        Ok(Self { texture, handle, width, height })
    }

    /// The JavaScript GPUTexture, for sampling on the shared GPUDevice.
    pub fn texture(&self) -> JsValue {
        self.handle.clone()
    }

    /// Write a contiguous, single-channel WGPU f32 tensor and submit pending GPU work.
    /// The tensor must contain exactly `width * height` values. Completion is asynchronous;
    /// the page can await the shared device queue before measuring or reusing the frame.
    pub fn write(&self, tensor: &Tensor) -> Result<()> {
        let device = tensor.wgpu_storage()?;
        ensure!(
            device.len() as u64 == self.width as u64 * self.height as u64,
            "Mask tensor size does not match the texture"
        );
        ensure!(
            device.datum_type() == tract_nnef::prelude::DatumType::F32,
            "Mask tensor must be f32"
        );
        let mut expected_stride = 1isize;
        for (&dimension, &stride) in device.shape().iter().zip(device.strides()).rev() {
            ensure!(dimension <= 1 || stride == expected_stride, "Mask must be contiguous");
            expected_stride *= dimension as isize;
        }
        tract_wgpu::tensor_to_texture(device, &self.texture, self.width, self.height)?;
        tract_wgpu::with_wgpu_queue(|queue| queue.flush())
    }
}
