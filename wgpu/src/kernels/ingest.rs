//! RGBA8 texture → NCHW f32 ingest.
//!
//! Native: [`Queue::write_texture`] (bytes_per_row 256-aligned).
//! Wasm: same CPU-bytes path, plus [`Queue::copy_external_image_to_texture`]
//! for `VideoFrame` / `ImageBitmap` (wgpu 30 web `create_external_texture`
//! is unimplemented). One GPU copy; the frame never lands as an f32 CPU tensor.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::{wgpu_context, with_wgpu_queue};

/// A per-channel affine map applied to RGB in [0, 1] as it is ingested:
/// `(value - mean) / std`, as vision models document their input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    pub const IDENTITY: Normalization = Normalization { mean: [0.0; 3], std: [1.0; 3] };

    fn scale_offset(&self) -> TractResult<[f32; 8]> {
        let mut out = [0.0; 8];
        for c in 0..3 {
            ensure!(
                self.std[c].is_finite() && self.std[c] != 0.0 && self.mean[c].is_finite(),
                "normalization needs a finite mean and a finite, nonzero std"
            );
            out[c] = self.std[c].recip();
            out[4 + c] = -self.mean[c] / self.std[c];
        }
        Ok(out)
    }
}

pub fn all_pipeline_keys(_shader_f16: bool) -> Vec<PipelineKey> {
    let mut keys = vec![PipelineKey {
        module: ModuleKey { kind: ModuleKind::Ingest, dtype: ShaderDtype::F32 },
        entry: EntryPoint::plain("rgba_to_nchw_f32"),
    }];
    for entry in ["rgb565_to_nchw_f32", "nv12_to_nchw_f32"] {
        keys.push(PipelineKey {
            module: ModuleKey { kind: ModuleKind::IngestBuffer, dtype: ShaderDtype::F32 },
            entry: EntryPoint::plain(entry),
        });
    }
    keys
}

fn packed_image_pixels(width: u32, height: u32) -> TractResult<u32> {
    ensure!(width > 0 && height > 0, "ingest size must be > 0");
    let pixels = width.checked_mul(height).context("ingest dimensions overflow")?;
    pixels.checked_mul(12).context("ingest output byte size exceeds u32")?;
    Ok(pixels)
}

fn ingest_buffer(
    width: u32,
    height: u32,
    bytes: &[u8],
    entry: &'static str,
) -> TractResult<DeviceTensor> {
    with_wgpu_queue(|q| {
        let limits = q.context().device().limits();
        let max_bytes = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        ensure!(bytes.len() as u64 <= max_bytes, "ingest input exceeds device buffer limit");
        ensure!(
            u64::from(width) * u64::from(height) * 12 <= max_bytes,
            "ingest output exceeds device buffer limit"
        );
        let input = q.context().create_storage_buffer(bytes);
        let output = output_tensor(width, height)?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::IngestBuffer, dtype: ShaderDtype::F32 },
            entry: EntryPoint::plain(entry),
        })?;
        let params = pack_u32s(&[element_offset(&output, 0).try_into()?, width, height, 0]);
        let dyn_off = q.alloc_uniform(&params)?;
        let bg = q.context().bind_group_ingest_buffer(
            &input,
            &get_wgpu_buffer(&output).inner,
            q.uniform(),
        )?;
        q.retain_tensor(&output);
        q.dispatch("ingest-buffer", &pipeline, &bg, dyn_off, u64::from(width) * u64::from(height))?;
        Ok(output)
    })
}

/// Upload tightly packed little-endian RGB565 and produce RGB NCHW f32 in [0, 1].
/// Requires exactly `width * height * 2` bytes and nonzero dimensions.
pub fn tensor_from_rgb565(width: u32, height: u32, bytes: &[u8]) -> TractResult<DeviceTensor> {
    let pixels = packed_image_pixels(width, height)?;
    ensure!(bytes.len() as u64 == u64::from(pixels) * 2, "invalid RGB565 byte length");
    ingest_buffer(width, height, bytes, "rgb565_to_nchw_f32")
}

/// Upload tightly packed NV12 using full-range BT.601 and produce RGB NCHW f32 in [0, 1].
/// Requires positive even dimensions and exactly `width * height * 3 / 2` bytes:
/// a Y plane followed by interleaved Cb/Cr, one pair per 2x2 luma block.
/// Chroma uses nearest-neighbor replication. Limited-range and HDR inputs must
/// be converted by the caller; no transfer-function or primaries conversion is applied.
pub fn tensor_from_nv12(width: u32, height: u32, bytes: &[u8]) -> TractResult<DeviceTensor> {
    let pixels = packed_image_pixels(width, height)?;
    ensure!(width.is_multiple_of(2) && height.is_multiple_of(2), "NV12 dimensions must be even");
    ensure!(bytes.len() as u64 == u64::from(pixels) * 3 / 2, "invalid NV12 byte length");
    ingest_buffer(width, height, bytes, "nv12_to_nchw_f32")
}

fn ingest_usages() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::RENDER_ATTACHMENT
}

fn create_rgba8_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tract-wgpu-ingest-rgba"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: ingest_usages(),
        view_formats: &[],
    })
}

fn pad_rgba_rows(width: u32, height: u32, rgba: &[u8]) -> TractResult<(Vec<u8>, u32)> {
    let row = width.checked_mul(4).context("width overflow")?;
    ensure!(
        rgba.len() == row as usize * height as usize,
        "RGBA8 length {} != {width}x{height}x4",
        rgba.len()
    );
    let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    if padded_row == row {
        return Ok((rgba.to_vec(), row));
    }
    let mut out = vec![0u8; padded_row as usize * height as usize];
    for y in 0..height as usize {
        let src = y * row as usize;
        let dst = y * padded_row as usize;
        out[dst..dst + row as usize].copy_from_slice(&rgba[src..src + row as usize]);
    }
    Ok((out, padded_row))
}

fn dispatch_rgba_to_nchw(
    q: &crate::WgpuQueue,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    output: &DeviceTensor,
    normalization: Normalization,
) -> TractResult<()> {
    let scale_offset = normalization.scale_offset()?;
    q.retain_tensor(output);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let pipeline = q.context().pipeline(PipelineKey {
        module: ModuleKey { kind: ModuleKind::Ingest, dtype: ShaderDtype::F32 },
        entry: EntryPoint::plain("rgba_to_nchw_f32"),
    })?;
    let out_buf = get_wgpu_buffer(output);
    let bg = q.context().bind_group_ingest(&view, &out_buf.inner, q.uniform())?;
    let mut words = vec![element_offset(output, 0) as u32, width, height, 0];
    words.extend(scale_offset.iter().map(|v| v.to_bits()));
    let params = pack_u32s(&words);
    let dyn_off = q.alloc_uniform(&params)?;
    q.dispatch("ingest", &pipeline, &bg, dyn_off, width as u64 * height as u64)?;
    q.retain_texture(texture);
    Ok(())
}

fn output_tensor(width: u32, height: u32) -> TractResult<DeviceTensor> {
    DeviceTensor::uninitialized_dt(DatumType::F32, &[1, 3, height as usize, width as usize])
}

/// Upload tightly packed RGBA8 (`width * height * 4` bytes) and convert to
/// NCHW f32 `[1, 3, H, W]` (alpha dropped). Works native and on wasm.
pub fn tensor_from_rgba8(width: u32, height: u32, rgba: &[u8]) -> TractResult<DeviceTensor> {
    tensor_from_rgba8_normalized(width, height, rgba, Normalization::IDENTITY)
}

pub(crate) fn tensor_from_rgba8_normalized(
    width: u32,
    height: u32,
    rgba: &[u8],
    normalization: Normalization,
) -> TractResult<DeviceTensor> {
    ensure!(width > 0 && height > 0, "ingest size must be > 0");
    wgpu_context();
    with_wgpu_queue(|q| {
        let (padded, bpr) = pad_rgba_rows(width, height, rgba)?;
        let texture = create_rgba8_texture(q.context().device(), width, height);
        q.context().queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let output = output_tensor(width, height)?;
        dispatch_rgba_to_nchw(q, texture, width, height, &output, normalization)?;
        Ok(output)
    })
}

/// One-GPU-copy ingest from a WebCodecs `VideoFrame` / canvas / bitmap.
/// Requires wgpu's web backend (`copy_external_image_to_texture`).
#[cfg(target_arch = "wasm32")]
pub fn tensor_from_external_image(
    source: wgpu::ExternalImageSource,
    width: u32,
    height: u32,
) -> TractResult<DeviceTensor> {
    tensor_from_external_image_normalized(source, width, height, Normalization::IDENTITY)
}

/// [`tensor_from_external_image`], with each channel mapped by `normalization`.
#[cfg(target_arch = "wasm32")]
pub fn tensor_from_external_image_normalized(
    source: wgpu::ExternalImageSource,
    width: u32,
    height: u32,
    normalization: Normalization,
) -> TractResult<DeviceTensor> {
    ensure!(width > 0 && height > 0, "ingest size must be > 0");
    wgpu_context();
    with_wgpu_queue(|q| {
        let texture = create_rgba8_texture(q.context().device(), width, height);
        q.context().queue().copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source,
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let output = output_tensor(width, height)?;
        dispatch_rgba_to_nchw(q, texture, width, height, &output, normalization)?;
        Ok(output)
    })
}

#[cfg(target_arch = "wasm32")]
pub fn tensor_from_video_frame(frame: web_sys::VideoFrame) -> TractResult<DeviceTensor> {
    let width = frame.display_width();
    let height = frame.display_height();
    tensor_from_external_image(wgpu::ExternalImageSource::VideoFrame(frame), width, height)
}

/// Writes a single-channel tensor into a texture the caller owns, so a mask can
/// be composited on the GPU instead of read back.
pub fn tensor_to_texture(
    mask: &DeviceTensor,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        ensure!(
            mask.len() >= (width * height) as usize,
            "mask of {} values is too small for {width}x{height}",
            mask.len()
        );
        q.retain_tensor(mask);
        let dt = ShaderDtype::from_datum(mask.datum_type())?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Export, dtype: dt },
            entry: EntryPoint::typed("export", dt),
        })?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = q.context().bind_group_export(&get_wgpu_buffer(mask).inner, &view, q.uniform())?;
        let params = pack_u32s(&[element_offset(mask, 0) as u32, width, height, 0]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.retain_texture(texture.clone());
        q.dispatch("export", &pipeline, &bg, dyn_off, (width * height) as u64)
    })
}
