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

pub fn all_pipeline_keys(_shader_f16: bool) -> Vec<PipelineKey> {
    vec![PipelineKey {
        module: ModuleKey { kind: ModuleKind::Ingest, dtype: ShaderDtype::F32 },
        entry: EntryPoint::plain("rgba_to_nchw_f32"),
    }]
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
) -> TractResult<()> {
    q.retain_tensor(output);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let pipeline = q.context().pipeline(PipelineKey {
        module: ModuleKey { kind: ModuleKind::Ingest, dtype: ShaderDtype::F32 },
        entry: EntryPoint::plain("rgba_to_nchw_f32"),
    })?;
    let out_buf = get_wgpu_buffer(output);
    let bg = q.context().bind_group_ingest(&view, &out_buf.inner, q.uniform())?;
    let params = pack_u32s(&[element_offset(output, 0) as u32, width, height, 0]);
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
        dispatch_rgba_to_nchw(q, texture, width, height, &output)?;
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
        dispatch_rgba_to_nchw(q, texture, width, height, &output)?;
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
