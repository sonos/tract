struct Params {
    off_out: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(offset: u32) -> u32 {
    return (src[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

fn store_rgb(i: u32, rgb: vec3<f32>) {
    let hw = params.width * params.height;
    let offset = params.off_out + i;
    dst[offset] = rgb.r;
    dst[offset + hw] = rgb.g;
    dst[offset + 2u * hw] = rgb.b;
}

@compute @workgroup_size(64)
fn rgb565_to_nchw_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    if (i >= params.width * params.height) { return; }
    let pixel = byte_at(2u * i) | (byte_at(2u * i + 1u) << 8u);
    store_rgb(i, vec3<f32>(
        f32((pixel >> 11u) & 31u) / 31.0,
        f32((pixel >> 5u) & 63u) / 63.0,
        f32(pixel & 31u) / 31.0,
    ));
}

@compute @workgroup_size(64)
fn nv12_to_nchw_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    let hw = params.width * params.height;
    if (i >= hw) { return; }
    let x = i % params.width;
    let y = i / params.width;
    let uv = hw + (y / 2u) * params.width + (x / 2u) * 2u;
    let luma = f32(byte_at(i));
    let cb = f32(byte_at(uv)) - 128.0;
    let cr = f32(byte_at(uv + 1u)) - 128.0;
    let rgb = vec3<f32>(
        luma + 1.402 * cr,
        luma - 0.344136 * cb - 0.714136 * cr,
        luma + 1.772 * cb,
    ) / 255.0;
    store_rgb(i, clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
}
