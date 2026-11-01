use tract_core::internal::*;
use tract_core::ops::cnn::{Deconv, KernelFormat};
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype,
    WORKGROUP, binary_ops_wgsl, epilogue_body, keys_for, pack_u32s, preamble, unary_ops_wgsl,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Deconv, &["conv_transpose2d"], shader_f16)
}

fn as_u32_i32(v: i32) -> u32 {
    v as u32
}

pub fn wgpu_deconv_dispatch(
    op: &Deconv,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(weights);
        for t in extras {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let pipeline = if epilogue.is_empty() {
            q.context().pipeline(PipelineKey {
                module: ModuleKey { kind: ModuleKind::Deconv, dtype: dt },
                entry: EntryPoint::typed("conv_transpose2d", dt),
            })?
        } else {
            let key = ("deconv", dt, epilogue, extras.len());
            q.context().chain_pipeline(
                key,
                layout,
                EntryPoint::typed("conv_transpose2d", dt),
                || deconv_module(dt, epilogue, extras.len()),
            )?
        };
        let in_shape = op.pool_spec.data_format.shape(input.shape())?;
        let out_shape = op.pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2, "tract-wgpu conv_transpose is 2D only");
        let n = *in_shape.n().unwrap_or(&1);
        let ci = *in_shape.c();
        let co = *out_shape.c();
        let ih = in_shape.hw_dims()[0];
        let iw = in_shape.hw_dims()[1];
        let oh = out_shape.hw_dims()[0];
        let ow = out_shape.hw_dims()[1];
        let kh = op.pool_spec.kernel_shape[0];
        let kw = op.pool_spec.kernel_shape[1];
        let groups = op.group;
        let ci_pg = op.pool_spec.input_channels / groups;
        let co_pg = op.pool_spec.output_channels / groups;
        let padding = op.pool_spec.padding.compute_for_deconv(
            in_shape.hw_dims(),
            &op.pool_spec.kernel_shape,
            &op.pool_spec.dilations(),
            &op.pool_spec.strides(),
            &op.adjustments,
        )?;
        let strides = op.pool_spec.strides();
        let dilations = op.pool_spec.dilations();

        let in_sn = *in_shape.n_stride().unwrap_or(&0);
        let in_sc = *in_shape.c_stride();
        let in_sh = in_shape.hw_strides()[0];
        let in_sw = in_shape.hw_strides()[1];
        let out_sn = *out_shape.n_stride().unwrap_or(&0);
        let out_sc = *out_shape.c_stride();
        let out_sh = out_shape.hw_strides()[0];
        let out_sw = out_shape.hw_strides()[1];
        let ws = weights.strides();
        ensure!(ws.len() >= 4);
        ensure!(
            op.kernel_format == KernelFormat::OIHW,
            "tract-wgpu conv_transpose reads [co, ci_pg, kh, kw]; rewrite_kernel_deconv_in_oihw \
             normalizes the kernel before translation"
        );
        let w_so = ws[0];
        let w_si = ws[1];
        let w_sh = ws[2];
        let w_sw = ws[3];

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(input), get_wgpu_buffer(weights)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(weights, 0) as u32,
            element_offset(output, 0) as u32,
            n as u32,
            ci as u32,
            co as u32,
            ih as u32,
            iw as u32,
            oh as u32,
            ow as u32,
            kh as u32,
            kw as u32,
            groups as u32,
            ci_pg as u32,
            co_pg as u32,
            0,
            padding[0].pad_before as u32,
            padding[1].pad_before as u32,
            strides[0] as u32,
            strides[1] as u32,
            dilations[0] as u32,
            dilations[1] as u32,
            as_u32_i32(in_sn as i32),
            as_u32_i32(in_sc as i32),
            as_u32_i32(in_sh as i32),
            as_u32_i32(in_sw as i32),
            as_u32_i32(w_so as i32),
            as_u32_i32(w_si as i32),
            as_u32_i32(w_sh as i32),
            as_u32_i32(w_sw as i32),
            as_u32_i32(out_sn as i32),
            as_u32_i32(out_sc as i32),
            as_u32_i32(out_sh as i32),
            as_u32_i32(out_sw as i32),
        ]);
        let mut tail = vec![0u32; 2];
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = crate::kernels::matmul::epilogue_mode(t, co)?;
        }
        tail.extend_from_slice(&off_extra);
        tail.extend_from_slice(&mode_extra);
        params.extend_from_slice(&pack_u32s(&tail));
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("deconv", &pipeline, &bg, dyn_off, (n * co * oh * ow) as u64)
    })
}

pub(crate) fn deconv_wgsl(dt: ShaderDtype) -> String {
    deconv_module(dt, &[], 0)
}

/// The gather-based transposed convolution, with an epilogue as
/// [`conv_module`] takes one.
///
/// [`conv_module`]: crate::kernels::conv::conv_module
pub fn deconv_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body(epilogue, "co");
    let mut s = preamble(dt);
    // Gather-based conv_transpose (ORT style). WGSL has no float atomics.
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    ci: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    groups: u32,
    ci_pg: u32,
    co_pg: u32,
    _p0: u32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_si: i32,
    w_sh: i32,
    w_sw: i32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    _pad: vec2<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
@compute @workgroup_size({WORKGROUP})
fn conv_transpose2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let nout = params.n * params.co * params.oh * params.ow;
    if (i >= nout) {{ return; }}
    var rest = i;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh; rest = rest / params.oh;
    let co = rest % params.co;
    let n = rest / params.co;
    let g = co / params.co_pg;
    let ci0 = g * params.ci_pg;
    var acc = 0.0;
    // Which taps land on an input pixel does not depend on the channel, and
    // finding out costs two integer divisions, so the channels run innermost.
    for (var kh = 0u; kh < params.kh; kh++) {{
        let ih_num = i32(oh) + params.pad_h - i32(kh) * params.dil_h;
        if (params.stride_h == 0 || ih_num % params.stride_h != 0) {{ continue; }}
        let ih = ih_num / params.stride_h;
        if (ih < 0 || ih >= i32(params.ih)) {{ continue; }}
        for (var kw = 0u; kw < params.kw; kw++) {{
            let iw_num = i32(ow) + params.pad_w - i32(kw) * params.dil_w;
            if (params.stride_w == 0 || iw_num % params.stride_w != 0) {{ continue; }}
            let iw = iw_num / params.stride_w;
            if (iw < 0 || iw >= i32(params.iw)) {{ continue; }}
            let in_base = i32(n) * params.in_sn
                + i32(ci0) * params.in_sc
                + ih * params.in_sh
                + iw * params.in_sw;
            let w_base = i32(co) * params.w_so
                + i32(kh) * params.w_sh
                + i32(kw) * params.w_sw;
            for (var ci = 0u; ci < params.ci_pg; ci++) {{
                let in_i = params.off_in + u32(in_base + i32(ci) * params.in_sc);
                let w_i = params.off_w + u32(w_base + i32(ci) * params.w_si);
                acc += f32(inp[in_i]) * f32(wgt[w_i]);
            }}
        }}
    }}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    var v = {t}(acc);
{epilogue}
    outp[out_i] = v;
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;
    use tract_core::transform::ModelTransform;

    use crate::WgpuTransform;
    use crate::test_utils::{fill, gpu_vs_cpu};

    /// The bias and activation after a transposed convolution fold into its
    /// kernel, and the result matches the CPU: the segmenter's last layer.
    #[test]
    fn deconv_bias_activation_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        for (ci, co, h, w) in [(16usize, 1usize, 9usize, 16usize), (3, 2, 5, 7)] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([1, ci, h, w]))?;
            let k = model
                .add_const("k", Tensor::from_shape(&[co, ci, 2, 2], &fill(3, ci * co * 4))?)?;
            let bias = model.add_const("bias", Tensor::from_shape(&[co], &fill(4, co))?)?;
            let deconv = Deconv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![2, 2],
                    padding: PaddingSpec::Valid,
                    dilations: None,
                    strides: Some(tvec![2, 2]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_format: KernelFormat::OIHW,
                adjustments: tvec![0, 0],
                group: 1,
            };
            let y = model.wire_node("deconv", deconv, &[x, k, bias])?[0];
            let y = model.wire_node("sigmoid", tract_core::ops::nn::sigmoid(), &[y])?[0];
            model.select_output_outlets(&[y])?;
            let mut folded = model.clone();
            WgpuTransform.transform(&mut folded)?;
            let deconv = folded
                .nodes()
                .iter()
                .find_map(|n| n.op_as::<crate::ops::deconv::WgpuDeconv>())
                .context("no WgpuDeconv after transform")?;
            ensure!(
                deconv.epilogue.len() == 2,
                "epilogue not folded: {:?}\n{}",
                deconv.epilogue,
                folded
                    .nodes()
                    .iter()
                    .map(|n| format!("{} {}", n.op.name(), n.name))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            gpu_vs_cpu(model, Tensor::from_shape(&[1, ci, h, w], &fill(5, ci * h * w))?)
                .with_context(|| format!("ci{ci} co{co} {h}x{w}"))?;
        }
        Ok(())
    }
}
