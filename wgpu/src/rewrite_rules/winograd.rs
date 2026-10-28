//! Winograd F(4,3) as a pure graph rewrite: a dense stride-1 3x3 `Conv`
//! becomes a tile-extracting grouped `Conv` whose weights bake in the B^T
//! transform, one batched `PrefixMatMul` accumulating the elementwise products
//! over input channels (the G-transformed weights uploaded once as a constant),
//! a second `PrefixMatMul` for the A^T output transform, then axis bookkeeping
//! and a crop. Every op used already has a wgpu kernel, and the axis ops fold
//! into their matmul/slice consumers through the axis-op fusion passes: with a
//! bias the decode permute folds into the bias add's strided read, so the tail
//! costs no copy at all.

use tract_core::internal::*;
use tract_core::ops::array::Slice;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::math::add;
use tract_core::ops::nn::DataFormat;

/// Input transform `B^T` of the F(4x4, 3x3) algorithm, applied as `B^T d B` to
/// each 6x6 input tile. Points {0, +-1, +-2, infinity} after Lavin & Gray.
#[rustfmt::skip]
const BT: [[f64; 6]; 6] = [
    [4.,  0., -5.,  0.,  1., 0.],
    [0., -4., -4.,  1.,  1., 0.],
    [0.,  4., -4., -1.,  1., 0.],
    [0., -2., -1.,  2.,  1., 0.],
    [0.,  2., -1., -2.,  1., 0.],
    [0.,  4.,  0., -5.,  0., 1.],
];

/// Filter transform `G` of F(4,3), applied as `G g G^T` to each 3x3 kernel.
#[rustfmt::skip]
const G: [[f64; 3]; 6] = [
    [1. / 4.,    0.,       0.     ],
    [-1. / 6.,  -1. / 6., -1. / 6.],
    [-1. / 6.,   1. / 6., -1. / 6.],
    [1. / 24.,   1. / 12., 1. / 6.],
    [1. / 24.,  -1. / 12., 1. / 6.],
    [0.,         0.,       1.     ],
];

/// Output transform `A^T` of F(4,3), applied as `A^T m A` to each 6x6 product
/// tile, yielding the 4x4 outputs.
#[rustfmt::skip]
const AT: [[f64; 6]; 4] = [
    [1., 1.,  1., 1.,  1., 0.],
    [0., 1., -1., 2., -2., 0.],
    [0., 1.,  1., 4.,  4., 0.],
    [0., 1., -1., 8., -8., 1.],
];

/// Smallest `ci * co` the rewrite takes. Winograd F(4,3) does 2.25x fewer
/// multiply-accumulates in its accumulation GEMM, but adds the extraction
/// convolution, the output-transform GEMM and the decode, whose cost grows with
/// `ci + co` and the tile count rather than `ci * co`. Below this product those
/// passes cost more than the GEMM saves, at any map size.
const WINOGRAD_MIN_CHANNEL_PRODUCT: usize = 16384;

/// Fewest 4x4 output tiles the rewrite takes.
const WINOGRAD_MIN_TILES: usize = 16;

/// `B^T ⊗ B^T` baked as the weights of a grouped 6x6 stride-4 convolution:
/// output channel `c*36 + (i*6+j)` of input-channel group `c` computes
/// `sum_ab BT[i][a] * BT[j][b] * x[c, 4th+a, 4tw+b]`, which is exactly tile
/// `d`'s transform `(B^T d B)[i,j]` for every tile at once.
fn input_transform_weights(ci: usize) -> TractResult<Tensor> {
    let mut w = vec![0f32; ci * 36 * 36];
    for c in 0..ci {
        for i in 0..6 {
            for j in 0..6 {
                for a in 0..6 {
                    for b in 0..6 {
                        w[((c * 36 + i * 6 + j) * 6 + a) * 6 + b] = (BT[i][a] * BT[j][b]) as f32;
                    }
                }
            }
        }
    }
    Tensor::from_shape(&[ci * 36, 1, 6, 6], &w)
}

/// `G g G^T` of every `(co, ci)` kernel pair, packed as `[36, n, co, ci]` for
/// the batched accumulation GEMM: `u[i*6+j][n][co][ci]`.
fn transformed_weights(kernel: &Tensor, n: usize, co: usize, ci: usize) -> TractResult<Tensor> {
    // `cast_to` panics on a named weight it cannot load; this read fails instead.
    kernel.try_as_plain_ram()?;
    let kernel = kernel.cast_to::<f32>()?;
    let k = kernel.to_plain_array_view::<f32>()?;
    let mut u = vec![0f32; 36 * n * co * ci];
    for o in 0..co {
        for c in 0..ci {
            let mut u6 = [[0f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let mut acc = 0f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            acc += G[i][a] * k[[o, c, a, b]] as f64 * G[j][b];
                        }
                    }
                    u6[i][j] = acc;
                }
            }
            for i in 0..6 {
                for j in 0..6 {
                    for nn in 0..n {
                        u[((i * 6 + j) * n + nn) * co * ci + o * ci + c] = u6[i][j] as f32;
                    }
                }
            }
        }
    }
    Tensor::from_shape(&[36, n, co, ci], &u)
}

/// `A^T ⊗ A^T` packed as `[16, 36]`: `at[i'*4+j'][i*6+j] = AT[i'][i]*AT[j'][j]`,
/// turning `Y = A^T M A` into one flat GEMM producing `[16, tiles]`.
fn output_transform() -> TractResult<Tensor> {
    let mut t = vec![0f32; 16 * 36];
    for ip in 0..4 {
        for jp in 0..4 {
            for i in 0..6 {
                for j in 0..6 {
                    t[(ip * 4 + jp) * 36 + i * 6 + j] = (AT[ip][i] * AT[jp][j]) as f32;
                }
            }
        }
    }
    Tensor::from_shape(&[16, 36], &t)
}

/// Rewrites a `Conv(stride 1, dilation 1, 3x3, group 1, f32 or f16, NCHW)` as
/// Winograd F(4,3). Emits only core ops the wgpu backend already runs: the
/// padding folds into the extraction convolution's own padding, the
/// `[36, ci, tiles]` operand is restrided into the accumulation GEMM by the
/// axis-op fusions, and the tail crop fuses into `GpuSlice`.
pub fn conv_to_winograd_f43(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let spec = &op.pool_spec;
    rule_if!(
        op.group == 1
            && op.q_params.is_none()
            && op.kernel_fmt == KernelFormat::OIHW
            && spec.data_format == DataFormat::NCHW
            && spec.kernel_shape == tvec![3, 3]
            && spec.stride(0) == 1
            && spec.stride(1) == 1
            && spec.dilation(0) == 1
            && spec.dilation(1) == 1
    );
    let facts = model.node_input_facts(node.id)?;
    rule_if!(facts.len() == 3);
    let dt = facts[0].datum_type;
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));
    rule_if_some!(
        kernel = facts[1].konst.as_ref().filter(|k| {
            k.datum_type() == dt && k.shape() == [op.output_channels(), op.input_channels(), 3, 3]
        })
    );
    let in_shape = spec.data_format.shape(&facts[0].shape)?;
    rule_if_some!(n = in_shape.n().and_then(|d| d.as_usize()));
    rule_if_some!(ih = in_shape.hw_dims()[0].as_usize());
    rule_if_some!(iw = in_shape.hw_dims()[1].as_usize());
    let padding = spec.computed_padding(in_shape.hw_dims());
    rule_if_some!(ph = padding[0].pad_before.as_usize());
    rule_if_some!(pw = padding[1].pad_before.as_usize());
    rule_if_some!(oh = padding[0].convoluted.as_usize());
    rule_if_some!(ow = padding[1].convoluted.as_usize());
    let (ci, co) = (op.input_channels(), op.output_channels());
    let nth = oh.div_ceil(4);
    let ntw = ow.div_ceil(4);
    let nt = nth * ntw;
    rule_if!(ci * co >= WINOGRAD_MIN_CHANNEL_PRODUCT && nt >= WINOGRAD_MIN_TILES);
    // The extraction conv's padding must cover the conv's own plus enough extra
    // to feed whole 6x6 windows: (ih + pad_h + extra_h - 6) / 4 + 1 == nth.
    let extra_h = 4 * nth + 2 - ih - ph;
    let extra_w = 4 * ntw + 2 - iw - pw;
    let x = n * co * nt;

    let mut patch = TypedModelPatch::default();
    let data = patch.tap_model(model, node.inputs[0])?;
    let w_in = patch.add_const(
        format!("{name}.wino.bt"),
        input_transform_weights(ci)?.cast_to_dt(dt)?.into_owned(),
    )?;
    let zero = patch.add_const(format!("{name}.wino.zero"), Tensor::zero_dt(dt, &[ci * 36])?)?;
    let w_u = patch.add_const(
        format!("{name}.wino.g"),
        transformed_weights(kernel, n, co, ci)?.cast_to_dt(dt)?.into_owned(),
    )?;
    let w_at = patch
        .add_const(format!("{name}.wino.at"), output_transform()?.cast_to_dt(dt)?.into_owned())?;

    let extract = Conv {
        pool_spec: PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![6, 6],
            padding: PaddingSpec::Explicit(tvec![ph, pw], tvec![extra_h, extra_w]),
            dilations: None,
            strides: Some(tvec![4, 4]),
            input_channels: ci,
            output_channels: ci * 36,
        },
        kernel_fmt: KernelFormat::OIHW,
        group: ci,
        q_params: None,
    };
    let mut wire = patch.wire_node(format!("{name}.wino.d"), extract, &[data, w_in, zero])?[0];
    wire = patch.wire_node(
        format!("{name}.wino.v"),
        AxisOp::Reshape(1, tvec![(ci * 36).to_dim()], tvec![ci.to_dim(), 36.to_dim()]),
        &[wire],
    )?[0];
    wire = patch.wire_node(
        format!("{name}.wino.vt"),
        AxisOp::Reshape(3, tvec![nth.to_dim(), ntw.to_dim()], tvec![nt.to_dim()]),
        &[wire],
    )?[0];
    wire = patch.wire_node(format!("{name}.wino.vb"), AxisOp::Move(2, 0), &[wire])?[0];
    let no_transpose = PrefixMatMul {
        transpose_a: false,
        transpose_b: false,
        transpose_c: false,
        quantize_output: None,
        operating_dt: None,
    };
    wire = patch.wire_node(format!("{name}.wino.m"), no_transpose, &[w_u, wire])?[0];
    wire = patch.wire_node(
        format!("{name}.wino.mf"),
        AxisOp::Reshape(1, tvec![n.to_dim(), co.to_dim(), nt.to_dim()], tvec![x.to_dim()]),
        &[wire],
    )?[0];
    // `matmul_blocked` only runs the all-transposed form, so the GEMM writes
    // [16, tiles]. Decoding it into image order interleaves (th, 4) and (tw, 4)
    // as rows and columns: a permute no view can express. The bias add runs on
    // the permuted 6-D view instead: the elementwise kernel reads its operand
    // through strides and writes contiguous, folding the decode into a dispatch
    // that already ran. Without a bias the two Moves stand alone.
    let all_transposed = PrefixMatMul {
        transpose_a: true,
        transpose_b: true,
        transpose_c: true,
        quantize_output: None,
        operating_dt: None,
    };
    wire = patch.wire_node(format!("{name}.wino.y"), all_transposed, &[wire, w_at])?[0];
    wire = patch.wire_node(
        format!("{name}.wino.tiles"),
        AxisOp::Reshape(
            0,
            tvec![16.to_dim(), x.to_dim()],
            tvec![4.to_dim(), 4.to_dim(), n.to_dim(), co.to_dim(), nth.to_dim(), ntw.to_dim()],
        ),
        &[wire],
    )?[0];
    wire = patch.wire_node(format!("{name}.wino.mj"), AxisOp::Move(1, 5), &[wire])?[0];
    wire = patch.wire_node(format!("{name}.wino.mi"), AxisOp::Move(0, 3), &[wire])?[0];
    let bias_fact = &facts[2];
    if !bias_fact.konst.as_ref().is_some_and(|k| k.is_all_zero().unwrap_or(false)) {
        rule_if!(bias_fact.shape.volume() == co.to_dim());
        let bias = patch.tap_model(model, node.inputs[2])?;
        let bias = patch.wire_node(
            format!("{name}.wino.bias"),
            AxisOp::Reshape(
                0,
                bias_fact.shape.to_tvec(),
                tvec![1.to_dim(), co.to_dim(), 1.to_dim(), 1.to_dim(), 1.to_dim(), 1.to_dim()],
            ),
            &[bias],
        )?[0];
        wire = patch.wire_node(format!("{name}.wino.add_bias"), add(), &[wire, bias])?[0];
    }
    wire = patch.wire_node(
        format!("{name}.wino.img"),
        AxisOp::Reshape(
            2,
            tvec![nth.to_dim(), 4.to_dim(), ntw.to_dim(), 4.to_dim()],
            tvec![(4 * nth).to_dim(), (4 * ntw).to_dim()],
        ),
        &[wire],
    )?[0];
    if oh < 4 * nth {
        wire = patch.wire_node(format!("{name}.wino.ch"), Slice::new(2, 0, oh), &[wire])?[0];
    }
    if ow < 4 * ntw {
        wire = patch.wire_node(format!("{name}.wino.cw"), Slice::new(3, 0, ow), &[wire])?[0];
    }
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}
