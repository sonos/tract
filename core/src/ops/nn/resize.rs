use crate::internal::*;
use crate::ops::array::Tile;

/// Maps an output coordinate back to the input axis. The ONNX coordinate
/// transformation modes that have a well-defined inverse without an input ROI.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CoordTransformer {
    HalfPixel,
    AlignCorners,
    Asymmetric,
    PytorchHalfPixel,
    HalfPixelSymmetric,
}

impl CoordTransformer {
    pub fn transform(&self, x_out: usize, scale: f32, len_in: usize, len_out: usize) -> f32 {
        match self {
            CoordTransformer::HalfPixel => (x_out as f32 + 0.5) / scale - 0.5,
            CoordTransformer::AlignCorners => {
                let output_width = scale * len_in as f32;
                if output_width == 1.0 {
                    0.0
                } else {
                    (x_out as f32 * (len_in as f32 - 1.0)) / (output_width - 1.0)
                }
            }
            CoordTransformer::Asymmetric => (x_out as f32) / scale,
            CoordTransformer::PytorchHalfPixel => {
                if len_out > 1 {
                    (x_out as f32 + 0.5) / scale - 0.5
                } else {
                    -0.5
                }
            }
            CoordTransformer::HalfPixelSymmetric => {
                let adjustment = len_out as f32 / (scale * len_in as f32);
                let offset = len_in as f32 / 2.0 * (1.0 - adjustment);
                offset + (x_out as f32 + 0.5) / scale - 0.5
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CoordTransformer::HalfPixel => "half_pixel",
            CoordTransformer::AlignCorners => "align_corners",
            CoordTransformer::Asymmetric => "asymmetric",
            CoordTransformer::PytorchHalfPixel => "pytorch_half_pixel",
            CoordTransformer::HalfPixelSymmetric => "half_pixel_symmetric",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "half_pixel" => CoordTransformer::HalfPixel,
            "align_corners" => CoordTransformer::AlignCorners,
            "asymmetric" => CoordTransformer::Asymmetric,
            "pytorch_half_pixel" => CoordTransformer::PytorchHalfPixel,
            "half_pixel_symmetric" => CoordTransformer::HalfPixelSymmetric,
            s => bail!("coordinate_transformation_mode: {s}"),
        })
    }
}

/// Interpolation kernel. `Linear` and `Nearest` use a 2-tap path; `Cubic` uses
/// a 4-tap kernel with the standard `a = -0.75` coefficient.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Interpolator {
    Linear,
    Nearest,
    Cubic,
}

impl Interpolator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Interpolator::Linear => "linear",
            Interpolator::Nearest => "nearest",
            Interpolator::Cubic => "cubic",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "linear" => Interpolator::Linear,
            "nearest" => Interpolator::Nearest,
            "cubic" => Interpolator::Cubic,
            s => bail!("mode: {s}"),
        })
    }
}

/// Number of input taps feeding one output element. Antialiasing widens the
/// footprint of `Linear` and `Cubic` by `1/scale` when downscaling, which is
/// what turns the interpolation into a low-pass filter.
pub fn window_size(interpolator: &Interpolator, antialias: bool, scale: f32) -> usize {
    let support = match interpolator {
        Interpolator::Nearest | Interpolator::Linear => 1.0f32,
        Interpolator::Cubic => 2.0,
    };
    if !antialias || scale >= 1.0 || matches!(interpolator, Interpolator::Nearest) {
        return 2 * support as usize;
    }
    let first = (-support / scale).floor() as isize + 1;
    (2 - 2 * first) as usize
}

/// Triangular kernel, low-passed by `scale` when antialiasing.
pub fn linear_weights(r: f32, scale: f32, antialias: bool, weights: &mut [f32]) {
    let scale = if antialias { scale.min(1.0) } else { 1.0 };
    fill_weights(r, scale, weights, |x| (1.0 - x.abs()).clamp(0.0, 1.0));
}

/// Cubic convolution kernel of coefficient `a`, low-passed by `scale` when
/// antialiasing.
pub fn cubic_weights(r: f32, scale: f32, a: f32, antialias: bool, weights: &mut [f32]) {
    let scale = if antialias { scale.min(1.0) } else { 1.0 };
    fill_weights(r, scale, weights, |x| cubic_kernel(x, a));
}

/// Evaluates `kernel` at the offset of each tap in the window from the source
/// coordinate, then renormalizes when the kernel was stretched by `scale`.
fn fill_weights(r: f32, scale: f32, weights: &mut [f32], kernel: impl Fn(f32) -> f32) {
    let first = 1.0 - (weights.len() / 2) as f32;
    for (k, w) in weights.iter_mut().enumerate() {
        *w = kernel((first + k as f32 - r) * scale);
    }
    if scale != 1.0 {
        let sum: f32 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sum);
    }
}

/// Standard Catmull-Rom-family cubic convolution kernel with coefficient `a`.
pub fn cubic_kernel(s: f32, a: f32) -> f32 {
    let abs_s = s.abs();
    if abs_s <= 1.0 {
        (a + 2.0) * abs_s * abs_s * abs_s - (a + 3.0) * abs_s * abs_s + 1.0
    } else if abs_s <= 2.0 {
        a * abs_s * abs_s * abs_s - 5.0 * a * abs_s * abs_s + 8.0 * a * abs_s - 4.0 * a
    } else {
        0.0
    }
}

/// Nearest-neighbour tie-breaking. Restricted to the two modes tract-core
/// supports: `Floor` (PyTorch `upsample_nearest`) and `RoundPreferCeil`
/// (`_upsample_nearest_exact`). The other ONNX modes stay in the ONNX op.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Nearest {
    Floor,
    RoundPreferCeil,
}

impl Nearest {
    /// True when the right (ceil) neighbour wins for a fractional offset.
    pub fn prefers_right(&self, x_ratio: f32) -> bool {
        match self {
            Nearest::Floor => false,
            Nearest::RoundPreferCeil => x_ratio >= 0.5,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Nearest::Floor => "floor",
            Nearest::RoundPreferCeil => "round_prefer_ceil",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "floor" => Nearest::Floor,
            "round_prefer_ceil" => Nearest::RoundPreferCeil,
            s => bail!("nearest_mode: {s}"),
        })
    }
}

/// Resampling plan for one axis: for each output index, the `window` input taps
/// it reads (already clamped into the axis) and their weights. `extrapolated`
/// flags output indices that map outside the input, which are filled with a
/// constant instead of being interpolated.
#[derive(Clone, Debug)]
pub struct AxisPlan {
    pub window: usize,
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub extrapolated: Vec<bool>,
}

/// Builds the resampling plan for one axis. `coord` maps an output index to a
/// source coordinate, or `None` when it falls outside the input. `weights`
/// fills a window for a fractional offset in `(0, 1]`, the tap at index `k`
/// sitting `k + 1 - window / 2` cells right of the source cell. With
/// `exclude_outside` the taps falling off the axis are dropped and the
/// remaining weights renormalized; otherwise they read the edge value.
pub fn plan_axis(
    len_in: usize,
    len_out: usize,
    window: usize,
    exclude_outside: bool,
    coord: impl Fn(usize) -> Option<f32>,
    weights: impl Fn(f32, &mut [f32]),
) -> AxisPlan {
    let mut plan = AxisPlan {
        window,
        indices: vec![0; window * len_out],
        weights: vec![0.0; window * len_out],
        extrapolated: vec![false; len_out],
    };
    for x in 0..len_out {
        let Some(x_in) = coord(x) else {
            plan.extrapolated[x] = true;
            continue;
        };
        let cell = x_in.ceil() - 1.0;
        let taps = &mut plan.weights[x * window..][..window];
        weights(x_in - cell, taps);
        let first = cell as isize + 1 - (window / 2) as isize;
        for (k, tap) in taps.iter_mut().enumerate() {
            let raw = first + k as isize;
            if exclude_outside && (raw < 0 || raw >= len_in as isize) {
                *tap = 0.0;
            }
            plan.indices[x * window + k] = raw.clamp(0, len_in as isize - 1) as usize;
        }
        if exclude_outside {
            let sum: f32 = taps.iter().sum();
            if sum != 0.0 {
                taps.iter_mut().for_each(|w| *w /= sum);
            }
        }
    }
    plan
}

/// Whether `plan` is exactly pixel replication — every output reading the
/// single input `x / scale` — which is the pattern
/// [`lower_nearest_integer_upsample`] produces. Only some coordinate transform
/// and tie-break pairs round that way, so the plan itself is the arbiter.
pub fn is_pixel_replication(plan: &AxisPlan, scale: usize) -> bool {
    !plan.extrapolated.contains(&true)
        && plan
            .indices
            .chunks_exact(plan.window)
            .zip(plan.weights.chunks_exact(plan.window))
            .enumerate()
            .all(|(x, (indices, weights))| {
                let mut taps = indices.iter().zip(weights).filter(|(_, w)| **w != 0.0);
                taps.next().is_some_and(|(i, w)| *w == 1.0 && *i == x / scale)
                    && taps.next().is_none()
            })
}

/// Resamples `axis` of a row-major `input` of shape `shape` into `output`,
/// which must be sized for the same shape with `axis` set to the plan length.
pub fn resample_axis(
    input: &[f32],
    shape: &[usize],
    axis: usize,
    plan: &AxisPlan,
    extrapolation_value: f32,
    output: &mut [f32],
) {
    let len_in = shape[axis];
    let len_out = plan.extrapolated.len();
    let inner: usize = shape[axis + 1..].iter().product();
    let window = plan.window;
    if len_in * inner == 0 || len_out * inner == 0 {
        return;
    }
    for (src, dst) in
        input.chunks_exact(len_in * inner).zip(output.chunks_exact_mut(len_out * inner))
    {
        for (x, dst) in dst.chunks_exact_mut(inner).enumerate() {
            if plan.extrapolated[x] {
                dst.fill(extrapolation_value);
                continue;
            }
            dst.fill(0.0);
            let indices = &plan.indices[x * window..][..window];
            let weights = &plan.weights[x * window..][..window];
            for (&i, &w) in indices.iter().zip(weights) {
                if w == 0.0 {
                    continue;
                }
                for (d, s) in dst.iter_mut().zip(&src[i * inner..][..inner]) {
                    *d += w * s;
                }
            }
        }
    }
}

/// Resamples `input` along the axes given by `scales`/`sizes`, the clean subset
/// of ONNX Resize: `interpolator` × `coord_transformer` × `nearest`, fixed
/// `cubic_coeff_a = -0.75`, no ROI and no `exclude_outside`. The ONNX op carries
/// the remaining edge cases and decltters into this op when it fits the subset.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Resize {
    pub coord_transformer: CoordTransformer,
    pub interpolator: Interpolator,
    pub nearest: Nearest,
    pub optional_scales_input: Option<usize>,
    pub optional_sizes_input: Option<usize>,
}

impl Resize {
    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        input_scale: Option<&Tensor>,
        input_sizes: Option<&Tensor>,
    ) -> TractResult<TVec<D>> {
        if let Some(scale) = input_scale
            && scale.len() == input_shape.len()
        {
            let mut shape = tvec!();
            for (i, s) in input_shape
                .iter()
                .zip(scale.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?.iter())
            {
                if s.round() == *s {
                    shape.push(i.clone() * (*s as usize));
                } else if let Ok(i) = i.to_usize() {
                    shape.push(((i as f32 * s) as usize).into());
                } else {
                    bail!(
                        "Can not compute output shape. inputs are {input_shape:?} and scale {scale:?}"
                    )
                }
            }
            return Ok(shape);
        }
        if let Some(sizes) = input_sizes
            && sizes.len() == input_shape.len()
        {
            return sizes
                .cast_to::<TDim>()?
                .try_as_plain()?
                .as_slice::<TDim>()?
                .iter()
                .map(|i| i.try_into())
                .collect();
        }
        bail!(
            "Neither sizes nor scales makes sense: input_shape: {:?}, scale: {:?}, sizes: {:?}",
            input_shape,
            input_scale,
            input_sizes,
        );
    }

    /// The resampling plan for one axis under this op's interpolator.
    pub fn plan_axis(&self, scale: f32, len_in: usize, len_out: usize) -> AxisPlan {
        let window = window_size(&self.interpolator, false, scale);
        let coord = |x| Some(self.coord_transformer.transform(x, scale, len_in, len_out));
        match self.interpolator {
            Interpolator::Linear => plan_axis(len_in, len_out, window, false, coord, |r, w| {
                linear_weights(r, scale, false, w)
            }),
            Interpolator::Cubic => plan_axis(len_in, len_out, window, false, coord, |r, w| {
                cubic_weights(r, scale, -0.75, false, w)
            }),
            Interpolator::Nearest => plan_axis(len_in, len_out, window, false, coord, |r, w| {
                let right = r == 1.0 || self.nearest.prefers_right(r);
                w[0] = !right as u8 as f32;
                w[1] = right as u8 as f32;
            }),
        }
    }
}

impl Op for Resize {
    fn name(&self) -> StaticName {
        "Resize".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Resize {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_dt = inputs[0].datum_type();
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            inputs[0].shape(),
            scales.map(|t| &**t),
            sizes.map(|t| &**t),
        )?;
        let scales: TVec<f32> = if let Some(scales) = scales.filter(|s| s.len() == inputs[0].rank())
        {
            scales.try_as_plain()?.as_slice::<f32>()?.into()
        } else {
            output_shape.iter().zip(inputs[0].shape()).map(|(o, i)| *o as f32 / *i as f32).collect()
        };
        let input = inputs.remove(0).into_tensor();
        let input = input.cast_to::<f32>()?;
        let mut shape: TVec<usize> = input.shape().into();
        let mut data: Vec<f32> = input.try_as_plain()?.as_slice::<f32>()?.to_vec();
        for (axis, scale) in scales.into_iter().enumerate() {
            let (len_in, len_out) = (shape[axis], output_shape[axis]);
            if len_in == len_out && scale == 1.0 {
                continue;
            }
            let plan = self.plan_axis(scale, len_in, len_out);
            let mut resampled = vec![0f32; data.len() / len_in * len_out];
            resample_axis(&data, &shape, axis, &plan, 0.0, &mut resampled);
            data = resampled;
            shape[axis] = len_out;
        }
        let out = tract_ndarray::ArrayD::from_shape_vec(&*shape, data)?.into_tensor();
        let out =
            if out.datum_type() == input_dt { out } else { out.cast_to_dt(input_dt)?.into_owned() };
        Ok(tvec!(out.into_tvalue()))
    }
}

impl TypedOp for Resize {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            &inputs[0].shape,
            scales.and_then(|f| f.konst.as_deref()),
            sizes.and_then(|f| f.konst.as_deref()),
        )?;
        Ok(tvec!(inputs[0].datum_type.fact(&output_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        rule_if!(matches!(self.interpolator, Interpolator::Nearest));
        rule_if_some!(scales_input = self.optional_scales_input);
        let scales_fact = model.outlet_fact(node.inputs[scales_input])?;
        rule_if_some!(scales_tensor = &scales_fact.konst);
        let scales: Vec<f32> =
            scales_tensor.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?.to_vec();
        let int_scales: Vec<usize> = scales.iter().map(|&s| s.round() as usize).collect();
        rule_if!(
            scales.iter().zip(&int_scales).all(|(&s, &i)| (s - i as f32).abs() <= 1e-5 && i != 0)
        );
        rule_if!(int_scales.iter().any(|&s| s != 1));
        let input_shape = &model.outlet_fact(node.inputs[0])?.shape;
        for (axis, &scale) in int_scales.iter().enumerate().filter(|&(_, &s)| s > 1) {
            let Some(len_in) = probe_length(&self.coord_transformer, &input_shape[axis]) else {
                return Ok(None);
            };
            rule_if!(is_pixel_replication(
                &self.plan_axis(scale as f32, len_in, len_in * scale),
                scale
            ));
        }

        lower_nearest_integer_upsample(model, node, &int_scales)
    }
}

/// An axis length to build a probe plan on. `HalfPixel` and `Asymmetric` map
/// coordinates without consulting the axis lengths, so a symbolic axis can
/// still be probed on a stand-in; the others cannot.
pub fn probe_length(coord_transformer: &CoordTransformer, len: &TDim) -> Option<usize> {
    len.to_usize().ok().or(match coord_transformer {
        CoordTransformer::HalfPixel | CoordTransformer::Asymmetric => Some(4),
        _ => None,
    })
}

/// Lowers a nearest-neighbour integer upsample to Reshape → Tile → Reshape: each
/// upsampled axis is split into a size-1 axis, tiled by its scale, then merged
/// back. Shared by the core and ONNX Resize declutters.
pub fn lower_nearest_integer_upsample(
    model: &TypedModel,
    node: &TypedNode,
    int_scales: &[usize],
) -> TractResult<Option<TypedModelPatch>> {
    let input_fact = model.outlet_fact(node.inputs[0])?;
    let input_shape = &input_fact.shape;

    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[0])?;

    let mut from_dims: TVec<TDim> = tvec![];
    let mut to_dims: TVec<TDim> = tvec![];
    let mut tile_multipliers: TVec<TDim> = tvec![];
    let mut first_upsampled = None;

    for (i, &scale) in int_scales.iter().enumerate() {
        from_dims.push(input_shape[i].clone());
        to_dims.push(input_shape[i].clone());
        tile_multipliers.push(1.into());
        if scale > 1 {
            if first_upsampled.is_none() {
                first_upsampled = Some(i);
            }
            to_dims.push(1.into());
            tile_multipliers.push(scale.into());
        }
    }

    if to_dims.len() > from_dims.len() {
        let first = first_upsampled.unwrap();
        wire = patch.wire_node(
            format!("{}.reshape_pre", node.name),
            AxisOp::Reshape(first, from_dims[first..].into(), to_dims[first..].into()),
            &[wire],
        )?[0];
    }

    wire = patch.wire_node(
        format!("{}.tile", node.name),
        Tile { multipliers: tile_multipliers },
        &[wire],
    )?[0];

    let tiled_shape: TVec<TDim> = to_dims
        .iter()
        .zip(int_scales.iter().flat_map(|&s| if s > 1 { vec![1usize, s] } else { vec![1] }))
        .map(|(d, s)| d.clone() * s)
        .collect();
    let mut final_dims: TVec<TDim> = tvec![];
    let mut idx = 0;
    for &scale in int_scales {
        if scale > 1 {
            final_dims.push(tiled_shape[idx].clone() * tiled_shape[idx + 1].clone());
            idx += 2;
        } else {
            final_dims.push(tiled_shape[idx].clone());
            idx += 1;
        }
    }

    if tiled_shape.len() > final_dims.len() {
        let first = first_upsampled.unwrap();
        wire = patch.wire_node(
            format!("{}.reshape_post", node.name),
            AxisOp::Reshape(first, tiled_shape[first..].into(), final_dims[first..].into()),
            &[wire],
        )?[0];
    }

    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_kernel_properties() {
        let a = -0.75f32;
        assert!((cubic_kernel(0.0, a) - 1.0).abs() < 1e-6);
        assert!(cubic_kernel(2.0, a).abs() < 1e-6);
        assert!(cubic_kernel(3.0, a).abs() < 1e-6);

        for t_int in 0..=100 {
            let t = t_int as f32 / 100.0;
            let sum = cubic_kernel(t + 1.0, a)
                + cubic_kernel(t, a)
                + cubic_kernel(1.0 - t, a)
                + cubic_kernel(2.0 - t, a);
            assert!((sum - 1.0).abs() < 1e-5, "kernel weights must sum to 1.0, got {sum} at t={t}");
        }
    }

    fn cubic_resize(input: Tensor, scales: &[f32]) -> Tensor {
        let scales = tract_ndarray::Array1::from(scales.to_vec()).into_tensor();
        let op = Resize {
            coord_transformer: CoordTransformer::HalfPixel,
            interpolator: Interpolator::Cubic,
            nearest: Nearest::Floor,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        };
        op.eval(tvec!(input.into_tvalue(), scales.into_tvalue())).unwrap().remove(0).into_tensor()
    }

    #[test]
    fn cubic_resize_1d_upsample() {
        let out = cubic_resize(tract_ndarray::arr1(&[0.0f32, 1.0, 2.0, 3.0]).into_tensor(), &[2.0]);
        let plain = out.try_as_plain().unwrap();
        let output = plain.as_slice::<f32>().unwrap();
        assert_eq!(output.len(), 8);
        assert!((output[0] - (-0.10546875)).abs() < 1e-4, "got {}", output[0]);
    }

    #[test]
    fn cubic_resize_2d_upsample() {
        let out = cubic_resize(
            tract_ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_tensor(),
            &[2.0, 2.0],
        );
        assert_eq!(out.shape(), &[4, 4]);
    }

    fn replicates(coord_transformer: CoordTransformer, nearest: Nearest, scale: usize) -> bool {
        let op = Resize {
            coord_transformer,
            interpolator: Interpolator::Nearest,
            nearest,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        };
        is_pixel_replication(&op.plan_axis(scale as f32, 4, 4 * scale), scale)
    }

    #[test]
    fn only_some_nearest_modes_replicate_pixels() {
        assert!(replicates(CoordTransformer::Asymmetric, Nearest::Floor, 2));
        assert!(replicates(CoordTransformer::HalfPixel, Nearest::RoundPreferCeil, 2));
        assert!(replicates(CoordTransformer::HalfPixel, Nearest::RoundPreferCeil, 3));
        assert!(!replicates(CoordTransformer::HalfPixel, Nearest::Floor, 2));
        assert!(!replicates(CoordTransformer::Asymmetric, Nearest::RoundPreferCeil, 2));
    }
}
