use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::nn::resize::{
    self, AxisPlan, CoordTransformer, Interpolator, cubic_weights, is_pixel_replication,
    linear_weights, lower_nearest_integer_upsample, plan_axis, probe_length, resample_axis,
    window_size,
};

/// Nearest-neighbour tie-breaking, the full ONNX set. `Floor` and
/// `RoundPreferCeil` are also supported by `tract_core::ops::nn::resize`; the
/// other two stay here in the ONNX edge-case op.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Nearest {
    Floor,
    Ceil,
    RoundPreferFloor,
    RoundPreferCeil,
}

impl Nearest {
    fn prefers_right(&self, x_ratio: f32) -> bool {
        match self {
            Nearest::Floor => false,
            Nearest::Ceil => true,
            Nearest::RoundPreferFloor => x_ratio > 0.5,
            Nearest::RoundPreferCeil => x_ratio >= 0.5,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Nearest::Floor => "floor",
            Nearest::Ceil => "ceil",
            Nearest::RoundPreferFloor => "round_prefer_floor",
            Nearest::RoundPreferCeil => "round_prefer_ceil",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "floor" => Nearest::Floor,
            "ceil" => Nearest::Ceil,
            "round_prefer_floor" => Nearest::RoundPreferFloor,
            "round_prefer_ceil" => Nearest::RoundPreferCeil,
            s => bail!("nearest_mode: {s}"),
        })
    }
}

/// ONNX `coordinate_transformation_mode`. Every mode but `tf_crop_and_resize`
/// inverts without an input ROI and is shared with tract-core.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CoordTransform {
    Plain(CoordTransformer),
    TfCropAndResize,
}

impl CoordTransform {
    pub fn as_str(&self) -> &'static str {
        match self {
            CoordTransform::Plain(t) => t.as_str(),
            CoordTransform::TfCropAndResize => "tf_crop_and_resize",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "tf_crop_and_resize" => CoordTransform::TfCropAndResize,
            s => CoordTransform::Plain(CoordTransformer::parse(s)?),
        })
    }
}

/// ONNX `keep_aspect_ratio_policy`: reconciles the requested `sizes` into a
/// single scale shared by every resized axis. Ignored when `scales` drives the
/// resize.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AspectRatio {
    Stretch,
    NotLarger,
    NotSmaller,
}

impl AspectRatio {
    pub fn as_str(&self) -> &'static str {
        match self {
            AspectRatio::Stretch => "stretch",
            AspectRatio::NotLarger => "not_larger",
            AspectRatio::NotSmaller => "not_smaller",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "stretch" => AspectRatio::Stretch,
            "not_larger" => AspectRatio::NotLarger,
            "not_smaller" => AspectRatio::NotSmaller,
            s => bail!("keep_aspect_ratio_policy: {s}"),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Resize {
    pub axes: Option<Vec<i64>>,
    pub coord_transformer: CoordTransform,
    pub interpolator: Interpolator,
    pub nearest: Nearest,
    pub antialias: bool,
    pub cubic_coeff_a_bits: u32,
    pub exclude_outside: bool,
    pub extrapolation_value_bits: u32,
    pub keep_aspect_ratio_policy: AspectRatio,
    pub optional_roi_input: Option<usize>,
    pub optional_scales_input: Option<usize>,
    pub optional_sizes_input: Option<usize>,
}

impl Resize {
    pub fn cubic_coeff_a(&self) -> f32 {
        f32::from_bits(self.cubic_coeff_a_bits)
    }

    pub fn extrapolation_value(&self) -> f32 {
        f32::from_bits(self.extrapolation_value_bits)
    }

    /// The axes `scales`, `sizes` and `roi` describe, negatives resolved. Axes
    /// left out keep their input length.
    pub fn resized_axes(&self, rank: usize) -> TractResult<TVec<usize>> {
        let Some(axes) = &self.axes else { return Ok((0..rank).collect()) };
        axes.iter()
            .map(|a| {
                let a = if *a < 0 { a + rank as i64 } else { *a };
                ensure!((0..rank as i64).contains(&a), "Resize axes {axes:?} out of rank {rank}");
                Ok(a as usize)
            })
            .collect()
    }

    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        input_scale: Option<&Tensor>,
        input_sizes: Option<&Tensor>,
    ) -> TractResult<TVec<D>> {
        let axes = self.resized_axes(input_shape.len())?;
        let mut shape: TVec<D> = input_shape.into();
        if let Some(scale) = input_scale.filter(|s| s.len() == axes.len()) {
            let scale = scale.cast_to::<f32>()?;
            for (&axis, s) in axes.iter().zip(scale.try_as_plain()?.as_slice::<f32>()?) {
                let i = &input_shape[axis];
                shape[axis] = if s.round() == *s {
                    i.clone() * (*s as usize)
                } else if let Ok(i) = i.to_usize() {
                    ((i as f32 * s) as usize).into()
                } else {
                    bail!(
                        "Can not compute output shape. inputs are {input_shape:?} and scale {scale:?}"
                    )
                };
            }
            return Ok(shape);
        }
        if let Some(sizes) = input_sizes.filter(|s| s.len() == axes.len()) {
            let sizes = sizes.cast_to::<TDim>()?;
            let sizes = sizes.try_as_plain()?.as_slice::<TDim>()?;
            if self.keep_aspect_ratio_policy == AspectRatio::Stretch {
                for (&axis, s) in axes.iter().zip(sizes) {
                    shape[axis] = s.try_into()?;
                }
            } else {
                let scale = self.aspect_ratio_scale(input_shape, &axes, sizes)?;
                for &axis in &axes {
                    let len = input_shape[axis].to_usize()?;
                    shape[axis] = ((scale * len as f32 + 0.5) as usize).into();
                }
            }
            return Ok(shape);
        }
        bail!(
            "Neither sizes nor scales makes sense: input_shape: {:?}, scale: {:?}, sizes: {:?}",
            input_shape,
            input_scale,
            input_sizes,
        );
    }

    /// The scale every resized axis takes under a non-`Stretch` policy: the
    /// smallest requested ratio to stay within `sizes`, the largest to cover it.
    fn aspect_ratio_scale<D: DimLike>(
        &self,
        input_shape: &[D],
        axes: &[usize],
        sizes: &[TDim],
    ) -> TractResult<f32> {
        let mut ratios: TVec<f32> = tvec!();
        for (&axis, size) in axes.iter().zip(sizes) {
            ratios.push(size.to_usize()? as f32 / input_shape[axis].to_usize()? as f32);
        }
        let pick = if self.keep_aspect_ratio_policy == AspectRatio::NotLarger {
            f32::min
        } else {
            f32::max
        };
        ratios.into_iter().reduce(pick).context("Resize sizes must not be empty")
    }

    /// Per-axis scale and output length over the full rank.
    fn resolve(
        &self,
        input_shape: &[usize],
        scales: Option<&Tensor>,
        sizes: Option<&Tensor>,
    ) -> TractResult<(TVec<f32>, TVec<usize>)> {
        let axes = self.resized_axes(input_shape.len())?;
        let output_shape = self.compute_output_shape(input_shape, scales, sizes)?;
        let mut per_axis: TVec<f32> = tvec!(1.0; input_shape.len());
        if let Some(scales) = scales.filter(|s| s.len() == axes.len()) {
            let scales = scales.cast_to::<f32>()?;
            for (&axis, s) in axes.iter().zip(scales.try_as_plain()?.as_slice::<f32>()?) {
                per_axis[axis] = *s;
            }
        } else if self.keep_aspect_ratio_policy != AspectRatio::Stretch {
            let sizes =
                sizes.context("Resize aspect ratio policy needs sizes")?.cast_to::<TDim>()?;
            let scale =
                self.aspect_ratio_scale(input_shape, &axes, sizes.try_as_plain()?.as_slice()?)?;
            for &axis in &axes {
                per_axis[axis] = scale;
            }
        } else {
            for &axis in &axes {
                per_axis[axis] = output_shape[axis] as f32 / input_shape[axis] as f32;
            }
        }
        Ok((per_axis, output_shape))
    }

    /// Normalized ROI `(start, end)` per axis, `(0, 1)` for the axes left out.
    fn roi(&self, rank: usize, roi: Option<&Tensor>) -> TractResult<TVec<(f32, f32)>> {
        let axes = self.resized_axes(rank)?;
        let Some(roi) = roi.filter(|r| r.len() == 2 * axes.len()) else {
            bail!("Resize in tf_crop_and_resize mode needs a roi of 2 x {} elements", axes.len())
        };
        let roi = roi.cast_to::<f32>()?;
        let roi = roi.try_as_plain()?.as_slice::<f32>()?;
        let mut per_axis: TVec<(f32, f32)> = tvec!((0.0, 1.0); rank);
        for (i, &axis) in axes.iter().enumerate() {
            per_axis[axis] = (roi[i], roi[i + axes.len()]);
        }
        Ok(per_axis)
    }

    fn plan_axis(&self, scale: f32, len_in: usize, len_out: usize, roi: (f32, f32)) -> AxisPlan {
        let window = window_size(&self.interpolator, self.antialias, scale);
        let coord: Box<dyn Fn(usize) -> Option<f32>> = match &self.coord_transformer {
            CoordTransform::Plain(t) => {
                let t = t.clone();
                Box::new(move |x| Some(t.transform(x, scale, len_in, len_out)))
            }
            CoordTransform::TfCropAndResize => {
                let last = len_in as f32 - 1.0;
                let span = last * (roi.1 - roi.0);
                let width = scale * len_in as f32;
                Box::new(move |x| {
                    let offset =
                        if width == 1.0 { span / 2.0 } else { x as f32 * span / (width - 1.0) };
                    let x = offset + roi.0 * last;
                    (x >= 0.0 && x <= last).then_some(x)
                })
            }
        };
        let exclude = self.exclude_outside;
        let (antialias, a) = (self.antialias, self.cubic_coeff_a());
        match self.interpolator {
            Interpolator::Linear => plan_axis(len_in, len_out, window, exclude, coord, |r, w| {
                linear_weights(r, scale, antialias, w)
            }),
            Interpolator::Cubic => plan_axis(len_in, len_out, window, exclude, coord, |r, w| {
                cubic_weights(r, scale, a, antialias, w)
            }),
            Interpolator::Nearest => plan_axis(len_in, len_out, window, exclude, coord, |r, w| {
                let right = r == 1.0 || self.nearest.prefers_right(r);
                w[0] = !right as u8 as f32;
                w[1] = right as u8 as f32;
            }),
        }
    }

    /// The clean subset reachable by `tract_core::ops::nn::resize::Resize`:
    /// default `cubic_coeff_a`, no `exclude_outside`, no antialiasing, no ROI,
    /// a stretching aspect ratio and a nearest mode core understands. `None`
    /// keeps the op as an ONNX edge-case op.
    fn as_core(&self) -> Option<resize::Resize> {
        if self.exclude_outside
            || self.antialias
            || self.keep_aspect_ratio_policy != AspectRatio::Stretch
        {
            return None;
        }
        let CoordTransform::Plain(coord_transformer) = &self.coord_transformer else {
            return None;
        };
        if self.interpolator == Interpolator::Cubic && self.cubic_coeff_a() != -0.75 {
            return None;
        }
        let nearest = match self.nearest {
            Nearest::Floor => resize::Nearest::Floor,
            Nearest::RoundPreferCeil => resize::Nearest::RoundPreferCeil,
            Nearest::Ceil | Nearest::RoundPreferFloor
                if self.interpolator != Interpolator::Nearest =>
            {
                resize::Nearest::Floor
            }
            _ => return None,
        };
        Some(resize::Resize {
            coord_transformer: coord_transformer.clone(),
            interpolator: self.interpolator.clone(),
            nearest,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        })
    }

    /// Spreads a per-`axes` `scales`/`sizes` constant over the full rank, so the
    /// core op — which carries no `axes` of its own — can take the node over.
    fn full_rank_aux(
        &self,
        input_shape: &ShapeFact,
        axes: &[usize],
        konst: &Tensor,
        sizes: bool,
    ) -> TractResult<Arc<Tensor>> {
        if sizes {
            let mut full: Vec<i64> = vec![0; input_shape.rank()];
            for (slot, dim) in full.iter_mut().zip(input_shape.iter()) {
                *slot = dim.to_usize()? as i64;
            }
            let konst = konst.cast_to::<i64>()?;
            for (&axis, v) in axes.iter().zip(konst.try_as_plain()?.as_slice::<i64>()?) {
                full[axis] = *v;
            }
            Ok(tract_ndarray::arr1(&full).into_arc_tensor())
        } else {
            let mut full = vec![1.0f32; input_shape.rank()];
            let konst = konst.cast_to::<f32>()?;
            for (&axis, v) in axes.iter().zip(konst.try_as_plain()?.as_slice::<f32>()?) {
                full[axis] = *v;
            }
            Ok(tract_ndarray::arr1(&full).into_arc_tensor())
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
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_dt = inputs[0].datum_type();
        let rank = inputs[0].rank();
        let tf_crop = self.coord_transformer == CoordTransform::TfCropAndResize;
        let roi = if tf_crop {
            self.roi(rank, self.optional_roi_input.and_then(|ix| inputs.get(ix)).map(|t| &**t))?
        } else {
            tvec!((0.0, 1.0); rank)
        };
        let (scales, output_shape) = self.resolve(
            inputs[0].shape(),
            self.optional_scales_input.and_then(|ix| inputs.get(ix)).map(|t| &**t),
            self.optional_sizes_input.and_then(|ix| inputs.get(ix)).map(|t| &**t),
        )?;
        let input = inputs.remove(0).into_tensor();
        let input = input.cast_to::<f32>()?;
        let mut shape: TVec<usize> = input.shape().into();
        let mut data: Vec<f32> = input.try_as_plain()?.as_slice::<f32>()?.to_vec();
        for (axis, scale) in scales.into_iter().enumerate() {
            let (len_in, len_out) = (shape[axis], output_shape[axis]);
            if len_in == len_out && scale == 1.0 && !tf_crop {
                continue;
            }
            let plan = self.plan_axis(scale, len_in, len_out, roi[axis]);
            let mut resampled = vec![0f32; data.len() / len_in * len_out];
            resample_axis(&data, &shape, axis, &plan, self.extrapolation_value(), &mut resampled);
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
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let rank = input_fact.rank();
        let axes = self.resized_axes(rank)?;
        if let Some(mut core_op) = self.as_core() {
            let konst = |ix: usize| -> Option<Arc<Tensor>> {
                model
                    .outlet_fact(node.inputs[ix])
                    .ok()?
                    .konst
                    .clone()
                    .filter(|k| k.len() == axes.len())
            };
            let active = self
                .optional_scales_input
                .filter(|&ix| konst(ix).is_some())
                .map(|ix| (ix, false))
                .or_else(|| {
                    self.optional_sizes_input.filter(|&ix| konst(ix).is_some()).map(|ix| (ix, true))
                });
            if let Some((ix, use_sizes)) = active {
                core_op.optional_scales_input = (!use_sizes).then_some(1);
                core_op.optional_sizes_input = use_sizes.then_some(1);
                let mut patch = TypedModelPatch::default();
                let data = patch.tap_model(model, node.inputs[0])?;
                let aux = if axes.len() == rank {
                    patch.tap_model(model, node.inputs[ix])?
                } else {
                    let full = self.full_rank_aux(
                        &input_fact.shape,
                        &axes,
                        &konst(ix).unwrap(),
                        use_sizes,
                    )?;
                    patch.add_const(format!("{}.resize_aux", node.name), full)?
                };
                let wire = patch.wire_node(&node.name, core_op, &[data, aux])?;
                patch.shunt_outside(model, node.id.into(), wire[0])?;
                return Ok(Some(patch));
            }
        }

        rule_if!(matches!(self.interpolator, Interpolator::Nearest));
        rule_if_some!(scales_input = self.optional_scales_input);
        let scales_fact = model.outlet_fact(node.inputs[scales_input])?;
        rule_if_some!(scales_tensor = &scales_fact.konst);
        rule_if!(scales_tensor.len() == rank);
        let scales: Vec<f32> =
            scales_tensor.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?.to_vec();
        let int_scales: Vec<usize> = scales.iter().map(|&s| s.round() as usize).collect();
        rule_if!(
            scales.iter().zip(&int_scales).all(|(&s, &i)| (s - i as f32).abs() <= 1e-5 && i != 0)
        );
        rule_if!(int_scales.iter().any(|&s| s != 1));
        let CoordTransform::Plain(coord_transformer) = &self.coord_transformer else {
            return Ok(None);
        };
        for (axis, &scale) in int_scales.iter().enumerate().filter(|&(_, &s)| s > 1) {
            let Some(len_in) = probe_length(coord_transformer, &input_fact.shape[axis]) else {
                return Ok(None);
            };
            rule_if!(is_pixel_replication(
                &self.plan_axis(scale as f32, len_in, len_in * scale, (0.0, 1.0)),
                scale
            ));
        }

        lower_nearest_integer_upsample(model, node, &int_scales)
    }
}

// --- NNEF serialization (edge-case op) ---

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_resize",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("scales"),
        TypeName::Scalar.tensor().named("roi").default(false),
        TypeName::String.named("coord_transformer").default("half_pixel"),
        TypeName::String.named("interpolator").default("nearest"),
        TypeName::String.named("nearest_mode").default("floor"),
        TypeName::Scalar.named("cubic_coeff_a").default(-0.75f32),
        TypeName::Logical.named("exclude_outside").default(false),
        TypeName::Logical.named("antialias").default(false),
        TypeName::Scalar.named("extrapolation_value").default(0.0f32),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Resize) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let input_shape = ast.model.outlet_fact(node.inputs[0])?.shape.to_tvec();
    let axes = op.resized_axes(input_shape.len())?;
    let passthrough = op.optional_scales_input.filter(|&ix| {
        axes.len() == input_shape.len()
            && ast
                .model
                .outlet_fact(node.inputs[ix])
                .map(|f| f.shape.volume() == axes.len().to_dim())
                .unwrap_or(false)
    });
    let scales = if let Some(ix) = passthrough {
        ast.mapping[&node.inputs[ix]].clone()
    } else {
        let output_shape = &node.outputs[0].fact.shape;
        let mut scales = vec![1.0f32; input_shape.len()];
        for &axis in &axes {
            let (i, o) = (input_shape[axis].to_usize()?, output_shape[axis].to_usize()?);
            scales[axis] = o as f32 / i as f32;
        }
        let scales = tract_ndarray::arr1(&scales).into_arc_tensor();
        ast.konst_variable(format!("{}.scales", node.name), &scales)?
    };
    let mut args = vec![
        ("coord_transformer", string(op.coord_transformer.as_str())),
        ("interpolator", string(op.interpolator.as_str())),
        ("nearest_mode", string(op.nearest.as_str())),
        ("cubic_coeff_a", numeric(op.cubic_coeff_a())),
        ("exclude_outside", logical(op.exclude_outside)),
        ("antialias", logical(op.antialias)),
        ("extrapolation_value", numeric(op.extrapolation_value())),
    ];
    if op.coord_transformer == CoordTransform::TfCropAndResize {
        let ix = op.optional_roi_input.context("tf_crop_and_resize needs a roi input")?;
        let roi = ast.model.outlet_fact(node.inputs[ix])?;
        let roi = roi.konst.as_ref().context("roi must be a constant for NNEF export")?;
        let roi = full_rank_roi(&axes, input_shape.len(), roi)?;
        let roi = ast.konst_variable(format!("{}.roi", node.name), &roi)?;
        args.push(("roi", (*roi).clone()));
    }
    Ok(Some(invocation("tract_onnx_resize", &[input, scales], &args)))
}

/// Spreads a per-`axes` ROI over the full rank, the layout the NNEF op expects.
fn full_rank_roi(axes: &[usize], rank: usize, roi: &Tensor) -> TractResult<Arc<Tensor>> {
    let roi = roi.cast_to::<f32>()?;
    let roi = roi.try_as_plain()?.as_slice::<f32>()?;
    let mut full = vec![0.0f32; rank];
    full.extend(std::iter::repeat_n(1.0f32, rank));
    for (i, &axis) in axes.iter().enumerate() {
        full[axis] = roi[i];
        full[rank + axis] = roi[i + axes.len()];
    }
    Ok(tract_ndarray::arr1(&full).into_arc_tensor())
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scales = invocation.named_arg_as(builder, "scales")?;
    let roi = invocation.optional_named_arg_as::<OutletId>(builder, "roi")?;
    let coord_transformer: String = invocation.named_arg_as(builder, "coord_transformer")?;
    let interpolator: String = invocation.named_arg_as(builder, "interpolator")?;
    let nearest_mode: String = invocation.named_arg_as(builder, "nearest_mode")?;
    let cubic_coeff_a: f32 = invocation.named_arg_as(builder, "cubic_coeff_a")?;
    let exclude_outside: bool = invocation.named_arg_as(builder, "exclude_outside")?;
    let antialias: bool = invocation.named_arg_as(builder, "antialias")?;
    let extrapolation_value: f32 = invocation.named_arg_as(builder, "extrapolation_value")?;

    let op = Resize {
        axes: None,
        coord_transformer: CoordTransform::parse(&coord_transformer)?,
        interpolator: Interpolator::parse(&interpolator)?,
        nearest: Nearest::parse(&nearest_mode)?,
        antialias,
        cubic_coeff_a_bits: cubic_coeff_a.to_bits(),
        exclude_outside,
        extrapolation_value_bits: extrapolation_value.to_bits(),
        keep_aspect_ratio_policy: AspectRatio::Stretch,
        optional_roi_input: roi.map(|_| 2),
        optional_scales_input: Some(1),
        optional_sizes_input: None,
    };

    let mut wires = tvec!(input, scales);
    wires.extend(roi);
    builder.wire(op, &wires)
}
