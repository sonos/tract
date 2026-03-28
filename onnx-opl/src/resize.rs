use tract_nnef::internal::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CoordTransformer {
    HalfPixel,
    AlignCorners,
    Asymmetric,
}

impl CoordTransformer {
    pub fn transform(&self, x_out: usize, scale: f32, len_in: usize, len_out: usize) -> f32 {
        match self {
            CoordTransformer::HalfPixel => (x_out as f32 + 0.5) / scale - 0.5,
            CoordTransformer::AlignCorners => {
                (x_out as f32 * (len_in as f32 - 1.0)) / (len_out as f32 - 1.0)
            }
            CoordTransformer::Asymmetric => (x_out as f32) / scale,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CoordTransformer::HalfPixel => "half_pixel",
            CoordTransformer::AlignCorners => "align_corners",
            CoordTransformer::Asymmetric => "asymmetric",
        }
    }

    pub fn from_str(s: &str) -> TractResult<Self> {
        Ok(match s {
            "half_pixel" => CoordTransformer::HalfPixel,
            "align_corners" => CoordTransformer::AlignCorners,
            "asymmetric" => CoordTransformer::Asymmetric,
            s => bail!("coordinate_transformation_mode: {s}"),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Interpolator {
    Linear,
    Nearest,
}

impl Interpolator {
    pub fn interpolate(
        &self,
        y_left: f32,
        y_right: f32,
        x_ratio: f32,
        nearest_mode: Nearest,
    ) -> f32 {
        match self {
            Interpolator::Linear => y_left * (1.0 - x_ratio) + y_right * x_ratio,
            Interpolator::Nearest => match nearest_mode {
                Nearest::Floor => y_left,
                Nearest::Ceil => y_right,
                Nearest::RoundPreferFloor => {
                    if x_ratio <= 0.5 {
                        y_left
                    } else {
                        y_right
                    }
                }
                Nearest::RoundPreferCeil => {
                    if x_ratio < 0.5 {
                        y_left
                    } else {
                        y_right
                    }
                }
            },
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Interpolator::Linear => "linear",
            Interpolator::Nearest => "nearest",
        }
    }

    pub fn from_str(s: &str) -> TractResult<Self> {
        Ok(match s {
            "linear" => Interpolator::Linear,
            "nearest" => Interpolator::Nearest,
            s => bail!("mode: {s}"),
        })
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Nearest {
    Floor,
    Ceil,
    RoundPreferFloor,
    RoundPreferCeil,
}

impl Nearest {
    pub fn as_str(&self) -> &'static str {
        match self {
            Nearest::Floor => "floor",
            Nearest::Ceil => "ceil",
            Nearest::RoundPreferFloor => "round_prefer_floor",
            Nearest::RoundPreferCeil => "round_prefer_ceil",
        }
    }

    pub fn from_str(s: &str) -> TractResult<Self> {
        Ok(match s {
            "floor" => Nearest::Floor,
            "ceil" => Nearest::Ceil,
            "round_prefer_floor" => Nearest::RoundPreferFloor,
            "round_prefer_ceil" => Nearest::RoundPreferCeil,
            s => bail!("nearest_mode: {s}"),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Resize {
    pub axes: Option<Vec<i64>>,
    pub coord_transformer: CoordTransformer,
    pub interpolator: Interpolator,
    pub nearest: Nearest,
    pub optional_roi_input: Option<usize>,
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
        if let Some(scale) = input_scale {
            if scale.len() == input_shape.len() {
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
        }
        if let Some(sizes) = input_sizes {
            if sizes.len() == input_shape.len() {
                return sizes
                    .cast_to::<TDim>()?
                    .try_as_plain()?
                    .as_slice::<TDim>()?
                    .iter()
                    .map(|i| i.try_into())
                    .collect();
            }
        }
        bail!(
            "Neither sizes nor scales makes sense: input_shape: {:?}, scale: {:?}, sizes: {:?}",
            input_shape,
            input_scale,
            input_sizes,
        );
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
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            inputs[0].shape(),
            scales.map(|t| &**t),
            sizes.map(|t| &**t),
        )?;
        let scales: TVec<f32> = if let Some(scales) = scales {
            scales.try_as_plain()?.as_slice::<f32>()?.into()
        } else {
            output_shape.iter().zip(inputs[0].shape()).map(|(o, i)| *o as f32 / *i as f32).collect()
        };
        let mut data = inputs.remove(0).into_tensor().into_plain_array::<f32>()?;
        for (axis, scale) in scales.into_iter().enumerate().filter(|(_, s)| *s != 1.0) {
            let mut new_shape: TVec<usize> = data.shape().into();
            new_shape[axis] = output_shape[axis];
            data = tract_ndarray::ArrayD::from_shape_fn(&*new_shape, |co_o| -> f32 {
                let x_out = co_o[axis];
                let x_in = self.coord_transformer.transform(
                    x_out,
                    scale,
                    data.shape()[axis],
                    new_shape[axis],
                );
                let mut co_i = co_o;
                let x_left = (x_in as usize).clamp(0, data.shape()[axis] - 1);
                co_i[axis] = x_left;
                let y_left = data[&co_i];
                let x_right = (x_left + 1).min(data.shape()[axis] - 1);
                co_i[axis] = x_right;
                let y_right = data[&co_i];
                let x_frac = x_in - x_left as f32;
                self.interpolator.interpolate(y_left, y_right, x_frac, self.nearest)
            })
        }
        Ok(tvec!(data.into_tvalue()))
    }
}

impl TypedOp for Resize {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let _roi = self.optional_roi_input.and_then(|ix| inputs.get(ix));
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
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}

// --- NNEF serialization ---

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
        TypeName::String.named("coord_transformer").default("half_pixel"),
        TypeName::String.named("interpolator").default("nearest"),
        TypeName::String.named("nearest_mode").default("floor"),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Resize) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let scales =
        ast.mapping[&node.inputs[op.optional_scales_input.context("no scales input")?]].clone();
    Ok(Some(invocation(
        "tract_onnx_resize",
        &[input, scales],
        &[
            ("coord_transformer", string(op.coord_transformer.as_str())),
            ("interpolator", string(op.interpolator.as_str())),
            ("nearest_mode", string(op.nearest.as_str())),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scales = invocation.named_arg_as(builder, "scales")?;
    let coord_transformer: String = invocation.named_arg_as(builder, "coord_transformer")?;
    let interpolator: String = invocation.named_arg_as(builder, "interpolator")?;
    let nearest_mode: String = invocation.named_arg_as(builder, "nearest_mode")?;

    let op = Resize {
        axes: None,
        coord_transformer: CoordTransformer::from_str(&coord_transformer)?,
        interpolator: Interpolator::from_str(&interpolator)?,
        nearest: Nearest::from_str(&nearest_mode)?,
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
    };

    builder.wire(op, &[input, scales])
}
