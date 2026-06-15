use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::nn::resize::{
    self, CoordTransformer, Interpolator, cubic_kernel, lower_nearest_integer_upsample,
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Resize {
    pub axes: Option<Vec<i64>>,
    pub coord_transformer: CoordTransformer,
    pub interpolator: Interpolator,
    pub nearest: Nearest,
    pub cubic_coeff_a_bits: u32,
    pub exclude_outside: bool,
    pub optional_roi_input: Option<usize>,
    pub optional_scales_input: Option<usize>,
    pub optional_sizes_input: Option<usize>,
}

impl Resize {
    pub fn cubic_coeff_a(&self) -> f32 {
        f32::from_bits(self.cubic_coeff_a_bits)
    }

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

    /// The clean subset reachable by `tract_core::ops::nn::resize::Resize`:
    /// default `cubic_coeff_a`, no `exclude_outside`, no ROI and a nearest mode
    /// core understands. `None` keeps the op as an ONNX edge-case op.
    fn as_core(&self) -> Option<resize::Resize> {
        if self.exclude_outside || self.optional_roi_input.is_some() {
            return None;
        }
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
            axes: self.axes.clone(),
            coord_transformer: self.coord_transformer.clone(),
            interpolator: self.interpolator.clone(),
            nearest,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        })
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
        let mut data = if input.datum_type() == f32::datum_type() {
            input.into_plain_array::<f32>()?
        } else {
            input.cast_to::<f32>()?.into_owned().into_plain_array::<f32>()?
        };
        for (axis, scale) in scales.into_iter().enumerate().filter(|(_, s)| *s != 1.0) {
            let mut new_shape: TVec<usize> = data.shape().into();
            new_shape[axis] = output_shape[axis];
            let input_len = data.shape()[axis];
            data = match self.interpolator {
                Interpolator::Cubic => {
                    let a = self.cubic_coeff_a();
                    let exclude = self.exclude_outside;
                    tract_ndarray::ArrayD::from_shape_fn(&*new_shape, |co_o| -> f32 {
                        let x_out = co_o[axis];
                        let x_in = self.coord_transformer.transform(
                            x_out,
                            scale,
                            input_len,
                            new_shape[axis],
                        );
                        let x_floor = x_in.floor() as isize;
                        let t = x_in - x_floor as f32;
                        let mut co_i = co_o;
                        let mut weights = [0.0f32; 4];
                        let mut values = [0.0f32; 4];
                        for (i, j) in (-1..=2isize).enumerate() {
                            let raw_idx = x_floor + j;
                            let w = cubic_kernel(t - j as f32, a);
                            if exclude && (raw_idx < 0 || raw_idx >= input_len as isize) {
                                weights[i] = 0.0;
                            } else {
                                weights[i] = w;
                                let idx = raw_idx.clamp(0, input_len as isize - 1) as usize;
                                co_i[axis] = idx;
                                values[i] = data[&co_i];
                            }
                        }
                        if exclude {
                            let sum: f32 = weights.iter().sum();
                            if sum != 0.0 {
                                for w in &mut weights {
                                    *w /= sum;
                                }
                            }
                        }
                        weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
                    })
                }
                _ => tract_ndarray::ArrayD::from_shape_fn(&*new_shape, |co_o| -> f32 {
                    let x_out = co_o[axis];
                    let x_in =
                        self.coord_transformer.transform(x_out, scale, input_len, new_shape[axis]);
                    let mut co_i = co_o;
                    let x_floor = x_in.floor() as isize;
                    let x_left = x_floor.clamp(0, input_len as isize - 1) as usize;
                    co_i[axis] = x_left;
                    let y_left = data[&co_i];
                    let x_right = (x_floor + 1).clamp(0, input_len as isize - 1) as usize;
                    co_i[axis] = x_right;
                    let y_right = data[&co_i];
                    let x_frac = x_in - x_floor as f32;
                    match self.interpolator {
                        Interpolator::Linear => y_left * (1.0 - x_frac) + y_right * x_frac,
                        Interpolator::Nearest => {
                            if self.nearest.prefers_right(x_frac) {
                                y_right
                            } else {
                                y_left
                            }
                        }
                        Interpolator::Cubic => unreachable!(),
                    }
                }),
            }
        }
        let out = data.into_tensor();
        let out =
            if out.datum_type() == input_dt { out } else { out.cast_to_dt(input_dt)?.into_owned() };
        Ok(tvec!(out.into_tvalue()))
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
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(mut core_op) = self.as_core() {
            let rank = model.outlet_fact(node.inputs[0])?.rank();
            let is_active = |ix: usize| -> bool {
                model
                    .outlet_fact(node.inputs[ix])
                    .ok()
                    .and_then(|f| f.konst.as_ref())
                    .map(|k| k.len() == rank)
                    .unwrap_or(false)
            };
            let active = self
                .optional_scales_input
                .filter(|&ix| is_active(ix))
                .map(|ix| (ix, false))
                .or_else(|| {
                    self.optional_sizes_input.filter(|&ix| is_active(ix)).map(|ix| (ix, true))
                });
            if let Some((ix, use_sizes)) = active {
                core_op.optional_scales_input = (!use_sizes).then_some(1);
                core_op.optional_sizes_input = use_sizes.then_some(1);
                let mut patch = TypedModelPatch::default();
                let data = patch.tap_model(model, node.inputs[0])?;
                let aux = patch.tap_model(model, node.inputs[ix])?;
                let wire = patch.wire_node(&node.name, core_op, &[data, aux])?;
                patch.shunt_outside(model, node.id.into(), wire[0])?;
                return Ok(Some(patch));
            }
        }

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
        TypeName::String.named("coord_transformer").default("half_pixel"),
        TypeName::String.named("interpolator").default("nearest"),
        TypeName::String.named("nearest_mode").default("floor"),
        TypeName::Scalar.named("cubic_coeff_a").default(-0.75f32),
        TypeName::Logical.named("exclude_outside").default(false),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Resize) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let scales = if let Some(scales_ix) = op.optional_scales_input {
        ast.mapping[&node.inputs[scales_ix]].clone()
    } else if let Some(sizes_ix) = op.optional_sizes_input {
        let input_shape = ast.model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let sizes_fact = ast.model.outlet_fact(node.inputs[sizes_ix])?;
        let sizes =
            sizes_fact.konst.as_ref().context("sizes must be a constant for NNEF export")?;
        let sizes = sizes.cast_to::<f32>()?;
        let sizes = sizes.try_as_plain()?.as_slice::<f32>()?;
        let scales: Vec<f32> = input_shape
            .iter()
            .zip(sizes.iter())
            .map(|(i, s)| i.to_usize().map(|i| *s / i as f32).unwrap_or(1.0))
            .collect();
        let scales_tensor = tract_ndarray::arr1(&scales).into_arc_tensor();
        ast.konst_variable(format!("{}.scales", node.name), &scales_tensor)?
    } else {
        bail!("Resize op has neither scales nor sizes input")
    };
    Ok(Some(invocation(
        "tract_onnx_resize",
        &[input, scales],
        &[
            ("coord_transformer", string(op.coord_transformer.as_str())),
            ("interpolator", string(op.interpolator.as_str())),
            ("nearest_mode", string(op.nearest.as_str())),
            ("cubic_coeff_a", numeric(op.cubic_coeff_a())),
            ("exclude_outside", logical(op.exclude_outside)),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scales = invocation.named_arg_as(builder, "scales")?;
    let coord_transformer: String = invocation.named_arg_as(builder, "coord_transformer")?;
    let interpolator: String = invocation.named_arg_as(builder, "interpolator")?;
    let nearest_mode: String = invocation.named_arg_as(builder, "nearest_mode")?;
    let cubic_coeff_a: f32 = invocation.named_arg_as(builder, "cubic_coeff_a")?;
    let exclude_outside: bool = invocation.named_arg_as(builder, "exclude_outside")?;

    let op = Resize {
        axes: None,
        coord_transformer: CoordTransformer::parse(&coord_transformer)?,
        interpolator: Interpolator::parse(&interpolator)?,
        nearest: Nearest::parse(&nearest_mode)?,
        cubic_coeff_a_bits: cubic_coeff_a.to_bits(),
        exclude_outside,
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
    };

    builder.wire(op, &[input, scales])
}
