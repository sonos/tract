use tract_nnef::internal::*;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CoordTransformer {
    HalfPixel,
    AlignCorners,
    Asymmetric,
    PytorchHalfPixel,
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
                    0.0
                }
            }
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CoordTransformer::HalfPixel => "half_pixel",
            CoordTransformer::AlignCorners => "align_corners",
            CoordTransformer::Asymmetric => "asymmetric",
            CoordTransformer::PytorchHalfPixel => "pytorch_half_pixel",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "half_pixel" => CoordTransformer::HalfPixel,
            "align_corners" => CoordTransformer::AlignCorners,
            "asymmetric" => CoordTransformer::Asymmetric,
            "pytorch_half_pixel" => CoordTransformer::PytorchHalfPixel,
            s => bail!("coordinate_transformation_mode: {s}"),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Interpolator {
    Linear,
    Nearest,
    Cubic,
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
            Interpolator::Cubic => {
                unreachable!("cubic interpolation uses a 4-tap kernel, not the 2-tap path")
            }
        }
    }

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

fn cubic_kernel(s: f32, a: f32) -> f32 {
    let abs_s = s.abs();
    if abs_s <= 1.0 {
        (a + 2.0) * abs_s * abs_s * abs_s - (a + 3.0) * abs_s * abs_s + 1.0
    } else if abs_s <= 2.0 {
        a * abs_s * abs_s * abs_s - 5.0 * a * abs_s * abs_s + 8.0 * a * abs_s - 4.0 * a
    } else {
        0.0
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
        let scales: TVec<f32> = if let Some(scales) = scales.filter(|s| s.len() == inputs[0].rank())
        {
            scales.try_as_plain()?.as_slice::<f32>()?.into()
        } else {
            output_shape.iter().zip(inputs[0].shape()).map(|(o, i)| *o as f32 / *i as f32).collect()
        };
        let mut data = inputs.remove(0).into_tensor().into_plain_array::<f32>()?;
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
                    let x_left = (x_in as usize).clamp(0, input_len - 1);
                    co_i[axis] = x_left;
                    let y_left = data[&co_i];
                    let x_right = (x_left + 1).min(input_len - 1);
                    co_i[axis] = x_right;
                    let y_right = data[&co_i];
                    let x_frac = x_in - x_left as f32;
                    self.interpolator.interpolate(y_left, y_right, x_frac, self.nearest)
                }),
            }
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
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        // Lower nearest-neighbor integer-scale upsamples to Reshape → Tile → Reshape
        rule_if!(matches!(self.interpolator, Interpolator::Nearest));
        rule_if_some!(scales_input = self.optional_scales_input);
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let scales_fact = model.outlet_fact(node.inputs[scales_input])?;
        rule_if_some!(scales_tensor = &scales_fact.konst);
        let scales: Vec<f32> =
            scales_tensor.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?.to_vec();

        // Check all scales are positive integers
        let int_scales: Vec<usize> = scales.iter().map(|&s| s.round() as usize).collect();
        rule_if!(
            scales.iter().zip(&int_scales).all(|(&s, &i)| (s - i as f32).abs() <= 1e-5 && i != 0)
        );
        // Only if at least one axis actually upsamples
        rule_if!(int_scales.iter().any(|&s| s != 1));

        let input_shape = &input_fact.shape;

        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, node.inputs[0])?;

        // Step 1: Reshape to interleave size-1 axes after each upsampled dim
        // e.g. (N, C, H, W) with scales (1,1,2,2) → (N, C, H, 1, W, 1)
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

        // Step 2: Tile the size-1 axes
        use tract_core::ops::array::Tile;
        wire = patch.wire_node(
            format!("{}.tile", node.name),
            Tile { multipliers: tile_multipliers },
            &[wire],
        )?[0];

        // Step 3: Reshape back to merge the tiled dims
        // e.g. (N, C, H, 2, W, 2) → (N, C, H*2, W*2)
        let tiled_shape: TVec<TDim> = to_dims
            .iter()
            .zip(int_scales.iter().flat_map(|&s| if s > 1 { vec![1usize, s] } else { vec![1] }))
            .map(|(d, s)| d.clone() * s)
            .collect();
        let mut final_dims: TVec<TDim> = tvec![];
        let mut idx = 0;
        for &scale in &int_scales {
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

    #[test]
    fn cubic_resize_1d_upsample() {
        let input = tract_ndarray::arr1(&[0.0f32, 1.0, 2.0, 3.0]);
        let input_tensor = input.into_tensor().into_tvalue();
        let scales = tract_ndarray::arr1(&[2.0f32]);
        let scales_tensor = scales.into_tensor().into_tvalue();
        let op = Resize {
            axes: None,
            coord_transformer: CoordTransformer::HalfPixel,
            interpolator: Interpolator::Cubic,
            nearest: Nearest::Floor,
            cubic_coeff_a_bits: (-0.75f32).to_bits(),
            exclude_outside: false,
            optional_roi_input: None,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        };
        let result = op.eval(tvec!(input_tensor, scales_tensor)).unwrap();
        let output = result[0].try_as_plain().unwrap().as_slice::<f32>().unwrap();
        assert_eq!(output.len(), 8);
        assert!((output[0] - (-0.10546875)).abs() < 1e-4, "got {}", output[0]);
    }

    #[test]
    fn cubic_resize_2d_upsample() {
        let input = tract_ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let input_tensor = input.into_tensor().into_tvalue();
        let scales = tract_ndarray::arr1(&[2.0f32, 2.0]);
        let scales_tensor = scales.into_tensor().into_tvalue();
        let op = Resize {
            axes: None,
            coord_transformer: CoordTransformer::HalfPixel,
            interpolator: Interpolator::Cubic,
            nearest: Nearest::Floor,
            cubic_coeff_a_bits: (-0.75f32).to_bits(),
            exclude_outside: false,
            optional_roi_input: None,
            optional_scales_input: Some(1),
            optional_sizes_input: None,
        };
        let result = op.eval(tvec!(input_tensor, scales_tensor)).unwrap();
        let shape = result[0].shape();
        assert_eq!(shape, &[4, 4]);
    }
}
