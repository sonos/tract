use crate::model::ParsingContext;
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;
use tract_nnef::tract_num_traits::Zero;

pub fn resize(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = match ctx.onnx_operator_set_version {
        10 => resize_10(node)?,
        11..=12 => resize_11(node)?,
        13..=17 => resize_13(node)?,
        18.. => resize_18(node)?,
        v => bail!("Unsupported operator set for Resize operator ({v})"),
    };
    Ok((Box::new(op), vec![]))
}

fn resize_10(node: &NodeProto) -> TractResult<Resize> {
    Ok(Resize {
        axes: None,
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_11(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(3);
    Ok(Resize {
        axes: None,
        optional_roi_input: Some(1),
        optional_scales_input: Some(2),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_13(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: None,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

fn resize_18(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: node.get_attr_opt_vec("axes")?,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: CoordTransformer::from_node(node)?,
        interpolator: Interpolator::from_node(node)?,
        nearest: Nearest::from_node(node)?,
    })
}

#[derive(Clone, Debug, Hash)]
enum CoordTransformer {
    HalfPixel,
    AlignCorners,
    Asymmetric,
}

impl CoordTransformer {
    fn transform(&self, x_out: usize, scale: f32, len_in: usize, len_out: usize) -> f32 {
        match self {
            CoordTransformer::HalfPixel => (x_out as f32 + 0.5) / scale - 0.5,
            CoordTransformer::AlignCorners => {
                (x_out as f32 * (len_in as f32 - 1.0)) / (len_out as f32 - 1.0)
            }
            CoordTransformer::Asymmetric => (x_out as f32) / scale,
        }
    }

    fn from_node(node: &NodeProto) -> TractResult<CoordTransformer> {
        Ok(match node.get_attr_opt("coordinate_transformation_mode")?.unwrap_or("half_pixel") {
            "align_corners" => CoordTransformer::AlignCorners,
            "half_pixel" => CoordTransformer::HalfPixel,
            "asymmetric" => CoordTransformer::Asymmetric,
            s => bail!("coordinate_transformation_mode: {}", s),
        })
    }
}

#[derive(Clone, Debug, Hash)]
enum Interpolator {
    Linear,
    Nearest,
}

impl Interpolator {
    fn interpolate(&self, y_left: f32, y_right: f32, x_ratio: f32, nearest_mode: Nearest) -> f32 {
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

    fn from_node(node: &NodeProto) -> TractResult<Interpolator> {
        Ok(match node.get_attr_opt("mode")?.unwrap_or("nearest") {
            "nearest" => Interpolator::Nearest,
            "linear" => Interpolator::Linear,
            s => bail!("mode: {}", s),
        })
    }
}

#[derive(Clone, Copy, Debug, Hash)]
enum Nearest {
    Floor,
    Ceil,
    RoundPreferFloor,
    RoundPreferCeil,
}

impl Nearest {
    fn from_node(node: &NodeProto) -> TractResult<Nearest> {
        Ok(match node.get_attr_opt("nearest_mode")?.unwrap_or("round_prefer_floor") {
            "floor" => Nearest::Floor,
            "ceil" => Nearest::Ceil,
            "round_prefer_floor" => Nearest::RoundPreferFloor,
            "round_prefer_ceil" => Nearest::RoundPreferCeil,
            s => bail!("nearest_mode: {}", s),
        })
    }
}

#[derive(Clone, new, Debug, Hash)]
struct Resize {
    axes: Option<Vec<i64>>,
    coord_transformer: CoordTransformer,
    interpolator: Interpolator,
    nearest: Nearest,
    optional_roi_input: Option<usize>,
    optional_scales_input: Option<usize>,
    optional_sizes_input: Option<usize>,
}

impl Op for Resize {
    fn name(&self) -> Cow<str> {
        "Resize".into()
    }

    op_as_typed_op!();
}

impl Resize {
    fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        input_scale: Option<&Tensor>,
        input_sizes: Option<&Tensor>,
    ) -> TractResult<TVec<D>> {
        if let Some(scale) = input_scale {
            if scale.len() == input_shape.len() {
                let mut shape = tvec!();
                for (i, s) in
                    input_shape.iter().zip(scale.cast_to::<f32>()?.as_slice::<f32>()?.iter())
                {
                    if s.round() == *s {
                        shape.push(i.clone() * (*s as usize));
                    } else if let Ok(i) = i.to_usize() {
                        shape.push(((i as f32 * s) as usize).into());
                    } else {
                        bail!("Can not compute output shape. inputs are {input_shape:?} and scale {scale:?}")
                    }
                }
                return Ok(shape);
            }
        }
        if let Some(sizes) = input_sizes {
            if sizes.len() == input_shape.len() {
                return sizes
                    .cast_to::<TDim>()?
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
            scales.as_slice::<f32>()?.into()
        } else {
            output_shape.iter().zip(inputs[0].shape()).map(|(o, i)| *o as f32 / *i as f32).collect()
        };
        let mut data = inputs.remove(0).into_tensor().into_array::<f32>()?;
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

impl InferenceRulesOp for Resize {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if let Some(scales) = self.optional_scales_input {
            s.given(&inputs[scales].shape[0], move |s, len| {
                if len.is_zero() {
                    rules_with_sizes(self, s, inputs, outputs)
                } else {
                    rules_with_scales(self, s, inputs, outputs)
                }
            })
        } else if self.optional_sizes_input.is_some() {
            rules_with_sizes(self, s, inputs, outputs)
        } else {
            /*
            // bogus 4 inputs case
            s.given_2(
            &inputs[0].rank,
            &inputs[self.optional_scales_input.unwrap()].shape,
            move |s, input_rank, scale_shape| {
            if scale_shape.len() == 0 || scale_shape[0] != input_rank.to_dim() {
            rules_with_sizes(self, s, inputs, outputs)
            } else {
            rules_with_scales(self, s, inputs, outputs)
            }
            },
            )
            */
            todo!()
        }
    }

    as_op!();
    to_typed!();
}

fn rules_with_scales<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let scales = &inputs[op.optional_scales_input.unwrap()];
    s.equals(&scales.datum_type, f32::datum_type())?;
    s.equals(&scales.rank, 1)?;
    s.equals(&scales.shape[0], inputs[0].rank.bex().to_dim())?;
    s.given_2(
        &inputs[0].shape,
        &inputs[op.optional_scales_input.unwrap()].value,
        move |s, input_shape, scales| {
            let output_size = op.compute_output_shape(&input_shape, Some(scales.as_ref()), None)?;
            let rank = input_shape.len();
            for i in 0..rank {
                s.equals(&outputs[0].shape[i], output_size[i].to_dim())?;
            }
            Ok(())
        },
    )
}

fn rules_with_sizes<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let sizes = &inputs[op.optional_sizes_input.unwrap()];
    s.equals(&sizes.rank, 1)?;
    s.equals(&sizes.shape[0], inputs[0].rank.bex().to_dim())?;
    s.given(&inputs[0].rank, move |s, rank| {
        for i in 0..(rank as usize) {
            s.equals(&outputs[0].shape[i], sizes.value[i].bex().to_dim())?;
        }
        Ok(())
    })
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
