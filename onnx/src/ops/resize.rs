use crate::model::ParsingContext;
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;

pub fn resize(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let coord_transformer =
        match node.get_attr_opt("coordinate_transformation_mode")?.unwrap_or("half_pixel") {
            "align_corners" => CoordTransformer::AlignCorners,
            "half_pixel" => CoordTransformer::HalfPixel,
            s => todo!("coordinate_transformation_mode: {}", s),
        };
    let interpolator = match node.get_attr("mode")? {
        "linear" => Interpolator::Linear,
        s => todo!("mode: {}", s),
    };
    let nearest = match node.get_attr_opt("nearest_mode")?.unwrap_or("round_prefer_floor") {
        "floor" => Nearest::Floor,
        "round_prefer_floor" => Nearest::RoundPreferFloor,
        s => todo!("nearest_mode: {}", s),
    };
    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        Box::new(Resize {
            optional_scales_input: options.next().unwrap(),
            optional_sizes_input: options.next().unwrap(),
            coord_transformer,
            interpolator,
            nearest,
        }),
        vec![],
    ))
}

#[derive(Clone, Debug, Hash)]
enum CoordTransformer {
    HalfPixel,
    AlignCorners,
}

impl CoordTransformer {
    fn transform(&self, x_out: usize, scale: f32, len_in: usize, len_out: usize) -> f32 {
        match self {
            CoordTransformer::HalfPixel => (x_out as f32 + 0.5) * scale - 0.5,
            CoordTransformer::AlignCorners => {
                (x_out as f32 * (len_in as f32 - 1.0)) / (len_out as f32 - 1.0)
            }
        }
    }
}

#[derive(Clone, Debug, Hash)]
enum Interpolator {
    Linear,
}

impl Interpolator {
    fn interpolate(&self, y_left: f32, y_right: f32, x_ratio: f32) -> f32 {
        match self {
            Interpolator::Linear => y_left * (1.0 - x_ratio) + y_right * x_ratio,
        }
    }
}

#[derive(Clone, Debug, Hash)]
enum Nearest {
    Floor,
    RoundPreferFloor,
}

#[derive(Clone, new, Debug, Hash)]
struct Resize {
    coord_transformer: CoordTransformer,
    interpolator: Interpolator,
    nearest: Nearest,
    optional_scales_input: Option<usize>,
    optional_sizes_input: Option<usize>,
}

tract_data::impl_dyn_hash!(Resize);

impl Op for Resize {
    fn name(&self) -> Cow<str> {
        "Resize".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl Resize {
    fn compute_output_shape(
        &self,
        input_shape: &[usize],
        input_scale: Option<&Tensor>,
        input_sizes: Option<&Tensor>,
    ) -> TractResult<TVec<usize>> {
        if let Some(scale) = input_scale {
            if scale.len() == input_shape.len() {
                let scales = scale.cast_to::<f32>()?;
                return Ok(input_shape
                    .iter()
                    .zip(scales.as_slice::<f32>()?.iter())
                    .map(|(input, scale)| ((*input as f32) * scale) as usize)
                    .collect());
            }
        }
        if let Some(sizes) = input_sizes {
            if sizes.len() == input_shape.len() {
                let size = sizes.cast_to::<i64>()?;
                return Ok(size.as_slice::<i64>()?.iter().map(|i| *i as usize).collect());
            }
        }
        bail!("Neither shape not scale makes sense: input_shape: {:?}, scale: {:?}, sizes: {:?}")
    }
}

impl EvalOp for Resize {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            inputs[0].shape(),
            scales.map(|t| &**t),
            sizes.map(|t| &**t),
        )?;
        let mut data = inputs.remove(0).into_tensor().into_array::<f32>()?;
        for axis in 0..data.ndim() {
            if output_shape[axis] == data.shape()[axis] {
                continue;
            } else if output_shape[axis] > data.shape()[axis] {
                let scale = output_shape[axis] as f32 / data.shape()[axis] as f32;
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
                    let mut co_i = co_o.clone();
                    let x_left = (x_in as usize).min(data.shape()[axis] - 1).max(0);
                    co_i[axis] = x_left;
                    let y_left = data[&co_i];
                    let x_right = (x_left + 1).min(data.shape()[axis] - 1);
                    co_i[axis] = x_right;
                    let y_right = data[&co_i];
                    let x_frac = x_in - x_left as f32;
                    self.interpolator.interpolate(y_left, y_right, x_frac)
                })
            }
        }
        Ok(tvec!(data.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Resize {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if inputs.len() == 3 && self.optional_scales_input == Some(2) {
            rules_with_scales(self, s, inputs, outputs)
        } else if inputs.len() == 3 && self.optional_sizes_input == Some(2) {
            rules_with_sizes(self, s, inputs, outputs)
        } else {
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
            let input_shape =
                input_shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<usize>>>()?;
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
        let input_shape = if let Some(s) = inputs[0].shape.as_finite() {
            s
        } else {
            bail!("Only constant input shape are supported in Resize")
        };
        let scales = self.optional_scales_input.and_then(|ix| inputs.get(ix));
        let sizes = self.optional_sizes_input.and_then(|ix| inputs.get(ix));
        let output_shape = self.compute_output_shape(
            &*input_shape,
            scales.and_then(|f| f.konst.as_deref()),
            sizes.and_then(|f| f.konst.as_deref()),
        )?;
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*output_shape)?))
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}
