use crate::model::ParsingContext;
use crate::pb::*;
use std::hash::Hash;
use tract_hir::internal::*;

pub fn resize(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let input_2_is_scales = node.input[2] != "";
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
    Ok((Box::new(Resize { input_2_is_scales, coord_transformer, interpolator, nearest }), vec![]))
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
    input_2_is_scales: bool,
}

tract_linalg::impl_dyn_hash!(Resize);

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
        input_2: &Tensor,
    ) -> TractResult<TVec<usize>> {
        if self.input_2_is_scales {
            let scales = input_2.cast_to::<f32>()?;
            Ok(input_shape
                .iter()
                .zip(scales.as_slice::<f32>()?.iter())
                .map(|(input, scale)| ((*input as f32) * scale) as usize)
                .collect())
        } else {
            let size = input_2.cast_to::<i64>()?;
            Ok(size.as_slice::<i64>()?.iter().map(|i| *i as usize).collect())
        }
    }
}

impl StatelessOp for Resize {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, _rois, input_2) = args_3!(inputs);
        let mut data = data.into_tensor().into_array::<f32>()?;
        for axis in 0..data.ndim() {
            let output_shape = self.compute_output_shape(data.shape(), &input_2)?;
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
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        let have_roi = false;
        if have_roi {
            s.equals(&inputs[1].rank, 1)?;
            s.equals(2 * inputs[0].rank.bex().to_dim(), inputs[1].shape[0].bex())?;
        }
        s.equals(&inputs[2].rank, 1)?;
        s.equals(inputs[0].rank.bex().to_dim(), inputs[2].shape[0].bex())?;
        s.given(&inputs[0].rank, move |s, rank| {
            let rank = rank as usize;
            if self.input_2_is_scales {
                if have_roi {
                    s.given_3(
                        &inputs[0].shape,
                        &inputs[1].value,
                        &inputs[2].value,
                        move |s, input_shape, rois, scales| {
                            let rois = rois.cast_to::<f32>()?;
                            let rois = rois.as_slice::<f32>()?;
                            let scales = scales.cast_to::<f32>()?;
                            let scales = scales.as_slice::<f32>()?;
                            for i in 0..rank {
                                let cropped =
                                    if have_roi { rois[i + rank] - rois[i] } else { 1.0f32 };
                                if let Ok(len) = input_shape[i].to_integer() {
                                    let output_len =
                                        (len as f32 * cropped * scales[i]).round() as usize;
                                    s.equals(&outputs[0].shape[i], output_len.to_dim())?;
                                }
                            }
                            Ok(())
                        },
                    )?;
                } else {
                    s.given_2(
                        &inputs[0].shape,
                        &inputs[2].value,
                        move |s, input_shape, scales| {
                            let scales = scales.cast_to::<f32>()?;
                            let scales = scales.as_slice::<f32>()?;
                            for i in 0..rank {
                                if let Ok(len) = input_shape[i].to_integer() {
                                    let output_len = (len as f32 * scales[i]).round() as usize;
                                    s.equals(&outputs[0].shape[i], output_len.to_dim())?;
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
            } else {
                for i in 0..(rank as usize) {
                    s.equals(&outputs[0].shape[i], inputs[2].value[i].bex().to_dim())?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Resize {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = if let Some(s) = inputs[0].shape.as_finite() {
            s
        } else {
            bail!("Only constant input shape are supported in Resize")
        };
        let input_2 = if let Some(t) = &inputs[2].konst {
            t
        } else {
            bail!("Only constant scale (or output size) are supported in Resize")
        };
        let output_shape = self.compute_output_shape(input_shape, &input_2)?;
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
