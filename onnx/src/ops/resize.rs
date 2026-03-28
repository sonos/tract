use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_nnef::tract_num_traits::Zero as _;
use tract_onnx_opl::resize::{CoordTransformer, Interpolator, Nearest, Resize};

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
    Ok((expand(ResizeInference(op)), vec![]))
}

fn resize_10(node: &NodeProto) -> TractResult<Resize> {
    Ok(Resize {
        axes: None,
        optional_roi_input: None,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
        coord_transformer: coord_transformer_from_node(node)?,
        interpolator: interpolator_from_node(node)?,
        nearest: nearest_from_node(node)?,
    })
}

fn resize_11(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(3);
    Ok(Resize {
        axes: None,
        optional_roi_input: Some(1),
        optional_scales_input: Some(2),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: coord_transformer_from_node(node)?,
        interpolator: interpolator_from_node(node)?,
        nearest: nearest_from_node(node)?,
    })
}

fn resize_13(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: None,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: coord_transformer_from_node(node)?,
        interpolator: interpolator_from_node(node)?,
        nearest: nearest_from_node(node)?,
    })
}

fn resize_18(node: &NodeProto) -> TractResult<Resize> {
    let mut options = crate::model::optional_inputs(node).skip(1);
    Ok(Resize {
        axes: node.get_attr_opt_vec("axes")?,
        optional_roi_input: options.next().unwrap(),
        optional_scales_input: options.next().unwrap(),
        optional_sizes_input: options.next().unwrap(),
        coord_transformer: coord_transformer_from_node(node)?,
        interpolator: interpolator_from_node(node)?,
        nearest: nearest_from_node(node)?,
    })
}

fn coord_transformer_from_node(node: &NodeProto) -> TractResult<CoordTransformer> {
    CoordTransformer::parse(
        node.get_attr_opt("coordinate_transformation_mode")?.unwrap_or("half_pixel"),
    )
}

fn interpolator_from_node(node: &NodeProto) -> TractResult<Interpolator> {
    Interpolator::parse(node.get_attr_opt("mode")?.unwrap_or("nearest"))
}

fn nearest_from_node(node: &NodeProto) -> TractResult<Nearest> {
    Nearest::parse(node.get_attr_opt("nearest_mode")?.unwrap_or("round_prefer_floor"))
}

#[derive(Clone, Debug)]
struct ResizeInference(Resize);

impl Expansion for ResizeInference {
    fn name(&self) -> StaticName {
        "Resize".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let op = &self.0;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if let Some(scales) = op.optional_scales_input {
            s.given(&inputs[scales].shape[0], move |s, len| {
                if len.is_zero() {
                    rules_with_sizes(op, s, inputs, outputs)
                } else {
                    rules_with_scales(op, s, inputs, outputs)
                }
            })
        } else if op.optional_sizes_input.is_some() {
            rules_with_sizes(op, s, inputs, outputs)
        } else {
            todo!()
        }
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        model.wire_node(name, self.0.clone(), inputs)
    }
}

fn rules_with_scales<'r, 'p: 'r, 's: 'r>(
    op: &'s Resize,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let scales_input = op.optional_scales_input.unwrap();
    let scales = &inputs[scales_input];
    s.equals(&scales.datum_type, f32::datum_type())?;
    s.equals(&scales.rank, 1)?;
    s.equals(&scales.shape[0], inputs[0].rank.bex().to_dim())?;
    s.given_2(&inputs[0].shape, &inputs[scales_input].value, move |s, input_shape, scales| {
        let output_size = op.compute_output_shape(&input_shape, Some(scales.as_ref()), None)?;
        let rank = input_shape.len();
        for i in 0..rank {
            s.equals(&outputs[0].shape[i], output_size[i].to_dim())?;
        }
        Ok(())
    })
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
