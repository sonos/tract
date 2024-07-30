use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::non_max_suppression::BoxRepr;

pub fn non_max_suppression(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let center_point_box =
        BoxRepr::from_i64(node.get_attr_opt("center_point_box")?.unwrap_or(0i64))?;

    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        expand(NonMaxSuppression {
            optional_max_output_boxes_per_class_input: options.next().unwrap(),
            optional_iou_threshold_input: options.next().unwrap(),
            optional_score_threshold_input: options.next().unwrap(),
            center_point_box,
            num_selected_indices_symbol: ctx.template.symbols.new_with_prefix("x"),
        }),
        vec![],
    ))
}

#[derive(Clone, new, Debug, Hash)]
struct NonMaxSuppression {
    optional_max_output_boxes_per_class_input: Option<usize>,
    optional_iou_threshold_input: Option<usize>,
    optional_score_threshold_input: Option<usize>,
    center_point_box: BoxRepr,
    num_selected_indices_symbol: Symbol,
}

impl Expansion for NonMaxSuppression {
    fn name(&self) -> Cow<str> {
        "NonMaxSuppression".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let input_count = 2
            + self.optional_max_output_boxes_per_class_input.is_some() as usize
            + self.optional_iou_threshold_input.is_some() as usize
            + self.optional_score_threshold_input.is_some() as usize;
        check_input_arity(inputs, input_count)?;
        check_output_arity(outputs, 1)?;

        // [out] selected_indices: shape=[num_selected_indices, 3], type=int64
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], self.num_selected_indices_symbol.to_dim())?;
        s.equals(&outputs[0].shape[1], 3usize.to_dim())?;
        s.equals(&outputs[0].datum_type, i64::datum_type())?;

        // [in] boxes: shape=[num_batches, spatial_dimension, 4], type=float
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[0].shape[2], 4usize.to_dim())?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;

        // [in] scores: shape=[num_batches, num_classes, spatial_dimension], type=float
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;

        // num_batches in boxes, scores
        s.equals(&inputs[0].shape[0], &inputs[1].shape[0])?;
        // spatial_dimension in boxes, scores
        s.equals(&inputs[0].shape[1], &inputs[1].shape[2])?;

        // [in, optional] max_output_boxes_per_class: scalar, type=int64
        if let Some(index) = self.optional_max_output_boxes_per_class_input {
            s.equals(&inputs[index].rank, 1)?;
            s.equals(&inputs[index].shape[0], 1usize.to_dim())?;
            s.equals(&inputs[index].datum_type, i64::datum_type())?;
        }

        // [in, optional] iou_threshold: scalar, type=float
        if let Some(index) = self.optional_iou_threshold_input {
            s.equals(&inputs[index].rank, 1)?;
            s.equals(&inputs[index].shape[0], 1usize.to_dim())?;
            s.equals(&inputs[index].datum_type, f32::datum_type())?;
        }

        // [in, optional] score_threshold: scalar, type=float
        if let Some(index) = self.optional_score_threshold_input {
            s.equals(&inputs[index].rank, 1)?;
            s.equals(&inputs[index].shape[0], 1usize.to_dim())?;
            s.equals(&inputs[index].datum_type, f32::datum_type())?;
        }

        Ok(())
    }

    fn wire(
        &self,
        name: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let max_output_boxes_per_class = self
            .optional_max_output_boxes_per_class_input
            .map(|index| Ok(inputs[index]))
            .unwrap_or_else(|| {
                model.add_const(format!("{name}.max_output_boxes_per_class"), tensor0(0i64))
            })?;
        let iou_threshold = self
            .optional_iou_threshold_input
            .map(|index| Ok(inputs[index]))
            .unwrap_or_else(|| model.add_const(format!("{name}.iou_threshold"), tensor0(0.0f32)))?;
        // score_threshold is an optional input, but we cannot assing it a meaningful default value
        let score_threshold = self.optional_score_threshold_input.map(|index| inputs[index]);

        let op = tract_onnx_opl::non_max_suppression::NonMaxSuppression {
            center_point_box: self.center_point_box,
            num_selected_indices_symbol: self.num_selected_indices_symbol.clone(),
            has_score_threshold: score_threshold.is_some(),
        };

        if let Some(score_threshold) = score_threshold {
            model.wire_node(
                name,
                op,
                &[
                    inputs[0], // boxes
                    inputs[1], // scores
                    max_output_boxes_per_class,
                    iou_threshold,
                    score_threshold,
                ],
            )
        } else {
            model.wire_node(
                name,
                op,
                &[
                    inputs[0], // boxes
                    inputs[1], // scores
                    max_output_boxes_per_class,
                    iou_threshold,
                ],
            )
        }
    }
}
