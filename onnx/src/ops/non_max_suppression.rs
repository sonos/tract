use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::non_max_suppression::BoxRepr;

pub fn non_max_suppression(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let center_point_box = BoxRepr::from_i64( node.get_attr_opt("center_point_box")?.unwrap_or(0i64))?;

    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        expand(NonMaxSuppression {
            optional_max_output_boxes_per_class_input: options.next().unwrap(),
            optional_iou_threshold_input: options.next().unwrap(),
            optional_score_threshold_input: options.next().unwrap(),
            center_point_box,
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
}

impl_dyn_hash!(NonMaxSuppression);

impl Expansion for NonMaxSuppression {
    fn name(&self) -> Cow<str> {
        "NonMaxSuppression".into()
    }

    op_onnx!();

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
        check_input_arity(&inputs, input_count)?;
        check_output_arity(&outputs, 1)?;

        // [out] selected_indices: shape=[num_selected_indices, 3], type=int64
        s.equals(&outputs[0].rank, 2)?;
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
        model.wire_node(
            name,
            tract_onnx_opl::non_max_suppression::NonMaxSuppression::new(self.center_point_box),
            inputs,
        )
    }
}
