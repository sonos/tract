use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::tract_ndarray::s;
use tract_hir::tract_ndarray::ArrayView1;

pub fn non_max_suppression(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let center_point_box = match node.get_attr_opt("center_point_box")?.unwrap_or(0i64) {
        0 => BoxRepr::TwoPoints,
        1 => BoxRepr::CenterWidthHeight,
        other => bail!("unsupported center_point_box attribute value: {}", other),
    };

    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        Box::new(NonMaxSuppression {
            optional_max_output_boxes_per_class_input: options.next().unwrap(),
            optional_iou_threshold_input: options.next().unwrap(),
            optional_score_threshold_input: options.next().unwrap(),
            center_point_box,
            num_selected_indices_symbol: Symbol::new('n'),
        }),
        vec![],
    ))
}

#[derive(Clone, Debug, Hash)]
enum BoxRepr {
    // boxes data format [y1, x1, y2, x2]
    TwoPoints,
    // boxes data format [x_center, y_center, width, height]
    CenterWidthHeight,
}

fn get_min_max(lhs: f32, rhs: f32) -> (f32, f32) {
    if lhs >= rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

impl BoxRepr {
    // iou: intersection over union
    fn should_suppress_by_iou(
        &self,
        box1: ArrayView1<f32>,
        box2: ArrayView1<f32>,
        iou_threshold: f32,
    ) -> bool {
        let (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max) = match self {
            BoxRepr::TwoPoints => {
                let (x1_min, x1_max) = get_min_max(box1[[1]], box1[[3]]);
                let (x2_min, x2_max) = get_min_max(box2[[1]], box2[[3]]);

                let (y1_min, y1_max) = get_min_max(box1[[0]], box1[[2]]);
                let (y2_min, y2_max) = get_min_max(box2[[0]], box2[[2]]);

                (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max)
            }
            BoxRepr::CenterWidthHeight => {
                let (box1_width_half, box1_height_half) = (box1[[2]] / 2.0, box1[[3]] / 2.0);
                let (box2_width_half, box2_height_half) = (box2[[2]] / 2.0, box2[[3]] / 2.0);

                let (x1_min, x1_max) = (box1[[0]] - box1_width_half, box1[[0]] + box1_width_half);
                let (x2_min, x2_max) = (box2[[0]] - box2_width_half, box2[[0]] + box2_width_half);

                let (y1_min, y1_max) = (box1[[1]] - box1_height_half, box1[[1]] + box1_height_half);
                let (y2_min, y2_max) = (box2[[1]] - box2_height_half, box2[[1]] + box2_height_half);

                (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max)
            }
        };

        let intersection_y_min = f32::max(y1_min, y2_min);
        let intersection_y_max = f32::min(y1_max, y2_max);
        if intersection_y_max <= intersection_y_min {
            return false;
        }

        let intersection_x_min = f32::max(x1_min, x2_min);
        let intersection_x_max = f32::min(x1_max, x2_max);
        if intersection_x_max <= intersection_x_min {
            return false;
        }

        let intersection_area =
            (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min);

        if intersection_area <= 0.0 {
            return false;
        }

        let area1 = (x1_max - x1_min) * (y1_max - y1_min);
        let area2 = (x2_max - x2_min) * (y2_max - y2_min);

        let union_area = area1 + area2 - intersection_area;

        if area1 <= 0.0 || area2 <= 0.0 || union_area <= 0.0 {
            return false;
        }

        let intersection_over_union = intersection_area / union_area;

        intersection_over_union > iou_threshold
    }
}

#[derive(Clone, new, Debug, Hash)]
struct NonMaxSuppression {
    optional_max_output_boxes_per_class_input: Option<usize>,
    optional_iou_threshold_input: Option<usize>,
    optional_score_threshold_input: Option<usize>,
    center_point_box: BoxRepr,
    num_selected_indices_symbol: Symbol,
}

impl_dyn_hash!(NonMaxSuppression);

impl Op for NonMaxSuppression {
    fn name(&self) -> Cow<str> {
        "NonMaxSuppression".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EvalOp for NonMaxSuppression {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {

        let mut max_output_boxes_per_class = *self
            .optional_max_output_boxes_per_class_input
            .and_then(|ix| inputs.get(ix))
            .map_or(Ok(&0i64), |val| val.to_scalar::<i64>())?;
        let iou_threshold = *self
            .optional_iou_threshold_input
            .and_then(|ix| inputs.get(ix))
            .map_or(Ok(&0f32), |val| val.to_scalar::<f32>())?;
        let score_threshold = self
            .optional_score_threshold_input
            .and_then(|ix| inputs.get(ix))
            .map_or(Ok::<_, TractError>(None), |val| Ok(Some(*val.to_scalar::<f32>()?)))?;

        let mut inputs = inputs.drain(..);
        let boxes = inputs.next().ok_or(anyhow!("Expected at least 2 args"))?;
        let scores = inputs.next().ok_or(anyhow!("Expected at least 2 args"))?;

        if max_output_boxes_per_class == 0 {
            max_output_boxes_per_class = i64::MAX;
        }
        if !(0.0..=1.0).contains(&iou_threshold) {
            bail!("iou_threshold must be between 0 and 1");
        }

        let num_batches = scores.shape()[0];
        let num_classes = scores.shape()[1];
        let num_dim = scores.shape()[2];

        let boxes = boxes.to_array_view::<f32>()?;
        let scores = scores.to_array_view::<f32>()?;

        if scores.iter().any(|el| el.is_nan()) {
            bail!("scores must not be NaN");
        }

        // items: (batch, class, index)
        let mut selected_global: TVec<(usize, usize, usize)> = tvec![];

        for batch in 0..num_batches {
            for class in 0..num_classes {
                // items: (score, index)
                let mut candidates: TVec<(f32, usize)> =
                    if let Some(score_threshold) = score_threshold {
                        (0..num_dim)
                            .map(|i| (scores[[batch, class, i]], i))
                            .filter(|(score, _)| *score > score_threshold)
                            .collect()
                    } else {
                        (0..num_dim).map(|i| (scores[[batch, class, i]], i)).collect()
                    };

                // unwrap: cannot panic because of the NaN check before
                candidates.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

                // items: (score, index)
                let mut selected_in_class: TVec<(f32, usize)> = tvec![];

                for (score, index) in candidates {
                    if selected_in_class.len() as i64 >= max_output_boxes_per_class {
                        break;
                    }

                    let box1 = boxes.slice(s![batch, index, ..]);
                    let suppr = selected_in_class.iter().any(|(_, index)| {
                        let box2 = boxes.slice(s![batch, *index, ..]);
                        self.center_point_box.should_suppress_by_iou(box1, box2, iou_threshold)
                    });
                    if !suppr {
                        selected_in_class.push((score, index));
                        selected_global.push((batch, class, index));
                    }
                }
            }
        }

        // output shape is [num_selected_indices, 3]; format is [batch_index, class_index, box_index]
        let num_selected = selected_global.len();
        let v = selected_global
            .into_iter()
            .flat_map(|(batch, class, index)| [batch as i64, class as i64, index as i64])
            .collect();
        let res = tract_ndarray::ArrayD::from_shape_vec(&*tvec![num_selected, 3], v)?;

        Ok(tvec![res.into_arc_tensor()])
    }
}

impl InferenceRulesOp for NonMaxSuppression {
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

    as_op!();
    to_typed!();
}

impl TypedOp for NonMaxSuppression {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec![f32::fact([TDim::Sym(self.num_selected_indices_symbol), 3usize.to_dim()])])
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}
