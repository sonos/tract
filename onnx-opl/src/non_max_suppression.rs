use std::cmp::Ordering;

use rustfft::num_traits::Float;
use tract_nnef::{
    internal::*,
    tract_ndarray::{s, ArrayView1},
};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_non_max_suppression",
        &parameters(),
        &[("output", TypeName::Integer.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

#[derive(Copy, Clone, Debug, Hash)]
pub enum BoxRepr {
    // boxes data format [y1, x1, y2, x2]
    TwoPoints,
    // boxes data format [x_center, y_center, width, height]
    CenterWidthHeight,
}

fn get_min_max<T: Float>(lhs: T, rhs: T) -> (T, T) {
    if lhs >= rhs {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

impl BoxRepr {
    pub fn from_i64(val: i64) -> TractResult<BoxRepr> {
        Ok(match val {
            0 => BoxRepr::TwoPoints,
            1 => BoxRepr::CenterWidthHeight,
            other => bail!("unsupported center_point_box argument value: {}", other),
        })
    }

    pub fn into_i64(self) -> i64 {
        match self {
            BoxRepr::TwoPoints => 0,
            BoxRepr::CenterWidthHeight => 1,
        }
    }

    // iou: intersection over union
    fn should_suppress_by_iou<T: Datum + Float>(
        &self,
        box1: ArrayView1<T>,
        box2: ArrayView1<T>,
        iou_threshold: T,
    ) -> bool {
        let two = T::one() + T::one();
        let (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max) = match self {
            BoxRepr::TwoPoints => {
                let (x1_min, x1_max) = get_min_max(box1[[1]], box1[[3]]);
                let (x2_min, x2_max) = get_min_max(box2[[1]], box2[[3]]);

                let (y1_min, y1_max) = get_min_max(box1[[0]], box1[[2]]);
                let (y2_min, y2_max) = get_min_max(box2[[0]], box2[[2]]);

                (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max)
            }
            BoxRepr::CenterWidthHeight => {
                let (box1_width_half, box1_height_half) = (box1[[2]] / two, box1[[3]] / two);
                let (box2_width_half, box2_height_half) = (box2[[2]] / two, box2[[3]] / two);

                let (x1_min, x1_max) = (box1[[0]] - box1_width_half, box1[[0]] + box1_width_half);
                let (x2_min, x2_max) = (box2[[0]] - box2_width_half, box2[[0]] + box2_width_half);

                let (y1_min, y1_max) = (box1[[1]] - box1_height_half, box1[[1]] + box1_height_half);
                let (y2_min, y2_max) = (box2[[1]] - box2_height_half, box2[[1]] + box2_height_half);

                (x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max)
            }
        };

        let intersection_y_min = T::max(y1_min, y2_min);
        let intersection_y_max = T::min(y1_max, y2_max);
        if intersection_y_max <= intersection_y_min {
            return false;
        }

        let intersection_x_min = T::max(x1_min, x2_min);
        let intersection_x_max = T::min(x1_max, x2_max);
        if intersection_x_max <= intersection_x_min {
            return false;
        }

        let intersection_area =
            (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min);

        if intersection_area.is_sign_negative() {
            return false;
        }

        let area1 = (x1_max - x1_min) * (y1_max - y1_min);
        let area2 = (x2_max - x2_min) * (y2_max - y2_min);

        let union_area = area1 + area2 - intersection_area;

        if area1.is_sign_negative() || area2.is_sign_negative() || union_area.is_sign_negative() {
            return false;
        }

        let intersection_over_union = intersection_area / union_area;

        intersection_over_union > iou_threshold
    }
}

#[derive(Debug, Clone, Hash)]
pub struct NonMaxSuppression {
    pub center_point_box: BoxRepr,
    pub num_selected_indices_symbol: Symbol,
    pub has_score_threshold: bool,
}

impl NonMaxSuppression {
    fn eval_t<T: Datum + Float>(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold) =
            if self.has_score_threshold {
                let (t1, t2, t3, t4, t5) = args_5!(inputs);
                (t1, t2, t3, t4, Some(t5))
            } else {
                let (t1, t2, t3, t4) = args_4!(inputs);
                (t1, t2, t3, t4, None)
            };

        let mut max_output_boxes_per_class = *max_output_boxes_per_class.to_scalar::<i64>()?;
        let iou_threshold = *iou_threshold.to_scalar::<T>()?;
        let score_threshold = score_threshold
            .map_or(Ok::<_, TractError>(None), |val| Ok(Some(*val.to_scalar::<T>()?)))?;

        if max_output_boxes_per_class == 0 {
            max_output_boxes_per_class = i64::MAX;
        }
        //        ensure!((0.0..=1.0).contains(&iou_threshold), "iou_threshold must be between 0 and 1");

        let num_batches = scores.shape()[0];
        let num_classes = scores.shape()[1];
        let num_dim = scores.shape()[2];

        let boxes = boxes.to_array_view::<T>()?;
        let scores = scores.to_array_view::<T>()?;

        // items: (batch, class, index)
        let mut selected_global: TVec<(usize, usize, usize)> = tvec![];

        for batch in 0..num_batches {
            for class in 0..num_classes {
                // items: (score, index)
                let mut candidates: TVec<(T, usize)> =
                    if let Some(score_threshold) = score_threshold {
                        (0..num_dim)
                            .map(|i| (scores[[batch, class, i]], i))
                            .filter(|(score, _)| *score > score_threshold)
                            .collect()
                    } else {
                        (0..num_dim).map(|i| (scores[[batch, class, i]], i)).collect()
                    };

                candidates.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

                // items: (score, index)
                let mut selected_in_class: TVec<(T, usize)> = tvec![];

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

        Ok(tvec![res.into_tvalue()])
    }
}

impl Op for NonMaxSuppression {
    fn name(&self) -> Cow<str> {
        "NonMaxSuppression".into()
    }

    op_as_typed_op!();
}

impl EvalOp for NonMaxSuppression {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let dt = inputs[0].datum_type();
        dispatch_floatlike!(Self::eval_t(dt)(self, inputs))
    }
}

impl TypedOp for NonMaxSuppression {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec![i64::fact([self.num_selected_indices_symbol.to_dim(), 3usize.to_dim()])])
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Integer.tensor().named("boxes"),
        TypeName::Scalar.tensor().named("scores"),
        TypeName::Integer.named("max_output_boxes_per_class").default(0),
        TypeName::Scalar.named("iou_threshold").default(0.0),
        TypeName::Scalar.named("score_threshold"),
        TypeName::Integer.named("center_point_box").default(0),
    ]
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &NonMaxSuppression,
) -> TractResult<Option<Arc<RValue>>> {
    let boxes = ast.mapping[&node.inputs[0]].clone();
    let scores = ast.mapping[&node.inputs[1]].clone();
    let max_output_boxes_per_class = ast.mapping[&node.inputs[2]].clone();
    let iou_threshold = ast.mapping[&node.inputs[3]].clone();
    let score_threshold = node.inputs.get(4).map(|v| ast.mapping[v].clone());

    let inv = if let Some(score_threshold) = score_threshold {
        invocation(
            "tract_onnx_non_max_suppression",
            &[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
            &[("center_point_box", numeric(op.center_point_box.into_i64()))],
        )
    } else {
        invocation(
            "tract_onnx_non_max_suppression",
            &[boxes, scores, max_output_boxes_per_class, iou_threshold],
            &[("center_point_box", numeric(op.center_point_box.into_i64()))],
        )
    };

    Ok(Some(inv))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let boxes = invocation.named_arg_as(builder, "boxes")?;
    let scores = invocation.named_arg_as(builder, "scores")?;
    let max_output_boxes_per_class =
        invocation.named_arg_as(builder, "max_output_boxes_per_class")?;
    let iou_threshold = invocation.named_arg_as(builder, "iou_threshold")?;
    let score_threshold = invocation.named_arg_as(builder, "score_threshold").ok();

    let center_point_box =
        BoxRepr::from_i64(invocation.named_arg_as(builder, "center_point_box")?)?;

    let n = builder.model.symbols.sym("n");
    let op = NonMaxSuppression {
        center_point_box,
        num_selected_indices_symbol: n,
        has_score_threshold: score_threshold.is_some(),
    };
    if let Some(score_threshold) = score_threshold {
        builder
            .wire(op, &[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold])
    } else {
        builder.wire(op, &[boxes, scores, max_output_boxes_per_class, iou_threshold])
    }
}
