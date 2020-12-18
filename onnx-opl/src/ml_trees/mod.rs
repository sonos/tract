use tract_nnef::internal::*;

mod tree;

pub use self::tree::Cmp;
pub use self::tree::{Aggregate, PostTransform, TreeEnsemble, TreeEnsembleData};

pub fn parse_post_transform(s: &str) -> TractResult<Option<PostTransform>> {
    match s {
        "NONE" => Ok(None),
        "SOFTMAX" => Ok(Some(PostTransform::Softmax)),
        "LOGISTIC" => Ok(Some(PostTransform::Logistic)),
        "SOFTMAX_ZERO" => Ok(Some(PostTransform::SoftmaxZero)),
        "PROBIT" => bail!("PROBIT unsupported"),
        _ => bail!("Invalid post transform: {}", s),
    }
}

pub fn parse_aggregate(s: &str) -> TractResult<Aggregate> {
    match s {
        "SUM" => Ok(Aggregate::Sum),
        "AVERAGE" => Ok(Aggregate::Avg),
        "MAX" => Ok(Aggregate::Max),
        "MIN" => Ok(Aggregate::Min),
        _ => bail!("Invalid aggregate function: {}", s),
    }
}

#[derive(Debug, Clone, Hash)]
pub struct TreeEnsembleClassifier {
    pub ensemble: TreeEnsemble,
    pub class_labels: Tensor,
}

impl_dyn_hash!(TreeEnsembleClassifier);

impl Op for TreeEnsembleClassifier {
    fn name(&self) -> Cow<str> {
        "TreeEnsembleClassifier".into()
    }

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }

    op_as_typed_op!();
}

impl EvalOp for TreeEnsembleClassifier {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let input = input.to_array_view::<f32>()?;
        let scores = self.ensemble.eval(input)?;
        let tops: Vec<usize> = scores
            .view()
            .into_dimensionality::<tract_ndarray::Ix2>()?
            .outer_iter()
            .map(|scores| {
                scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap()
                    .0 as usize
            })
            .collect();
        let categ = tops
            .iter()
            .map(|&ix| self.class_labels.slice(0, ix, ix + 1))
            .collect::<TractResult<Vec<Tensor>>>()?;
        let categ = Tensor::stack_tensors(0, &categ)?;
        Ok(tvec!(categ.into_arc_tensor(), scores.into_arc_tensor()))
    }
}

impl TypedOp for TreeEnsembleClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        Ok(tvec!(
            TypedFact::dt_shape(self.class_labels.datum_type(), [n.clone()].as_ref()),
            TypedFact::dt_shape(
                f32::datum_type(),
                [n.clone(), self.class_labels.len().to_dim()].as_ref()
            )
        ))
    }

    as_op!();
}
