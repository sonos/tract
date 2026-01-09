use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::array::TypedConcat;
use tract_hir::tract_core::ops::einsum::EinSum;
use tract_onnx_opl::ml::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LinearClassifier", linear_classifier);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransform {
    Softmax,
    Logistic,
}

pub fn parse_post_transform(s: &str) -> TractResult<Option<PostTransform>> {
    match s {
        "NONE" => Ok(None),
        "SOFTMAX" => Ok(Some(PostTransform::Softmax)),
        "LOGISTIC" => Ok(Some(PostTransform::Logistic)),
        "PROBIT" | "SOFTMAX_ZERO" => bail!("PROBIT and SOFTMAX_ZERO unsupported"),
        _ => bail!("Invalid post transform: {}", s),
    }
}

fn parse_class_data(node: &NodeProto) -> TractResult<Arc<Tensor>> {
    let ints = node.get_attr_opt_slice::<i64>("classlabels_ints")?;
    let strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?;
    match (ints, strs) {
        (Some(n), None) => Ok(rctensor1(n)),
        (None, Some(n)) => Ok(rctensor1(&n.iter().map(|d| d.to_string()).collect::<Vec<_>>())),
        (None, None) => bail!("cannot find neither 'classlabels_ints' not 'classlabels_strings'"),
        (Some(_), Some(_)) => {
            bail!("only one of 'classlabels_ints' and 'classlabels_strings' can be set")
        }
    }
}

#[derive(Debug, Clone, Hash)]
pub struct LinearClassifier {
    pub class_labels: Arc<Tensor>,
    pub coefficients: Arc<Tensor>,
    pub intercepts: Option<Arc<Tensor>>,
    pub post_transform: Option<PostTransform>,
    pub binary_result_layout: bool,
    pub num_models: usize,
}

fn linear_classifier(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let class_labels = parse_class_data(node)?;
    let n_classes = class_labels.len();
    let multi_class: i64 = node.get_attr_opt("multi_class")?.unwrap_or(0);
    let raw_coeffs: Vec<f32> = node.get_attr_vec("coefficients")?;
    node.expect(!raw_coeffs.is_empty(), "coefficients not empty")?;

    let intercepts_raw: Option<Vec<f32>> = node.get_attr_opt_vec("intercepts")?;

    let (e_prime, binary_result_layout) = match intercepts_raw.as_ref() {
        Some(v) => {
            node.expect(
                raw_coeffs.len() % v.len() == 0,
                "coefficients length must be a multiple of intercepts length",
            )?;
            let e_prime = v.len();
            let binary = n_classes == 2 && e_prime == 1 && multi_class == 0;
            (e_prime, binary)
        }
        None if n_classes == 2 && multi_class == 0 => (1, true),
        None if raw_coeffs.len() % n_classes == 0 => (n_classes, false),
        None => bail!(
            "coefficients length {} not compatible with number of classes {}",
            raw_coeffs.len(),
            n_classes
        ),
    };

    node.expect(
        raw_coeffs.len() % e_prime == 0,
        "coefficients length must be a multiple of number of models",
    )?;

    if binary_result_layout {
        node.expect(n_classes == 2, "binary result layout requires exactly 2 class labels")?;
    } else {
        node.expect(
            n_classes == e_prime,
            "class labels length must match number of models when not using binary single-model layout",
        )?;
    }

    let c = raw_coeffs.len() / e_prime;
    let coeffs_ec = tensor1(&raw_coeffs).into_shape(&[e_prime, c])?;
    let coefficients = coeffs_ec.permute_axes(&[1, 0])?.into_arc_tensor();

    let intercepts = match intercepts_raw {
        Some(v) => {
            node.expect(v.len() == e_prime, "intercepts length should match number of models")?;
            Some(rctensor1(&v))
        }
        None => None,
    };

    let post_transform =
        node.get_attr_opt("post_transform")?.map(parse_post_transform).transpose()?.unwrap_or(None);

    Ok((
        expand(LinearClassifier {
            class_labels,
            coefficients,
            intercepts,
            post_transform,
            binary_result_layout,
            num_models: e_prime,
        }),
        vec![],
    ))
}

impl Expansion for LinearClassifier {
    fn name(&self) -> StaticName {
        "LinearClassifier".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 2)?;

        s.equals(&outputs[0].datum_type, self.class_labels.datum_type())?;
        s.equals(&outputs[1].datum_type, DatumType::F32)?;

        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[1].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].shape[0], &inputs[0].shape[0])?;
        // Scores second dim depends on layout
        if self.binary_result_layout {
            s.equals(&outputs[1].shape[1], 2.to_dim())?;
        } else {
            s.equals(&outputs[1].shape[1], (self.num_models as i64).to_dim())?;
        }

        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::nn::*;

        let mut x = inputs[0];
        if model.outlet_fact(x)?.rank() == 1 {
            x = model.wire_node(format!("{prefix}.add_batch_axis"), AxisOp::Add(0), &[x])?[0];
        }

        if model.outlet_fact(x)?.datum_type != f32::datum_type() {
            x = model.wire_node(
                format!("{prefix}.to_f32"),
                tract_core::ops::cast::cast(f32::datum_type()),
                &[x],
            )?[0];
        }

        let w = model.add_const(format!("{prefix}.coefficients"), self.coefficients.clone())?;
        let axes = AxesMapping::for_numpy_matmul(2, false, false, false)?;
        let mut scores = model.wire_node(
            format!("{prefix}.matmul"),
            EinSum::new(axes, model.outlet_fact(x)?.datum_type),
            [x, w].as_ref(),
        )?;

        if let Some(intercepts) = self.intercepts.as_deref() {
            let bias = intercepts.clone().broadcast_into_rank(2)?.into_arc_tensor();
            let bias = model.add_const(format!("{prefix}.intercepts"), bias)?;
            scores = model.wire_node(
                format!("{prefix}.add_bias"),
                tract_core::ops::math::add(),
                &[scores[0], bias],
            )?;
        }

        let final_scores = if self.binary_result_layout {
            match self.post_transform {
                None => {
                    // logits [-s, s]
                    let m1 = model.add_const(format!("{prefix}.m1"), rctensor2(&[[-1f32]]))?;
                    let neg = model.wire_node(
                        format!("{prefix}.binary.neg"),
                        tract_core::ops::math::mul(),
                        &[scores[0], m1],
                    )?;
                    model.wire_node(
                        format!("{prefix}.binary.concat"),
                        TypedConcat::new(1),
                        &[neg[0], scores[0]],
                    )?
                }
                Some(PostTransform::Logistic) => {
                    // probabilities [1 - sigmoid(s), sigmoid(s)]
                    let p = model.wire_node(
                        format!("{prefix}.logistic"),
                        tract_core::ops::nn::sigmoid(),
                        &scores,
                    )?;
                    let one = model.add_const(prefix.to_string() + ".one", rctensor2(&[[1f32]]))?;
                    let complement = model.wire_node(
                        format!("{prefix}.binary.complement"),
                        tract_core::ops::math::sub(),
                        &[one, p[0]],
                    )?;
                    model.wire_node(
                        format!("{prefix}.binary.concat"),
                        TypedConcat::new(1),
                        &[complement[0], p[0]],
                    )?
                }
                Some(PostTransform::Softmax) => {
                    let m1 = model.add_const(format!("{prefix}.m1"), rctensor2(&[[-1f32]]))?;
                    let neg = model.wire_node(
                        format!("{prefix}.binary.neg"),
                        tract_core::ops::math::mul(),
                        &[scores[0], m1],
                    )?;
                    let logits2 = model.wire_node(
                        format!("{prefix}.binary.logits2"),
                        TypedConcat::new(1),
                        &[neg[0], scores[0]],
                    )?;
                    model.wire_node(
                        format!("{prefix}.softmax"),
                        tract_core::ops::nn::Softmax { axes: tvec!(1), ..Softmax::default() },
                        &logits2,
                    )?
                }
            }
        } else {
            let mut tmp = scores.clone();
            match self.post_transform {
                None => {}
                Some(PostTransform::Softmax) => {
                    tmp = model.wire_node(
                        format!("{prefix}.softmax"),
                        tract_core::ops::nn::Softmax { axes: tvec!(1), ..Softmax::default() },
                        &tmp,
                    )?;
                }
                Some(PostTransform::Logistic) => {
                    tmp = model.wire_node(
                        format!("{prefix}.logistic"),
                        tract_core::ops::nn::sigmoid(),
                        &tmp,
                    )?;
                }
            }
            tmp
        };

        let winners = model.wire_node(
            format!("{prefix}.argmax"),
            tract_core::ops::nn::Reduce::new(tvec!(1), tract_core::ops::nn::Reducer::ArgMax(false)),
            &final_scores,
        )?;
        let reduced = model.wire_node(
            format!("{prefix}.rm_axis"),
            tract_core::ops::change_axes::AxisOp::Rm(1),
            &winners,
        )?;
        let casted = model.wire_node(
            format!("{prefix}.casted"),
            tract_core::ops::cast::cast(i32::datum_type()),
            &reduced,
        )?;
        let labels = model.wire_node(
            format!("{prefix}.labels"),
            DirectLookup::new(
                self.class_labels.clone(),
                Tensor::zero_dt(self.class_labels.datum_type(), &[])?.into_arc_tensor(),
            )?,
            &casted,
        )?[0];

        Ok(tvec!(labels, final_scores[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }
}
