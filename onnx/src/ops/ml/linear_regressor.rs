use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::tract_core::ops::einsum::EinSum;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LinearRegressor", linear_regressor);
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

#[derive(Debug, Clone, Hash)]
pub struct LinearRegressor {
    pub coefficients: Arc<Tensor>,
    pub intercepts: Option<Arc<Tensor>>,
    pub post_transform: Option<PostTransform>,
    pub targets: usize,
}

fn linear_regressor(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let targets_i64: i64 = node.get_attr_opt("targets")?.unwrap_or(1);
    node.expect(targets_i64 > 0, "targets must be > 0")?;
    let targets: usize = usize::try_from(targets_i64)
        .map_err(|_| format_err!("targets out of range: {}", targets_i64))?;

    let raw_coeffs: Vec<f32> = node.get_attr_vec("coefficients")?;
    node.expect(!raw_coeffs.is_empty(), "coefficients not empty")?;

    node.expect(
        raw_coeffs.len() % targets == 0,
        "coefficients length must be a multiple of targets",
    )?;
    let c = raw_coeffs.len() / targets;

    let coeffs_tc = tensor1(&raw_coeffs).into_shape(&[targets, c])?;
    let coefficients = coeffs_tc.permute_axes(&[1, 0])?.into_arc_tensor();

    let intercepts: Option<Vec<f32>> = node.get_attr_opt_vec("intercepts")?;
    let intercepts = match intercepts {
        Some(v) => {
            node.expect(v.len() == targets, "intercepts length matches number of targets")?;
            Some(rctensor1(&v))
        }
        None => None,
    };

    let post_transform =
        node.get_attr_opt("post_transform")?.map(parse_post_transform).transpose()?.unwrap_or(None);

    Ok((expand(LinearRegressor { coefficients, intercepts, post_transform, targets }), vec![]))
}

impl Expansion for LinearRegressor {
    fn name(&self) -> StaticName {
        "LinearRegressor".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;

        s.equals(&outputs[0].datum_type, DatumType::F32)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[0].shape[1], self.targets.to_dim())?;
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
        let mut y = model.wire_node(
            format!("{prefix}.matmul"),
            EinSum::new(axes, model.outlet_fact(x)?.datum_type),
            [x, w].as_ref(),
        )?;

        if let Some(intercepts) = self.intercepts.as_deref() {
            let bias = intercepts.clone().broadcast_into_rank(2)?.into_arc_tensor();
            let bias = model.add_const(format!("{prefix}.intercepts"), bias)?;
            y = model.wire_node(
                format!("{prefix}.add_bias"),
                tract_core::ops::math::add(),
                &[y[0], bias],
            )?;
        }

        match self.post_transform {
            None => {}
            Some(PostTransform::Softmax) => {
                y = model.wire_node(
                    format!("{prefix}.softmax"),
                    tract_core::ops::nn::Softmax { axes: tvec!(1), ..Softmax::default() },
                    &y,
                )?;
            }
            Some(PostTransform::Logistic) => {
                y = model.wire_node(
                    format!("{prefix}.logistic"),
                    tract_core::ops::nn::sigmoid(),
                    &y,
                )?;
            }
        }

        Ok(tvec!(y[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }
}
