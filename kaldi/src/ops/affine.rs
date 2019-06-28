use tract_core::internal::*;
use tract_core::ndarray;

use crate::model::ParsingContext;

pub fn fixed_affine_component(ctx: &ParsingContext, name: &str) -> TractResult<Box<InferenceOp>> {
    let component = &ctx.proto_model.components[name];
    Ok(Box::new(Affine {
        linear_params: Arc::clone(component.attributes.get("LinearParams").ok_or("missing attribute LinearParams")?),
        bias_params: Arc::clone(component.attributes.get("BiasParams").ok_or("missing attribute ViasParams")?),
    }))
}

#[derive(Clone, Debug, new)]
struct Affine {
    linear_params: Arc<Tensor>,
    bias_params: Arc<Tensor>,
}

impl Affine {
    fn eval_t<T: Datum + num_traits::One + ndarray::LinalgScalar>(&self, input: &Tensor) -> TractResult<Tensor> {
        let array = input.to_array_view::<T>()?.into_dimensionality::<ndarray::Ix2>()?;
        let linear = self.linear_params.to_array_view::<T>()?.into_dimensionality::<ndarray::Ix2>()?;
        let bias = self.bias_params.to_array_view::<T>()?;
        let mut res = ndarray::Array2::from_shape_fn((array.shape()[0], bias.len()), |(_,x)| bias[x]);
        ndarray::linalg::general_mat_mul(T::one(), &linear, &array.t(), T::one(), &mut res.view_mut().reversed_axes());
        Ok(res.into_tensor())
    }
}

impl Op for Affine {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Affine".into()
    }
}

impl StatelessOp for Affine {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let output = dispatch_numbers!(Self::eval_t(input.datum_type())(self, &input))?;
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Affine {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, self.linear_params.datum_type())?;
        s.equals(&outputs[0].datum_type, self.linear_params.datum_type())?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[0].shape[1], &self.linear_params.shape()[0].to_dim())?;
        s.equals(&inputs[0].shape[1], &self.linear_params.shape()[1].to_dim())?;
        Ok(())
    }

    inference_op_as_op!();
}
