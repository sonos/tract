use tract_hir::internal::*;
use tract_hir::ops::nn::DataFormat;
use tract_num_traits::AsPrimitive;

#[derive(Debug, Clone, new, Default)]
pub struct BatchNorm {
    data_format: DataFormat,
    epsilon: f32,
    #[allow(dead_code)]
    spatial: bool,
}

impl BatchNorm {
    fn to_slope_and_inter<T>(
        &self,
        c_dim: usize,
        scale: &Tensor,
        beta: &Tensor,
        mean: &Tensor,
        var: &Tensor,
    ) -> TractResult<(Tensor, Tensor)>
    where
        T: Datum + tract_num_traits::Float,
        f32: AsPrimitive<T>,
    {
        let scale = scale.to_array_view::<T>()?.into_shape_with_order((c_dim,))?;
        let beta = beta.to_array_view::<T>()?.into_shape_with_order((c_dim,))?;
        let mean = mean.to_array_view::<T>()?.into_shape_with_order((c_dim,))?;
        let var = var.to_array_view::<T>()?.into_shape_with_order((c_dim,))?;

        let denominator = var.mapv(|x| (x + self.epsilon.as_()).sqrt());

        let slope = &scale / &denominator;
        let intercept = beta.to_owned() - (&mean * &scale) / denominator;
        Ok((slope.into_tensor(), intercept.into_tensor()))
    }
}

impl Expansion for BatchNorm {
    fn name(&self) -> Cow<str> {
        "BatchNorm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 5)?;
        check_output_arity(outputs, 1)?;
        s.equals_all(wrap!(
            &outputs[0].datum_type,
            &inputs[0].datum_type,
            &inputs[1].datum_type,
            &inputs[2].datum_type,
            &inputs[3].datum_type,
            &inputs[4].datum_type
        ))?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals_all(wrap!(
            &inputs[1].shape,
            &inputs[2].shape,
            &inputs[3].shape,
            &inputs[4].shape
        ))?;
        s.given(&inputs[0].shape, move |s, shape| {
            let shape = self.data_format.shape(shape)?;
            s.equals(&inputs[1].shape[0], shape.c_dim())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let x = target.outlet_fact(inputs[0])?;
        let params = (1..5)
            .map(|i| Ok(target.outlet_fact(inputs[i])?.konst.clone()))
            .collect::<TractResult<TVec<Option<Arc<Tensor>>>>>()?;

        if let (Some(scale), Some(beta), Some(mean), Some(var)) =
            (params[0].as_ref(), params[1].as_ref(), params[2].as_ref(), params[3].as_ref())
        {
            let x_shape = x.shape.to_tvec();
            let c_axis = self.data_format.shape(&x_shape)?.c_axis();
            let c_dim = self.data_format.shape(&x_shape)?.c_dim().to_usize()?;

            let (mut slope, mut inter) =
                dispatch_floatlike!(Self::to_slope_and_inter(x.datum_type)(
                    self, c_dim, scale, beta, mean, var
                ))?;

            let mut const_shape = tvec!(1; x_shape.len());
            const_shape[c_axis] = c_dim;

            slope.set_shape(&const_shape)?;
            inter.set_shape(&const_shape)?;

            let slope = target.add_const(prefix.to_string() + ".slope", slope)?;
            let inter = target.add_const(prefix.to_string() + ".inter", inter)?;
            let wire = target.wire_node(
                format!("{prefix}.mul"),
                tract_hir::ops::math::mul(),
                &[slope, inputs[0]],
            )?;
            return target.wire_node(prefix, tract_hir::ops::math::add(), &[inter, wire[0]]);
        }
        bail!("Params are not const")
    }
}
