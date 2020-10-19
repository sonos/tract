use crate::internal::*;
use tract_ndarray::*;
use tract_num_traits::{AsPrimitive, One, Zero};

#[derive(Debug, Clone, new, Default, Educe)]
#[educe(Hash)]
pub struct ConstantLike {
    #[educe(Hash(method = "hash_f32"))]
    value: f32,
}

tract_data::impl_dyn_hash!(ConstantLike);

impl Op for ConstantLike {
    fn name(&self) -> Cow<str> {
        "ConstantLike".into()
    }

    op_hir!();
    op_as_typed_op!();
}

impl EvalOp for ConstantLike {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(tensor0(self.value).broadcast_scalar_to_shape(input.shape())?.into_arc_tensor()))
    }
}

impl InferenceRulesOp for ConstantLike {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.given_2(&inputs[0].shape, &inputs[0].datum_type, move |s, shape, dt| {
            if shape.iter().all(|d| d.to_usize().is_ok()) {
                let shape: Vec<usize> = shape.iter().map(|d| d.to_usize().unwrap()).collect();
                let value = tensor0(self.value)
                    .cast_to_dt(dt)?
                    .broadcast_scalar_to_shape(&*shape)?
                    .into_arc_tensor();
                s.equals(&outputs[0].value, value)?;
            }
            Ok(())
        })
    }

    as_op!();
    to_typed!();
}

impl TypedOp for ConstantLike {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct EyeLike {
    dt: Option<DatumType>,
    k: isize,
}

tract_data::impl_dyn_hash!(EyeLike);

impl EyeLike {
    pub fn make<T>(&self, (r, c): (usize, usize)) -> TractResult<Arc<Tensor>>
    where
        T: Copy + Datum + One + Zero,
        f32: AsPrimitive<T>,
    {
        let mut array = Array2::<T>::zeros((r, c));
        for y in 0..r {
            let x = y as isize + self.k;
            if x >= 0 && x < c as isize {
                array[(y, x as usize)] = T::one()
            }
        }
        Ok(array.into_dyn().into_arc_tensor())
    }
}

impl Op for EyeLike {
    fn name(&self) -> Cow<str> {
        "EyeLike".into()
    }

    op_hir!();
    op_as_typed_op!();
}

impl EvalOp for EyeLike {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dt = self.dt.unwrap_or(input.datum_type());
        Ok(tvec!(dispatch_numbers!(Self::make(dt)(self, (input.shape()[0], input.shape()[1])))?))
    }
}

impl InferenceRulesOp for EyeLike {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        if let Some(dt) = self.dt {
            s.equals(&outputs[0].datum_type, dt)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.given(&inputs[0].shape, move |s, shape| {
            if let (Ok(r), Ok(c)) = (shape[0].to_usize(), shape[1].to_usize()) {
                let shape = (r, c);
                if let Some(dt) = self.dt {
                    let value = dispatch_numbers!(Self::make(dt)(self, shape))?;
                    s.equals(&outputs[0].value, value)?;
                } else {
                    s.given(&inputs[0].datum_type, move |s, dt| {
                        let value = dispatch_numbers!(Self::make(dt)(self, shape))?;
                        s.equals(&outputs[0].value, value)
                    })?;
                }
            }
            Ok(())
        })
    }

    as_op!();
    to_typed!();
}

impl TypedOp for EyeLike {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            self.dt.unwrap_or(inputs[0].datum_type),
            inputs[0].shape.clone()
        )?))
    }
}
