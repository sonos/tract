use crate::tfpb::node_def::NodeDef;
use tract_core::internal::*;

use super::philox::Philox4x32x10;

pub fn random_uniform(node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let dtype = node.get_attr_datum_type("T")?;
    let seed: u64 = node.get_attr_int("seed")?;
    let seed2: u64 = node.get_attr_int("seed2")?;
    Ok(Box::new(RandomUniform::new(dtype, seed, seed2)))
}

#[derive(Debug, Clone, new)]
pub struct RandomUniform {
    t: DatumType,
    seed1: u64,
    seed2: u64,
}

impl RandomUniform {
    pub fn make<T>(&self, shape: &Arc<Tensor>) -> TractResult<Arc<Tensor>>
    where
        T: Datum + Copy,
    {
        let shape: TVec<usize> =
            shape.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x as usize).collect();
        let mut rng = Philox4x32x10::weird_tf_constructor(self.seed1, self.seed2).flat_map(|big| {
            tvec![big as u32, (big >> 32) as u32, (big >> 64) as u32, (big >> 96) as u32].into_iter()
        });
        unsafe {
            let mut tensor = Tensor::uninitialized::<f32>(&*shape)?;
            tensor.as_slice_mut::<f32>()?.iter_mut().for_each(|x| {
                let mantissa = rng.next().unwrap() & 0x7fffff;
                let exp = 127 as u32;
                let f = exp << 23 | mantissa;
                *x = f32::from_bits(f) - 1.0
            });
            Ok(tensor.into_arc_tensor())
        }
    }
}

impl Op for RandomUniform {
    fn name(&self) -> Cow<str> {
        "RandomUniform".into()
    }

    fn validation(&self) -> Validation {
        if self.seed1 == 0 && self.seed2 == 0 {
            Validation::Random
        } else {
            Validation::Accurate
        }
    }
}

impl StatelessOp for RandomUniform {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(dispatch_numbers!(Self::make(self.t)(self, &inputs[0]))?))
    }
}

impl InferenceRulesOp for RandomUniform {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.t)?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        s.given(&inputs[0].value, move |s, value| {
            let shape: TVec<TDim> =
                value.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x.to_dim()).collect();
            s.equals(&outputs[0].shape, shape.bex())
        })?;
        Ok(())
    }

    inference_op_as_op!();
}
