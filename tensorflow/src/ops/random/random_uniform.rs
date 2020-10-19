use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;

use super::philox::Philox4x32x10;

pub fn random_uniform(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let seed: u64 = node.get_attr_int("seed")?;
    let seed2: u64 = node.get_attr_int("seed2")?;
    Ok(Box::new(RandomUniform::new(dtype, seed, seed2)))
}

pub fn random_uniform_int(
    _ctx: &ParsingContext,
    node: &NodeDef,
) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = node.get_attr_datum_type("Tout")?;
    let seed: u64 = node.get_attr_int("seed")?;
    let seed2: u64 = node.get_attr_int("seed2")?;
    Ok(Box::new(RandomUniformInt::new(dtype, seed, seed2)))
}

#[derive(Debug, Clone, new, Hash)]
pub struct RandomUniform {
    t: DatumType,
    seed1: u64,
    seed2: u64,
}

tract_data::impl_dyn_hash!(RandomUniform);

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

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for RandomUniform {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape: TVec<usize> =
            inputs[0].cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x as usize).collect();
        match self.t {
            DatumType::F32 => Ok(tvec!(make_f32(&*shape, self.seed1, self.seed2)?)),
            dt => bail!("RandomUniform not implemented for {:?}", dt),
        }
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

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref shape) = target.outlet_fact(mapping[&node.inputs[0]])?.konst {
            let op = TypedRandomUniform::new(
                self.t,
                self.seed1,
                self.seed2,
                shape.cast_to::<TDim>()?.as_slice::<TDim>()?.into(),
            );
            target.wire_node(&*node.name, op, &[node.inputs[0]])
        } else {
            bail!("Dynamic shape")
        }
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct TypedRandomUniform {
    t: DatumType,
    seed1: u64,
    seed2: u64,
    shape: TVec<TDim>,
}

tract_data::impl_dyn_hash!(TypedRandomUniform);

impl Op for TypedRandomUniform {
    fn name(&self) -> Cow<str> {
        "TypedRandomUniform".into()
    }

    fn validation(&self) -> Validation {
        if self.seed1 == 0 && self.seed2 == 0 {
            Validation::Random
        } else {
            Validation::Accurate
        }
    }

    op_tf!();
    op_as_typed_op!();
}

impl EvalOp for TypedRandomUniform {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = self.shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>()?;
        match self.t {
            DatumType::F32 => Ok(tvec!(make_f32(&*shape, self.seed1, self.seed2)?)),
            dt => bail!("RandomUniform not implemented for {:?}", dt),
        }
    }
}

impl TypedOp for TypedRandomUniform {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(self.t, &*self.shape)?))
    }

    as_op!();
}

pub fn make_f32(shape: &[usize], seed1: u64, seed2: u64) -> TractResult<Arc<Tensor>> {
    let mut rng = Philox4x32x10::weird_tf_constructor(seed1, seed2).u32_iter();
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

#[derive(Debug, Clone, new, Hash)]
pub struct RandomUniformInt {
    t: DatumType,
    seed1: u64,
    seed2: u64,
}

tract_data::impl_dyn_hash!(RandomUniformInt);

impl RandomUniformInt {
    pub fn make_i32(&self, shape: &[usize], lo: i32, hi: i32) -> TractResult<Arc<Tensor>> {
        let mut rng = Philox4x32x10::weird_tf_constructor(self.seed1, self.seed2).u32_iter();
        unsafe {
            let mut tensor = Tensor::uninitialized::<i32>(&*shape)?;
            tensor.as_slice_mut::<i32>()?.iter_mut().for_each(|x| {
                // reproduce TF casts, with no conviction
                let lo = lo as u32;
                let hi = hi as u32;
                *x = (lo + rng.next().unwrap() % (hi - lo)) as i32;
            });
            Ok(tensor.into_arc_tensor())
        }
    }
}

impl Op for RandomUniformInt {
    fn name(&self) -> Cow<str> {
        "RandomUniformInt".into()
    }

    fn validation(&self) -> Validation {
        if self.seed1 == 0 && self.seed2 == 0 {
            Validation::Random
        } else {
            Validation::Accurate
        }
    }

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for RandomUniformInt {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape: TVec<usize> =
            inputs[0].cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x as usize).collect();
        match self.t {
            DatumType::I32 => Ok(tvec!(Self::make_i32(
                self,
                &*shape,
                *inputs[1].to_scalar::<i32>()?,
                *inputs[2].to_scalar::<i32>()?
            )?)),
            dt => bail!("RandomUniformInt not implemented for {:?}", dt),
        }
    }
}

impl InferenceRulesOp for RandomUniformInt {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.t)?;
        s.equals(&inputs[1].datum_type, self.t)?;
        s.equals(&inputs[2].datum_type, self.t)?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&inputs[0].shape[0], outputs[0].rank.bex().to_dim())?;
        s.given(&inputs[0].value, move |s, value| {
            let shape: TVec<TDim> =
                value.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|&x| x.to_dim()).collect();
            s.equals(&outputs[0].shape, shape.bex())
        })?;
        Ok(())
    }

    as_op!();
}
