use rand::distributions::uniform::SampleUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::num_traits::Float;
use rand_distr::StandardNormal;
use tract_nnef::internal::*;
use tract_nnef::tract_core::trivial_op_state_freeeze;

pub fn register(registry: &mut Registry) {
    /*
    registry.register_primitive(
    "tract_onnx_multinomial",
    &parameters(),
    &[("output", TypeName::Scalar.tensor())],
    load
    );
    registry.register_dumper(TypeId::of::<Multinomial>(), dump);
    */
}

#[derive(Debug, Clone, Hash)]
pub enum Dist {
    Uniform { low: Tensor, high: Tensor },
    Normal { mean: Tensor, dev: Tensor },
}

#[derive(Debug, Clone, Hash)]
pub struct Random {
    pub fact: TypedFact,
    pub dist: Dist,
    pub seed: Option<u64>,
}

impl_dyn_hash!(Random);

impl Op for Random {
    fn name(&self) -> Cow<str> {
        "Random".into()
    }

    op_as_typed_op!();
}

impl TypedOp for Random {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    as_op!();
}

impl EvalOp for Random {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        let rng = self
            .seed
            .map(|s| SmallRng::seed_from_u64(s))
            .unwrap_or_else(|| SmallRng::from_entropy());
        Ok(Some(Box::new(RandomState(rng))))
    }
}

#[derive(Clone, Debug)]
struct RandomState(SmallRng);

impl OpState for RandomState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<Random>().context("op and state mismatch")?;
        let mut tensor = unsafe {
            Tensor::uninitialized_dt(
                op.fact.datum_type,
                &op.fact.shape.eval_to_usize(&session.resolved_symbols)?,
            )?
        };
        match &op.dist {
            Dist::Uniform { low, high } => match op.fact.datum_type {
                DatumType::F32 => sample_uniform::<f32>(&mut tensor, &mut self.0, low, high)?,
                DatumType::F64 => sample_uniform::<f64>(&mut tensor, &mut self.0, low, high)?,
                DatumType::F16 => {
                    sample_uniform::<f32>(&mut tensor, &mut self.0, low, high)?;
                    tensor = tensor.cast_to::<f16>()?.into_owned();
                }
                _ => bail!("Random only support float types"),
            },
            Dist::Normal { mean, dev } => match op.fact.datum_type {
                DatumType::F32 => sample_normal::<f32>(&mut tensor, &mut self.0, mean, dev)?,
                DatumType::F64 => sample_normal::<f64>(&mut tensor, &mut self.0, mean, dev)?,
                DatumType::F16 => {
                    sample_uniform::<f32>(&mut tensor, &mut self.0, mean, dev)?;
                    tensor = tensor.cast_to::<f16>()?.into_owned();
                }
                _ => bail!("Random only support float types"),
            },
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

trivial_op_state_freeeze!(RandomState);

fn sample_uniform<T: Datum + SampleUniform + Copy>(
    t: &mut Tensor,
    r: &mut SmallRng,
    low: &Tensor,
    high: &Tensor,
) -> TractResult<()> {
    let dist =
        rand::distributions::Uniform::new(low.cast_to_scalar::<T>()?, high.cast_to_scalar::<T>()?);
    t.as_slice_mut::<T>()?.iter_mut().zip(dist.sample_iter(r)).for_each(|(v, r)| *v = r);
    Ok(())
}

fn sample_normal<T: Datum + Float + Copy>(
    t: &mut Tensor,
    r: &mut SmallRng,
    mean: &Tensor,
    dev: &Tensor,
) -> TractResult<()>
where
    StandardNormal: Distribution<T>,
{
    let dist =
        rand_distr::Normal::<T>::new(mean.cast_to_scalar::<T>()?, dev.cast_to_scalar::<T>()?)?;
    t.as_slice_mut::<T>()?.iter_mut().zip(dist.sample_iter(r)).for_each(|(v, r)| *v = r);
    Ok(())
}
