use rand::distributions::uniform::SampleUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::num_traits::Float;
use rand_distr::StandardNormal;
use tract_nnef::internal::*;
use tract_nnef::ser::{array, tdims};
use tract_nnef::tract_core::trivial_op_state_freeeze;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_random",
        &[
            TypeName::String.named("datum_type"),
            TypeName::Integer.array().named("shape"),
            TypeName::String.named("dist"),
            TypeName::Scalar.array().named("parameters"),
            TypeName::Integer.named("seed"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let dt: DatumType = invocation.named_arg_as::<String>(builder, "datum_type")?.parse()?;
    let shape: TVec<TDim> = invocation.named_arg_as(builder, "shape")?;
    let fact = dt.fact(&shape);
    let dist: String = invocation.named_arg_as(builder, "dist")?;
    let parameters: TVec<Arc<Tensor>> = invocation.named_arg_as(builder, "parameters")?;
    let [p1, p2] = &*parameters else { bail!("Random expect two parameters") };
    let dist = match &*dist {
        "normal" => Dist::Normal { mean: p1.clone(), dev: p2.clone() },
        "uniform" => Dist::Uniform { low: p1.clone(), high: p2.clone() },
        _ => bail!("Unexpected distribution {}", dist),
    };
    let seed = invocation.get_named_arg_as(builder, "seed")?;
    let op = Random { fact, dist, seed };
    builder.wire(op, &[])
}

fn dump(_ast: &mut IntoAst, _node: &TypedNode, op: &Random) -> TractResult<Option<Arc<RValue>>> {
    let mut named = vec![
        ("datum_type", string(format!("{:?}", op.fact.datum_type))),
        ("shape", tdims(&op.fact.shape)),
    ];
    if let Some(seed) = op.seed {
        named.push(("seed", numeric(seed)));
    }
    match &op.dist {
        Dist::Uniform { low, high } => {
            named.push(("dist", string("uniform")));
            named.push((
                "parameters",
                array(&[
                    numeric(low.cast_to_scalar::<f32>()?),
                    numeric(high.cast_to_scalar::<f32>()?),
                ]),
            ));
        }
        Dist::Normal { mean, dev } => {
            named.push(("dist", string("normal")));
            named.push((
                "parameters",
                array(&[
                    numeric(mean.cast_to_scalar::<f32>()?),
                    numeric(dev.cast_to_scalar::<f32>()?),
                ]),
            ));
        }
    }
    Ok(Some(invocation("tract_onnx_random", &[], &named)))
}

#[derive(Debug, Clone, Hash)]
pub enum Dist {
    Uniform { low: Arc<Tensor>, high: Arc<Tensor> },
    Normal { mean: Arc<Tensor>, dev: Arc<Tensor> },
}

#[derive(Debug, Clone, Hash)]
pub struct Random {
    pub fact: TypedFact,
    pub dist: Dist,
    pub seed: Option<u64>,
}

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
        let rng = self.seed.map(SmallRng::seed_from_u64).unwrap_or_else(SmallRng::from_entropy);
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
