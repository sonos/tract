use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tract_nnef::internal::*;
use tract_nnef::tract_ndarray::s;
use tract_nnef::tract_num_traits::{AsPrimitive, Float, Zero};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_multinomial", 
        &parameters(),
        &[("output", TypeName::Scalar.tensor())], 
        load
    );
    registry.register_dumper(dump);
}

/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
#[derive(Clone, Debug)]
pub struct Multinomial {
    pub dtype: DatumType,
    pub sample_size: i32,
    pub seed: Option<f32>,
}

impl Multinomial {
    fn eval_t0<T1>(&self, input: TValue) -> TractResult<TValue>
    where
        T1: Datum + std::ops::SubAssign + Float + std::iter::Sum,
        Standard: Distribution<T1>,
    {
        match self.dtype {
            DatumType::I32 => self.eval_t::<T1, i32>(input),
            DatumType::I64 => self.eval_t::<T1, i64>(input),
            dt => bail!("Unsupported output datum type for Multinomial: {:?}", dt),
        }
    }
    fn eval_t<T1, T2>(&self, input: TValue) -> TractResult<TValue>
    where
        T1: Datum + std::ops::SubAssign + Float + std::iter::Sum,
        Standard: Distribution<T1>,
        T2: Datum + Zero + Copy,
        usize: AsPrimitive<T2>,
    {
        let batch_size = input.shape()[0];
        let class_size = input.shape()[1];

        let mut rng = self.seed.map_or_else(SmallRng::from_entropy, |seed| {
            SmallRng::seed_from_u64(seed.to_bits() as _)
        });

        // shape: [batch_size, class_size]
        let input = input.to_array_view::<T1>()?;

        // ONNX Multinomial inputs are "unnormalized log probabilities".
        // This means that we need to compute the maximum for each batch beforehand,
        //  and we also need to exp every value.

        let maximums: TVec<_> =
            input.rows().into_iter().map(|r| r.iter().map(|e| e.exp()).sum::<T1>()).collect();

        // shape: [batch_size, sample_size]
        let out_shape: &[_] = &[batch_size, self.sample_size as usize];
        let output = tract_ndarray::ArrayD::from_shape_fn(out_shape, |co_o| -> T2 {
            let batch = co_o[0];

            let mut rand = rng.gen::<T1>() * maximums[batch];
            let mut ret: T2 = usize::as_(class_size - 1);

            for (i, prob) in input.slice(s![batch, ..]).iter().enumerate() {
                let prob = prob.exp();
                if rand < prob {
                    ret = usize::as_(i);
                    break;
                }
                rand -= prob;
            }

            ret
        });

        Ok(output.into_tvalue())
    }
}

impl Op for Multinomial {
    fn name(&self) -> Cow<str> {
        "Multinomial".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Multinomial {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);

        let output = match input.datum_type() {
            // DatumType::F16 => self.eval_t0::<f16>(input), // TODO: implement random for f16
            DatumType::F32 => self.eval_t0::<f32>(input),
            DatumType::F64 => self.eval_t0::<f64>(input),
            dt => bail!("Unsupported input datum type for Multinomial: {:?}", dt),
        }?;

        Ok(tvec![output])
    }
}

impl TypedOp for Multinomial {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = if let Some(s) = inputs[0].shape.as_concrete() {
            s
        } else {
            bail!("Only constant input shape are supported in Multinomial")
        };

        let batch_size = input_shape[0];
        Ok(tvec!(self.dtype.fact([batch_size, self.sample_size as usize])))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Integer.tensor().named("input"),
        TypeName::Integer.named("dtype").default(6),
        TypeName::Integer.named("sample_size").default(1),
        TypeName::Integer.named("seed"),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Multinomial) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();

    let dtype = match op.dtype {
        DatumType::I32 => 6,
        DatumType::I64 => 7,
        dt => bail!("Unsupported datum type {:?} for ONNX Multinomial", dt),
    };

    let inv = if let Some(seed) = op.seed {
        invocation(
            "tract_onnx_multinomial",
            &[input],
            &[
                ("dtype", numeric(dtype)),
                ("sample_size", numeric(op.sample_size)),
                ("seed", numeric(seed)),
            ],
        )
    } else {
        invocation(
            "tract_onnx_multinomial",
            &[input],
            &[("dtype", numeric(dtype)), ("sample_size", numeric(op.sample_size))],
        )
    };

    Ok(Some(inv))
}

fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let dtype = match invocation.named_arg_as::<i64>(builder, "dtype")? {
        6 => DatumType::I32,
        7 => DatumType::I64,
        i => bail!("Unsupported datum type {} for ONNX Multinomial", i),
    };
    let sample_size = invocation.named_arg_as::<i64>(builder, "sample_size")? as _;
    let seed = invocation.named_arg_as(builder, "seed").ok();

    let op = Multinomial { dtype, sample_size, seed };
    builder.wire(op, &[input])
}
