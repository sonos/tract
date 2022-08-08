use crate::model::ParsingContext;
use crate::pb::*;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use tract_hir::internal::*;
use tract_hir::tract_ndarray::s;
use tract_hir::tract_num_traits::{AsPrimitive, Zero, Float};

pub fn multinomial(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dtype = match node.get_attr_opt("dtype")?.unwrap_or(6) {
        6 => DatumType::I32,
        7 => DatumType::I64,
        i => bail!("Unsupported datum type {} for ONNX Multinomial", i),
    };
    let sample_size = node.get_attr_opt("sample_size")?.unwrap_or(1);
    let seed = node.get_attr_opt("seed")?.unwrap_or(0.0f32);

    Ok((Box::new(Multinomial { dtype, sample_size, seed }), vec![]))
}

/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
#[derive(Clone, new, Debug, Educe)]
#[educe(Hash)]
struct Multinomial {
    dtype: DatumType,
    sample_size: i32,
    #[educe(Hash(method = "hash_f32"))]
    seed: f32,
}

impl Multinomial {
    fn eval_t0<T1>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>>
    where
        T1: Datum + std::ops::SubAssign + Float,
        Standard: Distribution<T1>,
    {
        match self.dtype {
            DatumType::I32 => self.eval_t::<T1, i32>(input),
            DatumType::I64 => self.eval_t::<T1, i64>(input),
            dt => bail!("Unsupported output datum type for Multinomial: {:?}", dt),
        }
    }
    fn eval_t<T1, T2>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>>
    where
        T1: Datum + std::ops::SubAssign + Float,
        Standard: Distribution<T1>,
        T2: Datum + Zero + Copy,
        usize: AsPrimitive<T2>,
    {
        let batch_size = input.shape()[0];
        let class_size = input.shape()[1];
        let out_shape: &[_] = &[batch_size, self.sample_size as usize];

        let input = input.to_array_view::<T1>()?;

        let output = tract_ndarray::ArrayD::from_shape_fn(out_shape, |co_o| -> T2 {
            let batch = co_o[0];

            let mut rand: T1 = rand::random();
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

        Ok(output.into_arc_tensor())
    }
}

impl_dyn_hash!(Multinomial);

impl Op for Multinomial {
    fn name(&self) -> Cow<str> {
        "Multinomial".into()
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EvalOp for Multinomial {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);

        let output = match input.datum_type() {
            // DatumType::F16 => self.eval_t0::<f16>(input),
            DatumType::F32 => self.eval_t0::<f32>(input),
            DatumType::F64 => self.eval_t0::<f64>(input),
            dt => bail!("Unsupported input datum type for Multinomial: {:?}", dt),
        }?;

        Ok(tvec![output])
    }
}

impl InferenceRulesOp for Multinomial {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        check_input_arity(&inputs, 1)?;

        // inputs[0]: tensor(float16), tensor(float), tensor(double) ; [batch_size, class_size]
        // outputs[0]: tensor(int32), tensor(int64) {depending on self.datum_type} ; [batch_size, sample_size]

        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?; // batch_size
        s.equals(&outputs[0].shape[1], self.sample_size.to_dim())?; // sample_size

        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Multinomial {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = if let Some(s) = inputs[0].shape.as_concrete() {
            s
        } else {
            bail!("Only constant input shape are supported in Multinomial")
        };

        let batch_size = input_shape[0];
        Ok(tvec!(self.dtype.fact(&[batch_size, self.sample_size as usize])))
    }

    fn declutter(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }
}
