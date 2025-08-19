use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::{Add, Mul};
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_transformers_sdpa",
        &[
            TypeName::Scalar.tensor().named("q"),
            TypeName::Scalar.tensor().named("k"),
            TypeName::Scalar.tensor().named("v"),
            TypeName::Scalar.tensor().named("mask"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scaled_masked_softmax,
    );
}

fn de_scaled_masked_softmax(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let v = invocation.named_arg_as(builder, "v")?;
    let mask = invocation.named_arg_as(builder, "mask")?;
    builder.wire(Sdpa { scale: None }, &[q, k, v, mask])
}

#[derive(Debug, Clone, Hash)]
pub struct Sdpa {
    pub scale: Option<Tensor>,
}

impl Op for Sdpa {
    fn name(&self) -> StaticName {
        "SDPA".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {:?}", self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for Sdpa {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let q = inputs.remove(0);
        let k = inputs.remove(0);
        let v = inputs.remove(0);
        let mask = inputs.remove(0);

        ensure!(q.rank() == 3, "Query tensor must be 3D");
        ensure!(k.rank() == 3, "Key tensor must be 3D");
        ensure!(v.rank() == 3, "Value tensor must be 3D");

        let d_k = k.shape()[2] as f32;

        let scale = if let Some(scale) = self.scale.as_ref() {
            let mut scale = scale.cast_to_dt(DatumType::F32)?.into_owned();
            scale.as_slice_mut::<f32>()?.iter_mut().for_each(|it| *it /= d_k.sqrt());
            scale
        } else {
            tensor0(1.0 / d_k.sqrt())
        };

        let axes = "amk,ank->amn".parse().unwrap();
        let q_dot_kt = EinSum { axes, operating_dt: DatumType::F32, q_params: None }
            .eval(tvec![q, k])?
            .remove(0);

        let scaled_input = Mul.eval(q_dot_kt, scale.into_tvalue(), DatumType::F32)?;
        let scaled_masked_input = Add.eval(scaled_input.into_tvalue(), mask, DatumType::F32)?;
        let attention_weights =
            Softmax::new(tvec![2], None, SoftmaxKind::Softmax(SoftmaxExp::Libc))
                .eval(tvec![scaled_masked_input.into()])?[0]
                .clone();

        let axes = "amk,akn->amn".parse().unwrap();
        let output = EinSum { axes, operating_dt: DatumType::F32, q_params: None }
            .eval(tvec![attention_weights, v])?;
        Ok(output)
    }
}

impl TypedOp for Sdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let q_shape = &inputs[0].shape;
        let v_shape = &inputs[2].shape;

        let output_shape = tvec!(q_shape[0].clone(), q_shape[1].clone(), v_shape[2].clone());
        let out_fact = inputs[0].datum_type().unwrap().fact(output_shape);
        Ok(tvec!(out_fact))
    }

    as_op!();
}
