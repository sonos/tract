use std::str::FromStr;

use tract_core::ops::array::Trilu;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::logic::Iff;
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
    let mut inputs = vec![q, k, v];
    if let Some(mask) = invocation.get_named_arg_as(builder, "mask")? {
        inputs.push(mask);
    };
    let scale: Option<f32> = invocation.get_named_arg_as(builder, "scale")?;
    let dt = DatumType::from_str(&invocation.named_arg_as::<String>(builder, "d_type")?)?;
    let inner_dt =
        DatumType::from_str(&invocation.named_arg_as::<String>(builder, "inner_dtype")?)?;
    let is_causal = invocation.named_arg_as(builder, "causal")?;
    builder.wire(Sdpa { scale: scale.map(|it| tensor0(it)), dt, inner_dt, is_causal }, &inputs)
}

#[derive(Debug, Clone, Hash)]
pub struct Sdpa {
    pub scale: Option<Tensor>,
    pub dt: DatumType,
    pub inner_dt: DatumType,
    pub is_causal: bool,
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
        let mut mask = if !inputs.is_empty() { Some(inputs.remove(0)) } else { None };

        let rank = q.rank();
        let k_shape = k.shape().iter().cloned().collect::<Vec<_>>();
        let q_shape = q.shape().iter().cloned().collect::<Vec<_>>();

        // Computing scaling factor for attention
        let d_k = k_shape[rank - 1] as f32;
        let scale = if let Some(scale) = self.scale.as_ref() {
            scale.cast_to_dt(self.inner_dt)?.into_owned()
        } else {
            tensor0(1.0 / d_k.sqrt()).cast_to_dt(self.inner_dt)?.into_owned()
        };

        // Construct causal mask if needed
        if self.is_causal {
            let (q_seq_len, k_seq_len) = (q_shape[rank - 2], k_shape[rank - 2]);
            let mut ones =
                unsafe { Tensor::uninitialized_dt(DatumType::F32, &[q_seq_len, k_seq_len])? };
            ones.fill_t(1.0_f32)?;

            // Build lower triangular matrix
            let k = tensor0(0_i64);
            let lower_triangle_ones =
                Trilu { upper: false }.eval(tvec![ones.into(), k.into()])?.remove(0);
            let cond_mask = lower_triangle_ones.cast_to::<bool>()?.into_owned();

            // Zeros for lower part
            let zeros = Tensor::zero_dt(self.inner_dt, &[q_seq_len, k_seq_len])?;

            // -inf for higher part
            let mut neg_infs =
                unsafe { Tensor::uninitialized_dt(self.inner_dt, &[q_seq_len, k_seq_len])? };
            neg_infs.fill_t(f32::NEG_INFINITY)?;

            let causal_mask_tensor = Iff
                .eval(tvec![
                    cond_mask.clone().into_tvalue(),
                    zeros.into_tvalue(),
                    neg_infs.into_tvalue(),
                ])?
                .remove(0);

            mask = Some(causal_mask_tensor);
        }

        // Computing attention matrix
        let axes = match rank {
            3 => "amk,ank->amn".parse().unwrap(),
            4 => "bhmk,bhnk->bhmn".parse().unwrap(),
            _ => unreachable!(),
        };
        let q_dot_kt = EinSum { axes, operating_dt: self.inner_dt, q_params: None }
            .eval(tvec![q, k])?
            .remove(0);
        let scaled_input = Mul.eval(q_dot_kt, scale.into_tvalue(), self.inner_dt)?;

        // Apply mask (causal or provided by user)
        let scaled_masked_input = if let Some(m) = mask {
            Add.eval(scaled_input.into_tvalue(), m, self.inner_dt)?
        } else {
            scaled_input
        };

        let attention_weights =
            Softmax::new(tvec![rank - 1], None, SoftmaxKind::Softmax(SoftmaxExp::Libc))
                .eval(tvec![scaled_masked_input.into()])?[0]
                .clone();

        // Final projection using V
        let axes = match rank {
            3 => "amk,akn->amn".parse().unwrap(),
            4 => "bhmn,bhnv->bhmv".parse().unwrap(),
            _ => unreachable!(),
        };
        let output = EinSum { axes, operating_dt: self.dt, q_params: None }
            .eval(tvec![attention_weights, v])?;
        Ok(output)
    }
}

impl TypedOp for Sdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if self.is_causal {
            ensure!(inputs.len() == 3, "Mask cannot be provided if is_causal=true")
        };
        let rank = inputs[0].rank();
        ensure!(rank == 3 || rank == 4, "Input tensors must be 3D or 4D");
        ensure!(
            inputs.iter().map(|it| it.rank()).all(|r| r == rank),
            "All inputs should have the same rank {}",
            rank
        );

        let q_shape = &inputs[0].shape.dims();
        let v_shape = &inputs[2].shape.dims();
        let output_shape = match rank {
            3 => {
                if let (&[b, seq_len, _], &[_, _, out_dim]) = (q_shape, v_shape) {
                    tvec!(b.clone(), seq_len.clone(), out_dim.clone())
                } else {
                    unreachable!()
                }
            }
            4 => {
                if let (&[b, n_heads, seq_len, _], &[_, _, _, out_dim]) = (q_shape, v_shape) {
                    tvec!(b.clone(), n_heads.clone(), seq_len.clone(), out_dim.clone())
                } else {
                    unreachable!()
                }
            }
            _ => unreachable!(),
        };

        let out_fact = inputs[0].datum_type().unwrap().fact(output_shape);
        Ok(tvec!(out_fact))
    }

    as_op!();
}
