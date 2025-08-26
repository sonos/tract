use infra::{Test, TestSuite};
use proptest::{
    prelude::{any, Arbitrary, BoxedStrategy, Just, Strategy},
    prop_oneof,
};
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_ndarray::{s, Array2, Array3, ArrayView2};
use tract_transformers::ops::sdpa::Sdpa;

use crate::tensor;

#[derive(Clone, Debug)]
pub struct SdpaProblemParams {
    q_emb: usize,
}

impl Default for SdpaProblemParams {
    fn default() -> Self {
        Self { q_emb: 32 }
    }
}

#[derive(Debug, Clone)]
pub struct SdpaProblem<F>
where
    F: Datum + Float,
{
    q: ArrayD<F>,
    k: ArrayD<F>,
    v: ArrayD<F>,
    mask: Option<ArrayD<F>>,
    scale: Option<F>,
    is_causal: bool,
}

impl Arbitrary for SdpaProblem<f32> {
    type Parameters = SdpaProblemParams;
    type Strategy = BoxedStrategy<SdpaProblem<f32>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        (1..3usize, 1..16usize)
            .prop_flat_map(move |(b, seq_len)| {
                let q = tensor(&[b, seq_len, args.q_emb]);
                let k = tensor(&[b, seq_len, args.q_emb]);
                let v = tensor(&[b, seq_len, args.q_emb]);

                let scale_strategy = prop_oneof![Just(None), (0.1f32..2.0).prop_map(Some)];
                let causal_strategy = any::<bool>();
                (Just((b, seq_len)), q, k, v, scale_strategy, causal_strategy)
            })
            .prop_flat_map(|((b, seq_len), q, k, v, scale, is_causal)| {
                let mask_strategy = if is_causal {
                    Just(None).boxed()
                } else {
                    prop_oneof![Just(None), tensor(&[b, seq_len, seq_len]).prop_map(Some)].boxed()
                };

                (Just(q), Just(k), Just(v), Just(scale), Just(is_causal), mask_strategy)
            })
            .prop_map(|(q, k, v, scale, is_causal, mask)| SdpaProblem {
                q,
                k,
                v,
                mask,
                scale,
                is_causal,
            })
            .boxed()
    }
}

impl SdpaProblem<f32> {
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let q = self.q.clone().into_tensor();
        let k = self.k.clone().into_tensor();
        let v = self.v.clone().into_tensor();
        let scale = self.scale.map(|it| tensor0(it));
        let q_in = model.add_source("Q", TypedFact::shape_and_dt_of(&q))?;
        let k_in = model.add_source("K", TypedFact::shape_and_dt_of(&k))?;
        let v_in = model.add_source("V", TypedFact::shape_and_dt_of(&v))?;
        let mut inputs = vec![q_in, k_in, v_in];
        if let Some(mask) = &self.mask {
            let mask_in = model
                .add_source("mask", TypedFact::shape_and_dt_of(&mask.clone().into_tensor()))?;
            inputs.push(mask_in);
        }
        let output = model.wire_node(
            "SDPA",
            Sdpa {
                scale,
                datum_type: DatumType::F32,
                acc_datum_type: DatumType::F32,
                is_causal: self.is_causal,
            },
            &inputs,
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn softmax(input: &Array2<f32>, axis: usize) -> Array2<f32> {
        let axis = tract_ndarray::Axis(axis);

        let max_per_axis =
            input.map_axis(axis, |lane| lane.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

        let shifted = input - &max_per_axis.insert_axis(axis);
        let exp = shifted.mapv(f32::exp);
        let sum_exp = exp.sum_axis(axis);

        let norm = sum_exp.insert_axis(axis);

        &exp / &norm
    }

    fn scaled_dot_product_attention_2d(
        queries: &ArrayView2<f32>,
        keys: &ArrayView2<f32>,
        values: &ArrayView2<f32>,
        mask: Option<ArrayView2<f32>>,
        scale: Option<f32>,
        is_causal: bool,
    ) -> Array2<f32> {
        let d_k = keys.shape()[1] as f32;
        let scale_factor = scale.unwrap_or(1.0 / d_k.sqrt());
        let q_dot_kt = queries.dot(&keys.t());
        let mut scaled_input = q_dot_kt * scale_factor;

        if is_causal {
            scaled_input.indexed_iter_mut().for_each(|((r, c), value)| {
                if c > r {
                    *value = f32::NEG_INFINITY;
                }
            });
        }

        if let Some(m) = mask {
            scaled_input += &m;
        }

        let att_weights = Self::softmax(&scaled_input, 1);
        let output = att_weights.dot(values);
        output
    }

    fn reference(&self) -> Array3<f32> {
        let [b, seq_len, _] = self.q.shape() else { unreachable!() };
        let [_, _, v_emb] = self.v.shape() else { unreachable!() };
        let mut output = Array3::<f32>::zeros((*b, *seq_len, *v_emb));
        for i in 0..*b {
            let q_slice = self.q.slice(s![i, .., ..]);
            let k_slice = self.k.slice(s![i, .., ..]);
            let v_slice = self.v.slice(s![i, .., ..]);
            let mask_slice = self.mask.as_ref().map(|m| m.slice(s![i, .., ..]));
            let output_2d = Self::scaled_dot_product_attention_2d(
                &q_slice,
                &k_slice,
                &v_slice,
                mask_slice,
                self.scale,
                self.is_causal,
            );

            output.slice_mut(s![i, .., ..]).assign(&output_2d);
        }

        output
    }
}

impl Test for SdpaProblem<f32> {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn tract_core::runtime::Runtime,
        approx: tract_core::internal::Approximation,
    ) -> infra::TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut inputs = tvec![
            self.q.clone().into_tvalue(),
            self.k.clone().into_tvalue(),
            self.v.clone().into_tvalue()
        ];
        if let Some(mask) = &self.mask {
            inputs.push(mask.clone().into_tvalue());
        }
        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    let params = SdpaProblemParams::default();
    suite.add_arbitrary::<SdpaProblem<f32>>("proptest_f32", params.clone());
    Ok(suite)
}
