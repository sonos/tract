use infra::{Test, TestSuite};
use proptest::prelude::{Arbitrary, BoxedStrategy, Strategy};
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
        Self { q_emb: 64 }
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
    scale: Option<F>,
    is_causal: bool,
}

impl<F> Arbitrary for SdpaProblem<F>
where
    F: Datum + Float,
{
    type Parameters = SdpaProblemParams;
    type Strategy = BoxedStrategy<SdpaProblem<F>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        (1..3usize, 1..64usize)
            .prop_flat_map(move |(b, seq_len)| {
                let q = tensor(&[b, seq_len, args.q_emb]);
                let k = tensor(&[b, seq_len, args.q_emb]);
                let v = tensor(&[b, seq_len, args.q_emb]);
                (q, k, v)
            })
            .prop_map(move |(q, k, v)| SdpaProblem { q, k, v, scale: None, is_causal: false })
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
        let output = model.wire_node(
            "SDPA",
            Sdpa {
                scale,
                datum_type: DatumType::F32,
                acc_datum_type: DatumType::F32,
                is_causal: self.is_causal,
            },
            &[q_in, k_in, v_in],
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
    ) -> Array2<f32> {
        let d_k = keys.shape()[1] as f32;
        let q_dot_kt = queries.dot(&keys.t());
        let scaled_input = q_dot_kt / d_k.sqrt();
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

            let output_2d = Self::scaled_dot_product_attention_2d(&q_slice, &k_slice, &v_slice);

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
        let mut output = runtime.prepare(model)?.run(tvec![
            self.q.clone().into_tvalue(),
            self.k.clone().into_tvalue(),
            self.v.clone().into_tvalue()
        ])?;
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
