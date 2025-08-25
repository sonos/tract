use infra::{Test, TestSuite};
use proptest::prelude::{Arbitrary, BoxedStrategy, Strategy};
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_ndarray::IxDyn;
use tract_transformers::ops::sdpa::Sdpa;

use crate::tensor;

#[derive(Clone, Debug)]
pub struct SdpaProblemParams {
    q_emb: usize,
}

impl Default for SdpaProblemParams {
    fn default() -> Self {
        Self { q_emb: 256 }
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
        (1..10usize, 1..128usize)
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

impl<F> SdpaProblem<F>
where
    F: Datum + Float,
{
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
                datum_type: F::datum_type(),
                acc_datum_type: DatumType::F32,
                is_causal: self.is_causal,
            },
            &[q_in, k_in, v_in],
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let [b, seq_len, _] = self.q.shape() else { unreachable!() };
        let [_, _, v_emb] = self.v.shape() else { unreachable!() };
        ArrayD::zeros(IxDyn(&[*b, *seq_len, *v_emb]))
    }
}

impl<F> Test for SdpaProblem<F>
where
    F: Float + Datum,
{
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
    suite.add_arbitrary::<SdpaProblem<f16>>("proptest_f16", params.clone());
    Ok(suite)
}
