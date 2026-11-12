use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_transformers::ops::causal_conv1d_update::CausalConv1dUpdate;

const TAPS: usize = 4;

fn values<F: Datum + Float>(shape: &[usize]) -> BoxedStrategy<ArrayD<F>> {
    let shape = shape.to_vec();
    let len = shape.iter().product::<usize>();
    vec((-8i8..=8).prop_map(|i| F::from(i as f32 / 8.0).unwrap()), len..=len)
        .prop_map(move |v| ArrayD::from_shape_vec(shape.clone(), v).unwrap())
        .boxed()
}

/// input `[b, C, S]`, weight `[C, k]`, state `[b, C, k]`, optional bias `[C]`, checked
/// against the convolution written out over `concat(state, input)`.
#[derive(Debug, Clone)]
pub struct CausalConv1dUpdateProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
    weight: ArrayD<F>,
    state: ArrayD<F>,
    bias: Option<ArrayD<F>>,
}

impl<F> Arbitrary for CausalConv1dUpdateProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (1usize..3, 1usize..9, 1usize..6, any::<bool>())
            .prop_flat_map(|(batch, channels, s_len, with_bias)| {
                let bias = if with_bias {
                    values::<F>(&[channels]).prop_map(Some).boxed()
                } else {
                    Just(None).boxed()
                };
                (
                    values::<F>(&[batch, channels, s_len]),
                    values::<F>(&[channels, TAPS]),
                    values::<F>(&[batch, channels, TAPS]),
                    bias,
                )
            })
            .prop_map(|(input, weight, state, bias)| Self { input, weight, state, bias })
            .boxed()
    }
}

impl<F> CausalConv1dUpdateProblem<F>
where
    F: Datum + Float,
{
    fn inputs(&self) -> TVec<Tensor> {
        let mut inputs = tvec![
            self.input.clone().into_tensor(),
            self.weight.clone().into_tensor(),
            self.state.clone().into_tensor(),
        ];
        inputs.extend(self.bias.clone().map(|b| b.into_tensor()));
        inputs
    }

    fn reference(&self) -> (ArrayD<F>, ArrayD<F>) {
        let f = |a: &ArrayD<F>| a.mapv(|v| v.to_f32().unwrap());
        let (input, weight, state) = (f(&self.input), f(&self.weight), f(&self.state));
        let bias = self.bias.as_ref().map(f);
        let &[_, _, s_len] = input.shape() else { unreachable!() };
        let full = |b: usize, c: usize, t: usize| {
            if t < TAPS { state[[b, c, t]] } else { input[[b, c, t - TAPS]] }
        };
        let output = ArrayD::from_shape_fn(input.shape(), |ix| {
            let (b, c, t) = (ix[0], ix[1], ix[2]);
            let sum = bias.as_ref().map_or(0.0, |bias| bias[[c]])
                + (0..TAPS).map(|tap| weight[[c, tap]] * full(b, c, t + 1 + tap)).sum::<f32>();
            F::from(sum / (1.0 + (-sum).exp())).unwrap()
        });
        let final_state = ArrayD::from_shape_fn(state.shape(), |ix| {
            F::from(full(ix[0], ix[1], s_len + ix[2])).unwrap()
        });
        (output, final_state)
    }
}

impl<F> Test for CausalConv1dUpdateProblem<F>
where
    F: Datum + Float,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let (output_ref, state_ref) = self.reference();
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wires = ["input", "weight", "initial_state", "bias"]
            .iter()
            .zip(self.inputs())
            .map(|(name, t)| model.add_source(*name, TypedFact::shape_and_dt_of(&t)))
            .collect::<TractResult<TVec<_>>>()?;
        let outputs = model.wire_node("conv", CausalConv1dUpdate, &wires)?;
        model.select_output_outlets(&outputs)?;
        let inputs = self.inputs().into_iter().map(|t| t.into_tvalue()).collect();
        let outputs = runtime.prepare(model)?.run(inputs)?;
        outputs[0].clone().into_tensor().close_enough(&output_ref.into_tensor(), approx)?;
        outputs[1].clone().into_tensor().close_enough(&state_ref.into_tensor(), approx)
    }
}

impl<F> CausalConv1dUpdateProblem<F>
where
    F: Datum + Float,
{
    /// Deterministic values in [-1, 1] for the given dims.
    fn sized(batch: usize, channels: usize, s_len: usize, with_bias: bool) -> Self {
        let fill = |shape: &[usize], seed: u64| {
            let len = shape.iter().product::<usize>() as u64;
            let v = (0..len)
                .map(|i| {
                    let unit = ((i + seed * 7919).wrapping_mul(2654435761) % 17) as f32 / 8.0;
                    F::from(unit - 1.0).unwrap()
                })
                .collect();
            ArrayD::from_shape_vec(shape, v).unwrap()
        };
        Self {
            input: fill(&[batch, channels, s_len], 1),
            weight: fill(&[channels, TAPS], 2),
            state: fill(&[batch, channels, TAPS], 3),
            bias: with_bias.then(|| fill(&[channels], 4)),
        }
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<CausalConv1dUpdateProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<CausalConv1dUpdateProblem<f16>>("proptest_f16", ());
    suite.add_test("bias_f16", CausalConv1dUpdateProblem::<f16>::sized(2, 48, 3, true));
    suite.add_test("no_bias_f16", CausalConv1dUpdateProblem::<f16>::sized(2, 48, 3, false));
    suite.add_test("bias_f32", CausalConv1dUpdateProblem::<f32>::sized(2, 48, 3, true));
    Ok(suite)
}
