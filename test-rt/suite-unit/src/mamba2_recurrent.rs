use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Dimension;
use tract_core::num_traits::Float;
use tract_transformers::ops::mamba2_recurrent::Mamba2Recurrent;

fn values<F: Datum + Float>(
    shape: &[usize],
    range: std::ops::RangeInclusive<i8>,
) -> BoxedStrategy<ArrayD<F>> {
    let shape = shape.to_vec();
    let len = shape.iter().product::<usize>();
    vec(range.prop_map(|i| F::from(i as f32 / 8.0).unwrap()), len..=len)
        .prop_map(move |v| ArrayD::from_shape_vec(shape.clone(), v).unwrap())
        .boxed()
}

/// x `[b, S, H, P]`, b/c `[b, S, G, N]`, log_decay `[b, S, H]`, state `[b, H, P, N]`.
#[derive(Debug, Clone)]
pub struct Mamba2RecurrentProblem<F>
where
    F: Datum + Float,
{
    x: ArrayD<F>,
    b: ArrayD<F>,
    c: ArrayD<F>,
    log_decay: ArrayD<F>,
    state: ArrayD<F>,
}

impl<F> Arbitrary for Mamba2RecurrentProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (1usize..3, 1usize..6, 1usize..3, 1usize..3, 1usize..6, 1usize..9)
            .prop_flat_map(|(batch, s_len, groups, ratio, head_dim, d_state)| {
                let heads = groups * ratio;
                (
                    values::<F>(&[batch, s_len, heads, head_dim], -8..=8),
                    values::<F>(&[batch, s_len, groups, d_state], -8..=8),
                    values::<F>(&[batch, s_len, groups, d_state], -8..=8),
                    values::<F>(&[batch, s_len, heads], -16..=0),
                    values::<F>(&[batch, heads, head_dim, d_state], -8..=8),
                )
            })
            .prop_map(|(x, b, c, log_decay, state)| Self { x, b, c, log_decay, state })
            .boxed()
    }
}

impl<F> Mamba2RecurrentProblem<F>
where
    F: Datum + Float,
{
    fn inputs(&self) -> TVec<Tensor> {
        tvec![
            self.x.clone().into_tensor(),
            self.b.clone().into_tensor(),
            self.c.clone().into_tensor(),
            self.log_decay.clone().into_tensor(),
            self.state.clone().into_tensor(),
        ]
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let wires = ["x", "b", "c", "log_decay", "initial_state"]
            .iter()
            .zip(self.inputs())
            .map(|(name, t)| model.add_source(*name, TypedFact::shape_and_dt_of(&t)))
            .collect::<TractResult<TVec<_>>>()?;
        let outputs = model.wire_node("mamba2", Mamba2Recurrent, &wires)?;
        model.select_output_outlets(&outputs)?;
        Ok(model)
    }

    /// Quadratic (SSD dual) form, independent of the recurrence:
    /// `y_t = sum_{j<=t} exp(sum_{j<k<=t} g_k) (c_t . b_j) x_j + exp(sum_{k<=t} g_k) state_0 . c_t`,
    /// with the final state accumulated the same way.
    fn reference(&self) -> (ArrayD<F>, ArrayD<F>) {
        let f = |a: &ArrayD<F>| a.mapv(|v| v.to_f32().unwrap());
        let (x, b, c, g, s0) =
            (f(&self.x), f(&self.b), f(&self.c), f(&self.log_decay), f(&self.state));
        let &[batch, s_len, heads, head_dim] = x.shape() else { unreachable!() };
        let (groups, d_state) = (b.shape()[2], b.shape()[3]);
        let ratio = heads / groups;
        let cum = |bi: usize, h: usize, from: usize, to: usize| -> f32 {
            (from..to).map(|k| g[[bi, k, h]]).sum::<f32>().exp()
        };
        let output = ArrayD::from_shape_fn(x.shape(), |ix| {
            let (bi, t, h, p) = (ix[0], ix[1], ix[2], ix[3]);
            let gr = h / ratio;
            let mut acc: f32 = (0..d_state).map(|n| s0[[bi, h, p, n]] * c[[bi, t, gr, n]]).sum();
            acc *= cum(bi, h, 0, t + 1);
            for j in 0..=t {
                let cb: f32 = (0..d_state).map(|n| c[[bi, t, gr, n]] * b[[bi, j, gr, n]]).sum();
                acc += cum(bi, h, j + 1, t + 1) * cb * x[[bi, j, h, p]];
            }
            F::from(acc).unwrap()
        });
        let state = ArrayD::from_shape_fn(vec![batch, heads, head_dim, d_state], |ix| {
            let (bi, h, p, n) = (ix[0], ix[1], ix[2], ix[3]);
            let mut acc = cum(bi, h, 0, s_len) * s0[[bi, h, p, n]];
            for j in 0..s_len {
                acc += cum(bi, h, j + 1, s_len) * x[[bi, j, h, p]] * b[[bi, j, h / ratio, n]];
            }
            F::from(acc).unwrap()
        });
        (output, state)
    }
}

impl<F> Test for Mamba2RecurrentProblem<F>
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
        let mut model = self.tract()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let inputs = self.inputs().into_iter().map(|t| t.into_tvalue()).collect();
        let outputs = runtime.prepare(model)?.run(inputs)?;
        outputs[0].clone().into_tensor().close_enough(&output_ref.into_tensor(), approx)?;
        outputs[1].clone().into_tensor().close_enough(&state_ref.into_tensor(), approx)
    }
}

impl<F> Mamba2RecurrentProblem<F>
where
    F: Datum + Float,
{
    /// Deterministic values in [-1, 1] (log_decay in [-2, 0]) for the given dims.
    fn sized(
        batch: usize,
        s_len: usize,
        groups: usize,
        heads: usize,
        head_dim: usize,
        d_state: usize,
    ) -> Self {
        let fill = |shape: &[usize], seed: usize, scale: f32, offset: f32| {
            ArrayD::from_shape_fn(shape, |ix| {
                let i = ix.slice().iter().fold(seed as u64, |acc, &v| acc * 31 + v as u64);
                let unit = (i.wrapping_mul(2654435761) % 1000) as f32 / 1000.0;
                F::from(unit * scale + offset).unwrap()
            })
        };
        Self {
            x: fill(&[batch, s_len, heads, head_dim], 1, 2.0, -1.0),
            b: fill(&[batch, s_len, groups, d_state], 2, 2.0, -1.0),
            c: fill(&[batch, s_len, groups, d_state], 3, 2.0, -1.0),
            log_decay: fill(&[batch, s_len, heads], 4, 2.0, -2.0),
            state: fill(&[batch, heads, head_dim, d_state], 5, 2.0, -1.0),
        }
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<Mamba2RecurrentProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<Mamba2RecurrentProblem<f16>>("proptest_f16", ());
    for d_state in [128, 200, 256] {
        suite.add_test(
            format!("state_{d_state}_f32"),
            Mamba2RecurrentProblem::<f32>::sized(1, 3, 2, 8, 16, d_state),
        );
        suite.add_test(
            format!("state_{d_state}_f16"),
            Mamba2RecurrentProblem::<f16>::sized(1, 3, 2, 8, 16, d_state),
        );
    }
    suite.add_test("batch_2", Mamba2RecurrentProblem::<f32>::sized(2, 4, 4, 8, 8, 64));
    Ok(suite)
}
