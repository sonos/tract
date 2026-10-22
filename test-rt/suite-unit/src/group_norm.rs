use std::collections::HashMap;
use std::fmt;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Dimension;
use tract_core::num_traits::Float;
use tract_core::ops::nn::GroupNorm;

use crate::tensor;

#[derive(Clone)]
pub struct GroupNormProblem<F>
where
    F: Datum + Float,
{
    input: Tensor,
    axis: usize,
    groups: usize,
    reduced_axes: TVec<usize>,
    eps: f32,
    _phantom: PhantomData<F>,
}

impl<F> fmt::Debug for GroupNormProblem<F>
where
    F: Datum + Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input:{:?} Axis:{} Groups:{} Reduced:{:?} Epsilon:{}",
            self.input, self.axis, self.groups, self.reduced_axes, self.eps
        )
    }
}

impl<F> Arbitrary for GroupNormProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<GroupNormProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (1usize..3, 1usize..5, 1usize..5, vec(1usize..7, 0..3), any::<bool>(), 1f32..1000f32)
            .prop_flat_map(|(n, groups, per_group, spatial, channel_last, eps)| {
                let c = groups * per_group;
                let mut shape = vec![n];
                let (axis, reduced_axes): (usize, TVec<usize>) = if channel_last {
                    shape.extend(&spatial);
                    shape.push(c);
                    (spatial.len() + 1, (1..spatial.len() + 1).collect())
                } else {
                    shape.push(c);
                    shape.extend(&spatial);
                    (1, (2..spatial.len() + 2).collect())
                };
                (tensor::<F>(&shape), Just((axis, groups, reduced_axes, eps / 1e5)))
            })
            .prop_map(|(input, (axis, groups, reduced_axes, eps))| Self {
                input: input.into(),
                axis,
                groups,
                reduced_axes,
                eps,
                _phantom: PhantomData,
            })
            .boxed()
    }
}

impl<F> GroupNormProblem<F>
where
    F: Datum + Float,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?;
        let op = GroupNorm {
            axis: self.axis,
            groups: self.groups,
            reduced_axes: self.reduced_axes.clone(),
            eps: tensor0(self.eps).into_arc_tensor(),
        };
        let output = model.wire_node("group_norm", op, &[input])?;
        model.select_output_outlets(&output)?;
        model.into_decluttered()
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.cast_to::<f32>().unwrap();
        let a = input.to_plain_array_view::<f32>().unwrap();
        let per_group = a.shape()[self.axis] / self.groups;
        let key = |idx: &[usize]| -> Vec<usize> {
            (0..idx.len())
                .filter(|ax| !self.reduced_axes.contains(ax))
                .map(|ax| if ax == self.axis { idx[ax] / per_group } else { idx[ax] })
                .collect()
        };
        let mut stats: HashMap<Vec<usize>, (f64, f64, f64)> = HashMap::new();
        for (idx, v) in a.indexed_iter() {
            let s = stats.entry(key(idx.slice())).or_default();
            s.0 += 1.0;
            s.1 += *v as f64;
            s.2 += (*v as f64) * (*v as f64);
        }
        ArrayD::from_shape_fn(a.raw_dim(), |idx| {
            let (n, sum, sum_sq) = stats[&key(idx.slice())];
            let mean = sum / n;
            let var = sum_sq / n - mean * mean;
            let y = (a[&idx] as f64 - mean) / (var + self.eps as f64).sqrt();
            F::from(y).unwrap()
        })
    }
}

impl<F> Test for GroupNormProblem<F>
where
    F: Datum + Float,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into()))?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<GroupNormProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<GroupNormProblem<f16>>("proptest_f16", ());
    suite.add(
        "channel_last_f32",
        GroupNormProblem::<f32> {
            input: Tensor::from_shape(
                &[2, 3, 2, 6],
                &(0..72).map(|i| ((i * 7) % 11) as f32 - 4.0).collect::<Vec<_>>(),
            )
            .unwrap(),
            axis: 3,
            groups: 3,
            reduced_axes: tvec!(1, 2),
            eps: 1e-5,
            _phantom: PhantomData,
        },
    );
    Ok(suite)
}
