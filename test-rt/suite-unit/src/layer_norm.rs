use std::fmt;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Axis;
use tract_core::num_traits::Float;
use tract_core::ops::nn::LayerNorm;

use crate::tensor;

#[derive(Clone)]
pub struct LayerNormProblem<F>
where
    F: Datum + Float,
{
    input: Tensor,
    axis: usize,
    affine: Option<(Vec<f32>, Vec<f32>)>,
    eps: f32,
    _phantom: PhantomData<F>,
}

impl<F> fmt::Debug for LayerNormProblem<F>
where
    F: Datum + Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input:{:?} Axis:{} Affine:{:?} Epsilon:{}",
            self.input, self.axis, self.affine, self.eps
        )
    }
}

impl<F> Arbitrary for LayerNormProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<LayerNormProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (vec(1usize..7, 1..4), any::<bool>(), 1f32..1000f32)
            .prop_flat_map(|(shape, affine, eps)| {
                let axis = 0..shape.len();
                (tensor::<F>(&shape), axis, Just(shape), Just(affine), Just(eps / 1e5))
            })
            .prop_flat_map(|(input, axis, shape, affine, eps)| {
                let dim = shape[axis];
                let affine = if affine {
                    (vec(-2f32..2f32, dim), vec(-2f32..2f32, dim)).prop_map(Some).boxed()
                } else {
                    Just(None).boxed()
                };
                (Just(input), Just(axis), affine, Just(eps))
            })
            .prop_map(|(input, axis, affine, eps)| Self {
                input: input.into(),
                axis,
                affine,
                eps,
                _phantom: PhantomData,
            })
            .boxed()
    }
}

impl<F> LayerNormProblem<F>
where
    F: Datum + Float,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let mut inputs = tvec!(model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?);
        if let Some((scale, bias)) = &self.affine {
            inputs.push(model.add_const("scale", tensor1(scale))?);
            inputs.push(model.add_const("bias", tensor1(bias))?);
        }
        let op = LayerNorm { axis: self.axis, eps: tensor0(self.eps).into_arc_tensor() };
        let output = model.wire_node("layer_norm", op, &inputs)?;
        model.select_output_outlets(&output)?;
        model.into_decluttered()
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.cast_to::<f32>().unwrap();
        let a = input.to_plain_array_view::<f32>().unwrap().mapv(|v| v as f64);
        let n = a.shape()[self.axis] as f64;
        let mean = a.sum_axis(Axis(self.axis)).insert_axis(Axis(self.axis)) / n;
        let centered = &a - &mean;
        let var =
            centered.mapv(|v| v * v).sum_axis(Axis(self.axis)).insert_axis(Axis(self.axis)) / n;
        let mut y = centered / var.mapv(|v| (v + self.eps as f64).sqrt());
        if let Some((scale, bias)) = &self.affine {
            for (d, mut lane) in y.axis_iter_mut(Axis(self.axis)).enumerate() {
                lane.mapv_inplace(|v| v * scale[d] as f64 + bias[d] as f64);
            }
        }
        y.mapv(|v| F::from(v).unwrap())
    }
}

impl<F> Test for LayerNormProblem<F>
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

fn rows<F: Datum + Float>(shape: &[usize], axis: usize, affine: bool) -> LayerNormProblem<F> {
    let len = shape.iter().product::<usize>();
    let data = (0..len)
        .map(|i| {
            F::from(((i * 37) % 101) as f32 / 20.0 - 2.0 + 6.0 * i as f32 / len as f32).unwrap()
        })
        .collect::<Vec<_>>();
    let dim = shape[axis];
    let affine = affine.then(|| {
        let scale = (0..dim).map(|d| 0.5 + (d % 7) as f32 / 4.0).collect();
        let bias = (0..dim).map(|d| (d % 5) as f32 / 3.0 - 0.5).collect();
        (scale, bias)
    });
    LayerNormProblem {
        input: Tensor::from_shape(shape, &data).unwrap(),
        axis,
        affine,
        eps: 1e-5,
        _phantom: PhantomData,
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<LayerNormProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<LayerNormProblem<f16>>("proptest_f16", ());
    suite.add("rows_640_affine_f16", rows::<f16>(&[2, 9, 640], 2, true));
    suite.add("rows_1280_f16", rows::<f16>(&[3, 1280], 1, false));
    suite.add("rows_2048_affine_f16", rows::<f16>(&[5, 2048], 1, true));
    suite.add("rows_1024_affine_f32", rows::<f32>(&[5, 1024], 1, true));
    suite.add("rows_too_long_f16", rows::<f16>(&[3, 2056], 1, true));
    suite.add("rows_not_vector_multiple_f32", rows::<f32>(&[4, 50], 1, true));
    suite.add("strided_axis_affine_f32", rows::<f32>(&[2, 96, 33], 1, true));
    suite.add("many_rows_f32", rows::<f32>(&[70000, 8], 1, true));
    Ok(suite)
}
