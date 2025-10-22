use core::f32;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax;

use crate::tensor;

#[derive(Debug, Clone)]
pub struct ScaledMaskedSoftmaxProblem<F>
where
    F: Datum + Float,
{
    input: ArrayD<F>,
    mask: ArrayD<F>,
    scale: F,
}

impl<F> Arbitrary for ScaledMaskedSoftmaxProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<ScaledMaskedSoftmaxProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        // ScaledMaskSoftmax assumes rank 3 for mask and input
        let dim = 1usize..20;
        vec(dim.clone(), 3..=3)
            .prop_flat_map(|shape| {
                (tensor::<F>(&shape), tensor::<f32>(&shape), -10..=10i32).prop_map(
                    |(input, mask, scale)| {
                        let mask = mask.mapv(|x| {
                            if x >= 0. {
                                F::from(0).unwrap()
                            } else {
                                F::from(f32::NEG_INFINITY).unwrap()
                            }
                        });
                        let scale = scale as f32 / 10.;
                        Self { input, mask, scale: F::from(scale).unwrap() }
                    },
                )
            })
            .boxed()
    }
}

impl<F> ScaledMaskedSoftmaxProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let input = model
            .add_source("input", TypedFact::shape_and_dt_of(&self.input.clone().into_tensor()))?;
        let mask = model
            .add_source("mask", TypedFact::shape_and_dt_of(&self.mask.clone().into_tensor()))?;

        let output = model.wire_node(
            "scaled_masked_softmax",
            ScaledMaskedSoftmax { scale: tensor0(self.scale).into_arc_tensor() },
            &[input, mask],
        )?;
        model.set_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn softmax(input: &ArrayD<F>, axis: usize) -> ArrayD<F> {
        let axis = tract_ndarray::Axis(axis);

        let max_per_axis = input.map_axis(axis, |lane| {
            lane.fold(F::from(f32::NEG_INFINITY).unwrap(), |a, &b| a.max(b))
        });

        let shifted = input - &max_per_axis.insert_axis(axis);
        let exp = shifted.mapv(F::exp);
        let sum_exp = exp.sum_axis(axis);

        let norm = sum_exp.insert_axis(axis);

        &exp / &norm
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.clone();
        let mask = self.mask.clone();

        let scaled_input = input.mapv(|x| x * self.scale);
        let masked_input = scaled_input + mask;

        Self::softmax(&masked_input, self.input.shape().len() - 1)
    }
}

impl<F> Test for ScaledMaskedSoftmaxProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        //dbg!(&self.input, &self.mask);
        let mut output = runtime
            .prepare(model)?
            .run(tvec![self.input.clone().into_tvalue(), self.mask.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        //dbg!(&reference, &output);
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<ScaledMaskedSoftmaxProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<ScaledMaskedSoftmaxProblem<f16>>("proptest_f16", ());

    suite.add(
        "trivial_f32_0",
        ScaledMaskedSoftmaxProblem {
            input: tensor3(&[[[0f32]]]).into_array()?,
            mask: tensor3(&[[[0f32]]]).into_array()?,
            scale: 1f32,
        },
    );
    suite.add(
        "trivial_f32_1",
        ScaledMaskedSoftmaxProblem {
            input: tensor3(&[[[0f32, 0f32], [0f32, 0f32]]]).into_array()?,
            mask: tensor3(&[[[f32::NEG_INFINITY, 0f32], [0f32, 0f32]]]).into_array()?,
            scale: 0f32,
        },
    );
    Ok(suite)
}
