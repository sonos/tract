use core::f32;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::{Array5, ArrayD, Axis};
use tract_core::num_traits::{Float, Zero};
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
        // ScaledMaskSoftmax accepts ranks 2 to 5
        (vec(1usize..20, 2..=5), any::<bool>(), any::<bool>())
            .prop_flat_map(|(q_shape, broadcast_batch, broadcast_head)| {
                let mut m_shape = q_shape.clone();
                if broadcast_batch {
                    m_shape[0] = 1
                };
                if broadcast_head {
                    m_shape[1] = 1
                };
                (tensor::<F>(&q_shape), tensor::<f32>(&m_shape), -10..=10i32).prop_map(
                    |(input, mask, scale)| {
                        let mask = mask.mapv(|x| {
                            if x >= 0. {
                                F::from(0).unwrap()
                            } else {
                                F::from(f32::NEG_INFINITY).unwrap()
                            }
                        });
                        let scale = if scale.is_zero() { 1.0 } else { scale as f32 / 10. };
                        Self {
                            input: input.into_dimensionality().unwrap(),
                            mask: mask.into_dimensionality().unwrap(),
                            scale: F::from(scale).unwrap(),
                        }
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

    fn softmax(input: &Array5<F>, axis: usize) -> TractResult<Array5<F>> {
        let axis = tract_ndarray::Axis(axis);

        let max_per_axis = input.map_axis(axis, |lane| {
            lane.fold(F::from(f32::NEG_INFINITY).unwrap(), |a, &b| a.max(b))
        });

        let shifted = input - &max_per_axis.insert_axis(axis);
        let exp = shifted.mapv(F::exp);
        let sum_exp = exp.sum_axis(axis);

        let norm = sum_exp.insert_axis(axis);

        Ok(&exp / &norm)
    }

    fn reference(&self) -> TractResult<ArrayD<F>> {
        ensure!(self.input.ndim() == self.mask.ndim());
        let mut input = self.input.view();
        let mut mask = self.mask.clone();
        while input.ndim() < 5 {
            input.insert_axis_inplace(Axis(0));
            mask.insert_axis_inplace(Axis(0));
        }

        let scaled_input = input.mapv(|x| x * self.scale);
        let masked_input = scaled_input + mask;

        let mut output = Self::softmax(&masked_input.into_dimensionality()?, 4)?.into_dyn();
        while output.ndim() > self.input.ndim() {
            output.index_axis_inplace(Axis(0), 0);
        }
        Ok(output)
    }
}

impl<F> Test for ScaledMaskedSoftmaxProblem<F>
where
    F: Datum + Float,
    f32: From<F>,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        ensure!(!self.scale.is_zero());
        let reference = self.reference()?.into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        //dbg!(&self.input, &self.mask);
        let mut output = runtime
            .prepare(model)?
            .run(tvec![self.input.clone().into_tvalue(), self.mask.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        // dbg!(&reference, &output);
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
            input: tensor4(&[[[[0f32]]]]).into_array()?.into_dimensionality()?,
            mask: tensor4(&[[[[0f32]]]]).into_array()?.into_dimensionality()?,
            scale: 1f32,
        },
    );

    suite.add(
        "trivial_f32_1",
        ScaledMaskedSoftmaxProblem {
            input: tensor4(&[[[[0f32, 0f32], [0f32, 0f32]]]])
                .into_array()?
                .into_dimensionality()?,
            mask: tensor4(&[[[[f32::NEG_INFINITY, 0f32], [0f32, 0f32]]]])
                .into_array()?
                .into_dimensionality()?,
            scale: 1f32,
        },
    );

    suite.add(
        "trivial_f32_2",
        ScaledMaskedSoftmaxProblem {
            input: arr4(&[[[[0f32, 0f32]]]]).into_dimensionality()?,
            mask: arr4(&[[[[f32::NEG_INFINITY, 0f32]]]]).into_dimensionality()?,
            scale: 1f32,
        },
    );

    suite.add(
        "trivial_f32_3",
        ScaledMaskedSoftmaxProblem {
            input: tensor4(&[[[[0f32, 0f32]]]]).into_array()?.into_dimensionality()?,
            mask: tensor4(&[[[[0f32, 0f32]]]]).into_array()?.into_dimensionality()?,
            scale: 1f32,
        },
    );
    Ok(suite)
}
