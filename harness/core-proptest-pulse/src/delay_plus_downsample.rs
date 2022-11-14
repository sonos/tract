use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_hir::internal::*;
use tract_hir::ops::array;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops::Downsample;

use super::*;

#[derive(Debug, Clone)]
struct DelayPlusDownsampleProblem {
    input: usize,
    pulse: usize,
    delay: usize,
    stride: usize,
    modulo: usize,
}

fn t(n: usize) -> ArrayD<f32> {
    arr1(&*(0..n).map(|x| x as f32).collect_vec()).into_shape(vec![1, n, 1]).unwrap()
}

impl Arbitrary for DelayPlusDownsampleProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..100, 1usize..4, 0usize..100, 1usize..4)
            .prop_flat_map(|(input, pulse_mul, delay, stride)| {
                (
                    Just(input + stride + delay),
                    Just(pulse_mul * stride),
                    Just(delay),
                    Just(stride),
                    0..stride,
                )
            })
            .prop_map(|(input, pulse, delay, stride, modulo)| DelayPlusDownsampleProblem {
                input,
                pulse,
                delay,
                stride,
                modulo,
            })
            .boxed()
    }
}

impl DelayPlusDownsampleProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = InferenceModel::default();
        let s = model.symbol_table.sym("S");
        let a = model
            .add_source("a", f32::fact(dims!(1, s, 1)).into())
            .unwrap();
        let crop =
            model.wire_node("delay", expand(array::Crop::new(1, self.delay, 0)), &[a]).unwrap();
        let ds = model
            .wire_node(
                "ds",
                Downsample { axis: 1, stride: self.stride as isize, modulo: self.modulo },
                &crop,
            )
            .unwrap();
        model.set_output_outlets(&ds).unwrap();
        let model = model.into_typed().unwrap();
        dbg!(&model);
        proptest_regular_against_pulse(model, self.pulse as _, t(self.input), 1)
    }
}

proptest! {
    #[test]
    fn proptest(pb in DelayPlusDownsampleProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn test_modulo() {
    DelayPlusDownsampleProblem { input: 3, pulse: 2, delay: 0, stride: 2, modulo: 1 }.run().unwrap()
}

#[test]
fn test_delay() {
    DelayPlusDownsampleProblem { input: 3, pulse: 2, delay: 1, stride: 2, modulo: 0 }.run().unwrap()
}

