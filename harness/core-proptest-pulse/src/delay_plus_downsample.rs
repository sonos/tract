use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_core::ops::Downsample;
use tract_core::tract_data::itertools::Itertools;

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
    arr1(&(0..n).map(|x| x as f32).collect_vec()).into_shape_with_order(vec![1, n, 1]).unwrap()
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
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, 1))).unwrap();
        let crop =
//            model.wire_node("delay", expand(array::Crop::new(1, self.delay, 0)), &[a]).unwrap();
            model.wire_node("delay", Slice::new(1, self.delay, s), &[a]).unwrap();
        let ds = model
            .wire_node(
                "ds",
                Downsample { axis: 1, stride: self.stride as isize, modulo: self.modulo },
                &crop,
            )
            .unwrap();
        model.set_output_outlets(&ds).unwrap();
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

#[test]
fn test_from_convs() {
    DelayPlusDownsampleProblem { input: 5, pulse: 2, delay: 1, stride: 2, modulo: 0 }
        .run()
        .unwrap();
}

#[test]
fn test_delayed_stride() {
    DelayPlusDownsampleProblem { input: 9, pulse: 2, delay: 1, stride: 2, modulo: 0 }.run().unwrap()
}

#[test]
fn test_big_delay() {
    DelayPlusDownsampleProblem { input: 6, pulse: 1, delay: 4, stride: 1, modulo: 0 }.run().unwrap()
}

#[test]
fn test_huge_delay() {
    DelayPlusDownsampleProblem { input: 4, pulse: 2, delay: 1, stride: 2, modulo: 0 }.run().unwrap()
}
