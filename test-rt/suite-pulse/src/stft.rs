use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ops::fft::Stft;
use tract_core::runtime::Runtime;

use crate::pulse_and_compare;

/// STFT with the streaming axis distinct from the STFT axis must be
/// pulsifiable: every non-STFT axis is a 1-to-1 passthrough once
/// `Stft::axes_mapping` declares the relationship. Without the mapping the
/// pulse pass bails with "could not track pulsing axis" as soon as a batched
/// STFT pipeline streams its batch axis.
///
/// Input is (B_stream, T, 2): axis 0 streams, the STFT runs on T, and the
/// trailing 2 holds (re, im). One pulse carries `pulse` batch elements, each
/// getting a full-length STFT.
#[derive(Clone, Debug)]
pub struct StftProblem {
    pub batch_len: usize,
    pub pulse: usize,
    pub time_len: usize,
    pub frame: usize,
    pub stride: usize,
}

impl Arbitrary for StftProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (
            1usize..6,
            1usize..3,
            4usize..16,
            proptest::sample::select(vec![2usize, 4]),
            proptest::sample::select(vec![1usize, 2]),
        )
            .prop_filter("a frame longer than the signal yields no frame", |(_, _, t, f, _)| t >= f)
            .prop_map(|(batch_len, pulse, time_len, frame, stride)| StftProblem {
                batch_len,
                pulse,
                time_len,
                frame,
                stride,
            })
            .boxed()
    }
}

impl Test for StftProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(s, self.time_len, 2)))?;
        model.wire_node(
            "stft",
            Stft { axis: 1, frame: self.frame, stride: self.stride, window: None },
            &[a],
        )?;
        model.auto_outputs()?;

        let time_len = self.time_len;
        let input: ArrayD<f32> = ArrayD::from_shape_fn(vec![self.batch_len, time_len, 2], |idx| {
            (idx[0] * time_len * 2 + idx[1] * 2 + idx[2]) as f32 * 0.01
        });
        pulse_and_compare(runtime, approx, model, self.pulse, input, 0)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<StftProblem>("proptest", ());
    let case = |batch_len, pulse, time_len, frame, stride| StftProblem {
        batch_len,
        pulse,
        time_len,
        frame,
        stride,
    };
    suite.add_test("batch_axis_4_pulse2_t8_frame4_stride2", case(4, 2, 8, 4, 2));
    suite.add_test("batch_axis_3_pulse1_t6_frame3_stride1", case(3, 1, 6, 3, 1));
    suite.add_test("batch_axis_2_pulse2_t12_frame4_stride4", case(2, 2, 12, 4, 4));
    Ok(suite)
}
