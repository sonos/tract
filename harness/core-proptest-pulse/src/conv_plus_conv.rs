use proptest::proptest;
use proptest::test_runner::TestCaseResult;
use tract_core::dimfact;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::shapefactoid;
use tract_core::infer::*;

use super::*;

#[derive(Debug, Clone)]
struct ConvOp {
    stride: usize,
    dilation: usize,
    ker: Array3<f32>,
}
impl ConvOp {
    fn chain(&self, name: &str, model: &mut InferenceModel, after: OutletId) -> OutletId {
        let filters = model.add_const(format!("{}-kernel", name), self.ker.clone()).unwrap();
        let mut conv = tract_core::ops::cnn::Conv::default();
        conv.dilations = Some(tvec!(self.dilation));
        conv.strides = Some(tvec!(self.stride));
        model.wire_node(name, conv, &[after, filters]).unwrap()[0]
    }
}

impl Arbitrary for ConvOp {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (1usize..3, 1usize..3, vec(1usize..3))
            .prop_map(|(stride, dilation, ker)| ConvOp {
                stride,
                dilation,
                ker: Array3::from_shape_vec((1, 1, ker.len()), ker).unwrap(),
            })
            .boxed()
    }
}

#[derive(Debug, Clone)]
struct ConvPlusConvProblem {
    input: Array3<f32>,
    pulse: usize,
    conv1: ConvOp,
    conv2: ConvOp,
}

impl Arbitrary for ConvPlusConvProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> BoxedStrategy<Self> {
        (ConvOp::arbitrary(), ConvOp::arbitrary(), 1usize..3)
            .prop_flat_map(|(conv1, conv2, pulse_factor)| {
                let pulse = conv1.stride * conv2.stride * pulse_factor;
                let min_input = 10usize;
                (Just(conv1), Just(conv2), Just(pulse), vec(min_input..3 * min_input))
            })
            .prop_map(|(conv1, conv2, pulse, input)| {
                let input = Array3::from_shape_vec((1, 1, input.len()), input).unwrap(); // NCHW
                ConvPlusConvProblem { input, pulse, conv1, conv2 }
            })
            .boxed()
    }
}

impl ConvPlusConvProblem {
    pub fn run(&self) -> TestCaseResult {
        let mut model = InferenceModel::default();
        let input = model
            .add_source("a", InferenceFact::dt_shape(f32::datum_type(), shapefactoid!(1, 1, S)))
            .unwrap();
        let id = self.conv1.chain("conv1", &mut model, input);
        let _id = self.conv2.chain("conv2", &mut model, id);
        model.auto_outputs().unwrap();
        proptest_regular_against_pulse(model, self.pulse as _, self.input.clone().into_dyn(), 2)
    }
}

proptest! {
    #[test]
    fn proptest(pb in ConvPlusConvProblem::arbitrary()) { pb.run().unwrap() }
}

#[test]
fn prob_1() {
    let cpc = ConvPlusConvProblem {
        input: Array3::from_shape_fn((1, 1, 7), |(_, _, x)| x as f32),
        pulse: 1,
        conv1: ConvOp { stride: 1, dilation: 1, ker: arr3(&[[[1f32]]]) },
        conv2: ConvOp { stride: 1, dilation: 2, ker: arr3(&[[[1f32, 2.0]]]) },
    };
    cpc.run().unwrap();
}
