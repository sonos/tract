use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::num_traits::Float;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_ndarray::*;

pub mod apply_rope;
pub mod bin_einsum;
pub mod conv_f32;
pub mod conv_q;
pub mod deconv;
pub mod downsample;
pub mod gelu_approximate;
pub mod matmul_q40;
pub mod q_binary;
pub mod q_elmwise;
pub mod q_flavours;
pub mod q_helpers;
pub mod rms_norm;
pub mod scaled_masked_softmax;
pub mod silu;
pub mod slice;

pub fn suite() -> TractResult<TestSuite> {
    let mut suite: TestSuite = Default::default();
    suite.add("bin_einsum", bin_einsum::suite()?);
    suite.add("conv_f32", conv_f32::suite()?);
    suite.add("conv_q", conv_q::suite()?);
    suite.add("deconv", deconv::suite()?);
    suite.add("downsample", downsample::suite()?);
    suite.add("matmul_q40", matmul_q40::suite()?);
    suite.add("q_flavours", q_flavours::suite()?);
    suite.add("rms_norm", rms_norm::suite()?);
    suite.add("apply_rope", apply_rope::suite()?);
    suite.add("gelu_approximate", gelu_approximate::suite()?);
    suite.add("scaled_masked_softmax", scaled_masked_softmax::suite()?);
    suite.add("silu", silu::suite()?);
    suite.add("slice", slice::suite()?);
    suite.add("q_binary", q_binary::suite()?);
    suite.add("q_elmwise", q_elmwise::suite()?);
    Ok(suite)
}

pub fn tensor<'a, F: Datum + Float>(
    shape: impl IntoIterator<Item = &'a usize>,
) -> BoxedStrategy<ArrayD<F>> {
    let shape = shape.into_iter().copied().collect::<Vec<_>>();
    let len = shape.iter().product::<usize>();
    vec((-10i8..=10i8).prop_map(|i| F::from(i).unwrap()), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.to_vec(), vec).unwrap())
        .boxed()
}
pub fn qtensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<i8>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>(), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn shapes(rank: usize) -> BoxedStrategy<(Vec<usize>, Vec<usize>)> {
    vec((1usize..5, 0usize..5).prop_map(|(k, exceed)| (k, k + exceed)), rank..=rank)
        .prop_map(|v| v.into_iter().unzip())
        .boxed()
}

pub fn data_format() -> impl Strategy<Value = DataFormat> {
    prop_oneof![
        Just(DataFormat::CHW),
        Just(DataFormat::HWC),
        Just(DataFormat::NCHW),
        Just(DataFormat::NHWC)
    ]
}

pub fn kernel_format() -> impl Strategy<Value = KernelFormat> {
    prop_oneof![
        Just(KernelFormat::OIHW),
        /* Just(KernelFormat::OHWI), */ Just(KernelFormat::HWIO)
    ]
}
