use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_ndarray::*;

pub mod conv_f32;
pub mod conv_q;
pub mod deconv;
pub mod downsample;
pub mod q_binary;
pub mod q_elmwise;
pub mod q_flavours;
pub mod q_helpers;
pub mod slice;

pub fn suite() -> TractResult<TestSuite> {
    let mut suite: TestSuite = Default::default();
    suite.add("conv_f32", conv_f32::suite()?);
    suite.add("conv_q", conv_q::suite()?);
    suite.add("deconv", deconv::suite()?);
    suite.add("downsample", downsample::suite()?);
    suite.add("q_flavours", q_flavours::suite()?);
    suite.add("slice", slice::suite()?);
    suite.add("q_binary", q_binary::suite()?);
    suite.add("q_elmwise", q_elmwise::suite()?);
    Ok(suite)
}

pub fn tensor<'a>(shape: impl IntoIterator<Item = &'a usize>) -> BoxedStrategy<ArrayD<f32>> {
    let shape = shape.into_iter().copied().collect::<Vec<_>>();
    let len = shape.iter().product::<usize>();
    vec((-10i8..=10i8).prop_map(|i| i as f32), len..=len)
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
