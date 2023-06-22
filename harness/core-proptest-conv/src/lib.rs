use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_ndarray::*;

pub fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub fn tensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<f32>> {
    let len = shape.iter().product::<usize>();
    vec((-10i8..=10i8).prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn qtensor(shape: Vec<usize>) -> BoxedStrategy<ArrayD<i8>> {
    let len = shape.iter().product::<usize>();
    vec(any::<i8>(), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap())
        .boxed()
}

pub fn shapes(rank: usize) -> BoxedStrategy<(Vec<usize>, Vec<usize>)> {
    vec((1usize..4, 0usize..5).prop_map(|(k, exceed)| (k, k + exceed)), rank..=rank)
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
    prop_oneof![Just(KernelFormat::OIHW), /* Just(KernelFormat::OHWI), */ Just(KernelFormat::HWIO)]
}


