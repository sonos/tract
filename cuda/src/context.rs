use cust::prelude::*;

use tract_gpu::device::DeviceContext;
use tract_gpu::tensor::OwnedDeviceTensor;

use std::cell::RefCell;

use std::ops::Deref;
use std::sync::OnceLock;

use tract_core::internal::*;

use crate::tensor::CudaTensor;

thread_local! {
    pub static CUDA_STREAM: RefCell<CudaStream> = RefCell::new(CudaStream::new());
}

pub fn cuda_context() -> CudaContext {
    static INSTANCE: OnceLock<CudaContext> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let ctxt = CudaContext::new().expect("Could not create Metal context");
            tract_gpu::device::set_context(Box::new(ctxt.clone()))
                .expect("Could not set Metal context");
            ctxt
        })
        .clone()
}

#[derive(Debug, Clone)]
pub struct CudaContext {
    inner: Context,
}

impl CudaContext {
    pub fn new() -> TractResult<Self> {
        let context =
            cust::quick_init().with_context(|| "Could not find system default Metal device")?;

        let ctxt = Self { inner: context };
        Ok(ctxt)
    }
}

impl Deref for CudaContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DeviceContext for CudaContext {
    fn synchronize(&self) -> TractResult<()> {
        CUDA_STREAM.with_borrow(|stream| stream.wait_until_completed())
    }

    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        let data = tensor.as_bytes();
        static ZERO: [u8; 1] = [0];
        // Handle empty data
        let data = if data.is_empty() { &ZERO } else { data };

        Ok(Box::new(CudaTensor::from_bytes(
            data,
            tensor.datum_type(),
            tensor.shape(),
            tensor.strides(),
        )))
    }
}

#[derive(Debug)]
pub struct CudaStream {
    pub stream: Stream,
}

impl Default for CudaStream {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaStream {
    pub fn new() -> Self {
        Self { stream: Stream::new(StreamFlags::NON_BLOCKING, None).unwrap() }
    }

    pub fn wait_until_completed(&self) -> TractResult<()> {
        self.stream.synchronize().map_err(|e| e.into())
    }
}
