use metal::{CommandBuffer, ComputeCommandEncoder, ComputeCommandEncoderRef};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct TCommandBuffer {
    inner: CommandBuffer,
    encoder: ComputeCommandEncoder,
}

impl TCommandBuffer {
    pub fn new(command_buffer: CommandBuffer) -> Self {
        let encoder = command_buffer.new_compute_command_encoder().to_owned();

        TCommandBuffer { inner: command_buffer, encoder }
    }

    pub fn encoder(&self) -> &ComputeCommandEncoder {
        &self.encoder
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        encode_cb(&self.encoder);
    }
}

impl Deref for TCommandBuffer {
    type Target = CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TCommandBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
