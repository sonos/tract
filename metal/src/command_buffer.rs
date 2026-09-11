use crate::profile::MetalProfileHandle;
use metal::{
    CommandBuffer, ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePassDescriptor,
};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

/// A command buffer and the compute pass its dispatches share. A profiled
/// buffer gives each dispatch a pass of its own instead, so the device's
/// timestamp counters -- which it samples at a pass boundary -- bracket one
/// dispatch each.
#[derive(Debug, Clone)]
pub struct TCommandBuffer {
    inner: CommandBuffer,
    encoder: Rc<RefCell<Option<ComputeCommandEncoder>>>,
    profile: Option<Rc<MetalProfileHandle>>,
}

impl TCommandBuffer {
    pub fn new(command_buffer: CommandBuffer, profile: Option<Rc<MetalProfileHandle>>) -> Self {
        TCommandBuffer { inner: command_buffer, encoder: Rc::new(RefCell::new(None)), profile }
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        if let Some(handle) = self.profile.as_ref() {
            let slot = handle.profile.borrow_mut().next_slot(handle.node_id.get());
            if let Some((buffer_ix, start, end)) = slot {
                self.end_encoding();
                let descriptor = ComputePassDescriptor::new();
                let attachment = descriptor
                    .sample_buffer_attachments()
                    .object_at(0)
                    .expect("A compute pass takes a sample buffer attachment at 0");
                {
                    let profile = handle.profile.borrow();
                    attachment.set_sample_buffer(profile.buffer(buffer_ix));
                }
                attachment.set_start_of_encoder_sample_index(start);
                attachment.set_end_of_encoder_sample_index(end);
                let encoder = self.inner.compute_command_encoder_with_descriptor(descriptor);
                encode_cb(encoder);
                encoder.end_encoding();
                return;
            }
        }
        encode_cb(&self.shared_encoder());
    }

    /// Close the pass the dispatches share, if one is open. A command buffer
    /// holds one pass at a time, so this runs before a profiled dispatch opens
    /// its own and before the buffer is committed.
    pub fn end_encoding(&self) {
        if let Some(encoder) = self.encoder.borrow_mut().take() {
            encoder.end_encoding();
        }
    }

    fn shared_encoder(&self) -> ComputeCommandEncoder {
        self.encoder
            .borrow_mut()
            .get_or_insert_with(|| self.inner.new_compute_command_encoder().to_owned())
            .to_owned()
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
