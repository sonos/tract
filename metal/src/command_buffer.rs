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
        // Both Metal encoder constructors return autoreleased objects.
        objc::rc::autoreleasepool(|| self.encode_in_pool(encode_cb));
    }

    fn encode_in_pool(&self, encode_cb: impl Fn(&ComputeCommandEncoderRef)) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MetalStream;
    use metal::foreign_types::ForeignTypeRef;
    use objc::rc::{WeakPtr, autoreleasepool};

    fn check_command_objects_released(profile: bool) -> tract_core::internal::TractResult<()> {
        // An outer pool makes a missing inner drain observable without leaking
        // test objects into the worker thread's lifetime.
        autoreleasepool(|| {
            let stream = MetalStream::new();
            if profile {
                let device = metal::Device::system_default().expect("Metal device");
                if !device
                    .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
                {
                    eprintln!(
                        "Skipping profiled lifetime test: no stage-boundary counter sampling"
                    );
                    return Ok(());
                }
                stream.enable_profiling()?;
            }
            for _ in 0..4 {
                let command = stream.command_buffer();
                let weak_command = unsafe { WeakPtr::new(command.as_ptr().cast()) };
                let encoders = RefCell::new(Vec::new());
                for _ in 0..2 {
                    command.encode(|encoder| {
                        encoders
                            .borrow_mut()
                            .push(unsafe { WeakPtr::new(encoder.as_ptr().cast()) });
                    });
                }
                // The owned command must survive its construction pool.
                assert!(!weak_command.load().is_null());
                stream.wait_until_completed()?;
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                drop(command);
                // GPU completion can precede the driver's final reference release.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while !weak_command.load().is_null() && std::time::Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(weak_command.load().is_null(), "completed command retained by outer pool");
                for encoder in encoders.into_inner() {
                    assert!(encoder.load().is_null(), "completed encoder retained by outer pool");
                }
            }
            Ok(())
        })
    }

    #[test]
    fn command_objects_released_without_caller_pool_drain() -> tract_core::internal::TractResult<()>
    {
        check_command_objects_released(false)
    }

    #[test]
    fn profiled_command_objects_released_without_caller_pool_drain()
    -> tract_core::internal::TractResult<()> {
        check_command_objects_released(true)
    }
}
