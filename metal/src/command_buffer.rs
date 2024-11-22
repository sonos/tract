use anyhow::Result;
use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePassDescriptor, ComputePassDescriptorRef, CounterSampleBufferDescriptor, CounterSampleBufferRef, Device, MTLResourceOptions, NSRange
};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

const NUM_SAMPLES: u64 = 2;

fn handle_compute_pass_sample_buffer_attachment(
    compute_pass_descriptor: &ComputePassDescriptorRef,
    counter_sample_buffer: &CounterSampleBufferRef,
) {
    let sample_buffer_attachment_descriptor =
        compute_pass_descriptor.sample_buffer_attachments().object_at(0).unwrap();

    sample_buffer_attachment_descriptor.set_sample_buffer(counter_sample_buffer);
    sample_buffer_attachment_descriptor.set_start_of_encoder_sample_index(0);
    sample_buffer_attachment_descriptor.set_end_of_encoder_sample_index(1);
}

#[derive(Debug, Clone)]
pub struct ProfileBuffers {
    device: Device,
    buffers: Vec<Buffer>,
    used: usize,
}

impl ProfileBuffers {
    pub fn new(device: Device) -> Result<Self> {
            let buffers: Vec<Buffer> = (0..5)
            .map(|_| device.new_buffer(
                (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
                MTLResourceOptions::StorageModeShared,
            ))
            .collect();
            
            Ok(Self {
                device,
                buffers,
                used: 0,
            })
    }

    pub fn next(&mut self) -> &Buffer {
        if self.used == self.buffers.len() {
            self.buffers.push(self.device.new_buffer(
                (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }

        let buffer = &self.buffers[self.used];
        self.used += 1;
        buffer
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn get_buffer_at_idx(&self, idx: usize) -> &Buffer {
        &self.buffers[idx]
    }
}

#[derive(Debug, Clone)]
pub struct Profiler {
    device: Device,
    sampling_buffers: Arc<Mutex<ProfileBuffers>>,
}

impl Profiler {
    pub fn new(device: Device, sampling_buffers: Arc<Mutex<ProfileBuffers>>) -> Self {
        Self { device, sampling_buffers }
    }
}

#[derive(Debug, Clone)]
// Define ProfileCommandBuffer as a wrapper around CommandBuffer
pub struct TractCommandBuffer {
    inner: CommandBuffer,
    profiler: Option<Profiler>,
}

impl TractCommandBuffer {
    pub fn new(command_buffer: CommandBuffer) -> Self {
        TractCommandBuffer { inner: command_buffer, profiler: None }
    }

    pub fn attach_profiler(
        &mut self,
        device: Device,
        sampling_buffers: Arc<Mutex<ProfileBuffers>>,
    ) {
        self.profiler = Some(Profiler::new(device, sampling_buffers));
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        if let Some(profiler) = &mut self.profiler {
            let compute_pass_descriptor = ComputePassDescriptor::new();

            let counter_sample_buffer_desc = CounterSampleBufferDescriptor::new();
            counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
            counter_sample_buffer_desc.set_sample_count(NUM_SAMPLES);
            let counter_sets = profiler.device.counter_sets();
            let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

            counter_sample_buffer_desc
                .set_counter_set(timestamp_counter.expect("No timestamp counter found"));
            let counter_sample_buffer = profiler
                .device
                .new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc)
                .unwrap();

            handle_compute_pass_sample_buffer_attachment(
                compute_pass_descriptor,
                &counter_sample_buffer,
            );

            let encoder =
                self.inner.compute_command_encoder_with_descriptor(compute_pass_descriptor);
            encode_cb(encoder);

            let blit_encoder = self.inner.new_blit_command_encoder();
            blit_encoder.resolve_counters(
                &counter_sample_buffer,
                NSRange::new(0_u64, NUM_SAMPLES),
                profiler.sampling_buffers.lock().unwrap().next(),
                0_u64,
            );
            blit_encoder.end_encoding();
        } else {
            let encoder = self.inner.new_compute_command_encoder();
            encode_cb(encoder);
        };
    }
}

impl Deref for TractCommandBuffer {
    type Target = CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TractCommandBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
