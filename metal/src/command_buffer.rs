use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePassDescriptor, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, MTLResourceOptions, NSRange,
};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

const NUM_SAMPLES: u64 = 2;
const DEFAULT_BUFFER_PER_NODE: u64 = 2;

#[derive(Debug, Clone)]
pub struct ProfileBuffers {
    device: Device,
    buffers: Vec<Buffer>,
    used: usize,
}

impl ProfileBuffers {
    pub fn new(device: Device) -> Self {
        let buffers = (0..DEFAULT_BUFFER_PER_NODE)
            .map(|_| {
                device.new_buffer(
                    (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        ProfileBuffers { device, buffers, used: 0 }
    }

    pub fn get_buffer(&mut self) -> &Buffer {
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

    pub fn buffers(&self) -> &Vec<Buffer> {
        &self.buffers
    }
}

#[derive(Debug, Clone)]
pub struct MetalProfiler {
    profile_buffers: Vec<ProfileBuffers>,
    current_node_id: Option<usize>,
    counter_sample_buffer: CounterSampleBuffer,
    compute_pass_descriptor: ComputePassDescriptor,
}

impl MetalProfiler {
    pub fn new(device: Device, num_nodes: usize) -> Self {
        let compute_pass_descriptor = ComputePassDescriptor::new();

        let counter_sample_buffer_desc = CounterSampleBufferDescriptor::new();
        counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
        counter_sample_buffer_desc.set_sample_count(NUM_SAMPLES);
        let counter_sets = device.counter_sets();
        let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

        counter_sample_buffer_desc
            .set_counter_set(timestamp_counter.expect("No timestamp counter found"));

        let counter_sample_buffer =
            device.new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc).unwrap();

        let sample_buffer_attachment_descriptor =
            compute_pass_descriptor.sample_buffer_attachments().object_at(0).unwrap();

        sample_buffer_attachment_descriptor.set_sample_buffer(&counter_sample_buffer);
        sample_buffer_attachment_descriptor.set_start_of_encoder_sample_index(0);
        sample_buffer_attachment_descriptor.set_end_of_encoder_sample_index(1);

        let profile_buffers =
            (0..num_nodes).map(|_| ProfileBuffers::new(device.to_owned())).collect();

        Self {
            profile_buffers,
            current_node_id: None,
            counter_sample_buffer,
            compute_pass_descriptor: compute_pass_descriptor.to_owned(),
        }
    }

    pub fn add_node_entry(&mut self, node_id: usize) {
        self.current_node_id = Some(node_id);
    }

    pub fn get_buffer(&mut self) -> &Buffer {
        let current_node_id = self.current_node_id.expect(
            "Metal profile doesn't have any current node id to attach a sampling buffer to",
        );
        self.profile_buffers[current_node_id].get_buffer()
    }

    pub fn get_profile_data(&self) -> Vec<u64> {
        let mut res = vec![0; self.profile_buffers.len()];

        self.profile_buffers.iter().enumerate().for_each(|(key, v)| {
            let mut node_duration_ns = 0;
            v.buffers.iter().for_each(|buffer| unsafe {
                let slice = std::slice::from_raw_parts(
                    buffer.contents() as *const u64,
                    NUM_SAMPLES as usize,
                );
                node_duration_ns += slice[1] - slice[0];
            });
            res[key] = node_duration_ns;
        });
        res
    }
}

#[derive(Debug, Clone)]
pub struct TCommandBuffer {
    inner: CommandBuffer,
    profiler: Option<Rc<RefCell<MetalProfiler>>>,
}

impl TCommandBuffer {
    pub fn new(
        command_buffer: CommandBuffer,
        profiler: Option<Rc<RefCell<MetalProfiler>>>,
    ) -> Self {
        TCommandBuffer { inner: command_buffer, profiler }
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        if let Some(profiler) = &self.profiler {
            let mut profiler = profiler.borrow_mut();

            let encoder = self
                .inner
                .compute_command_encoder_with_descriptor(&profiler.compute_pass_descriptor);

            encode_cb(encoder);

            let blit_encoder = self.inner.new_blit_command_encoder();
            blit_encoder.resolve_counters(
                &profiler.counter_sample_buffer.clone(),
                NSRange::new(0_u64, NUM_SAMPLES),
                profiler.get_buffer(),
                0_u64,
            );
            blit_encoder.end_encoding();
        } else {
            let encoder = self.inner.new_compute_command_encoder();
            encode_cb(encoder);
        };
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
