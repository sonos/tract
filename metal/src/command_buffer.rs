use anyhow::Result;
use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePassDescriptor,
    ComputePassDescriptorRef, Counter, CounterSampleBuffer, CounterSampleBufferDescriptor,
    CounterSampleBufferRef, Device, MTLResourceOptions, NSRange,
};
use std::{
    borrow::{Borrow, BorrowMut},
    cell::{RefCell, RefMut},
    collections::HashMap,
    hash::Hash,
    ops::{Deref, DerefMut},
    rc::Rc,
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
    buffers: Vec<Rc<RefCell<Buffer>>>,
    used: usize,
}

impl ProfileBuffers {
    pub fn new(device: Device) -> Result<Self> {
        let buffers: Vec<Rc<RefCell<Buffer>>> = vec![Rc::new(RefCell::new(device.new_buffer(
            (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )))];

        Ok(Self { device, buffers, used: 0 })
    }

    pub fn next(&mut self) -> Rc<RefCell<Buffer>> {
        if self.used == self.buffers.len() {
            let new_buffer = Rc::new(RefCell::new(self.device.new_buffer(
                (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
                MTLResourceOptions::StorageModeShared,
            )));
            self.buffers.push(new_buffer.clone());
        }

        let buffer = self.buffers[self.used].clone();
        self.used += 1;
        buffer
    }

    pub fn get_buffer_at_idx(&self, idx: usize) -> Rc<RefCell<Buffer>> {
        self.buffers[idx].clone()
    }
}

#[derive(Debug, Clone)]
pub struct MetalProfiler {
    device: Device,
    sampling_buffers: RefCell<HashMap<usize, Vec<Buffer>>>,
    current_node: RefCell<Option<usize>>,
    counter_sample_buffer: CounterSampleBuffer,
    compute_pass_descriptor: ComputePassDescriptor,
}

impl MetalProfiler {
    pub fn new(device: Device) -> Self {
        let compute_pass_descriptor = ComputePassDescriptor::new();

        let counter_sample_buffer_desc = CounterSampleBufferDescriptor::new();
        counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
        counter_sample_buffer_desc.set_sample_count(NUM_SAMPLES);
        let counter_sets = device.counter_sets();
        let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

        counter_sample_buffer_desc
            .set_counter_set(timestamp_counter.expect("No timestamp counter found"));

        dbg!("Creating a new counter sample buffer");
        let counter_sample_buffer =
            device.new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc).unwrap();

        handle_compute_pass_sample_buffer_attachment(
            compute_pass_descriptor,
            &counter_sample_buffer,
        );

        Self {
            device: device.to_owned(),
            sampling_buffers: RefCell::new(HashMap::new()),
            current_node: RefCell::new(None),
            counter_sample_buffer,
            compute_pass_descriptor: compute_pass_descriptor.to_owned(),
        }
    }

    pub fn add_node_entry(&self, node_id: usize) {
        self.sampling_buffers.borrow_mut().insert(node_id, vec![]);
        self.current_node.replace(Some(node_id));
    }

    pub fn add_buffer(&self, buffer: Buffer) {
        let current_node = self.current_node.borrow().unwrap();

        let mut sample_buffers = self.sampling_buffers.borrow_mut();
        let node_values = sample_buffers.get_mut(&current_node).expect("No buffer");

        node_values.push(buffer);
    }

    pub fn get_buffers(&self) -> HashMap<usize, u64> {
        let mut formatted_hashmap: HashMap<usize, u64> = HashMap::new();
        self.sampling_buffers.borrow().iter().for_each(|(key, v)| {
            let mut node_duration_ms = 0;
            v.iter().for_each(|buffer| {
                unsafe {
                    let slice = std::slice::from_raw_parts(
                        buffer.contents() as *const u64,
                        NUM_SAMPLES as usize,
                    );
                    node_duration_ms += slice[1] - slice[0];
                }
            });
            formatted_hashmap.insert(*key, node_duration_ms);
        });
        formatted_hashmap
    }
}

#[derive(Debug, Clone)]
// Define ProfileCommandBuffer as a wrapper around CommandBuffer
pub struct TractCommandBuffer {
    inner: CommandBuffer,
    profiler: Option<Rc<MetalProfiler>>,
}

impl TractCommandBuffer {
    pub fn new(command_buffer: CommandBuffer, profiler: Option<Rc<MetalProfiler>>) -> Self {
        dbg!("Creating new command buffer");
        TractCommandBuffer { inner: command_buffer, profiler }
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        if let Some(profiler) = self.profiler.borrow() {
            let destination_buffer = profiler.device.new_buffer(
                (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = self
                .inner
                .compute_command_encoder_with_descriptor(&profiler.compute_pass_descriptor);

            encode_cb(encoder);

            let blit_encoder = self.inner.new_blit_command_encoder();
            blit_encoder.resolve_counters(
                &profiler.counter_sample_buffer,
                NSRange::new(0_u64, NUM_SAMPLES),
                &destination_buffer,
                0_u64,
            );
            blit_encoder.end_encoding();

            profiler.add_buffer(destination_buffer);
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
