//! Per-dispatch GPU timing, from the device's own timestamp counters.
//!
//! A turn's dispatches share one compute pass, and an Apple GPU samples
//! counters at a pass boundary rather than a dispatch one, so a profiled turn
//! gives each dispatch a pass of its own and reads the two timestamps that
//! pass is bracketed by. That serialises the turn, which is what a profile is
//! for.

use metal::{
    CounterSampleBuffer, CounterSampleBufferDescriptor, CounterSampleBufferRef, CounterSet, Device,
    MTLCounterSamplingPoint, MTLStorageMode, NSRange, NSUInteger,
};
use std::cell::{Cell, RefCell};
use std::time::Duration;
use tract_core::internal::*;

/// The name Metal gives the counter set carrying GPU timestamps.
const TIMESTAMP_COUNTER_SET: &str = "timestamp";

/// A counter sample buffer holds 32 KB at most and a timestamp sample takes 8,
/// so one buffer brackets this many dispatches, two samples to a dispatch. A
/// turn takes as many buffers as its dispatches need.
const DISPATCHES_PER_BUFFER: usize = 32768 / 8 / 2;

/// Buffers a profiled turn will allocate before it stops recording.
const MAX_BUFFERS: usize = 16;

/// What a profiled command buffer needs of its stream: where the samples go,
/// and which node is being evaluated.
#[derive(Debug)]
pub struct MetalProfileHandle {
    pub profile: RefCell<MetalProfile>,
    pub node_id: Cell<usize>,
}

#[derive(Debug)]
pub struct MetalProfile {
    device: Device,
    counter_set: CounterSet,
    /// One buffer per `DISPATCHES_PER_BUFFER` dispatches a turn reached, grown
    /// as a turn needs them and reused by the turns after it.
    buffers: Vec<CounterSampleBuffer>,
    /// The node each pair of samples was taken for, in the order taken.
    nodes: Vec<usize>,
    /// Where the device's clock and the host's stood when profiling began, the
    /// pair a tick is scaled to seconds by.
    epoch: (u64, u64),
    untimed: usize,
}

impl MetalProfile {
    pub fn new(device: &Device) -> TractResult<MetalProfile> {
        ensure!(
            device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary),
            "This device samples no counter at a pass boundary, so a dispatch cannot be timed"
        );
        let counter_set = device
            .counter_sets()
            .into_iter()
            .find(|set| set.name() == TIMESTAMP_COUNTER_SET)
            .context("This device exposes no timestamp counter set")?;
        Ok(MetalProfile {
            device: device.to_owned(),
            counter_set: counter_set.to_owned(),
            buffers: vec![],
            nodes: vec![],
            epoch: sample_clocks(device),
            untimed: 0,
        })
    }

    fn new_buffer(&self) -> TractResult<CounterSampleBuffer> {
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&self.counter_set);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(2 * DISPATCHES_PER_BUFFER as u64);
        self.device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|e| anyhow!("{e}"))
            .context("Could not allocate a counter sample buffer")
    }

    pub fn buffer(&self, ix: usize) -> &CounterSampleBufferRef {
        &self.buffers[ix]
    }

    /// The buffer the next dispatch's pass samples into and where in it its
    /// start and end go, or `None` once a turn has taken all the buffers it is
    /// allowed.
    pub fn next_slot(&mut self, node_id: usize) -> Option<(usize, NSUInteger, NSUInteger)> {
        let taken = self.nodes.len();
        let buffer_ix = taken / DISPATCHES_PER_BUFFER;
        if buffer_ix >= MAX_BUFFERS {
            self.untimed += 1;
            return None;
        }
        while self.buffers.len() <= buffer_ix {
            match self.new_buffer() {
                Ok(buffer) => self.buffers.push(buffer),
                Err(e) => {
                    log::warn!("A profiled turn outgrew its counter sample buffers: {e}");
                    self.untimed += 1;
                    return None;
                }
            }
        }
        let start = 2 * (taken % DISPATCHES_PER_BUFFER) as NSUInteger;
        self.nodes.push(node_id);
        Some((buffer_ix, start, start + 1))
    }

    /// What each dispatch spent on the device, and which node asked for it.
    /// Sound only once the command buffer carrying them has completed.
    pub fn drain(&mut self, device: &Device) -> TractResult<Vec<(usize, Duration)>> {
        if self.untimed > 0 {
            log::warn!(
                "{} dispatches went untimed past the {} a turn records",
                self.untimed,
                MAX_BUFFERS * DISPATCHES_PER_BUFFER
            );
            self.untimed = 0;
        }
        let nodes = std::mem::take(&mut self.nodes);
        if nodes.is_empty() {
            return Ok(vec![]);
        }
        let seconds_per_tick = self.seconds_per_tick(device);
        let mut timings = Vec::with_capacity(nodes.len());
        for (buffer_ix, buffer) in self.buffers.iter().enumerate() {
            let first = buffer_ix * DISPATCHES_PER_BUFFER;
            if first >= nodes.len() {
                break;
            }
            let held = (nodes.len() - first).min(DISPATCHES_PER_BUFFER);
            let ticks = resolve_timestamps(buffer, 2 * held)?;
            for dispatch in 0..held {
                let elapsed = ticks[2 * dispatch + 1].saturating_sub(ticks[2 * dispatch]);
                timings.push((
                    nodes[first + dispatch],
                    Duration::from_secs_f64(elapsed as f64 * seconds_per_tick),
                ));
            }
        }
        Ok(timings)
    }

    /// The device's clock has its own unit, and the pair of clocks it can be
    /// read against gives the scale: how far each ran between the two readings.
    /// A device whose clock did not move leaves the ticks unscaled.
    fn seconds_per_tick(&self, device: &Device) -> f64 {
        let (cpu, gpu) = sample_clocks(device);
        let device_ticks = gpu.saturating_sub(self.epoch.1);
        let host_nanos = cpu.saturating_sub(self.epoch.0);
        if device_ticks == 0 || host_nanos == 0 {
            return 1e-9;
        }
        host_nanos as f64 * 1e-9 / device_ticks as f64
    }
}

/// The first `wanted` timestamps a sample buffer holds.
///
/// `CounterSampleBufferRef::resolve_counter_range` asks for the bytes of a
/// freshly reserved empty `Vec`, which is none of them, and then claims the
/// range was filled -- so it hands back whatever the allocation held. This
/// sends `resolveCounterRange:` itself and copies the length it actually
/// wants. A timestamp resolves to one `u64` a sample.
// The objc msg_send!/sel! macros expand to a cargo-clippy cfg check that older
// toolchains report at the call site; the allow covers it.
#[allow(unexpected_cfgs)]
fn resolve_timestamps(buffer: &CounterSampleBufferRef, wanted: usize) -> TractResult<Vec<u64>> {
    use metal::foreign_types::ForeignTypeRef;
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};

    let mut timestamps = vec![0u64; wanted];
    let bytes = std::mem::size_of_val(timestamps.as_slice()) as u64;
    unsafe {
        let buffer: *mut Object = buffer.as_ptr() as *mut Object;
        let resolved: *mut Object =
            msg_send![buffer, resolveCounterRange: NSRange::new(0, wanted as u64)];
        ensure!(!resolved.is_null(), "The device resolved none of {wanted} timestamps sampled");
        let () = msg_send![resolved, getBytes: timestamps.as_mut_ptr() length: bytes];
    }
    Ok(timestamps)
}

fn sample_clocks(device: &Device) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0, 0);
    device.sample_timestamps(&mut cpu, &mut gpu);
    (cpu, gpu)
}
