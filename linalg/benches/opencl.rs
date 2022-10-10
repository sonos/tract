use std::ptr::null_mut;

use criterion::*;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    error_codes::ClError,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::{cl_float, CL_NON_BLOCKING},
};

fn context() -> Context {
    let platforms = get_platforms().unwrap();
    let device = platforms[0].get_devices(CL_DEVICE_TYPE_GPU).unwrap().remove(0);
    let device = Device::new(device);
    Context::from_device(&device).expect("Context::from_device failed")
}

fn queue(context: &Context) -> CommandQueue {
    CommandQueue::create_with_properties(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
        0,
    )
    .expect("CommandQueue::create failed")
}

fn b(c: &mut Criterion, ctx: &Context, name: &str, mut steps: impl FnMut(&mut CommandQueue)) {
    let mut queue = queue(ctx);
    c.bench_function(name, |b| {
        b.iter(|| {
            steps(&mut queue);
            queue.flush();
        })
    });
}

const v: [f32; 1024*1024] = [0f32; 1024 * 1024];

fn write_buffer(c: &mut Criterion, ctx: &Context, name: &str, size: usize) {
    let mut cl = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, size, null_mut()).unwrap();
    b(c, &ctx, name, |q| {
        q.enqueue_write_buffer(&mut cl, CL_NON_BLOCKING, 0, &v[0..size], &[]).unwrap();
    });
}

fn all(c: &mut Criterion) {
    let ctx = context();
    b(c, &ctx, "empty", |_| ());
    write_buffer(c, &ctx, "write_1kf32_buffer", 1024);
    write_buffer(c, &ctx, "write_1Mf32_buffer", 1024 * 1024);
}

criterion_group!(benches, all);
criterion_main!(benches);
