use std::ptr::null_mut;

use criterion::*;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
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

fn empty(c: &mut Criterion) {
    c.bench_function("empty", |b| {
        let ctx = context();
        let queue = queue(&ctx);
        b.iter(|| {
            queue.finish().unwrap();
        })
    });
}

fn write_buffer(c: &mut Criterion) {
    let mut g = c.benchmark_group("write_buffer");
    for size in &[64, 256, 1024, 8 * 1024, 32 * 1024, 128 * 1024, 1024 * 1024] {
        g.bench_with_input(BenchmarkId::new("write_buf", size.to_string()), size, |b, s| {
            let ctx = context();
            let q = queue(&ctx);
            let v = vec![0f32; *s];
            let mut cl =
                Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, v.len(), null_mut()).unwrap();
            b.iter(move || {
                q.enqueue_write_buffer(&mut cl, CL_NON_BLOCKING, 0, &v, &[]).unwrap();
                q.finish().unwrap();
            });
        })
        .throughput(Throughput::Elements(*size as _));
    }
}

static GEMV1: &'static str =
    "__kernel void gemv1(__global const float * a,__global const float * x,
                    __global float * y,int m,int n) {
          float sum = 0.0f;
          int i = get_global_id(0); // row index
          for (int k=0;k<n;k++)
            {
              sum += a[i + m*k] * x[k];
            }
          y[i] = sum;
        }";

fn gemv1(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemv1");
    for loc in [1, 2, 4, 8, 16] {
        for m in [16, 32, 64, 128, 256, 1024] {
            for n in [16, 32, 64, 128, 256, 1024] {
                g.bench_with_input(
                    BenchmarkId::new("square", format!("{}x{}by{}", m, n, loc)),
                    &(m,n),
                    |b, &(m, n)| {
                        let ctx = context();
                        let a = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, m * n, null_mut())
                            .unwrap();
                        let x =
                            Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, n, null_mut()).unwrap();
                        let y =
                            Buffer::<cl_float>::create(&ctx, CL_MEM_READ_WRITE, m, null_mut()).unwrap();

                        let queue = queue(&ctx);
                        let program = Program::create_and_build_from_source(&ctx, GEMV1, "").unwrap();
                        let kernel = Kernel::create(&program, "gemv1").expect("Kernel::create failed");
                        b.iter(|| {
                            let mut run = ExecuteKernel::new(&kernel);
                            run.set_arg(&a)
                                .set_arg(&x)
                                .set_arg(&y)
                                .set_arg(&(m as i32))
                                .set_arg(&(n as i32))
                                .set_global_work_sizes(&[m])
                                .set_local_work_sizes(&[loc])
                                .enqueue_nd_range(&queue)
                                .unwrap();
                            queue.finish().unwrap();
                        });
                    },
                )
                .throughput(Throughput::Elements((m * m) as _));
            }
        }
    }
}

criterion_group!(benches, empty, gemv1, write_buffer);
criterion_main!(benches);
