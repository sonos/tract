use std::ptr::null_mut;

use criterion::{measurement::Measurement, *};

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

static GEMV1: &'static str = "__kernel void gemv1(__global const float *a,__global const float *x,
                    __global float *y, int m, int n) {
          float sum = 0.0f;
          int row = get_global_id(0);
          for (int k=0 ; k<n ; k++) {
              sum += a[row*n + k] * x[k];
          }
          y[row] = sum;
    }";

fn profile_gemv1() {
    // let (m, n, iters) = (1024, 32, 10000);
    // let (m, n, iters) = (1024, 1024, 10000);
    let (m, n, l, iters) = (16, 16, 2, 10000);
    let ctx = context();
    let a = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, m * n, null_mut()).unwrap();
    let x = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, n, null_mut()).unwrap();
    let y = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_WRITE, m, null_mut()).unwrap();

    let queue = queue(&ctx);
    let program = Program::create_and_build_from_source(&ctx, GEMV1, "").unwrap();
    let kernel = Kernel::create(&program, "gemv1").expect("Kernel::create failed");
    let mut ns = 0;
    for i in 0..iters {
        let mut run = ExecuteKernel::new(&kernel);
        let event = run
            .set_arg(&a)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&(m as i32))
            .set_arg(&(n as i32))
            .set_global_work_sizes(&[m])
            .set_local_work_sizes(&[2])
            .enqueue_nd_range(&queue)
            .unwrap();
        event.wait().unwrap();
        ns += event.profiling_command_end().unwrap() - event.profiling_command_start().unwrap();
    }
    let gigaflops = (m * n) as f32 / ns as f32 * iters as f32;
    dbg!(gigaflops);
}

fn bench_gemv1_bench_one(c: &mut Criterion, name: &str, m: usize, n: usize, loc: usize) {
    c.benchmark_group(format!("gemv1-{}-{}x{}by{}", name, m, n, loc))
        .throughput(Throughput::Elements((m * n) as _))
        .bench_function("loop", |b| {
            let ctx = context();
            let a = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, m * n, null_mut()).unwrap();
            let x = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, n, null_mut()).unwrap();
            let y = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_WRITE, m, null_mut()).unwrap();

            let queue = queue(&ctx);
            let program = Program::create_and_build_from_source(&ctx, GEMV1, "").unwrap();
            let kernel = Kernel::create(&program, "gemv1").expect("Kernel::create failed");
            b.iter(|| {
                let mut run = ExecuteKernel::new(&kernel);
                let event = run
                    .set_arg(&a)
                    .set_arg(&x)
                    .set_arg(&y)
                    .set_arg(&(m as i32))
                    .set_arg(&(n as i32))
                    .set_global_work_sizes(&[m])
                    .set_local_work_sizes(&[loc])
                    .enqueue_nd_range(&queue)
                    .unwrap();
                event.wait().unwrap();
            });
        });
}


static GEMV2: &'static str = "
#define ROW_DIM 0
#define COL_DIM 1
__kernel void gemv2(__global const float * a,
                    __global const float * x,
		    __global float * y,
		    __local float * work,
		    int m, int n)
{
  float sum = (float)0;
  for (int k=get_global_id(COL_DIM);k<n;k+=get_global_size(COL_DIM)) {
      sum += a[get_global_id(ROW_DIM)+m*k] * x[k];
  }

  int rows = get_local_size(ROW_DIM); // rows in group
  int cols = get_local_size(COL_DIM); // initial cols in group
  int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
  int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols
  work[ii+rows*jj] = sum;
  barrier(CLK_LOCAL_MEM_FENCE); // sync group

  while ( cols > 1 ) {
      cols >>= 1;
      if (jj < cols) work[ii+rows*jj] += work[ii+rows*(jj+cols)];
      barrier(CLK_LOCAL_MEM_FENCE); // sync group
  }

  if ( jj == 0 ) y[get_global_id(ROW_DIM)] = work[ii];
}";

fn bench_gemv2_bench_one(c: &mut Criterion, name: &str, m: usize, n: usize, loc: usize) {
    let p = 4;
    c.benchmark_group(format!("gemv2-{}-{}x{}by{}", name, m, n, loc))
        .throughput(Throughput::Elements((m * n) as _))
        .bench_function("loop", |b| {
            let ctx = context();
            let a = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, m * n, null_mut()).unwrap();
            let x = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_ONLY, n, null_mut()).unwrap();
            let y = Buffer::<cl_float>::create(&ctx, CL_MEM_READ_WRITE, m, null_mut()).unwrap();

            let queue = queue(&ctx);
            let program = Program::create_and_build_from_source(&ctx, GEMV2, "").unwrap();
            let kernel = Kernel::create(&program, "gemv2").expect("Kernel::create failed");
            b.iter(|| {
                let mut run = ExecuteKernel::new(&kernel);
                let event = run
                    .set_arg(&a)
                    .set_arg(&x)
                    .set_arg(&y)
                    .set_arg_local_buffer(loc * p)
                    .set_arg(&(m as i32))
                    .set_arg(&(n as i32))
                    .set_global_work_sizes(&[m, p])
                    .set_local_work_sizes(&[loc, p])
                    .enqueue_nd_range(&queue)
                    .unwrap();
                event.wait().unwrap();
            });
        });
}

fn gemv(c: &mut Criterion) {
    for loc in [1, 2, 4, 8, 16] {
        for m in [16, 32, 64, 128, 256, 1024] {
            for n in [16, 32, 64, 128, 256, 1024] {
                bench_gemv1_bench_one(c, "gemv1", m, n, loc);
                bench_gemv2_bench_one(c, "gemv2", m, n, loc);
            }
        }
    }
}

criterion_group!(benches, empty, gemv, write_buffer);
criterion_main!(benches);

/*
fn main() {
profile_gemv1()
}
*/
