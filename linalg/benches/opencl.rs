use std::{io::Write, ptr::null_mut};
mod nano;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
    types::cl_float,
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

fn empty(q: &CommandQueue) {
    q.finish().unwrap();
}

fn kernel(c: &Context, q: &CommandQueue, g: usize, l: usize, body: &str) -> f64 {
    let code =
        "__kernel void ker(__global const float *a, __global float *b) {".to_string() + body + "}";
    let program = Program::create_and_build_from_source(c, &code, "").unwrap();
    let a = Buffer::<cl_float>::create(&c, CL_MEM_READ_ONLY, 8 * 1024 * 1024, null_mut()).unwrap();
    let b = Buffer::<cl_float>::create(&c, CL_MEM_READ_WRITE, 8 * 1024 * 1024, null_mut()).unwrap();
    let kernel = Kernel::create(&program, "ker").expect("Kernel::create failed");
    b1!({
        let mut task = ExecuteKernel::new(&kernel);
        task.set_arg(&a).set_arg(&b).set_global_work_sizes(&[g]);
        if l > 0 {
            task.set_local_work_sizes(&[l]);
        }
        let event = task.enqueue_nd_range(q).unwrap();
        event.wait().unwrap();
    })
}

fn main() {
    let c = context();
    let q = queue(&c);
    let empty = b32!(empty(&q));
    println!("empty: {:10.3} µs", empty * 1e6);
    let gs = [16, 128, 512, 2048];
    let ls = [0, 1, 2, 4, 8, 16];
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let kernels = vec![
        ("noop", ""),
        /* 
         * these ones are optimised away
         *
        ("global", "int row = get_global_id(0);"),
        ("load1", "float x = a[0];"),
        ("ld@glob", "float x = a[get_global_id(0)];"),
        ("ld1k", "for (int i = 0; i < 1024; i++) { float x = a[i]; }"),
        ("ld10k", "for (int i = 0; i < 1024*1024; i++) { float x = a[i]; }"),
        */
        ("cp1k", "for (int i = 0; i < 1024; i++) { b[i] = a[i]; }"),
        ("cp1k@", "for (int i = 0; i < 1024; i++) { b[i * 1024 + get_global_id(0)] = a[i * 1024 + get_global_id(0)]; }"),
//        ("cp1k@t", "for (int i = 0; i < 1024; i++) { b[i + 1024 * get_global_id(0)] = a[i + 1024 * get_global_id(0)]; }"),
        ("cp1k@v", "for (int i = 0; i < 128; i++) { vstore4(vload4(0, &a[i * 4 * 128 + get_global_id(0)]), 0, &b[i * 4 * 128 + get_global_id(0)]); }"),
        // these ones are reading the same data again and again, hitting cache
        /*
        ("sum1k", "float s = 0; for (int i = 0; i < 1024; i++) { s += a[i]; } b[0] = s;"),
        ("sum1kdb", "float s = 0; for (int i = 0; i < 1024; i++) { float v = a[i]; s += v + v; } b[0] = s;"),
        ("sum1ksq", "float s = 0; for (int i = 0; i < 1024; i++) { float v = a[i]; s += v * v; } b[0] = s;"),

        ("pdt1k", "float s = 0; for (int i = 0; i < 1024; i++) { s *= a[i]; } b[0] = s;"),
        ("pdt1k20", "float s = 0; for (int i = 0; i < 1024; i++) { for(int j = 0; j < 20; j++ ) { s *= a[i]; } } b[0] = s;"),
        ("sum1k20", "float s = 0; for (int i = 0; i < 1024; i++) { for(int j = 0; j < 20; j++ ) { s += a[i]; } } b[0] = s;"),
        */
        // ("sum1k@", "float s = 0; int offset = 1024 * get_global_id(0); for (int i = 0; i < 1024; i++) { s += a[offset + i]; } b[offset] = s;"),
        // should we worry about the optimiser and in-loop compute ? no.
        ("sum128", "float s = 0; int row = get_global_id(0); for (int i = 0; i < 128; i++) { s += a[i * 128 + row]; } b[row] = s;"),
        /*
        ("sum128@1", "float s = 0; int offset = get_global_id(0) * 128; a += offset; for (int i = 0; i < 128; i++) { s += a[i]; } b[offset] = s;"),
        ("sum128@2", "float s = 0; int offset = get_global_id(0) * 128; for (int i = 0; i < 128; i++) { s += a[offset + i]; } b[offset] = s;"),
        ("sum128@3", "float s = 0; for (int i = 0; i < 128; i++) { s += a[128 * get_global_id(0) + i]; } b[get_global_id(0)] = s;"),
        */
        // what about row/col major storage ?
        ("sum128t", "float s = 0; int row = get_global_id(0); for (int i = 0; i < 128; i++) { s += a[i + 128 * row]; } b[row] = s;"),
//        ("cp1M", "for (int i = 0; i < 1024 * 1024; i++) { b[i] = a[i]; }"),
        ("sum128v", "float4 s = (float4)(0.0f, 0.0f, 0.0f, 0.0f); int row = get_global_id(0);
                        for (int i = 0; i < 32; i++) { s = s + vload4(0, &a[i * 4 * 128 + row]); } b[row] = s.x + s.y + s.z + s.w;"),
    ];
    print!("{:8} {:8}   ", "", "");
    for (name, _) in &kernels {
        print!("{:>10}", name);
    }
    println!("");
    for g in gs {
        for l in ls {
            print!("{:8} {:8} : ", g, l);
            for (_name, body) in &kernels {
                print!("{:10.0}", 1e6 * kernel(&c, &q, g, l, body));
                std::io::stdout().flush().unwrap();
            }
            println!("");
        }
    }
}
