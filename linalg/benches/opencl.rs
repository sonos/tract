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
    types::CL_BLOCKING,
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
    let freq = Device::new(c.default_device()).max_clock_frequency().unwrap();
    let empty = b32!(empty(&q));
    let noop = kernel(&c, &q, 1, 1, "");
    println!("empty: {:10.3}µs noop: {:10.3}µs freq:{}MHz", empty * 1e6, noop * 1e6, freq);
    let gs = [16, 128, 512, 2048];
    let ls = [0, 1, 2, 4, 8, 16];
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let mut kernels = vec![
        ("noop", 0, ""),
        /* 
         * these ones are optimised away
         *
        ("global", "int row = get_global_id(0);"),
        ("load1", "float x = a[0];"),
        ("ld@glob", "float x = a[get_global_id(0)];"),
        ("ld1k", "for (int i = 0; i < 1024; i++) { float x = a[i]; }"),
        ("ld10k", "for (int i = 0; i < 1024*1024; i++) { float x = a[i]; }"),
        */
        ("cp1k", 1024, "for (int i = 0; i < 1024; i++) { b[i] = a[i]; }"),
        ("cp1k@", 1024, "for (int i = 0; i < 1024; i++) { b[i * 1024 + get_global_id(0)] = a[i * 1024 + get_global_id(0)]; }"),
//        ("cp1k@t", "for (int i = 0; i < 1024; i++) { b[i + 1024 * get_global_id(0)] = a[i + 1024 * get_global_id(0)]; }"),
        ("cp1k@v", 128 * 4, "for (int i = 0; i < 128; i++) { vstore4(vload4(0, &a[i * 4 * 128 + get_global_id(0)]), 0, &b[i * 4 * 128 + get_global_id(0)]); }"),
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
        ("sum128", 128, "float s = 0; int row = get_global_id(0); for (int i = 0; i < 128; i++) { s += a[i * 128 + row]; } b[row] = s;"),
        /*
        ("sum128@1", "float s = 0; int offset = get_global_id(0) * 128; a += offset; for (int i = 0; i < 128; i++) { s += a[i]; } b[offset] = s;"),
        ("sum128@2", "float s = 0; int offset = get_global_id(0) * 128; for (int i = 0; i < 128; i++) { s += a[offset + i]; } b[offset] = s;"),
        ("sum128@3", "float s = 0; for (int i = 0; i < 128; i++) { s += a[128 * get_global_id(0) + i]; } b[get_global_id(0)] = s;"),
        */
        // what about row/col major storage ?
        ("sum128t", 128, "float s = 0; int row = get_global_id(0); for (int i = 0; i < 128; i++) { s += a[i + 128 * row]; } b[row] = s;"),
//        ("cp1M", "for (int i = 0; i < 1024 * 1024; i++) { b[i] = a[i]; }"),
        ("sum128v", 4* 128, "float4 s = (float4)(0.0f, 0.0f, 0.0f, 0.0f); int row = get_global_id(0);
                        for (int i = 0; i < 32; i++) { s = s + vload4(0, &a[i * 4 * 128 + row]); } b[row] = s.x + s.y + s.z + s.w;"),
        ("sum128", 128, "float s = 0; int row = get_global_id(0); for (int i = 0; i < 128; i++) { s += a[i * 128 + row]; } b[row] = s;"),
        ("add1024", 1024, "float s = 12; for (int i = 0; i < 1024; i++) { s += s; } b[get_global_id(0)] = s;"),
        ("add1024un", 1024, r#"float s = 12;
            #pragma unroll 256
            for (int ii =0; ii< 1024; ii++) {
                s += s;
            }
        b[get_global_id(0)] = s;"#),
        ("mul1024", 1024, "float s = 12; for (int i = 0; i < 1024; i++) { s *= s; } b[get_global_id(0)] = s;"),
        ("mul1024un", 1024, r#"float s = 12;
            #pragma unroll 256
            for (int ii =0; ii< 1024; ii++) {
                s *= s;
            }
        b[get_global_id(0)] = s;"#),
        ("sumsq1024un", 1024 * 2, r#"float s = 12;
            #pragma unroll 256
            for (int ii =0; ii<1024; ii++) {
                s += s*s;
            }
        b[get_global_id(0)] = s;"#),
        ("sumaddsq1024un", 1024 * 3, r#"float s = 12; float t = 42;
            #pragma unroll 256
            for (int ii =0; ii<1024; ii++) {
                s = s + s * s + t;
            }
        b[get_global_id(0)] = s;"#),
        ("sumfma1024un", 1024 * 3, r#"float s = 12;
            #pragma unroll 256
            for (int ii =0; ii<1024; ii++) {
                s = mad(s, s, s);
            }
        b[get_global_id(0)] = s;"#),
        ("sumdot1024un", 256 * 4, r#"
            float4 v = vload4(0, a);
            #pragma unroll 256
            for (int ii =0; ii<256; ii++) {
                v.x = dot(v, v);
                v.x = dot(v, v);
                v.x = dot(v, v);
                v.x = dot(v, v);
            }
            b[get_global_id(0)] = v.x + v.y + v.z + v.w;"#),
    ];
    print!("{:8} {:8}   ", "", "");
    if let Some(filter) = std::env::args().nth(1) {
        kernels.retain(|(n, _, _)| n.contains(&filter));
    }
    for (name, _, _) in &kernels {
        print!("{:>20}", name);
    }
    println!("");
    for g in gs {
        for l in ls {
            print!("{:8} {:8} :  ", g, l);
            for (_name, len, body) in &kernels {
                std::io::stdout().flush().unwrap();
                let time = (kernel(&c, &q, g, l, body) - noop).max(0.);
                print!("{:10.0} {:1.2} ", time * 1e6, (*len * g) as f32 / time as f32 / 1e6 / freq as f32);
            }
            println!("");
        }
    }
}

/*
fn main() {
    let c = context();
    let q = queue(&c);
    let code = "__kernel void ker(__global float *a, __global float *b) { a[0] = dot(vload4(0, a), vload4(0, b)); }";
    let program = Program::create_and_build_from_source(&c, &code, "").unwrap();
    let mut a = Buffer::<cl_float>::create(&c, CL_MEM_READ_ONLY, 4, null_mut()).unwrap();
    let mut b = Buffer::<cl_float>::create(&c, CL_MEM_READ_WRITE, 4, null_mut()).unwrap();

    let mut ha = [1.0f32, 2.0, 3.0, 4.0];
    let mut hb = [5.0f32, 6.0, 7.0, 8.0];
    let kernel = Kernel::create(&program, "ker").expect("Kernel::create failed");
    let mut send_a = q.enqueue_write_buffer(&mut a, CL_BLOCKING, 0, &ha, &[]).unwrap();
    let mut send_b = q.enqueue_write_buffer(&mut b, CL_BLOCKING, 0, &hb, &[]).unwrap();
    let mut task = ExecuteKernel::new(&kernel);
    task.set_arg(&a).set_arg(&b).set_global_work_sizes(&[1]);
    let event = task.enqueue_nd_range(&q).unwrap();
    event.wait().unwrap();
    let mut read_a = q.enqueue_read_buffer(&mut a, CL_BLOCKING, 0, &mut ha, &[]).unwrap();
    println!("{:?} {:?}", ha, hb);
}
*/
