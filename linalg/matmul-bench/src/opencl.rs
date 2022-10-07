use std::ptr::null_mut;

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

pub struct Gpu {
    context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    mr: usize,
    nr: usize,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    fn create(k: &str, mr: usize, nr: usize) -> Self {
        let platforms = get_platforms().unwrap();
        let device = platforms[0].get_devices(CL_DEVICE_TYPE_GPU).unwrap().remove(0);
        let device = Device::new(device);
        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = CommandQueue::create_with_properties(
            &context,
            context.default_device(),
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .expect("CommandQueue::create failed");

        let kernel_cl = r#"
    __kernel void gemm_0(const int M, const int K, const int N,
                          const __global float* A,
                          const __global float* B,
                          __global float* C) {

        const int m = get_global_id(0);
        const int n = get_global_id(1);

        float acc = 0.0f;
        for (int k=0; k<K; k++) {
            acc += A[m * K + k] * B[k * N + n];
        }

        C[m * N + n] = acc;
    }

    __kernel void gemm_1(const int M, const int K, const int N,
                            const __global float* A,
                            const __global float* B,
                            __global float* C) {

          const int m = get_global_id(0) * 4;
          const int n = get_global_id(1) * 4;

          float acc00 = 0.0f;
          float acc10 = 0.0f;
          float acc20 = 0.0f;
          float acc30 = 0.0f;

          float acc01 = 0.0f;
          float acc11 = 0.0f;
          float acc21 = 0.0f;
          float acc31 = 0.0f;

          float acc02 = 0.0f;
          float acc12 = 0.0f;
          float acc22 = 0.0f;
          float acc32 = 0.0f;

          float acc03 = 0.0f;
          float acc13 = 0.0f;
          float acc23 =0.0f;
          float acc33 = 0.0f;

          for (int k=0; k<K; k++) {
              float a0 = A[m * K + k];
              float a1 = A[m * K + k + M];
              float a2 = A[m * K + k + 2 * M];
              float a3 = A[m * K + k + 3 * M];

              float b0 = B[k * N + n];
              float b1 = B[k * N + n + 1];
              float b2 = B[k * N + n + 2];
              float b3 = B[k * N + n + 3];

              acc00 += a0 * b0;
              acc10 += a1 * b0;
              acc20 += a2 * b0;
              acc30 += a3 * b0;

              acc01 += a0 * b1;
              acc11 += a1 * b1;
              acc21 += a2 * b1;
              acc31 += a3 * b1;

              acc02 += a0 * b2;
              acc12 += a1 * b2;
              acc22 += a2 * b2;
              acc32 += a3 * b2;

              acc03 += a0 * b3;
              acc13 += a1 * b3;
              acc23 += a2 * b3;
              acc33 += a3 * b3;
          }

          C[m * N + n] = acc00;
          C[m * N + n + M] = acc10;
          C[m * N + n + 2 * M] = acc20;
          C[m * N + n + 3 * M] = acc30;

          C[m * N + n + 1] = acc01;
          C[m * N + n + M + 1] = acc11;
          C[m * N + n + 2 * M + 1] = acc21;
          C[m * N + n + 3 * M + 1] = acc31;

          C[m * N + n + 2] = acc02;
          C[m * N + n + M + 2] = acc12;
          C[m * N + n + 2 * M + 2] = acc22;
          C[m * N + n + 3 * M + 2] = acc32;

          C[m * N + n + 3] = acc03;
          C[m * N + n + M + 3] = acc13;
          C[m * N + n + 2 * M + 3] = acc23;
          C[m * N + n + 3 * M + 3] = acc33;
      }

      // packed
      __kernel void gemm_2(const int M, const int K, const int N,
                            const __global float* A,
                            const __global float* B,
                            __global float* C) {

          const int m = get_global_id(0);
          const int n = get_global_id(1);

          #pragma promote_to_registers
          float4 acc[4];

          for (int i=0; i<4; i++) {
            acc[i].x = 0;
            acc[i].y = 0;
            acc[i].z = 0;
            acc[i].w = 0;
          }

          const __global float *pa = &A[m*K*4];
          const __global float *pb = &B[n*K*4];

          for (int k=0; k<K; k++) {
            #pragma promote_to_registers
            float4 a = vload4(k, pa);
            #pragma promote_to_registers
            float4 b = vload4(k, pb);

            // #define mac(a, b, c) c += a * b;
            #define mac(a, b, c) c = mad(a, b, c);

            #pragma unroll
            for (int i = 0; i<4; i++) {
              float va;
              switch(i) {
                case 0: va = a.x; break;
                case 1: va = a.y; break;
                case 2: va = a.z; break;
                case 3: va = a.w; break;
              }
              mac(va, b.x, acc[i].x)
              mac(va, b.y, acc[i].y)
              mac(va, b.z, acc[i].z)
              mac(va, b.w, acc[i].w)
          }
        }

        #pragma unroll
        for (int i = 0; i<4; i++) {
          int offset = n + i * N / 4 + m * N;
          vstore4(acc[i], offset, C);
        }
      }
      "#;

        let program = Program::create_and_build_from_source(&context, kernel_cl, "").unwrap();
        let kernel = Kernel::create(&program, k).expect("Kernel::create failed");

        println!("device: {}", device.name().unwrap());

        Gpu { context, queue, kernel, mr, nr }
    }
}

#[derive(Default)]
struct Params {
    packed: bool,
    local_sizes: Option<(usize, usize)>,
}

impl Gpu {
    fn run(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        params: Params,
    ) -> Result<(), ClError> {
        let mut a_cl =
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, m * k, null_mut())?;
        let mut b_cl =
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, k * n, null_mut())?;

        let packed_a = crate::pack_a(a, m, k, self.mr);
        let packed_b = crate::pack_b(b, k, n, self.nr);

        let (pa, pb) = if params.packed { (&*packed_a, &*packed_b) } else { (a, b) };

        let mut c_cl =
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, m * n, null_mut())?;

        let write_a = self.queue.enqueue_write_buffer(&mut a_cl, CL_NON_BLOCKING, 0, pa, &[])?;
        let write_b = self.queue.enqueue_write_buffer(&mut b_cl, CL_NON_BLOCKING, 0, pb, &[])?;

        let mut run = ExecuteKernel::new(&self.kernel);
        run.set_arg(&(m as i32))
            .set_arg(&(k as i32))
            .set_arg(&(n as i32))
            .set_arg(&a_cl)
            .set_arg(&b_cl)
            .set_arg(&c_cl)
            .set_global_work_sizes(&[m / self.mr, n / self.nr])
            .set_event_wait_list(&[write_a.get(), write_b.get()]);
        if let Some((mr, nr)) = params.local_sizes {
            run.set_local_work_sizes(&[mr, nr]);
        }
        let run = run.enqueue_nd_range(&self.queue).unwrap();

        let read_c =
            self.queue.enqueue_read_buffer(&mut c_cl, CL_NON_BLOCKING, 0, c, &[run.get()])?;
        read_c.wait()?;
        Ok(())
    }
}

#[allow(non_upper_case_globals)]
mod kernels {
    pub use super::*;
    use std::sync::Mutex;

    macro_rules! kernel {
        ($id:ident, $mr: expr, $nr: expr) => {
            lazy_static::lazy_static! {
                pub static ref $id: Mutex<Gpu> = {
                    Mutex::new(Gpu::create(stringify!($id), $mr, $nr))
                };
            }
        };
    }

    kernel!(gemm_1, 4, 4);
    kernel!(gemm_2, 4, 4);
}

pub fn opencl_gemm1(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    kernels::gemm_1.lock().unwrap().run(m, k, n, a, b, c, Params::default()).unwrap();
}

pub fn opencl_gemm_1_with_local_2x2(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    kernels::gemm_1
        .lock()
        .unwrap()
        .run(m, k, n, a, b, c, Params { local_sizes: Some((2, 2)), ..Params::default() })
        .unwrap();
}

pub fn opencl_gemm_2_pack(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    kernels::gemm_2
        .lock()
        .unwrap()
        .run(m, k, n, a, b, c, Params { packed: true, local_sizes: Some((2,2)), ..Params::default() })
        .unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::t;

    t!(opencl_gemm1);
    t!(opencl_gemm_1_with_local_2x2);
    t!(opencl_gemm_2_pack);
}
