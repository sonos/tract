use crate::matmul::{BasicMatMul, GemmImpl, GemmKernel, MfaGemm, MlxGemm};
use criterion::measurement::WallTime;
use criterion::*;
use ggml::Context;
use tract_core::internal::*;
use tract_gpu::tensor::IntoDevice;
use tract_linalg::mmm::AsInputValue;
use tract_metal::kernels::matmul::GgmlGemm;
use tract_metal::kernels::{matmul, LibraryName};
use tract_metal::MetalStream;

pub fn ggml_matmul(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    let ggml_dt = match dt {
        DatumType::F32 => ggml::Type::F32,
        DatumType::F16 => ggml::Type::F16,
        _ => unimplemented!(),
    };

    let ctxt = Context::new_with_allocate(500_000_000);

    let mut t = ctxt.new_tensor_3d(ggml_dt, 1, 2, 3);
    t.zero_data();

    let mut a = ctxt.new_tensor_2d(ggml_dt, k, m);
    a.zero_data();
    let mut b = ctxt.new_tensor_2d(ggml_dt, k, n); // intern transposition
    b.zero_data();

    crit.bench_function(&format!("ggml_{:?}", dt), |be| {
        be.iter(|| {
            let ctxt = Context::new_with_allocate(500_000_000);
            let mut a = ctxt.new_tensor_2d(ggml_dt, k, m);
            a.zero_data();
            let mut b = ctxt.new_tensor_2d(ggml_dt, k, n); // intern transposition
            b.zero_data();
            let c = ctxt.op_mul_mat(&a, &b);
            let mut graph = ctxt.create_compute_graph();
            graph.build_forward_expand(&c);

            let mut execution_plan = ggml::GraphExecutionPlan::new(&mut graph, 1);
            execution_plan.execute(&ctxt);
        });
    });
}

pub fn tract_with_packing(
    crit: &mut BenchmarkGroup<WallTime>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    use tract_linalg::mmm::FusedSpec;
    let a = Tensor::zero_dt(dt, &[batch, m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[batch, k, n]).unwrap();
    let mut c = Tensor::zero_dt(dt, &[m, n]).unwrap();

    // mk,kn -> mn
    unsafe {
        let mmm = tract_linalg::ops().mmm(dt, Some(m), Some(k), Some(n)).unwrap();

        let c_storage = mmm.c_view(Some(0), Some(1));

        let mut scratch = mmm.allocate_scratch_space();

        let (packer_a, packer_b) = &mmm.packings()[0];

        crit.bench_function(&format!("tract_with_packing_{:?}", dt), |be| {
            let packed_a = packer_a.prepare_one(&a, 1, 0).unwrap();
            let packed_b = packer_b.prepare_one(&b, 0, 1).unwrap();

            be.iter(|| {
                mmm.run_with_scratch_space(
                    m,
                    n,
                    &mut *scratch,
                    &[
                        FusedSpec::AddMatMul {
                            packing: 0,
                            a: AsInputValue::Borrowed(&(*packed_a)),
                            b: AsInputValue::Borrowed(&(*packed_b)),
                        },
                        FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
                    ],
                )
                .unwrap()
            });
        });
    }
}

pub fn metal_gemm<K: GemmKernel>(
    crit: &mut BenchmarkGroup<WallTime>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
    is_ggml: bool,
) {
    let stream = MetalStream::new();
    stream.load_library(LibraryName::MfaLib).unwrap();

    let a = Tensor::zero_dt(dt, &[batch, m, k]).unwrap();
    let b = if is_ggml {
        Tensor::zero_dt(dt, &[batch, n, k]).unwrap()
    } else {
        Tensor::zero_dt(dt, &[batch, k, n]).unwrap()
    };

    let metal_a = a.into_device().unwrap();
    let metal_b = b.into_device().unwrap();
    // Warmup
    let _ = GemmImpl::<MfaGemm>::default().eval(&stream, &metal_a, &metal_b).unwrap();

    crit.bench_function(&format!("tract_metal_gemm_{}_{:?}", K::name(), dt), |be| {
        be.iter(|| {
            let _ = GemmImpl::<K>::new(false, is_ggml).eval(&stream, &metal_a, &metal_b).unwrap();
        });
    });
}

fn matmul(c: &mut Criterion, b: usize, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}x{}", b, m, k, n));
    c.throughput(Throughput::Elements((m * k * n) as _));
    // ggml_matmul(&mut c, m, k, n, f32::datum_type());

    for dt in [f32::datum_type(), f16::datum_type()] {
        metal_gemm::<BasicMatMul>(&mut c, b, m, k, n, dt, false);
        metal_gemm::<MlxGemm>(&mut c, b, m, k, n, dt, false);
        metal_gemm::<MfaGemm>(&mut c, b, m, k, n, dt, false);
        metal_gemm::<GgmlGemm>(&mut c, b, m, k, n, dt, true);
    }
    // ggml_matmul(&mut c, m, k, n, f16::datum_type());
    // tract_with_packing(&mut c, b, m, k, n, f32::datum_type());
    //tract_with_packing(&mut c, b, m, k, n, f16::datum_type());
    c.finish();
}

#[allow(unused)]
fn tinyllama(c: &mut Criterion) {
    let shapes = vec![
        (32, 1, 25, 32),
        (1, 32003, 2048, 1),
        (1, 1, 2048, 32003),
        // (32003, 2048, 6),
        // (1, 32, 32),
        // (1, 4, 4),
        // (1, 4096, 4096),
        // (1, 2048, 2048),
        // (1, 1024, 1024),
        // (1, 128, 128),
        // (1, 64, 3),
        // (1, 64, 1),
        // (1, 5632, 2048),
        // (1, 3, 64),
        // (1, 64, 13),
        // (1, 12, 64),
        // (1, 2048, 5632),
        // (1, 2048, 32003),
        // (1, 2048, 2048),
        // (1, 2048, 256),
    ];
    for (b, m, k, n) in shapes {
        matmul(c, b, m, k, n);
    }
}

#[allow(unused)]
fn big(c: &mut Criterion) {
    matmul(c, 1, 2048, 2048, 1);
    matmul(c, 1, 1, 2048, 2048);
    matmul(c, 1, 2048, 2048, 2048);
    matmul(c, 1, 4096, 4096, 4096);
}

#[allow(unused)]
fn wavenet(c: &mut Criterion) {
    matmul(c, 1, 32, 32, 8);
    matmul(c, 1, 16, 60, 8);
}

#[allow(unused)]
fn asr_15_m(c: &mut Criterion) {
    matmul(c, 1, 768, 200, 24);
    matmul(c, 1, 768, 2304, 24);
    matmul(c, 1, 768, 2304, 8);
    matmul(c, 1, 768, 384, 1);
}

#[allow(unused)]
fn inception(c: &mut Criterion) {
    matmul(c, 1, 64, 288, 21609);
}

#[allow(unused)]
fn whisper_base(c: &mut Criterion) {
    matmul(c, 1, 512, 512, 1500);
}

criterion_group!(benches, tinyllama); //big, wavenet, asr_15_m, inception, whisper_base);
criterion_main!(benches);
