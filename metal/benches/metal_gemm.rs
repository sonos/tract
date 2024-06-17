use criterion::measurement::WallTime;
use criterion::*;
use ggml::Context;
// use ggml;
use tract_core::internal::*;
use tract_metal::*;

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
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    use tract_linalg::frame::mmm::FusedSpec;
    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    let mut c = Tensor::zero_dt(dt, &[m, n]).unwrap();

    // mk,kn -> mn
    unsafe {
        let mmm = tract_linalg::ops().mmm(dt, dt, dt, Some(m), Some(k), Some(n)).unwrap();

        let c_storage = mmm.c_view(0, 1);

        let mut scratch = mmm.allocate_scratch_space();

        crit.bench_function(&format!("tract_with_packing_{:?}", dt), |be| {
            let packed_a = mmm.a_pack().pack_tensor(&a, 1, 0).unwrap();
            let packed_b = mmm.b_pack().pack_tensor(&b, 0, 1).unwrap();

            be.iter(|| {
                mmm.run_with_scratch_space(
                    m,
                    n,
                    &mut *scratch,
                    &[
                        FusedSpec::AddMatMul { a: packed_a.as_ref(), b: packed_b.as_ref() },
                        FusedSpec::Store(c_storage.wrap(&mut c.view_mut())),
                    ],
                )
                .unwrap()
            });
        });
    }
}

pub fn metal_mat_vec(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    let context = MetalContext::new();
    context.shared_context().load_library(LibraryName::MfaLib).unwrap();

    let a = Tensor::zero_dt(dt, &[m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[k, n]).unwrap();
    let metal_a = a.into_metal().unwrap();
    let metal_b = b.into_metal().unwrap();
    // Warmup
    let _ = tract_metal::kernels::mat_vec(&context, &metal_a, &metal_b).unwrap();

    crit.bench_function(&format!("tract_metal_mat_vec_{:?}", dt), |be| {
        be.iter(|| {
            let _ = tract_metal::kernels::mat_vec(&context, &metal_a, &metal_b).unwrap();
        });
    });
    // metal_mat_vec
}

pub fn metal_gemm(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    let context = MetalContext::new();
    context.shared_context().load_library(LibraryName::MfaLib).unwrap();

    let a = Tensor::zero_dt(dt, &[1, m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[1, k, n]).unwrap();
    let metal_a = a.into_metal().unwrap();
    let metal_b = b.into_metal().unwrap();
    // Warmup
    let _ = tract_metal::kernels::mfa_gemm(&context, &metal_a, &metal_b).unwrap();

    crit.bench_function(&format!("tract_metal_gemm_{:?}", dt), |be| {
        be.iter(|| {
            let _ = tract_metal::kernels::mfa_gemm(&context, &metal_a, &metal_b).unwrap();
        });
    });
}

pub fn metal_gemm_without_cache(
    crit: &mut BenchmarkGroup<WallTime>,
    m: usize,
    k: usize,
    n: usize,
    dt: DatumType,
) {
    let context = MetalContext::new();
    context.shared_context().load_library(LibraryName::MfaLib).unwrap();

    let a = Tensor::zero_dt(dt, &[1, m, k]).unwrap();
    let b = Tensor::zero_dt(dt, &[1, k, n]).unwrap();
    let metal_a = a.into_metal().unwrap();
    let metal_b = b.into_metal().unwrap();
    // Warmup
    let _ = tract_metal::kernels::mfa_gemm(&context, &metal_a, &metal_b).unwrap();

    crit.bench_function(&format!("tract_metal_gemm_without_cache_{:?}", dt), |be| {
        be.iter(|| {
            context.shared_context().flush_pipeline_cache().unwrap();
            let _ = tract_metal::kernels::mfa_gemm(&context, &metal_a, &metal_b).unwrap();
        });
    });
}

pub fn metal_tile_8x8(crit: &mut BenchmarkGroup<WallTime>, dim: usize, dt: DatumType) {
    let context = MetalContext::new();
    crit.bench_function(&format!("tract_metal_mmm_tile_8x8_{:?}", dt), |be| {
        let a = Tensor::zero_dt(dt, &[dim, dim]).unwrap();
        let b = Tensor::zero_dt(dt, &[dim, dim]).unwrap();
        let metal_a = a.into_metal().unwrap();
        let metal_b = b.into_metal().unwrap();

        be.iter(|| {
            let _ = tract_metal::kernels::mmm_tile_8x8(&context, &metal_a, &metal_b).unwrap();
        });
    });
}

fn matmul(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let mut c = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    c.throughput(Throughput::Elements((m * k * n) as _));
    ggml_matmul(&mut c, m, k, n, f32::datum_type());
    if m == 1 || n == 1 {
        metal_mat_vec(&mut c, m, k, n, f32::datum_type());
    }
    tract_with_packing(&mut c, m, k, n, f32::datum_type());
    metal_gemm(&mut c, m, k, n, f32::datum_type());
    metal_gemm_without_cache(&mut c, m, k, n, f32::datum_type());
    // ggml_matmul(&mut c, m, k, n, f16::datum_type());
    tract_with_packing(&mut c, m, k, n, f16::datum_type());
    metal_gemm(&mut c, m, k, n, f16::datum_type());
    c.finish();
}

fn tinyllama(c: &mut Criterion) {
    let shapes = vec![
        (32003, 2048, 6),
        (1, 32, 32),
        (1, 4, 4),
        (1, 4096, 4096),
        (1, 2048, 2048),
        (1, 1024, 1024),
        (1, 128, 128),
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
    for (m, k, n) in shapes {
        matmul(c, m, k, n);
    }
}

fn big(c: &mut Criterion) {
    matmul(c, 2048, 2048, 1);
    matmul(c, 1, 2048, 2048);
    matmul(c, 2048, 2048, 2048);
    matmul(c, 4096, 4096, 4096);
}

fn wavenet(c: &mut Criterion) {
    matmul(c, 32, 32, 8);
    matmul(c, 16, 60, 8);
}

fn asr_15_m(c: &mut Criterion) {
    matmul(c, 768, 200, 24);
    matmul(c, 768, 2304, 24);
    matmul(c, 768, 2304, 8);
    matmul(c, 768, 384, 1);
}

fn inception(c: &mut Criterion) {
    matmul(c, 64, 288, 21609);
}

fn whisper_base(c: &mut Criterion) {
    matmul(c, 512, 512, 1500);
}

criterion_group!(benches, tinyllama, big, wavenet, asr_15_m, inception, whisper_base);
criterion_main!(benches);
