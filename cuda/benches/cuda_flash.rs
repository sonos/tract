use criterion::measurement::WallTime;
use criterion::*;

use tract_core::internal::*;
use tract_cuda::CUDA_STREAM;
use tract_cuda::kernels::flash_attn::CudaFlashAttn;
use tract_cuda::kernels::ggml_flash_attn::GgmlFlashAttn;
use tract_gpu::tensor::IntoDevice;

pub fn cuda_ggml_flash(
    crit: &mut BenchmarkGroup<WallTime>,
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    past_seq_len: usize,
    seq_len: usize,
    out_dim: usize,
) {
    CUDA_STREAM.with(|stream| {
        let q = Tensor::zero_dt(DatumType::F32, &[batch, q_heads, seq_len, out_dim]).unwrap();
        let k = Tensor::zero_dt(
            DatumType::F16,
            &[batch, kv_heads, (past_seq_len + seq_len).next_multiple_of(256), out_dim],
        )
        .unwrap();
        let v = Tensor::zero_dt(
            DatumType::F16,
            &[batch, kv_heads, (past_seq_len + seq_len).next_multiple_of(256), out_dim],
        )
        .unwrap();
        let mask = Tensor::zero_dt(
            DatumType::F16,
            &[1, 1, seq_len.next_multiple_of(16), (past_seq_len + seq_len).next_multiple_of(256)],
        )
        .unwrap();

        let cuda_q = q.into_device().unwrap();
        let cuda_k = k.into_device().unwrap();
        let cuda_v = v.into_device().unwrap();
        let cuda_mask = mask.into_device().unwrap();

        crit.bench_function(&format!("tract_cuda_ggml_flash"), |be| {
            be.iter(|| {
                let _ =
                    GgmlFlashAttn.eval(stream, &cuda_q, &cuda_k, &cuda_v, &cuda_mask, 1.0).unwrap();
            });
        });
    })
}

pub fn cuda_minimal_flash(
    crit: &mut BenchmarkGroup<WallTime>,
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    past_seq_len: usize,
    seq_len: usize,
    out_dim: usize,
) {
    CUDA_STREAM.with(|stream| {
        let q = Tensor::zero_dt(DatumType::F16, &[batch, q_heads, seq_len, out_dim]).unwrap();
        let k =
            Tensor::zero_dt(DatumType::F16, &[batch, kv_heads, past_seq_len + seq_len, out_dim])
                .unwrap();
        let v =
            Tensor::zero_dt(DatumType::F16, &[batch, kv_heads, past_seq_len + seq_len, out_dim])
                .unwrap();
        let mask =
            Tensor::zero_dt(DatumType::F16, &[1, 1, seq_len, past_seq_len + seq_len]).unwrap();

        let cuda_q = q.into_device().unwrap();
        let cuda_k = k.into_device().unwrap();
        let cuda_v = v.into_device().unwrap();
        let cuda_mask = mask.into_device().unwrap();

        crit.bench_function(&format!("tract_cuda_minimal_flash"), |be| {
            be.iter(|| {
                let _ = CudaFlashAttn
                    .eval(stream, &cuda_q, &cuda_k, &cuda_v, Some(&cuda_mask), 1.0, false)
                    .unwrap();
            });
        });
    })
}

fn flash_attn(
    c: &mut Criterion,
    b: usize,
    qh: usize,
    kh: usize,
    p: usize,
    s: usize,
    out_dim: usize,
) {
    let mut c = c.benchmark_group(format!(
        "Batch: {b}. Q_head: {qh} KV_head: {kh} P: {p} S {s} d: {out_dim}"
    ));
    c.throughput(Throughput::Elements((4 * b * qh * s * (s + p) * out_dim) as _));

    cuda_ggml_flash(&mut c, b, qh, kh, p, s, out_dim);
    cuda_minimal_flash(&mut c, b, qh, kh, p, s, out_dim);
    c.finish();
}

#[allow(unused)]
fn small(c: &mut Criterion) {
    let shapes = vec![
        //(1, 1, 1, 0, 1, 64),
        //(1, 1, 1, 4096, 4096, 128),
        //(1, 8, 8, 4096, 4096, 128),
        //(1, 1, 1, 0, 64, 64),
        //(1, 1, 1, 0, 256, 64),
        //(1, 1, 1, 0, 256, 128),
        //(1, 4, 4, 256, 256, 64),
        //(1, 8, 8, 512, 512, 128),
        //(1, 1, 1, 4096, 4096, 64),
        //(1, 4, 4, 256, 2048, 64),
        //(1, 32, 32, 512, 1024, 128),
        //(1, 1, 1, 0, 64, 128),
        //(1, 8, 8, 0, 64, 128),
        (1, 1, 1, 0, 128, 128),
        (1, 1, 1, 64, 64, 128),
        (1, 1, 1, 32, 128, 128),
        (1, 1, 1, 128, 128, 128),
        (1, 1, 1, 64, 64, 128),
        (1, 1, 1, 0, 256, 128),
        (1, 32, 4, 0, 512, 128),
        (2, 32, 4, 0, 512, 128),
        (1, 32, 4, 256, 256, 64),
    ];
    for (b, qh, kh, p, s, out_dim) in shapes {
        flash_attn(c, b, qh, kh, p, s, out_dim);
    }
}

#[allow(unused)]
fn benched_models_pp(c: &mut Criterion) {
    let shapes = vec![
        // Llama3.2-1B
        (1, 32, 8, 0, 512, 64),
        (1, 32, 8, 0, 2048, 64),
        (1, 32, 8, 0, 1, 64),
        (1, 32, 8, 127, 1, 64),
        // Llama3.1-8B / Qwen3-7B
        (1, 32, 8, 0, 512, 128),
        (1, 32, 8, 0, 2048, 128),
        (1, 32, 8, 0, 1, 128),
        (1, 32, 8, 127, 1, 128),
        // OpenELM-270M
        (1, 20, 20, 0, 512, 64),
        (1, 20, 20, 0, 2048, 64),
        (1, 20, 20, 0, 1, 64),
        (1, 20, 20, 127, 1, 64),
        // Qwen2.5-7B
        (1, 28, 4, 0, 512, 128),
        (1, 28, 4, 0, 2048, 128),
        (1, 28, 4, 0, 1, 128),
        (1, 28, 4, 127, 1, 128),
        // Qwen3-1.7B
        (1, 16, 8, 0, 512, 128),
        (1, 16, 8, 0, 2048, 128),
        (1, 16, 8, 0, 1, 128),
        (1, 16, 8, 127, 1, 128),
    ];
    for (b, qh, kh, p, s, out_dim) in shapes {
        flash_attn(c, b, qh, kh, p, s, out_dim);
    }
}

criterion_group!(benches, benched_models_pp); //big, wavenet, asr_15_m, inception, whisper_base);
criterion_main!(benches);
