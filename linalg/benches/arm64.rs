use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};

use tract_data::prelude::*;
use tract_linalg::frame::mmm::LinearSpec;
use tract_linalg::frame::mmm::MatMatMulKer;
use tract_linalg::frame::mmm::MatMatMulKerSpec;
use tract_linalg::mmm::{PanelStore, Tile};

fn ker<T: Datum + Copy + num_traits::Zero, K: MatMatMulKer<T>>(
    criterion: &mut BenchmarkGroup<WallTime>,
    k: usize,
) {
    let item_size = T::datum_type().size_of();
    let a = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_a()) * K::mr()],
        K::alignment_bytes_packed_a(),
    )
    .unwrap();
    let b = Tensor::zero_aligned::<T>(
        &[(k + K::end_padding_packed_b()) * K::nr()],
        K::alignment_bytes_packed_b(),
    )
    .unwrap();
    let mut c = Tensor::zero::<T>(&[K::mr() * K::nr()]).unwrap();
    let ref a = PanelStore::Packed { ptr: unsafe { a.as_ptr_unchecked::<u8>() as _ } };
    let ref b = PanelStore::Packed { ptr: unsafe { b.as_ptr_unchecked::<u8>() as _ } };
    let ref c = Tile {
        ptr: unsafe { c.as_ptr_mut_unchecked::<u8>() as _ },
        item_size,
        col_byte_stride: (item_size * K::mr()) as isize,
        row_byte_stride: item_size as isize,
    };
    let ref linear = LinearSpec::Mul { k };
    let op = MatMatMulKerSpec { a, b, c, linear, non_linear: std::ptr::null() };
    criterion.bench_function(format!("{}/{}x{}", K::name(), K::mr(), K::nr()), |b| {
        b.iter(|| K::kernel(&op))
    });
}

fn run(criterion: &mut Criterion) {
    use tract_linalg::arm64::*;
    let k = 1000;
    let mut criterion = criterion.benchmark_group(format!("k{}", k));
    ker::<f32, MatMatMulF32x16x4>(&mut criterion, k);
    ker::<f32, MatMatMulF32x12x8>(&mut criterion, k);
    ker::<f32, MatMatMulF32x8x8>(&mut criterion, k);
    ker::<f32, MatMatMulF32x16x4A53>(&mut criterion, k);
    ker::<f32, MatMatMulF32x12x8A53>(&mut criterion, k);
    ker::<f32, MatMatMulF32x8x8A53>(&mut criterion, k);
    criterion.finish();
}

criterion_group!(benches, run);
criterion_main!(benches);
