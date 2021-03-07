use criterion::*;

mod utils;
use utils::*;

fn s16x60x8(c: &mut Criterion) {
    packed_packed(c, "wavenet", 32, 32, 8, true); // postproc
    packed_packed(c, "wavenet", 16, 60, 8, true);
}

criterion_group!(benches, s16x60x8);
criterion_main!(benches);
