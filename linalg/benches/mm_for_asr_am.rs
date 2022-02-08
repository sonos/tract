use criterion::*;

mod utils;
use utils::*;

fn all(c: &mut Criterion) {
    // packed_packed: co, ci, n
//    direct_conv(c, "asr_2M", 24, 5, 40, 200, 1); // lda
    packed_packed(c, "asr_2M", 256, 200, 24); // tdnn1
//    direct_conv(c, "asr_2M", 24, 3, 256, 256, 1); // tdnn2
//    direct_conv(c, "asr_2M", 24, 3, 256, 256, 3); // tdnn3
    packed_packed(c, "asr_2M", 256, 256, 8); // fastlstm1 and 2 (input) x 8 (4 prod x 2 layers)
    packed_packed(c, "asr_2M", 256, 128, 1); // fastlstm1 and 2 (hidden) x 64 (4 prod x 2 layers x 8 loops)
    packed_packed(c, "asr_2M", 256, 256, 1); // fastlstm1 and 2 (rp) x 16 (2 layers x 8 loops)
//    direct_conv(c, "asr_2M", 8, 3, 256, 256, 1); // tdnn4, tdd5 (x2)
    packed_packed(c, "asr_2M", 1690, 256, 8); // output

    // 8M
    packed_packed(c, "asr_8M", 512, 200, 24); // tdnn1
    packed_packed(c, "asr_8M", 512, 512, 24); // tdnn2
    packed_packed(c, "asr_8M", 512, 256, 1); // fastlstm1 and 2 (four parts, rec mat*vec)
    packed_vec(c, "asr_8M", 512, 256, 1); // fastlstm1 and 2 (four parts, rec mat*vec)

    // pseudo 15M
    packed_packed(c, "asr_pseudo15M", 768, 200, 24); // tdnn1
    packed_packed(c, "asr_pseudo15M", 768, 2304, 24); // tdnn2
    packed_packed(c, "asr_pseudo15M", 768, 2304, 8); // tdnn3,4,5
    packed_packed(c, "asr_pseudo15M", 768, 768, 8); // fastlstm1 and 2 (four parts, rec mat*mat)
    packed_packed(c, "asr_pseudo15M", 768, 384, 1); // fastlstm1 and 2 (four parts, rec mat*vec)
    packed_vec(c, "asr_pseudo15M", 768, 384, 1); // fastlstm1 and 2 (four parts, rec mat*vec)

    // 15M
    packed_vec(c, "asr_15M", 768, 256, 1); // fastlstm1 and 2 (four parts, rec mat*vec)
}

criterion_group!(benches, all);
criterion_main!(benches);
