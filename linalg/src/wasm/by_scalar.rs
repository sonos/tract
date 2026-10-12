// `f32::max`/`f32::min` lower to per-lane NaN-checking scalar code on wasm32. `pmax(s, x)` is
// `s < x ? x : s` and `pmin(s, x)` is `x < s ? x : s`: a NaN in the buffer yields `s` and
// signed zeros resolve to `s`, as the scalar code does. Only a NaN scalar needs a branch: it
// leaves the buffer unchanged, and pmax/pmin would propagate it.

routine_by_scalar_rust!(wasm32;
    f32,
    wasm_max_by_scalar_f32_16n,
    16,
    4,
    fn run(buf: &mut [f32], s: f32) {
        use std::arch::wasm32::*;
        if s.is_nan() {
            return;
        }
        let sv = f32x4_splat(s);
        for c in buf.chunks_exact_mut(16) {
            let p = c.as_mut_ptr() as *mut v128;
            for j in 0..4 {
                unsafe { v128_store(p.add(j), f32x4_pmax(sv, v128_load(p.add(j)))) };
            }
        }
    },
    op(Max),
    bin
);

routine_by_scalar_rust!(wasm32;
    f32,
    wasm_min_by_scalar_f32_16n,
    16,
    4,
    fn run(buf: &mut [f32], s: f32) {
        use std::arch::wasm32::*;
        if s.is_nan() {
            return;
        }
        let sv = f32x4_splat(s);
        for c in buf.chunks_exact_mut(16) {
            let p = c.as_mut_ptr() as *mut v128;
            for j in 0..4 {
                unsafe { v128_store(p.add(j), f32x4_pmin(sv, v128_load(p.add(j)))) };
            }
        }
    },
    op(Min),
    bin
);

#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod tests {
    use super::*;
    use crate::frame::element_wise::ElementWiseKer;

    /// Bit-for-bit against `f32::max`/`f32::min`, on the values the proptests leave out:
    /// NaN in the buffer, signed zeros, infinities, and a NaN scalar.
    #[test]
    fn edge_values_match_scalar_semantics() {
        let values =
            [f32::NAN, -0.0, 0.0, 1.5, -1.5, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE];
        for s in [0.0f32, -0.0, 1.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for len in [1usize, 7, 16, 17, 40] {
                let buf: Vec<f32> = (0..len).map(|i| values[i % values.len()]).collect();
                let mut got = buf.clone();
                wasm_max_by_scalar_f32_16n::ew().run_with_params(&mut got, s).unwrap();
                let want: Vec<u32> = buf.iter().map(|x| x.max(s).to_bits()).collect();
                assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want, "max s={s}");
                let mut got = buf.clone();
                wasm_min_by_scalar_f32_16n::ew().run_with_params(&mut got, s).unwrap();
                let want: Vec<u32> = buf.iter().map(|x| x.min(s).to_bits()).collect();
                assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want, "min s={s}");
            }
        }
    }
}
