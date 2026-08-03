#![allow(clippy::excessive_precision)]
use crate::frame::element_wise::ElementWiseKer;
use tract_data::internal::*;

// Tanh-form GELU approximation matching tract's GeluApproximate (pow=3, the
// canonical Hendrycks-Gimpel/Open-AI form):
//
//     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The fast variant (pow=2) is not exposed here; the graph op falls back to
// scalar when fast_impl=true.

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const COEF: f32 = 0.044715;

/// The scalar reference every GELU kernel in this module is defined against.
/// `HGeluLut8`'s table is built from this function, so the table is bit-identical
/// to `HGelu8` by construction rather than by approximation.
#[inline]
fn gelu(v: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

#[derive(Clone, Debug)]
pub struct SGelu4;

impl ElementWiseKer<f32> for SGelu4 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = gelu(*px));
    }
}

#[derive(Clone, Debug)]
pub struct HGelu8;

impl ElementWiseKer<f16> for HGelu8 {
    fn name() -> &'static str {
        "generic"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        8
    }

    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        x.iter_mut().for_each(|px| *px = f16::from_f32(gelu(px.to_f32())));
    }
}

/// Every f16 bit pattern mapped through `gelu`, so the whole activation is one
/// load per element. 128 KiB, built on first use: a model with no f16 GELU never
/// pays for it, and the build costs 65536 scalar evaluations.
fn gelu_lut() -> &'static [u16; 1 << 16] {
    static LUT: std::sync::OnceLock<Box<[u16; 1 << 16]>> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = Box::new([0u16; 1 << 16]);
        for (bits, slot) in lut.iter_mut().enumerate() {
            *slot = f16::from_f32(gelu(f16::from_bits(bits as u16).to_f32())).to_bits();
        }
        lut
    })
}

#[derive(Clone, Debug)]
pub struct HGeluLut8;

impl ElementWiseKer<f16> for HGeluLut8 {
    fn name() -> &'static str {
        "lut"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        8
    }

    fn run(x: &mut [f16], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        let lut = gelu_lut();
        x.iter_mut().for_each(|px| *px = f16::from_bits(lut[px.to_bits() as usize]));
    }
}

#[cfg(test)]
#[macro_use]
pub mod s {
    gelu_frame_tests!(true, f32, crate::generic::gelu::SGelu4);
}

#[cfg(test)]
#[macro_use]
pub mod h {
    gelu_frame_tests!(true, tract_data::internal::f16, crate::generic::gelu::HGelu8);
}

#[cfg(test)]
mod lut {
    use super::*;

    #[test]
    fn lut_matches_scalar_kernel_on_every_f16() {
        let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let mut reference = all.clone();
        let mut lut = all;
        HGelu8::ew().run(&mut reference).unwrap();
        HGeluLut8::ew().run(&mut lut).unwrap();
        let mismatch = reference
            .iter()
            .zip(&lut)
            .position(|(a, b)| a.to_bits() != b.to_bits())
            .map(|i| (f16::from_bits(i as u16), reference[i], lut[i]));
        assert_eq!(mismatch, None);
    }

    fn ordered(x: f16) -> i32 {
        let b = x.to_bits();
        if b & 0x8000 != 0 { !b as i32 } else { (b | 0x8000) as i32 }
    }

    #[test]
    fn registered_kernel_tracks_the_scalar_kernel_on_every_f16() {
        let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let mut reference = all.clone();
        let mut registered = all;
        HGelu8::ew().run(&mut reference).unwrap();
        (crate::ops().gelu_f16)().run(&mut registered).unwrap();
        let worst = reference
            .iter()
            .zip(&registered)
            .filter(|(a, b)| !(a.is_nan() && b.is_nan()))
            .map(|(a, b)| (ordered(*a) - ordered(*b)).abs())
            .max()
            .unwrap();
        assert!(worst <= 1, "registered gelu_f16 drifts {worst} ulp from the scalar kernel");
    }
}
