/// Generic scalar reference implementation of fused row-wise RmsNorm.
///   out_i = x_i * rsqrt(mean(x_i²) + eps)
///
/// Replaces tract-core's 4-call composition (`Reducer::MeanOfSquares` + `Add` +
/// `Rsqrt` + `Mul`) with a single 2-pass kernel. Overridden by AVX-512 on
/// x86_64; non-x86 / non-AVX512 hosts keep this scalar version.
pub fn rms_norm_f32(buf: &mut [f32], eps: f32) {
    if buf.is_empty() {
        return;
    }
    let n = buf.len() as f32;
    let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / n;
    let inv_std = (mean_sq + eps).sqrt().recip();
    for x in buf.iter_mut() {
        *x *= inv_std;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_enough(got: f32, want: f32) -> bool {
        (got - want).abs() < 1e-5
    }

    #[test]
    fn rms_norm_constant() {
        // RmsNorm of all-ones with eps=0: mean(1²)=1, rsqrt=1, output=1.
        let mut buf = [1.0; 16];
        rms_norm_f32(&mut buf, 0.0);
        for v in buf {
            assert!(close_enough(v, 1.0), "got {v}, want 1.0");
        }
    }

    #[test]
    fn rms_norm_one_two_three_four() {
        // mean(1+4+9+16)/4 = 7.5, rsqrt(7.5) ≈ 0.3651
        let mut buf = [1.0_f32, 2.0, 3.0, 4.0];
        rms_norm_f32(&mut buf, 0.0);
        let inv = (7.5_f32).sqrt().recip();
        for (i, v) in buf.iter().enumerate() {
            let want = (i + 1) as f32 * inv;
            assert!(close_enough(*v, want), "i={i}: got {v}, want {want}");
        }
    }

    #[test]
    fn rms_norm_eps_added_under_root() {
        // eps inside the sqrt, not added afterward.
        let mut buf = [0.0_f32; 4];
        rms_norm_f32(&mut buf, 1e-5);
        for v in buf {
            // 0 * anything = 0; just verify no NaN/inf.
            assert!(v.is_finite());
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn rms_norm_empty() {
        let mut buf: [f32; 0] = [];
        rms_norm_f32(&mut buf, 1e-5);
    }
}
