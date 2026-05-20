//! Direct, register-blocked convolution for the "channel-mixing temporal conv"
//! shape class: NCHW, kernel width 1 (extent only on H), unit stride/dilation on
//! the contiguous W axis, grouped, with a *small* number of output channels per
//! group (`ocg`).
//!
//! For such convs the im2col + matmul lowering is inefficient: the per-group
//! matmul is `M = ocg` (tiny, e.g. 5) × `K = icg·KH` × `N = H·W`, so the matmul
//! kernel's m-tile is mostly wasted — exactly the same pathology as a low-M GEMV.
//! ORT side-steps it with a direct conv.
//!
//! This op computes the conv directly: for each (group, output-row, block of the
//! contiguous W axis) it holds `ocg` accumulators in registers and reduces over
//! `(kh, icg)`, loading each input row ONCE and reusing it across all `ocg`
//! outputs (the same input-reuse a GEMM gets). Measured on df_dec's `df_convp.1`
//! (group=2, 64→10ch, kernel [5,1], 100×96): 0.77 ms native / 0.79 ms wasm vs
//! 1.72 / 2.42 ms for tract's lazy im2col and 1.13 ms for ORT — a 2.2–3.1× win,
//! bit-exact.
//!
//! Eligibility is checked in `Conv::codegen`; anything outside the supported
//! shape class falls back to im2col.

use crate::internal::*;

/// Width of the inner SIMD-vectorised block over the contiguous W axis.
const WB: usize = 16;

/// Direct blocked conv. Inputs: X [N, C, H, W] (NCHW, f32), kernel
/// [OC, ICG·KH] (group-major: row `oc` holds its group's `icg·KH` weights,
/// i-major/h-minor), bias [OC]. Output [N, OC, H_out, W].
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BlockedConv {
    pub n: usize,
    pub c_in: usize,
    pub h_in: usize,
    pub w: usize,
    pub oc: usize,
    pub group: usize,
    pub kh: usize,
    pub stride_h: usize,
    pub dil_h: usize,
    pub pad_before_h: usize,
    pub h_out: usize,
}

impl BlockedConv {
    #[inline]
    fn icg(&self) -> usize {
        self.c_in / self.group
    }
    #[inline]
    fn ocg(&self) -> usize {
        self.oc / self.group
    }
}

impl Op for BlockedConv {
    fn name(&self) -> StaticName {
        "BlockedConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "N={} C={}->OC={} group={} kh={} (icg={} ocg={}) HxW={}x{} -> H_out={} pad_before={} stride_h={} dil_h={}",
            self.n,
            self.c_in,
            self.oc,
            self.group,
            self.kh,
            self.icg(),
            self.ocg(),
            self.h_in,
            self.w,
            self.h_out,
            self.pad_before_h,
            self.stride_h,
            self.dil_h,
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for BlockedConv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let x_t = inputs[0].cast_to::<f32>()?;
        let k_t = inputs[1].cast_to::<f32>()?;
        let b_t = inputs[2].cast_to::<f32>()?;
        // SAFETY: just cast to f32; conv I/O tensors are standard (contiguous) layout.
        let x = unsafe { x_t.as_slice_unchecked::<f32>() };
        let kernel = unsafe { k_t.as_slice_unchecked::<f32>() };
        let bias_raw = unsafe { b_t.as_slice_unchecked::<f32>() };
        // Normalise bias to a per-output-channel vector (it may arrive as a
        // scalar zero, empty, or already [oc]).
        let bias_vec: Vec<f32> = match bias_raw.len() {
            0 => vec![0.0; self.oc],
            1 => vec![bias_raw[0]; self.oc],
            _ => bias_raw.to_vec(),
        };
        let bias = bias_vec.as_slice();

        let mut output =
            unsafe { Tensor::uninitialized::<f32>(&[self.n, self.oc, self.h_out, self.w])? };
        let out = unsafe { output.as_slice_mut_unchecked::<f32>() };

        let ocg = self.ocg();
        match ocg {
            1 => self.run::<1>(x, kernel, bias, out),
            2 => self.run::<2>(x, kernel, bias, out),
            3 => self.run::<3>(x, kernel, bias, out),
            4 => self.run::<4>(x, kernel, bias, out),
            5 => self.run::<5>(x, kernel, bias, out),
            6 => self.run::<6>(x, kernel, bias, out),
            8 => self.run::<8>(x, kernel, bias, out),
            _ => self.run_generic(x, kernel, bias, out),
        }

        Ok(tvec!(output.into_tvalue()))
    }
}

impl BlockedConv {
    /// Const-OCG fast path: `ocg` accumulators of WB lanes held in registers.
    ///
    /// The hot loop (full WB-wide blocks) touches `acc` ONLY at compile-time
    /// constant offsets `[ocl][j]` (ocl<OCG, j<WB, both const) and stores it
    /// whole — so LLVM's SROA promotes the OCG·WB accumulators to SSA registers
    /// and keeps them resident across the runtime `(kh, icg)` reduction (this is
    /// what makes it ~2.4× faster than a runtime-length-access variant, matching
    /// the standalone microbench). `get_unchecked` keeps the runtime-derived
    /// input/kernel/output indices bounds-check-free; all are provably in range
    /// from the shape invariants. The `w % WB` remainder uses a scalar tail.
    // Index loops are deliberate here: const offsets into `acc` are what let SROA
    // keep the accumulators register-resident; iterator forms regressed codegen.
    #[allow(clippy::needless_range_loop)]
    fn run<const OCG: usize>(&self, x: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
        let (icg, w, h_in, h_out, kh) = (self.icg(), self.w, self.h_in, self.h_out, self.kh);
        let (sh, dh, pb) =
            (self.stride_h as isize, self.dil_h as isize, self.pad_before_h as isize);
        let kstride_oc = icg * kh; // weights row stride per output channel
        let n_full = w / WB; // full WB-wide blocks; remainder handled after
        for ni in 0..self.n {
            let x_n = &x[ni * self.c_in * h_in * w..];
            let out_n = &mut out[ni * self.oc * h_out * w..];
            for g in 0..self.group {
                let oc0 = g * OCG;
                let ic0 = g * icg;
                for oh in 0..h_out {
                    // ---- full WB blocks: all-const acc access -> register-resident ----
                    for blk in 0..n_full {
                        let wb = blk * WB;
                        let mut acc = [[0f32; WB]; OCG];
                        for ocl in 0..OCG {
                            let b = bias[oc0 + ocl];
                            for j in 0..WB {
                                acc[ocl][j] = b;
                            }
                        }
                        for kh_i in 0..kh {
                            let ih = oh as isize * sh + kh_i as isize * dh - pb;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let row0 = ((ic0 * h_in + ih as usize) * w + wb) as isize;
                            for icl in 0..icg {
                                let row_base = (row0 + (icl * h_in * w) as isize) as usize;
                                for ocl in 0..OCG {
                                    let wv = unsafe {
                                        *kernel.get_unchecked(
                                            (oc0 + ocl) * kstride_oc + icl * kh + kh_i,
                                        )
                                    };
                                    let a = &mut acc[ocl];
                                    for j in 0..WB {
                                        a[j] += unsafe { *x_n.get_unchecked(row_base + j) } * wv;
                                    }
                                }
                            }
                        }
                        for ocl in 0..OCG {
                            let ob = ((oc0 + ocl) * h_out + oh) * w + wb;
                            for j in 0..WB {
                                unsafe { *out_n.get_unchecked_mut(ob + j) = acc[ocl][j] };
                            }
                        }
                    }
                    // ---- remainder (w % WB != 0): scalar tail accumulated in place ----
                    let wb = n_full * WB;
                    if wb < w {
                        let rem = w - wb;
                        for ocl in 0..OCG {
                            let b = bias[oc0 + ocl];
                            let ob = ((oc0 + ocl) * h_out + oh) * w + wb;
                            for j in 0..rem {
                                out_n[ob + j] = b;
                            }
                        }
                        for kh_i in 0..kh {
                            let ih = oh as isize * sh + kh_i as isize * dh - pb;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for icl in 0..icg {
                                let row_base = ((ic0 + icl) * h_in + ih) * w + wb;
                                for ocl in 0..OCG {
                                    let wv = kernel[(oc0 + ocl) * kstride_oc + icl * kh + kh_i];
                                    let ob = ((oc0 + ocl) * h_out + oh) * w + wb;
                                    for j in 0..rem {
                                        out_n[ob + j] += x_n[row_base + j] * wv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Generic fallback for `ocg` outside the const-dispatched set. Correct but
    /// not register-blocked (heap accumulators). Rarely hit for the eligible class.
    #[allow(clippy::needless_range_loop)]
    fn run_generic(&self, x: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
        let (icg, ocg, w, h_in, h_out, kh) =
            (self.icg(), self.ocg(), self.w, self.h_in, self.h_out, self.kh);
        let (sh, dh, pb) =
            (self.stride_h as isize, self.dil_h as isize, self.pad_before_h as isize);
        let kstride_oc = icg * kh;
        let mut acc = vec![0f32; ocg * w];
        for ni in 0..self.n {
            let x_n = &x[ni * self.c_in * h_in * w..];
            let out_n = &mut out[ni * self.oc * h_out * w..];
            for g in 0..self.group {
                let oc0 = g * ocg;
                let ic0 = g * icg;
                for oh in 0..h_out {
                    for ocl in 0..ocg {
                        let b = bias[oc0 + ocl];
                        for j in 0..w {
                            acc[ocl * w + j] = b;
                        }
                    }
                    for kh_i in 0..kh {
                        let ih = oh as isize * sh + kh_i as isize * dh - pb;
                        if ih < 0 || ih >= h_in as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        for icl in 0..icg {
                            let ic = ic0 + icl;
                            let row = &x_n[(ic * h_in + ih) * w..(ic * h_in + ih) * w + w];
                            for ocl in 0..ocg {
                                let wv = kernel[(oc0 + ocl) * kstride_oc + icl * kh + kh_i];
                                let a = &mut acc[ocl * w..ocl * w + w];
                                for j in 0..w {
                                    a[j] += row[j] * wv;
                                }
                            }
                        }
                    }
                    for ocl in 0..ocg {
                        let ob = ((oc0 + ocl) * h_out + oh) * w;
                        out_n[ob..ob + w].copy_from_slice(&acc[ocl * w..ocl * w + w]);
                    }
                }
            }
        }
    }
}

impl TypedOp for BlockedConv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "BlockedConv expects 3 inputs (X, kernel, bias)");
        Ok(tvec!(f32::datum_type().fact([self.n, self.oc, self.h_out, self.w])))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let macs = self.n * self.oc * self.h_out * self.w * self.icg() * self.kh;
        Ok(tvec!((Cost::FMA(f32::datum_type()), macs.to_dim())))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent scalar reference for the eligible conv class (NCHW, kw=1,
    /// unit stride/dilation on W). `kernel` is `[oc, icg*kh]` (group-major,
    /// i-major/h-minor); input channel for output `oc` is `(oc/ocg)*icg + icl`.
    #[allow(clippy::too_many_arguments)]
    fn reference(op: &BlockedConv, x: &[f32], kernel: &[f32], bias: &[f32]) -> Vec<f32> {
        let (icg, ocg) = (op.icg(), op.ocg());
        let (h_in, w, kh) = (op.h_in, op.w, op.kh);
        let (sh, dh, pb) = (op.stride_h as isize, op.dil_h as isize, op.pad_before_h as isize);
        let mut out = vec![0f32; op.n * op.oc * op.h_out * w];
        for ni in 0..op.n {
            for oc in 0..op.oc {
                let g = oc / ocg;
                for oh in 0..op.h_out {
                    for wi in 0..w {
                        let mut acc = bias[oc];
                        for kh_i in 0..kh {
                            let ih = oh as isize * sh + kh_i as isize * dh - pb;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            for icl in 0..icg {
                                let ic = g * icg + icl;
                                let xv = x[((ni * op.c_in + ic) * h_in + ih) * w + wi];
                                acc += xv * kernel[oc * (icg * kh) + icl * kh + kh_i];
                            }
                        }
                        out[((ni * op.oc + oc) * op.h_out + oh) * w + wi] = acc;
                    }
                }
            }
        }
        out
    }

    fn run_case(c_in: usize, oc: usize, group: usize, kh: usize, h_in: usize, w: usize, pb: usize) {
        let icg = c_in / group;
        let h_out = h_in + pb - (kh - 1); // stride=dil=1, pad_after=0
        let op = BlockedConv {
            n: 1,
            c_in,
            h_in,
            w,
            oc,
            group,
            kh,
            stride_h: 1,
            dil_h: 1,
            pad_before_h: pb,
            h_out,
        };
        let x: Vec<f32> = (0..c_in * h_in * w).map(|i| ((i as f32 * 0.137).sin()) * 0.7).collect();
        let kernel: Vec<f32> =
            (0..oc * icg * kh).map(|i| ((i as f32 * 0.091).cos()) * 0.3).collect();
        let bias: Vec<f32> = (0..oc).map(|i| (i as f32 * 0.05) - 0.1).collect();

        let want = reference(&op, &x, &kernel, &bias);
        let got = op
            .eval(tvec![
                Tensor::from_shape(&[1, c_in, h_in, w], &x).unwrap().into_tvalue(),
                Tensor::from_shape(&[oc, icg * kh], &kernel).unwrap().into_tvalue(),
                Tensor::from_shape(&[oc], &bias).unwrap().into_tvalue(),
            ])
            .unwrap();
        let got_view = got[0].to_plain_array_view::<f32>().unwrap();
        let got = got_view.as_slice().unwrap();
        assert_eq!(got.len(), want.len());
        let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(
            max_abs < 1e-5,
            "BlockedConv mismatch (c_in={c_in} oc={oc} g={group} kh={kh} h={h_in} w={w} pb={pb}): max_abs={max_abs}"
        );
    }

    #[test]
    fn blocked_conv_matches_reference() {
        // df_convp.1-like: group=2, ocg=5, kh=5, causal pad, w multiple of WB.
        run_case(64, 10, 2, 5, 12, 96, 4);
        // full block + remainder (w=20 = 16 + 4), ocg=2.
        run_case(4, 4, 2, 3, 5, 20, 1);
        // remainder-only (w=5 < WB), ocg=3.
        run_case(8, 6, 2, 4, 7, 5, 2);
        // group=1, ocg=3, no padding.
        run_case(6, 3, 1, 3, 8, 33, 0);
        // ocg=1 edge.
        run_case(4, 2, 2, 2, 6, 17, 1);
    }
}
