use crate::mmm::PackedExoticFact;

use super::*;
use num_traits::{AsPrimitive, Float, Zero};
use std::alloc::Layout;

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct BaseQ4_0<const QK: usize = 32>;

pub const Q4_0: BaseQ4_0 = BaseQ4_0::<32>;

impl<const QK: usize> Debug for BaseQ4_0<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if QK == 32 { write!(f, "Q4_0") } else { write!(f, "BaseQ4_0<{QK}>") }
    }
}

impl<const QK: usize> BaseQ4_0<QK> {
    fn quant_block<T>(&self, block: &[T], quant: &mut [u8])
    where
        f32: AsPrimitive<i8> + From<T>,
        T: Debug + Float,
    {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut writer = NibbleWriter::for_slice(quant);
        let mut amax = T::zero();
        let mut max = T::zero();
        for v in block {
            if amax < v.abs() {
                amax = v.abs();
                max = *v;
            }
        }
        let scale = f32::from(max) / -8f32;
        let r_scale = if scale.is_zero() { 0f32 } else { scale.recip() };
        writer.write_f16(f16::from_f32(scale));

        for idx in 0..block.len() {
            // Quant block in GGML nibble order
            let ggml_idx = (block.len() / 2) * (idx % 2) + (idx / 2);
            let i: i8 = (f32::from(block[ggml_idx]) * r_scale + 8.5f32).as_();
            writer.write_i4(i.min(15));
        }
    }

    fn dequant_block<T: Float + 'static>(&self, quant: &[u8], block: &mut [T])
    where
        f16: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut nibbles = NibbleReader::for_slice(quant);
        let d: T = nibbles.read_f16().as_();
        for idx in 0..block.len() {
            let ggml_idx = (block.len() / 2) * (idx % 2) + (idx / 2);
            block[ggml_idx] = (nibbles.read_i4() - 8).as_() * d;
        }
    }

    unsafe fn extract_panel_t<T: Float + Debug + 'static>(
        &self,
        value: &EagerPackedInput,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()>
    where
        f16: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        let pbqf: &PackedBlockQuantFormat =
            value.fact.format.downcast_ref().with_context(|| {
                format!("Expecing PackedBlockQuantFormat, found {:?}", value.fact.format)
            })?;
        ensure!(pbqf.r == target.r);
        ensure!(value.fact.k % self.block_len() == 0);
        ensure!(*pbqf.bq == *(self as &dyn BlockQuant));
        let scratch =
            unsafe { std::slice::from_raw_parts_mut(scratch as *mut T, value.fact.k * target.r) };
        let blocks_for_k = value.fact.k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let input = &value.packed[panel * target.r * row_bytes..];
        let mut scales = vec![T::zero(); target.r];
        let mut scratch = scratch.iter_mut();
        let zipped_order = zipped_order(pbqf.r, pbqf.zip);
        let mut weights = vec![0i8; pbqf.r];
        let panel_block_bytes = target.r * self.block_bytes();
        let (scale_offset, weights_offset) = if pbqf.scales_at_end {
            (panel_block_bytes - target.r * f16::datum_type().size_of(), 0)
        } else {
            (0, target.r * f16::datum_type().size_of())
        };
        for block in 0..blocks_for_k {
            let block = &input[block * panel_block_bytes..][..panel_block_bytes];
            let mut s_reader = NibbleReader::for_slice(&block[scale_offset..]);
            let mut w_reader = NibbleReader::for_slice(&block[weights_offset..]);
            for s in &mut scales {
                *s = s_reader.read_f16().as_();
            }
            for _ in 0..self.block_len() {
                for &o in &zipped_order {
                    weights[o] = w_reader.read_i4();
                }
                for (w, s) in weights.iter().zip(scales.iter()) {
                    *scratch.next().unwrap() = *s * (*w - 8).as_();
                }
            }
        }
        Ok(())
    }

    fn extract_at_mn_t<T: Float + Debug + 'static>(
        &self,
        value: &EagerPackedInput,
        mn: usize,
        target: &mut [T],
    ) -> TractResult<()>
    where
        f16: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        let pbqf: &PackedBlockQuantFormat =
            value.fact.format.downcast_ref().with_context(|| {
                format!("Expecing PackedBlockQuantFormat, found {:?}", value.fact.format)
            })?;
        ensure!(value.fact.k % self.block_len() == 0);
        ensure!(*pbqf.bq == *(self as &dyn BlockQuant));
        ensure!(value.fact.mn.to_usize().ok().map(|it| mn < it).unwrap_or(true));
        ensure!(value.fact.k == target.len());
        let blocks_for_k = value.fact.k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let panel = mn / pbqf.r;
        let value = &value.packed[panel * pbqf.r * row_bytes..];
        let mut target = target.iter_mut();
        let zipped_order =
            zipped_order(pbqf.r, pbqf.zip).iter().position(|x| *x == mn % pbqf.r).unwrap();

        let panel_block_bytes = pbqf.r * self.block_bytes();
        let (scale_offset, weights_offset) = if pbqf.scales_at_end {
            (panel_block_bytes - pbqf.r * f16::datum_type().size_of(), 0)
        } else {
            (0, pbqf.r * f16::datum_type().size_of())
        };
        unsafe {
            for block in 0..blocks_for_k {
                let block = value.as_ptr().add(block * panel_block_bytes);
                let scale = *((block.add(scale_offset) as *const f16).add(mn % pbqf.r));
                let scale: T = scale.as_();
                for i in 0..self.block_len() {
                    let byte = *block.add(weights_offset + i * pbqf.r / 2 + zipped_order / 2);
                    let nib = if zipped_order % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    *target.next().unwrap() = scale * ((nib as i8) - 8).as_();
                }
            }
        }
        Ok(())
    }

    /// Quantize one activation row to symmetric int8 with a per-block f32 scale (block length
    /// `QK`): `scale_b = max(|a| in block) / 127`, `q[i] = round(a[i] / scale_b)`. Returns
    /// `(q, scales)` with `q.len() == a.len()` and `scales.len() == a.len() / QK`.
    fn quantize_row_q8(&self, a: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let nb = a.len() / QK;
        let mut q = vec![0i8; a.len()];
        let mut scales = vec![0f32; nb];
        for b in 0..nb {
            let blk = &a[b * QK..][..QK];
            let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scales[b] = scale;
            let r = scale.recip();
            for i in 0..QK {
                q[b * QK + i] = (blk[i] * r).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (q, scales)
    }

    /// W4A8 GEMV (`M == 1`): `y[n] = Σ_k dequant(W[n, k]) · a[k]`, computed as per-block int8 dot
    /// products of the unpacked 4-bit weights with an int8-quantized activation, each scaled by
    /// `w_block_scale · a_block_scale`. `weight` is this format's storage for a `[n, k]` matrix
    /// (row-major blocks, `k % QK == 0`); `a` is the length-`k` activation, quantized to int8 on
    /// the fly, so the result is near- but not bit-exact against the f32-dequant path.
    pub fn w4a8_gemv(&self, weight: &[u8], n: usize, k: usize, a: &[f32]) -> TractResult<Vec<f32>> {
        ensure!(k % QK == 0, "W4A8 GEMV needs K a multiple of {QK}, got {k}");
        ensure!(a.len() == k && weight.len() == n * (k / QK) * self.block_bytes());
        let nb = k / QK;
        let (aq, asc) = self.quantize_row_q8(a);
        let bb = self.block_bytes();
        let mut y = vec![0f32; n];
        let mut wi8 = [0i8; QK];
        for ni in 0..n {
            let mut acc = 0f32;
            for b in 0..nb {
                let blk = &weight[(ni * nb + b) * bb..][..bb];
                let wscale = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                for idx in 0..QK {
                    let byte = blk[2 + idx / 2];
                    let nib = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 } as i8;
                    wi8[(QK / 2) * (idx % 2) + idx / 2] = nib - 8;
                }
                let aqb = &aq[b * QK..][..QK];
                // Keep this contiguous so LLVM lowers the i8->i32 MAC to a NEON int8 dot.
                let mut dot = 0i32;
                for i in 0..QK {
                    dot += wi8[i] as i32 * aqb[i] as i32;
                }
                acc += dot as f32 * wscale * asc[b];
            }
            y[ni] = acc;
        }
        Ok(y)
    }

    /// W4A8 GEMM (`M >= 1`): `Y[m, n] = Σ_k dequant(W[n, k]) · A[m, k]`, computed as per-block
    /// int8 dot products of the unpacked 4-bit weights with int8-quantized activations, each
    /// scaled by `w_block_scale · a_block_scale`. `weight` is this format's storage for a `[n, k]`
    /// matrix (row-major blocks, `k % QK == 0`); `a` is the `[m, k]` row-major activation,
    /// quantized to int8 on the fly; `out` is the `[m, n]` row-major result, overwritten. Each
    /// weight block is decoded once and reused across the `m` activation rows, so the result is
    /// near- but not bit-exact against the f32-dequant path (and bit-exact against `w4a8_gemv`
    /// run per row).
    pub fn w4a8_gemm(
        &self,
        weight: &[u8],
        n: usize,
        k: usize,
        a: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> TractResult<()> {
        ensure!(k % QK == 0, "W4A8 GEMM needs K a multiple of {QK}, got {k}");
        ensure!(a.len() == m * k && weight.len() == n * (k / QK) * self.block_bytes());
        ensure!(out.len() == m * n);
        let nb = k / QK;
        let bb = self.block_bytes();

        let mut aq = vec![0i8; m * k];
        let mut asc = vec![0f32; m * nb];
        for mi in 0..m {
            let (q, s) = self.quantize_row_q8(&a[mi * k..][..k]);
            aq[mi * k..][..k].copy_from_slice(&q);
            asc[mi * nb..][..nb].copy_from_slice(&s);
        }

        #[cfg(target_arch = "aarch64")]
        let n_done = if QK == 32 && std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: guarded by the FEAT_DotProd check; QK == 32.
            unsafe { w4a8_gemm_sdot_4(weight, n, k, &aq, &asc, m, out) }
        } else {
            0
        };
        #[cfg(not(target_arch = "aarch64"))]
        let n_done = 0usize;

        let mut wi8 = [0i8; QK];
        let mut accs = vec![0f32; m];
        for ni in n_done..n {
            accs.iter_mut().for_each(|x| *x = 0.0);
            for b in 0..nb {
                let blk = &weight[(ni * nb + b) * bb..][..bb];
                let wscale = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                for idx in 0..QK {
                    let byte = blk[2 + idx / 2];
                    let nib = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 } as i8;
                    wi8[(QK / 2) * (idx % 2) + idx / 2] = nib - 8;
                }
                for mi in 0..m {
                    let aqb = &aq[mi * k + b * QK..][..QK];
                    // Keep this contiguous so LLVM lowers the i8->i32 MAC to a NEON int8 dot.
                    let mut dot = 0i32;
                    for i in 0..QK {
                        dot += wi8[i] as i32 * aqb[i] as i32;
                    }
                    accs[mi] += dot as f32 * wscale * asc[mi * nb + b];
                }
            }
            for mi in 0..m {
                out[mi * n + ni] = accs[mi];
            }
        }
        Ok(())
    }
}

/// W4A8 GEMM fast path for `QK == 32` on FEAT_DotProd CPUs. Processes 4 output columns at a time:
/// the 4 columns' k-values are packed so each by-element `SDOT` accumulates the four columns' dot
/// products directly into the four accumulator lanes, so there is no per-block horizontal reduce.
/// `aq`/`asc` are the pre-quantized activation (`[m, k]` int8) and its per-block scales. Returns
/// the number of columns handled (a multiple of 4); the caller does the `n % 4` tail in scalar.
///
/// SAFETY: caller must guarantee the running CPU has FEAT_DotProd.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn w4a8_gemm_sdot_4(
    weight: &[u8],
    n: usize,
    k: usize,
    aq: &[i8],
    asc: &[f32],
    m: usize,
    out: &mut [f32],
) -> usize {
    use std::arch::aarch64::*;
    const QK: usize = 32;
    let nb = k / QK;
    let bb = 18; // Q4_0 block: 2-byte f16 scale + 16 nibble bytes
    let n4 = n & !3;
    let mut wpack = [0i8; 8 * 16]; // 8 k-groups x [4 cols x 4 k]
    let mut wsc = [0f32; 4];
    let mut accs = vec![0f32; m * 4];
    for nt in (0..n4).step_by(4) {
        accs.iter_mut().for_each(|x| *x = 0.0);
        for b in 0..nb {
            for c in 0..4 {
                let blk = &weight[((nt + c) * nb + b) * bb..][..bb];
                wsc[c] = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                for idx in 0..QK {
                    let byte = blk[2 + idx / 2];
                    let nib = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 } as i8;
                    let elem = (QK / 2) * (idx % 2) + idx / 2;
                    wpack[(elem / 4) * 16 + c * 4 + (elem % 4)] = nib - 8;
                }
            }
            let wsc_v = vld1q_f32(wsc.as_ptr());
            let w: [int8x16_t; 8] = core::array::from_fn(|i| vld1q_s8(wpack[i * 16..].as_ptr()));
            for mi in 0..m {
                let ap = aq[mi * k + b * QK..].as_ptr();
                let a0 = vld1q_s8(ap);
                let a1 = vld1q_s8(ap.add(16));
                let mut acc = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {a0:v}.4b[0]",
                    "sdot {acc:v}.4s, {w1:v}.16b, {a0:v}.4b[1]",
                    "sdot {acc:v}.4s, {w2:v}.16b, {a0:v}.4b[2]",
                    "sdot {acc:v}.4s, {w3:v}.16b, {a0:v}.4b[3]",
                    "sdot {acc:v}.4s, {w4:v}.16b, {a1:v}.4b[0]",
                    "sdot {acc:v}.4s, {w5:v}.16b, {a1:v}.4b[1]",
                    "sdot {acc:v}.4s, {w6:v}.16b, {a1:v}.4b[2]",
                    "sdot {acc:v}.4s, {w7:v}.16b, {a1:v}.4b[3]",
                    acc = inout(vreg) acc,
                    w0 = in(vreg) w[0], w1 = in(vreg) w[1], w2 = in(vreg) w[2], w3 = in(vreg) w[3],
                    w4 = in(vreg) w[4], w5 = in(vreg) w[5], w6 = in(vreg) w[6], w7 = in(vreg) w[7],
                    a0 = in(vreg) a0, a1 = in(vreg) a1,
                    options(pure, nomem, nostack),
                );
                let scaled = vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(acc), wsc_v), asc[mi * nb + b]);
                let cur = vld1q_f32(accs[mi * 4..].as_ptr());
                vst1q_f32(accs[mi * 4..].as_mut_ptr(), vaddq_f32(cur, scaled));
            }
        }
        for mi in 0..m {
            out[mi * n + nt..mi * n + nt + 4].copy_from_slice(&accs[mi * 4..mi * 4 + 4]);
        }
    }
    n4
}

fn zipped_order(r: usize, zip: usize) -> Vec<usize> {
    if zip == 0 {
        (0..r).collect_vec()
    } else {
        (0..r)
            .map(|i| {
                let vec_pair_ix = i / (2 * zip);
                let lane = (i % (2 * zip)) / 2;
                let side = i % 2;
                vec_pair_ix * 2 * zip + side * zip + lane
            })
            .collect_vec()
    }
}

impl<const QK: usize> BlockQuant for BaseQ4_0<QK> {
    fn block_len(&self) -> usize {
        QK
    }

    fn block_bytes(&self) -> usize {
        2 + self.block_len() / 2
    }

    fn quant_block_f32(&self, block: &[f32], quant: &mut [u8]) {
        self.quant_block(block, quant)
    }

    fn quant_block_f16(&self, block: &[f16], quant: &mut [u8]) {
        self.quant_block(block, quant)
    }

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
        self.dequant_block(quant, block)
    }

    fn dequant_block_f16(&self, quant: &[u8], block: &mut [f16]) {
        self.dequant_block(quant, block)
    }

    // s0_0 n0_0 n0_1 n0_2 n0_3 ... n0_30n0_31 s0_32 n0_32n0_33 ...
    // s1_0 n1_0 n1_1 n1_2 n1_3 ... n1_30n1_31 s1_32 n1_32n1_33 ...
    //
    //  becomes (with r=4)
    //
    //  s0_0  s1_0  s2_0  s3_0  n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
    //  s0_32 s1_32 s2_32 s3_32 n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
    //  ...
    fn pack(
        &self,
        input: &[u8],
        k: usize,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> TractResult<EagerPackedInput> {
        ensure!(input.len() % self.block_bytes() == 0);
        ensure!(k % self.block_len() == 0);
        // ensure!(input.len() == k * r / self.block_len() * self.block_bytes());
        ensure!(zip < r);
        let m = if input.len() == 0 {
            0
        } else {
            input.len() / self.block_bytes() * self.block_len() / k
        };
        let panels = m.divceil(r);
        let blocks_for_k = k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let panel_bytes = row_bytes * r;
        let mut blob =
            unsafe { Blob::for_layout(Layout::from_size_align(panel_bytes * panels, 128)?) };
        let mut writer = NibbleWriter::for_slice(&mut blob);
        let order = zipped_order(r, zip);
        let mut scales = vec![f16::zero(); r];
        for p in 0..panels {
            let input = &input[(r * p) * row_bytes..];
            let mut readers = (0..r)
                .map(|r| {
                    // manage partial panel
                    let offset = if r * row_bytes < input.len() { r * row_bytes } else { 0 };
                    NibbleReader::for_slice(&input[offset..])
                })
                .collect_vec();
            let mut temp_nibbles = vec![vec![0i8; self.block_len()]; r];
            for _ in 0..blocks_for_k {
                for (row, reader) in readers.iter_mut().enumerate() {
                    scales[row] = reader.read_f16();
                    temp_nibbles[row] =
                        (0..self.block_len()).map(|_| reader.read_i4()).collect_vec();
                }
                if !scales_at_end {
                    scales.iter().for_each(|s| writer.write_f16(*s))
                }
                for pos in 0..self.block_len() {
                    for &row in &order {
                        let ggml_idx = pos / (self.block_len() / 2) + (2 * pos) % self.block_len();
                        let nib = temp_nibbles[row][ggml_idx];
                        writer.write_i4(nib);
                    }
                }
                if scales_at_end {
                    scales.iter().for_each(|s| writer.write_f16(*s))
                }
            }
        }
        Ok(EagerPackedInput {
            fact: PackedExoticFact {
                format: Box::new(PackedBlockQuantFormat {
                    bq: Box::new(*self),
                    r,
                    zip,
                    scales_at_end,
                }),
                mn: m.to_dim(),
                k,
            },
            packed: blob.into(),
            panel_bytes,
            mn: m,
        })
    }

    unsafe fn extract_packed_panel(
        &self,
        value: &EagerPackedInput,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()> {
        unsafe {
            dispatch_floatlike!(Self::extract_panel_t(target.dt)(
                self, value, target, panel, scratch
            ))
        }
    }

    fn extract_at_mn_f16(
        &self,
        value: &EagerPackedInput,
        mn: usize,
        target: &mut [f16],
    ) -> TractResult<()> {
        self.extract_at_mn_t(value, mn, target)
    }

    fn extract_at_mn_f32(
        &self,
        value: &EagerPackedInput,
        mn: usize,
        target: &mut [f32],
    ) -> TractResult<()> {
        self.extract_at_mn_t(value, mn, target)
    }
}

impl<const QK: usize> Display for BaseQ4_0<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q4_0")
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use tract_data::internal::tract_ndarray::Array2;

    use crate::pack::PackedFormat;

    use super::*;

    fn test_loop_f32(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.to_vec();
        while input.len() % b.block_len() != 0 {
            input.push(0f32);
        }
        let quant = b.quant_f32(&input).unwrap();
        let result = b.dequant_f32(&quant).unwrap();
        let view = &result.try_as_plain().unwrap().as_slice::<f32>().unwrap()[..data.len()];
        assert_eq!(data, view);
    }

    fn test_loop_f16(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.iter().map(|f| f16::from_f32(*f)).collect_vec();
        while input.len() % b.block_len() != 0 {
            input.push(f16::zero());
        }
        let quant = b.quant_f16(&input).unwrap();
        let result = b.dequant_f16(&quant).unwrap();
        let view = &result.try_as_plain().unwrap().as_slice::<f16>().unwrap();
        assert_eq!(&input, view);
    }

    #[test]
    fn loop_q4f32_pos() {
        test_loop_f32(Q4_0, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn loop_q4f16_pos() {
        test_loop_f16(Q4_0, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn loop_q4f32_neg() {
        test_loop_f32(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4f16_neg() {
        test_loop_f16(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_pos() {
        test_loop_f32(Q4_0, &[1234.0]);
        test_loop_f16(Q4_0, &[1234.0]);
    }

    #[test]
    fn loop_q4_big_neg() {
        test_loop_f32(Q4_0, &[-1234.0]);
        test_loop_f16(Q4_0, &[-1234.0]);
    }

    #[test]
    #[ignore = "perf: cargo test -p tract-linalg --release w4a8_gemm_bench -- --ignored --nocapture"]
    fn w4a8_gemm_bench() -> TractResult<()> {
        use std::time::Instant;
        let (m, n, k) = (64usize, 1536usize, 384usize);
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 53) as f32 - 26.0) / 9.0).collect();
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 19 % 41) as f32 - 20.0) / 7.0).collect();
        let q = Q4_0.quant_f32(&w)?;
        let mut out = vec![0f32; m * n];
        for _ in 0..5 {
            Q4_0.w4a8_gemm(&q, n, k, &a, m, &mut out)?;
        }
        let iters = 300;
        let t = Instant::now();
        for _ in 0..iters {
            Q4_0.w4a8_gemm(&q, n, k, &a, m, &mut out)?;
        }
        eprintln!(
            "w4a8_gemm prefill m={m} n={n} k={k}: {} us/iter",
            t.elapsed().as_micros() as usize / iters
        );
        Ok(())
    }

    #[test]
    fn w4a8_gemm_matches_gemv() -> TractResult<()> {
        let (m, n, k) = (5usize, 7usize, 64usize);
        let wf: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 53) as f32 - 26.0) / 9.0).collect();
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 19 % 41) as f32 - 20.0) / 7.0).collect();
        let qbytes = Q4_0.quant_f32(&wf)?;
        let mut out = vec![0f32; m * n];
        Q4_0.w4a8_gemm(&qbytes, n, k, &a, m, &mut out)?;
        for mi in 0..m {
            let row = Q4_0.w4a8_gemv(&qbytes, n, k, &a[mi * k..][..k])?;
            assert_eq!(&out[mi * n..][..n], &row[..], "row {mi}");
        }
        Ok(())
    }

    #[test]
    fn w4a8_gemm_approx_dequant_matmul() -> TractResult<()> {
        let (m, n, k) = (4usize, 6usize, 96usize);
        let wf: Vec<f32> = (0..n * k).map(|i| ((i * 31 % 47) as f32 - 23.0) / 11.0).collect();
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 23 % 43) as f32 - 21.0) / 8.0).collect();
        let qbytes = Q4_0.quant_f32(&wf)?;
        let mut out = vec![0f32; m * n];
        Q4_0.w4a8_gemm(&qbytes, n, k, &a, m, &mut out)?;
        let wdeq = Q4_0.dequant_f32(&qbytes)?;
        let wdeq = wdeq.try_as_plain()?.as_slice::<f32>()?;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += wdeq[ni * k + ki] * a[mi * k + ki];
                }
                let got = out[mi * n + ni];
                assert!(
                    (got - acc).abs() <= 0.15 * acc.abs() + 1.0,
                    "[{mi},{ni}] got {got} ref {acc}"
                );
            }
        }
        Ok(())
    }

    fn test_extract_f32(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.to_vec();
        while input.len() % b.block_len() != 0 {
            input.push(0f32);
        }
        let quant = b.quant_f32(&input).unwrap();
        for (ix, v) in data.iter().enumerate() {
            assert_eq!(b.extract_at_offset_f32(&quant, ix).round(), *v);
        }
    }

    #[test]
    fn extract_q40f32_pos() {
        let data = (1..).map(|i| ((i % 14) - 6) as f32).take(5 * Q4_0.block_len()).collect_vec();
        test_extract_f32(Q4_0, &data);
    }

    fn test_pack_then_extract_panel(
        q: impl BlockQuant,
        k: usize,
        m: usize,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> TractResult<()> {
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor();
        let weights_f32 = q
            .dequant_f32(&q.quant_f32(weights_orig.try_as_plain()?.as_slice::<f32>()?)?)?
            .into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let q4 = q.quant_f32(weights_f32.try_as_plain()?.as_slice::<f32>()?)?;
        let packed_q4 = q.pack(&q4, k, r, zip, scales_at_end)?;

        for panel in 0..packed_f32.panels_count() {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(panel, None)?;
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                let mut panel_q4 = Tensor::zero::<f32>(&[k * r])?;
                q.extract_packed_panel(
                    &packed_q4,
                    &packer,
                    panel,
                    panel_q4.as_bytes_mut().as_mut_ptr(),
                )?;
                assert_eq!(panel_q4.try_as_plain()?.as_slice::<f32>()?, panel_f32);
            }
        }
        Ok(())
    }

    #[test]
    fn pack_then_extract_panel() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ4_0::<2>, 4, 4, 2, 0, false)
    }

    #[test]
    fn pack_then_extract_panel_with_zip() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ4_0::<2>, 2, 8, 8, 4, false)
    }

    #[test]
    fn pack_then_extract_panel_with_scales_at_end() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ4_0::<2>, 2, 4, 4, 0, true)
    }

    fn test_pack_then_extract_row(
        q: impl BlockQuant,
        k: usize,
        m: usize,
        r: usize,
        zip: usize,
        scales_at_end: bool,
    ) -> TractResult<()> {
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor();
        let weights_f32 = q
            .dequant_f32(&q.quant_f32(weights_orig.try_as_plain()?.as_slice::<f32>()?)?)?
            .into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let q4 = q.quant_f32(weights_f32.try_as_plain()?.as_slice::<f32>()?)?;
        let packed_q4 = q.pack(&q4, k, r, zip, scales_at_end)?;

        for row in 0..packed_f32.mn() {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(row / r, None)?;
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                let row_f32 = (0..k).map(|ix| panel_f32[row % r + r * ix]).collect_vec();

                let mut q4 = vec![0f32; k];
                q.extract_at_mn_f32(&packed_q4, row, &mut q4)?;
                assert_eq!(q4, row_f32);
            }
        }
        Ok(())
    }

    #[test]
    fn pack_then_extract_row() -> TractResult<()> {
        test_pack_then_extract_row(BaseQ4_0::<2>, 4, 4, 2, 0, false)
    }

    #[test]
    fn pack_then_extract_row_with_zip() -> TractResult<()> {
        test_pack_then_extract_row(BaseQ4_0::<2>, 2, 8, 8, 4, false)
    }

    #[test]
    fn pack_then_extract_row_with_scales_at_end() -> TractResult<()> {
        test_pack_then_extract_row(BaseQ4_0::<2>, 2, 4, 4, 0, true)
    }

    #[test]
    fn w4a8_gemv_near_f32_dequant() {
        let (n, k) = (8usize, 64usize);
        let mut rng = 1u64;
        let mut rnd = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
        let a: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let blob = Q4_0.quant_f32(&w).unwrap();
        let deq = Q4_0.dequant_f32(&blob).unwrap();
        let deq = deq.try_as_plain().unwrap();
        let deq = deq.as_slice::<f32>().unwrap();
        let mut y_ref = vec![0f32; n];
        for ni in 0..n {
            for kk in 0..k {
                y_ref[ni] += deq[ni * k + kk] * a[kk];
            }
        }
        let y = Q4_0.w4a8_gemv(&blob, n, k, &a).unwrap();
        let num: f32 = y.iter().zip(&y_ref).map(|(x, r)| (x - r).powi(2)).sum::<f32>().sqrt();
        let den: f32 = y_ref.iter().map(|r| r * r).sum::<f32>().sqrt();
        assert!(num / den.max(1e-9) < 0.02, "W4A8 vs f32-dequant GEMV deviation too large");
    }
}
