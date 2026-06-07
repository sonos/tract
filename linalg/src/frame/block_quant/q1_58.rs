use crate::mmm::PackedExoticFact;

use super::*;
use num_traits::{AsPrimitive, Float, Zero};
use std::alloc::Layout;

/// Ternary (BitNet b1.58) block-quant: weights in {-1, 0, +1}, one f16 scale per block,
/// codes packed 2 bits each (four per byte). With the default block of 32 this is
/// `2 + 32/4 = 10` bytes per 32 weights == 2.5 bits/weight (vs Q4_0's 4.5 and f16's 16).
///
/// Codes on the wire: `-1 -> 0`, `0 -> 1`, `+1 -> 2` (code 3 unused), so a code `c`
/// dequantizes to `(c as i8 - 1) * scale`. Quantization uses abs-mean scaling
/// (`scale = mean(|w|)`), the standard BitNet b1.58 recipe.
///
/// Origin: the underlying scheme is BitNet b1.58 (Ma et al., "The Era of 1-bit LLMs:
/// All Large Language Models are in 1.58 Bits", 2024); the byte/block layout follows
/// llama.cpp's `TQ`-style ternary formats. It was identified as a transferable
/// footprint/speed win for tract while surveying the openfluke "Loom" engine
/// (its `bitnet_cpu` ternary kernels: <https://github.com/openfluke/loom>).
///
/// Prior art:
/// - BitNet b1.58 — Ma et al., "The Era of 1-bit LLMs: All Large Language Models
///   are in 1.58 Bits", 2024 (arXiv:2402.17764): the ternary + abs-mean recipe.
/// - llama.cpp ternary block formats `TQ1_0` (1.69 bpw, 5 trits/byte) and `TQ2_0`
///   (2.06 bpw, 2 bits/elem): this format is the 2-bit-packed `TQ2_0`-style layout.
/// - Microsoft `bitnet.cpp` ("1-bit AI Infra", arXiv:2410.16144) — `I2_S`/`TL1`/`TL2`
///   ternary CPU kernels — and `T-MAC` (arXiv:2407.00088) lookup-table low-bit mpGEMM.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct BaseQ1_58<const QK: usize = 32>;

pub const Q1_58: BaseQ1_58 = BaseQ1_58::<32>;

impl<const QK: usize> Debug for BaseQ1_58<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if QK == 32 { write!(f, "Q1_58") } else { write!(f, "BaseQ1_58<{QK}>") }
    }
}

impl<const QK: usize> BaseQ1_58<QK> {
    fn quant_block<T>(&self, block: &[T], quant: &mut [u8])
    where
        f32: From<T>,
        T: Debug + Float,
    {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut writer = CrumbWriter::for_slice(quant);
        let mut sum_abs = 0f32;
        for v in block {
            sum_abs += f32::from(*v).abs();
        }
        let scale = sum_abs / block.len() as f32;
        let r_scale = if scale == 0f32 { 0f32 } else { scale.recip() };
        writer.write_f16(f16::from_f32(scale));
        for v in block {
            // round-to-nearest, clamp to {-1, 0, +1}
            let q = (f32::from(*v) * r_scale).round().clamp(-1f32, 1f32) as i8;
            writer.write_crumb((q + 1) as u8);
        }
    }

    fn dequant_block<T: Float + 'static>(&self, quant: &[u8], block: &mut [T])
    where
        f16: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut reader = CrumbReader::for_slice(quant);
        let d: T = reader.read_f16().as_();
        for v in block {
            let code = reader.read_crumb();
            *v = (code as i8 - 1).as_() * d;
        }
    }

    /// Phase B helper: dequantize a block into ternary codes `{-1, 0, +1}` (as i8) plus the
    /// block's f16 scale, *without* applying the scale. This is the input an integer
    /// (int8xint8 -> i32) GEMM kernel consumes: the i32 accumulator stays exact and the
    /// scale is applied once per block in the epilogue.
    pub fn dequant_block_i8(&self, quant: &[u8], codes: &mut [i8]) -> f16 {
        assert!(quant.len() == self.block_bytes());
        assert!(codes.len() == self.block_len());
        let mut reader = CrumbReader::for_slice(quant);
        let d = reader.read_f16();
        for c in codes.iter_mut() {
            *c = reader.read_crumb() as i8 - 1;
        }
        d
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
        let mut codes = vec![0u8; pbqf.r];
        let panel_block_bytes = target.r * self.block_bytes();
        let (scale_offset, weights_offset) = if pbqf.scales_at_end {
            (panel_block_bytes - target.r * f16::datum_type().size_of(), 0)
        } else {
            (0, target.r * f16::datum_type().size_of())
        };
        for block in 0..blocks_for_k {
            let block = &input[block * panel_block_bytes..][..panel_block_bytes];
            let mut s_reader = CrumbReader::for_slice(&block[scale_offset..]);
            let mut w_reader = CrumbReader::for_slice(&block[weights_offset..]);
            for s in &mut scales {
                *s = s_reader.read_f16().as_();
            }
            for _ in 0..self.block_len() {
                for c in &mut codes {
                    *c = w_reader.read_crumb();
                }
                for (c, s) in codes.iter().zip(scales.iter()) {
                    *scratch.next().unwrap() = *s * (*c as i8 - 1).as_();
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
        let row = mn % pbqf.r;
        let panel_block_bytes = pbqf.r * self.block_bytes();
        let (scale_offset, weights_offset) = if pbqf.scales_at_end {
            (panel_block_bytes - pbqf.r * f16::datum_type().size_of(), 0)
        } else {
            (0, pbqf.r * f16::datum_type().size_of())
        };
        unsafe {
            for block in 0..blocks_for_k {
                let block = value.as_ptr().add(block * panel_block_bytes);
                let scale = *((block.add(scale_offset) as *const f16).add(row));
                let scale: T = scale.as_();
                for i in 0..self.block_len() {
                    // crumb index of (position i, this row) within the weight section
                    let ci = i * pbqf.r + row;
                    let byte = *block.add(weights_offset + ci / 4);
                    let code = (byte >> (2 * (ci % 4))) & 0x3;
                    *target.next().unwrap() = scale * (code as i8 - 1).as_();
                }
            }
        }
        Ok(())
    }
}

impl<const QK: usize> BlockQuant for BaseQ1_58<QK> {
    fn block_len(&self) -> usize {
        QK
    }

    fn block_bytes(&self) -> usize {
        // one f16 scale + QK 2-bit codes (four per byte)
        2 + self.block_len() / 4
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

    // s0_0 c0_0 c0_1 .. c0_31 s0_32 c0_32 ..        (per row: f16 scale + QK packed crumbs)
    // s1_0 c1_0 c1_1 .. c1_31 s1_32 c1_32 ..
    //
    //  becomes (with r=4, scales-first)
    //
    //  s0_0 s1_0 s2_0 s3_0  [c0_0 c1_0 c2_0 c3_0  c0_1 c1_1 c2_1 c3_1 ...]  (codes packed 4/byte)
    //  s0_32 s1_32 s2_32 s3_32  [c0_0 c1_0 c2_0 c3_0 ...]
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
        ensure!(zip == 0, "No zipping required for Q1_58");
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
        let mut writer = CrumbWriter::for_slice(&mut blob);
        let mut scales = vec![f16::zero(); r];
        for p in 0..panels {
            let input = &input[(r * p) * row_bytes..];
            let mut readers = (0..r)
                .map(|r| {
                    // manage partial panel
                    let offset = if r * row_bytes < input.len() { r * row_bytes } else { 0 };
                    CrumbReader::for_slice(&input[offset..])
                })
                .collect_vec();
            let mut temp_codes = vec![vec![0u8; self.block_len()]; r];
            for _ in 0..blocks_for_k {
                for (row, reader) in readers.iter_mut().enumerate() {
                    scales[row] = reader.read_f16();
                    temp_codes[row] =
                        (0..self.block_len()).map(|_| reader.read_crumb()).collect_vec();
                }
                if !scales_at_end {
                    scales.iter().for_each(|s| writer.write_f16(*s))
                }
                for pos in 0..self.block_len() {
                    for row in &temp_codes {
                        writer.write_crumb(row[pos]);
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

impl<const QK: usize> Display for BaseQ1_58<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q1_58")
    }
}

#[cfg(test)]
mod tests {
    use tract_data::internal::tract_ndarray::Array2;

    use crate::pack::PackedFormat;

    use super::*;

    // Ternary data the format reproduces exactly: a *full* block whose non-zero entries
    // all share one magnitude (so abs-mean == that magnitude and round(w/scale) lands on
    // {-1, +1}). Padding with zeros would shift the abs-mean, so we fill whole blocks.
    fn test_loop_exact_f32(b: impl BlockQuant, data: &[f32]) {
        assert!(data.len() % b.block_len() == 0);
        let quant = b.quant_f32(data).unwrap();
        let result = b.dequant_f32(&quant).unwrap();
        let view = result.try_as_plain().unwrap().as_slice::<f32>().unwrap();
        assert_eq!(data, view);
    }

    fn test_loop_exact_f16(b: impl BlockQuant, data: &[f32]) {
        assert!(data.len() % b.block_len() == 0);
        let input = data.iter().map(|f| f16::from_f32(*f)).collect_vec();
        let quant = b.quant_f16(&input).unwrap();
        let result = b.dequant_f16(&quant).unwrap();
        let view = result.try_as_plain().unwrap().as_slice::<f16>().unwrap();
        assert_eq!(&input, view);
    }

    // a full 32-block of alternating +/-2
    fn alt_block(mag: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| if i % 2 == 0 { mag } else { -mag }).collect()
    }

    #[test]
    fn loop_q1_58_f32_alt() {
        test_loop_exact_f32(Q1_58, &alt_block(2.0, 32));
    }

    #[test]
    fn loop_q1_58_f16_alt() {
        test_loop_exact_f16(Q1_58, &alt_block(2.0, 32));
    }

    #[test]
    fn loop_q1_58_f32_zeros() {
        test_loop_exact_f32(Q1_58, &vec![0.0; 32]);
    }

    #[test]
    fn loop_q1_58_small_block() {
        test_loop_exact_f32(BaseQ1_58::<4>, &[5.0, -5.0, 5.0, 5.0]);
    }

    // abs-mean reconstruction on a known mixed block (block_len 4, no padding).
    #[test]
    fn quant_absmean_known() {
        let q = BaseQ1_58::<4>;
        // block of 4: |.| mean = (3+1+0+2)/4 = 1.5 ; round(w/1.5) -> codes
        let data = [3.0f32, -1.0, 0.0, -2.0];
        let quant = q.quant_f32(&data).unwrap();
        let deq = q.dequant_f32(&quant).unwrap();
        let got = deq.try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec();
        // scale = 1.5 ; codes: round(3/1.5)=2->clamp 1 ; round(-1/1.5)=-1 ; 0 ; round(-2/1.5)=-1
        assert_eq!(got, vec![1.5, -1.5, 0.0, -1.5]);
    }

    fn test_pack_then_extract_panel(
        q: impl BlockQuant,
        k: usize,
        m: usize,
        r: usize,
        scales_at_end: bool,
    ) -> TractResult<()> {
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 7) as f32 - 3.)
                .into_tensor();
        // abs-mean ternary quant is not idempotent, so derive both the f32 reference and
        // the packed form from a *single* quantization `qt`.
        let qt = q.quant_f32(weights_orig.try_as_plain()?.as_slice::<f32>()?)?;
        let weights_f32 = q.dequant_f32(&qt)?.into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let packed_qt = q.pack(&qt, k, r, 0, scales_at_end)?;

        for panel in 0..packed_f32.panels_count() {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(panel, None)?;
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                let mut panel_qt = Tensor::zero::<f32>(&[k * r])?;
                q.extract_packed_panel(
                    &packed_qt,
                    &packer,
                    panel,
                    panel_qt.as_bytes_mut().as_mut_ptr(),
                )?;
                assert_eq!(panel_qt.try_as_plain()?.as_slice::<f32>()?, panel_f32);
            }
        }
        Ok(())
    }

    #[test]
    fn pack_then_extract_panel() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ1_58::<4>, 8, 4, 2, false)
    }

    #[test]
    fn pack_then_extract_panel_with_scales_at_end() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ1_58::<4>, 8, 4, 4, true)
    }

    #[test]
    fn pack_then_extract_panel_r8() -> TractResult<()> {
        test_pack_then_extract_panel(BaseQ1_58::<8>, 16, 8, 8, false)
    }

    fn test_pack_then_extract_row(
        q: impl BlockQuant,
        k: usize,
        m: usize,
        r: usize,
        scales_at_end: bool,
    ) -> TractResult<()> {
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 7) as f32 - 3.)
                .into_tensor();
        let qt = q.quant_f32(weights_orig.try_as_plain()?.as_slice::<f32>()?)?;
        let weights_f32 = q.dequant_f32(&qt)?.into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let packed_qt = q.pack(&qt, k, r, 0, scales_at_end)?;

        for row in 0..packed_f32.mn() {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(row / r, None)?;
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                let row_f32 = (0..k).map(|ix| panel_f32[row % r + r * ix]).collect_vec();

                let mut qt = vec![0f32; k];
                q.extract_at_mn_f32(&packed_qt, row, &mut qt)?;
                assert_eq!(qt, row_f32);
            }
        }
        Ok(())
    }

    #[test]
    fn pack_then_extract_row() -> TractResult<()> {
        test_pack_then_extract_row(BaseQ1_58::<4>, 8, 4, 2, false)
    }

    #[test]
    fn pack_then_extract_row_with_scales_at_end() -> TractResult<()> {
        test_pack_then_extract_row(BaseQ1_58::<4>, 8, 4, 4, true)
    }

    // ---- Phase B: integer (int8 x int8 -> i32) dot reproduces the f32 dequant path ----

    /// Per-tensor symmetric int8 activation quantization (max-abs), the BitNet b1.58 recipe.
    fn quant_activations_i8(x: &[f32]) -> (Vec<i8>, f32) {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let r = if scale == 0.0 { 0.0 } else { scale.recip() };
        (x.iter().map(|v| (v * r).round().clamp(-127., 127.) as i8).collect(), scale)
    }

    /// GEMV y[M] = W[M,K] . x[K], with W ternary-quantized (per-block f16 scale) and x
    /// int8-quantized (one tensor scale). The dot is done as i8xi8 -> i32 accumulation
    /// per block; the block weight-scale and the activation-scale are applied in the
    /// epilogue. Compares against the f32 dequant-then-matmul reference.
    #[test]
    fn phase_b_integer_gemv_matches_f32() -> TractResult<()> {
        let q = BaseQ1_58::<32>;
        let (m, k) = (6usize, 96usize);
        let bl = q.block_len();
        let blocks_for_k = k / bl;

        let weights =
            Array2::from_shape_fn((m, k), |(i, j)| (((i * 7 + j * 13) % 11) as f32 - 5.0) * 0.5);
        let x: Vec<f32> = (0..k).map(|j| ((j % 9) as f32 - 4.0) * 0.25).collect();

        // f32 reference: dequantize weights blockwise, plain matmul.
        let mut ref_y = vec![0f32; m];
        let mut block_q = vec![0u8; q.block_bytes()];
        let mut block_deq = vec![0f32; bl];
        for i in 0..m {
            let mut acc = 0f32;
            for b in 0..blocks_for_k {
                let row_block = &weights.as_slice().unwrap()[i * k + b * bl..][..bl];
                q.quant_block_f32(row_block, &mut block_q);
                q.dequant_block_f32(&block_q, &mut block_deq);
                for j in 0..bl {
                    acc += block_deq[j] * x[b * bl + j];
                }
            }
            ref_y[i] = acc;
        }

        // integer path: ternary codes (i8) x int8 activations -> i32, scaled per block.
        let (x_i8, x_scale) = quant_activations_i8(&x);
        let mut int_y = vec![0f32; m];
        let mut codes = vec![0i8; bl];
        for i in 0..m {
            let mut acc_f = 0f32;
            for b in 0..blocks_for_k {
                let row_block = &weights.as_slice().unwrap()[i * k + b * bl..][..bl];
                q.quant_block_f32(row_block, &mut block_q);
                let w_scale = q.dequant_block_i8(&block_q, &mut codes).to_f32();
                let mut acc_i32: i32 = 0;
                for j in 0..bl {
                    acc_i32 += codes[j] as i32 * x_i8[b * bl + j] as i32;
                }
                acc_f += acc_i32 as f32 * w_scale * x_scale;
            }
            int_y[i] = acc_f;
        }

        // The integer path differs from the f32 reference only by the activation int8
        // rounding (weights are bit-identical between the two paths).
        for i in 0..m {
            let err = (int_y[i] - ref_y[i]).abs();
            assert!(
                err <= 0.02 * ref_y[i].abs().max(1.0),
                "row {i}: int8={} f32={} err={}",
                int_y[i],
                ref_y[i],
                err
            );
        }
        Ok(())
    }

    // Reports two things per format: (1) the packed weight *footprint* (the unconditional
    // Tier-A win), and (2) the cost of the scalar `extract_packed_panel` unpack into the
    // f16 kernel panel.
    //
    // NOTE on interpretation: the unpack here is *compute-bound* on the scalar bit-unpack,
    // and on a typical box the packed buffer fits in L3 — so this does NOT measure the
    // DRAM-bandwidth win. (Byte-aligned Q8_1 even unpacks faster per byte than the
    // bit-packed Q1_58/Q4_0.) The Tier-A *speed* win only appears in a real decode GEMV
    // whose weights exceed last-level cache AND with a vectorized unpack (or via the
    // Tier-B integer path that skips the f16 expansion). The footprint win, by contrast,
    // is unconditional. See doc/loom-ternary-blockquant.md.
    //
    //   cargo test -p tract-linalg --lib block_quant::q1_58::tests::measure_unpack \
    //       --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure_unpack() -> TractResult<()> {
        use std::time::Instant;

        fn run(q: impl BlockQuant, name: &str, m: usize, k: usize, r: usize) -> TractResult<()> {
            let weights: Vec<f32> =
                (0..m * k).map(|i| ((i * 2654435761) % 7) as f32 - 3.0).collect();
            let quant = q.quant_f32(&weights)?;
            let packed = q.pack(&quant, k, r, 0, false)?;
            let bytes = packed.packed.len();
            let packer = PackedFormat::new(f16::datum_type(), r, 128);
            let mut scratch = Tensor::zero::<f16>(&[k * r])?;
            let panels = m.div_ceil(r);

            // warmup + timed loop
            let iters = (2_000_000_000usize / bytes.max(1)).clamp(3, 200);
            for _ in 0..2 {
                for p in 0..panels {
                    unsafe {
                        q.extract_packed_panel(
                            &packed,
                            &packer,
                            p,
                            scratch.as_bytes_mut().as_mut_ptr(),
                        )?;
                    }
                }
            }
            let t = Instant::now();
            for _ in 0..iters {
                for p in 0..panels {
                    unsafe {
                        q.extract_packed_panel(
                            &packed,
                            &packer,
                            p,
                            scratch.as_bytes_mut().as_mut_ptr(),
                        )?;
                    }
                }
            }
            let secs = t.elapsed().as_secs_f64();
            let total_bytes = bytes as f64 * iters as f64;
            println!(
                "{name:7} {m}x{k}: packed {:.2} MB ({:.3} B/w), scalar-unpack {:.2} GB/s, {:.0} us/pass",
                bytes as f64 / 1e6,
                bytes as f64 / (m * k) as f64,
                total_bytes / secs / 1e9,
                secs / iters as f64 * 1e6,
            );
            Ok(())
        }

        let (m, k, r) = (4096, 4096, 8);
        println!("--- per-matmul weight unpack (stream packed weights -> f16 panel) ---");
        run(Q1_58, "Q1_58", m, k, r)?;
        run(Q4_0, "Q4_0", m, k, r)?;
        run(Q8_1, "Q8_1", m, k, r)?;
        Ok(())
    }

    // Quantifies the SIMD unpack win: scalar `extract_packed_panel` vs the AVX2
    // `packed_32_q1_58_to_f32` extractor, both turning a Q1_58 r=32 panel into an f32
    // panel. This is the step that dominated the scalar decode GEMV.
    //
    //   cargo test -p tract-linalg --lib block_quant::q1_58::tests::measure_simd_unpack \
    //       --release -- --ignored --nocapture
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn measure_simd_unpack() -> TractResult<()> {
        use crate::pack::PackedFormat;
        use std::time::Instant;

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            println!("avx2/f16c not available, skipping");
            return Ok(());
        }
        let q = BaseQ1_58::<32>;
        let (m, k) = (4096usize, 4096usize);
        let weights: Vec<f32> = (0..m * k).map(|i| ((i * 2654435761) % 7) as f32 - 3.0).collect();
        let quant = q.quant_f32(&weights)?;
        let packed = q.pack(&quant, k, 32, 0, false)?;
        let panels = m / 32;
        let mut scratch = vec![0f32; k * 32];
        let packer = PackedFormat::new(f32::datum_type(), 32, 32);
        let simd = crate::x86_64_fma::panel_extract::packed_32_q1_58_to_f32.kernel;

        let bytes = packed.packed.len() as f64;
        let iters = 50;
        // scalar
        for _ in 0..2 {
            for p in 0..panels {
                unsafe {
                    q.extract_packed_panel(&packed, &packer, p, scratch.as_mut_ptr() as *mut u8)?;
                }
            }
        }
        let t = Instant::now();
        for _ in 0..iters {
            for p in 0..panels {
                unsafe {
                    q.extract_packed_panel(&packed, &packer, p, scratch.as_mut_ptr() as *mut u8)?;
                }
            }
        }
        let scalar = t.elapsed().as_secs_f64() / iters as f64;
        // simd
        for _ in 0..2 {
            for p in 0..panels {
                unsafe {
                    simd(
                        packed.packed.as_ptr().add(p * packed.panel_bytes),
                        scratch.as_mut_ptr() as *mut u8,
                        k,
                    );
                }
            }
        }
        let t = Instant::now();
        for _ in 0..iters {
            for p in 0..panels {
                unsafe {
                    simd(
                        packed.packed.as_ptr().add(p * packed.panel_bytes),
                        scratch.as_mut_ptr() as *mut u8,
                        k,
                    );
                }
            }
        }
        let simd_s = t.elapsed().as_secs_f64() / iters as f64;
        println!("--- Q1_58 unpack {m}x{k} ({:.1} MB packed) ---", bytes / 1e6);
        println!(
            "scalar extract_packed_panel: {:6.0} us, {:.1} GB/s",
            scalar * 1e6,
            bytes / scalar / 1e9
        );
        println!(
            "AVX2  packed_32_q1_58_to_f32: {:6.0} us, {:.1} GB/s ({:.1}x faster)",
            simd_s * 1e6,
            bytes / simd_s / 1e9,
            scalar / simd_s
        );
        Ok(())
    }

    // Decode-shaped (N=1) end-to-end GEMV through the *real* generic mmm kernel, weights
    // sized to exceed this box's L3, comparing plain f32 weights vs Q4_0 vs Q1_58. This is
    // the honest end-to-end Tier-A measurement (kernel compute + weight streaming together).
    //
    //   cargo test -p tract-linalg --lib block_quant::q1_58::tests::measure_decode_gemv \
    //       --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn measure_decode_gemv() -> TractResult<()> {
        use crate::generic::mmm::generic_f32_4x1;
        use crate::mmm::{AsInputValue, FusedSpec};
        use std::time::Instant;

        let (m, k) = (4096usize, 32768usize); // f32 512MB, Q4_0 72MB, Q1_58 40MB (all > 33MB L3)
        let mmm = generic_f32_4x1.mmm();

        let w = Tensor::zero::<f32>(&[m, k])?; // values irrelevant to timing
        let x = Tensor::zero::<f32>(&[k, 1])?;

        // (label, packing index on generic_f32_4x1, bytes/weight)
        let cases = [("f32 ", 2usize, 4.0f64), ("Q4_0", 7, 18.0 / 32.0), ("Q1_58", 9, 10.0 / 32.0)];

        println!("--- decode GEMV {m}x{k} x1 through generic_f32_4x1 (L3 = 33 MB) ---");
        for (name, packing, bpw) in cases {
            let (pa_fmt, pb_fmt) = &mmm.packings()[packing];
            let pa = pa_fmt.prepare_one(&w, 1, 0)?;
            let pb = pb_fmt.prepare_one(&x, 0, 1)?;
            let mut c = Tensor::zero::<f32>(&[m])?;
            let mb = m as f64 * k as f64 * bpw / 1e6;

            let mut once = || -> TractResult<()> {
                unsafe {
                    mmm.run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing,
                            },
                            FusedSpec::Store(mmm.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                }
            };
            once()?; // warmup
            let iters = 5;
            let t = Instant::now();
            for _ in 0..iters {
                once()?;
            }
            let ms = t.elapsed().as_secs_f64() / iters as f64 * 1e3;
            println!(
                "{name}: weights {mb:6.1} MB, {ms:7.1} ms/token, {:.1} GB/s (weight stream)",
                mb / 1e3 / (ms / 1e3),
            );
        }
        Ok(())
    }
}
