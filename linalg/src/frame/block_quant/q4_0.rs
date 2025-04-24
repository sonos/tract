use crate::mmm::PackedOpaqueFact;

use super::*;
use num_traits::{AsPrimitive, Float, Zero};
use std::alloc::Layout;

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct BaseQ4_0<const QK: usize = 32>;

pub const Q4_0: BaseQ4_0 = BaseQ4_0::<32>;

impl<const QK: usize> Debug for BaseQ4_0<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if QK == 32 {
            write!(f, "Q4_0")
        } else {
            write!(f, "BaseQ4_0<{QK}>")
        }
    }
}

impl<const QK: usize> BaseQ4_0<QK> {
    fn quant_block<T>(&self, block: &[T], quant: &mut [u8])
    where
        f32: AsPrimitive<T>,
        T: Debug + Float + AsPrimitive<f16> + AsPrimitive<i8> + 'static,
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
        let scale: T = max / (-8f32).as_();
        let r_scale = if scale.is_zero() { T::zero() } else { scale.recip() };
        writer.write_f16(scale.as_());

        for idx in 0..block.len() {
            // Quant block in GGML nibble order
            let ggml_idx = (block.len() / 2) * (idx % 2) + (idx / 2);
            let i: i8 = (block[ggml_idx] * r_scale + (8.5f32).as_()).as_();
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
        ensure!(pbqf.bq.same_as(self));
        let scratch = std::slice::from_raw_parts_mut(scratch as *mut T, value.fact.k * target.r);
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
        ensure!(pbqf.bq.same_as(self));
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
    fn same_as(&self, other: &dyn BlockQuant) -> bool {
        other.downcast_ref::<Self>().map(|other| other == self).unwrap_or(false)
    }

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
    //  s0_0 S1_0 S2_0 s3_0  n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
    //  s0_32 S1_32 S2_32 s3_32  n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
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
            fact: PackedOpaqueFact {
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
        dispatch_floatlike!(Self::extract_panel_t(target.dt)(self, value, target, panel, scratch))
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
        let view = &result.as_slice::<f32>().unwrap()[..data.len()];
        assert_eq!(data, view);
    }

    fn test_loop_f16(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.iter().map(|f| f16::from_f32(*f)).collect_vec();
        while input.len() % b.block_len() != 0 {
            input.push(f16::zero());
        }
        let quant = b.quant_f16(&input).unwrap();
        let result = b.dequant_f16(&quant).unwrap();
        let view = &result.as_slice::<f16>().unwrap();
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
    fn loop_q4f16_beg() {
        test_loop_f16(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_pos() {
        test_loop_f32(Q4_0, &[1234.0]);
        test_loop_f16(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_neg() {
        test_loop_f32(Q4_0, &[-1234.0]);
        test_loop_f16(Q4_0, &[-1234.0]);
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
        let weights_f32 =
            q.dequant_f32(&q.quant_f32(weights_orig.as_slice::<f32>()?)?)?.into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let q4 = q.quant_f32(weights_f32.as_slice::<f32>()?)?;
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
                assert_eq!(panel_q4.as_slice::<f32>()?, panel_f32);
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
        let weights_f32 =
            q.dequant_f32(&q.quant_f32(weights_orig.as_slice::<f32>()?)?)?.into_shape(&[m, k])?;
        let packer = PackedFormat::new(f32::datum_type(), r, 128);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;

        let q4 = q.quant_f32(weights_f32.as_slice::<f32>()?)?;
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
}
