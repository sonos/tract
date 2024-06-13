use super::*;
use num_traits::{AsPrimitive, Float};
use std::alloc::Layout;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BaseQ4_0<const QK: usize = 32>;

pub static Q4_0: BaseQ4_0 = BaseQ4_0::<32>;

impl<const QK: usize> BaseQ4_0<QK> {
    fn quant_block<T>(&self, block: &[T], quant: &mut [u8])
    where
        f32: AsPrimitive<T>,
        T: Float + AsPrimitive<f16> + AsPrimitive<i8> + 'static,
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
        let d: T = max / (-8f32).as_();
        let id = if d.is_zero() { T::zero() } else { d.recip() };
        writer.write_f16(d.as_());

        for x in block {
            let i: i8 = (*x * id + (8.5f32).as_()).as_();
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
        for x in block {
            *x = (nibbles.read_i4() - 8).as_() * d;
        }
    }

    unsafe fn repack_panel_t<T: Float + 'static>(
        &self,
        value: &PackedBlockQuantValue,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()>
    where
        f16: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        ensure!(value.format.r == target.r);
        ensure!(value.k % self.block_len() == 0);
        let scratch = std::slice::from_raw_parts_mut(scratch as *mut T, value.k * target.r);
        let blocks_for_k = value.k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let mut input =
            NibbleReader::for_slice(&value.packed_block_quant_data[panel * target.r * row_bytes..]);
        let mut scales = vec![T::zero(); target.r];
        let mut scratch = scratch.iter_mut();
        for _ in 0..blocks_for_k {
            for s in &mut scales {
                *s = input.read_f16().as_();
            }
            for _ in 0..self.block_len() {
                for &s in &scales {
                    *scratch.next().unwrap() = s * (input.read_i4() - 8).as_();
                }
            }
        }
        Ok(())
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
    fn pack(&self, input: &[u8], k: usize, r: usize) -> TractResult<PackedBlockQuantValue> {
        assert!(input.len() % self.block_bytes() == 0);
        assert!(k % self.block_len() == 0);
        let m = input.len() / self.block_bytes() * self.block_len() / k;
        assert!(m % r == 0);

        let panels = m.divceil(r);
        let blocks_for_k = k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let panel_bytes = row_bytes * r;
        let mut blob =
            unsafe { Blob::for_layout(Layout::from_size_align(panel_bytes * panels, 128)?) };
        let mut writer = NibbleWriter::for_slice(&mut blob);
        for p in 0..panels {
            let input = &input[r * p * row_bytes..];
            let mut readers =
                (0..r).map(|r| NibbleReader::for_slice(&input[r * row_bytes..])).collect_vec();
            for _ in 0..blocks_for_k {
                readers.iter_mut().for_each(|r| writer.write_f16(r.read_f16()));
                for _ in 0..self.block_len() {
                    for r in &mut readers {
                        writer.write_i4(r.read_i4());
                    }
                }
            }
        }
        Ok(PackedBlockQuantValue {
            format: PackedBlockQuantFormat { bq: Box::new(*self), r },
            packed_block_quant_data: blob,
            mn: m,
            k,
        })
    }

    unsafe fn repack_panel(
        &self,
        value: &PackedBlockQuantValue,
        target: &PackedFormat,
        panel: usize,
        scratch: *mut u8,
    ) -> TractResult<()> {
        dispatch_floatlike!(Self::repack_panel_t(target.dt)(self, value, target, panel, scratch))
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

    use crate::frame::PackedFormat;

    use super::*;

    fn cycle_f32(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.to_vec();
        while input.len() % b.block_len() != 0 {
            input.push(0f32);
        }
        let quant = b.quant_f32(&input).unwrap();
        let result = b.dequant_f32(&quant).unwrap();
        let view = &result.as_slice::<f32>().unwrap()[..data.len()];
        assert_eq!(data, view);
    }

    fn cycle_f16(b: impl BlockQuant, data: &[f32]) {
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
    fn loop_q4_pos() {
        cycle_f32(Q4_0, &[1.0, 2.0, 3.0, 4.0]);
        cycle_f16(Q4_0, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn loop_q4_neg() {
        cycle_f32(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
        cycle_f16(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_pos() {
        cycle_f32(Q4_0, &[1234.0]);
        cycle_f16(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_neg() {
        cycle_f32(Q4_0, &[-1234.0]);
        cycle_f16(Q4_0, &[-1234.0]);
    }

    #[test]
    fn packing() -> TractResult<()> {
        let (q, k, m, r) = (BaseQ4_0::<2>, 4, 4, 2);
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor();
        let weights_f32 =
            q.dequant_f32(&q.quant_f32(weights_orig.as_slice::<f32>()?)?)?.into_shape(&[m, k])?;
        eprintln!("{:?}", weights_f32.to_array_view::<f32>()?);
        let packer = PackedFormat::new(f32::datum_type(), r, 128, 0);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;
        assert_eq!(packed_f32.panels_count(), 2);

        let q4 = q.quant_f32(&weights_f32.as_slice::<f32>()?)?;
        let packed_q4 = q.pack(&q4, k, r)?;

        for panel in 0..2 {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(panel, None)?;
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                eprintln!("{panel_f32:?}");
                let mut panel_q4 = Tensor::zero::<f32>(&[k * r])?;
                q.repack_panel(&packed_q4, &packer, panel, panel_q4.as_bytes_mut().as_mut_ptr())?;
                eprintln!("{panel_q4:?}");
                assert_eq!(panel_q4.as_slice::<f32>()?, panel_f32);
            }
        }
        Ok(())
    }
}
