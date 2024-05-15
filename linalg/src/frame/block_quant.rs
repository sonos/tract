use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use tract_data::internal::*;
use tract_data::itertools::Itertools;

use std::alloc::Layout;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::{Cursor, Read, Write};

use crate::mmm::MMMInput;

use super::Packer;

pub trait BlockQuant: Debug + Display + Clone + Send + Sync + Hash {
    fn block_len(&self) -> usize;

    fn block_bytes(&self) -> usize;

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]);
    fn quant_block(&self, block: &[f32], quant: &mut [u8]);

    fn quant(&self, input: &[f32]) -> TractResult<Blob> {
        unsafe {
            let blocks = input.len() / self.block_len();
            let mut quant = Blob::for_layout(
                Layout::from_size_align(blocks * self.block_bytes(), 128).unwrap(),
            );
            for b in 0..blocks {
                let block = &input[b * self.block_len()..][..self.block_len()];
                let qblock = &mut quant[b * self.block_bytes()..][..self.block_bytes()];
                self.quant_block(block, qblock);
            }
            Ok(quant)
        }
    }

    fn dequant_f32(&self, input: &[u8]) -> TractResult<Tensor> {
        unsafe {
            let blocks = input.len() / self.block_bytes();
            let mut tensor = Tensor::uninitialized::<f32>(&[blocks * self.block_len()])?;
            let slice = tensor.as_slice_mut::<f32>()?;
            for b in 0..blocks {
                let block = &mut slice[b * self.block_len()..][..self.block_len()];
                let qblock = &input[b * self.block_bytes()..][..self.block_bytes()];
                self.dequant_block_f32(qblock, block);
            }
            Ok(tensor)
        }
    }

    fn pack(&self, input: &[u8], k: usize, r: usize) -> TractResult<Blob>;
    fn panel_f32(&self, packed: &Blob, k: usize, r: usize, panel: usize, scratch: &mut [f32]);
}

#[derive(Copy, Clone, Debug, Hash)]
pub struct BaseQ4_0<const QK: usize = 32>;

pub static Q4_0: BaseQ4_0 = BaseQ4_0::<32>;

impl<const QK: usize> BlockQuant for BaseQ4_0<QK> {
    fn block_len(&self) -> usize {
        QK
    }

    fn block_bytes(&self) -> usize {
        2 + self.block_len() / 2
    }

    fn quant_block(&self, block: &[f32], quant: &mut [u8]) {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut writer = NibbleWriter::for_slice(quant);
        let mut amax = 0f32;
        let mut max = 0f32;
        for v in block {
            if amax < v.abs() {
                amax = v.abs();
                max = *v;
            }
        }
        let d = max / -8f32;
        let id = if d == 0.0 { 0f32 } else { d.recip() };
        writer.write_f16(f16::from_f32(d));

        for x in block {
            writer.write_i4(((*x * id + 8.5f32) as i8).min(15));
        }
    }

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut nibbles = NibbleReader::for_slice(quant);
        let d = nibbles.read_f16().to_f32();
        for x in block {
            *x = (nibbles.read_i4() - 8) as f32 * d;
        }
    }

    // s0_0 n0_0 n0_1 n0_2 n0_3 ... n0_30n0_31 s0_32 n0_32n0_33 ...
    // s1_0 n1_0 n1_1 n1_2 n1_3 ... n1_30n1_31 s1_32 n1_32n1_33 ...
    //
    //  becomes (with r=4)
    //
    //  s0_0 S1_0 S2_0 s3_0  n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
    //  s0_32 S1_32 S2_32 s3_32  n0_0 n1_0 n2_0 n3_0  n0_1 n1_1 n2_1 n3_1 ... n0_33 n1_33 n2_33 n3_33
    //  ...
    fn pack(&self, input: &[u8], k: usize, r: usize) -> TractResult<Blob> {
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
        Ok(blob)
    }

    fn panel_f32(&self, packed: &Blob, k: usize, r: usize, panel: usize, scratch: &mut [f32]) {
        assert!(k % self.block_len() == 0);
        let blocks_for_k = k / self.block_len();
        let row_bytes = blocks_for_k * self.block_bytes();
        let mut input = NibbleReader::for_slice(&packed[panel * r * row_bytes..]);
        let mut scales = vec![0f32; r];
        let mut scratch = scratch.iter_mut();
        for _ in 0..blocks_for_k {
            for s in &mut scales {
                *s = input.read_f16().to_f32();
            }
            for _ in 0..self.block_len() {
                for s in &scales {
                    *scratch.next().unwrap() = s * (input.read_i4() - 8) as f32;
                }
            }
        }
    }
}

impl<const QK: usize> Display for BaseQ4_0<QK> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q4_0")
    }
}

#[derive(Clone, Debug, Hash)]
pub struct PackedBlockQuant<BQ: BlockQuant> {
    format: BQ,
    data: Blob,
    pack: Packer,
    mn: usize,
    k: usize,
}

impl<BQ: BlockQuant> Display for PackedBlockQuant<BQ> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Packed{} (m={} k={} r={})", self.format, self.mn, self.k, self.pack.r)
    }
}

impl<BQ: BlockQuant> MMMInput for PackedBlockQuant<BQ> {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        Some(self.pack.single_panel_layout(self.k, 4))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let buffer = buffer.unwrap();
        let mut scratch = unsafe {
            std::slice::from_raw_parts_mut(buffer as *mut f32, self.pack.single_panel_len(self.k))
        };
        self.format.panel_f32(&self.data, self.k, self.pack.r, i, &mut scratch);
        buffer
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn r(&self) -> usize {
        self.pack.r
    }
    fn k(&self) -> usize {
        self.k
    }
}

pub struct NibbleReader<R> {
    second_half: Option<i8>,
    reader: R,
}

impl<'s> NibbleReader<Cursor<&'s [u8]>> {
    fn for_slice(slice: &'s [u8]) -> Self {
        NibbleReader::new(Cursor::new(slice))
    }
}

impl<R: Read> NibbleReader<R> {
    fn new(reader: R) -> NibbleReader<R> {
        NibbleReader { reader, second_half: None }
    }

    fn read_f16(&mut self) -> f16 {
        assert!(self.second_half.is_none());
        f16::from_bits(self.reader.read_u16::<LE>().unwrap())
    }

    fn read_i4(&mut self) -> i8 {
        if let Some(second) = self.second_half.take() {
            second
        } else {
            let byte = self.reader.read_u8().unwrap();
            self.second_half = Some((byte >> 4) as i8);
            (byte & 0x0F) as i8
        }
    }
}

pub struct NibbleWriter<W> {
    first_half: Option<i8>,
    writer: W,
}

impl<'s> NibbleWriter<Cursor<&'s mut [u8]>> {
    fn for_slice(slice: &'s mut [u8]) -> Self {
        NibbleWriter::new(Cursor::new(slice))
    }
}

impl<W: Write> NibbleWriter<W> {
    pub fn new(writer: W) -> NibbleWriter<W> {
        NibbleWriter { writer, first_half: None }
    }

    fn write_f16(&mut self, f: f16) {
        assert!(self.first_half.is_none());
        self.writer.write_u16::<LE>(f.to_bits()).unwrap()
    }

    fn write_i4(&mut self, q: i8) {
        if let Some(first) = self.first_half.take() {
            self.writer.write_u8(first as u8 | ((q as u8) << 4)).unwrap()
        } else {
            self.first_half = Some(q);
        }
    }
}

#[cfg(test)]
mod tests {
    use tract_data::internal::tract_ndarray::Array2;

    use crate::frame::Packer;

    use super::*;

    fn cycle(b: impl BlockQuant, data: &[f32]) {
        let mut input = data.to_vec();
        while input.len() % b.block_len() != 0 {
            input.push(0f32);
        }
        let quant = b.quant(&input).unwrap();
        let result = b.dequant_f32(&quant).unwrap();
        let view = &result.as_slice::<f32>().unwrap()[..data.len()];
        assert_eq!(data, view);
    }

    #[test]
    fn loop_q4_pos() {
        cycle(Q4_0, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn loop_q4_neg() {
        cycle(Q4_0, &[-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn loop_q4_big_pos() {
        cycle(Q4_0, &[1234.0]);
    }

    #[test]
    fn loop_q4_big_neg() {
        cycle(Q4_0, &[-1234.0]);
    }

    #[test]
    fn packing() -> TractResult<()> {
        let (q, k, m, r) = (BaseQ4_0::<2>, 4, 4, 2);
        let weights_orig =
            Array2::from_shape_fn((m, k), |(m, k)| ((m * 31 + k * 17) % 20) as f32 - 10.)
                .into_tensor();
        let weights_f32 =
            q.dequant_f32(&q.quant(weights_orig.as_slice::<f32>()?)?)?.into_shape(&[m, k])?;
        eprintln!("{:?}", weights_f32.to_array_view::<f32>()?);
        let packer = Packer::new(r, 128, 0);
        let packed_f32 = packer.pack_tensor(&weights_f32, 1, 0)?;
        assert_eq!(packed_f32.panels_count(), 2);

        let q4 = q.quant(&weights_f32.as_slice::<f32>()?)?;
        let packed_q4 = q.pack(&q4, k, r)?;

        for panel in 0..2 {
            unsafe {
                let panel_f32 = packed_f32.panel_bytes(panel, None);
                let panel_f32 = std::slice::from_raw_parts(panel_f32 as *const f32, k * r);
                eprintln!("{panel_f32:?}");
                let mut panel_q4 = Tensor::zero::<f32>(&[k * r])?;
                q.panel_f32(&packed_q4, k, r, panel, panel_q4.as_slice_mut()?);
                eprintln!("{panel_q4:?}");
                assert_eq!(panel_q4.as_slice::<f32>()?, panel_f32);
            }
        }
        Ok(())
    }
}
