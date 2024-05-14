use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use tract_data::internal::*;
use tract_data::itertools::Itertools;

use std::alloc::Layout;
use std::io::{Cursor, Read, Write};

pub trait BlockQuant {
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
}

pub struct Q4_0;

impl BlockQuant for Q4_0 {
    fn block_len(&self) -> usize {
        32
    }

    fn block_bytes(&self) -> usize {
        2 + self.block_len() / 2
    }

    fn quant_block(&self, block: &[f32], quant: &mut [u8]) {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut writer = Cursor::new(quant);
        let mut writer = NibbleWriter::new(&mut writer);
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

        for &x in block {
            writer.write_i4(((x * id + 8.5f32) as i8).min(15));
        }
    }

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut reader = Cursor::new(quant);
        let mut nibbles = NibbleReader::new(&mut reader);
        let d = nibbles.read_f16().to_f32();
        block.iter_mut().for_each(|x| {
            *x = (nibbles.read_i4() - 8) as f32 * d;
        })
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
        unsafe {
            let mut blob = Blob::for_layout(Layout::from_size_align(panel_bytes * panels, 128)?);
            let mut writer = Cursor::new(&mut blob[..]);
            let mut writer = NibbleWriter::new(&mut writer);
            for p in 0..panels {
                let input = &input[r * p * row_bytes..];
                let mut line_readers =
                    (0..r).map(|r| Cursor::new(&input[r * row_bytes..])).collect_vec();
                let mut readers = line_readers.iter_mut().map(NibbleReader::new).collect_vec();
                for _ in 0..blocks_for_k {
                    readers.iter_mut().for_each(|r| writer.write_f16(r.read_f16()));
                    for _ in 0..self.block_len() / 2 {
                        for _ in [0, 1] {
                            for r in &mut readers {
                                writer.write_i4(r.read_i4());
                            }
                        }
                    }
                }
            }
            Ok(blob)
        }
    }
}

pub struct NibbleReader<'r, R> {
    second_half: Option<i8>,
    reader: &'r mut R,
}

impl<'r, R: Read> NibbleReader<'r, R> {
    pub fn new(reader: &'r mut R) -> NibbleReader<R> {
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

pub struct NibbleWriter<'r, W> {
    first_half: Option<i8>,
    writer: &'r mut W,
}

impl<'r, W: Write> NibbleWriter<'r, W> {
    pub fn new(writer: &'r mut W) -> NibbleWriter<W> {
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
}
