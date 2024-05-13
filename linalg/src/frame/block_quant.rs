use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use tract_data::internal::*;

use std::alloc::Layout;
use std::io::{Cursor, Write};

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
        let df16 = f16::from_f32(d).to_bits();
        writer.write_u16::<LE>(df16).unwrap();

        for i in 0..16 {
            let x0 = block[2 * i] * id;
            let x1 = block[2 * i + 1] * id;

            let xi0 = ((x0 + 8.5f32) as u8).min(15);
            let xi1 = ((x1 + 8.5f32) as u8).min(15);

            writer.write(&[xi0 | (xi1 << 4)]).unwrap();
        }
    }

    fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
        assert!(quant.len() == self.block_bytes());
        assert!(block.len() == self.block_len());
        let mut reader = Cursor::new(quant);
        let d = f16::from_bits(reader.read_u16::<LE>().unwrap()).to_f32();
        for i in 0..16 {
            let byte = reader.read_u8().unwrap();
            block[2 * i] = ((byte & 0x0F) as i8 - 8) as f32 * d;
            block[2 * i + 1] = ((byte >> 4) as i8 - 8) as f32 * d;
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
