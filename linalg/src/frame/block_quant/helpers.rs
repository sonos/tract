use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};
use tract_data::internal::*;

pub struct NibbleReader<R> {
    second_half: Option<i8>,
    reader: R,
}

impl<'s> NibbleReader<Cursor<&'s [u8]>> {
    pub fn for_slice(slice: &'s [u8]) -> Self {
        NibbleReader::new(Cursor::new(slice))
    }
}

impl<R: Read> NibbleReader<R> {
    pub fn new(reader: R) -> NibbleReader<R> {
        NibbleReader { reader, second_half: None }
    }

    pub fn read_f16(&mut self) -> f16 {
        assert!(self.second_half.is_none());
        f16::from_bits(self.reader.read_u16::<LE>().unwrap())
    }

    pub fn read_i4(&mut self) -> i8 {
        if let Some(second) = self.second_half.take() {
            second
        } else {
            let byte = self.reader.read_u8().unwrap();
            self.second_half = Some((byte >> 4) as i8);
            (byte & 0x0F) as i8
        }
    }

    pub fn read_i8(&mut self) -> i8 {
        self.reader.read_i8().unwrap()
    }
}

pub struct NibbleWriter<W> {
    first_half: Option<i8>,
    writer: W,
}

impl<'s> NibbleWriter<Cursor<&'s mut [u8]>> {
    pub fn for_slice(slice: &'s mut [u8]) -> Self {
        NibbleWriter::new(Cursor::new(slice))
    }
}

impl<W: Write> NibbleWriter<W> {
    pub fn new(writer: W) -> NibbleWriter<W> {
        NibbleWriter { writer, first_half: None }
    }

    pub fn write_f16(&mut self, f: f16) {
        assert!(self.first_half.is_none());
        self.writer.write_u16::<LE>(f.to_bits()).unwrap()
    }

    pub fn write_i4(&mut self, q: i8) {
        if let Some(first) = self.first_half.take() {
            self.writer.write_u8(first as u8 | ((q as u8) << 4)).unwrap()
        } else {
            self.first_half = Some(q);
        }
    }

    pub fn write_i8(&mut self, q: i8) {
        self.writer.write_i8(q).unwrap()
    }
}

/// Reads/writes 2-bit values ("crumbs"), four per byte, least-significant pair first.
/// Used by the ternary (Q1_58 / BitNet b1.58) block-quant format. f16 reads/writes are
/// only legal on a crumb (byte) boundary, which the ternary layouts always respect.
pub struct CrumbReader<R> {
    acc: u8,
    remaining: u8,
    reader: R,
}

impl<'s> CrumbReader<Cursor<&'s [u8]>> {
    pub fn for_slice(slice: &'s [u8]) -> Self {
        CrumbReader::new(Cursor::new(slice))
    }
}

impl<R: Read> CrumbReader<R> {
    pub fn new(reader: R) -> CrumbReader<R> {
        CrumbReader { reader, acc: 0, remaining: 0 }
    }

    pub fn read_f16(&mut self) -> f16 {
        assert!(self.remaining == 0);
        f16::from_bits(self.reader.read_u16::<LE>().unwrap())
    }

    /// Returns the 2-bit code (0..=3).
    pub fn read_crumb(&mut self) -> u8 {
        if self.remaining == 0 {
            self.acc = self.reader.read_u8().unwrap();
            self.remaining = 4;
        }
        let c = self.acc & 0x3;
        self.acc >>= 2;
        self.remaining -= 1;
        c
    }
}

pub struct CrumbWriter<W> {
    acc: u8,
    filled: u8,
    writer: W,
}

impl<'s> CrumbWriter<Cursor<&'s mut [u8]>> {
    pub fn for_slice(slice: &'s mut [u8]) -> Self {
        CrumbWriter::new(Cursor::new(slice))
    }
}

impl<W: Write> CrumbWriter<W> {
    pub fn new(writer: W) -> CrumbWriter<W> {
        CrumbWriter { writer, acc: 0, filled: 0 }
    }

    pub fn write_f16(&mut self, f: f16) {
        assert!(self.filled == 0);
        self.writer.write_u16::<LE>(f.to_bits()).unwrap()
    }

    /// Writes a 2-bit code (only the low two bits of `c` are used).
    pub fn write_crumb(&mut self, c: u8) {
        self.acc |= (c & 0x3) << (2 * self.filled);
        self.filled += 1;
        if self.filled == 4 {
            self.writer.write_u8(self.acc).unwrap();
            self.acc = 0;
            self.filled = 0;
        }
    }
}
