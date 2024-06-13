use byteorder::{ReadBytesExt, WriteBytesExt, LE};
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
}
