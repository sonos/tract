use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use tract_core::internal::*;

const TRACT_ITEM_TYPE_VENDOR: u16 = (b'T' as u16) << 8u16 | b'R' as u16;

#[repr(C)]
#[derive(Debug)]
struct Header {
    magic: [u8; 2],
    version_maj: u8,
    version_min: u8,
    data_size_bytes: u32,
    rank: u32,
    dims: [u32; 8],
    bits_per_item: u32,
    item_type: u16,
    item_type_vendor: u16,
    item_type_params_deprecated: [u8; 32],
    padding: [u32; 11],
}

pub fn read_tensor<R: std::io::Read>(mut reader: R) -> TractResult<Tensor> {
    unsafe {
        let mut header: Header = std::mem::zeroed();
        let buffer: &mut [u8; 128] = std::mem::transmute(&mut header);
        reader.read_exact(buffer)?;
        if header.magic != [0x4e, 0xef] {
            bail!("Wrong magic number");
        };
        if header.version_maj != 1 && header.version_min != 0 {
            bail!("Wrong version number");
        }
        if header.rank > 8 {
            bail!("Wrong tensor rank {}", header.rank);
        }
        let shape: TVec<usize> =
            header.dims[0..header.rank as usize].iter().map(|d| *d as _).collect();
        let len = shape.iter().product::<usize>();
        if header.bits_per_item != 0xFFFFFFFF
            && len * (header.bits_per_item as usize / 8) != header.data_size_bytes as usize
        {
            bail!(
                "Shape and len mismatch: shape:{:?}, bits_per_item:{}, bytes:{} ",
                shape,
                header.bits_per_item,
                header.data_size_bytes
            );
        }
        if header.item_type_vendor != 0 && header.item_type_vendor != TRACT_ITEM_TYPE_VENDOR {
            bail!("Unknownn item type vendor {}", header.item_type_vendor);
        }
        let dt = match (header.item_type_vendor, header.item_type, header.bits_per_item) {
            (0, 0, 16) => DatumType::F16,
            (0, 0, 32) => DatumType::F32,
            (0, 0, 64) => DatumType::F64,
            (0, 1, 8) => DatumType::U8,
            (0, 1, 16) => DatumType::U16,
            (0, 1, 32) => DatumType::U32,
            (0, 1, 64) => DatumType::U64,
            (0, 4, 8) => DatumType::I8,
            (0, 4, 16) => DatumType::I16,
            (0, 4, 32) => DatumType::I32,
            (0, 4, 64) => DatumType::I64,
            (TRACT_ITEM_TYPE_VENDOR, 0x1000, 0xFFFF) => DatumType::String,
            _ => bail!(
                "Unsupported type in tensor type:{} bits_per_item:{}",
                header.item_type,
                header.bits_per_item
            ),
        };
        if dt.is_copy() {
            let mut tensor = Tensor::uninitialized_dt(dt, &shape)?;
            reader.read_exact(tensor.as_bytes_mut())?;
            Ok(tensor)
        } else if dt == DatumType::String {
            let mut tensor = Tensor::zero_dt(dt, &shape)?;
            for item in tensor.as_slice_mut_unchecked::<String>() {
                let len: u32 = reader.read_u32::<LE>()?;
                let mut bytes = Vec::with_capacity(len as usize);
                bytes.set_len(len as usize);
                reader.read_exact(&mut bytes)?;
                *item = String::from_utf8(bytes)?;
            }
            Ok(tensor)
        } else {
            todo!()
        }
    }
}

pub fn write_tensor<W: std::io::Write>(w: &mut W, tensor: &Tensor) -> TractResult<()> {
    unsafe {
        let mut header: Header = std::mem::zeroed();
        header.magic = [0x4e, 0xef];
        header.version_maj = 1;
        header.version_min = 0;
        if tensor.rank() > 8 {
            bail!("Only rank up to 8 are supported");
        }
        header.rank = tensor.rank() as u32;
        for d in 0..tensor.rank() {
            header.dims[d] = tensor.shape()[d] as u32;
        }
        header.data_size_bytes = (tensor.len() * tensor.datum_type().size_of()) as u32;
        header.bits_per_item = (tensor.datum_type().size_of() * 8) as u32;
        header.item_type = if tensor.datum_type().is_float() {
            0
        } else if tensor.datum_type().is_signed() {
            4
        } else if tensor.datum_type().is_unsigned() {
            1
        } else if tensor.datum_type() == DatumType::String {
            header.item_type_vendor = TRACT_ITEM_TYPE_VENDOR;
            header.bits_per_item = 0xFFFF;
            0x1000
        } else {
            bail!("Don't know how to serialize {:?}", tensor.datum_type())
        };
        let header_buf: &[u8; 128] = std::mem::transmute(&header);
        w.write_all(&*header_buf)?;
        if tensor.datum_type().is_copy() {
            w.write_all(tensor.as_bytes())?;
        } else if tensor.datum_type() == DatumType::String {
            for s in tensor.as_slice_unchecked::<String>() {
                w.write_u32::<LE>(s.as_bytes().len() as u32)?;
                w.write_all(s.as_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn header_is_128_bytes() {
        assert_eq!(std::mem::size_of::<Header>(), 128);
    }
}
