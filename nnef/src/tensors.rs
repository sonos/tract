use std::io::{Read, Write};

use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use tract_core::internal::*;
use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0};

const TRACT_ITEM_TYPE_VENDOR: u16 = ((b'T' as u16) << 8u16) | b'R' as u16;

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

impl Default for Header {
    fn default() -> Self {
        let mut header: Header = unsafe { std::mem::zeroed() };
        header.magic = [0x4e, 0xef];
        header.version_maj = 1;
        header.version_min = 0;
        header
    }
}

impl Header {
    pub fn write(&self, w: &mut impl Write) -> TractResult<()> {
        let slice = unsafe { std::mem::transmute::<&Header, &[u8; 128]>(self) };
        w.write_all(slice)?;
        Ok(())
    }

    pub fn read(reader: &mut impl Read) -> TractResult<Header> {
        let mut header: Header = unsafe { std::mem::zeroed() };
        reader.read_exact(unsafe {
            std::mem::transmute::<&mut Header, &mut [u8; 128]>(&mut header)
        })?;
        if header.magic != [0x4e, 0xef] {
            bail!("Wrong magic number");
        };
        if header.version_maj != 1 && header.version_min != 0 {
            bail!("Wrong version number");
        }
        if header.rank > 8 {
            bail!("Wrong tensor rank {}", header.rank);
        }
        Ok(header)
    }
}

pub fn read_tensor(mut reader: impl Read) -> TractResult<Tensor> {
    let header = Header::read(&mut reader)?;
    let shape: TVec<usize> = header.dims[0..header.rank as usize].iter().map(|d| *d as _).collect();
    let len = shape.iter().product::<usize>();

    if header.item_type == 5 {
        let expected_bit_size = len * header.bits_per_item as usize;
        let real_bit_size = header.data_size_bytes as usize * 8;
        if !(real_bit_size - 8 <= expected_bit_size && expected_bit_size <= real_bit_size) {
            bail!(
                "Shape and len mismatch: shape:{:?}, bits_per_item:{}, bytes:{} ",
                shape,
                header.bits_per_item,
                header.data_size_bytes
            );
        }
    } else if header.bits_per_item != u32::MAX
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

    // last checked with spec 1.0.5: https://registry.khronos.org/NNEF/specs/1.0/nnef-1.0.5.html
    //
    // Quantized types are not instanciated as DatumType::Q* here since
    // quant infos are joined later from .quant file (
    //  see: ops/nnef/deser.rs
    // )
    let dt = match (header.item_type_vendor, header.item_type, header.bits_per_item) {
        // 0 - 0b0000 - float values in IEEE format, valid bits per item is 16, 32, 64
        (0, 0, 16) => DatumType::F16,
        (0, 0, 32) => DatumType::F32,
        (0, 0, 64) => DatumType::F64,

        // 1 - 0b0001 - unsigned integer values, maximum bits per item is 64.
        (0, 1, 8) => DatumType::U8,
        (0, 1, 16) => DatumType::U16,
        (0, 1, 32) => DatumType::U32,
        (0, 1, 64) => DatumType::U64,

        // 2 - 0b0010 - quantized unsigned integer values, maximum bits per item is 64.
        (0, 2, 8) => DatumType::U8,
        (0, 2, 16) => DatumType::U16,
        (0, 2, 32) => DatumType::U32,
        (0, 2, 64) => DatumType::U64,

        // 3 - 0b0011 - quantized signed integer values, maximum bits per item is 64.
        (0, 3, 8) => DatumType::I8,
        (0, 3, 16) => DatumType::I16,
        (0, 3, 32) => DatumType::I32,
        (0, 3, 64) => DatumType::I64,

        // 4 - 0b0100 - signed integer values, maximum bits per item is 64.
        (0, 4, 8) => DatumType::I8,
        (0, 4, 16) => DatumType::I16,
        (0, 4, 32) => DatumType::I32,
        (0, 4, 64) => DatumType::I64,

        // 5 - 0b0101 - bool values, 1 bit or 8 bits (0 means false, non-zero means true)
        (0, 5, 1 | 8) => DatumType::Bool,
        (TRACT_ITEM_TYPE_VENDOR, 0x1000, _) => DatumType::String,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 0, 32) => DatumType::ComplexF16,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 0, 64) => DatumType::ComplexF32,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 0, 128) => DatumType::ComplexF64,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 4, 32) => DatumType::ComplexI16,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 4, 64) => DatumType::ComplexI32,
        #[cfg(feature = "complex")]
        (TRACT_ITEM_TYPE_VENDOR, 4, 128) => DatumType::ComplexI64,
        (TRACT_ITEM_TYPE_VENDOR, it, _)
            if ((it & 0x2000) == 0x2000) || ((it & 0x3000) == 0x3000) =>
        {
            return read_block_quant_value(&mut reader, &header);
        }
        _ => bail!(
            "Unsupported type in tensor type:{} bits_per_item:{}",
            header.item_type,
            header.bits_per_item
        ),
    };
    if dt.is_copy() {
        let mut tensor = unsafe { Tensor::uninitialized_dt(dt, &shape)? };
        if dt == DatumType::Bool && header.bits_per_item == 1 {
            let buf = tensor.as_slice_mut::<bool>()?;

            let mut current_byte = 0;
            for (ix, value) in buf.iter_mut().enumerate() {
                let bit_ix = ix % 8;
                if bit_ix == 0 {
                    current_byte = reader.read_u8()?;
                }
                *value = ((current_byte >> (7 - bit_ix)) & 0x1) != 0;
            }
        } else {
            reader.read_exact(tensor.as_bytes_mut())?;
        }
        Ok(tensor)
    } else if dt == DatumType::String {
        let mut tensor = Tensor::zero_dt(dt, &shape)?;
        for item in tensor.as_slice_mut::<String>()? {
            let len: u32 = reader.read_u32::<LE>()?;
            let mut bytes = Vec::with_capacity(len as usize);
            #[allow(clippy::uninit_vec)]
            unsafe {
                bytes.set_len(len as usize);
            };
            reader.read_exact(&mut bytes)?;
            *item = String::from_utf8(bytes)?;
        }
        Ok(tensor)
    } else {
        todo!()
    }
}

pub fn write_tensor(w: &mut impl Write, tensor: &Tensor) -> TractResult<()> {
    ensure!(tensor.datum_type() != TDim::datum_type());
    if tensor.datum_type() == Opaque::datum_type() {
        ensure!(tensor.rank() == 0);
        if let Some(bqv) = tensor.to_scalar::<Opaque>()?.downcast_ref::<BlockQuantValue>() {
            return write_block_quant_value(w, bqv);
        }
    }
    let mut header = Header::default();
    if tensor.rank() > 8 {
        bail!("Only rank up to 8 are supported");
    }
    header.rank = tensor.rank() as u32;
    for d in 0..tensor.rank() {
        header.dims[d] = tensor.shape()[d] as u32;
    }
    header.data_size_bytes = (tensor.len() * tensor.datum_type().size_of()) as u32;
    header.bits_per_item = (tensor.datum_type().size_of() * 8) as u32;

    let (itv, it) = match tensor.datum_type() {
        DatumType::F16 | DatumType::F32 | DatumType::F64 => (0, 0),
        DatumType::U8 | DatumType::U16 | DatumType::U32 | DatumType::U64 | DatumType::QU8(_) => {
            (0, 2)
        }
        DatumType::I8
        | DatumType::I16
        | DatumType::I32
        | DatumType::I64
        | DatumType::QI8(_)
        | DatumType::QI32(_) => (0, 3),
        DatumType::String => {
            header.bits_per_item = u32::MAX;
            (TRACT_ITEM_TYPE_VENDOR, 0x1000)
        }
        #[cfg(feature = "complex")]
        DatumType::ComplexF16 | DatumType::ComplexF32 | DatumType::ComplexF64 => {
            (TRACT_ITEM_TYPE_VENDOR, 0)
        }
        #[cfg(feature = "complex")]
        DatumType::ComplexI16 | DatumType::ComplexI32 | DatumType::ComplexI64 => {
            (TRACT_ITEM_TYPE_VENDOR, 4)
        }
        DatumType::Bool => (0, 5),
        DatumType::TDim | DatumType::Blob | DatumType::Opaque => {
            bail!("Don't know how to serialize {:?}", tensor.datum_type())
        }
    };
    header.item_type = it;
    header.item_type_vendor = itv;
    header.write(w)?;
    if tensor.datum_type().is_copy() {
        w.write_all(tensor.as_bytes())?;
    } else if tensor.datum_type() == DatumType::String {
        for s in tensor.as_slice::<String>()? {
            w.write_u32::<LE>(s.len() as u32)?;
            w.write_all(s.as_bytes())?;
        }
    }
    Ok(())
}

pub fn tract_to_gguf_q4_0_packing(data: &mut Blob) -> TractResult<()> {
    let block_size = Q4_0.block_bytes();
    ensure!(data.layout().size() % block_size == 0);

    let n_block = data.layout().size() / block_size;
    let data_bytes = data.as_bytes_mut();

    for b in 0..n_block {
        let offset = b * block_size + 2;
        let nibbles = &mut data_bytes[offset..offset + 16];
        let second_part: &mut [u8; 8] = &mut [0; 8];
        second_part.clone_from_slice(&nibbles[8..]);
        for i in (0..16).rev() {
            let lsb = if i % 2 == 0 { nibbles[i / 2] & 0x0F } else { (nibbles[i / 2] & 0xF0) >> 4 };
            let msb = if i % 2 == 0 {
                (second_part[i / 2] & 0x0F) << 4
            } else {
                second_part[i / 2] & 0xF0
            };
            nibbles[i] = msb | lsb;
        }
    }
    Ok(())
}

fn read_block_quant_value(r: &mut impl Read, header: &Header) -> TractResult<Tensor> {
    let format = if matches!(header.item_type, 0x2040 | 0x3040) {
        Q4_0
    } else {
        bail!("Unexpected block quant format")
    };
    ensure!(header.rank >= 2);
    let shape: TVec<_> =
        header.dims.iter().take(header.rank as usize).map(|d| *d as usize).collect();
    let q_m = shape[0];
    let q_k = shape.iter().skip(1).product::<usize>();
    ensure!(q_k % format.block_len() == 0);
    let expected_len = (q_m * q_k) / format.block_len() * format.block_bytes();
    ensure!(expected_len == header.data_size_bytes as _);
    let mut blob = unsafe { Blob::new_for_size_and_align(expected_len, 128) };
    r.read_exact(&mut blob)?;
    if header.item_type == 0x2040 {
        tract_to_gguf_q4_0_packing(&mut blob)?;
    }
    let fact = BlockQuantFact::new(Box::new(format), shape);
    let bqv = BlockQuantValue { value: Arc::new(blob), fact };
    let tensor = tensor0(Opaque(Arc::new(bqv)));
    Ok(tensor)
}

#[allow(clippy::field_reassign_with_default)]
fn write_block_quant_value(w: &mut impl Write, value: &BlockQuantValue) -> TractResult<()> {
    ensure!(value.fact.format.same_as(&Q4_0));

    let mut header = Header::default();
    header.rank = value.fact.shape().len() as u32;
    for (h, v) in header.dims.iter_mut().zip(value.fact.shape().iter()) {
        *h = *v as u32;
    }
    header.bits_per_item = u32::MAX;
    header.data_size_bytes = value.value.len() as _;
    header.item_type_vendor = TRACT_ITEM_TYPE_VENDOR;
    // 0x3040 3 is for GGML formats, 0 for Q formats then 4 and 0
    header.item_type = 0x3040;
    header.write(w)?;
    w.write_all(&value.value)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn header_is_128_bytes() {
        assert_eq!(std::mem::size_of::<Header>(), 128);
    }

    #[test]
    #[cfg(feature = "complex")]
    fn serde_tensor_complex_f32() -> TractResult<()> {
        let t = tensor2(&[
            [Complex::new(1.0f32, 2.0), Complex::new(2.0, 1.0), Complex::new(3.5, 2.4)],
            [Complex::new(3.0, 4.5), Complex::new(3.0, 2.5), Complex::new(1.5, 2.5)],
        ]);
        let mut buffer = Vec::<u8>::new();
        write_tensor(&mut buffer, &t)?;
        let serde_tensor = read_tensor(buffer.as_slice())?;
        assert_eq!(t, serde_tensor);
        Ok(())
    }

    #[test]
    #[cfg(feature = "complex")]
    fn serde_tensor_complex_f64() -> TractResult<()> {
        let t = tensor2(&[
            [Complex::new(1.0f64, 2.0), Complex::new(2.0, 1.0), Complex::new(3.5, 2.4)],
            [Complex::new(3.0, 4.5), Complex::new(3.0, 2.5), Complex::new(1.5, 2.5)],
        ]);
        let mut buffer = Vec::<u8>::new();
        write_tensor(&mut buffer, &t)?;
        let serde_tensor = read_tensor(buffer.as_slice())?;
        assert_eq!(t, serde_tensor);
        Ok(())
    }

    #[test]
    #[cfg(feature = "complex")]
    fn serde_tensor_complex_i32() -> TractResult<()> {
        let t = tensor2(&[
            [Complex::new(1i32, 2), Complex::new(2, 1), Complex::new(3, 2)],
            [Complex::new(3, 4), Complex::new(3, 2), Complex::new(1, 2)],
        ]);
        let mut buffer = Vec::<u8>::new();
        write_tensor(&mut buffer, &t)?;
        let serde_tensor = read_tensor(buffer.as_slice())?;
        assert_eq!(t, serde_tensor);
        Ok(())
    }

    #[test]
    #[cfg(feature = "complex")]
    fn serde_tensor_complex_i64() -> TractResult<()> {
        let t = tensor2(&[
            [Complex::new(1i64, 2), Complex::new(2, 1), Complex::new(3, 2)],
            [Complex::new(3, 4), Complex::new(3, 2), Complex::new(1, 2)],
        ]);
        let mut buffer = Vec::<u8>::new();
        write_tensor(&mut buffer, &t)?;
        let serde_tensor = read_tensor(buffer.as_slice())?;
        assert_eq!(t, serde_tensor);
        Ok(())
    }
}
