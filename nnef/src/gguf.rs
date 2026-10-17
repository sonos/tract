//! Read a GGUF v3 tensor table into NNEF `.dat` blobs.
//!
//! GGUF stores `ne[0]` as the contiguous axis, and the APEX blocks are
//! quantized along that axis. The tract tensor reverses those dimensions so
//! the block axis is last. F32, F16, and the seven block payloads are copied.
//! Every other ggml type is refused by name. No importance matrix is read.

use std::io::Cursor;
use std::io::Read;
use std::sync::Arc;

use byteorder::LE;
use byteorder::ReadBytesExt;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantStorage;
use tract_core::tract_linalg::block_quant::IQ2_S;
use tract_core::tract_linalg::block_quant::IQ4_XS;
use tract_core::tract_linalg::block_quant::Q3_K;
use tract_core::tract_linalg::block_quant::Q4_K;
use tract_core::tract_linalg::block_quant::Q5_K;
use tract_core::tract_linalg::block_quant::Q6_K;
use tract_core::tract_linalg::block_quant::Q8_0;

use crate::tensors::write_tensor;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: usize = 32;
const MAX_STRING_BYTES: u64 = 1024 * 1024;

/// One GGUF tensor encoded as an NNEF `.dat` blob.
#[derive(Debug)]
pub struct GgufNnefTensor {
    /// Name from the GGUF tensor table.
    pub name: String,
    /// Bytes produced by [`write_tensor`](crate::tensors::write_tensor).
    pub dat: Vec<u8>,
}

/// Copy each tensor in a little-endian GGUF v3 file into an NNEF `.dat` blob.
///
/// Accepts F32, F16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, and IQ4_XS.
/// `general.alignment` defaults to 32 and, when present, is a `uint32`
/// multiple of 8. A tensor whose payload does not fit in an NNEF blob is
/// refused. The ggml bytes are not dequantized and not requantized.
pub fn gguf_v3_to_nnef_dats(file: &[u8]) -> TractResult<Vec<GgufNnefTensor>> {
    let mut cursor = Cursor::new(file);
    let magic = cursor.read_u32::<LE>()?;
    ensure!(magic == GGUF_MAGIC, "not a GGUF file");
    let version = cursor.read_u32::<LE>()?;
    ensure!(version == GGUF_VERSION, "GGUF version {version} is not supported");
    let tensor_count = read_count(&mut cursor, "tensor count")?;
    let kv_count = read_count(&mut cursor, "metadata count")?;
    let alignment = read_metadata(&mut cursor, kv_count)?;
    let infos = read_tensor_infos(&mut cursor, tensor_count)?;
    let data_start = align_up(cursor.position() as usize, alignment)?;
    ensure!(data_start <= file.len(), "GGUF tensor data starts past the end of the file");
    let data = &file[data_start..];
    infos.into_iter().map(|info| tensor_dat(info, data, alignment)).collect()
}

struct TensorInfo {
    name: String,
    dims: TVec<usize>,
    ggml_type: u32,
    offset: usize,
}

fn read_metadata(cursor: &mut Cursor<&[u8]>, kv_count: usize) -> TractResult<usize> {
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let key = read_string(cursor)?;
        let value_type = cursor.read_u32::<LE>()?;
        if key == "general.alignment" {
            ensure!(value_type == 4, "general.alignment must be a uint32");
            let value = cursor.read_u32::<LE>()? as usize;
            ensure!(
                value >= 8 && value.is_multiple_of(8),
                "general.alignment {value} is not a multiple of 8"
            );
            alignment = value;
        } else {
            skip_value(cursor, value_type, 0)?;
        }
    }
    Ok(alignment)
}

fn read_tensor_infos(
    cursor: &mut Cursor<&[u8]>,
    tensor_count: usize,
) -> TractResult<Vec<TensorInfo>> {
    // The count is untrusted. Do not reserve from it.
    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(cursor)?;
        ensure!(!name.is_empty(), "GGUF tensor name is empty");
        let n_dims = cursor.read_u32::<LE>()? as usize;
        ensure!((1..=8).contains(&n_dims), "GGUF tensor {name} has rank {n_dims}");
        let mut dims = tvec!();
        for _ in 0..n_dims {
            let dim = read_count(cursor, &format!("dimension of {name}"))?;
            ensure!(
                dim <= u32::MAX as usize,
                "GGUF tensor {name} has a dimension that does not fit NNEF"
            );
            dims.push(dim);
        }
        let ggml_type = cursor.read_u32::<LE>()?;
        let offset = read_count(cursor, &format!("offset of {name}"))?;
        infos.push(TensorInfo { name, dims, ggml_type, offset });
    }
    Ok(infos)
}

fn tensor_dat(info: TensorInfo, data: &[u8], alignment: usize) -> TractResult<GgufNnefTensor> {
    ensure!(
        info.offset.is_multiple_of(alignment),
        "GGUF tensor {} offset {} is not aligned",
        info.name,
        info.offset
    );
    // GGUF ne[0] is the contiguous block axis. NNEF stores that axis last.
    let shape: TVec<usize> = info.dims.iter().rev().copied().collect();
    let tensor = if let Some(format) = block_format(info.ggml_type) {
        block_tensor(&info, data, &shape, format)?
    } else if let Some(dt) = plain_type(info.ggml_type) {
        plain_tensor(&info, data, &shape, dt)?
    } else {
        let type_name = ggml_type_name(info.ggml_type).unwrap_or("unknown");
        bail!(
            "GGUF tensor {} has type {type_name} ({}), which this reader does not copy",
            info.name,
            info.ggml_type
        );
    };
    let mut dat = Vec::new();
    write_tensor(&mut dat, &tensor)?;
    Ok(GgufNnefTensor { name: info.name, dat })
}

fn block_tensor(
    info: &TensorInfo,
    data: &[u8],
    shape: &[usize],
    format: Box<dyn BlockQuant>,
) -> TractResult<Tensor> {
    let Some((&k, prefix)) = shape.split_last() else {
        bail!("GGUF tensor {} has no dimensions", info.name);
    };
    ensure!(
        k.is_multiple_of(format.block_len()),
        "GGUF tensor {} K {k} is not a multiple of the block",
        info.name
    );
    let m = prefix
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("block-quant shape overflows")?;
    let row_bytes = (k / format.block_len())
        .checked_mul(format.block_bytes())
        .context("block-quant row overflows")?;
    let nbytes = m.checked_mul(row_bytes).context("block-quant length overflows")?;
    let payload = payload(info, data, nbytes)?;
    let storage = BlockQuantStorage::new(format, m, k, Arc::new(Blob::from_bytes(payload)?))?;
    Ok(storage.into_tensor_with_shape(f32::datum_type(), shape))
}

fn plain_tensor(
    info: &TensorInfo,
    data: &[u8],
    shape: &[usize],
    dt: DatumType,
) -> TractResult<Tensor> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .context("tensor length overflows")?;
    let nbytes = len.checked_mul(dt.size_of()).context("tensor byte length overflows")?;
    let payload = payload(info, data, nbytes)?;
    let mut tensor = Tensor::zero_dt(dt, shape)?;
    tensor.try_as_plain_ram_mut()?.as_bytes_mut().copy_from_slice(payload);
    Ok(tensor)
}

fn payload<'a>(info: &TensorInfo, data: &'a [u8], nbytes: usize) -> TractResult<&'a [u8]> {
    ensure!(nbytes <= u32::MAX as usize, "GGUF tensor {} does not fit in an NNEF blob", info.name);
    let end = info.offset.checked_add(nbytes).context("tensor extent overflows")?;
    ensure!(end <= data.len(), "GGUF tensor {} extends past the end of the file", info.name);
    Ok(&data[info.offset..end])
}

fn block_format(ggml_type: u32) -> Option<Box<dyn BlockQuant>> {
    Some(match ggml_type {
        8 => Box::new(Q8_0),
        11 => Box::new(Q3_K),
        12 => Box::new(Q4_K),
        13 => Box::new(Q5_K),
        14 => Box::new(Q6_K),
        22 => Box::new(IQ2_S),
        23 => Box::new(IQ4_XS),
        _ => return None,
    })
}

fn plain_type(ggml_type: u32) -> Option<DatumType> {
    match ggml_type {
        0 => Some(DatumType::F32),
        1 => Some(DatumType::F16),
        _ => None,
    }
}

fn ggml_type_name(ggml_type: u32) -> Option<&'static str> {
    Some(match ggml_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => return None,
    })
}

fn read_count(cursor: &mut Cursor<&[u8]>, what: &str) -> TractResult<usize> {
    let value = cursor.read_u64::<LE>()?;
    usize::try_from(value).with_context(|| format!("GGUF {what} does not fit usize"))
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> TractResult<String> {
    let len = cursor.read_u64::<LE>()?;
    ensure!(len <= MAX_STRING_BYTES, "GGUF string is longer than {MAX_STRING_BYTES} bytes");
    ensure!(len <= remaining(cursor), "GGUF string extends past the end of the file");
    let mut bytes = vec![0u8; len as usize];
    cursor.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF string is not UTF-8")
}

fn skip_value(cursor: &mut Cursor<&[u8]>, value_type: u32, depth: u32) -> TractResult<()> {
    ensure!(depth < 8, "GGUF metadata array is nested too deeply");
    match value_type {
        0 | 1 | 7 => {
            cursor.read_u8()?;
        }
        2 | 3 => {
            cursor.read_u16::<LE>()?;
        }
        4..=6 => {
            cursor.read_u32::<LE>()?;
        }
        10..=12 => {
            cursor.read_u64::<LE>()?;
        }
        8 => {
            read_string(cursor)?;
        }
        9 => {
            let elem = cursor.read_u32::<LE>()?;
            let len = cursor.read_u64::<LE>()?;
            ensure!(len <= remaining(cursor), "GGUF metadata array is longer than the file");
            for _ in 0..len {
                skip_value(cursor, elem, depth + 1)?;
            }
        }
        other => bail!("unknown GGUF metadata type {other}"),
    }
    Ok(())
}

fn remaining(cursor: &Cursor<&[u8]>) -> u64 {
    cursor.get_ref().len().saturating_sub(cursor.position() as usize) as u64
}

fn align_up(pos: usize, alignment: usize) -> TractResult<usize> {
    let rem = pos % alignment;
    if rem == 0 {
        return Ok(pos);
    }
    pos.checked_add(alignment - rem).context("GGUF alignment overflows")
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensors::read_tensor;

    struct Raw<'a> {
        name: &'a str,
        ggml_type: u32,
        dims: &'a [u64],
        data: &'a [u8],
    }

    fn push_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_str(buf: &mut Vec<u8>, value: &str) {
        push_u64(buf, value.len() as u64);
        buf.extend_from_slice(value.as_bytes());
    }

    /// `alignment` `None` omits `general.alignment`, so the reader must use 32.
    fn gguf_file(alignment: Option<u32>, tensors: &[Raw<'_>]) -> Vec<u8> {
        let align = alignment.unwrap_or(32) as usize;
        let mut buf = Vec::new();
        push_u32(&mut buf, GGUF_MAGIC);
        push_u32(&mut buf, GGUF_VERSION);
        push_u64(&mut buf, tensors.len() as u64);
        let kv_count = 2 + usize::from(alignment.is_some());
        push_u64(&mut buf, kv_count as u64);
        push_str(&mut buf, "general.architecture");
        push_u32(&mut buf, 8);
        push_str(&mut buf, "llama");
        push_str(&mut buf, "general.tags");
        push_u32(&mut buf, 9);
        push_u32(&mut buf, 8);
        push_u64(&mut buf, 1);
        push_str(&mut buf, "test");
        if let Some(alignment) = alignment {
            push_str(&mut buf, "general.alignment");
            push_u32(&mut buf, 4);
            push_u32(&mut buf, alignment);
        }
        let mut offset = 0u64;
        let mut placed = Vec::new();
        for tensor in tensors {
            let aligned = offset.div_ceil(align as u64) * align as u64;
            placed.push((tensor, aligned));
            offset = aligned + tensor.data.len() as u64;
        }
        for (tensor, offset) in &placed {
            push_str(&mut buf, tensor.name);
            push_u32(&mut buf, tensor.dims.len() as u32);
            for dim in tensor.dims {
                push_u64(&mut buf, *dim);
            }
            push_u32(&mut buf, tensor.ggml_type);
            push_u64(&mut buf, *offset);
        }
        let data_start = buf.len().div_ceil(align) * align;
        buf.resize(data_start, 0);
        let mut cursor = 0u64;
        for (tensor, offset) in &placed {
            buf.resize(buf.len() + (*offset - cursor) as usize, 0);
            buf.extend_from_slice(tensor.data);
            cursor = offset + tensor.data.len() as u64;
        }
        buf
    }

    fn known_matrix() -> Tensor {
        let data: Vec<f32> =
            (0..4 * 256).map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) / 7.0).collect();
        Tensor::from_shape(&[4, 256], &data).unwrap()
    }

    #[test]
    fn apex_blocks_round_trip_ggml_bytes() -> TractResult<()> {
        let matrix = known_matrix();
        let values = matrix.try_as_plain_ram()?.as_slice::<f32>()?;
        let formats: [(&dyn BlockQuant, u32); 7] = [
            (&Q8_0, 8),
            (&Q3_K, 11),
            (&Q4_K, 12),
            (&Q5_K, 13),
            (&Q6_K, 14),
            (&IQ2_S, 22),
            (&IQ4_XS, 23),
        ];
        for (format, ggml_type) in formats {
            let quant = format.quant_f32(values)?;
            let name = format.to_string();
            let file = gguf_file(
                Some(32),
                &[Raw { name: &name, ggml_type, dims: &[256, 4], data: quant.as_bytes() }],
            );
            let dats = gguf_v3_to_nnef_dats(&file)?;
            assert_eq!(dats.len(), 1);
            assert_eq!(dats[0].name, name);
            assert_eq!(&dats[0].dat[128..], quant.as_bytes());
            let back = read_tensor(dats[0].dat.as_slice())?;
            assert_eq!(back.shape(), &[4, 256]);
            let stored = back.try_storage_as::<BlockQuantStorage>()?;
            let got = stored.format().dequant_f32(stored.value().as_bytes())?;
            let expect = format.dequant_f32(quant.as_bytes())?;
            assert_eq!(
                got.try_as_plain_ram()?.as_slice::<f32>()?,
                expect.try_as_plain_ram()?.as_slice::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn f32_and_f16_pass_through() -> TractResult<()> {
        let f32_values = Tensor::from_shape(&[2, 3], &[1.0f32, -2.0, 0.5, 4.0, -0.25, 8.0])?;
        let f16_values = f32_values.cast_to::<f16>()?.into_owned();
        let file = gguf_file(
            None,
            &[
                Raw { name: "plain.f32", ggml_type: 0, dims: &[3, 2], data: f32_values.as_bytes() },
                Raw { name: "plain.f16", ggml_type: 1, dims: &[3, 2], data: f16_values.as_bytes() },
            ],
        );
        let dats = gguf_v3_to_nnef_dats(&file)?;
        assert_eq!(dats.len(), 2);
        let back_f32 = read_tensor(dats[0].dat.as_slice())?;
        let back_f16 = read_tensor(dats[1].dat.as_slice())?;
        assert_eq!(back_f32, f32_values);
        assert_eq!(back_f16, f16_values);
        Ok(())
    }

    #[test]
    fn alignment_64_still_reads_f32() -> TractResult<()> {
        let values = Tensor::from_shape(&[1, 4], &[1.0f32, 2.0, 3.0, 4.0])?;
        let file = gguf_file(
            Some(64),
            &[Raw { name: "w", ggml_type: 0, dims: &[4, 1], data: values.as_bytes() }],
        );
        let dats = gguf_v3_to_nnef_dats(&file)?;
        assert_eq!(read_tensor(dats[0].dat.as_slice())?, values);
        Ok(())
    }

    #[test]
    fn other_quantized_types_are_refused_by_name() {
        for (ggml_type, name) in [(2u32, "Q4_0"), (16, "IQ2_XXS"), (17, "IQ2_XS"), (29, "IQ1_M")] {
            let file = gguf_file(
                Some(32),
                &[Raw { name: "w", ggml_type, dims: &[32, 1], data: &[0u8; 18] }],
            );
            let err = gguf_v3_to_nnef_dats(&file).unwrap_err().to_string();
            assert!(err.contains(name), "{err}");
        }
    }

    #[test]
    fn version_2_is_refused() {
        let mut file = Vec::new();
        push_u32(&mut file, GGUF_MAGIC);
        push_u32(&mut file, 2);
        let err = gguf_v3_to_nnef_dats(&file).unwrap_err().to_string();
        assert!(err.contains("version 2"), "{err}");
    }
}
