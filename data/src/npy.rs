//! Reading and writing numpy `.npy` tensor files.
//!
//! Supports format versions 1.0, 2.0 and 3.0 holding a plain array of bool, integer
//! (8 to 64 bits) or float (f16, f32, f64) items, in either byte order and either memory
//! order. Structured, object, string and complex dtypes are rejected.
//!
//! The header is a Python dict literal. Only the subset numpy itself writes is accepted:
//! the three keys `descr`, `fortran_order` and `shape` (each exactly once, any order),
//! single-quoted strings, `True`/`False` and a flat tuple of non-negative integers.

use std::io::{Read, Write};

use crate::internal::*;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{char, one_of, space0, u8 as dec_u8, u64 as dec_u64};
use nom::combinator::{all_consuming, map, map_opt, map_res, opt, value};
use nom::multi::separated_list1;
use nom::sequence::{delimited, preceded, terminated};
use nom::{Finish, IResult, Parser};

const MAGIC: &[u8] = b"\x93NUMPY";
const MAX_HEADER_LEN: usize = 65535;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ByteOrder {
    Little,
    Big,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    datum_type: DatumType,
    byte_order: ByteOrder,
    fortran_order: bool,
    shape: TVec<usize>,
}

#[derive(Clone, Debug)]
enum Entry {
    Descr(DatumType, ByteOrder),
    FortranOrder(bool),
    Shape(TVec<usize>),
}

type R<'i, O> = IResult<&'i str, O>;

fn spaced<'i, O>(
    p: impl Parser<&'i str, Output = O, Error = nom::error::Error<&'i str>>,
) -> impl Parser<&'i str, Output = O, Error = nom::error::Error<&'i str>> {
    delimited(space0, p, space0)
}

fn key<'i>(
    name: &'static str,
) -> impl Parser<&'i str, Output = (), Error = nom::error::Error<&'i str>> {
    value((), (spaced(delimited(char('\''), tag(name), char('\''))), char(':')))
}

fn descr(i: &str) -> R<'_, (DatumType, ByteOrder)> {
    map_opt(
        delimited(char('\''), (one_of("<>|"), one_of("biuf"), dec_u8), char('\'')),
        |(order, kind, size)| {
            let byte_order = match (order, size) {
                ('|', 1) | ('<', _) => ByteOrder::Little,
                ('>', _) => ByteOrder::Big,
                _ => return None,
            };
            let dt = match (kind, size) {
                ('b', 1) => DatumType::Bool,
                ('u', 1) => DatumType::U8,
                ('u', 2) => DatumType::U16,
                ('u', 4) => DatumType::U32,
                ('u', 8) => DatumType::U64,
                ('i', 1) => DatumType::I8,
                ('i', 2) => DatumType::I16,
                ('i', 4) => DatumType::I32,
                ('i', 8) => DatumType::I64,
                ('f', 2) => DatumType::F16,
                ('f', 4) => DatumType::F32,
                ('f', 8) => DatumType::F64,
                _ => return None,
            };
            Some((dt, byte_order))
        },
    )
    .parse(i)
}

fn shape(i: &str) -> R<'_, TVec<usize>> {
    let dim = map_res(dec_u64, usize::try_from);
    delimited(
        char('('),
        map(
            opt(terminated(separated_list1(char(','), spaced(dim)), opt(spaced(char(','))))),
            |dims| dims.unwrap_or_default().into(),
        ),
        preceded(space0, char(')')),
    )
    .parse(i)
}

fn entry(i: &str) -> R<'_, Entry> {
    spaced(alt((
        preceded(key("descr"), map(spaced(descr), |(dt, bo)| Entry::Descr(dt, bo))),
        preceded(
            key("fortran_order"),
            map(
                spaced(alt((value(true, tag("True")), value(false, tag("False"))))),
                Entry::FortranOrder,
            ),
        ),
        preceded(key("shape"), map(spaced(shape), Entry::Shape)),
    )))
    .parse(i)
}

fn dict(i: &str) -> R<'_, Vec<Entry>> {
    spaced(delimited(
        char('{'),
        terminated(separated_list1(char(','), entry), opt(spaced(char(',')))),
        char('}'),
    ))
    .parse(i)
}

fn parse_header(bytes: &[u8]) -> TractResult<Header> {
    let text = std::str::from_utf8(bytes).context("npy header is not valid UTF-8")?;
    let text = text.strip_suffix('\n').context("npy header does not end with a newline")?;
    let (_, entries) = all_consuming(dict)
        .parse(text)
        .finish()
        .map_err(|e| format_err!("Invalid npy header {text:?}: {e:?}"))?;
    let (mut descr, mut fortran_order, mut shape) = (None, None, None);
    for entry in entries {
        let duplicate = match entry {
            Entry::Descr(dt, bo) => descr.replace((dt, bo)).is_some(),
            Entry::FortranOrder(f) => fortran_order.replace(f).is_some(),
            Entry::Shape(s) => shape.replace(s).is_some(),
        };
        ensure!(!duplicate, "Duplicate key in npy header {text:?}");
    }
    let (datum_type, byte_order) = descr.context("npy header misses `descr`")?;
    Ok(Header {
        datum_type,
        byte_order,
        fortran_order: fortran_order.context("npy header misses `fortran_order`")?,
        shape: shape.context("npy header misses `shape`")?,
    })
}

const NATIVE_ORDER: ByteOrder =
    if cfg!(target_endian = "little") { ByteOrder::Little } else { ByteOrder::Big };

fn read_u16_le(reader: &mut impl Read) -> TractResult<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32_le(reader: &mut impl Read) -> TractResult<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Reads a tensor from a `.npy` stream of at most `max_len` bytes.
///
/// `max_len` is the stream length when known (file size, npz entry size), `u64::MAX`
/// otherwise. The tensor is allocated from the header before its data is read, so a
/// header announcing more than `max_len` bytes is rejected upfront. With `u64::MAX`, a
/// forged header can trigger an allocation as large as it likes. Header size is capped
/// at 64KiB.
pub fn read_npy(mut reader: impl Read, max_len: u64) -> TractResult<Tensor> {
    let mut preamble = [0u8; 8];
    reader.read_exact(&mut preamble)?;
    ensure!(preamble[..6] == *MAGIC, "Not a npy file (bad magic string)");
    let (major, minor) = (preamble[6], preamble[7]);
    let (header_len, prefix_len) = match major {
        1 => (read_u16_le(&mut reader)? as usize, 10),
        2 | 3 => (usize::try_from(read_u32_le(&mut reader)?)?, 12),
        _ => bail!("Unsupported npy format version {major}.{minor}"),
    };
    ensure!(header_len <= MAX_HEADER_LEN, "npy header too long ({header_len} bytes)");
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let header = parse_header(&header)?;

    let dt = header.datum_type;
    let len = header
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .with_context(|| format!("npy shape {:?} overflows", header.shape))?;
    let size = len.checked_mul(dt.size_of()).context("npy data size overflows")?;
    ensure!(
        ((prefix_len + header_len) as u64).saturating_add(size as u64) <= max_len,
        "npy header announces {size} bytes of {dt:?} data, more than the {max_len} bytes available",
    );

    let mut shape = header.shape.clone();
    if header.fortran_order {
        shape.reverse();
    }
    let storage_dt = if dt == DatumType::Bool { DatumType::U8 } else { dt };
    // SAFETY: storage_dt is a plain number type, valid for any bit pattern, and every byte is
    // overwritten by read_exact before the tensor is used.
    let mut tensor = unsafe { Tensor::uninitialized_dt(storage_dt, &shape)? };
    reader.read_exact(tensor.as_bytes_mut()).with_context(|| {
        format!("npy data truncated: shape {:?} of {dt:?} needs {size} bytes", header.shape)
    })?;
    if header.byte_order != NATIVE_ORDER && dt.size_of() > 1 {
        tensor.as_bytes_mut().chunks_exact_mut(dt.size_of()).for_each(|item| item.reverse());
    }
    if dt == DatumType::Bool {
        tensor = tensor.cast_to::<bool>()?.into_owned();
    }
    if header.fortran_order {
        let axes: TVec<usize> = (0..shape.len()).rev().collect();
        tensor.permute_axes(&axes)
    } else {
        Ok(tensor)
    }
}

/// The numpy type descriptor a tensor of this datum type is written with, if any.
///
/// Items are written in the host byte order. Quantized types map to their storage
/// integer type, the quantization parameters are lost.
pub fn npy_descr(dt: DatumType) -> Option<String> {
    let (kind, size) = match dt.unquantized() {
        DatumType::Bool => ('b', 1),
        DatumType::U8 => ('u', 1),
        DatumType::U16 => ('u', 2),
        DatumType::U32 => ('u', 4),
        DatumType::U64 => ('u', 8),
        DatumType::I8 => ('i', 1),
        DatumType::I16 => ('i', 2),
        DatumType::I32 => ('i', 4),
        DatumType::I64 => ('i', 8),
        DatumType::F16 => ('f', 2),
        DatumType::F32 => ('f', 4),
        DatumType::F64 => ('f', 8),
        _ => return None,
    };
    let order = match (size, NATIVE_ORDER) {
        (1, _) => '|',
        (_, ByteOrder::Little) => '<',
        (_, ByteOrder::Big) => '>',
    };
    Some(format!("{order}{kind}{size}"))
}

/// Writes a tensor as a `.npy` stream, laid out exactly as `numpy.save` would.
///
/// Fails on datum types [`npy_descr`] does not cover.
pub fn write_npy(w: &mut impl Write, tensor: &Tensor) -> TractResult<()> {
    let dt = tensor.datum_type();
    let descr = npy_descr(dt).with_context(|| format!("Can not write {dt:?} tensor as npy"))?;
    let data = tensor.try_as_plain_ram()?;
    let shape = match tensor.shape() {
        [d] => format!("({d},)"),
        dims => format!("({})", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let v1_padded = (10 + header.len() + 1).next_multiple_of(64);
    let prefix_len = if v1_padded - 10 <= u16::MAX as usize { 10 } else { 12 };
    let padded = (prefix_len + header.len() + 1).next_multiple_of(64);
    header.extend(std::iter::repeat_n(' ', padded - prefix_len - header.len() - 1));
    header.push('\n');
    w.write_all(MAGIC)?;
    if prefix_len == 10 {
        w.write_all(&[1, 0])?;
        w.write_all(&(header.len() as u16).to_le_bytes())?;
    } else {
        w.write_all(&[2, 0])?;
        w.write_all(&u32::try_from(header.len())?.to_le_bytes())?;
    }
    w.write_all(header.as_bytes())?;
    w.write_all(data.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(text: &str) -> TractResult<Header> {
        parse_header(format!("{text}\n").as_bytes())
    }

    #[test]
    fn numpy_header() {
        let h =
            header("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }      ").unwrap();
        assert_eq!(
            h,
            Header {
                datum_type: DatumType::F32,
                byte_order: ByteOrder::Little,
                fortran_order: false,
                shape: tvec!(3, 4)
            }
        );
    }

    #[test]
    fn shapes() {
        let shape = |s: &str| {
            header(&format!("{{'descr': '|u1', 'fortran_order': False, 'shape': {s}}}"))
                .map(|h| h.shape)
        };
        assert_eq!(shape("()").unwrap(), TVec::<usize>::new());
        assert_eq!(shape("(5,)").unwrap(), tvec!(5));
        assert_eq!(shape("( 5 , )").unwrap(), tvec!(5));
        assert_eq!(shape("(2, 0, 7)").unwrap(), tvec!(2, 0, 7));
        assert!(shape("(,)").is_err());
        assert!(shape("(-1,)").is_err());
        assert!(shape("(1.5,)").is_err());
        assert!(shape("((1,),)").is_err());
        assert!(shape("(99999999999999999999,)").is_err());
    }

    #[test]
    fn descrs() {
        let descr = |d: &str| {
            header(&format!("{{'shape': (), 'fortran_order': True, 'descr': '{d}'}}"))
                .map(|h| (h.datum_type, h.byte_order))
        };
        assert_eq!(descr("|b1").unwrap(), (DatumType::Bool, ByteOrder::Little));
        assert_eq!(descr("<f2").unwrap(), (DatumType::F16, ByteOrder::Little));
        assert_eq!(descr(">i8").unwrap(), (DatumType::I64, ByteOrder::Big));
        assert!(descr("|f4").is_err());
        assert!(descr("=f4").is_err());
        assert!(descr("<c8").is_err());
        assert!(descr("<f16").is_err());
        assert!(descr("<U10").is_err());
        assert!(descr("<f999").is_err());
    }

    #[test]
    fn bad_headers() {
        let ok = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,)}";
        assert!(header(ok).is_ok());
        assert!(parse_header(ok.as_bytes()).is_err());
        assert!(header("{'descr': '<f4', 'fortran_order': False}").is_err());
        assert!(
            header("{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': ()}")
                .is_err()
        );
        assert!(header("{'descr': '<f4', 'fortran_order': False, 'shape': (), 'x': 1}").is_err());
        assert!(header("{\"descr\": '<f4', 'fortran_order': False, 'shape': ()}").is_err());
        assert!(header("{'descr': [('a', '<f4')], 'fortran_order': False, 'shape': ()}").is_err());
        assert!(header("{'descr': '<f4', 'fortran_order': 0, 'shape': ()}").is_err());
        assert!(header("{'descr': '<f4', 'fortran_order': False, 'shape': ()} x").is_err());
        assert!(header("{}").is_err());
    }

    #[test]
    fn nested_garbage_is_rejected_quickly() {
        let nested = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {}{}}}",
            "(".repeat(30000),
            ")".repeat(30000)
        );
        let start = std::time::Instant::now();
        assert!(header(&nested).is_err());
        assert!(start.elapsed().as_secs() < 1);
    }

    fn npy_bytes(header: &str, version: u8, data: &[u8]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend([version, 0]);
        if version == 1 {
            bytes.extend((header.len() as u16).to_le_bytes());
        } else {
            bytes.extend((header.len() as u32).to_le_bytes());
        }
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        bytes
    }

    fn read_all(bytes: &[u8]) -> TractResult<Tensor> {
        read_npy(bytes, bytes.len() as u64)
    }

    #[test]
    fn read_rejects_oversized_claims() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000000,), }\n";
        assert!(read_all(&npy_bytes(header, 1, &[0u8; 16])).is_err());
        let header =
            "{'descr': '<f8', 'fortran_order': False, 'shape': (4611686018427387904, 4), }\n";
        assert!(read_npy(&*npy_bytes(header, 1, &[]), u64::MAX).is_err());
    }

    #[test]
    fn read_rejects_truncated_data() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (8,), }\n";
        assert!(read_npy(&*npy_bytes(header, 1, &[0u8; 16]), u64::MAX).is_err());
    }

    #[test]
    fn read_bool_normalizes() {
        let header = "{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }\n";
        assert_eq!(
            read_all(&npy_bytes(header, 1, &[2, 0, 255])).unwrap(),
            tensor1(&[true, false, true])
        );
    }

    #[test]
    fn read_rejects_huge_header() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend([2, 0]);
        bytes.extend(u32::MAX.to_le_bytes());
        assert!(read_all(&bytes).is_err());
    }

    fn round_trip(t: Tensor) {
        let mut bytes = vec![];
        write_npy(&mut bytes, &t).unwrap();
        assert_eq!((bytes.len() - t.len() * t.datum_type().size_of()) % 64, 0);
        assert_eq!(read_all(&bytes).unwrap(), t);
    }

    #[test]
    fn round_trips() {
        round_trip(tensor0(true));
        round_trip(tensor1(&[0u8, 255]));
        round_trip(tensor1(&[-1i8, 1]));
        round_trip(tensor2(&[[1u16, 2], [3, 4]]));
        round_trip(tensor1(&[u32::MAX]));
        round_trip(tensor1(&[u64::MAX]));
        round_trip(tensor1(&[i16::MIN]));
        round_trip(tensor1(&[i32::MIN]));
        round_trip(tensor1(&[i64::MIN]));
        round_trip(tensor1(&[f16::from_f32(1.5), f16::NEG_INFINITY]));
        round_trip(tensor3(&[[[1.0f32, -2.5]]]));
        round_trip(tensor1(&[std::f64::consts::PI]));
        round_trip(Tensor::zero::<f32>(&[0, 3]).unwrap());
    }

    fn numpy_file(dict: &str, data: &str) -> Vec<u8> {
        let mut header = format!("{dict:<117}\n");
        assert_eq!(header.len(), 118);
        let mut bytes = MAGIC.to_vec();
        bytes.extend([1, 0, 118, 0]);
        bytes.extend(header.drain(..).map(|c| c as u8));
        bytes.extend(
            (0..data.len()).step_by(2).map(|i| u8::from_str_radix(&data[i..][..2], 16).unwrap()),
        );
        bytes
    }

    #[test]
    fn numpy_files() {
        // Produced by `numpy.save` (numpy 2.4).
        let cases = [
            (
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
                "0000803f00000040",
                tensor1(&[1f32, 2.]),
                true,
            ),
            (
                "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 3), }",
                "000000340038003a003c003d",
                tensor2(&[[0f32, 0.25, 0.5], [0.75, 1., 1.25]])
                    .cast_to::<f16>()
                    .unwrap()
                    .into_owned(),
                true,
            ),
            (
                "{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }",
                "010001",
                tensor1(&[true, false, true]),
                true,
            ),
            (
                "{'descr': '<i8', 'fortran_order': False, 'shape': (), }",
                "f9ffffffffffffff",
                tensor0(-7i64),
                true,
            ),
            (
                "{'descr': '|u1', 'fortran_order': False, 'shape': (0, 2), }",
                "",
                Tensor::zero::<u8>(&[0, 2]).unwrap(),
                true,
            ),
            (
                "{'descr': '>u2', 'fortran_order': True, 'shape': (2, 3), }",
                "000100040002000500030006",
                tensor2(&[[1u16, 2, 3], [4, 5, 6]]),
                false,
            ),
        ];
        for (dict, data, tensor, same_bytes) in cases {
            let file = numpy_file(dict, data);
            assert_eq!(read_all(&file).unwrap(), tensor, "{dict}");
            if same_bytes && cfg!(target_endian = "little") {
                let mut written = vec![];
                write_npy(&mut written, &tensor).unwrap();
                assert_eq!(written, file, "{dict}");
            }
        }
    }

    #[test]
    fn write_rejects_unsupported() {
        assert!(write_npy(&mut vec![], &tensor0("foo".to_string())).is_err());
        assert!(write_npy(&mut vec![], &tensor0(TDim::from(3))).is_err());
    }
}
