//! Wire (de)serialization: sub-models as NNEF, activation tensors as
//! self-describing `.dat`, both wrapped in length-prefixed frames.

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};
use tract_core::prelude::*;
use tract_nnef::framework::Nnef;
use tract_nnef::tensors::{read_tensor, write_tensor};
use tract_transformers::WithTractTransformers;

/// NNEF framework extended with the transformer op registry — REQUIRED on both
/// the writing and reading side, or LLM ops (SDPA / KV-cache / RoPE) fail to
/// round-trip.
pub fn nnef() -> Nnef {
    tract_nnef::nnef().with_tract_transformers()
}

/// Serialize a sub-model to an in-memory NNEF tar (uncompressed).
pub fn model_to_bytes(model: &TypedModel) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    nnef().write(model, &mut buf)?;
    Ok(buf)
}

/// Inverse of [`model_to_bytes`].
pub fn model_from_bytes(bytes: &[u8]) -> Result<TypedModel> {
    nnef().model_for_read(&mut std::io::Cursor::new(bytes))
}

/// Serialize one activation tensor (self-describing: carries dtype + shape).
pub fn tensor_to_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_tensor(&mut buf, t)?;
    Ok(buf)
}

/// Inverse of [`tensor_to_bytes`].
pub fn tensor_from_bytes(bytes: &[u8]) -> Result<Tensor> {
    read_tensor(std::io::Cursor::new(bytes))
}

/// Write `payload` as a `u64`-length-prefixed frame.
pub fn write_frame(w: &mut impl Write, payload: &[u8]) -> Result<()> {
    w.write_u64::<LittleEndian>(payload.len() as u64)?;
    w.write_all(payload)?;
    Ok(())
}

/// Read one frame written by [`write_frame`].
pub fn read_frame(r: &mut impl Read) -> Result<Vec<u8>> {
    let len = r.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Write a count-prefixed batch of tensor frames into a stream. Takes owned
/// `Tensor`s (the wire form) — `Tensor` is `Send`, unlike `TValue`.
pub fn write_tensors(w: &mut impl Write, tensors: &[Tensor]) -> Result<()> {
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    for t in tensors {
        write_frame(w, &tensor_to_bytes(t)?)?;
    }
    Ok(())
}

/// Read a batch written by [`write_tensors`].
pub fn read_tensors(r: &mut impl Read) -> Result<TVec<Tensor>> {
    let count = r.read_u64::<LittleEndian>()? as usize;
    let mut out = tvec!();
    for _ in 0..count {
        let frame = read_frame(r)?;
        out.push(tensor_from_bytes(&frame)?);
    }
    Ok(out)
}
