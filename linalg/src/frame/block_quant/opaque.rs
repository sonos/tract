//! Packing for block-quant formats whose bytes stay in ggml block order.
//!
//! A packed panel is `r` consecutive rows of native blocks, so the packed blob is the
//! row-major input followed by zeroed padding rows. Extraction dequantizes a panel
//! into the `[k, r]` layout the f32 and f16 matmul kernels consume.
use super::*;
use crate::mmm::PackedExoticFact;

pub(super) fn store_f16(dst: &mut [u8], v: f32) {
    dst[..2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

pub(super) fn load_f16(src: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32()
}

fn row_bytes(bq: &dyn BlockQuant, k: usize) -> usize {
    k / bq.block_len() * bq.block_bytes()
}

/// Packs `input`, rows of `k` values in `bq` blocks, into panels of `r` rows.
pub(super) fn pack(
    bq: &dyn BlockQuant,
    input: &[u8],
    k: usize,
    r: usize,
    zip: usize,
    scales_at_end: bool,
) -> TractResult<EagerPackedInput> {
    ensure!(zip == 0, "{bq} packing does not interleave rows");
    ensure!(k > 0 && k % bq.block_len() == 0, "K {k} is not a multiple of {bq} blocks");
    let row_bytes = row_bytes(bq, k);
    ensure!(input.len() % row_bytes == 0);
    let m = input.len() / row_bytes;
    let panel_bytes = row_bytes * r;
    let mut blob =
        unsafe { Blob::for_layout(Layout::from_size_align(panel_bytes * m.div_ceil(r), 128)?) };
    blob[..input.len()].copy_from_slice(input);
    blob[input.len()..].fill(0);
    Ok(EagerPackedInput {
        fact: PackedExoticFact {
            format: Box::new(PackedBlockQuantFormat::new(bq, r, zip, scales_at_end)),
            mn: m.to_dim(),
            k,
        },
        packed: blob.into(),
        panel_bytes,
        mn: m,
    })
}

/// Dequantizes `panel` of `value` into `scratch`, a `[k, r]` buffer of `target.dt`.
///
/// # Safety
///
/// `scratch` must be valid for writes of `k * r` values of `target.dt`.
pub(super) unsafe fn extract_packed_panel(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    target: &PackedFormat,
    panel: usize,
    scratch: *mut u8,
) -> TractResult<()> {
    let pbqf = value
        .fact
        .format
        .downcast_ref::<PackedBlockQuantFormat>()
        .context("expected a PackedBlockQuantFormat")?;
    ensure!(pbqf.r == target.r);
    ensure!(*pbqf.bq == *bq);
    let (k, r) = (value.fact.k, pbqf.r);
    let src = &value.packed[panel * value.panel_bytes..][..value.panel_bytes];
    if target.dt == f32::datum_type() {
        let out = unsafe { std::slice::from_raw_parts_mut(scratch as *mut f32, k * r) };
        dequant_panel(bq, src, r, out, |q, b| bq.dequant_block_f32(q, b));
    } else if target.dt == f16::datum_type() {
        let out = unsafe { std::slice::from_raw_parts_mut(scratch as *mut f16, k * r) };
        dequant_panel(bq, src, r, out, |q, b| bq.dequant_block_f16(q, b));
    } else {
        bail!("{bq} panels extract to f32 or f16, not {:?}", target.dt);
    }
    Ok(())
}

fn dequant_panel<T: Copy + Zero>(
    bq: &dyn BlockQuant,
    panel: &[u8],
    r: usize,
    out: &mut [T],
    dequant_block: impl Fn(&[u8], &mut [T]),
) {
    let bl = bq.block_len();
    let row_bytes = panel.len() / r;
    let mut block = vec![T::zero(); bl];
    for (row, src) in panel.chunks_exact(row_bytes).enumerate() {
        for (b, q) in src.chunks_exact(bq.block_bytes()).enumerate() {
            dequant_block(q, &mut block);
            for (i, v) in block.iter().enumerate() {
                out[(b * bl + i) * r + row] = *v;
            }
        }
    }
}

fn packed_row<'a>(
    bq: &dyn BlockQuant,
    value: &'a EagerPackedInput,
    mn: usize,
    k: usize,
) -> TractResult<&'a [u8]> {
    ensure!(value.fact.k == k);
    ensure!(mn < value.mn);
    let row_bytes = row_bytes(bq, k);
    Ok(&value.packed[mn * row_bytes..][..row_bytes])
}

/// Dequantizes row `mn` of `value` into `target`, which holds `k` values.
pub(super) fn extract_at_mn_f32(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    mn: usize,
    target: &mut [f32],
) -> TractResult<()> {
    let row = packed_row(bq, value, mn, target.len())?;
    for (q, block) in
        row.chunks_exact(bq.block_bytes()).zip(target.chunks_exact_mut(bq.block_len()))
    {
        bq.dequant_block_f32(q, block);
    }
    Ok(())
}

/// Dequantizes row `mn` of `value` into `target`, which holds `k` values.
pub(super) fn extract_at_mn_f16(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    mn: usize,
    target: &mut [f16],
) -> TractResult<()> {
    let row = packed_row(bq, value, mn, target.len())?;
    for (q, block) in
        row.chunks_exact(bq.block_bytes()).zip(target.chunks_exact_mut(bq.block_len()))
    {
        bq.dequant_block_f16(q, block);
    }
    Ok(())
}

/// Implements [`BlockQuant`] for a ggml block of `$len` values in `$bytes` bytes.
///
/// `$quant` and `$dequant` are the f32 reference routines, `fn(&[f32], &mut [u8])` and
/// `fn(&[u8], &mut [f32])`. f16 goes through f32, and packing keeps the ggml bytes.
macro_rules! impl_opaque_block_quant {
    ($ty:ty, $name:literal, $len:expr, $bytes:expr, $quant:path, $dequant:path) => {
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str($name)
            }
        }

        impl BlockQuant for $ty {
            fn block_len(&self) -> usize {
                $len
            }

            fn block_bytes(&self) -> usize {
                $bytes
            }

            fn quant_block_f32(&self, block: &[f32], quant: &mut [u8]) {
                assert_eq!(block.len(), $len);
                assert_eq!(quant.len(), $bytes);
                $quant(block, quant);
            }

            fn quant_block_f16(&self, block: &[f16], quant: &mut [u8]) {
                let mut tmp = [0f32; $len];
                for (dst, src) in tmp.iter_mut().zip(block) {
                    *dst = src.to_f32();
                }
                self.quant_block_f32(&tmp, quant);
            }

            fn dequant_block_f32(&self, quant: &[u8], block: &mut [f32]) {
                assert_eq!(block.len(), $len);
                assert_eq!(quant.len(), $bytes);
                $dequant(quant, block);
            }

            fn dequant_block_f16(&self, quant: &[u8], block: &mut [f16]) {
                let mut tmp = [0f32; $len];
                self.dequant_block_f32(quant, &mut tmp);
                for (dst, src) in block.iter_mut().zip(tmp) {
                    *dst = f16::from_f32(src);
                }
            }

            fn pack(
                &self,
                input: &[u8],
                k: usize,
                r: usize,
                zip: usize,
                scales_at_end: bool,
            ) -> TractResult<EagerPackedInput> {
                super::opaque::pack(self, input, k, r, zip, scales_at_end)
            }

            unsafe fn extract_packed_panel(
                &self,
                value: &EagerPackedInput,
                target: &PackedFormat,
                panel: usize,
                scratch: *mut u8,
            ) -> TractResult<()> {
                unsafe { super::opaque::extract_packed_panel(self, value, target, panel, scratch) }
            }

            fn extract_at_mn_f16(
                &self,
                value: &EagerPackedInput,
                mn: usize,
                target: &mut [f16],
            ) -> TractResult<()> {
                super::opaque::extract_at_mn_f16(self, value, mn, target)
            }

            fn extract_at_mn_f32(
                &self,
                value: &EagerPackedInput,
                mn: usize,
                target: &mut [f32],
            ) -> TractResult<()> {
                super::opaque::extract_at_mn_f32(self, value, mn, target)
            }
        }
    };
}
pub(super) use impl_opaque_block_quant;

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_then_extract(bq: &dyn BlockQuant) -> TractResult<()> {
        let (m, r) = (5, 4);
        let k = 2 * bq.block_len();
        let input: Vec<f32> = (0..m * k).map(|i| ((i * 31 % 41) as f32 - 20.0) / 8.0).collect();
        let quant = bq.quant_f32(&input)?;
        let reference = bq.dequant_f32(&quant)?.into_shape(&[m, k])?;
        let packed = bq.pack(&quant, k, r, 0, false)?;
        let f32_packer = PackedFormat::new(f32::datum_type(), r, 128);
        let f32_packed = f32_packer.pack_tensor(&reference, 1, 0)?;
        for panel in 0..f32_packed.panels_count() {
            let mut extracted = vec![0f32; k * r];
            unsafe {
                bq.extract_packed_panel(&packed, &f32_packer, panel, extracted.as_mut_ptr() as _)?;
                let expected = f32_packed.panel_bytes(panel, None)? as *const f32;
                assert_eq!(extracted, std::slice::from_raw_parts(expected, k * r));
            }
        }
        let reference = reference.try_as_plain_ram()?;
        let reference = reference.as_slice::<f32>()?;
        for mn in 0..m {
            let expected = &reference[mn * k..][..k];
            let mut row = vec![0f32; k];
            bq.extract_at_mn_f32(&packed, mn, &mut row)?;
            assert_eq!(row, expected);
            let mut row = vec![f16::zero(); k];
            bq.extract_at_mn_f16(&packed, mn, &mut row)?;
            assert!(row.iter().zip(expected).all(|(a, b)| *a == f16::from_f32(*b)));
        }
        Ok(())
    }

    #[test]
    fn q8_0_pack_then_extract() -> TractResult<()> {
        pack_then_extract(&Q8_0)
    }

    #[test]
    fn q4_k_pack_then_extract() -> TractResult<()> {
        pack_then_extract(&Q4_K)
    }
}
