//! Panel packing for block-quant formats whose bytes stay in ggml order.
//! A panel is `r` rows of native blocks. Extraction dequantizes into the
//! `[k, r]` panel the f32/f16 matmul kernels consume.
use super::*;
use crate::mmm::PackedExoticFact;
use crate::pack::PackedFormat;
use num_traits::Zero;

pub fn pack_opaque(
    bq: &dyn BlockQuant,
    input: &[u8],
    k: usize,
    r: usize,
    zip: usize,
    scales_at_end: bool,
) -> TractResult<EagerPackedInput> {
    ensure!(zip == 0, "opaque block-quant packing does not interleave rows");
    ensure!(k % bq.block_len() == 0, "K {} is not a multiple of block {}", k, bq.block_len());
    ensure!(input.len() % bq.block_bytes() == 0);
    let m = if input.is_empty() || k == 0 {
        0
    } else {
        input.len() / bq.block_bytes() * bq.block_len() / k
    };
    let panels = m.div_ceil(r);
    let blocks_for_k = k / bq.block_len();
    let row_bytes = blocks_for_k * bq.block_bytes();
    let panel_bytes = row_bytes * r;
    let mut blob = unsafe { Blob::for_layout(Layout::from_size_align(panel_bytes * panels, 128)?) };
    for p in 0..panels {
        for row in 0..r {
            let src_row = p * r + row;
            if src_row >= m {
                break;
            }
            let src = &input[src_row * row_bytes..][..row_bytes];
            let dst_at = p * panel_bytes + row * row_bytes;
            blob[dst_at..dst_at + row_bytes].copy_from_slice(src);
        }
    }
    Ok(EagerPackedInput {
        fact: PackedExoticFact {
            format: Box::new(PackedBlockQuantFormat {
                bq: dyn_clone::clone_box(bq),
                r,
                zip,
                scales_at_end,
            }),
            mn: m.to_dim(),
            k,
        },
        packed: blob.into(),
        panel_bytes,
        mn: m,
    })
}

pub unsafe fn extract_packed_panel(
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
        .context("opaque extract expected a PackedBlockQuantFormat")?;
    ensure!(pbqf.r == target.r);
    ensure!(*pbqf.bq == *bq);
    let row_bytes = (value.fact.k / bq.block_len()) * bq.block_bytes();
    let panel_ptr = unsafe { value.packed.as_ptr().add(panel * pbqf.r * row_bytes) };
    unsafe {
        if target.dt == f32::datum_type() {
            dequant_panel_f32(bq, pbqf.r, panel_ptr, value.fact.k, scratch as *mut f32);
        } else if target.dt == f16::datum_type() {
            dequant_panel_f16(bq, pbqf.r, panel_ptr, value.fact.k, scratch as *mut f16);
        } else {
            bail!("opaque block-quant extract supports f32 and f16, got {:?}", target.dt);
        }
    }
    Ok(())
}

pub unsafe fn dequant_panel_f32(
    bq: &dyn BlockQuant,
    r: usize,
    input: *const u8,
    k: usize,
    output: *mut f32,
) {
    if k == 0 || r == 0 {
        return;
    }
    let blocks = k / bq.block_len();
    let row_bytes = blocks * bq.block_bytes();
    let src = unsafe { std::slice::from_raw_parts(input, r * row_bytes) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, k * r) };
    let mut tmp = [0.0f32; 256];
    let bl = bq.block_len();
    for row in 0..r {
        for b in 0..blocks {
            let q = &src[row * row_bytes + b * bq.block_bytes()..][..bq.block_bytes()];
            bq.dequant_block_f32(q, &mut tmp[..bl]);
            for i in 0..bl {
                dst[(b * bl + i) * r + row] = tmp[i];
            }
        }
    }
}

pub unsafe fn dequant_panel_f16(
    bq: &dyn BlockQuant,
    r: usize,
    input: *const u8,
    k: usize,
    output: *mut f16,
) {
    if k == 0 || r == 0 {
        return;
    }
    let blocks = k / bq.block_len();
    let row_bytes = blocks * bq.block_bytes();
    let src = unsafe { std::slice::from_raw_parts(input, r * row_bytes) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, k * r) };
    let mut tmp = [f16::zero(); 256];
    let bl = bq.block_len();
    for row in 0..r {
        for b in 0..blocks {
            let q = &src[row * row_bytes + b * bq.block_bytes()..][..bq.block_bytes()];
            bq.dequant_block_f16(q, &mut tmp[..bl]);
            for i in 0..bl {
                dst[(b * bl + i) * r + row] = tmp[i];
            }
        }
    }
}

pub fn extract_at_mn_f32(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    mn: usize,
    target: &mut [f32],
) -> TractResult<()> {
    extract_at_mn(bq, value, mn, target, true)
}

pub fn extract_at_mn_f16(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    mn: usize,
    target: &mut [f16],
) -> TractResult<()> {
    extract_at_mn(bq, value, mn, target, false)
}

fn extract_at_mn<T: Datum + Copy>(
    bq: &dyn BlockQuant,
    value: &EagerPackedInput,
    mn: usize,
    target: &mut [T],
    f32_out: bool,
) -> TractResult<()> {
    let pbqf = value.fact.format.downcast_ref::<PackedBlockQuantFormat>().unwrap();
    ensure!(value.fact.k == target.len());
    ensure!(mn < value.mn);
    let blocks = value.fact.k / bq.block_len();
    let row_bytes = blocks * bq.block_bytes();
    let panel = mn / pbqf.r;
    let row = mn % pbqf.r;
    let src = &value.packed[panel * pbqf.r * row_bytes + row * row_bytes..][..row_bytes];
    if f32_out {
        let mut tmp = [0.0f32; 256];
        let dst = unsafe {
            std::slice::from_raw_parts_mut(target.as_mut_ptr() as *mut f32, target.len())
        };
        for b in 0..blocks {
            let q = &src[b * bq.block_bytes()..][..bq.block_bytes()];
            bq.dequant_block_f32(q, &mut tmp[..bq.block_len()]);
            dst[b * bq.block_len()..][..bq.block_len()].copy_from_slice(&tmp[..bq.block_len()]);
        }
    } else {
        let mut tmp = [f16::zero(); 256];
        let dst = unsafe {
            std::slice::from_raw_parts_mut(target.as_mut_ptr() as *mut f16, target.len())
        };
        for b in 0..blocks {
            let q = &src[b * bq.block_bytes()..][..bq.block_bytes()];
            bq.dequant_block_f16(q, &mut tmp[..bq.block_len()]);
            dst[b * bq.block_len()..][..bq.block_len()].copy_from_slice(&tmp[..bq.block_len()]);
        }
    }
    Ok(())
}
