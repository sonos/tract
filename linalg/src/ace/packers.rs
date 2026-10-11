//! Portable ACE packers: the bf16 K=2-inner packer and the unified MX
//! scaled-block packer. Both are pure Rust (no `#[cfg]`), so they build and are
//! validated on any target — the real ACE kernel will consume these exact byte
//! layouts.
//!
//! The MX packer is the crux of ACE integration in tract: `FusedKerSpec::AddMatMul`
//! carries only `{k, pa, pb, packing}` with no slot for the Block Scale Register,
//! so the per-block E8M0 scales must travel *inside* the packed byte stream. We
//! place them in a contiguous tail region after the elements, which keeps the
//! element region byte-identical to a run of `PackedI8K4` K-blocks (so the kernel
//! reads each MX block as a clean 512-byte `[elem; 512]`) and matches how real ACE
//! loads the 16-wide scale strip into the Block Scale Register once per block.

use tract_data::internal::*;

use super::format::{AceMxElem, f32_to_bf16_rne, quantize_mx_block};
use super::isa::{ACE_MX_BLOCK_K, ACE_TILE_DIM};
use crate::WeightType;
use crate::frame::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage,
};

// ===========================================================================
// bf16, K=2-inner (portable clone of x86_64_fma::amx_bf16::PackedBf16K2)
// ===========================================================================

/// bf16 K=2-inner packer for ACE `top2bf16ps`. Source is f32, truncated to bf16
/// (round-to-nearest-even) at pack time. Per K=2 block of an r-row panel: byte
/// `(kb*r*2 + lm*2 + ki)*2` holds the bf16 of `src[2*kb+ki, panel_row + lm]`,
/// i.e. one ZMM = 16×2 bf16 in lane order `row*2 + ki` — exactly what
/// [`super::isa::ace_top2bf16ps`] consumes.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedAceBf16K2 {
    pub r: usize,
    pub align: usize,
}

impl PackedAceBf16K2 {
    pub fn new(r: usize) -> Self {
        PackedAceBf16K2 { r, align: 64 }
    }
    fn k_padded(&self, k: usize) -> usize {
        k.div_ceil(2) * 2
    }
    fn panel(&self, k: usize) -> usize {
        self.k_padded(k) * self.r * 2
    }
    pub fn pack_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let kp = self.k_padded(k);
        let pl = kp * self.r * 2;
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * pl, self.align) };
        blob.as_bytes_mut().fill(0); // zero bf16 = +0.0, safe K/mn tail
        let kblocks = kp / 2;
        unsafe {
            let src = t.as_ptr_unchecked::<f32>();
            let dst = blob.as_mut_ptr() as *mut u16;
            for p in 0..panels {
                let pw = self.r.min(mn - p * self.r);
                let panel = dst.add(p * (kp * self.r));
                let mn0 = (p * self.r) as isize;
                for kb in 0..kblocks {
                    for ki in 0..2 {
                        let kk = kb * 2 + ki;
                        if kk >= k {
                            break;
                        }
                        let srow = src.offset(kk as isize * ks + mn0 * ms);
                        let dblock = panel.add(kb * self.r * 2 + ki);
                        for lm in 0..pw {
                            *dblock.add(lm * 2) = f32_to_bf16_rne(*srow.offset(lm as isize * ms));
                        }
                    }
                }
            }
        }
        Ok(Box::new(EagerPackedInput {
            fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
            packed: blob.into(),
            panel_bytes: pl,
            mn,
        }))
    }
}

impl std::fmt::Display for PackedAceBf16K2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AceBf16K2[{}]", self.r)
    }
}

impl MMMInputFormat for PackedAceBf16K2 {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        Ok(PackedMatrixStorage::new(self.prepare_one(t, k_axis, mn_axis)?)
            .into_tensor(t.datum_type()))
    }
    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        self.pack_view(&t.view(), k_axis, mn_axis)
    }
    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }
    fn r(&self) -> usize {
        self.r
    }
    fn k_alignment(&self) -> usize {
        2
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedAceBf16K2>().filter(|x| x.r == self.r).map(|_| self as _)
    }
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        mn.divceil(self.r) * self.panel(k.to_usize().unwrap_or(0))
    }
    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }
    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

// ===========================================================================
// MX scaled block (MXFP8 / MXINT8), K=4-inner elements + tail scale strip
// ===========================================================================

/// Unified OCP-MX packer for MXFP8 (E4M3 elements) and MXINT8 (i8 elements).
/// Source is f32, quantized to MX at pack time via [`quantize_mx_block`].
///
/// Per r-row panel, two contiguous regions (for `kp = ceil(k/32)*32`, `nb = kp/32`):
///   * ELEMENTS (offset 0, `kp*r` bytes) — K=4-inner like `PackedI8K4`:
///     byte `(k/4)*(r*4) + row*4 + (k%4)`. Each MX block is a clean `r*32 = 512`-byte
///     run (`block_base = blk*512`) consumed by the `ace_top_mx*_block` kernels.
///   * SCALES (offset `kp*r`, `nb*r` bytes) — one E8M0 byte per (row, block):
///     byte `kp*r + blk*r + row`. Padding (mn-tail rows, K-tail) keeps scale = 127
///     (E8M0 2^0) and elements = 0, so padded lanes contribute `0*0*1 = 0`, never NaN.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedAceScaledBlock {
    pub r: usize,
    pub elem: AceMxElem,
    pub align: usize,
}

impl PackedAceScaledBlock {
    pub fn mxfp8(r: usize) -> Self {
        // The kernels' block readers assume r == tile dim (block = r*32 = 512 B,
        // scale tail at nb*512); see mmm.rs ace_matmul_mx*.
        debug_assert_eq!(r, ACE_TILE_DIM, "ACE MX packer requires r == tile dim (16)");
        PackedAceScaledBlock { r, elem: AceMxElem::MxFp8, align: 64 }
    }
    pub fn mxint8(r: usize) -> Self {
        debug_assert_eq!(r, ACE_TILE_DIM, "ACE MX packer requires r == tile dim (16)");
        PackedAceScaledBlock { r, elem: AceMxElem::MxInt8, align: 64 }
    }
    fn panel(&self, k: usize) -> usize {
        let kp = k.next_multiple_of(ACE_MX_BLOCK_K);
        let nb = kp / ACE_MX_BLOCK_K;
        (kp * self.r + nb * self.r).next_multiple_of(self.align)
    }
    pub fn pack_view(
        &self,
        t: &TensorView,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        let k = t.shape()[k_axis];
        let mn = t.shape()[mn_axis];
        let kp = k.next_multiple_of(ACE_MX_BLOCK_K);
        let nb = kp / ACE_MX_BLOCK_K;
        let elem_bytes = kp * self.r;
        let scale_bytes = nb * self.r;
        let panel_bytes = self.panel(k);
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * panel_bytes, self.align) };
        blob.as_bytes_mut().fill(0);
        unsafe {
            let src = t.as_ptr_unchecked::<f32>();
            let base = blob.as_mut_ptr();
            for p in 0..panels {
                let pw = self.r.min(mn - p * self.r);
                let panel = base.add(p * panel_bytes);
                let scales = panel.add(elem_bytes);
                // E8M0 1.0 everywhere first -> padded rows/blocks stay benign (never 0xFF).
                std::slice::from_raw_parts_mut(scales, scale_bytes).fill(127);
                for row in 0..pw {
                    let mn_i = (p * self.r + row) as isize;
                    for blk in 0..nb {
                        let mut vals = [0f32; ACE_MX_BLOCK_K];
                        for (i, vv) in vals.iter_mut().enumerate() {
                            let kk = blk * ACE_MX_BLOCK_K + i;
                            if kk < k {
                                *vv = *src.offset(kk as isize * ks + mn_i * ms);
                            }
                        }
                        let mut ebytes = [0u8; ACE_MX_BLOCK_K];
                        *scales.add(blk * self.r + row) =
                            quantize_mx_block(&vals, self.elem, &mut ebytes);
                        let blk_base = panel.add(blk * self.r * ACE_MX_BLOCK_K);
                        for (i, &eb) in ebytes.iter().enumerate() {
                            *blk_base.add((i / 4) * (self.r * 4) + row * 4 + (i % 4)) = eb;
                        }
                    }
                }
            }
        }
        Ok(Box::new(EagerPackedInput {
            fact: PackedExoticFact { format: Box::new(self.clone()), mn: mn.to_dim(), k },
            packed: blob.into(),
            panel_bytes,
            mn,
        }))
    }
}

impl std::fmt::Display for PackedAceScaledBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match self.elem {
            AceMxElem::MxFp8 => "MxFp8",
            AceMxElem::MxInt8 => "MxInt8",
        };
        write!(f, "{}[{}]", name, self.r)
    }
}

impl MMMInputFormat for PackedAceScaledBlock {
    fn prepare_tensor(&self, t: &Tensor, k_axis: usize, mn_axis: usize) -> TractResult<Tensor> {
        Ok(PackedMatrixStorage::new(self.prepare_one(t, k_axis, mn_axis)?)
            .into_tensor(t.datum_type()))
    }
    fn prepare_one(
        &self,
        t: &Tensor,
        k_axis: usize,
        mn_axis: usize,
    ) -> TractResult<Box<dyn MMMInputValue>> {
        self.pack_view(&t.view(), k_axis, mn_axis)
    }
    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }
    fn r(&self) -> usize {
        self.r
    }
    fn k_alignment(&self) -> usize {
        ACE_MX_BLOCK_K // 32: pad K to whole MX blocks
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedAceScaledBlock>()
            .filter(|x| x.r == self.r && x.elem == self.elem)
            .map(|_| self as _)
    }
    fn mem_size(&self, k: TDim, mn: TDim) -> TDim {
        mn.divceil(self.r) * self.panel(k.to_usize().unwrap_or(0))
    }
    fn extract_at_mn_f16(&self, _: &EagerPackedInput, _: usize, _: &mut [f16]) -> TractResult<()> {
        bail!("no f16 extract")
    }
    fn extract_at_mn_f32(&self, _: &EagerPackedInput, _: usize, _: &mut [f32]) -> TractResult<()> {
        bail!("no f32 extract")
    }
}

// `r` must be the ACE tile width for the kernels' fixed-shape block readers.
const _: () = assert!(ACE_TILE_DIM == 16);
