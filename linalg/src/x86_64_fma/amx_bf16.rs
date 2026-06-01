// Intel AMX bf16 support: f32 -> bf16 packers and the AMX bf16 runtime gate.
//
// The kernel `avx512amx_mmm_f32_16x16` uses TDPBF16PS (bf16 x bf16 -> f32) to
// accelerate f32 matmul on Sapphire Rapids+ AMX hardware. The inputs are
// truncated from f32 to bf16 at pack time (round-to-nearest-even, matching
// Intel's VCVTNEPS2BF16 semantics); the f32 accumulators are bit-identical
// to a "scalar bf16 multiply + f32 accumulate" reference but DIFFER from a
// pure-f32 FMA reference by ~1 / 2^8 relative per multiply (bf16 has 8
// mantissa bits vs f32's 23). This precision loss is the same as oneDNN
// "fast-math" f32 matmul on AMX and is acceptable for inference workloads
// (LLMs, CNNs) that already tolerate bf16.
//
// Tile geometry mirrors the i32 16x16 kernel: 16 rows x 64 colsb per tile.
// Per TDPBF16PS: 16 M-rows x 16 N-cols x 32 K-bf16 = 8192 fma operations
// per single instruction -- the same throughput as TDPBSSD.

use std::sync::OnceLock;

use tract_data::internal::*;

use crate::WeightType;
use crate::frame::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage,
};

/// Detect AMX-BF16 + AMX-TILE via CPUID leaf 7 sub-leaf 0 (EDX bits 22, 24).
/// AMX-BF16 is the bit-22 capability; AMX-TILE (bit 24) is mandatory for any
/// AMX use. Returns false unless both are present.
fn cpu_has_amx_bf16() -> bool {
    if !std::is_x86_feature_detected!("avx512f") {
        return false;
    }
    let r = std::arch::x86_64::__cpuid_count(7, 0);
    const AMX_BF16: u32 = 1 << 22;
    const AMX_TILE: u32 = 1 << 24;
    (r.edx & AMX_BF16) != 0 && (r.edx & AMX_TILE) != 0
}

/// Returns true iff Intel AMX bf16 is available AND the OS has granted this
/// process permission to use the AMX tile-data XSAVE state. Reuses the
/// arch_prctl XCOMP-perm request mechanism from the int8 path -- the same
/// XFEATURE_XTILEDATA permission gates both data types.
pub fn has_amx_bf16() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| cpu_has_amx_bf16() && super::amx::request_amx_tile_xcomp_perm())
}

/// Convert an f32 to bf16 with round-to-nearest-even (matches Intel's
/// VCVTNEPS2BF16). NaN inputs are preserved as quiet NaN. Used by the bf16
/// packers below (scalar; AMX hardware is on Sapphire Rapids+ which has the
/// AVX-512-BF16 intrinsic for batched conversion, but packing is amortised
/// over many kernel calls so the scalar path is fine).
#[inline]
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    // NaN check: exponent all-ones and mantissa nonzero.
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        // Quiet NaN: set the top mantissa bit of the bf16 result.
        ((bits >> 16) as u16) | 0x0040
    } else {
        // round-to-nearest-even: add 0x7FFF + (lsb of bf16) before truncating.
        let lsb = (bits >> 16) & 1;
        let rounding = 0x0000_7FFF + lsb;
        (bits.wrapping_add(rounding) >> 16) as u16
    }
}

/// AMX-friendly A packing for f32 matmul via bf16. Per `r`-row panel, the
/// M-rows are laid out row-major in bf16 across `K_padded` contiguous bf16
/// per row (K_padded = ceil(K/32)*32, so each row is a whole number of
/// AMX K-tile widths). Source is f32; conversion happens at pack time.
///
///   panel_bytes = r * K_padded * 2  (each bf16 = 2 bytes)
///
/// AMX `tileloadd` with stride = K_padded*2 reads exactly 16 M-rows of
/// 64 bytes (= 32 bf16) per call -- one inner-K iter's worth.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedAmxBf16A {
    pub r: usize,
    pub align: usize,
}

impl PackedAmxBf16A {
    pub fn new(r: usize) -> Self {
        PackedAmxBf16A { r, align: 64 }
    }
    fn k_padded(&self, k: usize) -> usize {
        k.div_ceil(32) * 32
    }
    fn panel(&self, k: usize) -> usize {
        self.k_padded(k) * self.r * 2
    }
    pub fn single_panel_len(&self, k: usize) -> usize {
        self.panel(k)
    }
    pub fn len(&self, k: usize, mn: usize) -> usize {
        mn.div_ceil(self.r) * self.panel(k)
    }
    pub fn alignment(&self) -> usize {
        self.align
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
        let pl = kp * self.r * 2; // bytes per panel
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * pl, self.align) };
        blob.as_bytes_mut().fill(0);
        unsafe {
            let src = t.as_ptr_unchecked::<f32>();
            let dst = blob.as_mut_ptr() as *mut u16;
            for p in 0..panels {
                let pw = self.r.min(mn - p * self.r);
                let panel = dst.add(p * (kp * self.r)); // panel_offset in u16 elements
                let mn0 = (p * self.r) as isize;
                for lm in 0..pw {
                    let drow = panel.add(lm * kp);
                    let srow_base = src.offset((mn0 + lm as isize) * ms);
                    for kk in 0..k {
                        let v = *srow_base.offset(kk as isize * ks);
                        *drow.add(kk) = f32_to_bf16_rne(v);
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

impl std::fmt::Display for PackedAmxBf16A {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AmxBf16A[{}]", self.r)
    }
}

impl MMMInputFormat for PackedAmxBf16A {
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
    fn k_alignment(&self) -> usize {
        // tdpbf16ps consumes 32 bf16 per K-step.
        32
    }
    fn r(&self) -> usize {
        self.r
    }
    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedAmxBf16A>().filter(|x| x.r == self.r).map(|_| self as _)
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

/// AMX-friendly B packing for f32 matmul via bf16 (analog of PackedI8K4 but
/// K=2-inner instead of K=4-inner -- tdpbf16ps groups 2 bf16 per K-step).
///
/// Per K=2 block: r N-cols x 2 K-bf16 = r * 2 * 2 bytes = 4r bytes.
/// Block layout: byte (n*4 + ki*2..(n*4 + ki*2 + 2)) = bf16 of B[2kb+ki, n].
/// For r=16: 64 bytes per K=2 block, 16 blocks per K=32 AMX tile -> 1024 B.
///
/// One AMX `tileloadd` with stride = 4r bytes reads 16 K-pair-rows of
/// r * 4 bytes each = one inner-K iter's worth of B.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedBf16K2 {
    pub r: usize,
    pub align: usize,
}

impl PackedBf16K2 {
    pub fn new(r: usize) -> Self {
        PackedBf16K2 { r, align: 64 }
    }
    fn k_padded(&self, k: usize) -> usize {
        k.div_ceil(2) * 2
    }
    fn panel(&self, k: usize) -> usize {
        self.k_padded(k) * self.r * 2
    }
    pub fn single_panel_len(&self, k: usize) -> usize {
        self.panel(k)
    }
    pub fn len(&self, k: usize, mn: usize) -> usize {
        mn.div_ceil(self.r) * self.panel(k)
    }
    pub fn alignment(&self) -> usize {
        self.align
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
        let pl = kp * self.r * 2; // bytes per panel
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * pl, self.align) };
        blob.as_bytes_mut().fill(0);
        let (ks, ms) = (st[k_axis], st[mn_axis]);
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
                            let v = *srow.offset(lm as isize * ms);
                            *dblock.add(lm * 2) = f32_to_bf16_rne(v);
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

impl std::fmt::Display for PackedBf16K2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Bf16K2[{}]", self.r)
    }
}

impl MMMInputFormat for PackedBf16K2 {
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
    fn k_alignment(&self) -> usize {
        2
    }
    fn r(&self) -> usize {
        self.r
    }
    fn precursor(&self) -> WeightType {
        WeightType::Plain(f32::datum_type())
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedBf16K2>().filter(|x| x.r == self.r).map(|_| self as _)
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
