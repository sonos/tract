// Intel AMX int8 support: A packing format and runtime gate.
//
// The kernel `avx512amx_mmm_i32_8x8` uses TDPBSSD (signed-signed). Per
// iteration of its inner loop it consumes one 8x64-byte A tile and one
// 16x32-byte B tile and updates an 8x8 i32 C tile. The B-side packing
// matches the existing K=4-inner `PackedI8K4` layout, so it is reused
// unchanged. The A-side packing is novel: AMX's tile-A semantics require
// M-major-within-panel row-major bytes, which is incompatible with the
// K-major-outer `PackedI8K4`. `PackedAmxA` below produces that layout.
//
// Runtime gate: CPUID `amx-int8` is necessary but not sufficient on Linux —
// the kernel must also call `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`
// to receive AMX tile-data XSAVE permission from the kernel before any tile
// instruction can run. `has_amx_int8()` performs both checks once and caches
// the result; it returns false on non-Linux even if CPUID reports AMX.

use std::sync::OnceLock;

use tract_data::internal::*;

use crate::WeightType;
use crate::frame::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, PackedExoticFact, PackedMatrixStorage,
};

/// Detect AMX-INT8 + AMX-TILE via CPUID leaf 7 sub-leaf 0 (EDX bits 24-25).
/// Stable-Rust friendly: `is_x86_feature_detected!("amx-int8")` is gated on
/// the nightly `x86_amx_intrinsics` feature, so we read CPUID by hand.
fn cpu_has_amx_int8() -> bool {
    if !std::is_x86_feature_detected!("avx512f") {
        return false;
    }
    let r = std::arch::x86_64::__cpuid_count(7, 0);
    // bit 24 = AMX-TILE, bit 25 = AMX-INT8 in EDX.
    const AMX_TILE: u32 = 1 << 24;
    const AMX_INT8: u32 = 1 << 25;
    (r.edx & AMX_TILE) != 0 && (r.edx & AMX_INT8) != 0
}

/// Linux only: ask the kernel for permission to use the AMX tile-data XSAVE
/// state via `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`. Returns
/// true if the kernel grants permission (or if the process already has it).
#[cfg(target_os = "linux")]
unsafe fn request_amx_xcomp_perm() -> bool {
    // x86_64 syscall: rax=158 (arch_prctl), rdi=0x1023 (REQ_XCOMP_PERM),
    // rsi=18 (XFEATURE_XTILEDATA). Returns 0 on success.
    let rc: i64;
    unsafe {
        std::arch::asm!(
            "syscall",
            in("rax") 158i64,
            in("rdi") 0x1023i64,
            in("rsi") 18i64,
            lateout("rax") rc,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    rc == 0
}

/// Returns true iff Intel AMX int8 is available AND the OS has granted this
/// process permission to use the AMX tile-data XSAVE state. Result is
/// memoised — the arch_prctl call has process-wide effect and only needs to
/// run once.
pub fn has_amx_int8() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        if !cpu_has_amx_int8() {
            return false;
        }
        #[cfg(target_os = "linux")]
        {
            unsafe { request_amx_xcomp_perm() }
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    })
}

/// AMX-friendly A packing: per `r`-row panel, M-rows are laid out row-major
/// across `K_padded = ceil(K / 64) * 64` contiguous bytes per row. AMX's
/// `tileloadd` with stride = K_padded reads exactly 8 contiguous M-rows of
/// 64 K-bytes each per call.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PackedAmxA {
    pub r: usize,
    pub align: usize,
}

impl PackedAmxA {
    pub fn new(r: usize) -> Self {
        PackedAmxA { r, align: 64 }
    }
    fn k_padded(&self, k: usize) -> usize {
        k.div_ceil(64) * 64
    }
    fn panel(&self, k: usize) -> usize {
        self.k_padded(k) * self.r
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
        let pl = kp * self.r;
        let panels = mn.div_ceil(self.r);
        let st = t.strides();
        let (ks, ms) = (st[k_axis], st[mn_axis]);
        let mut blob = unsafe { Blob::new_for_size_and_align(panels * pl, self.align) };
        blob.as_bytes_mut().fill(0);
        unsafe {
            let src = t.as_ptr_unchecked::<i8>();
            let dst = blob.as_mut_ptr() as *mut i8;
            for p in 0..panels {
                let pw = self.r.min(mn - p * self.r);
                let panel = dst.add(p * pl);
                let mn0 = (p * self.r) as isize;
                for lm in 0..pw {
                    let drow = panel.add(lm * kp);
                    let srow_base = src.offset((mn0 + lm as isize) * ms);
                    for kk in 0..k {
                        *drow.add(kk) = *srow_base.offset(kk as isize * ks);
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

impl std::fmt::Display for PackedAmxA {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "AmxA[{}]", self.r)
    }
}

impl MMMInputFormat for PackedAmxA {
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
        WeightType::Plain(i8::datum_type())
    }
    fn r(&self) -> usize {
        self.r
    }
    fn k_alignment(&self) -> usize {
        // AMX consumes K=64 bytes per tdpbssd inner iteration; the packer
        // already pads internally, but expose the alignment so upstream
        // schedulers can reason about K-blocking.
        64
    }
    fn merge_with<'o, 'a: 'o, 'b: 'o>(
        &'a self,
        o: &'b dyn MMMInputFormat,
    ) -> Option<&'o dyn MMMInputFormat> {
        o.downcast_ref::<PackedAmxA>().filter(|x| x.r == self.r).map(|_| self as _)
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
