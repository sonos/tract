use crate::frame::pack::PackedFormat;

use super::*;
use std::borrow::Cow;
use std::fmt::Debug;

use crate::LADatum;

pub trait MatMatMulKer: Clone + Debug + Send + Sync + 'static {
    type Acc: LADatum;
    fn name(&self) -> &str;
    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn quality(&self) -> ImplementationQuality;

    /// The preference its author spelled out for this kernel, before the instruction-set
    /// default is added in. Zero for a kernel that claims nothing.
    fn declared_boost(&self) -> isize;

    /// [`Self::declared_boost`] plus the default owed to the instruction set the kernel was
    /// written for, [`crate::isa::TIER_BOOST`] per tier. Ranking reads this one.
    fn dynamic_boost(&self) -> isize {
        self.declared_boost() + self.isa().tier() as isize * crate::isa::TIER_BOOST
    }

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)];
    fn stores(&self) -> Cow<'_, [DatumType]>;

    #[allow(unused_variables)]
    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn is_supported_here(&self) -> bool {
        true
    }

    /// What the instruction set must offer for this kernel to run here.
    fn isa(&self) -> crate::isa::IsaReq {
        crate::isa::IsaReq::ANY
    }

    /// Whether the border-tile store scratch should be laid out row-major
    /// (n contiguous) instead of the default column-major (mr contiguous).
    /// Set by kernels whose store has an aligned row-major bulk path.
    fn stores_row_major_tile(&self) -> bool {
        false
    }
}

type Kernel<Acc> = unsafe fn(&[FusedKerSpec<Acc>]) -> isize;

#[derive(Clone)]
pub struct DynKernel<const MR: usize, const NR: usize, Acc: LADatum> {
    pub name: String,
    pub kernel: Kernel<Acc>,
    pub quality: ImplementationQuality,
    pub packings: Vec<(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)>,
    pub stores: Vec<DatumType>,
    /// False when this build did not assemble the kernel's asm, its arch not being the one the
    /// kernel was written for. The kernel struct still exists, so it stays introspectable, but
    /// it is never supported here and calling it bails.
    pub bound: bool,
    /// What the instruction set must offer for this kernel to run here at all.
    pub isa: crate::isa::IsaReq,
    pub boost: fn() -> isize,
    pub can_fuse: fn(&FusedSpec) -> bool,
    pub row_major_store: bool,
}

impl<const MR: usize, const NR: usize, Acc: LADatum> DynKernel<MR, NR, Acc> {
    pub fn new(
        name: &str,
        kernel: Kernel<Acc>,
        packing_a: PackedFormat,
        packing_b: PackedFormat,
        quality: ImplementationQuality,
    ) -> Self {
        let kernel = DynKernel {
            name: name.to_string(),
            kernel,
            quality,
            packings: vec![],
            stores: vec![Acc::datum_type()],
            bound: true,
            isa: crate::isa::IsaReq::ANY,
            boost: || 0,
            can_fuse: |_| true,
            row_major_store: false,
        };
        kernel.with_packing(packing_a, packing_b)
    }

    /// Sets what the instruction set must offer for this kernel to run here — the `isa(..)` of
    /// the kernel macros. Runnability only, and it is a set of declared tokens, nothing runtime:
    /// a preference spelled here would also skip the kernel's tests. Use [`Self::with_boost`].
    pub fn with_isa(mut self, isa: crate::isa::IsaReq) -> Self {
        self.isa = isa;
        self
    }

    /// Sets the tie-break behind [`MatMatMulKer::dynamic_boost`] — the `boost(..)` of the kernel
    /// macros, and the one place a runtime preference belongs.
    pub fn with_boost(mut self, f: fn() -> isize) -> Self {
        self.boost = f;
        self
    }

    pub fn with_packing(mut self, a: impl MMMInputFormat, b: impl MMMInputFormat) -> Self {
        self.packings.push((Box::new(a), Box::new(b)));
        self
    }

    pub fn with_packing_a(self, a: impl MMMInputFormat) -> Self {
        let b = self.regular_pack_b();
        self.with_packing(a, b)
    }

    pub fn regular_pack_a(&self) -> PackedFormat {
        *self.packings[0].0.clone().downcast::<PackedFormat>().unwrap()
    }

    pub fn regular_pack_b(&self) -> PackedFormat {
        *self.packings[0].1.clone().downcast::<PackedFormat>().unwrap()
    }

    pub fn with_can_fuse(self, can_fuse: fn(&FusedSpec) -> bool) -> Self {
        Self { can_fuse, ..self }
    }

    pub fn with_store<D: LADatum>(mut self) -> Self {
        self.stores.push(D::datum_type());
        self
    }

    pub fn mmm(&self) -> Box<dyn MatMatMul> {
        Box::new(self.clone())
    }
}

impl<const MR: usize, const NR: usize, Acc: LADatum> Debug for DynKernel<MR, NR, Acc> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<const MR: usize, const NR: usize, Acc: LADatum> MatMatMulKer for DynKernel<MR, NR, Acc> {
    type Acc = Acc;
    fn name(&self) -> &str {
        &self.name
    }

    fn mr(&self) -> usize {
        MR
    }

    fn nr(&self) -> usize {
        NR
    }

    fn quality(&self) -> ImplementationQuality {
        self.quality
    }

    fn is_supported_here(&self) -> bool {
        self.bound && self.isa.satisfied_by(crate::isa::native())
    }

    fn isa(&self) -> crate::isa::IsaReq {
        self.isa
    }

    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        (self.can_fuse)(spec)
    }

    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize {
        unsafe { (self.kernel)(op) }
    }

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)] {
        &self.packings
    }

    fn stores(&self) -> Cow<'_, [DatumType]> {
        Cow::Borrowed(&self.stores)
    }

    fn declared_boost(&self) -> isize {
        (self.boost)()
    }

    fn stores_row_major_tile(&self) -> bool {
        self.row_major_store
    }
}
