use pack::PackedFormat;

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

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)];
    fn stores(&self) -> Cow<[DatumType]>;

    #[allow(unused_variables)]
    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn is_supported_here(&self) -> bool {
        true
    }
}

type Kernel<Acc> = unsafe fn(&[FusedKerSpec<Acc>]) -> isize;

#[derive(Clone)]
pub struct DynKernel<const MR: usize, const NR: usize, Acc: LADatum> {
    pub name: String,
    pub kernel: Kernel<Acc>,
    pub default_packing_alignments: (usize, usize),
    pub packings: Vec<(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)>,
    pub stores: Vec<DatumType>,
    pub supported_predicate: fn() -> bool,
    pub can_fuse: fn(&FusedSpec) -> bool,
}

impl<const MR: usize, const NR: usize, Acc: LADatum> DynKernel<MR, NR, Acc> {
    pub fn new(
        name: &str,
        kernel: Kernel<Acc>,
        default_packing_alignments: (usize, usize),
    ) -> Self {
        let kernel = DynKernel {
            name: name.to_string(),
            kernel,
            packings: vec![],
            stores: vec![Acc::datum_type()],
            supported_predicate: || true,
            default_packing_alignments,
            can_fuse: |_| true,
        };
        let a = kernel.regular_pack_a();
        let b = kernel.regular_pack_b();
        kernel.with_packing(a, b)
    }

    pub fn with_platform_condition(mut self, f: fn() -> bool) -> Self {
        self.supported_predicate = f;
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
        PackedFormat::new(Acc::datum_type(), MR, self.default_packing_alignments.0)
    }

    pub fn regular_pack_b(&self) -> PackedFormat {
        PackedFormat::new(Acc::datum_type(), NR, self.default_packing_alignments.1)
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

    fn is_supported_here(&self) -> bool {
        (self.supported_predicate)()
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

    fn stores(&self) -> Cow<[DatumType]> {
        Cow::Borrowed(&self.stores)
    }
}
