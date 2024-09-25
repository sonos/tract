use pack::PackedFormat;
use tract_itertools::Itertools;

use super::*;
use std::borrow::Cow;
use std::fmt::Debug;

use crate::LADatum;

pub trait MatMatMulKer: Clone + Debug + Send + Sync + 'static {
    type Acc: LADatum;
    fn name(&self) -> Cow<str>;
    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn packings(&self) -> Cow<[(&dyn MMMInputFormat, &dyn MMMInputFormat)]>;

    #[allow(unused_variables)]
    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        true
    }
}

type F<Acc> = unsafe extern "C" fn(*const FusedKerSpec<Acc>) -> isize;

#[derive(Clone)]
pub struct DynKernel<const MR: usize, const NR: usize, Acc: LADatum> {
    name: String,
    kernel: F<Acc>,
    packings: Vec<(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)>,
}

impl<const MR: usize, const NR: usize, Acc: LADatum> DynKernel<MR, NR, Acc> {
    pub fn new(name: &str, kernel: F<Acc>) -> Self {
        let kernel = DynKernel { name: name.to_string(), kernel, packings: vec![] };
        let a = kernel.regular_pack_a();
        let b = kernel.regular_pack_b();
        kernel.with_packing(a, b)
    }

    pub fn with_packing(mut self, a: Box<dyn MMMInputFormat>, b: Box<dyn MMMInputFormat>) -> Self {
        self.packings.push((a, b));
        self
    }

    pub fn regular_pack_a(&self) -> Box<dyn MMMInputFormat> {
        Box::new(PackedFormat::new(Acc::datum_type(), MR, vector_size()))
    }

    pub fn regular_pack_b(&self) -> Box<dyn MMMInputFormat> {
        Box::new(PackedFormat::new(Acc::datum_type(), NR, vector_size()))
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
    fn name(&self) -> Cow<str> {
        Cow::Borrowed(&self.name)
    }

    fn mr(&self) -> usize {
        MR
    }

    fn nr(&self) -> usize {
        NR
    }

    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize {
        unsafe { (self.kernel)(op.as_ptr()) }
    }

    fn packings(&self) -> Cow<[(&dyn MMMInputFormat, &dyn MMMInputFormat)]> {
        Cow::Owned(self.packings.iter().map(|p| (&*p.0, &*p.1)).collect_vec())
    }
}
