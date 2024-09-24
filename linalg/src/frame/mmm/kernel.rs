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

#[derive(Clone)]
struct DynKernel<const MR: usize, const NR: usize, Acc: LADatum> {
    name: String,
    kernel: unsafe fn(&[FusedKerSpec<Acc>]) -> isize,
    packings: Vec<(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)>,
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
        unsafe { (self.kernel)(op) }
    }

    fn packings(&self) -> Cow<[(&dyn MMMInputFormat, &dyn MMMInputFormat)]> {
        Cow::Owned(self.packings.iter().map(|p| (&*p.0, &*p.1)).collect_vec())
    }
}
