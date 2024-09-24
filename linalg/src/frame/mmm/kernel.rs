use std::borrow::Cow;
use std::fmt::Debug;
use super::*;

use crate::LADatum;

pub trait MatMatMulKer: Copy + Clone + Debug + Send + Sync + 'static {
    type Acc: LADatum;
    fn name(&self) -> Cow<'static, str>;
    fn kernel(&self, op: &[FusedKerSpec<Self::Acc>]) -> isize;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn packings(&self) -> Cow<[(&dyn MMMInputFormat, &dyn MMMInputFormat)]>;

    #[allow(unused_variables)]
    fn prefetch(&self, ptr: *const u8, len: usize) {}

    #[allow(unused_variables)]
    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        true
    }
}
