use super::ScratchSpaceImpl;
use super::*;
use crate::frame::Packer;
use crate::multithread::Executor;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fmt;
use std::marker::PhantomData;
use tract_data::internal::*;

pub trait MatMatMul:
    fmt::Debug + fmt::Display + dyn_clone::DynClone + Send + Sync + std::any::Any
{
    fn kernel_name(&self) -> &'static str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn a_pack(&self) -> Packer;
    fn b_pack(&self) -> Packer;

    fn internal_type(&self) -> DatumType;

    unsafe fn c_view(&self, m_axis: usize, n_axis: usize) -> OutputStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec;

    fn can_fuse(&self, spec: &FusedSpec) -> bool;

    unsafe fn run(&self, m: usize, n: usize, non_linear: &[FusedSpec]) -> TractResult<()> {
        let mut scratch = self.allocate_scratch_space();
        self.run_with_scratch_space(m, n, &mut *scratch, non_linear)
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace>;
    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool;
    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

impl PartialEq for Box<dyn MatMatMul> {
    fn eq(&self, other: &Box<dyn MatMatMul>) -> bool {
        self.as_ref().type_id() == other.as_ref().type_id()
    }
}

impl std::hash::Hash for Box<dyn MatMatMul> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().type_id().hash(state)
    }
}

#[derive(Clone)]
pub struct MatMatMulImpl<K: MatMatMulKer> {
    phantom: PhantomData<K>,
}

unsafe impl<K: MatMatMulKer> Send for MatMatMulImpl<K> {}
unsafe impl<K: MatMatMulKer> Sync for MatMatMulImpl<K> {}

impl<K: MatMatMulKer> Default for MatMatMulImpl<K> {
    fn default() -> Self {
        MatMatMulImpl { phantom: PhantomData }
    }
}

impl<K: MatMatMulKer> fmt::Display for MatMatMulImpl<K> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self)
    }
}

impl<K: MatMatMulKer> fmt::Debug for MatMatMulImpl<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MMM ({} {}x{})", K::name(), K::mr(), K::nr())
    }
}

impl<K: MatMatMulKer> MatMatMul for MatMatMulImpl<K> {
    fn kernel_name(&self) -> &'static str {
        K::name()
    }

    fn mr(&self) -> usize {
        K::mr()
    }

    fn nr(&self) -> usize {
        K::nr()
    }

    fn a_pack(&self) -> Packer {
        Packer::new(K::mr(), K::alignment_bytes_packed_a(), K::end_padding_packed_a())
    }

    fn b_pack(&self) -> Packer {
        Packer::new(K::nr(), K::alignment_bytes_packed_b(), K::end_padding_packed_b())
    }

    fn internal_type(&self) -> DatumType {
        K::Acc::datum_type()
    }

    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        K::can_fuse(spec)
    }

    unsafe fn c_view(&self, m_axis: usize, n_axis: usize) -> OutputStoreSpec {
        OutputStoreSpec::View { m_axis, n_axis, mr: K::mr(), nr: K::nr() }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec {
        OutputStoreSpec::Strides {
            row_byte_stride: row_stride * item_size as isize,
            col_byte_stride: col_stride * item_size as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::<ScratchSpaceImpl<K::Acc>>::default()
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceImpl<K::Acc>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()> {
        let scratch = scratch
            .downcast_mut::<ScratchSpaceImpl<K::Acc>>()
            .context("Wrong scratch space type")?;
        scratch.prepare::<K>(m, n, non_linear)?;
        if n == 1 && K::nr() == 1 {
            run_with_scratch_space_vec::<K>(m, scratch, non_linear)
        } else if non_linear.iter().any(|f| f.prefer_col_outer()) {
            run_with_scratch_space_col_outer::<K>(m, n, scratch, non_linear)
        } else {
            run_with_scratch_space_row_outer::<K>(m, n, scratch, non_linear)
        }
    }
}

unsafe fn run_with_scratch_space_vec<K: MatMatMulKer>(
    m: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    match crate::multithread::current_tract_executor() {
        Executor::SingleThread => {
            for ia in 0..m.divceil(K::mr()) {
                scratch.run::<K>(non_linear, ia, 0)?;
            }
            Ok(())
        }
        Executor::MultiThread(pool) => pool.install(|| {
            (0..m.div_ceil(K::mr()))
                .into_par_iter()
                .try_for_each(|ia| scratch.run::<K>(non_linear, ia, 0))
        }),
    }
}

unsafe fn run_with_scratch_space_col_outer<K: MatMatMulKer>(
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    match crate::multithread::current_tract_executor() {
        Executor::SingleThread => {
            for ib in 0..n.divceil(K::nr()) {
                for ia in 0..m.divceil(K::mr()) {
                    scratch.run::<K>(non_linear, ia, ib)?;
                }
            }
            Ok(())
        }
        Executor::MultiThread(pool) => pool.install(|| {
            (0..n.div_ceil(K::nr())).into_par_iter().try_for_each(|ib| {
                for ia in 0..m.divceil(K::mr()) {
                    scratch.run::<K>(non_linear, ia, ib)?;
                }
                Ok(())
            })
        }),
    }
}

unsafe fn run_with_scratch_space_row_outer<K: MatMatMulKer>(
    m: usize,
    n: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    match crate::multithread::current_tract_executor() {
        Executor::SingleThread => {
            for ia in 0..m.divceil(K::mr()) {
                for ib in 0..n.divceil(K::nr()) {
                    scratch.run::<K>(non_linear, ia, ib)?;
                }
            }
            Ok(())
        }
        Executor::MultiThread(pool) => pool.install(|| {
            pool.install(|| {
                (0..m.div_ceil(K::mr())).into_par_iter().try_for_each(|ia| {
                    for ib in 0..n.divceil(K::nr()) {
                        scratch.run::<K>(non_linear, ia, ib)?;
                    }
                    Ok(())
                })
            })
        }),
    }
}
