use super::ScratchSpaceImpl;
use super::*;
use crate::frame::Packer;
use crate::LADatum;
use anyhow::Context;
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::anyhow;
use tract_data::internal::*;

pub trait MatMatMul:
    Debug + fmt::Display + dyn_clone::DynClone + Send + Sync + std::any::Any
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

    unsafe fn run(&self, m: usize, n: usize, non_linear: &[FusedSpec]) -> anyhow::Result<()> {
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
    ) -> anyhow::Result<()>;
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
pub struct MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
    phantom: PhantomData<(K, TI)>,
}

unsafe impl<K, TI> Send for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
}

unsafe impl<K, TI> Sync for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
}

impl<K, TI> Default for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
    fn default() -> Self {
        MatMatMulImpl { phantom: PhantomData }
    }
}

impl<K, TI> fmt::Debug for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MMM ({} {}x{})", K::name(), K::mr(), K::nr())
    }
}

impl<K, TI> MatMatMul for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI> + 'static,
{
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
        TI::datum_type()
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
        Box::<ScratchSpaceImpl<TI>>::default()
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceImpl<TI>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let scratch =
            scratch.downcast_mut::<ScratchSpaceImpl<TI>>().context("Wrong scratch space type")?;
        scratch.prepare::<K>(m, n, non_linear)?;
        if n == 1 && K::nr() == 1 {
            self.run_with_scratch_space_vec(m, scratch, non_linear);
        } else if non_linear.iter().any(|f| f.prefer_col_outer()) {
            self.run_with_scratch_space_col_outer(m, n, scratch, non_linear);
        } else {
            self.run_with_scratch_space_row_outer(m, n, scratch, non_linear);
        }
        Ok(())
    }
}

impl<TI: LADatum, K: MatMatMulKer<TI>> MatMatMulImpl<K, TI> {
    unsafe fn run_with_scratch_space_vec(
        &self,
        m: usize,
        scratch: &mut ScratchSpaceImpl<TI>,
        non_linear: &[FusedSpec],
    ) {
        for ia in 0..m.divceil(K::mr()) {
            scratch.run::<K>(non_linear, ia, 0);
        }
    }

    unsafe fn run_with_scratch_space_col_outer(
        &self,
        m: usize,
        n: usize,
        scratch: &mut ScratchSpaceImpl<TI>,
        non_linear: &[FusedSpec],
    ) {
        for ib in 0..n.divceil(K::nr()) {
            for ia in 0..m.divceil(K::mr()) {
                scratch.run::<K>(non_linear, ia, ib);
            }
        }
    }

    unsafe fn run_with_scratch_space_row_outer(
        &self,
        m: usize,
        n: usize,
        scratch: &mut ScratchSpaceImpl<TI>,
        non_linear: &[FusedSpec],
    ) {
        for ia in 0..m.divceil(K::mr()) {
            for ib in 0..n.divceil(K::nr()) {
                scratch.run::<K>(non_linear, ia, ib);
            }
        }
    }
}

impl<K, TI> fmt::Display for MatMatMulImpl<K, TI>
where
    TI: LADatum,
    K: MatMatMulKer<TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "({} {}x{})", K::name(), K::mr(), K::nr())
    }
}
