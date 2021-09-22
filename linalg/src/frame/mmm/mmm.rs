use super::ScratchSpaceFusedNonLinear;
use super::*;
use crate::frame::Packer;
use anyhow::Context;
use num_traits::{AsPrimitive, Zero};
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg};
use tract_data::anyhow;
use tract_data::internal::*;

pub trait MatMatMul:
    Debug + fmt::Display + dyn_clone::DynClone + Send + Sync + std::any::Any
{
    fn kernel_name(&self) -> &'static str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn a_pack(&self, k: usize) -> Packer;
    fn b_pack(&self, k: usize) -> Packer;

    fn internal_type(&self) -> DatumType;

    unsafe fn a_packed(&self, item_size: usize, k: usize) -> PackedStoreSpec;

    unsafe fn b_packed(&self, item_size: usize, k: usize) -> InputStoreSpec;
    unsafe fn b_from_data_and_offsets(
        &self,
        item_size: usize,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> InputStoreSpec;

    unsafe fn c_view(&self) -> OutputStoreSpec;
    unsafe fn c_view_with_axis(&self, m_axis: usize, n_axis: usize) -> OutputStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec;

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

    unsafe fn run_with_scratch_space_vec(
        &self,
        m: usize,
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
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    prefetch: Option<&'static (dyn Fn(*const u8, usize) + Send + Sync)>,
    phantom: PhantomData<(K, TI)>,
}

unsafe impl<K, TI> Send for MatMatMulImpl<K, TI>
where
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

unsafe impl<K, TI> Sync for MatMatMulImpl<K, TI>
where
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

impl<K, TI> fmt::Debug for MatMatMulImpl<K, TI>
where
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MMM ({} {}x{})", K::name(), K::mr(), K::nr())
    }
}

impl<K, TI> MatMatMulImpl<K, TI>
where
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    pub fn new() -> MatMatMulImpl<K, TI> {
        MatMatMulImpl { prefetch: crate::ops().prefetch, phantom: PhantomData }
    }
}

impl<K, TI> MatMatMul for MatMatMulImpl<K, TI>
where
    TI: Datum + Copy + Add + Mul<Output = TI> + Zero + Debug + 'static + Neg<Output = TI>,
    K: MatMatMulKer<TI> + 'static,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
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

    fn a_pack(&self, k: usize) -> Packer {
        Packer::new(k, K::mr(), K::alignment_bytes_packed_a(), K::end_padding_packed_a())
    }

    fn b_pack(&self, k: usize) -> Packer {
        Packer::new(k, K::nr(), K::alignment_bytes_packed_b(), K::end_padding_packed_b())
    }

    fn internal_type(&self) -> DatumType {
        TI::datum_type()
    }

    unsafe fn a_packed(&self, item_size: usize, k: usize) -> PackedStoreSpec {
        PackedStoreSpec { panel_bytes: (k * K::mr() * item_size) }
    }

    unsafe fn b_packed(&self, item_size: usize, k: usize) -> InputStoreSpec {
        let panel_bytes = k * K::nr() * item_size;
        InputStoreSpec::Packed(PackedStoreSpec { panel_bytes })
    }

    unsafe fn b_from_data_and_offsets(
        &self,
        item_size: usize,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> InputStoreSpec {
        debug_assert!(rows_offsets.len() > 0);
        debug_assert!(cols_offsets.len() > 0);
        // repeat the last offset to get to the panel boundary (pad to next multiple of nr)
        let wanted = (cols_offsets.len() + K::nr() - 1) / K::nr() * K::nr();
        let mut col_byte_offsets: Vec<_> =
            cols_offsets.iter().map(|o| o * item_size as isize).collect();
        while col_byte_offsets.len() < wanted {
            col_byte_offsets.push(*col_byte_offsets.last().unwrap());
        }
        // repeat the last offset four times to simplify kernel loop unrolling
        let mut row_byte_offsets: Vec<_> = Vec::with_capacity(rows_offsets.len() + 4);
        row_byte_offsets.set_len(rows_offsets.len() + 4);
        for i in 0..rows_offsets.len() {
            *row_byte_offsets.get_unchecked_mut(i) =
                *rows_offsets.get_unchecked(i) * item_size as isize;
        }
        let pad = *row_byte_offsets.get_unchecked(rows_offsets.len() - 1);
        for i in 0..4 {
            *row_byte_offsets.get_unchecked_mut(rows_offsets.len() + i) = pad;
        }
        InputStoreSpec::OffsetsAndPtrs { col_byte_offsets, row_byte_offsets, nr: K::nr() }
    }

    unsafe fn c_view(&self) -> OutputStoreSpec {
        OutputStoreSpec::View { axes: None, mr: K::mr(), nr: K::nr() }
    }

    unsafe fn c_view_with_axis(&self, m_axis: usize, n_axis: usize) -> OutputStoreSpec {
        OutputStoreSpec::View { axes: Some((m_axis, n_axis)), mr: K::mr(), nr: K::nr() }
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
            row_item_stride: row_stride,
            col_item_stride: col_stride,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::new(ScratchSpaceFusedNonLinear::<TI>::default())
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceFusedNonLinear<TI>>().is_some()
    }

    unsafe fn run_with_scratch_space_vec(
        &self,
        m: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let mr = K::mr();
        let scratch = scratch
            .downcast_mut::<ScratchSpaceFusedNonLinear<TI>>()
            .context("Wrong scratch space type")?;
        scratch.prepare::<K>(non_linear);
        for ia in 0..m / mr {
            scratch.for_valid_tile::<K>(&non_linear, ia, 0);
            let err = K::kernel(&scratch.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {}", err);
        }
        if m % mr != 0 {
            scratch.for_border_tile::<K>(&non_linear, m / mr, 0);
            let err = K::kernel(&scratch.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {}", err);
            scratch.postprocess_tile::<K>(&non_linear, m / mr, 0, m % mr, 1);
        }
        Ok(())
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let mr = K::mr();
        let nr = K::nr();
        if n == 1 && K::nr() == 1 {
            return self.run_with_scratch_space_vec(m, scratch, &non_linear);
        }
        let scratch = scratch
            .downcast_mut::<ScratchSpaceFusedNonLinear<TI>>()
            .context("Wrong scratch space type")?;
        scratch.prepare::<K>(non_linear);
        for ia in 0..m / mr {
            for ib in 0..n / nr {
                scratch.for_valid_tile::<K>(&non_linear, ia, ib);
                let err = K::kernel(&scratch.uspecs());
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
            if n % nr != 0 {
                scratch.for_border_tile::<K>(&non_linear, ia, n / nr);
                let err = K::kernel(&scratch.uspecs());
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                scratch.postprocess_tile::<K>(&non_linear, ia, n / nr, mr, n % nr);
            }
        }
        if m % mr != 0 {
            for ib in 0..n / nr {
                scratch.for_border_tile::<K>(&non_linear, m / mr, ib);
                let err = K::kernel(&scratch.uspecs());
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                scratch.postprocess_tile::<K>(&non_linear, m / mr, ib, m % mr, nr);
            }
            if n % nr != 0 {
                scratch.for_border_tile::<K>(&non_linear, m / mr, n / nr);
                let err = K::kernel(&scratch.uspecs());
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                scratch.postprocess_tile::<K>(&non_linear, m / mr, n / nr, m % mr, n % nr);
            }
        }
        Ok(())
    }
}

impl<K, TI> fmt::Display for MatMatMulImpl<K, TI>
where
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "({} {}x{})", K::name(), K::mr(), K::nr())
    }
}
