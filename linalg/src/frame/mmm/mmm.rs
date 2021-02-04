use super::fuse::ScratchSpaceFusedNonLinear;
use super::*;
use crate::frame::Packer;
use num_traits::{AsPrimitive, Bounded, Zero};
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg};
use tract_data::anyhow;
use tract_data::internal::*;

pub trait MatMatMul:
    Debug + fmt::Display + dyn_clone::DynClone + Send + Sync + std::any::Any
{
    fn a_pack(&self) -> Packer;
    fn b_pack(&self) -> Packer;

    fn internal_type(&self) -> DatumType;

    unsafe fn a_packed(&self) -> MatrixStoreSpec;

    unsafe fn b_packed(&self) -> MatrixStoreSpec;
    unsafe fn b_from_data_and_offsets(
        &self,
        dt: DatumType,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> MatrixStoreSpec;
    unsafe fn b_vec_from_data_and_stride(&self, dt: DatumType, stride: isize) -> MatrixStoreSpec;
    unsafe fn b_vec_from_data(&self, dt: DatumType) -> MatrixStoreSpec;

    unsafe fn c_view(&self) -> MatrixStoreSpec;
    unsafe fn c_view_with_axis(&self, m_axis: usize, n_axis: usize) -> MatrixStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        row_stride: isize,
        col_stride: isize,
    ) -> MatrixStoreSpec;
    unsafe fn c_vec_from_data_and_stride(&self, stride: isize) -> MatrixStoreSpec;
    unsafe fn c_vec_from_data(&self) -> MatrixStoreSpec;

    unsafe fn run(
        &self,
        a: &MatrixStore,
        b: &MatrixStore,
        c: &mut MatrixStore,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let mut scratch = self.allocate_scratch_space();
        self.run_with_scratch_space(&mut *scratch, a, b, c, non_linear)
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace>;
    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool;
    unsafe fn run_with_scratch_space(
        &self,
        scratch: &mut dyn ScratchSpace,
        a: &MatrixStore,
        b: &MatrixStore,
        c: &mut MatrixStore,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

#[derive(Clone)]
pub struct MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    pub m: usize,
    pub k: usize,
    pub n: usize,

    prefetch: Option<&'static (dyn Fn(*const u8, usize) + Send + Sync)>,
    phantom: PhantomData<(K, TC, TI)>,
}

unsafe impl<K, TC, TI> Send for MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

unsafe impl<K, TC, TI> Sync for MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

impl<K, TC, TI> fmt::Debug for MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MMM ({})", K::name())
    }
}

impl<K, TC, TI> MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    pub fn new(m: usize, k: usize, n: usize) -> MatMatMulImpl<K, TC, TI> {
        MatMatMulImpl { m, k, n, prefetch: crate::ops().prefetch, phantom: PhantomData }
    }

    #[inline]
    fn prefetch(&self, a: &PanelStore, b: &PanelStore) {
        if let Some(prefetch) = self.prefetch {
            if let PanelStore::Packed { ptr } = a {
                prefetch(*ptr as *const u8, 512);
            }
            match b {
                PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                _ => (),
            }
        }
    }
}

impl<K, TC, TI> MatMatMul for MatMatMulImpl<K, TC, TI>
where
    TC: Datum + Copy + Debug + 'static + Bounded + AsPrimitive<TI>,
    TI: Datum + Copy + Add + Mul<Output = TI> + Zero + Debug + 'static + Neg<Output = TI>,
    K: MatMatMulKer<TI> + 'static,
    i32: AsPrimitive<TI>,
    usize: AsPrimitive<TI>,
{
    fn a_pack(&self) -> Packer {
        Packer::new(self.k, K::mr(), K::alignment_bytes_packed_a(), K::end_padding_packed_a())
    }

    fn b_pack(&self) -> Packer {
        Packer::new(self.k, K::nr(), K::alignment_bytes_packed_b(), K::end_padding_packed_b())
    }

    fn internal_type(&self) -> DatumType {
        TI::datum_type()
    }

    unsafe fn a_packed(&self) -> MatrixStoreSpec {
        MatrixStoreSpec::Packed { panel_len: (self.k * K::mr()) }
    }

    unsafe fn b_packed(&self) -> MatrixStoreSpec {
        MatrixStoreSpec::Packed { panel_len: (self.k * K::nr()) }
    }

    unsafe fn b_from_data_and_offsets(
        &self,
        dt: DatumType,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> MatrixStoreSpec {
        debug_assert!(rows_offsets.len() > 0);
        debug_assert!(cols_offsets.len() > 0);
        // repeat the last offset to get to the panel boundary (pad to next multiple of nr)
        let wanted = (cols_offsets.len() + K::nr() - 1) / K::nr() * K::nr();
        let mut col_byte_offsets: Vec<_> =
            cols_offsets.iter().map(|o| o * dt.size_of() as isize).collect();
        while col_byte_offsets.len() < wanted {
            col_byte_offsets.push(*col_byte_offsets.last().unwrap());
        }
        // repeat the last offset four times to simplify kernel loop unrolling
        let mut row_byte_offsets: Vec<_> = Vec::with_capacity(rows_offsets.len() + 4);
        row_byte_offsets.set_len(rows_offsets.len() + 4);
        for i in 0..rows_offsets.len() {
            *row_byte_offsets.get_unchecked_mut(i) =
                *rows_offsets.get_unchecked(i) * dt.size_of() as isize;
        }
        let pad = *row_byte_offsets.get_unchecked(rows_offsets.len() - 1);
        for i in 0..4 {
            *row_byte_offsets.get_unchecked_mut(rows_offsets.len() + i) = pad;
        }
        MatrixStoreSpec::OffsetsAndPtrs { col_byte_offsets, row_byte_offsets, nr: K::nr() }
    }

    unsafe fn b_vec_from_data_and_stride(&self, dt: DatumType, stride: isize) -> MatrixStoreSpec {
        MatrixStoreSpec::VecStride {
            byte_stride: stride * dt.size_of() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn b_vec_from_data(&self, dt: DatumType) -> MatrixStoreSpec {
        self.b_vec_from_data_and_stride(dt, 1)
    }

    unsafe fn c_view(&self) -> MatrixStoreSpec {
        MatrixStoreSpec::View { axes: None }
    }

    unsafe fn c_view_with_axis(&self, m_axis: usize, n_axis: usize) -> MatrixStoreSpec {
        MatrixStoreSpec::View { axes: Some((m_axis, n_axis)) }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        row_stride: isize,
        col_stride: isize,
    ) -> MatrixStoreSpec {
        MatrixStoreSpec::Strides {
            row_byte_stride: row_stride * std::mem::size_of::<TC>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<TC>() as isize,
        }
    }

    unsafe fn c_vec_from_data_and_stride(&self, stride: isize) -> MatrixStoreSpec {
        MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TC>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data(&self) -> MatrixStoreSpec {
        self.c_vec_from_data_and_stride(1)
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::new(ScratchSpaceFusedNonLinear::<TI>::default())
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceFusedNonLinear<TI>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        scratch: &mut dyn ScratchSpace,
        a: &MatrixStore,
        b: &MatrixStore,
        c: &mut MatrixStore,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        use anyhow::Context;
        let mr = K::mr();
        let nr = K::nr();
        anyhow::ensure!(c.tensor.datum_type() == TC::datum_type());
        let m = self.m;
        let n = self.n;
        let scratch = scratch
            .downcast_mut::<ScratchSpaceFusedNonLinear<TI>>()
            .context("Wrong scratch space type")?;

        let ref linear = LinearSpec::k(self.k);
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                let ref b = b.panel_b(nr, ib, nr);
                self.prefetch(a, b);
                scratch.clear();
                let ref direct_c = c.tile_c(ia, ib, mr, nr);
                let non_linear = scratch.for_tile::<TC, K>(&non_linear, ia, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: direct_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
            if let MatrixStoreSpec::VecStride { .. } = c.spec {
                let ref b = b.panel_b(nr, n / nr, n % nr);
                self.prefetch(a, b);
                scratch.clear();
                let non_linear = scratch.for_tile::<TC, K>(&non_linear, ia, n / nr);
                let ref direct_c = c.tile_c(ia, 0, mr, nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: direct_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            } else if n % nr != 0 {
                let ref b = b.panel_b(nr, n / nr, n % nr);
                self.prefetch(a, b);
                scratch.clear();
                let tmpc = scratch.tmp_tile_c(TC::datum_type(), mr, nr);
                let non_linear = scratch.for_tile::<TC, K>(&non_linear, ia, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: &tmpc,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile::<TC>(ia, n / nr, mr, n % nr, &tmpc, mr, nr);
            }
        }
        if m % mr != 0 {
            let ref panel_a = a.panel_a(m / mr);
            for ib in 0..n / nr {
                let ref b = b.panel_b(nr, ib, nr);
                self.prefetch(panel_a, b);
                scratch.clear();
                let tmpc = scratch.tmp_tile_c(TC::datum_type(), mr, nr);
                let non_linear = scratch.for_tile::<TC, K>(&non_linear, m / mr, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: &tmpc,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile::<TC>(m / mr, ib, m % mr, nr, &tmpc, mr, nr);
            }
            if n % nr != 0 {
                let ref b = b.panel_b(nr, n / nr, n % nr);
                self.prefetch(panel_a, b);
                scratch.clear();
                // FIXME: can we write straight to C if n == 1 ?
                let tmpc = scratch.tmp_tile_c(TC::datum_type(), mr, nr);
                let non_linear = scratch.for_tile::<TC, K>(&non_linear, m / mr, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: &tmpc,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile::<TC>(m / mr, n / nr, m % mr, n % nr, &tmpc, mr, nr);
            }
        }
        Ok(())
    }
}

impl<K, TC, TI> fmt::Display for MatMatMulImpl<K, TC, TI>
where
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "(m:{}, k:{}, n:{}) ({} {}x{})",
            self.m,
            self.k,
            self.n,
            K::name(),
            K::mr(),
            K::nr()
        )
    }
}
