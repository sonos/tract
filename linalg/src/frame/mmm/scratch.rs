use num_traits::Zero;
use std::fmt::Debug;

use super::{FusedKerSpec, FusedSpec, MatMatMulKer, MatrixStore, PanelStore};
use downcast_rs::{impl_downcast, Downcast};
use tract_data::prelude::*;

use std::alloc::Layout;

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

pub struct ScratchSpaceFusedNonLinear<TI: Copy> {
    uspecs: TVec<FusedKerSpec<TI>>,
    buffers: TVec<(bool, Layout, *mut u8)>,
}

impl<TI: Copy> Default for ScratchSpaceFusedNonLinear<TI> {
    fn default() -> ScratchSpaceFusedNonLinear<TI> {
        ScratchSpaceFusedNonLinear { uspecs: tvec![], buffers: tvec![] }
    }
}

impl<TI: Copy + 'static> ScratchSpace for ScratchSpaceFusedNonLinear<TI> {}
unsafe impl<TI: Copy + 'static> Send for ScratchSpaceFusedNonLinear<TI> {}

impl<TI: Copy> Drop for ScratchSpaceFusedNonLinear<TI> {
    fn drop(&mut self) {
        unsafe { self.buffers.drain(..).for_each(|(_, lo, buf)| std::alloc::dealloc(buf, lo)) }
    }
}

impl<TI: Copy> ScratchSpaceFusedNonLinear<TI> {
    pub fn clear(&mut self) {
        self.buffers.iter_mut().for_each(|(used, _, _)| *used = false);
    }

    fn get_raw_buffer(&mut self, bytes: usize) -> *mut u8 {
        if let Some(buf) =
            self.buffers.iter_mut().find(|(used, layout, _)| !used && layout.size() == bytes)
        {
            buf.0 = true;
            buf.2
        } else {
            let layout = Layout::from_size_align(bytes, 4).unwrap();
            let buf = unsafe { std::alloc::alloc(layout) };
            self.buffers.push((true, layout, buf));
            buf
        }
    }

    fn get_temp_slice<'a, T: Datum>(&mut self, len: usize) -> &'a mut [T] {
        let buf = self.get_raw_buffer(std::mem::size_of::<T>() * len) as *mut T;
        unsafe { std::slice::from_raw_parts_mut(buf, len) }
    }

    #[inline]
    pub unsafe fn for_tile<TC, K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        c_store: &MatrixStore,
    ) -> *const FusedKerSpec<TI>
    where
        TC: Datum + Copy,
        TI: Datum + Copy + Debug + Zero,
    {
        if specs.is_empty() {
            std::ptr::null()
        } else {
            self.fused_for_tile::<TC, K>(specs, down, right, c_store)
        }
    }

    unsafe fn fused_for_tile<TC, K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        c_store: &MatrixStore,
    ) -> *const FusedKerSpec<TI>
    where
        TC: Datum + Copy,
        TI: Datum + Copy + Debug + Zero,
    {
        self.uspecs.clear();
        for spec in specs {
            let s = match spec {
                FusedSpec::Min(m) => FusedKerSpec::Min(*m.to_scalar_unchecked()),
                FusedSpec::Max(m) => FusedKerSpec::Max(*m.to_scalar_unchecked()),
                FusedSpec::AddC => FusedKerSpec::AddC,
                FusedSpec::PerRowAdd(v)
                | FusedSpec::PerRowMul(v)
                | FusedSpec::PerColMul(v)
                | FusedSpec::PerColAdd(v) => {
                    let (dir, r) =
                        if matches!(spec, FusedSpec::PerColAdd(_) | FusedSpec::PerColMul(_)) {
                            (right, K::nr())
                        } else {
                            (down, K::mr())
                        };
                    let have = v.len().saturating_sub(dir * r).min(r);
                    let ptr = if have < K::mr() {
                        let buf = self.get_temp_slice(r);
                        buf[have..].iter_mut().for_each(|x| *x = TI::zero());
                        if have > 0 {
                            buf[..have].copy_from_slice(&v.as_slice_unchecked()[dir * r..][..have]);
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(dir * r)
                    };
                    match spec {
                        FusedSpec::PerRowAdd(_) => FusedKerSpec::PerRowAdd(ptr),
                        FusedSpec::PerRowMul(_) => FusedKerSpec::PerRowMul(ptr),
                        FusedSpec::PerColAdd(_) => FusedKerSpec::PerColAdd(ptr),
                        FusedSpec::PerColMul(_) => FusedKerSpec::PerColMul(ptr),
                        _ => unreachable!(),
                    }
                }
                FusedSpec::AddRowColProducts(rows, cols) => {
                    let have = rows.len() - down * K::mr();
                    let row_ptr = if have < K::mr() {
                        let buf = self.get_temp_slice(K::mr());
                        buf[..have]
                            .copy_from_slice(&rows.as_slice_unchecked()[down * K::mr()..][..have]);
                        buf.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let have = cols.len() - right * K::nr();
                    let col_ptr = if have < K::nr() {
                        let buf = self.get_temp_slice(K::nr());
                        buf[..have]
                            .copy_from_slice(&cols.as_slice_unchecked()[right * K::nr()..][..have]);
                        buf.as_ptr()
                    } else {
                        cols.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    FusedKerSpec::AddRowColProducts(row_ptr, col_ptr)
                }
                FusedSpec::ScalarMul(t) => FusedKerSpec::ScalarMul(*t.to_scalar_unchecked()),
                FusedSpec::ScalarAdd(t) => FusedKerSpec::ScalarAdd(*t.to_scalar_unchecked()),
                FusedSpec::QTowardsEven(m, s) => {
                    FusedKerSpec::QTowardsEven(*m.to_scalar_unchecked(), *s)
                }
                FusedSpec::QTowardsPlusInf(m, s) => {
                    FusedKerSpec::QTowardsPlusInf(*m.to_scalar_unchecked(), *s)
                }
                FusedSpec::QAway(m, s) => FusedKerSpec::QAway(*m.to_scalar_unchecked(), *s),
                FusedSpec::AddUnicast(tensor) => {
                    let (rsc, csc, item_count) = match c_store {
                        MatrixStore::Strides { row_item_stride, col_item_stride, item_count, .. } => {
                            (*row_item_stride, *col_item_stride, *item_count)
                        }
                        MatrixStore::VecStride { item_stride, item_count, .. } => (*item_stride, 1, *item_count),
                        _ => panic!(),
                    };
                    let tile_offset = rsc * down as isize * K::mr() as isize + csc * right as isize * K::nr() as isize;
                    let tile_ptr: *const TI = tensor.as_ptr_unchecked::<TI>().offset(tile_offset);

                    let last_tile_value = tile_ptr.offset((K::mr() - 1) as isize * rsc + (K::nr() - 1) as isize * csc);
                    if last_tile_value >= tensor.as_ptr_unchecked::<TI>().offset(item_count as isize) {
                        let tmp_d_tile = self.get_temp_slice::<TI>(K::mr() * K::nr());
                        for r in 0..K::mr() as isize {
                            for c in 0..K::nr() as isize {
                                let inner_offset = c * csc + r * rsc;
                                if inner_offset + tile_offset < item_count as isize {
                                    tmp_d_tile[r as usize + c as usize * K::mr()] = *tile_ptr.offset(inner_offset);
                                }
                            }
                        }
                        FusedKerSpec::AddUnicast(
                            tmp_d_tile.as_ptr(),
                            std::mem::size_of::<TI>(),
                            std::mem::size_of::<TI>() * K::mr(),
                        )
                    } else {
                        FusedKerSpec::AddUnicast(
                            tile_ptr,
                            rsc as usize * std::mem::size_of::<TI>(),
                            csc as usize * std::mem::size_of::<TI>(),
                        )
                    }
                }
            };
            self.uspecs.push(s);
        }
        self.uspecs.push(FusedKerSpec::Done);
        self.uspecs.as_ptr()
    }

    pub unsafe fn tmp_tile_c(&mut self, c: DatumType, mr: usize, nr: usize) -> PanelStore {
        let ptr = self.get_raw_buffer(mr * nr * c.size_of());
        PanelStore::Strides {
            ptr: ptr as _,
            item_size: c.size_of(),
            row_byte_stride: c.size_of() as isize,
            col_byte_stride: (c.size_of() * mr) as isize,
        }
    }
}
