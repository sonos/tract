use num_traits::Zero;
use std::fmt::Debug;

use super::{FusedKerSpec, FusedSpec, InputStoreKer, MatMatMulKer, OutputStoreKer};
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

impl<TI: Copy + Datum + Zero> ScratchSpaceFusedNonLinear<TI> {
    #[inline]
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

    fn get_temp<'a, T>(&mut self) -> &'a mut T {
        let buf = self.get_raw_buffer(std::mem::size_of::<T>()) as *mut T;
        unsafe { buf.as_mut().unwrap() }
    }

    #[inline]
    pub unsafe fn prepare<K: MatMatMulKer<TI>>(&mut self, specs: &[FusedSpec]) {
        self.uspecs = tvec!(FusedKerSpec::Done; specs.len() + 1)
    }

    #[inline]
    pub unsafe fn for_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        valid: bool,
    ) {
        self.clear();
        for ix in 0..specs.len() {
            let spec = &specs[ix];
            self.uspecs[ix] = match spec {
                FusedSpec::Min(m) => FusedKerSpec::Min(*m.to_scalar_unchecked()),
                FusedSpec::Max(m) => FusedKerSpec::Max(*m.to_scalar_unchecked()),
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
                FusedSpec::AddUnicast(store) => {
                    if valid {
                        FusedKerSpec::AddUnicast(store.tile_c(down, right))
                    } else {
                        let row_byte_stride = store.row_byte_stride;
                        let col_byte_stride = store.col_byte_stride;
                        let tile_offset = row_byte_stride * down as isize * K::mr() as isize
                            + col_byte_stride * right as isize * K::nr() as isize;
                        let tile_ptr = store.ptr.offset(tile_offset);
                        let tmp_d_tile = self.get_temp_slice::<TI>(K::mr() * K::nr());
                        if store.item_size == 4 {
                            // assumes TI is f32 or i32
                            for r in 0..K::mr() as isize {
                                for c in 0..K::nr() as isize {
                                    let inner_offset = c * col_byte_stride + r * row_byte_stride;
                                    if inner_offset + tile_offset < 4 * store.item_count as isize {
                                        tmp_d_tile[r as usize + c as usize * K::mr()] =
                                            *(tile_ptr.offset(inner_offset) as *const TI);
                                    }
                                }
                            }
                        } else {
                            unreachable!();
                        }
                        FusedKerSpec::AddUnicast(OutputStoreKer {
                            ptr: tmp_d_tile.as_ptr() as _,
                            row_byte_stride: std::mem::size_of::<TI>() as isize,
                            col_byte_stride: (std::mem::size_of::<TI>() * K::mr()) as isize,
                            item_size: std::mem::size_of::<TI>(),
                        })
                    }
                }
                FusedSpec::QScale(s, rp, m) => FusedKerSpec::QScale(*s, *rp, *m),
                FusedSpec::Store(c_store) => {
                    if valid {
                        FusedKerSpec::Store(c_store.tile_c(down, right))
                    } else {
                        let tmpc = self.tmp_tile_c(c_store.item_size(), K::mr(), K::nr());
                        FusedKerSpec::Store(tmpc)
                    }
                }
                FusedSpec::AddMatMul { k, a, b } => {
                    let pa = a.panel(down);
                    K::prefetch(pa.ptr as _, 512);
                    let pb: &mut InputStoreKer = self.get_temp();
                    let mut tb = b.panel_b(right);
                    std::mem::swap(pb, &mut tb);
                    FusedKerSpec::AddMatMul { k: *k, pa: a.panel(down).ptr, pb, cpu_variant: 0 }
                }
            };
        }
    }

    #[inline]
    pub fn uspecs(&self) -> &[FusedKerSpec<TI>] {
        &self.uspecs
    }

    #[inline]
    pub unsafe fn tmp_tile_c(&mut self, item_size: usize, mr: usize, nr: usize) -> OutputStoreKer {
        let ptr = self.get_raw_buffer(mr * nr * item_size);
        OutputStoreKer {
            ptr: ptr as _,
            item_size,
            row_byte_stride: item_size as isize,
            col_byte_stride: (item_size * mr) as isize,
        }
    }

    pub unsafe fn postprocess_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        m_remnant: usize,
        n_remnant: usize,
    ) where
        TI: Datum + Copy + Debug + Zero,
    {
        for (i, spec) in specs.iter().enumerate() {
            let ker_spec = self.uspecs.get_unchecked(i);
            if let (FusedSpec::Store(c_store), FusedKerSpec::Store(tmp)) = (spec, ker_spec) {
                c_store.set_from_tile(down, right, m_remnant, n_remnant, &tmp)
            }
        }
    }
}
