use num_traits::Zero;
use std::fmt::Debug;

use super::{FusedKerSpec, FusedSpec, InputStoreKer, MatMatMulKer, OutputStoreKer, PackedStoreKer};
use downcast_rs::{impl_downcast, Downcast};
use tract_data::prelude::*;

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

#[derive(Default)]
pub struct ScratchSpaceFusedNonLinear<TI: Copy> {
    uspecs: TVec<FusedKerSpec<TI>>,
    buffer: Vec<u8>,
    loc_dependant: TVec<(usize, usize)>,
}

impl<TI: Copy + 'static> ScratchSpace for ScratchSpaceFusedNonLinear<TI> {}
unsafe impl<TI: Copy + 'static> Send for ScratchSpaceFusedNonLinear<TI> {}

impl<TI: Copy + Datum + Zero> ScratchSpaceFusedNonLinear<TI> {
    pub unsafe fn prepare<K: MatMatMulKer<TI>>(&mut self, specs: &[FusedSpec]) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        self.uspecs = tvec!(FusedKerSpec::Done; specs.len() + 1);
        self.loc_dependant.clear();
        for ix in 0..specs.len() {
            let spec = &specs[ix];
            self.uspecs[ix] = match spec {
                FS::Min(m) => FKS::Min(*m.to_scalar_unchecked()),
                FS::Max(m) => FKS::Max(*m.to_scalar_unchecked()),
                FS::ScalarMul(t) => FKS::ScalarMul(*t.to_scalar_unchecked()),
                FS::ScalarAdd(t) => FKS::ScalarAdd(*t.to_scalar_unchecked()),
                FS::QScale(s, rp, m) => FKS::QScale(*s, *rp, *m),
                FS::PerRowAdd(_) | FS::PerRowMul(_) => {
                    self.loc_dependant.push((ix, TI::datum_type().size_of() * K::mr()));
                    FusedKerSpec::Done
                }
                FS::PerColAdd(_) | FS::PerColMul(_) => {
                    self.loc_dependant.push((ix, TI::datum_type().size_of() * K::nr()));
                    FusedKerSpec::Done
                }
                FS::AddRowColProducts(_, _) => {
                    self.loc_dependant.push((ix, TI::datum_type().size_of() * (K::mr() + K::nr())));
                    FusedKerSpec::Done
                }
                FS::Store(_) | FS::AddUnicast(_) => {
                    self.loc_dependant.push((ix, TI::datum_type().size_of() * K::mr() * K::nr()));
                    FusedKerSpec::Done
                }
                FS::AddMatMul { .. } => {
                    self.loc_dependant.push((ix, std::mem::size_of::<InputStoreKer>()));
                    FusedKerSpec::Done
                }
            }
        }
        let size: usize = self.loc_dependant.iter().map(|pair| pair.1).sum();
        if self.buffer.len() < size {
            self.buffer.reserve(size - self.buffer.len())
        }
        let mut offset = 0;
        for ld in &mut self.loc_dependant {
            let size = ld.1;
            ld.1 = offset;
            offset += size;
        }
    }

    #[inline]
    pub unsafe fn for_valid_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceFusedNonLinear { uspecs, loc_dependant, buffer } = self;
        for (ix, offset) in loc_dependant.iter_mut() {
            let spec = &specs[*ix];
            uspecs[*ix] = match spec {
                FS::PerRowAdd(v) => FKS::PerRowAdd(v.as_ptr_unchecked::<TI>().add(down * K::mr())),
                FS::PerRowMul(v) => FKS::PerRowMul(v.as_ptr_unchecked::<TI>().add(down * K::mr())),
                FS::PerColAdd(v) => FKS::PerColAdd(v.as_ptr_unchecked::<TI>().add(right * K::nr())),
                FS::PerColMul(v) => FKS::PerColMul(v.as_ptr_unchecked::<TI>().add(right * K::nr())),
                FS::AddRowColProducts(rows, cols) => {
                    let row_ptr = rows.as_ptr_unchecked::<TI>().add(down * K::mr());
                    let col_ptr = cols.as_ptr_unchecked::<TI>().add(right * K::nr());
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => FKS::AddUnicast(store.tile_c(down, right)),
                FS::Store(c_store) => FKS::Store(c_store.tile_c(down, right)),
                FS::AddMatMul { k, a, b } => {
                    let pa = a.panel(down).ptr;
                    K::prefetch(pa as _, 512);
                    let scratch =
                        buffer.as_mut_ptr().offset(*offset as isize) as *mut InputStoreKer;
                    *scratch = b.panel_b(right);
                    FKS::AddMatMul { k: *k, pa, pb: scratch, cpu_variant: 0 }
                }
                _ => std::hint::unreachable_unchecked(),
            };
        }
    }

    #[inline(never)]
    pub unsafe fn for_border_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceFusedNonLinear { uspecs, loc_dependant, buffer } = self;
        for (ix, offset) in loc_dependant.iter_mut() {
            let spec = &specs[*ix];
            let scratch = buffer.as_mut_ptr().offset(*offset as isize);
            uspecs[*ix] = match spec {
                FS::PerRowAdd(v) | FS::PerRowMul(v) | FS::PerColMul(v) | FS::PerColAdd(v) => {
                    let (dir, r) = if matches!(spec, FS::PerColAdd(_) | FS::PerColMul(_)) {
                        (right, K::nr())
                    } else {
                        (down, K::mr())
                    };
                    let buf = std::slice::from_raw_parts_mut(scratch as *mut TI, r);
                    let have = v.len().saturating_sub(dir * r).min(r);
                    let ptr = if have < K::mr() {
                        buf[have..].iter_mut().for_each(|x| *x = TI::zero());
                        if have > 0 {
                            buf[..have].copy_from_slice(&v.as_slice_unchecked()[dir * r..][..have]);
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(dir * r)
                    };
                    match spec {
                        FS::PerRowAdd(_) => FKS::PerRowAdd(ptr),
                        FS::PerRowMul(_) => FKS::PerRowMul(ptr),
                        FS::PerColAdd(_) => FKS::PerColAdd(ptr),
                        FS::PerColMul(_) => FKS::PerColMul(ptr),
                        _ => std::hint::unreachable_unchecked(),
                    }
                }
                FS::AddRowColProducts(rows, cols) => {
                    let r = std::slice::from_raw_parts_mut(scratch as *mut TI, K::mr());
                    let have = rows.len() - down * K::mr();
                    let row_ptr = if have < K::mr() {
                        r[..have]
                            .copy_from_slice(&rows.as_slice_unchecked()[down * K::mr()..][..have]);
                        r.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let c = std::slice::from_raw_parts_mut(
                        (scratch as *mut TI).offset(K::mr() as isize),
                        K::nr(),
                    );
                    let have = cols.len() - right * K::nr();
                    let col_ptr = if have < K::nr() {
                        c[..have]
                            .copy_from_slice(&cols.as_slice_unchecked()[right * K::nr()..][..have]);
                        c.as_ptr()
                    } else {
                        cols.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => {
                    let row_byte_stride = store.row_byte_stride;
                    let col_byte_stride = store.col_byte_stride;
                    let tile_offset = row_byte_stride * down as isize * K::mr() as isize
                        + col_byte_stride * right as isize * K::nr() as isize;
                    let tile_ptr = store.ptr.offset(tile_offset);
                    let tmp_d_tile =
                        std::slice::from_raw_parts_mut(scratch as *mut TI, K::mr() * K::nr());
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
                        std::hint::unreachable_unchecked();
                    }
                    FKS::AddUnicast(OutputStoreKer {
                        ptr: tmp_d_tile.as_ptr() as _,
                        row_byte_stride: std::mem::size_of::<TI>() as isize,
                        col_byte_stride: (std::mem::size_of::<TI>() * K::mr()) as isize,
                        item_size: std::mem::size_of::<TI>(),
                    })
                }
                FS::Store(c_store) => {
                    let tmpc = OutputStoreKer {
                        ptr: scratch as _,
                        item_size: c_store.item_size,
                        row_byte_stride: c_store.item_size as isize,
                        col_byte_stride: (c_store.item_size * K::mr()) as isize,
                    };
                    FKS::Store(tmpc)
                }
                FS::AddMatMul { k, a, b } => {
                    let pa = a.panel(down).ptr;
                    K::prefetch(pa as _, 512);
                    let scratch =
                        buffer.as_mut_ptr().offset(*offset as isize) as *mut InputStoreKer;
                    *scratch = b.panel_b(right);
                    FKS::AddMatMul { k: *k, pa, pb: scratch, cpu_variant: 0 }
                }
                _ => std::hint::unreachable_unchecked(),
            };
        }
    }

    #[inline]
    pub fn uspecs(&self) -> &[FusedKerSpec<TI>] {
        &self.uspecs
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
