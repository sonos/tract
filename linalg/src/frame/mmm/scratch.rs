use num_traits::Zero;
use std::fmt::Debug;

use super::{BinOp, FusedKerSpec, FusedSpec, InputStoreKer, MatMatMulKer, OutputStoreKer};
use downcast_rs::{impl_downcast, Downcast};
use tract_data::prelude::*;

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

#[derive(Default, Debug)]
pub struct ScratchSpaceFusedNonLinear<TI: Copy> {
    uspecs: Vec<FusedKerSpec<TI>>,
    pub buffer: Vec<u8>,
    loc_dependant: TVec<(usize, *const u8)>,
}

impl<TI: Copy + 'static> ScratchSpace for ScratchSpaceFusedNonLinear<TI> {}
unsafe impl<TI: Copy + 'static> Send for ScratchSpaceFusedNonLinear<TI> {}

impl<TI: Copy + Datum + Zero> ScratchSpaceFusedNonLinear<TI> {
    pub unsafe fn prepare<K: MatMatMulKer<TI>>(&mut self, specs: &[FusedSpec]) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        self.uspecs.clear();
        self.loc_dependant.clear();
        self.uspecs.reserve(specs.len() + 1);
        let mut offset = 0;
        // we're cheating here, storing offset as the buf pointer first
        for ix in 0..specs.len() {
            let spec = &specs[ix];
            let uspec = match spec {
                FS::BinScalar(t, op) => match op {
                    BinOp::Min => FKS::ScalarMin(*t.to_scalar_unchecked()),
                    BinOp::Max => FKS::ScalarMax(*t.to_scalar_unchecked()),
                    BinOp::Mul => FKS::ScalarMul(*t.to_scalar_unchecked()),
                    BinOp::Add => FKS::ScalarAdd(*t.to_scalar_unchecked()),
                    BinOp::Sub => FKS::ScalarSub(*t.to_scalar_unchecked()),
                    BinOp::SubF => FKS::ScalarSubF(*t.to_scalar_unchecked()),
                },
                FS::QScale(s, rp, m) => FKS::QScale(*s, *rp, *m),
                FS::BinPerRow(_, _) => {
                    self.loc_dependant.push((ix, offset as _));
                    offset += TI::datum_type().size_of() * K::mr();
                    FusedKerSpec::Done
                }
                FS::BinPerCol(_, _) => {
                    self.loc_dependant.push((ix, offset as _));
                    offset += TI::datum_type().size_of() * K::nr();
                    FusedKerSpec::Done
                }
                FS::AddRowColProducts(_, _) => {
                    self.loc_dependant.push((ix, offset as _));
                    offset += TI::datum_type().size_of() * (K::mr() + K::nr());
                    FusedKerSpec::Done
                }
                FS::Store(_) | FS::AddUnicast(_) => {
                    self.loc_dependant.push((ix, offset as _));
                    offset += TI::datum_type().size_of() * K::mr() * K::nr();
                    FusedKerSpec::Done
                }
                FS::AddMatMul { .. } => {
                    self.loc_dependant.push((ix, offset as _));
                    offset += std::mem::size_of::<InputStoreKer>();
                    FusedKerSpec::Done
                }
            };
            self.uspecs.push(uspec);
        }
        self.uspecs.push(FKS::Done);
        self.buffer.resize(offset, 0);
        for (_, loc) in &mut self.loc_dependant {
            *loc = self.buffer.as_ptr().offset(*loc as _);
        }
    }

    #[inline(always)]
    pub unsafe fn for_valid_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceFusedNonLinear { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 1 == uspecs.len());
        for (ix, ptr) in loc_dependant.iter_mut() {
            let spec = specs.get_unchecked(*ix);
            *uspecs.get_unchecked_mut(*ix) = match spec {
                FS::BinPerRow(v, op) => {
                    let v = v.as_ptr_unchecked::<TI>().add(down * K::mr());
                    match op {
                        BinOp::Min => FKS::PerRowMin(v),
                        BinOp::Max => FKS::PerRowMax(v),
                        BinOp::Add => FKS::PerRowAdd(v),
                        BinOp::Mul => FKS::PerRowMul(v),
                        BinOp::Sub => FKS::PerRowSub(v),
                        BinOp::SubF => FKS::PerRowSubF(v),
                    }
                }
                FS::BinPerCol(v, op) => {
                    let v = v.as_ptr_unchecked::<TI>().add(right * K::nr());
                    match op {
                        BinOp::Min => FKS::PerColMin(v),
                        BinOp::Max => FKS::PerColMax(v),
                        BinOp::Add => FKS::PerColAdd(v),
                        BinOp::Mul => FKS::PerColMul(v),
                        BinOp::Sub => FKS::PerColSub(v),
                        BinOp::SubF => FKS::PerColSubF(v),
                    }
                }
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
                    let scratch = *ptr as *mut InputStoreKer;
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
        let ScratchSpaceFusedNonLinear { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 1 == uspecs.len());
        for (ix, ptr) in loc_dependant.iter_mut() {
            let spec = specs.get_unchecked(*ix);
            *uspecs.get_unchecked_mut(*ix) = match spec {
                FS::BinPerRow(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(*ptr as *mut TI, K::mr());
                    let have = v.len().saturating_sub(down * K::mr()).min(K::mr());
                    let ptr = if have < K::mr() {
                        if have > 0 {
                            buf.get_unchecked_mut(..have).copy_from_slice(
                                &v.as_slice_unchecked()
                                    .get_unchecked(down * K::mr()..)
                                    .get_unchecked(..have),
                            );
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    match op {
                        BinOp::Min => FKS::PerRowMin(ptr),
                        BinOp::Max => FKS::PerRowMax(ptr),
                        BinOp::Add => FKS::PerRowAdd(ptr),
                        BinOp::Mul => FKS::PerRowMul(ptr),
                        BinOp::Sub => FKS::PerRowSub(ptr),
                        BinOp::SubF => FKS::PerRowSubF(ptr),
                    }
                }
                FS::BinPerCol(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(*ptr as *mut TI, K::nr());
                    let have = v.len().saturating_sub(right * K::nr()).min(K::nr());
                    let ptr = if have < K::nr() {
                        if have > 0 {
                            buf.get_unchecked_mut(..have).copy_from_slice(
                                &v.as_slice_unchecked()
                                    .get_unchecked(right * K::nr()..)
                                    .get_unchecked(..have),
                            );
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    match op {
                        BinOp::Min => FKS::PerColMin(ptr),
                        BinOp::Max => FKS::PerColMax(ptr),
                        BinOp::Add => FKS::PerColAdd(ptr),
                        BinOp::Mul => FKS::PerColMul(ptr),
                        BinOp::Sub => FKS::PerColSub(ptr),
                        BinOp::SubF => FKS::PerColSubF(ptr),
                    }
                }
                FS::AddRowColProducts(rows, cols) => {
                    let r = std::slice::from_raw_parts_mut(*ptr as *mut TI, K::mr());
                    let have = rows.len() - down * K::mr();
                    let row_ptr = if have < K::mr() {
                        r.get_unchecked_mut(..have).copy_from_slice(
                            &rows
                                .as_slice_unchecked()
                                .get_unchecked(down * K::mr()..)
                                .get_unchecked(..have),
                        );
                        r.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let c = std::slice::from_raw_parts_mut(
                        (*ptr as *mut TI).offset(K::mr() as isize),
                        K::nr(),
                    );
                    let have = cols.len() - right * K::nr();
                    let col_ptr = if have < K::nr() {
                        c.get_unchecked_mut(..have).copy_from_slice(
                            &cols
                                .as_slice_unchecked()
                                .get_unchecked(right * K::nr()..)
                                .get_unchecked(..have),
                        );
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
                        std::slice::from_raw_parts_mut(*ptr as *mut TI, K::mr() * K::nr());
                    debug_assert_eq!(store.item_size, 4);
                    // assumes TI is f32 or i32
                    for r in 0..K::mr() as isize {
                        for c in 0..K::nr() as isize {
                            let inner_offset = c * col_byte_stride + r * row_byte_stride;
                            if inner_offset + tile_offset < 4 * store.item_count as isize {
                                *tmp_d_tile.get_unchecked_mut(r as usize + c as usize * K::mr()) =
                                    *(tile_ptr.offset(inner_offset) as *const TI);
                            }
                        }
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
                        ptr: *ptr as _,
                        item_size: c_store.item_size,
                        row_byte_stride: c_store.item_size as isize,
                        col_byte_stride: (c_store.item_size * K::mr()) as isize,
                    };
                    FKS::Store(tmpc)
                }
                FS::AddMatMul { k, a, b } => {
                    let pa = a.panel(down).ptr;
                    K::prefetch(pa as _, 512);
                    let scratch = *ptr as *mut InputStoreKer;
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
