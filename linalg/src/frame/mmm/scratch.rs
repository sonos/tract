use super::{BinOp, FusedKerSpec, FusedSpec, MatMatMulKer, OutputStoreKer};
use crate::LADatum;
use downcast_rs::{impl_downcast, Downcast};
use std::alloc::Layout;
use std::fmt::Debug;
use tract_data::internal::num_integer::Integer;
use tract_data::internal::*;

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

#[derive(Debug)]
pub struct ScratchSpaceImpl<TI: LADatum> {
    uspecs: Vec<FusedKerSpec<TI>>,
    layout: Layout,
    buffer: *mut u8,
    loc_dependant: TVec<LocDependant>,
    valid_down_tiles: usize,
    remnant_down: usize,
    valid_right_tiles: usize,
    remnant_right: usize,
}

impl<TI: LADatum> Default for ScratchSpaceImpl<TI> {
    fn default() -> Self {
        ScratchSpaceImpl {
            uspecs: vec![],
            layout: unsafe { Layout::from_size_align_unchecked(0, 1) },
            buffer: std::ptr::null_mut(),
            loc_dependant: tvec!(),
            valid_down_tiles: 0,
            remnant_down: 0,
            valid_right_tiles: 0,
            remnant_right: 0,
        }
    }
}

#[derive(Debug, new)]
struct LocDependant {
    spec: usize,
    uspec: usize,
    loc: *const u8,
    buffer: Option<*mut u8>,
}

impl<TI: LADatum> ScratchSpace for ScratchSpaceImpl<TI> {}
unsafe impl<TI: LADatum> Send for ScratchSpaceImpl<TI> {}

impl<TI: LADatum> Drop for ScratchSpaceImpl<TI> {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            unsafe {
                std::alloc::dealloc(self.buffer as _, self.layout);
            }
        }
    }
}

#[derive(Debug)]
struct AddMatMulTemp {
    ptr: *const u8,
    panel_id: usize,
    is_b: bool,
}

impl<TI: LADatum> ScratchSpaceImpl<TI> {
    pub unsafe fn prepare<K: MatMatMulKer<TI>>(
        &mut self,
        m: usize,
        n: usize,
        specs: &[FusedSpec],
    ) -> TractResult<()> {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        self.uspecs.clear();
        self.loc_dependant.clear();
        self.uspecs.reserve(specs.len() + 2);
        self.uspecs.push(FusedKerSpec::Clear);
        self.valid_down_tiles = m / K::mr();
        self.remnant_down = m % K::mr();
        self.valid_right_tiles = n / K::nr();
        self.remnant_right = n % K::nr();
        let mut offset = 0;
        let mut align = std::mem::size_of::<*const ()>();
        fn ld(spec: usize, uspec: usize, loc: *mut u8) -> LocDependant {
            LocDependant { spec, uspec, loc, buffer: None }
        }
        // we're cheating here, storing offset as the buf pointer first
        for (ix, spec) in specs.iter().enumerate() {
            let uspec = match spec {
                FS::BinScalar(t, op) => match op {
                    BinOp::Min => FKS::ScalarMin(*t.to_scalar()?),
                    BinOp::Max => FKS::ScalarMax(*t.to_scalar()?),
                    BinOp::Mul => FKS::ScalarMul(*t.to_scalar()?),
                    BinOp::Add => FKS::ScalarAdd(*t.to_scalar()?),
                    BinOp::Sub => FKS::ScalarSub(*t.to_scalar()?),
                    BinOp::SubF => FKS::ScalarSubF(*t.to_scalar()?),
                },
                FS::ShiftLeft(s) => FKS::ShiftLeft(*s),
                FS::RoundingShiftRight(s, rp) => FKS::RoundingShiftRight(*s, *rp),
                FS::QScale(s, rp, m) => FKS::QScale(*s, *rp, *m),
                FS::BinPerRow(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset as _));
                    offset += TI::datum_type().size_of() * K::mr();
                    FusedKerSpec::Done
                }
                FS::BinPerCol(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset as _));
                    offset += TI::datum_type().size_of() * K::nr();
                    FusedKerSpec::Done
                }
                FS::AddRowColProducts(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset as _));
                    offset += TI::datum_type().size_of() * (K::mr() + K::nr());
                    FusedKerSpec::Done
                }
                FS::Store(_) | FS::AddUnicast(_) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset as _));
                    offset += TI::datum_type().size_of() * K::mr() * K::nr();
                    FusedKerSpec::Done
                }
                FS::LeakyRelu(t) => FKS::LeakyRelu(*t.to_scalar()?),
                FS::AddMatMul { a, b, .. } => {
                    for input in [a, b] {
                        /*
                        if cfg!(debug_assertions) {
                            if let InputStore::Packed { ptr, .. } = a {
                                ensure!(*ptr as usize % K::alignment_bytes_packed_a() == 0);
                            }
                            if let InputStore::Packed { ptr, .. } = b {
                                ensure!(*ptr as usize % K::alignment_bytes_packed_b() == 0);
                            }
                        }
                        */
                        let mut ld = ld(ix, self.uspecs.len(), offset as _);
                        offset += std::mem::size_of::<AddMatMulTemp>();
                        if let Some(tmp) = input.scratch_panel_buffer_layout() {
                            align = tmp.align().lcm(&align);
                            offset = Integer::next_multiple_of(&offset, &tmp.align());
                            ld.buffer = Some(offset as _);
                            offset += tmp.size();
                        }
                        self.loc_dependant.push(ld);
                    }
                    FusedKerSpec::Done
                }
            };
            self.uspecs.push(uspec);
        }
        self.uspecs.push(FKS::Done);
        if offset > self.layout.size() || align > self.layout.align() {
            if !self.buffer.is_null() {
                std::alloc::dealloc(self.buffer as _, self.layout);
            }
            self.layout = Layout::from_size_align_unchecked(offset, align);
            self.buffer = std::alloc::alloc(self.layout);
            assert!(!self.buffer.is_null());
        }
        let mut mat_mul_half_done = false;
        for LocDependant { loc, buffer, spec, .. } in &mut self.loc_dependant {
            *loc = self.buffer.offset(*loc as _);
            if let Some(b) = buffer {
                *b = self.buffer.offset(*b as _);
            }
            let spec = specs.get_unchecked(*spec);
            #[allow(clippy::single_match)]
            match spec {
                FS::AddMatMul { .. } => {
                    let scratch = &mut *(*loc as *mut AddMatMulTemp);
                    scratch.is_b = mat_mul_half_done;
                    scratch.panel_id = usize::MAX;
                    mat_mul_half_done = !mat_mul_half_done;
                }
                _ => (),
            };
        }
        Ok(())
    }

    pub unsafe fn run<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        if down < self.valid_down_tiles && right < self.valid_right_tiles {
            self.for_valid_tile::<K>(specs, down, right);
            let err = K::kernel(self.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {err}");
        } else {
            let remnant_down =
                if down < self.valid_down_tiles { K::mr() } else { self.remnant_down };
            let remnant_right =
                if right < self.valid_right_tiles { K::nr() } else { self.remnant_right };
            self.for_border_tile::<K>(specs, down, right, remnant_down, remnant_right);
            let err = K::kernel(self.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {err}");
            self.postprocess_tile::<K>(specs, down, right, remnant_down, remnant_right);
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
        let ScratchSpaceImpl { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 2 == uspecs.len());
        let mut adhoc_pa: *const u8 = std::ptr::null();
        for LocDependant { spec, uspec, loc, buffer } in loc_dependant.iter_mut() {
            let spec = specs.get_unchecked(*spec);
            *uspecs.get_unchecked_mut(*uspec) = match spec {
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
                FS::AddMatMul { a, b } => {
                    let scratch = &mut *(*loc as *mut AddMatMulTemp);
                    if !scratch.is_b {
                        if scratch.panel_id != down {
                            scratch.ptr = a.panel_bytes(down, *buffer);
                            scratch.panel_id = down;
                        }
                        adhoc_pa = scratch.ptr;
                        FKS::Done // will be overriden by the second pass for B. absolutely not
                                  // done.
                    } else {
                        if scratch.panel_id != right {
                            scratch.ptr = b.panel_bytes(right, *buffer);
                            scratch.panel_id = right;
                        }
                        FKS::AddMatMul { k: b.k(), pa: adhoc_pa, pb: scratch.ptr, cpu_variant: 0 }
                    }
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
        m_remnant: usize,
        n_remnant: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceImpl { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 2 == uspecs.len());
        let mut adhoc_pa: *const u8 = std::ptr::null();
        for LocDependant { spec, uspec, loc, buffer } in loc_dependant.iter_mut() {
            let spec = specs.get_unchecked(*spec);
            *uspecs.get_unchecked_mut(*uspec) = match spec {
                FS::BinPerRow(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(*loc as *mut TI, K::mr());
                    let ptr = if m_remnant < K::mr() {
                        if m_remnant > 0 {
                            buf.get_unchecked_mut(..m_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(down * K::mr()..)
                                    .get_unchecked(..m_remnant),
                            );
                        }
                        if cfg!(debug_assertions) {
                            buf.get_unchecked_mut(m_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
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
                    let buf = std::slice::from_raw_parts_mut(*loc as *mut TI, K::nr());
                    let ptr = if n_remnant < K::nr() {
                        if n_remnant > 0 {
                            buf.get_unchecked_mut(..n_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(right * K::nr()..)
                                    .get_unchecked(..n_remnant),
                            );
                        }
                        if cfg!(debug_assertions) {
                            buf.get_unchecked_mut(n_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
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
                    let r = std::slice::from_raw_parts_mut(*loc as *mut TI, K::mr());
                    let row_ptr = if m_remnant < K::mr() {
                        r.get_unchecked_mut(..m_remnant).copy_from_slice(
                            rows.as_slice_unchecked()
                                .get_unchecked(down * K::mr()..)
                                .get_unchecked(..m_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(m_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        r.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let c = std::slice::from_raw_parts_mut((*loc as *mut TI).add(K::mr()), K::nr());
                    let col_ptr = if n_remnant < K::nr() {
                        c.get_unchecked_mut(..n_remnant).copy_from_slice(
                            cols.as_slice_unchecked()
                                .get_unchecked(right * K::nr()..)
                                .get_unchecked(..n_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(n_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
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
                        std::slice::from_raw_parts_mut(*loc as *mut TI, K::mr() * K::nr());
                    if cfg!(debug_assertions) {
                        tmp_d_tile.iter_mut().for_each(|t| *t = TI::zero());
                    }
                    for r in 0..m_remnant as isize {
                        for c in 0..n_remnant as isize {
                            let inner_offset = c * col_byte_stride + r * row_byte_stride;
                            if inner_offset + tile_offset
                                < (store.item_size * store.item_count) as isize
                            {
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
                        ptr: *loc as _,
                        item_size: c_store.item_size,
                        row_byte_stride: c_store.item_size as isize,
                        col_byte_stride: (c_store.item_size * K::mr()) as isize,
                    };
                    FKS::Store(tmpc)
                }
                FS::AddMatMul { a, b } => {
                    let scratch = &mut *(*loc as *mut AddMatMulTemp);
                    if !scratch.is_b {
                        if scratch.panel_id != down {
                            scratch.ptr = a.panel_bytes(down, *buffer);
                            scratch.panel_id = down;
                        }
                        adhoc_pa = scratch.ptr;
                        FKS::Done // will be overriden by the second pass for B. absolutely not
                                  // done.
                    } else {
                        if scratch.panel_id != right {
                            scratch.ptr = b.panel_bytes(right, *buffer);
                            scratch.panel_id = right;
                        }
                        FKS::AddMatMul { k: b.k(), pa: adhoc_pa, pb: scratch.ptr, cpu_variant: 0 }
                    }
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
        TI: LADatum,
    {
        for LocDependant { spec, uspec, .. } in self.loc_dependant.iter() {
            let spec = specs.get_unchecked(*spec);
            let ker_spec = self.uspecs.get_unchecked(*uspec);
            if let (FusedSpec::Store(c_store), FusedKerSpec::Store(tmp)) = (spec, ker_spec) {
                c_store.set_from_tile(down, right, m_remnant, n_remnant, tmp)
            }
        }
    }
}
