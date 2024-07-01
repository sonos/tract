use super::{BinOp, FusedKerSpec, FusedSpec, MatMatMulKer, OutputStoreKer};
use crate::LADatum;
use downcast_rs::{impl_downcast, Downcast};
use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use tract_data::internal::num_integer::Integer;
use tract_data::internal::*;

static GENERATION: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static TLS: RefCell<TLSScratch> = Default::default();
}

#[derive(Default, Debug)]
struct TLSScratch {
    generation: usize,
    blob: Blob,
    ker_specs_16: Vec<FusedKerSpec<f16>>,
    ker_specs_32: Vec<FusedKerSpec<f32>>,
    ker_specs_64: Vec<FusedKerSpec<f64>>,
}

impl TLSScratch {
    #[allow(unknown_lints, clippy::missing_transmute_annotations)]
    fn ker_specs<TI: LADatum>(&mut self) -> &mut Vec<FusedKerSpec<TI>> {
        unsafe {
            if TI::datum_type() == f32::datum_type() || TI::datum_type() == i32::datum_type() {
                std::mem::transmute(&mut self.ker_specs_32)
            } else if TI::datum_type() == f16::datum_type() {
                std::mem::transmute(&mut self.ker_specs_16)
            } else if TI::datum_type() == f64::datum_type() {
                std::mem::transmute(&mut self.ker_specs_64)
            } else {
                todo!();
            }
        }
    }

    fn sync<TI: LADatum>(&mut self, scratch: &ScratchSpaceImpl<TI>) {
        if self.generation == scratch.generation {
            return;
        }
        let ker_specs = self.ker_specs::<TI>();
        ker_specs.clear();
        ker_specs.extend_from_slice(&scratch.ker_specs);

        unsafe {
            self.blob.ensure_size_and_align(scratch.blob_size, scratch.blob_align);

            for LocDependant { loc, ker_spec, .. } in &scratch.loc_dependant {
                #[allow(clippy::single_match)]
                if matches!(scratch.ker_specs[*ker_spec], FusedKerSpec::AddMatMul { .. }) {
                    let scratch = &mut *(self.blob.as_ptr().add(*loc) as *mut AddMatMulTemp);
                    scratch.panel_a_id = usize::MAX;
                    scratch.panel_b_id = usize::MAX;
                };
            }
        }
        self.generation = scratch.generation;
    }
}

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

#[derive(Debug, Default)]
pub struct ScratchSpaceImpl<TI: LADatum> {
    generation: usize,
    blob_size: usize,
    blob_align: usize,
    ker_specs: Vec<FusedKerSpec<TI>>,
    loc_dependant: TVec<LocDependant>,
    valid_down_tiles: usize,
    remnant_down: usize,
    valid_right_tiles: usize,
    remnant_right: usize,
}

#[derive(Debug, new)]
struct LocDependant {
    spec: usize,
    ker_spec: usize,
    // offset for the location dependant structure
    loc: usize,
    // offset of its associated dynamic-size buffers
    buffer_a: Option<usize>,
    buffer_b: Option<usize>,
}

impl<TI: LADatum> ScratchSpace for ScratchSpaceImpl<TI> {}
unsafe impl<TI: LADatum> Send for ScratchSpaceImpl<TI> {}

#[derive(Debug)]
struct AddMatMulTemp {
    ptr_a: *const u8,
    panel_a_id: usize,
    ptr_b: *const u8,
    panel_b_id: usize,
}

impl<TI: LADatum> ScratchSpaceImpl<TI> {
    pub unsafe fn prepare(
        &mut self,
        ker: &impl MatMatMulKer<Acc = TI>,
        m: usize,
        n: usize,
        specs: &[FusedSpec],
    ) -> TractResult<()> {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        self.ker_specs.clear();
        self.loc_dependant.clear();
        self.ker_specs.reserve(specs.len() + 2);
        self.ker_specs.push(FusedKerSpec::Clear);
        self.valid_down_tiles = m / ker.mr();
        self.remnant_down = m % ker.mr();
        self.valid_right_tiles = n / ker.nr();
        self.remnant_right = n % ker.nr();
        let mut offset = 0;
        let mut align = std::mem::size_of::<*const ()>();
        fn ld(spec: usize, uspec: usize, loc: usize) -> LocDependant {
            LocDependant { spec, ker_spec: uspec, loc, buffer_a: None, buffer_b: None }
        }
        for (ix, spec) in specs.iter().enumerate() {
            let ker_spec = match spec {
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
                    self.loc_dependant.push(ld(ix, self.ker_specs.len(), offset));
                    offset += TI::datum_type().size_of() * ker.mr();
                    FusedKerSpec::Done
                }
                FS::BinPerCol(_, _) => {
                    self.loc_dependant.push(ld(ix, self.ker_specs.len(), offset));
                    offset += TI::datum_type().size_of() * ker.nr();
                    FusedKerSpec::Done
                }
                FS::AddRowColProducts(_, _) => {
                    self.loc_dependant.push(ld(ix, self.ker_specs.len(), offset));
                    offset += TI::datum_type().size_of() * (ker.mr() + ker.nr());
                    FusedKerSpec::Done
                }
                FS::Store(_) | FS::AddUnicast(_) => {
                    self.loc_dependant.push(ld(ix, self.ker_specs.len(), offset));
                    offset += TI::datum_type().size_of() * ker.mr() * ker.nr();
                    FusedKerSpec::Done
                }
                FS::LeakyRelu(t) => FKS::LeakyRelu(*t.to_scalar()?),
                FS::AddMatMul { a, b, packing } => {
                    let mut ld = ld(ix, self.ker_specs.len(), offset);
                    offset += std::mem::size_of::<AddMatMulTemp>();
                    if let Some(tmp) = a.scratch_panel_buffer_layout() {
                        align = tmp.align().lcm(&align);
                        offset = Integer::next_multiple_of(&offset, &tmp.align());
                        ld.buffer_a = Some(offset);
                        offset += tmp.size();
                    }
                    if let Some(tmp) = b.scratch_panel_buffer_layout() {
                        align = tmp.align().lcm(&align);
                        offset = Integer::next_multiple_of(&offset, &tmp.align());
                        ld.buffer_b = Some(offset);
                        offset += tmp.size();
                    }
                    self.loc_dependant.push(ld);
                    FusedKerSpec::AddMatMul {
                        k: 0,
                        pa: std::ptr::null(),
                        pb: std::ptr::null(),
                        packing: *packing,
                    }
                }
            };
            self.ker_specs.push(ker_spec);
        }
        self.ker_specs.push(FKS::Done);
        self.blob_size = offset;
        self.blob_align = align;

        self.generation = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub unsafe fn run(
        &self,
        ker: &impl MatMatMulKer<Acc = TI>,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) -> TractResult<()> {
        TLS.with_borrow_mut(|tls| {
            tls.sync(self);
            if down < self.valid_down_tiles && right < self.valid_right_tiles {
                self.for_valid_tile(ker, specs, tls, down, right)?;
                let err = ker.kernel(tls.ker_specs());
                debug_assert_eq!(err, 0, "Kernel return error {err}");
            } else {
                let remnant_down =
                    if down < self.valid_down_tiles { ker.mr() } else { self.remnant_down };
                let remnant_right =
                    if right < self.valid_right_tiles { ker.nr() } else { self.remnant_right };
                self.for_border_tile(ker, specs, tls, down, right, remnant_down, remnant_right)?;
                let err = ker.kernel(tls.ker_specs());
                debug_assert_eq!(err, 0, "Kernel return error {err}");
                self.postprocess_tile(specs, tls, down, right, remnant_down, remnant_right)?;
            }
            Ok(())
        })
    }

    #[inline(always)]
    unsafe fn for_valid_tile(
        &self,
        ker: &impl MatMatMulKer<Acc = TI>,
        specs: &[FusedSpec],
        tls: &mut TLSScratch,
        down: usize,
        right: usize,
    ) -> TractResult<()> {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceImpl { ker_specs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 2 == ker_specs.len());
        for LocDependant { spec, ker_spec, loc, buffer_a, buffer_b } in loc_dependant {
            let spec = specs.get_unchecked(*spec);
            let it = match spec {
                FS::BinPerRow(v, op) => {
                    let v = v.as_ptr_unchecked::<TI>().add(down * ker.mr());
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
                    let v = v.as_ptr_unchecked::<TI>().add(right * ker.nr());
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
                    let row_ptr = rows.as_ptr_unchecked::<TI>().add(down * ker.mr());
                    let col_ptr = cols.as_ptr_unchecked::<TI>().add(right * ker.nr());
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => FKS::AddUnicast(store.tile_c(down, right)),
                FS::Store(c_store) => FKS::Store(c_store.tile_c(down, right)),
                FS::AddMatMul { a, b, packing } => {
                    let scratch =
                        (tls.blob.as_mut_ptr().add(*loc) as *mut AddMatMulTemp).as_mut().unwrap();
                    if scratch.panel_a_id != down {
                        scratch.ptr_a =
                            a.panel_bytes(down, buffer_a.map(|o| tls.blob.as_mut_ptr().add(o)))?;
                        scratch.panel_a_id = down;
                    }
                    if scratch.panel_b_id != right {
                        scratch.ptr_b =
                            b.panel_bytes(right, buffer_b.map(|o| tls.blob.as_mut_ptr().add(o)))?;
                        scratch.panel_b_id = right;
                    }
                    FKS::AddMatMul {
                        k: b.k(),
                        pa: scratch.ptr_a,
                        pb: scratch.ptr_b,
                        packing: *packing,
                    }
                }
                _ => std::hint::unreachable_unchecked(),
            };
            *tls.ker_specs().get_unchecked_mut(*ker_spec) = it;
        }
        Ok(())
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn for_border_tile(
        &self,
        ker: &impl MatMatMulKer<Acc = TI>,
        specs: &[FusedSpec],
        tls: &mut TLSScratch,
        down: usize,
        right: usize,
        m_remnant: usize,
        n_remnant: usize,
    ) -> TractResult<()> {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        for LocDependant { spec, ker_spec: uspec, loc, buffer_a, buffer_b } in &self.loc_dependant {
            let loc = tls.blob.as_mut_ptr().add(*loc);
            let spec = specs.get_unchecked(*spec);
            let it = match spec {
                FS::BinPerRow(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(loc as *mut TI, ker.mr());
                    let ptr = if m_remnant < ker.mr() {
                        if m_remnant > 0 {
                            buf.get_unchecked_mut(..m_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(down * ker.mr()..)
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
                        v.as_ptr_unchecked::<TI>().add(down * ker.mr())
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
                    let buf = std::slice::from_raw_parts_mut(loc as *mut TI, ker.nr());
                    let ptr = if n_remnant < ker.nr() {
                        if n_remnant > 0 {
                            buf.get_unchecked_mut(..n_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(right * ker.nr()..)
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
                        v.as_ptr_unchecked::<TI>().add(right * ker.nr())
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
                    let r = std::slice::from_raw_parts_mut(loc as *mut TI, ker.mr());
                    let row_ptr = if m_remnant < ker.mr() {
                        r.get_unchecked_mut(..m_remnant).copy_from_slice(
                            rows.as_slice_unchecked()
                                .get_unchecked(down * ker.mr()..)
                                .get_unchecked(..m_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(m_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        r.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * ker.mr())
                    };
                    let c =
                        std::slice::from_raw_parts_mut((loc as *mut TI).add(ker.mr()), ker.nr());
                    let col_ptr = if n_remnant < ker.nr() {
                        c.get_unchecked_mut(..n_remnant).copy_from_slice(
                            cols.as_slice_unchecked()
                                .get_unchecked(right * ker.nr()..)
                                .get_unchecked(..n_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(n_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        c.as_ptr()
                    } else {
                        cols.as_ptr_unchecked::<TI>().add(right * ker.nr())
                    };
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => {
                    let row_byte_stride = store.row_byte_stride;
                    let col_byte_stride = store.col_byte_stride;
                    let tile_offset = row_byte_stride * down as isize * ker.mr() as isize
                        + col_byte_stride * right as isize * ker.nr() as isize;
                    let tile_ptr = store.ptr.offset(tile_offset);
                    let tmp_d_tile =
                        std::slice::from_raw_parts_mut(loc as *mut TI, ker.mr() * ker.nr());
                    if cfg!(debug_assertions) {
                        tmp_d_tile.iter_mut().for_each(|t| *t = TI::zero());
                    }
                    for r in 0..m_remnant as isize {
                        for c in 0..n_remnant as isize {
                            let inner_offset = c * col_byte_stride + r * row_byte_stride;
                            if inner_offset + tile_offset
                                < (store.item_size * store.item_count) as isize
                            {
                                *tmp_d_tile.get_unchecked_mut(r as usize + c as usize * ker.mr()) =
                                    *(tile_ptr.offset(inner_offset) as *const TI);
                            }
                        }
                    }
                    FKS::AddUnicast(OutputStoreKer {
                        ptr: tmp_d_tile.as_ptr() as _,
                        row_byte_stride: std::mem::size_of::<TI>() as isize,
                        col_byte_stride: (std::mem::size_of::<TI>() * ker.mr()) as isize,
                        item_size: std::mem::size_of::<TI>(),
                    })
                }
                FS::Store(c_store) => {
                    let tmpc = OutputStoreKer {
                        ptr: loc as _,
                        item_size: c_store.item_size,
                        row_byte_stride: c_store.item_size as isize,
                        col_byte_stride: (c_store.item_size * ker.mr()) as isize,
                    };
                    FKS::Store(tmpc)
                }
                FS::AddMatMul { a, b, packing } => {
                    let scratch = (loc as *mut AddMatMulTemp).as_mut().unwrap();
                    if scratch.panel_a_id != down {
                        scratch.ptr_a =
                            a.panel_bytes(down, buffer_a.map(|o| tls.blob.as_mut_ptr().add(o)))?;
                        scratch.panel_a_id = down;
                    }
                    if scratch.panel_b_id != right {
                        scratch.ptr_b =
                            b.panel_bytes(right, buffer_b.map(|o| tls.blob.as_mut_ptr().add(o)))?;
                        scratch.panel_b_id = right;
                    }
                    FKS::AddMatMul {
                        k: b.k(),
                        pa: scratch.ptr_a,
                        pb: scratch.ptr_b,
                        packing: *packing,
                    }
                }
                _ => std::hint::unreachable_unchecked(),
            };
            *tls.ker_specs().get_unchecked_mut(*uspec) = it;
        }
        Ok(())
    }

    #[inline]
    pub fn uspecs(&self) -> &[FusedKerSpec<TI>] {
        &self.ker_specs
    }

    unsafe fn postprocess_tile(
        &self,
        specs: &[FusedSpec],
        tls: &mut TLSScratch,
        down: usize,
        right: usize,
        m_remnant: usize,
        n_remnant: usize,
    ) -> TractResult<()>
    where
        TI: LADatum,
    {
        for LocDependant { spec, ker_spec: uspec, .. } in self.loc_dependant.iter() {
            let spec = specs.get_unchecked(*spec);
            let ker_spec = tls.ker_specs::<TI>().get_unchecked(*uspec);
            if let (FusedSpec::Store(c_store), FusedKerSpec::Store(tmp)) = (spec, ker_spec) {
                c_store.set_from_tile(down, right, m_remnant, n_remnant, tmp)
            }
        }
        Ok(())
    }
}
