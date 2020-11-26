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

    fn a_storage(&self) -> &MatrixStoreSpec;
    fn b_storage(&self) -> &MatrixStoreSpec;
    fn c_storage(&self) -> &MatrixStoreSpec;

    fn internal_type(&self) -> DatumType;

    /*
    unsafe fn set_zero_point_a(&mut self, value: Tensor);
    unsafe fn set_zero_point_b(&mut self, value: Tensor);
    unsafe fn set_zero_point_c(&mut self, value: Tensor);
    unsafe fn set_scale_factor(&mut self, factor: f32);
    */

    unsafe fn b_from_data_and_offsets(&mut self, rows_offsets: &[isize], cols_offsets: &[isize]);

    unsafe fn b_vec_from_data_and_stride(&mut self, stride: isize);
    unsafe fn b_vec_from_data(&mut self);

    unsafe fn c_from_data_and_strides(&mut self, row_stride: isize, col_stride: isize);

    unsafe fn c_vec_from_data_and_stride(&mut self, stride: isize);
    unsafe fn c_vec_from_data(&mut self);

    unsafe fn run(
        &self,
        a: &TensorView,
        b: &TensorView,
        c: &mut TensorView,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

#[derive(Debug, Clone)]
pub struct MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    pub m: usize,
    pub k: usize,
    pub n: usize,

    pub a_storage: MatrixStoreSpec,
    pub b_storage: MatrixStoreSpec,
    pub c_storage: MatrixStoreSpec,

    /*
    pub zero_point_a: Option<Tensor>,
    pub zero_point_b: Option<Tensor>,
    pub zero_point_c: Option<Tensor>,
    pub scale_factor: Option<(TI, usize)>,
    */

    phantom: PhantomData<(K, TA, TB, TC, TI)>,
}

unsafe impl<K, TA, TB, TC, TI> Send for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

unsafe impl<K, TA, TB, TC, TI> Sync for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
}

impl<K, TA, TB, TC, TI> MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
{
    pub fn new(m: usize, k: usize, n: usize) -> MatMatMulImpl<K, TA, TB, TC, TI> {
        MatMatMulImpl {
            m,
            k,
            n,
            a_storage: MatrixStoreSpec::Packed { panel_len: (k * K::mr()) },
            b_storage: MatrixStoreSpec::Packed { panel_len: (k * K::nr()) },
            c_storage: MatrixStoreSpec::Strides {
                row_byte_stride: (n * std::mem::size_of::<TC>()) as isize,
                col_byte_stride: (std::mem::size_of::<TC>()) as isize,
                mr: K::mr(),
                nr: K::nr(),
            },
            /*
            zero_point_a: None,
            zero_point_b: None,
            zero_point_c: None,
            scale_factor: None,
            */
            phantom: PhantomData,
        }
    }
}

impl<K, TA, TB, TC, TI> MatMatMul for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero + Debug + 'static + AsPrimitive<TI>,
    TB: Datum + Copy + Zero + Debug + 'static + AsPrimitive<TI>,
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

    fn a_storage(&self) -> &MatrixStoreSpec {
        &self.a_storage
    }
    fn b_storage(&self) -> &MatrixStoreSpec {
        &self.b_storage
    }
    fn c_storage(&self) -> &MatrixStoreSpec {
        &self.c_storage
    }

    unsafe fn b_from_data_and_offsets(&mut self, rows_offsets: &[isize], cols_offsets: &[isize]) {
        debug_assert!(rows_offsets.len() > 0);
        debug_assert!(cols_offsets.len() > 0);
        // repeat the last offset to get to the panel boundary (pad to next multiple of nr)
        let wanted = (cols_offsets.len() + K::nr() - 1) / K::nr() * K::nr();
        let mut col_byte_offsets: Vec<_> =
            cols_offsets.iter().map(|o| o * std::mem::size_of::<TB>() as isize).collect();
        while col_byte_offsets.len() < wanted {
            col_byte_offsets.push(*col_byte_offsets.last().unwrap());
        }
        // repeat the last offset four times to simplify kernel loop unrolling
        let mut row_byte_offsets: Vec<_> = Vec::with_capacity(rows_offsets.len() + 4);
        row_byte_offsets.set_len(rows_offsets.len() + 4);
        for i in 0..rows_offsets.len() {
            *row_byte_offsets.get_unchecked_mut(i) =
                *rows_offsets.get_unchecked(i) * std::mem::size_of::<TB>() as isize;
        }
        let pad = *row_byte_offsets.get_unchecked(rows_offsets.len() - 1);
        for i in 0..4 {
            *row_byte_offsets.get_unchecked_mut(rows_offsets.len() + i) = pad;
        }
        self.b_storage =
            MatrixStoreSpec::OffsetsAndPtrs { col_byte_offsets, row_byte_offsets, nr: K::nr() };
    }

    unsafe fn b_vec_from_data_and_stride(&mut self, stride: isize) {
        self.b_storage = MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TB>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn b_vec_from_data(&mut self) {
        self.b_vec_from_data_and_stride(1)
    }

    unsafe fn c_from_data_and_strides(&mut self, row_stride: isize, col_stride: isize) {
        self.c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: row_stride * std::mem::size_of::<TC>() as isize,
            col_byte_stride: col_stride * std::mem::size_of::<TC>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data_and_stride(&mut self, stride: isize) {
        self.c_storage = MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TC>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn c_vec_from_data(&mut self) {
        self.c_vec_from_data_and_stride(1)
    }

    unsafe fn run(
        &self,
        a: &TensorView,
        b: &TensorView,
        c: &mut TensorView,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let mr = K::mr();
        let nr = K::nr();
        debug_assert_eq!(a.datum_type(), TA::datum_type());
        debug_assert_eq!(b.datum_type(), TB::datum_type());
        debug_assert_eq!(c.datum_type(), TC::datum_type());
        let prefetch = crate::ops().prefetch.as_ref();
        let m = self.m;
        let n = self.n;
        let mut scratch = ScratchSpaceFusedNonLinear::default();
        let mut tmpc = Vec::with_capacity(mr * nr);
        tmpc.set_len(mr * nr);
        let tmp_c_storage = MatrixStoreSpec::Strides {
            row_byte_stride: (std::mem::size_of::<TC>() * nr) as isize,
            col_byte_stride: std::mem::size_of::<TC>() as isize,
            mr,
            nr,
        };
        let ref mut tmp_tile = tmp_c_storage.wrap(tmpc.as_ptr());

        let a = a.as_ptr_unchecked::<TA>();
        let b = b.as_ptr_unchecked::<TB>();
        let c = c.as_ptr_mut_unchecked::<TC>();
        let ref linear = LinearSpec::k(self.k);
        let mut non_linear = non_linear.to_vec();
        /*
        if let Some(ref a0) = self.zero_point_a {
            let mut sum_b_over_k = self.sum_b_over_k(b);
            for n in 0..self.n {
                sum_b_over_k[n] = sum_b_over_k[n].neg();
            }
            let term = if a0.rank() == 0 {
                for n in 0..self.n {
                    sum_b_over_k[n] = sum_b_over_k[n] * a0.to_scalar_unchecked::<TA>().as_();
                }
                FusedSpec::PerColAdd(tensor1(&*sum_b_over_k))
            } else {
                let a0 = a0.cast_to::<TI>().unwrap();
                FusedSpec::AddRowColProducts(a0.into_owned(), tensor1(&*sum_b_over_k))
            };
            non_linear.insert(0, term);
        }
        if let Some(ref b0) = self.zero_point_b {
            let mut sum_a_over_k = self.sum_a_over_k(a);
            for m in 0..self.m {
                sum_a_over_k[m] = sum_a_over_k[m].neg();
                if let Some(ref a0) = self.zero_point_a {
                    if a0.rank() == 0 {
                        sum_a_over_k[m] =
                            a0.to_scalar_unchecked::<TA>().as_() * self.k.as_() + sum_a_over_k[m];
                    } else {
                        sum_a_over_k[m] =
                            a0.as_slice_unchecked::<TA>()[m].as_() * self.k.as_() + sum_a_over_k[m];
                    }
                }
            }
            let term = if b0.rank() == 0 {
                for m in 0..self.m {
                    sum_a_over_k[m] = sum_a_over_k[m] * b0.to_scalar_unchecked::<TB>().as_();
                }
                FusedSpec::PerRowAdd(tensor1(&*sum_a_over_k))
            } else {
                let b0 = tensor1(
                    &*b0.as_slice_unchecked::<TB>().iter().map(|b| b.as_()).collect::<Vec<_>>(),
                );
                FusedSpec::AddRowColProducts(tensor1(&*sum_a_over_k), b0)
            };
            non_linear.insert(0, term);
        }
        if let Some(scale) = self.scale_factor {
            non_linear.push(FusedSpec::QTowardsPlusInf(tensor0(scale.0), scale.1));
        }
        if let Some(c0) = &self.zero_point_c {
            non_linear.push(FusedSpec::ScalarAdd(c0.cast_to::<TI>().unwrap().into_owned()));
        }
        // makeshift Q detection
        //        if TC::datum_type().size_of() < TI::datum_type().size_of() && self.scale_factor.is_some() {
        if TC::datum_type().size_of() < TI::datum_type().size_of()
            && (self.scale_factor.is_some()
                || self.zero_point_a.is_some()
                || self.zero_point_b.is_some()
                || self.zero_point_c.is_some()
                || self.scale_factor.is_some())
        {
            non_linear.push(FusedSpec::Min(tensor0(TC::max_value().as_())));
            non_linear.push(FusedSpec::Max(tensor0(TC::min_value().as_())));
        }
        */
        let a = self.a_storage.wrap(a);
        let b = self.b_storage.wrap(b);
        //        eprintln!("{:?} {:?}", a, b);
        let mut c = self.c_storage.wrap(c);
        for ia in 0..m / mr {
            let ref a = a.panel_a(ia);
            for ib in 0..n / nr {
                if let PanelStore::Packed { ptr } = a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, ib, nr);
                match b {
                    PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let ref direct_c = c.tile_c(ia, ib);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, ia, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: direct_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
            }
            if n % nr != 0 {
                if let PanelStore::Packed { ptr } = a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, n / nr, n % nr);
                match b {
                    PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let ref tmp_tile_c = tmp_tile.tile_c(0, 0);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, ia, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(ia, n / nr, mr, n % nr, &*tmpc);
            }
        }
        if m % mr != 0 {
            let ref panel_a = a.panel_a(m / mr);
            let ref tmp_tile_c = tmp_tile.tile_c(0, 0);
            for ib in 0..n / nr {
                if let PanelStore::Packed { ptr } = panel_a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, ib, nr);
                match b {
                    PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, m / mr, ib);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(m / mr, ib, m % mr, nr, &*tmpc);
            }
            if n % nr != 0 {
                if let PanelStore::Packed { ptr } = panel_a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, n / nr, n % nr);
                match b {
                    PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, m / mr, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: panel_a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile(m / mr, n / nr, m % mr, n % nr, &*tmpc);
            }
        }
        Ok(())
    }

    /*
    unsafe fn set_zero_point_a(&mut self, value: Tensor) {
        self.zero_point_a = Some(value)
    }

    unsafe fn set_zero_point_b(&mut self, value: Tensor) {
        self.zero_point_b = Some(value)
    }

    unsafe fn set_zero_point_c(&mut self, value: Tensor) {
        self.zero_point_c = Some(value)
    }

    unsafe fn set_scale_factor(&mut self, factor: f32) {
        // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/util/gemmlowp_common.h#L16
        let factor_bits = factor.to_bits();
        let current_exponent = factor_bits >> 23;
        let bumped_multi = f32::from_bits(factor_bits & 0x007fffff | 0x3f000000);
        let int_multi = (bumped_multi * (1i64 << 31) as f32).round() as i32;
        let shift = 126 - current_exponent;
        self.scale_factor = Some((int_multi.as_(), shift as usize));
    }
    */
}

impl<K, TA, TB, TC, TI> MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + Debug + 'static + AsPrimitive<TI>,
    TB: Copy + Zero + Debug + 'static + AsPrimitive<TI>,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI> + 'static,
    i32: AsPrimitive<TI>,
{
    /*
    fn sum_a_over_k(&self, mut a: *const TA) -> Vec<TI> {
        match &self.a_storage {
            MatrixStoreSpec::Packed { .. } => {
                let mr = K::mr();
                let mut result = vec![TI::zero(); self.m];
                unsafe {
                    for p in 0..(self.m / mr) {
                        for _k in 0..self.k {
                            for row in 0..mr {
                                result[p * mr + row] = result[p * mr + row] + (*a).as_();
                                a = a.offset(1);
                            }
                        }
                    }
                    if self.m % mr != 0 {
                        let p = self.m / mr;
                        for _k in 0..self.k {
                            for row in 0..mr {
                                if row < self.m % mr {
                                    result[p * mr + row] = result[p * mr + row] + (*a).as_();
                                }
                                a = a.offset(1);
                            }
                        }
                    }
                }
                result
            }
            a => panic!("Storage for A {:?} not supported for quantized ops", a),
        }
    }

    fn sum_b_over_k(&self, mut b: *const TB) -> Vec<TI> {
        let mut result = vec![TI::zero(); self.n];
        match &self.b_storage {
            MatrixStoreSpec::Packed { .. } => unsafe {
                let nr = K::nr();
                for p in 0..(self.n / nr) {
                    for _k in 0..self.k {
                        for col in 0..nr {
                            result[p * nr + col] = result[p * nr + col] + (*b).as_();
                            b = b.offset(1);
                        }
                    }
                }
                if self.n % nr != 0 {
                    let p = self.n / nr;
                    for _k in 0..self.k {
                        for col in 0..nr {
                            if col < self.n % nr {
                                result[p * nr + col] = result[p * nr + col] + (*b).as_();
                            }
                            b = b.offset(1);
                        }
                    }
                }
            },
            MatrixStoreSpec::OffsetsAndPtrs { row_byte_offsets, col_byte_offsets, .. } => unsafe {
                for n in 0..self.n {
                    for k in 0..self.k {
                        let offset = (row_byte_offsets[k] + col_byte_offsets[n])
                            / std::mem::size_of::<TB>() as isize;
                        result[n] = result[n] + (*b.offset(offset)).as_();
                    }
                }
            },
            b => panic!("Storage {:?} for B not supported for quantized ops", b),
        }
        result
    }
    */
}

impl<K, TA, TB, TC, TI> fmt::Display for MatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + 'static,
    TB: Copy + Zero + 'static,
    TC: Copy + Debug + 'static,
    TI: Copy + Add + Mul + Zero + Debug + 'static,
    K: MatMatMulKer<TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "A:{}, B:{} C:{} (m:{}, k:{}, n:{}) ({} {}x{})",
            self.a_storage,
            self.b_storage,
            self.c_storage,
            self.m,
            self.k,
            self.n,
            K::name(),
            K::mr(),
            K::nr()
        )
    }
}
