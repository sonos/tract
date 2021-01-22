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
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> MatrixStoreSpec;
    unsafe fn b_vec_from_data_and_stride(&self, stride: isize) -> MatrixStoreSpec;
    unsafe fn b_vec_from_data(&self) -> MatrixStoreSpec;

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
        MatMatMulImpl { m, k, n, phantom: PhantomData }
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

    unsafe fn a_packed(&self) -> MatrixStoreSpec {
        MatrixStoreSpec::Packed { panel_len: (self.k * K::mr()) }
    }

    unsafe fn b_packed(&self) -> MatrixStoreSpec {
        MatrixStoreSpec::Packed { panel_len: (self.k * K::nr()) }
    }

    unsafe fn b_from_data_and_offsets(
        &self,
        rows_offsets: &[isize],
        cols_offsets: &[isize],
    ) -> MatrixStoreSpec {
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
        MatrixStoreSpec::OffsetsAndPtrs { col_byte_offsets, row_byte_offsets, nr: K::nr() }
    }

    unsafe fn b_vec_from_data_and_stride(&self, stride: isize) -> MatrixStoreSpec {
        MatrixStoreSpec::VecStride {
            byte_stride: stride * std::mem::size_of::<TB>() as isize,
            mr: K::mr(),
            nr: K::nr(),
        }
    }

    unsafe fn b_vec_from_data(&self) -> MatrixStoreSpec {
        self.b_vec_from_data_and_stride(1)
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

    unsafe fn run(
        &self,
        a: &MatrixStore,
        b: &MatrixStore,
        c: &mut MatrixStore,
        non_linear: &[FusedSpec],
    ) -> anyhow::Result<()> {
        let mr = K::mr();
        let nr = K::nr();
        anyhow::ensure!(a.tensor.datum_type() == TA::datum_type());
        anyhow::ensure!(b.tensor.datum_type() == TB::datum_type());
        anyhow::ensure!(c.tensor.datum_type() == TC::datum_type());
        let prefetch = crate::ops().prefetch.as_ref();
        let m = self.m;
        let n = self.n;
        let mut scratch = ScratchSpaceFusedNonLinear::default();
        let mut tmpc_buffer = Tensor::uninitialized::<TC>(&[nr, mr])?;
        let tmp_c_storage = self.c_view_with_axis(1, 0);
        let tmpc_view = tmpc_buffer.view_mut();
        let tmpc = tmp_c_storage.wrap(&tmpc_view);

        let ref linear = LinearSpec::k(self.k);
        let non_linear = non_linear.to_vec();
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
        // FIXME prefetch a are a bit weird
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
                let ref direct_c = c.tile_c(ia, ib, mr, nr);
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
            if let MatrixStoreSpec::VecStride { .. } = c.spec {
                if let PanelStore::Packed { ptr } = a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, n / nr, n % nr);
                match b {
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, ia, n / nr);
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
                if let PanelStore::Packed { ptr } = a {
                    prefetch(*ptr as *const u8, 512);
                }
                let ref b = b.panel_b(nr, n / nr, n % nr);
                match b {
                    PanelStore::Packed { ptr } => prefetch(*ptr as *const u8, 512),
                    PanelStore::VecStride { ptr, .. } => prefetch(*ptr as *const u8, 128),
                    _ => (),
                }
                let ref tmp_tile_c = tmpc.tile_c(0, 0, mr, nr);
                let non_linear = scratch.for_tile::<TA, TB, TC, K>(&non_linear, ia, n / nr);
                let err = K::kernel(&MatMatMulKerSpec {
                    a: a as _,
                    b: b as _,
                    c: tmp_tile_c as _,
                    linear,
                    non_linear,
                });
                debug_assert_eq!(err, 0, "Kernel return error {}", err);
                c.set_from_tile::<TC>(ia, n / nr, mr, n % nr, tmpc.tensor, mr, nr);
            }
        }
        if m % mr != 0 {
            let ref panel_a = a.panel_a(m / mr);
            let ref tmp_tile_c = tmpc.tile_c(0, 0, mr, nr);
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
                c.set_from_tile::<TC>(m / mr, ib, m % mr, nr, tmpc.tensor, mr, nr);
            }
            if n % nr != 0 {
                // FIXME: can we write straight to C if n == 1 ?
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
                c.set_from_tile::<TC>(m / mr, n / nr, m % mr, n % nr, tmpc.tensor, mr, nr);
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
