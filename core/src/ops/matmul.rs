pub mod lir_unary;
pub mod mir;
pub mod mir_quant;
pub mod mir_quant_unary;
pub mod mir_unary;
pub mod pack;

#[cfg(test)]
mod change_axis_test;

use crate::internal::*;
use tract_itertools::Itertools;
use tract_linalg::mmm::FusedSpec;
use tract_ndarray::prelude::*;

pub use self::mir::MatMul;
pub use self::mir_quant::{MatMulQParams, QMatMul};
pub use self::mir_unary::MatMulUnary;
use self::pack::MatMatMulPack;

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub struct MatMulAxes {
    pub a_m: usize,
    pub a_k: usize,
    pub b_k: usize,
    pub b_n: usize,
    pub c_m: usize,
    pub c_n: usize,
}

impl Default for MatMulAxes {
    fn default() -> Self {
        Self::default_for_rank(2)
    }
}

impl MatMulAxes {
    pub fn default_for_rank(rank: usize) -> Self {
        Self::default_for_ranks(rank, rank, rank)
    }

    pub fn default_for_ranks(a: usize, b: usize, c: usize) -> Self {
        MatMulAxes { a_m: a - 2, a_k: a - 1, b_k: b - 2, b_n: b - 1, c_m: c - 2, c_n: c - 1 }
    }

    pub fn transposing_a(self) -> Self {
        MatMulAxes { a_m: self.a_k, a_k: self.a_m, ..self }
    }

    pub fn transposing_b(self) -> Self {
        MatMulAxes { b_n: self.b_k, b_k: self.b_n, ..self }
    }

    pub fn transposing_c(self) -> Self {
        MatMulAxes { c_n: self.c_m, c_m: self.c_n, ..self }
    }

    pub fn transposing(self, a: bool, b: bool, c: bool) -> Self {
        let mut it = self;
        if a {
            it = it.transposing_a();
        }
        if b {
            it = it.transposing_b();
        }
        if c {
            it = it.transposing_c();
        }
        it
    }

    // return matching axis index in a and c
    fn follow_axis_from_b(&self, in_b: usize) -> (usize, usize) {
        let ix = in_b - (self.b_k < in_b) as usize - (self.b_n < in_b) as usize;
        let in_a = (0..).filter(|&i| i != self.a_m && i != self.a_k).nth(ix).unwrap();
        let in_c = (0..).filter(|&i| i != self.c_m && i != self.c_n).nth(ix).unwrap();
        (in_a, in_c)
    }

    // return matching axis index in a and b
    fn follow_axis_from_c(&self, in_c: usize) -> (usize, usize) {
        let ix = in_c - (self.c_m < in_c) as usize - (self.c_n < in_c) as usize;
        let in_a = (0..).filter(|&i| i != self.a_m && i != self.a_k).nth(ix).unwrap();
        let in_b = (0..).filter(|&i| i != self.b_k && i != self.b_n).nth(ix).unwrap();
        (in_a, in_b)
    }

    #[allow(clippy::type_complexity)]
    pub fn change_axis_from_b(
        &self,
        change: &AxisOp,
        bc_rank: usize,
    ) -> TractResult<(MatMulAxes, Option<AxisOp>, Option<AxisOp>, Option<AxisOp>)> {
        match change {
            AxisOp::Rm(in_b) => {
                // adhoc: change n_axis as they do not matter
                // FIXME: remove me if matmul becomes einsum
                if *in_b == self.b_n {
                    if let Some(axis) = (0..bc_rank)
                        .filter(|axis| *axis != self.b_n && *axis != self.b_k)
                        .rev()
                        .next()
                    {
                        let (_, new_c_n) = self.follow_axis_from_b(axis);
                        let new_axes = Self { b_n: axis, c_n: new_c_n, ..*self };
                        let (in_a, in_c) = new_axes.follow_axis_from_b(*in_b);
                        return new_axes.remove_untouched_axis(in_a, *in_b, in_c);
                    }
                }
                ensure!(*in_b != self.b_k && *in_b != self.b_n);
                let (in_a, in_c) = self.follow_axis_from_b(*in_b);
                self.remove_untouched_axis(in_a, *in_b, in_c)
            }
            AxisOp::Add(in_b) => {
                // adhoc to try to keep inner axis addition inner
                // FIXME: remove me if matmul becomes einsum
                let (in_a, in_c) = if *in_b == self.b_k + 1 {
                    (0, self.c_m + 1)
                } else if *in_b == self.b_n + 1 {
                    (0, self.c_n + 1)
                } else {
                    self.follow_axis_from_b(*in_b)
                };
                self.insert_untouched_axis(in_a, *in_b, in_c)
            }
            AxisOp::Reshape(in_b, before, after) => {
                ensure!(self.b_n < *in_b || self.b_n >= *in_b + before.len());
                ensure!(self.b_k < *in_b || self.b_k >= *in_b + before.len());
                let (in_a, in_c) = self.follow_axis_from_b(*in_b);
                self.reshape_untouched_axes(in_a, *in_b, in_c, before, after)
            }
            AxisOp::Move(from, to) => {
                // only deal with spectific cases for now:
                // movement inside prefix, unaffecting k and n -> propagate to c
                if *from.max(to) < self.b_k.min(self.b_n) && *from.max(to) < self.c_n.min(self.c_m)
                {
                    Ok((
                        *self,
                        None,
                        Some(AxisOp::Move(*from, *to)),
                        Some(AxisOp::Move(*from, *to)),
                    ))
                    // moving k or n to a different position : absorb it
                } else if *from == self.b_k || *from == self.b_n {
                    let b_n = change.transform_axis(self.b_n).unwrap();
                    let b_k = change.transform_axis(self.b_k).unwrap();
                    Ok((
                        MatMulAxes { b_n, b_k, ..*self },
                        None,
                        Some(AxisOp::Move(*from, *to)),
                        None,
                    ))
                } else {
                    bail!("Unsupported move");
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn change_axis_from_c(
        &self,
        change: &AxisOp,
        bc_rank: usize,
    ) -> TractResult<(MatMulAxes, Option<AxisOp>, Option<AxisOp>, Option<AxisOp>)> {
        match change {
            AxisOp::Rm(in_c) => {
                // adhoc: change n_axis as they do not matter
                // FIXME: remove me if matmul becomes einsum
                if *in_c == self.c_n {
                    if let Some(axis) = (0..bc_rank)
                        .filter(|axis| *axis != self.c_n && *axis != self.c_m)
                        .rev()
                        .next()
                    {
                        let (_, new_b_n) = self.follow_axis_from_c(axis);
                        let new_axes = Self { c_n: axis, b_n: new_b_n, ..*self };
                        let (in_a, in_b) = new_axes.follow_axis_from_c(*in_c);
                        return new_axes.remove_untouched_axis(in_a, in_b, *in_c);
                    }
                }
                ensure!(*in_c != self.c_m && *in_c != self.c_n);
                let (in_a, in_b) = self.follow_axis_from_c(*in_c);
                self.remove_untouched_axis(in_a, in_b, *in_c)
            }
            AxisOp::Add(in_c) => {
                let (in_a, in_b) = self.follow_axis_from_c(*in_c);
                self.insert_untouched_axis(in_a, in_b, *in_c)
            }
            AxisOp::Reshape(in_c, before, after) => {
                ensure!(self.c_n < *in_c || self.c_n >= *in_c + before.len());
                ensure!(self.c_m < *in_c || self.c_m >= *in_c + before.len());
                let (in_a, in_b) = self.follow_axis_from_c(*in_c);
                self.reshape_untouched_axes(in_a, in_b, *in_c, before, after)
            }
            AxisOp::Move(from, to) => {
                // only deal with spectific cases for now:
                // movement inside prefix, unaffecting m and n -> propagate to b
                if *from.max(to) < self.b_k.min(self.b_n) && *from.max(to) < self.c_n.min(self.c_m)
                {
                    Ok((
                        *self,
                        None,
                        Some(AxisOp::Move(*from, *to)),
                        Some(AxisOp::Move(*from, *to)),
                    ))
                    // moving m or n to a different position : absorb it
                } else if *from == self.c_m || *from == self.c_n {
                    let c_n = change.transform_axis(self.c_n).unwrap();
                    let c_m = change.transform_axis(self.c_m).unwrap();
                    Ok((
                        MatMulAxes { c_n, c_m, ..*self },
                        None,
                        None,
                        Some(AxisOp::Move(*from, *to)),
                    ))
                } else {
                    bail!("Unsupported move");
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn remove_untouched_axis(
        &self,
        in_a: usize,
        in_b: usize,
        in_c: usize,
    ) -> TractResult<(MatMulAxes, Option<AxisOp>, Option<AxisOp>, Option<AxisOp>)> {
        let axes = MatMulAxes {
            a_m: self.a_m - (in_a < self.a_m) as usize,
            a_k: self.a_k - (in_a < self.a_k) as usize,
            b_k: self.b_k - (in_b < self.b_k) as usize,
            b_n: self.b_n - (in_b < self.b_n) as usize,
            c_m: self.c_m - (in_c < self.c_m) as usize,
            c_n: self.c_n - (in_c < self.c_n) as usize,
        };
        Ok((axes, Some(AxisOp::Rm(in_a)), Some(AxisOp::Rm(in_b)), Some(AxisOp::Rm(in_c))))
    }

    #[allow(clippy::type_complexity)]
    fn insert_untouched_axis(
        &self,
        in_a: usize,
        in_b: usize,
        in_c: usize,
    ) -> TractResult<(MatMulAxes, Option<AxisOp>, Option<AxisOp>, Option<AxisOp>)> {
        let axes = MatMulAxes {
            a_m: self.a_m + (in_a <= self.a_m) as usize,
            a_k: self.a_k + (in_a <= self.a_k) as usize,
            b_k: self.b_k + (in_b <= self.b_k) as usize,
            b_n: self.b_n + (in_b <= self.b_n) as usize,
            c_m: self.c_m + (in_c <= self.c_m) as usize,
            c_n: self.c_n + (in_c <= self.c_n) as usize,
        };
        Ok((axes, Some(AxisOp::Add(in_a)), Some(AxisOp::Add(in_b)), Some(AxisOp::Add(in_c))))
    }

    #[allow(clippy::type_complexity)]
    fn reshape_untouched_axes(
        &self,
        in_a: usize,
        in_b: usize,
        in_c: usize,
        before: &[TDim],
        after: &[TDim],
    ) -> TractResult<(MatMulAxes, Option<AxisOp>, Option<AxisOp>, Option<AxisOp>)> {
        let diff = after.len() as isize - before.len() as isize;
        let axes = MatMulAxes {
            a_m: (self.a_m as isize + ((in_a <= self.a_m) as usize as isize * diff)) as usize,
            a_k: (self.a_k as isize + ((in_a <= self.a_k) as usize as isize * diff)) as usize,
            b_k: (self.b_k as isize + ((in_b <= self.b_k) as usize as isize * diff)) as usize,
            b_n: (self.b_n as isize + ((in_b <= self.b_n) as usize as isize * diff)) as usize,
            c_m: (self.c_m as isize + ((in_c <= self.c_m) as usize as isize * diff)) as usize,
            c_n: (self.c_n as isize + ((in_c <= self.c_n) as usize as isize * diff)) as usize,
        };
        Ok((
            axes,
            Some(AxisOp::Reshape(
                in_a,
                std::iter::repeat_with(|| 1.to_dim()).take(before.len()).collect(),
                std::iter::repeat_with(|| 1.to_dim()).take(after.len()).collect(),
            )),
            Some(AxisOp::Reshape(in_b, before.into(), after.into())),
            Some(AxisOp::Reshape(in_c, before.into(), after.into())),
        ))
    }

    pub fn to_array(&self) -> [usize; 6] {
        [self.a_m, self.a_k, self.b_k, self.b_n, self.c_m, self.c_n]
    }

    pub fn from_array(array: &[usize]) -> TractResult<Self> {
        anyhow::ensure!(
            array.len() == 6,
            "MatMulAxes requires exactly six axis numbers, got {:?}",
            array
        );
        Ok(MatMulAxes {
            a_m: array[0],
            a_k: array[1],
            b_k: array[2],
            b_n: array[3],
            c_m: array[4],
            c_n: array[5],
        })
    }
}

pub fn compute_shape<D: DimLike>(
    ashape: &[D],
    bshape: &[D],
    axes: MatMulAxes,
) -> TractResult<(D, D, D, TVec<D>)> {
    let a_shape_bc: TVec<D> = ashape
        .iter()
        .enumerate()
        .filter_map(
            |(ix, dim)| if ix != axes.a_m && ix != axes.a_k { Some(dim.clone()) } else { None },
        )
        .collect();
    let b_shape_bc = bshape
        .iter()
        .enumerate()
        .filter_map(
            |(ix, dim)| if ix != axes.b_k && ix != axes.b_n { Some(dim.clone()) } else { None },
        )
        .collect();
    let mut c_shape = crate::broadcast::multi_broadcast(&[a_shape_bc, b_shape_bc])
        .ok_or_else(|| format_err!("Could not broadcast"))?;
    // dbg!(&c_shape);
    let (m, ka) = (ashape[axes.a_m].clone(), ashape[axes.a_k].clone());
    let (kb, n) = (bshape[axes.b_k].clone(), bshape[axes.b_n].clone());
    if ka != kb {
        bail!(
            "Inconsistent matmul: a: {} b: {}, axes: am:{} ak:{} bk:{} bn:{} cm:{} cn:{}",
            ashape.iter().join(","),
            bshape.iter().join(","),
            axes.a_m,
            axes.a_k,
            axes.b_k,
            axes.b_n,
            axes.c_m,
            axes.c_n
        );
    }
    if axes.c_m < axes.c_n {
        c_shape.insert(axes.c_m, m.clone());
        c_shape.insert(axes.c_n, n.clone());
    } else {
        c_shape.insert(axes.c_n, n.clone());
        c_shape.insert(axes.c_m, m.clone());
    }
    Ok((m, ka, n, c_shape))
}

pub fn output_type(input: DatumType) -> DatumType {
    if input.is_float() {
        input
    } else {
        i32::datum_type()
    }
}

pub(super) fn eval(a: &Tensor, b: &Tensor, axes: MatMulAxes) -> TractResult<Tensor> {
    unsafe {
        let (m, k, n, c_shape) = compute_shape(a.shape(), b.shape(), axes)?;
        let c_dt = output_type(a.datum_type());
        let mm = tract_linalg::ops()
            .mmm(a.datum_type(), b.datum_type(), c_dt, Some(m), Some(k), Some(n))
            .with_context(|| {
                format!(
                    "No matrix multiplier for {:?}x{:?} to {:?}",
                    a.datum_type(),
                    b.datum_type(),
                    c_dt
                )
            })?;
        let c_storage = mm.c_view(axes.c_m, axes.c_n);
        let c = Tensor::uninitialized_dt(c_dt, &c_shape)?;

        let a_pack = mm.a_pack();
        let b_pack = mm.b_pack();

        let mut packed_a = Tensor::uninitialized_aligned_dt(
            a.datum_type(),
            &[a_pack.len(k, m)],
            a_pack.alignment(),
        )?;
        let mut packed_b = Tensor::uninitialized_aligned_dt(
            b.datum_type(),
            &[b_pack.len(k, n)],
            b_pack.alignment(),
        )?;

        // FIXME: what does it look with putting m and n in C at 1 instead of removing ?

        let mut a_bc_shape: TVec<usize> = a.shape().into();
        a_bc_shape.remove(axes.a_m.max(axes.a_k));
        a_bc_shape.remove(axes.a_m.min(axes.a_k));

        let mut b_bc_shape: TVec<usize> = b.shape().into();
        b_bc_shape.remove(axes.b_n.max(axes.b_k));
        b_bc_shape.remove(axes.b_n.min(axes.b_k));

        let mut a_strides: TVec<isize> = a.strides().into();
        a_strides.remove(axes.a_m.max(axes.a_k));
        a_strides.remove(axes.a_m.min(axes.a_k));

        let mut b_strides: TVec<isize> = b.strides().into();
        b_strides.remove(axes.b_n.max(axes.b_k));
        b_strides.remove(axes.b_n.min(axes.b_k));

        let mut c_bc_shape = c_shape;
        c_bc_shape.remove(axes.c_m.max(axes.c_n));
        c_bc_shape.remove(axes.c_m.min(axes.c_n));

        let mut c_strides: TVec<isize> = c.strides().into();
        c_strides.remove(axes.c_m.max(axes.c_n));
        c_strides.remove(axes.c_m.min(axes.c_n));

        for prefix in tract_ndarray::indices(&*c_bc_shape).into_iter() {
            let mut a_offset = 0;
            let mut b_offset = 0;
            let mut c_offset = 0;
            for (axis, &dim) in prefix.slice().iter().enumerate() {
                if a_bc_shape[axis] > 1 {
                    a_offset += a_strides[axis] * dim as isize * a.datum_type().size_of() as isize;
                }
                if b_bc_shape[axis] > 1 {
                    b_offset += b_strides[axis] * dim as isize * b.datum_type().size_of() as isize;
                }
                c_offset += c_strides[axis] * dim as isize * c_dt.size_of() as isize;
            }
            a_pack.pack(
                packed_a.view_mut(),
                TensorView::from_bytes(a, a_offset, a.shape(), a.strides()),
                axes.a_k,
                axes.a_m,
            );
            b_pack.pack(
                packed_b.view_mut(),
                TensorView::from_bytes(b, b_offset, b.shape(), b.strides()),
                axes.b_k,
                axes.b_n,
            );
            mm.run(
                m,
                n,
                &[
                    FusedSpec::AddMatMul {
                        a: mm.a_packed(a.datum_type().size_of(), k).wrap(&packed_a.view()),
                        b: mm.b_packed(b.datum_type().size_of(), k).wrap(&packed_b.view())?,
                        k,
                    },
                    FusedSpec::Store(c_storage.wrap(&TensorView::from_bytes(
                        &c,
                        c_offset,
                        c.shape(),
                        c.strides(),
                    ))),
                ],
            )?;
        }
        Ok(c)
    }
}

pub(super) fn cost<A: DimLike + Clone, B: DimLike + Clone>(
    a: &[A],
    b: &[B],
    dt: DatumType,
    axes: MatMulAxes,
) -> TractResult<TVec<(Cost, TDim)>> {
    let (m, k, n, c_shape) = compute_shape(
        &a.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        &b.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        axes,
    )?;
    let mul = c_shape.iter().rev().skip(2).cloned().product();
    Ok(tvec!((Cost::FMA(dt), [mul, m.to_dim(), k.to_dim(), n.to_dim()].iter().product())))
}
