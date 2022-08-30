pub mod lir_unary;
pub mod mir;
pub mod mir_quant;
pub mod mir_quant_unary;
pub mod mir_unary;
pub mod pack;

use crate::internal::*;
use tract_itertools::Itertools;
use tract_linalg::mmm::FusedSpec;
use tract_ndarray::prelude::*;

pub use self::mir::MatMul;
pub use self::mir_quant::{MatMulQParams, QMatMul};
pub use self::mir_unary::MatMulUnary;
use self::pack::MatMatMulPack;

#[derive(PartialEq, Clone, Debug, Copy, Hash)]
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

    pub fn change_axis_from_b(
        &self,
        change: &AxisOp,
    ) -> TractResult<(MatMulAxes, AxisOp, AxisOp, AxisOp)> {
        match change {
            AxisOp::Rm(ix) => {
                ensure!(*ix != self.b_k && *ix != self.b_n);
                let index_as_untouched_axis =
                    ix - (self.b_k < *ix) as usize - (self.b_n < *ix) as usize;
                self.remove_untouched_axis(index_as_untouched_axis)
            }
            AxisOp::Add(in_b) => {
                if *in_b == self.b_n + 1 {
                    self.insert_untouched_axis(self.a_m + 1, *in_b, self.c_n + 1)
                } else if *in_b == self.b_k + 1 {
                    self.insert_untouched_axis(self.a_k + 1, *in_b, self.c_m + 1)
                } else {
                    let ix = in_b - (self.b_k < *in_b) as usize - (self.b_n < *in_b) as usize;
                    let in_a = ix + (ix > self.a_m) as usize + (ix > self.a_k) as usize;
                    let in_c = ix + (ix > self.c_m) as usize + (ix > self.c_n) as usize;
                    self.insert_untouched_axis(in_a, *in_b, in_c)
                }
            }
            _ => bail!("Invalid change"),
        }
    }

    pub fn change_axis_from_c(
        &self,
        change: &AxisOp,
    ) -> TractResult<(MatMulAxes, AxisOp, AxisOp, AxisOp)> {
        match change {
            AxisOp::Rm(ix) => {
                ensure!(*ix != self.c_m && *ix != self.c_n);
                let index_as_untouched_axis =
                    ix - (self.c_m < *ix) as usize - (self.c_n < *ix) as usize;
                self.remove_untouched_axis(index_as_untouched_axis)
            }
            /*
            AxisOp::Add(ix) => {
            let index_as_untouched_axis =
            ix - (self.c_m < *ix) as usize - (self.c_n < *ix) as usize;
            self.insert_untouched_axis(index_as_untouched_axis)
            }
            */
            _ => bail!("Invalid change"),
        }
    }

    fn remove_untouched_axis(
        &self,
        ix: usize,
    ) -> TractResult<(MatMulAxes, AxisOp, AxisOp, AxisOp)> {
        let axes = MatMulAxes {
            a_m: self.a_m - (ix < self.a_m) as usize,
            a_k: self.a_k - (ix < self.a_k) as usize,
            b_k: self.b_k - (ix < self.b_k) as usize,
            b_n: self.b_n - (ix < self.b_n) as usize,
            c_m: self.c_m - (ix < self.c_m) as usize,
            c_n: self.c_n - (ix < self.c_n) as usize,
        };
        let in_a = ix + (ix > self.a_m) as usize + (ix > self.a_k) as usize;
        let in_b = ix + (ix > self.b_k) as usize + (ix > self.b_n) as usize;
        let in_c = ix + (ix > self.c_m) as usize + (ix > self.c_n) as usize;
        Ok((axes, AxisOp::Rm(in_a), AxisOp::Rm(in_b), AxisOp::Rm(in_c)))
    }

    fn insert_untouched_axis(
        &self,
        in_a: usize,
        in_b: usize,
        in_c: usize,
    ) -> TractResult<(MatMulAxes, AxisOp, AxisOp, AxisOp)> {
        let axes = MatMulAxes {
            a_m: self.a_m + (in_a <= self.a_m) as usize,
            a_k: self.a_k + (in_a <= self.a_k) as usize,
            b_k: self.b_k + (in_b <= self.b_k) as usize,
            b_n: self.b_n + (in_b <= self.b_n) as usize,
            c_m: self.c_m + (in_c <= self.c_m) as usize,
            c_n: self.c_n + (in_c <= self.c_n) as usize,
        };
        Ok((axes, AxisOp::Add(in_a), AxisOp::Add(in_b), AxisOp::Add(in_c)))
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
        let rank = a.rank();
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
        let mut c = Tensor::uninitialized_dt(c_dt, &c_shape)?;

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

        let mut c_bc_shape = c_shape.clone();
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

#[cfg(test)]
mod change_axis_test {
    use proptest::prelude::*;
    use proptest::strategy::{BoxedStrategy, Strategy};

    use super::*;

    fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
        let len = shape.iter().product::<usize>();
        let shape = shape.to_vec();
        proptest::collection::vec(any::<i8>().prop_map(|i| i as f32), len..=len)
            .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap().into_tensor())
            .boxed()
    }

    fn strat_for_b_axes(rank: usize) -> impl Strategy<Value = (usize, usize)> {
        assert!(rank >= 2);
        (0..rank)
            .prop_flat_map(move |bn| (Just(bn), (0..rank - 1)))
            .prop_map(|(bn, raw_bk)| (bn, raw_bk + (raw_bk >= bn) as usize))
    }

    #[derive(Clone, Debug)]
    struct ChangeAxisMatmulProblem {
        input: Tensor,
        change: AxisOp,
        matmul: MatMulUnary,
    }

    impl Arbitrary for ChangeAxisMatmulProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<ChangeAxisMatmulProblem>;
        fn arbitrary_with(_parameters: Self::Parameters) -> Self::Strategy {
            proptest::collection::vec(1..10usize, 2..5)
                .prop_flat_map(|shape_input| {
                    (tensor(&shape_input), AxisOp::arbitrary_with(shape_input.into()))
                })
                .prop_flat_map(|(input, change)| {
                    let mut matmul_input_shape: TVec<usize> = input.shape().into();
                    change.change_shape_array(&mut matmul_input_shape, false).unwrap();
                    (Just(input), Just(change), Just(matmul_input_shape))
                })
                .prop_filter("rank must be >= 2", |(_, _, matmul_input_shape)| {
                    matmul_input_shape.len() >= 2
                })
                .prop_flat_map(|(input, change, matmul_input_shape)| {
                    (
                        Just(input),
                        Just(change),
                        Just(matmul_input_shape.clone()),
                        strat_for_b_axes(matmul_input_shape.len()),
                        1usize..=6,
                    )
                })
                .prop_flat_map(|(input, change, matmul_input_shape, (b_k, b_n), m)| {
                    let k = matmul_input_shape[b_k];
                    (Just((input, change, matmul_input_shape, b_k, b_n)), tensor(&[m, k]))
                })
                .prop_map(|((input, change, matmul_input_shape, b_k, b_n), a)| {
                    let mut axes = MatMulAxes::default_for_rank(matmul_input_shape.len());
                    axes.b_n = b_n;
                    axes.b_k = b_k;
                    ChangeAxisMatmulProblem {
                        input,
                        change,
                        matmul: MatMulUnary {
                            a: a.broadcast_into_rank(matmul_input_shape.len())
                                .unwrap()
                                .into_arc_tensor(),
                            axes,
                        },
                    }
                })
                .boxed()
        }
    }

    impl ChangeAxisMatmulProblem {
        fn model(&self) -> TypedModel {
            let mut model = TypedModel::default();
            let source = model.add_source("source", f32::fact(self.input.shape())).unwrap();
            let changed = model.wire_node("change", self.change.clone(), &[source]).unwrap();
            let output = model.wire_node("mm", self.matmul.clone(), &changed).unwrap();
            model.set_output_outlets(&output).unwrap();
            model
        }
        fn reference(&self) -> Tensor {
            let model = self.model();
            let mut outputs =
                model.into_runnable().unwrap().run(tvec!(self.input.clone())).unwrap();
            outputs.remove(0).into_tensor()
        }

        fn swapped(&self) -> Option<Tensor> {
            let model = self.model();
            self.matmul
                .change_axes(&model, &model.nodes[2], InOut::In(0), &self.change.recip())
                .unwrap()
                .map(|changed_mm| {
                    let mut model = TypedModel::default();
                    let source = model.add_source("source", f32::fact(self.input.shape())).unwrap();
                    let mul = model
                        .wire_node(
                            "mm",
                            changed_mm
                                .substitute_op
                                .clone()
                                .unwrap_or(Box::new(self.matmul.clone())),
                            &[source],
                        )
                        .unwrap();
                    let change_after = changed_mm
                        .wire_changes
                        .iter()
                        .find(|(io, _change)| *io == InOut::Out(0))
                        .map(|(_io, change)| change)
                        .unwrap();
                    let changed = model.wire_node("change", change_after.clone(), &mul).unwrap();
                    model.set_output_outlets(&changed).unwrap();
                    let mut outputs =
                        model.into_runnable().unwrap().run(tvec!(self.input.clone())).unwrap();
                    outputs.remove(0).into_tensor()
                })
        }
    }

    proptest! {
        #[test]
        fn proptest_validity(pb in any::<ChangeAxisMatmulProblem>()) {
            pb.reference();
        }

        #[test]
        fn proptest_equals(pb in any::<ChangeAxisMatmulProblem>()) {
            if let Some(swapped) = pb.swapped() {
                prop_assert_eq!(swapped, pb.reference());
            }
        }
    }

    #[test]
    fn rm0() {
        let pb = ChangeAxisMatmulProblem {
            input: Tensor::zero::<f32>(&[3,1,1]).unwrap(),
            change: AxisOp::Rm(1),
            matmul: MatMulUnary {
                a: Tensor::zero::<f32>(&[1,3]).unwrap().into_arc_tensor(),
                axes: MatMulAxes { a_m: 0, a_k: 1, b_k: 0, b_n: 1, c_m: 0, c_n: 1 }
            }
        };
        assert_eq!(pb.swapped().unwrap(), pb.reference());
    }
}
