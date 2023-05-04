use tract_ndarray::Ix2;
use tract_num_traits::One;

use super::EinSum;
use crate::internal::*;
use super::codegen::*;

pub fn decompose(op: &EinSum, model: &TypedModel, node: &TypedNode) -> TractResult<TypedModel> {
    let mut substitute = TypedModel::default();
    let inputs = node
        .inputs
        .iter()
        .enumerate()
        .map(|(ix, input)| {
            substitute.add_source(
                format!("adhoc_source_{ix}"),
                model.outlet_fact(*input)?.without_value(),
            )
        })
        .collect::<TractResult<Vec<_>>>()?;
    let outputs = substitute.wire_node(&node.name, op.clone(), &inputs)?;
    substitute.set_output_outlets(&outputs)?;
    decompose_einsums_in_place(&mut substitute)?;
    Ok(substitute)
}

pub fn decompose_einsums_in_place(model: &mut TypedModel) -> TractResult<()> {
    'top: loop {
        dbg!(&model);
        for n in model.eval_order()? {
            let node = &model.nodes[n];
            if let Some(einsum) = node.op_as::<EinSum>() {
                if let Some(patch) = step(einsum, model, node)? {
                    patch.apply(model)?;
                    model.compact().unwrap();
                    continue 'top;
                }
            }
        }
        return Ok(());
    }
}

pub fn step(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    assert!(
        (node.inputs.len() == 2 && op.q_params.is_none())
            || (node.inputs.len() == 9 && op.q_params.is_some())
    );
    let (m_axis, k_axis, n_axis) = match ensure_mkn_axes(op, model, node)? {
        AxesOrPatch::Axes(m, k, n) => (m, k, n),
        AxesOrPatch::Patch(p) => return Ok(Some(p))
    };
    let a = model.outlet_fact(node.inputs[0])?;
    let b = model.outlet_fact(node.inputs[1])?;
    let c = &node.outputs[0].fact;
    assert_eq!(a.rank(), b.rank());
    assert_eq!(a.rank(), c.rank());
    let rank = a.rank();
    let a_m = m_axis.inputs[0][0];
    let a_k = k_axis.inputs[0][0];
    let b_k = k_axis.inputs[1][0];
    let b_n = n_axis.inputs[1][0];
    let c_m = m_axis.outputs[0][0];
    let c_n = n_axis.outputs[0][0];
    assert!(a_m >= rank - 2);
    assert!(a_k >= rank - 2);
    assert!(b_k >= rank - 2);
    assert!(b_n >= rank - 2);
    assert!(c_m >= rank - 2);
    assert!(c_n >= rank - 2);
    let transpose_a = a_m > a_k;
    let transpose_b = b_k > b_n;
    let transpose_c = c_m > c_n;
    Ok(Some(TypedModelPatch::replace_single_op(
        model,
        node,
        &node.inputs,
        BasicMatMul { transpose_a, transpose_b, transpose_c },
    )?))
}

#[derive(Clone, Debug)]
pub struct BasicMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_c: bool,
}

impl BasicMatMul {
    fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let mut output: TVec<D> = (0..a.len() - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[a.len() - 2 + self.transpose_a as usize].clone());
        output.push(b[b.len() - 2 + !self.transpose_b as usize].clone());
        if self.transpose_c {
            let len = output.len();
            output.swap(len - 2, len - 1);
        }
        output
    }

    fn eval_t<T: Datum + tract_ndarray::LinalgScalar>(
        &self,
        c: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
    ) -> TractResult<()> {
        use crate::ndarray::Dimension;
        let a = a.to_array_view::<T>()?;
        let b = b.to_array_view::<T>()?;
        let mut c = c.to_array_view_mut::<T>()?;
        for prefix in tract_ndarray::indices(&c.shape()[..c.ndim() - 2]) {
            let mut a = a.view();
            let mut b = b.view();
            let mut c = c.view_mut();
            for d in prefix.slice().iter() {
                a.index_axis_inplace(tract_ndarray::Axis(0), *d.max(&a.shape()[0]));
                b.index_axis_inplace(tract_ndarray::Axis(0), *d.max(&b.shape()[0]));
                c.index_axis_inplace(tract_ndarray::Axis(0), *d);
            }
            let a = a.into_dimensionality::<Ix2>().unwrap();
            let b = b.into_dimensionality::<Ix2>().unwrap();
            let mut c = c.into_dimensionality::<Ix2>().unwrap();
            let a = if self.transpose_a { a.t() } else { a };
            let b = if self.transpose_b { b.t() } else { b };
            if self.transpose_c {
                c.assign(&b.t().dot(&a.t()))
            } else {
                c.assign(&a.dot(&b))
            }
        }
        Ok(())
    }
}

impl Op for BasicMatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for BasicMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        let mut c = Tensor::zero_dt(a.datum_type(), &self.output_shape(a.shape(), b.shape()))?;
        dispatch_numbers!(Self::eval_t(a.datum_type())(self, &mut c, &a, &b)).unwrap();
        Ok(tvec!(c.into_tvalue()))
    }
}

impl TypedOp for BasicMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let a = inputs[0];
        let b = inputs[1];
        ensure!(a.rank() == b.rank());
        ensure!(a.rank() >= 2);
        ensure!(
            a.shape[a.rank() - 2 + !self.transpose_a as usize]
                == b.shape[b.rank() - 2 + self.transpose_b as usize]
        );
        Ok(tvec!(a.datum_type.fact(self.output_shape(&a.shape, &b.shape))))
    }

    as_op!();
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::{TestCaseResult, TestRunner};
    use tract_data::itertools::Itertools;

    pub fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
        let shape = shape.to_vec();
        let len = shape.iter().product::<usize>();
        vec((-10i8..=10i8).prop_map(|i| i as f32), len..=len)
            .prop_map(move |vec| tensor1(&vec).into_shape(&shape).unwrap())
            .boxed()
    }

    fn full_shapes(e: &AxesMapping) -> BoxedStrategy<(Vec<usize>, Vec<usize>)> {
        let e = e.clone();
        let inputs_axes = e
            .iter_all_axes()
            .filter(|axis| axis.inputs[0].len() + axis.inputs[1].len() > 0)
            .cloned()
            .collect_vec();
        let dims = vec![2usize..6; inputs_axes.len()];
        dims.prop_map(move |dims| {
            let a: Vec<usize> = e
                .axes(InOut::In(0))
                .map(|a| dims[inputs_axes.iter().position(|b| a == b).unwrap()])
                .collect_vec();
            let b: Vec<usize> = e
                .axes(InOut::In(1))
                .map(|a| dims[inputs_axes.iter().position(|b| a == b).unwrap()])
                .collect_vec();
            (a, b)
        })
        .boxed()
    }

    fn test_expr(expr: &str) {
        let expr = expr.to_string();
        let mut runner = TestRunner::default();
        let axes: AxesMapping = expr.parse().unwrap();
        fn is_k(axes: &AxesMapping, input: usize, position: usize) -> bool {
            let axis = axes.axis((InOut::In(input), position)).unwrap();
            axis.inputs[1 - input].len() == 1 && axis.outputs[0].len() == 0
        }
        let cases = full_shapes(&axes)
            .prop_flat_map(|(a, b)| {
                (
                    a.iter()
                        .enumerate()
                        .map(|(ix, d)| {
                            if is_k(&axes, 0, ix) {
                                prop_oneof![Just(*d)].boxed()
                            } else {
                                prop_oneof![Just(1usize), Just(*d)].boxed()
                            }
                        })
                        .collect_vec(),
                    b.iter()
                        .enumerate()
                        .map(|(ix, d)| {
                            if is_k(&axes, 1, ix) {
                                prop_oneof![Just(*d)].boxed()
                            } else {
                                prop_oneof![Just(1usize), Just(*d)].boxed()
                            }
                        })
                        .collect_vec(),
                )
            })
            .prop_flat_map(|(a_shape, b_shape)| (tensor(&a_shape), tensor(&b_shape)))
            .prop_map(|(a, b)| DeconvProblem { expr: expr.clone(), a, b });
        runner.run(&cases, |pb| pb.check()).unwrap()
    }

    #[derive(Debug, Clone, PartialEq)]
    struct DeconvProblem {
        expr: String,
        a: Tensor,
        b: Tensor,
    }

    impl DeconvProblem {
        fn check(&self) -> TestCaseResult {
            let mut model = TypedModel::default();
            let sa = model.add_source("a", f32::fact(self.a.shape())).unwrap();
            let sb = model.add_source("b", f32::fact(self.b.shape())).unwrap();
            let einsum = model
                .wire_node(
                    "einsum",
                    EinSum::new(self.expr.parse().unwrap(), f32::datum_type()),
                    &[sa, sb],
                )
                .unwrap();
            model.set_output_outlets(&einsum).unwrap();
            let a = self.a.clone().into_tvalue();
            let b = self.b.clone().into_tvalue();
            let inputs = tvec!(a, b);
            let reference =
                TypedRunnableModel::new(&model).unwrap().run(inputs.clone()).unwrap().remove(0);
            decompose_einsums_in_place(&mut model).unwrap();
            model.compact().unwrap();
            prop_assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
            let test = TypedRunnableModel::new(&model).unwrap().run(inputs).unwrap().remove(0);
            reference.close_enough(&test, true).unwrap();
            Ok(())
        }
    }

    #[rustfmt::skip] #[test] fn prop_mk_kn_mn() { test_expr("mk,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_km_kn_mn() { test_expr("km,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_nk_mn() { test_expr("mk,nk->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_kn_nm() { test_expr("mk,kn->nm") }
    #[rustfmt::skip] #[test] fn prop_k_kn_mn() { test_expr("k,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_k_mn() { test_expr("mk,k->mn") }
    #[rustfmt::skip] #[test] fn prop_m_n_mn() { test_expr("m,n->mn") }

    #[test]
    fn k_kn_mn_0() -> TestCaseResult {
        DeconvProblem {
            expr: "k,kn->mn".to_string(),
            a: tensor1(&[0f32, 0f32]),
            b: tensor2(&[[0f32, 0.], [0., 0.]]),
        }
        .check()
    }

    #[test]
    fn mk_k_mn_0() -> TestCaseResult {
        DeconvProblem {
            expr: "mk,k->mn".to_string(),
            a: tensor2(&[[0f32, 0.], [0., 0.]]),
            b: tensor1(&[0f32, 0f32]),
        }
        .check()
    }

    #[test]
    fn mk_kn_nm_0() -> TestCaseResult {
        DeconvProblem {
            expr: "mk,kn->mn".to_string(),
            a: Tensor::zero::<f32>(&[3, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2, 2]).unwrap(),
        }
        .check()
    }
}
