use tract_ndarray::Ix2;
use tract_num_traits::One;

use super::codegen::*;
use super::EinSum;
use crate::internal::*;

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
    decompose_one_in_place(&mut substitute)?;
    Ok(substitute)
}

fn decompose_one_in_place(model: &mut TypedModel) -> TractResult<()> {
    ensure!(model.nodes.iter().filter(|n| n.op_is::<EinSum>()).count() == 1);
    let (m, k, n) = loop {
        let node = model.nodes.iter().find(|n| n.op_is::<EinSum>()).unwrap();
        let op = node.op_as::<EinSum>().unwrap();
        match ensure_mkn_axes(op, model, node)? {
            AxesOrPatch::Axes(m, k, n) => break (m, k, n),
            AxesOrPatch::Patch(p) => {
                p.apply(model)?;
                model.compact()?;
            }
            AxesOrPatch::NotAMatMul(axis) => bail!("{} is not a matmul because of axis {}", op.axes, axis.repr),
        };
    };
    let (m, k, n) = (m.repr, k.repr, n.repr);
    let node = model.nodes.iter().find(|n| n.op_is::<EinSum>()).unwrap();
    let op = node.op_as::<EinSum>().unwrap();
    let node_name = &node.name;
    let prefix: String =
        op.axes.iter_all_axes().filter(|a| ![m, k, n].contains(&a.repr)).map(|a| a.repr).collect();
    let mut patch = TypedModelPatch::default();
    let mut wire =
        node.inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<TVec<_>>>()?;

    let a_order_es: String = op.axes.axes(InOut::In(0)).map(|a| a.repr).collect();
    let a_order_mm = format!("{prefix}{m}{k}");
    let a_order_mm_t = format!("{prefix}{k}{m}");
    let a_transform = format!("{}->{}", a_order_es, a_order_mm)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let a_transform_t = format!("{}->{}", a_order_es, a_order_mm_t)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_a = a_transform.len() > a_transform_t.len();
    let a_transform = if transpose_a { a_transform_t } else { a_transform };
    for (ix, op) in a_transform.into_iter().enumerate() {
        wire[0] = patch.wire_node(format!("{node_name}.fix_a.{ix}"), op, &[wire[0]])?[0];
    }

    let b_order_es: String = op.axes.axes(InOut::In(1)).map(|a| a.repr).collect();
    let b_order_mm = format!("{prefix}{k}{n}");
    let b_order_mm_t = format!("{prefix}{n}{k}");
    let b_transform = format!("{}->{}", b_order_es, b_order_mm)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let b_transform_t = format!("{}->{}", b_order_es, b_order_mm_t)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_b = b_transform.len() > b_transform_t.len();
    let b_transform = if transpose_b { b_transform_t } else { b_transform };
    for (ix, op) in b_transform.into_iter().enumerate() {
        wire[1] = patch.wire_node(format!("{node_name}.fix_b.{ix}"), op, &[wire[1]])?[0];
    }

    let c_order_es: String = op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();
    let c_order_mm = format!("{prefix}{m}{n}");
    let c_order_mm_t = format!("{prefix}{n}{m}");
    let c_transform = format!("{}->{}", c_order_mm, c_order_es)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let c_transform_t = format!("{}->{}", c_order_mm_t, c_order_es)
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_c = c_transform.len() > c_transform_t.len();
    let c_transform = if transpose_c { c_transform_t } else { c_transform };

    wire =
        patch.wire_node(node_name, BasicMatMul { transpose_a, transpose_b, transpose_c }, &wire)?;

    for (ix, op) in c_transform.into_iter().enumerate() {
        wire = patch.wire_node(format!("{node_name}.fix_c.{ix}"), op, &wire)?;
    }
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    patch.apply(model)?;
    model.compact()?;
    Ok(())
}

#[derive(Clone, Debug, Default)]
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
            for &d in prefix.slice().iter() {
                a.index_axis_inplace(tract_ndarray::Axis(0), d.min(a.shape()[0] - 1));
                b.index_axis_inplace(tract_ndarray::Axis(0), d.min(b.shape()[0] - 1));
                c.index_axis_inplace(tract_ndarray::Axis(0), d);
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

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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

    fn test_expr(expr: &str) -> TestCaseResult {
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
            .prop_map(|(a, b)| EinSumProblem { expr: expr.clone(), a, b });
        runner.run(&cases, |pb| pb.check().map_err(|e| TestCaseError::fail(e.to_string())))?;
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq)]
    struct EinSumProblem {
        expr: String,
        a: Tensor,
        b: Tensor,
    }

    impl EinSumProblem {
        fn check(&self) -> TractResult<()> {
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
            decompose_one_in_place(&mut model)?;
            assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
            let test = TypedRunnableModel::new(&model).unwrap().run(inputs).unwrap().remove(0);
            reference.close_enough(&test, true).unwrap();
            Ok(())
        }
    }

    #[rustfmt::skip] #[test] fn prop_mk_kn_mn() -> TestCaseResult { test_expr("mk,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_km_kn_mn() -> TestCaseResult { test_expr("km,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_nk_mn() -> TestCaseResult { test_expr("mk,nk->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_kn_nm() -> TestCaseResult { test_expr("mk,kn->nm") }
    #[rustfmt::skip] #[test] fn prop_k_kn_mn() -> TestCaseResult { test_expr("k,kn->mn") }
    #[rustfmt::skip] #[test] fn prop_mk_k_mn() -> TestCaseResult { test_expr("mk,k->mn") }
    #[rustfmt::skip] #[test] fn prop_m_n_mn() -> TestCaseResult { test_expr("m,n->mn") }
    #[rustfmt::skip] #[test] fn prop_amk_akn_amn() -> TestCaseResult { test_expr("amk,akn->amn") }
    #[rustfmt::skip] #[test] fn prop_mk_akn_amn() -> TestCaseResult { test_expr("mk,akn->amn") }
    #[rustfmt::skip] #[test] fn prop_btgi_gih_tgh() -> TestCaseResult { test_expr("btgi,gih->tgh") }
    #[rustfmt::skip] #[test] fn prop_tgi_gih_btgh() -> TestCaseResult { test_expr("tgi,gih->btgh") }

    #[test]
    fn k_kn_mn_0() -> TractResult<()> {
        EinSumProblem {
            expr: "k,kn->mn".to_string(),
            a: tensor1(&[0f32, 0f32]),
            b: tensor2(&[[0f32, 0.], [0., 0.]]),
        }
        .check()
    }

    #[test]
    fn mk_k_mn_0() -> TractResult<()> {
        EinSumProblem {
            expr: "mk,k->mn".to_string(),
            a: Tensor::zero::<f32>(&[2, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2]).unwrap(),
        }
        .check()
    }

    #[test]
    fn mk_k_mn_1() -> TractResult<()> {
        EinSumProblem {
            expr: "mk,k->mn".to_string(),
            a: Tensor::zero::<f32>(&[1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2]).unwrap(),
        }
        .check()
    }

    #[test]
    fn mk_kn_nm_0() -> TractResult<()> {
        EinSumProblem {
            expr: "mk,kn->mn".to_string(),
            a: Tensor::zero::<f32>(&[3, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2, 2]).unwrap(),
        }
        .check()
    }

    #[test]
    fn amk_akn_amn_0() -> TractResult<()> {
        EinSumProblem {
            expr: "amk,akn->amn".to_string(),
            a: Tensor::zero::<f32>(&[1, 1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 2, 1]).unwrap(),
        }
        .check()
    }

    #[test]
    fn amk_akn_amn_1() -> TractResult<()> {
        EinSumProblem {
            expr: "amk,akn->amn".to_string(),
            a: Tensor::zero::<f32>(&[2, 1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 2, 1]).unwrap(),
        }
        .check()
    }

    #[test]
    fn amk_akn_amn_2() -> TractResult<()> {
        EinSumProblem {
            expr: "amk,akn->amn".to_string(),
            a: Tensor::zero::<f32>(&[1, 1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2, 2, 2]).unwrap(),
        }
        .check()
    }

    #[test]
    fn amk_akn_amn_3() -> TractResult<()> {
        EinSumProblem {
            expr: "amk,akn->amn".to_string(),
            a: Tensor::zero::<f32>(&[1, 1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[2, 2, 1]).unwrap(),
        }
        .check()
    }

    #[test]
    fn km_anbck_bmn_0() -> TractResult<()> {
        EinSumProblem {
            expr: "km,anbck->bmn".to_string(),
            a: Tensor::zero::<f32>(&[2, 1]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 1, 1, 1, 2]).unwrap(),
        }
        .check()
    }

    #[test]
    fn btgi_gih_tgh_0() -> TractResult<()> {
        EinSumProblem {
            expr: "btgi,gih->tgh ".to_string(),
            a: Tensor::zero::<f32>(&[2, 1, 1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 2, 1]).unwrap(),
        }
        .check()
    }

    #[test]
    fn q() -> TractResult<()> {
        let qp = QParams::ZpScale { zero_point: 0, scale: 0.1 };
        let op = EinSum {
            axes: "mk,kn,m,,,,,,->mn".parse()?,
            operating_dt: i32::datum_type(),
            q_params: Some(DatumType::QI8(qp)),
        };
        let mut model = TypedModelPatch::default();
        let inputs = [
            model.add_source("a", DatumType::QI8(qp).fact(&[3, 2]))?,
            model.add_source("b", DatumType::QI8(qp).fact(&[2, 4]))?,
            model.add_source("bias", i32::datum_type().fact(&[3]))?,
            model.add_source("a0", i8::datum_type().scalar_fact())?,
            model.add_source("a_scale", f32::datum_type().scalar_fact())?,
            model.add_source("b0", i8::datum_type().scalar_fact())?,
            model.add_source("b_scale", f32::datum_type().scalar_fact())?,
            model.add_source("c0", i8::datum_type().scalar_fact())?,
            model.add_source("c_scale", f32::datum_type().scalar_fact())?,
        ];
        let wire = model.wire_node("einsum", op.clone(), &inputs)?;
        model.set_output_outlets(&wire)?;
        let mut sub = op.decompose_in_legacy_ops(&model, model.node(wire[0].node))?;
        sub.compact()?;
        assert!(sub.nodes.iter().all(|n| !n.op_is::<EinSum>()));
        Ok(())
    }
}
