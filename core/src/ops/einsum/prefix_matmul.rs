use tract_data::itertools::Itertools;
use tract_linalg::Scaler;
use tract_ndarray::Ix2;
use tract_num_traits::One;

use super::einsum_matmul::EinSumMatMul;
use super::eval::dequant_inputs;
use crate::internal::*;
use crate::ops::einsum::block_quant_aware_input_shape;
use crate::ops::konst::Const;

pub fn rewrite_einsum_to_prefix_matmul(model: &mut TypedModel) -> TractResult<()> {
    super::einsum_matmul::detect_all(model)?;
    Rewriter::default().with_rule_for("einsum-to-prefix-matmul", rule).rewrite(&(), model)
}

fn rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &EinSumMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    // F: 2 inputs
    // Q: 9 inputs
    if !((op.q_params.is_none() && node.inputs.len() == 2)
        || (op.q_params.is_some() && node.inputs.len() == 9))
    {
        return Ok(None);
    }
    if op.q_params.is_some()
        && model.node_input_facts(node.id)?.iter().skip(3).any(|i| i.konst.is_none())
    {
        return Ok(None);
    }
    let prefix: String = op
        .axes
        .iter_all_axes()
        .filter(|a| ![op.m_axis, op.k_axis, op.n_axis].contains(&a.repr))
        .map(|a| a.repr)
        .collect();
    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &node.inputs)?;
    let mut wire = tvec!(inputs[0], inputs[1]);

    let (m, k, n) = (op.m_axis, op.k_axis, op.n_axis);
    let a_order_es: String = op.axes.axes(InOut::In(0)).map(|a| a.repr).collect();
    let a_order_mm = format!("{prefix}{m}{k}");
    let a_order_mm_t = format!("{prefix}{k}{m}");
    let a_transform = format!("{a_order_es}->{a_order_mm}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let a_transform_t = format!("{a_order_es}->{a_order_mm_t}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_a = a_transform.len() > a_transform_t.len();
    let a_transform = if transpose_a { a_transform_t } else { a_transform };
    let name = format!("{node_name}.fix_a");
    for op in a_transform {
        wire[0] = patch.wire_node(&name, op, &[wire[0]])?[0];
    }
    // terrible hack to maintain opaque fact through eager propatagation of constant through the
    // axes transformation
    if let Some(op) = patch.node_mut(wire[0].node).op_as_mut::<Const>() {
        *op = Const::new_with_opt_opaque_fact(
            op.val().clone(),
            model.outlet_fact(node.inputs[0])?.opaque_fact.clone(),
        )?;
    }
    patch
        .outlet_fact_mut(wire[0])?
        .opaque_fact
        .clone_from(&model.outlet_fact(node.inputs[0])?.opaque_fact);
    // end of hack

    let b_order_es: String = op.axes.axes(InOut::In(1)).map(|a| a.repr).collect();
    let b_order_mm = format!("{prefix}{k}{n}");
    let b_order_mm_t = format!("{prefix}{n}{k}");
    let b_transform = format!("{b_order_es}->{b_order_mm}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let b_transform_t = format!("{b_order_es}->{b_order_mm_t}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_b = b_transform.len() > b_transform_t.len();
    let b_transform = if transpose_b { b_transform_t } else { b_transform };
    let name = format!("{node_name}.fix_b");
    for op in b_transform {
        wire[1] = patch.wire_node(&name, op, &[wire[1]])?[0];
    }

    let c_order_es: String = op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();
    let c_order_mm = format!("{prefix}{m}{n}");
    let c_order_mm_t = format!("{prefix}{n}{m}");
    let c_transform = format!("{c_order_mm}->{c_order_es}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let c_transform_t = format!("{c_order_mm_t}->{c_order_es}")
        .parse::<AxesMapping>()?
        .translate_to_axis_ops()?;
    let transpose_c = c_transform.len() > c_transform_t.len();
    let c_transform = if transpose_c { c_transform_t } else { c_transform };
    let quantize_output = if let Some(qp) = op.q_params {
        let qparams: Vec<&Tensor> = inputs[3..9]
            .iter()
            .map(|f| {
                patch
                    .outlet_fact(*f)?
                    .konst
                    .as_deref()
                    .context("Can only translate fixed scalar quantization")
            })
            .try_collect()?;
        Some(qp.with_qparams(QParams::ZpScale {
            zero_point: qparams[4].cast_to_scalar::<i32>()?,
            scale: qparams[5].cast_to_scalar::<f32>()?,
        }))
    } else {
        None
    };
    wire = patch.wire_node(
        node_name,
        PrefixMatMul { transpose_a, transpose_b, transpose_c, quantize_output },
        &wire,
    )?;

    for (ix, op) in c_transform.into_iter().enumerate() {
        wire = patch.wire_node(format!("{node_name}.fix_c.{ix}"), op, &wire)?;
    }
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

#[derive(Clone, Debug, Copy, Default)]
pub struct PrefixMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_c: bool,
    pub quantize_output: Option<DatumType>,
}

impl PrefixMatMul {
    fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2 + self.transpose_a as usize].clone());
        output.push(b[rank - 2 + !self.transpose_b as usize].clone());
        if self.transpose_c {
            output.swap(rank - 2, rank - 1);
        }
        output
    }

    fn mm<Acc: Datum + tract_ndarray::LinalgScalar>(
        &self,
        acc: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
    ) -> TractResult<()> {
        use crate::ndarray::Dimension;
        let a = a.to_array_view::<Acc>()?;
        let b = b.to_array_view::<Acc>()?;
        let mut c = acc.to_array_view_mut::<Acc>()?;
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

impl Op for PrefixMatMul {
    fn name(&self) -> Cow<str> {
        "PrefixMatMul".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "transpose_a: {} transpose_b: {} transpose_c: {} q: {:?}",
            self.transpose_a, self.transpose_b, self.transpose_c, self.quantize_output
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for PrefixMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let c_dt = if inputs[0].datum_type().is_number() {
            inputs[0].datum_type()
        } else if inputs[1].datum_type().is_number() {
            inputs[1].datum_type()
        } else {
            f32::datum_type()
        };
        let inputs = dequant_inputs(c_dt, inputs)?;

        let output_shape = self.output_shape(inputs[0].shape(), inputs[1].shape());

        if let Some(qp) = self.quantize_output {
            let mut acc = Tensor::zero_dt(i32::datum_type(), &output_shape)?;
            let mut a_i32 = inputs[0].cast_to::<i32>()?.into_owned();
            a_i32
                .as_slice_mut::<i32>()?
                .iter_mut()
                .for_each(|x| *x -= inputs[0].datum_type().zp_scale().0);
            let mut b_i32 = inputs[1].cast_to::<i32>()?.into_owned();
            b_i32
                .as_slice_mut::<i32>()?
                .iter_mut()
                .for_each(|x| *x -= inputs[1].datum_type().zp_scale().0);
            self.mm::<i32>(&mut acc, &a_i32, &b_i32)?;
            let scale = inputs[0].datum_type().zp_scale().1 * inputs[1].datum_type().zp_scale().1
                / qp.zp_scale().1;
            let scaler = Scaler::new(scale, tract_linalg::mmm::RoundingPolicy::Even);
            acc.to_array_view_mut::<i32>()?.iter_mut().for_each(|x| *x = *x * scaler);
            let mut c: Tensor = acc.cast_to_dt(qp.unquantized())?.into_owned();
            unsafe { c.set_datum_type(qp) };
            Ok(tvec!(c.into_tvalue()))
        } else {
            let mut c = Tensor::zero_dt(c_dt, &output_shape)?;
            dispatch_floatlike!(Self::mm(c_dt)(self, &mut c, &inputs[0], &inputs[1]))?;
            Ok(tvec!(c.into_tvalue()))
        }
    }
}

impl TypedOp for PrefixMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };
        let a_shape = block_quant_aware_input_shape(inputs[0])?;
        let b_shape = block_quant_aware_input_shape(inputs[1])?;
        let dt = self.quantize_output.unwrap_or(if a.datum_type.is_number() {
            a.datum_type
        } else {
            b.datum_type
        });
        Ok(tvec!(dt.fact(self.output_shape(&a_shape, &b_shape))))
    }

    as_op!();
}

#[cfg(test)]
mod test {
    use crate::ops::einsum::EinSum;

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
        fn is_disapearing_axis(axes: &AxesMapping, input: usize, position: usize) -> bool {
            let axis = axes.axis((InOut::In(input), position)).unwrap();
            axis.outputs[0].len() == 0
        }
        let cases = full_shapes(&axes)
            .prop_flat_map(|(a, b)| {
                (
                    a.iter()
                        .enumerate()
                        .map(|(ix, d)| {
                            if is_k(&axes, 0, ix) {
                                prop_oneof![Just(*d)].boxed()
                            } else if is_disapearing_axis(&axes, 0, ix) {
                                Just(1).boxed()
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
                            } else if is_disapearing_axis(&axes, 1, ix) {
                                Just(1).boxed()
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
            rewrite_einsum_to_prefix_matmul(&mut model)?;
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
    fn q() -> TractResult<()> {
        let qp = QParams::ZpScale { zero_point: 0, scale: 0.1 };
        let op = EinSum {
            axes: "mk,kn,m,,,,,,->mn".parse()?,
            operating_dt: i32::datum_type(),
            q_params: Some(DatumType::QI8(qp)),
        };
        let mut model = TypedModelPatch::default();
        let inputs = [
            model.add_source("a", DatumType::QI8(qp).fact([3, 2]))?,
            model.add_source("b", DatumType::QI8(qp).fact([2, 4]))?,
            model.add_source("bias", i32::datum_type().fact([3]))?,
            model.add_const("a0", tensor0(qp.zp_scale().0))?,
            model.add_const("a_scale", tensor0(qp.zp_scale().1))?,
            model.add_const("b0", tensor0(qp.zp_scale().0))?,
            model.add_const("b_scale", tensor0(qp.zp_scale().1))?,
            model.add_const("c0", tensor0(qp.zp_scale().0))?,
            model.add_const("c_scale", tensor0(qp.zp_scale().1))?,
        ];
        let wire = model.wire_node("einsum", op.clone(), &inputs)?;
        model.set_output_outlets(&wire)?;
        rewrite_einsum_to_prefix_matmul(&mut model)?;
        assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
        Ok(())
    }
}
