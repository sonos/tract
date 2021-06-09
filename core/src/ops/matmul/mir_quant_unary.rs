use ops::matmul::mir_quant::wire_matmul_quant;

use crate::internal::*;
use crate::ops;
use crate::ops::matmul::*;
use mir_quant::MatMulQParams;

#[derive(Debug, Clone, new, Hash)]
pub struct QMatMulUnary {
    pub a: Arc<Tensor>,
    pub bias: Option<Arc<Tensor>>,
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub output_type: DatumType,
    pub params: MatMulQParams,
}

impl_dyn_hash!(QMatMulUnary);

impl QMatMulUnary {
    pub fn with_a_trans(self, a_trans: bool) -> QMatMulUnary {
        QMatMulUnary { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> QMatMulUnary {
        QMatMulUnary { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> QMatMulUnary {
        QMatMulUnary { c_trans, ..self }
    }
}

impl Op for QMatMulUnary {
    fn name(&self) -> Cow<str> {
        "QMatMulUnary".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for QMatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if &inputs[0].rank() != &self.a.rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], self.a);
        }

        let mut model = TypedModel::default();
        let a = model.add_const("source_a", self.a.clone())?;
        let b = model.add_const("source_b", inputs[0].clone())?;
        let bias = if let Some(bias) = self.bias.clone() {
            Some(model.add_const("source_bias", bias)?)
        } else {
            None
        };

        let params = self
            .params
            .iter()
            .map(|(name, qp)| {
                model.add_const(format!("source_{}", name), qp.tensor(&inputs).clone())
            })
            .collect::<TractResult<Vec<_>>>()?;

        let new_op = MatMulUnary {
            a: self.a.clone(),
            a_trans: self.a_trans,
            b_trans: self.b_trans,
            c_trans: self.c_trans,
        };
        let result = model.wire_node("adhoc.matmul", new_op, &[b])?[0];
        let result = wire_matmul_quant(
            &mut model,
            "adhoc",
            a,
            self.a_trans,
            b,
            self.b_trans,
            bias,
            self.c_trans,
            result,
            self.output_type,
            &params,
        )?;
        model.set_output_outlets(&[result])?;
        model.into_runnable()?.run(tvec![])
    }
}

impl TypedOp for QMatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs.len() != 1 + self.params.input_count() {
            bail!(
                "Inconsistent q matmul unary. expects {} inputs, got {}",
                1 + self.params.input_count(),
                inputs.len()
            );
        }
        if inputs[0].rank() != self.a.rank() {
            bail!("Inconsistent matmul between {:?} and {:?} (rank mismatch)", inputs[0], self.a);
        }
        let (_m, _k, _n, c_shape) = compute_shape(
            &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            &inputs[0].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;

        if let Some(bias) = &self.bias {
            if bias.rank() == 2 {
                let expected_bias_shape = if self.c_trans {
                    [1, c_shape[c_shape.len() - 1].to_usize()?]
                } else {
                    [c_shape[c_shape.len() - 2].to_usize()?, 1]
                };
                anyhow::ensure!(bias.shape() == expected_bias_shape);
            } else {
                anyhow::ensure!(bias.len() == 1);
            };
        }

        Ok(tvec!(TypedFact::dt_shape(self.output_type, c_shape)))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        super::mir_unary::mir_unary_invariants(model, node, &self.a, self.b_trans, self.c_trans)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let b = &model.outlet_fact(node.inputs[0])?;
        match change {
            AxisOp::Move(from, to) => {
                if *from == b.rank() - 2 && *to == b.rank() - 1 {
                    let op = QMatMulUnary {
                        b_trans: !self.b_trans,
                        c_trans: !self.c_trans,
                        ..self.clone()
                    };
                    Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
                } else {
                    Ok(None)
                }
            }
            AxisOp::Add(axis) if *axis < b.rank() - 1 => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis)?;
                let op =
                    Some(Box::new(QMatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            // b is [.. 1, n], can add axis to the right and transpose
            AxisOp::Add(axis) if *axis == b.rank() && b.shape[b.rank() - 2] == 1.to_dim() => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis - 2)?;
                let op = QMatMulUnary {
                    b_trans: !self.b_trans,
                    c_trans: !self.c_trans,
                    a: a.into_arc_tensor(),
                    ..self.clone()
                };
                Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
            }
            AxisOp::Rm(axis) if b.rank() - axis > 2 => {
                let mut a = self.a.clone().into_tensor();
                a.remove_axis(*axis)?;
                let op =
                    Some(Box::new(QMatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            _ => return Ok(None),
        }
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let b_fact = model.outlet_fact(node.inputs[0])?;
        let c_fact = &self.output_facts(&[b_fact])?[0];
        if axis + self.c_trans as usize == c_fact.shape.rank() {
            let a_split_axis = self.a.rank() - 1 - !self.a_trans as usize;
            let a = self.a.slice(a_split_axis, start, end)?.into_arc_tensor();
            let bias = if let Some(bias) = self.bias.as_ref().filter(|b| b.len() > 1) {
                debug_assert_eq!(bias.rank(), 2);
                let bias_axis = if self.c_trans { 0 } else { 1 };
                Some(bias.slice(bias_axis, start, end)?.into_arc_tensor())
            } else {
                self.bias.clone()
            };
            let wire = patch.tap_model(model, node.inputs[0])?;
            return Ok(Some(
                patch.wire_node(
                    format!("{}.sliced-m-{}-{}", node.name, start, end),
                    Self { a, bias, ..self.clone() },
                    &[wire],
                )?[0],
            ));
        }
        return Ok(None);
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        super::mir_unary::mir_unary_declutter_leading_concat_on_k(
            model,
            node,
            &self.a,
            self.a_trans,
            self.b_trans,
            &|a: Arc<Tensor>| Box::new(QMatMulUnary { a, ..self.clone() }),
        )
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.a_trans,
            self.b_trans,
        )
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();

        if let Some((inputs, qp)) = self.params.inline_static(model, node)? {
            let mut patch = TypedModelPatch::new("inlining matmul quantized params");
            let inputs: Vec<OutletId> =
                inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<_>>()?;
            let op = Self { params: qp, ..self.clone() };
            let wire = patch.wire_node(&node.name, op, &inputs)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            return Ok(Some(patch));
        }

        let a = patch.wire_node(
            format!("{}.a_const", &node.name),
            ops::konst::Const(self.a.clone()),
            &[],
        )?[0];
        let b = patch.tap_model(model, node.inputs[0])?;
        let bias = if let Some(bias) = self.bias.clone() {
            Some(patch.add_const(format!("{}.bias_const", &node.name), bias)?)
        } else {
            None
        };

        let params = self
            .params
            .iter()
            .map(|(name, qp)| match qp {
                AttrOrInput::Input(o) => patch.tap_model(model, node.inputs[*o]),
                AttrOrInput::Attr(t) => {
                    patch.add_const(format!("{}_{}", node.name, name), t.clone())
                }
            })
            .collect::<TractResult<Vec<OutletId>>>()?;
        let new_op = MatMulUnary {
            a: self.a.clone(),
            a_trans: self.a_trans,
            b_trans: self.b_trans,
            c_trans: self.c_trans,
        };
        let result = patch.wire_node(format!("{}.matmul", &node.name), new_op, &[b])?[0];
        let result = wire_matmul_quant(
            &mut patch,
            &node.name,
            a,
            self.a_trans,
            b,
            self.b_trans,
            bias,
            self.c_trans,
            result,
            self.output_type,
            &params,
        )?;
        patch.shunt_outside(model, node.id.into(), result)?;
        Ok(Some(patch))
    }

    as_op!();
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_ndarray::prelude::*;

    proptest! {
        #[test]
        fn prop(pb in any::<QMatMulUnaryProblem>()) {
            pb.check()
        }
    }

    #[test]
    fn c0() {
        QMatMulUnaryProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: arr2(&[[0]]),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check()
    }

    #[test]
    fn b_scale() {
        QMatMulUnaryProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: arr2(&[[0]]),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 2.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn sat() {
        QMatMulUnaryProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            bias: arr2(&[[0]]),
            a0: -17,
            b0: 1,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 0.25,
        }
        .check();
    }

    #[test]
    fn rounding() {
        QMatMulUnaryProblem {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            bias: arr2(&[[0]]),
            a0: 27,
            b0: -1,
            c0: 1,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn neg_rounding() {
        QMatMulUnaryProblem {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            bias: arr2(&[[0]]),
            a0: -11,
            b0: -45,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn rounding_ties_2() {
        QMatMulUnaryProblem {
            a: arr2(&[[47], [0]]),
            b: arr2(&[[1, 0, 30]]),
            bias: arr2(&[[0]]),
            a0: 86,
            b0: 19,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn rounding_ties_3() {
        QMatMulUnaryProblem {
            a: arr2(&[[-30]]),
            b: arr2(&[[0, 107, 0]]),
            bias: arr2(&[[0]]),
            a0: -59,
            b0: 117,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.15,
            c_scale: 0.6,
        };
    }

    #[test]
    fn onnx_test_matmulinteger() {
        QMatMulUnaryProblem {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            bias: arr2(&[[0]]),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check()
    }

    #[derive(Debug)]
    struct QMatMulUnaryProblem {
        a: Array2<i8>,
        b: Array2<i8>,
        bias: Array2<i32>,
        a0: i8,
        b0: i8,
        c0: i8,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    }

    fn round_ties_to_right(x: f32) -> i32 {
        (x + 0.5).floor() as i32
    }

    impl QMatMulUnaryProblem {
        fn check(&self) {
            let r = self.reference();
            let t = self.tract();
            if r.iter().zip(t.iter()).any(|(r, t)| r.max(t) - r.min(t) > 1) {
                panic!("mismatch! refernce: {:?} tract: {:?}", r, t)
            }
        }

        fn reference(&self) -> Array2<i8> {
            let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
            let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
            let c = a.dot(&b);
            let c = c.map(|&x| round_ties_to_right(x / self.c_scale) + self.c0 as i32);
            c.map(|&x| x.max(-128).min(127) as i8)
        }

        fn tract(&self) -> Array2<i8> {
            let mut model = TypedModel::default();
            let mut inputs = tvec!();
            inputs.push(
                model
                    .add_source(
                        "b",
                        TypedFact::dt_shape(i8::datum_type(), &[self.b.nrows(), self.b.ncols()]),
                    )
                    .unwrap(),
            );
            inputs.push(model.add_source("a0", TypedFact::scalar::<i8>()).unwrap());
            inputs.push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
            inputs.push(model.add_source("b0", TypedFact::scalar::<i8>()).unwrap());
            inputs.push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
            inputs.push(model.add_source("c0", TypedFact::scalar::<i8>()).unwrap());
            inputs.push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
            let result = model
                .wire_node(
                    "qmm",
                    QMatMulUnary::new(
                        self.a.clone().into_arc_tensor(),
                        self.bias.clone().into_arc_tensor(),
                        false,
                        false,
                        false,
                        i8::datum_type(),
                        MatMulQParams::all_dynamic(1),
                    ),
                    &inputs,
                )
                .unwrap();
            model.set_output_outlets(&result).unwrap();
            let mut result = model
                .into_runnable()
                .unwrap()
                .run(tvec!(
                    self.b.clone().into_tensor(),
                    self.a0.into(),
                    self.a_scale.into(),
                    self.b0.into(),
                    self.b_scale.into(),
                    self.c0.into(),
                    self.c_scale.into(),
                ))
                .unwrap();
            result
                .remove(0)
                .into_tensor()
                .into_array::<i8>()
                .unwrap()
                .into_dimensionality()
                .unwrap()
        }
    }

    fn scale() -> BoxedStrategy<f32> {
        prop_oneof![Just(1.0), (1i32..=20).prop_map(|x| x as f32 / 20.0)].boxed()
    }

    impl Arbitrary for QMatMulUnaryProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<QMatMulUnaryProblem>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (1usize..=4, 1usize..=4, 1usize..=4)
                .prop_flat_map(|(m, k, n)| {
                    (
                        Just((m, k, n)),
                        vec(any::<i8>(), m * k..=m * k),
                        vec(any::<i8>(), k * n..=k * n),
                        any::<i8>(),
                        any::<i8>(),
                        any::<i8>(),
                        scale(),
                        scale(),
                        scale(),
                    )
                })
                .prop_map(|((m, k, n), a, b, a0, b0, c0, a_scale, b_scale, c_scale)| {
                    QMatMulUnaryProblem {
                        a: Array2::from_shape_vec((m, k), a).unwrap(),
                        b: Array2::from_shape_vec((k, n), b).unwrap(),
                        bias: arr2(&[[0i32]]),
                        a0,
                        b0,
                        c0,
                        a_scale,
                        b_scale,
                        c_scale,
                    }
                })
                .boxed()
        }
    }
}
