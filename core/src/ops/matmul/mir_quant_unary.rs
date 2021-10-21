use ops::matmul::mir_quant::wire_matmul_quant;

use crate::internal::*;
use crate::ops;
use crate::ops::matmul::mir_quant::{combine_scales, requant, wire_offset_u8_as_i8};
use crate::ops::matmul::*;
use mir_quant::MatMulQParams;
use mir_quant::QParamKind;

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
        let t_a = self.a.offset_u8_as_i8();
        let a = model.add_const("source_a", self.a.clone())?;
        let b = model.add_const("source_b", inputs[0].clone())?;
        let bias = if let Some(bias) = self.bias.clone() {
            Some(model.add_const("source_bias", bias)?)
        } else {
            None
        };

        let mut input_outlets = tvec![a];
        for (i, t) in inputs.iter().enumerate().skip(1) {
            input_outlets.push(model.add_const(format!("source_{}", i), t.clone())?)
        }

        let mut params = self.params.as_outlet_ids(
            &mut model,
            "qmatmul_unary",
            &input_outlets,
            self.a.datum_type(),
            inputs[0].datum_type(),
            self.output_type,
        )?;
        let a = wire_offset_u8_as_i8(&mut model, "adhoc", a, "a", &mut params[0], "a0")?;
        let b = wire_offset_u8_as_i8(&mut model, "adhoc", b, "b", &mut params[2], "b0")?;

        let new_op = MatMulUnary {
            a: t_a,
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

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        if self.params.iter().any(|qp| match &qp.1 {
            &QParamKind::Attr(t) => t.rank() > 0,
            &QParamKind::FromInput(ix) => inputs[*ix].rank() > 0,
            &QParamKind::FromQType => false,
        }) {
            Ok(Invariants::none())
        } else {
            let mut invs = super::mir_unary::mir_unary_invariants(
                &inputs[0],
                &outputs[0],
                &self.a,
                self.b_trans,
                self.c_trans,
            )?;
            for axis in &mut invs.axes {
                axis.inputs.extend(std::iter::repeat(None).take(inputs.len() - 1));
            }
            Ok(invs)
        }
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
                    let mut bias = self.bias.clone();
                    if let Some(b) = &mut bias {
                        if b.rank() == 2 {
                            *b = b.clone().into_tensor().permute_axes(&[1, 0])?.into_arc_tensor();
                        }
                    }
                    let op = QMatMulUnary {
                        b_trans: !self.b_trans,
                        c_trans: !self.c_trans,
                        bias,
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
        suffix: &str,
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
                let bias_axis = if self.c_trans { 1 } else { 0 };
                Some(bias.slice(bias_axis, start, end)?.into_arc_tensor())
            } else {
                self.bias.clone()
            };
            let wire = patch.tap_model(model, node.inputs[0])?;
            return Ok(Some(
                patch.wire_node(
                    format!("{}.{}", node.name, suffix),
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
        use crate::ops::array::concat::ConcatSlice;
        use crate::ops::array::TypedConcat;
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            let k_axis = self.a.rank() - 1 - self.a_trans as usize;
            if concat.axis == input_fact.shape.rank() - 1 && self.b_trans {
                let mut input = 0;
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_usize())
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                let mut params_for_split = self.params.clone();
                params_for_split.a_scale = tensor0(1.0f32).into();
                params_for_split.b_scale = tensor0(1.0f32).into();
                params_for_split.c_scale = tensor0(1.0f32).into();
                params_for_split.c0 = tensor0(0i32).into();
                let input_outlets = node
                    .inputs
                    .iter()
                    .skip(1)
                    .map(|o| patch.tap_model(model, *o))
                    .collect::<TractResult<TVec<_>>>()?;
                let params_outlets = self.params.as_outlet_ids(
                    &mut patch,
                    &*node.name,
                    &input_outlets,
                    self.a.datum_type(),
                    model.node_input_facts(node.id)?[0].datum_type,
                    self.output_type,
                )?;

                let scale = combine_scales(
                    &mut patch,
                    &node.name,
                    params_outlets[1],
                    params_outlets[3],
                    params_outlets[5],
                )?;
                let c0 = params_outlets[4];

                for (ix, slice) in concat.slices.iter().enumerate() {
                    let wire = match slice {
                        ConcatSlice::Const(t) => patch.add_const(
                            format!("{}.const-{}", node.name, ix),
                            t.clone().into_arc_tensor(),
                        )?,
                        ConcatSlice::Var => {
                            input += 1;
                            patch.tap_model(model, concat_node.inputs[input - 1])?
                        }
                    };
                    let mut a = self.a.slice(k_axis, offsets[ix], offsets[ix + 1])?;
                    while a.rank() > 0 && a.shape()[0] == 1 {
                        a.remove_axis(0)?;
                    }
                    let wire = patch.wire_node(
                        format!("{}.k-{}-{}", node.name, offsets[ix], offsets[ix + 1]),
                        Self {
                            a: a.into_arc_tensor(),
                            output_type: DatumType::I32,
                            bias: self.bias.clone().filter(|_| ix == 0),
                            params: params_for_split.clone(),
                            ..self.clone()
                        },
                        &[wire],
                    )?[0];
                    wires.push(wire)
                }
                let mut wire = wires[0];
                for (ix, w) in wires[1..].iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}.k-add-{}", node.name, ix),
                        crate::ops::binary::TypedBinOp(Box::new(crate::ops::math::Add)),
                        &[wire, *w],
                    )?[0];
                }
                wire = requant(&mut patch, &node.name, wire, self.output_type, scale, c0)?;
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
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
        let t_a = self.a.offset_u8_as_i8();

        if let Some((inputs, qp)) = self.params.inline_static(model, node)? {
            let mut patch = TypedModelPatch::new("inlining matmul quantized params");
            let inputs: Vec<OutletId> =
                inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<_>>()?;
            let op = Self {
                a: t_a,
                params: MatMulQParams { a0: qp.a0.offset_u8_as_i8(&patch, &inputs)?, ..qp },
                ..self.clone()
            };
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
        let mut input_outlets = tvec![a];
        for i in node.inputs.iter().skip(1) {
            input_outlets.push(patch.tap_model(model, *i)?)
        }
        let mut params = self.params.as_outlet_ids(
            &mut patch,
            &*node.name,
            &input_outlets,
            self.a.datum_type(),
            model.node_input_facts(node.id)?[0].datum_type,
            self.output_type,
        )?;

        let a = wire_offset_u8_as_i8(&mut patch, &node.name, a, "a", &mut params[0], "a0")?;
        let b = wire_offset_u8_as_i8(&mut patch, &node.name, b, "b", &mut params[2], "b0")?;

        let new_op = MatMulUnary {
            a: t_a,
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
        fn prop_i8_i8_i8(pb in any::<QMatMulUnaryProblemI8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_i8_u8(pb in any::<QMatMulUnaryProblemI8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_i8(pb in any::<QMatMulUnaryProblemI8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_i8(pb in any::<QMatMulUnaryProblemU8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_u8(pb in any::<QMatMulUnaryProblemI8U8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_u8(pb in any::<QMatMulUnaryProblemU8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_i8(pb in any::<QMatMulUnaryProblemU8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_u8(pb in any::<QMatMulUnaryProblemU8U8U8>()) {
            pb.check();
        }
    }

    #[test]
    fn c0() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn b_scale() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 2.0,
            c_scale: 1.0,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn sat() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            bias: tensor0(0i32),
            a0: -17,
            b0: 1,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 0.25,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn rounding() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 27,
            b0: -1,
            c0: 1,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 1.0,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn neg_rounding() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            bias: tensor0(0i32),
            a0: -11,
            b0: -45,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 1.0,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn rounding_ties_2() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[47], [0]]),
            b: arr2(&[[1, 0, 30]]),
            bias: tensor0(0i32),
            a0: 86,
            b0: 19,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 0.6,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn rounding_ties_3() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[-30]]),
            b: arr2(&[[0, 107, 0]]),
            bias: tensor0(0i32),
            a0: -59,
            b0: 117,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.15,
            c_scale: 0.6,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn onnx_test_matmulinteger() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            bias: tensor0(0i32),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    fn round_ties_to_right(x: f32) -> i32 {
        (x + 0.5).floor() as i32
    }

    fn scale() -> BoxedStrategy<f32> {
        prop_oneof![Just(1.0), (1i32..=20).prop_map(|x| x as f32 / 20.0)].boxed()
    }

    macro_rules! impl_qmmup {
        ($name:ident, $a:ty, $b:ty, $c:ty $(,)?) => {
            #[derive(Debug)]
            struct $name {
                a: Array2<$a>,
                b: Array2<$b>,
                bias: Tensor,
                a0: $a,
                b0: $b,
                c0: $c,
                a_scale: f32,
                b_scale: f32,
                c_scale: f32,
                opt: bool,
                dyn_qp: bool,
            }

            impl $name {
                fn check(&self) {
                    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
                let r = self.reference();
                    let t = self.tract();
                    assert!(
                        r.iter().zip(t.iter()).all(|(r, t)| r.max(t) - r.min(t) <= 1),
                        "mismatch! optimized plan: {}, dynamic qparams: {}, reference: {:?}, tract: {:?}",
                        self.opt,
                        self.dyn_qp,
                        r,
                        t,
                        );

            }

            fn reference(&self) -> Array2<$c> {
                let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
                let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
                let c = a.dot(&b);
                let c = c.map(|&x| round_ties_to_right(x / self.c_scale) + self.c0 as i32);
                c.map(|&x| x.max(<$c>::MIN as i32).min(<$c>::MAX as i32) as $c)
            }

            fn tract(&self) -> Array2<$c> {
                let mut model = TypedModel::default();
                let mut inputs = tvec![];
                inputs.push(
                    model
                    .add_source(
                        "b",
                        TypedFact::dt_shape(
                            <$b>::datum_type(),
                            &[self.b.nrows(), self.b.ncols()],
                            ),
                            )
                    .unwrap(),
                    );
                let qparams = if self.dyn_qp {
                    inputs.push(model.add_source("a0", TypedFact::scalar::<$a>()).unwrap());
                    inputs.push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
                    inputs.push(model.add_source("b0", TypedFact::scalar::<$b>()).unwrap());
                    inputs.push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
                    inputs.push(model.add_source("c0", TypedFact::scalar::<$c>()).unwrap());
                    inputs.push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
                    MatMulQParams::all_dynamic(1)
                } else {
                    MatMulQParams {
                        a0: tensor0::<$a>(self.a0).into(),
                        a_scale: tensor0::<f32>(self.a_scale).into(),
                        b0: tensor0::<$b>(self.b0).into(),
                        b_scale: tensor0::<f32>(self.b_scale).into(),
                        c0: tensor0::<$c>(self.c0).into(),
                        c_scale: tensor0::<f32>(self.c_scale).into(),
                    }
                };
                let result = model
                    .wire_node(
                        "qmmu",
                        QMatMulUnary::new(
                            self.a.clone().into_arc_tensor(),
                            Some(self.bias.clone().into_arc_tensor()),
                            false,
                            false,
                            false,
                            <$c>::datum_type(),
                            qparams,
                            ),
                            &inputs,
                            )
                    .unwrap();
                model.set_output_outlets(&result).unwrap();

                let inputs = if self.dyn_qp {
                    tvec![
                        self.b.clone().into_tensor(),
                        self.a0.into(),
                        self.a_scale.into(),
                        self.b0.into(),
                        self.b_scale.into(),
                        self.c0.into(),
                        self.c_scale.into(),
                    ]
                } else {
                    tvec![self.b.clone().into_tensor()]
                };
                let model = if self.opt { model.into_optimized().unwrap() } else { model };
                let mut outputs = model
                    .into_runnable()
                    .unwrap()
                    .run(inputs)
                    .unwrap();
                outputs
                    .remove(0)
                    .into_tensor()
                    .into_array::<$c>()
                    .unwrap()
                    .into_dimensionality()
                    .unwrap()
            }
        }

        impl Arbitrary for $name {
            type Parameters = ();
            type Strategy = BoxedStrategy<$name>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                (1usize..=4, 1usize..=4, 1usize..=4)
                    .prop_flat_map(|(m, k, n)| {
                        (
                            Just((m, k, n)),
                            vec(any::<$a>(), m * k..=m * k),
                            vec(any::<$b>(), k * n..=k * n),
                            any::<$a>(),
                            any::<$b>(),
                            any::<$c>(),
                            scale(),
                            scale(),
                            scale(),
                            any::<bool>(),
                            any::<bool>(),
                            )
                    })
                .prop_map(|((m, k, n), a, b, a0, b0, c0, a_scale, b_scale, c_scale, opt, dyn_qp)| {
                    $name {
                        a: Array2::from_shape_vec((m, k), a).unwrap(),
                        b: Array2::from_shape_vec((k, n), b).unwrap(),
                        bias: tensor0(0i32),
                        a0,
                        b0,
                        c0,
                        a_scale,
                        b_scale,
                        c_scale,
                        opt,
                        dyn_qp
                    }
                })
                .boxed()
            }
        }
    };
}

    impl_qmmup! { QMatMulUnaryProblemI8I8I8, i8, i8, i8 }
    impl_qmmup! { QMatMulUnaryProblemI8I8U8, i8, i8, u8 }
    impl_qmmup! { QMatMulUnaryProblemI8U8I8, i8, u8, i8 }
    impl_qmmup! { QMatMulUnaryProblemU8I8I8, u8, i8, i8 }
    impl_qmmup! { QMatMulUnaryProblemI8U8U8, i8, u8, u8 }
    impl_qmmup! { QMatMulUnaryProblemU8I8U8, u8, i8, u8 }
    impl_qmmup! { QMatMulUnaryProblemU8U8I8, u8, u8, i8 }
    impl_qmmup! { QMatMulUnaryProblemU8U8U8, u8, u8, u8 }

    #[test]
    fn test_qmmup_i8_i8_i8() {
        QMatMulUnaryProblemI8I8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmup_i8_i8_u8() {
        QMatMulUnaryProblemI8I8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmup_i8_u8_i8() {
        QMatMulUnaryProblemI8U8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmup_u8_i8_i8() {
        QMatMulUnaryProblemU8I8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmup_i8_u8_u8() {
        QMatMulUnaryProblemI8U8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmup_u8_i8_u8() {
        QMatMulUnaryProblemU8I8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmup_u8_u8_i8() {
        QMatMulUnaryProblemU8U8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmup_u8_u8_u8() {
        QMatMulUnaryProblemU8U8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmup_u8_u8_u8_2() {
        QMatMulUnaryProblemU8U8U8 {
            a: arr2(&[[129, 129], [129, 128]]),
            b: arr2(&[[129, 0], [0, 129]]),
            bias: tensor0(0i32),
            a0: 128,
            b0: 128,
            c0: 0,
            a_scale: 1.,
            b_scale: 1.,
            c_scale: 1.,
            opt: true,
            dyn_qp: true,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmup_u8_u8_u8_3() {
        QMatMulUnaryProblemU8U8U8 {
            a: arr2(&[[60, 196], [114, 142]]),
            b: arr2(&[[0, 0, 0], [0, 0, 0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 0,
            a_scale: 1.,
            b_scale: 1.,
            c_scale: 1.,
            opt: true,
            dyn_qp: true,
        }
        .check();
    }

    #[test]
    fn test_qmmup_u8_u8_u8_4() {
        QMatMulUnaryProblemU8U8U8 {
            a: arr2(&[[0], [0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0),
            a0: 0,
            b0: 0,
            c0: 0,
            a_scale: 0.05,
            b_scale: 1.0,
            c_scale: 1.0,
            opt: true,
            dyn_qp: false,
        }
        .check()
    }
}
