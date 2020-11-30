use ops::binary::wire_with_rank_broadcast;

use crate::internal::*;
use crate::ops;
use crate::ops::matmul::*;

#[derive(Debug, Clone, new, Hash)]
pub struct QMatMul {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub output_type: DatumType,
}

impl_dyn_hash!(QMatMul);

impl QMatMul {
    pub fn with_a_trans(self, a_trans: bool) -> QMatMul {
        QMatMul { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> QMatMul {
        QMatMul { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> QMatMul {
        QMatMul { c_trans, ..self }
    }
}

impl Op for QMatMul {
    fn name(&self) -> Cow<str> {
        "QMatMul".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for QMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if &inputs[0].rank() != &inputs[3].rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], inputs[1]);
        }

        let mut model = TypedModel::default();
        let a = model.add_const("source_a", inputs.remove(0))?;
        let a_scale = model.add_const("source_a_scale", inputs.remove(0))?;
        let a0 = model.add_const("source_a0", inputs.remove(0))?;
        let b = model.add_const("source_b", inputs.remove(0))?;
        let b_scale = model.add_const("source_b_scale", inputs.remove(0))?;
        let b0 = model.add_const("source_b0", inputs.remove(0))?;
        let c_scale = model.add_const("source_c_scale", inputs.remove(0))?;
        let c0 = model.add_const("source_c0", inputs.remove(0))?;

        let result = self.wire(&mut model, "adhoc", a, a_scale, a0, b, b_scale, b0, c_scale, c0)?;
        model.set_output_outlets(&[result])?;
        model.into_runnable()?.run(inputs.into_iter().map(|i| i.into_tensor()).collect())
    }
}

impl TypedOp for QMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[3].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[3]
            );
        }
        let dt = inputs[7].datum_type;
        let (_m, _k, _n, c_shape) = compute_shape(
            &inputs[0].shape,
            &inputs[3].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;
        Ok(tvec!(TypedFact::dt_shape(dt, c_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        let a = patch.tap_model(model, node.inputs[0])?;
        let a_scale = patch.tap_model(model, node.inputs[1])?;
        let a0 = patch.tap_model(model, node.inputs[2])?;
        let b = patch.tap_model(model, node.inputs[3])?;
        let b_scale = patch.tap_model(model, node.inputs[4])?;
        let b0 = patch.tap_model(model, node.inputs[5])?;
        let c_scale = patch.tap_model(model, node.inputs[6])?;
        let c0 = patch.tap_model(model, node.inputs[7])?;

        let result =
            self.wire(&mut patch, &node.name, a, a_scale, a0, b, b_scale, b0, c_scale, c0)?;
        patch.shunt_outside(model, node.id.into(), result)?;
        Ok(Some(patch))
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

    as_op!();
}

impl QMatMul {
    fn wire(
        &self,
        model: &mut TypedModel,
        name: &str,
        a: OutletId,
        a_scale: OutletId,
        a0: OutletId,
        b: OutletId,
        b_scale: OutletId,
        b0: OutletId,
        c_scale: OutletId,
        c0: OutletId,
    ) -> TractResult<OutletId> {
        let a_fact = model.outlet_fact(a)?.clone();
        let rank = a_fact.rank();
        let k = model.outlet_fact(a)?.shape[rank - 2 + !self.a_trans as usize].clone();

        let ab_scale = wire_with_rank_broadcast(
            &format!("{}.ab_scale", name),
            model,
            ops::math::mul::bin_typed(),
            &[a_scale, b_scale],
        )?[0];
        let abc_scale = wire_with_rank_broadcast(
            &format!("{}.abc_scales", name),
            model,
            ops::math::div::bin_typed(),
            &[ab_scale, c_scale],
        )?[0];

        let a_i32 = model.wire_node(
            format!("{}.a_as_i32", name),
            ops::cast::cast(i32::datum_type()),
            &[a],
        )?[0];
        let b_i32 = model.wire_node(
            format!("{}.b_as_i32", name),
            ops::cast::cast(i32::datum_type()),
            &[b],
        )?[0];
        let sum_a = model.wire_node(
            format!("{}.sum_a", name),
            ops::nn::Reduce::new(tvec!(rank - 2 + !self.a_trans as usize), ops::nn::Reducer::Sum),
            &[a_i32],
        )?[0];
        let sum_b = model.wire_node(
            format!("{}.sum_b", name),
            ops::nn::Reduce::new(tvec!(rank - 2 + self.b_trans as usize), ops::nn::Reducer::Sum),
            &[b_i32],
        )?[0];

        let new_op = MatMul { a_trans: self.a_trans, b_trans: self.b_trans, c_trans: self.c_trans };
        let result = model.wire_node(format!("{}.matmul", &name), new_op, &[a, b])?[0];
        let result = compensate_zero_points(model, name, result, k, a0, b0, sum_a, sum_b)?;
        requant(model, name, result, self.output_type, abc_scale, c0)
    }
}

pub(crate) fn compensate_zero_points(
    model: &mut TypedModel,
    name: &str,
    result: OutletId,
    k: TDim,
    a0: OutletId,
    b0: OutletId,
    sum_a: OutletId,
    sum_b: OutletId,
) -> TractResult<OutletId> {
    let a0 =
        model.wire_node(format!("{}.cast_a0", name), ops::cast::cast(i32::datum_type()), &[a0])?[0];

    let b0 =
        model.wire_node(format!("{}.cast_b0", name), ops::cast::cast(i32::datum_type()), &[b0])?[0];

    let k = model.add_const(format!("{}.k", name), rctensor0(k.clone()))?;
    let k =
        model.wire_node(format!("{}.cast_k", name), ops::cast::cast(i32::datum_type()), &[k])?[0];

    let a0_sum_b = wire_with_rank_broadcast(
        &format!("{}.a0_sum_b", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0, sum_b],
    )?[0];

    let b0_sum_a = wire_with_rank_broadcast(
        &format!("{}.b0_sum_a", name),
        model,
        ops::math::mul::bin_typed(),
        &[b0, sum_a],
    )?[0];

    let a0_k = wire_with_rank_broadcast(
        &format!("{}.a0_k", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0, k],
    )?[0];

    let a0_k_b0 = wire_with_rank_broadcast(
        &format!("{}.a0_k_b0", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0_k, b0],
    )?[0];

    let result = model.wire_node(
        &format!("{}.minus_a0_B", &name),
        ops::math::sub::bin_typed(),
        &[result, a0_sum_b],
    )?[0];
    let result = model.wire_node(
        &format!("{}.minus_b0_A", &name),
        ops::math::sub::bin_typed(),
        &[result, b0_sum_a],
    )?[0];

    let result = wire_with_rank_broadcast(
        &format!("{}.plus_a0_k_b0", &name),
        model,
        ops::math::add::bin_typed(),
        &[result, a0_k_b0],
    )?[0];
    Ok(result)
}

pub(crate) fn requant(
    model: &mut TypedModel,
    name: &str,
    wire: OutletId,
    dt: DatumType,
    scale: OutletId,
    zero_point: OutletId,
) -> TractResult<OutletId> {
    let wire = model.wire_node(
        format!("{}.dequant", name),
        ops::cast::cast(f32::datum_type()),
        &[wire],
    )?[0];

    let wire = wire_with_rank_broadcast(
        &format!("{}.scale", name),
        model,
        ops::math::mul::bin_typed(),
        &[wire, scale],
    )?;

    let wire = model.wire_node(format!("{}.round", name), ops::math::round(), &wire)?[0];

    let wire = model.wire_node(
        format!("{}.requant", name),
        ops::cast::cast(i32::datum_type()),
        &[wire],
    )?[0];

    let zero_point = model.wire_node(
        format!("{}.cast_c0", name),
        ops::cast::cast(i32::datum_type()),
        &[zero_point],
    )?[0];

    let wire = wire_with_rank_broadcast(
        &format!("{}.zeropoint", name),
        model,
        ops::math::add::bin_typed(),
        &[wire, zero_point],
    )?[0];

    if dt == i32::datum_type() {
        return Ok(wire);
    }

    let rank = model.outlet_fact(wire)?.rank();
    let inf = tensor0(if dt == DatumType::I8 { -128i32 } else { 0 })
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let sup = tensor0(if dt == DatumType::I8 { 127i32 } else { 255 })
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let wire = model.wire_node(format!("{}.min", name), ops::math::min::unary(sup), &[wire])?;
    let wire = model.wire_node(format!("{}.max", name), ops::math::max::unary(inf), &wire)?;
    let wire = model.wire_node(format!("{}.cast", name), ops::cast::cast(dt), &wire)?;
    Ok(wire[0])
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_ndarray::prelude::*;

    proptest! {
        #[test]
        fn prop(pb in any::<QMatMulProblem>()) {
            assert_eq!(pb.tract(), pb.reference());
        }
    }

    #[test]
    fn c0() {
        let pb = QMatMulProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[test]
    fn b_scale() {
        let pb = QMatMulProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 2.0,
            c_scale: 1.0,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[test]
    fn sat() {
        let pb = QMatMulProblem {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            a0: -17,
            b0: 1,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 0.25,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[test]
    fn rounding() {
        let pb = QMatMulProblem {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            a0: 27,
            b0: -1,
            c0: 1,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 1.0,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[test]
    fn neg_rounding() {
        let pb = QMatMulProblem {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            a0: -11,
            b0: -45,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 1.0,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[test]
    fn onnx_test_matmulinteger() {
        let pb = QMatMulProblem {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        };
        assert_eq!(pb.reference(), pb.tract())
    }

    #[derive(Debug)]
    struct QMatMulProblem {
        a: Array2<i8>,
        b: Array2<i8>,
        a0: i8,
        b0: i8,
        c0: i8,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    }

    impl QMatMulProblem {
        fn reference(&self) -> Array2<i8> {
            let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
            let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
            let c = a.dot(&b);
            let c = c.map(|&x| ((x / self.c_scale).round() + self.c0 as f32));
            c.map(|&x| (x as i32).max(-128).min(127) as i8)
        }

        fn tract(&self) -> Array2<i8> {
            let mut model = TypedModel::default();
            let mut inputs = tvec!();
            inputs.push(
                model
                    .add_source(
                        "a",
                        TypedFact::dt_shape(i8::datum_type(), &[self.a.nrows(), self.a.ncols()]),
                    )
                    .unwrap(),
            );
            inputs.push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
            inputs.push(model.add_source("a0", TypedFact::scalar::<i8>()).unwrap());
            inputs.push(
                model
                    .add_source(
                        "b",
                        TypedFact::dt_shape(i8::datum_type(), &[self.b.nrows(), self.b.ncols()]),
                    )
                    .unwrap(),
            );
            inputs.push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
            inputs.push(model.add_source("b0", TypedFact::scalar::<i8>()).unwrap());
            inputs.push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
            inputs.push(model.add_source("c0", TypedFact::scalar::<i8>()).unwrap());
            let result = model
                .wire_node("qmm", QMatMul::new(false, false, false, i8::datum_type()), &inputs)
                .unwrap();
            model.set_output_outlets(&result).unwrap();
            let mut result = model
                .into_runnable()
                .unwrap()
                .run(tvec!(
                    self.a.clone().into_tensor(),
                    self.a_scale.into(),
                    self.a0.into(),
                    self.b.clone().into_tensor(),
                    self.b_scale.into(),
                    self.b0.into(),
                    self.c_scale.into(),
                    self.c0.into()
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

    impl Arbitrary for QMatMulProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<QMatMulProblem>;
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
                    QMatMulProblem {
                        a: Array2::from_shape_vec((m, k), a).unwrap(),
                        b: Array2::from_shape_vec((k, n), b).unwrap(),
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
