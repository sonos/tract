use anyhow::ensure;
use ops::binary::wire_with_rank_broadcast;

use crate::internal::*;
use crate::ops;
use crate::ops::einsum::EinSum;
use crate::ops::matmul::*;
use crate::ops::quant::offset_u8_as_i8_elementwise;

pub fn offset_u8_as_i8(param: &Arc<Tensor>) -> TractResult<AttrOrInput> {
    match param.datum_type().unquantized() {
        DatumType::U8 => {
            Ok(param.to_array_view()?.mapv(offset_u8_as_i8_elementwise).into_arc_tensor().into())
        }
        DatumType::I32 => {
            Ok(param.to_array_view()?.mapv(|i: i32| i - 128).into_arc_tensor().into())
        }
        _ => Ok(param.clone().into()),
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MatMulQParams {
    pub a0: AttrOrInput,
    pub a_scale: AttrOrInput,
    pub b0: AttrOrInput,
    pub b_scale: AttrOrInput,
    pub c0: AttrOrInput,
    pub c_scale: AttrOrInput,
}

impl MatMulQParams {
    pub fn noop_static(dt: DatumType) -> MatMulQParams {
        MatMulQParams {
            a0: Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor().into(),
            a_scale: rctensor0(1f32).into(),
            b0: Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor().into(),
            b_scale: rctensor0(1f32).into(),
            c0: Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor().into(),
            c_scale: rctensor0(1f32).into(),
        }
    }

    pub fn all_dynamic(offset: usize) -> MatMulQParams {
        MatMulQParams {
            a0: offset.into(),
            a_scale: (offset + 1).into(),
            b0: (offset + 2).into(),
            b_scale: (offset + 3).into(),
            c0: (offset + 4).into(),
            c_scale: (offset + 5).into(),
        }
    }

    pub fn all_from_qtype(a: &QParams, b: &QParams, c: &QParams) -> TractResult<MatMulQParams> {
        Ok(MatMulQParams {
            a0: tensor0(a.zp_scale().0).into(),
            a_scale: tensor0(a.zp_scale().1).into(),
            b0: tensor0(b.zp_scale().0).into(),
            b_scale: tensor0(b.zp_scale().1).into(),
            c0: tensor0(c.zp_scale().0).into(),
            c_scale: tensor0(c.zp_scale().1).into(),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &AttrOrInput)> {
        vec![
            ("a0", &self.a0),
            ("a_scale", &self.a_scale),
            ("b0", &self.b0),
            ("b_scale", &self.b_scale),
            ("c0", &self.c0),
            ("c_scale", &self.c_scale),
        ]
        .into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut AttrOrInput)> {
        vec![
            ("a0", &mut self.a0),
            ("a_scale", &mut self.a_scale),
            ("b0", &mut self.b0),
            ("b_scale", &mut self.b_scale),
            ("c0", &mut self.c0),
            ("c_scale", &mut self.c_scale),
        ]
        .into_iter()
    }

    pub fn inline_static(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<(Vec<OutletId>, MatMulQParams)>> {
        let mut new = self.clone();
        let mut inputs = vec![];
        for (ix, input) in node.inputs.iter().enumerate() {
            if let (Some(position), Some(k)) = (
                self.iter().position(|qp| &AttrOrInput::Input(ix) == qp.1),
                model.outlet_fact(*input)?.konst.as_ref(),
            ) {
                *new.iter_mut().nth(position).unwrap().1 = AttrOrInput::Attr(k.clone());
                new.remove_input(inputs.len());
            } else {
                inputs.push(*input)
            }
        }
        Ok(Some((inputs, new)).filter(|pair| &pair.1 != self))
    }

    pub fn remove_input(&mut self, ix: usize) {
        for (_, qp) in self.iter_mut() {
            qp.remove_input(ix);
        }
    }

    pub fn insert_input(&mut self, ix: usize) {
        for (_, qp) in self.iter_mut() {
            qp.insert_input(ix)
        }
    }

    pub fn input_count(&self) -> usize {
        self.iter().filter(|qp| matches!(qp.1, AttrOrInput::Input(_))).count()
    }

    pub fn as_outlet_ids(
        &self,
        model: &mut TypedModel,
        node_name: &str,
        input_wires: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut params_outlets = tvec!();
        for (param_name, param) in self.iter() {
            params_outlets.push(param.as_input(
                model,
                input_wires,
                format!("{node_name}.{param_name}"),
            )?)
        }
        Ok(params_outlets)
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct QMatMul {
    pub axes: MatMulAxes,
    pub output_type: DatumType,
    pub params: MatMulQParams,
}

impl Op for QMatMul {
    fn name(&self) -> Cow<str> {
        "QMatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for QMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(
            inputs[0].rank() == inputs[1].rank(),
            "Rank mismatch {:?} vs {:?}",
            inputs[0],
            inputs[1]
        );

        let mut model = TypedModel::default();
        let a = model.add_const("source_a", inputs[0].clone().into_arc_tensor())?;
        let b = model.add_const("source_b", inputs[1].clone().into_arc_tensor())?;
        let bias = model.add_const("source_bias", inputs[2].clone().into_arc_tensor())?;

        let mut input_outlets = tvec![a, b, bias];
        for (i, t) in inputs.iter().enumerate().skip(3) {
            input_outlets.push(model.add_const(format!("source_{i}"), t.clone().into_arc_tensor())?)
        }

        let mut params = self.params.as_outlet_ids(&mut model, "qmatmul_unary", &input_outlets)?;

        let a = wire_offset_u8_as_i8(&mut model, "adhoc", a, "a", &mut params[0], "a0")?;
        let b = wire_offset_u8_as_i8(&mut model, "adhoc", b, "b", &mut params[2], "b0")?;

        let new_op = MatMul { axes: self.axes };
        let result = model.wire_node("adhoc.matmul", new_op, &[a, b])?[0];
        let result = wire_matmul_quant(
            &mut model,
            "adhoc",
            a,
            b,
            Some(bias),
            self.axes,
            result,
            self.output_type,
            &params,
        )?;
        model.set_output_outlets(&[result])?;
        model.into_runnable()?.run(tvec![])
    }
}

impl TypedOp for QMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs.len() != 3 + self.params.input_count() {
            bail!(
                "Inconsistent q matmul. expects {} inputs, got {}",
                3 + self.params.input_count(),
                inputs.len()
            );
        }
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let (_m, _k, _n, c_shape) = compute_shape(&inputs[0].shape, &inputs[1].shape, self.axes)?;

        let bias = &inputs[2];
        #[allow(clippy::comparison_chain)]
        if bias.rank() > 1 {
            anyhow::bail!("Bias must be either scalar or vector (rank 0 or 1).");
        } else if bias.rank() == 1 {
            let expected_len = &c_shape[self.axes.c_m];
            anyhow::ensure!(
                &bias.shape[0] == expected_len,
                "got: {:?} expected len: {:?}",
                bias,
                expected_len
            );
        };

        Ok(tvec!(self.output_type.fact(c_shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.axes,
        )
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        assert!(a_fact.rank() == b_fact.rank());
        let mut axes = self.axes.to_axis_mapping(a_fact.rank())?;
        let mut patch = TypedModelPatch::new("QMatMul as Einsum");
        let name = &node.name;
        // a, b, bias
        let mut inputs: Vec<OutletId> = (node.inputs[0..3])
            .iter()
            .map(|i| patch.tap_model(model, *i))
            .collect::<TractResult<Vec<_>>>()?;
        let bias_fact = model.outlet_fact(node.inputs[2])?;
        axes = axes.add_input(bias_fact.rank())?;
        if bias_fact.rank() == 1 {
            axes = axes.with_input_axis_named(2, 0, '$')?.linking('m', '$')?;
        }
        for (param_name, param_value) in self.params.iter() {
            let outlet = match &param_value {
                AttrOrInput::Attr(k) => patch.add_const(format!("{name}.{param_name}"), k.clone())?,
                AttrOrInput::Input(i) => patch.tap_model(model, node.inputs[*i])?,
            };
            inputs.push(outlet);
            axes = axes.add_input(patch.outlet_fact(outlet)?.rank())?;
        }
        let op = EinSum::new(axes, DatumType::I32, Some(self.output_type));
        let output = patch.wire_node(&node.name, op, &inputs)?[0];
        patch.shunt_outside(model, node.id.into(), output)?;
        Ok(Some(patch))
    }

    as_op!();
}

/// Wires the offsetting of a matrix and zero point node.
///
/// Only wires nodes of u8 type and leaves nodes of different type untouched.
pub(crate) fn wire_offset_u8_as_i8(
    model: &mut TypedModel,
    model_name: &str,
    matrix: OutletId,
    matrix_name: &str,
    zero_point: &mut OutletId,
    zero_point_name: &str,
) -> TractResult<OutletId> {
    let fact = model.outlet_fact(matrix)?;
    if let DatumType::U8 = fact.datum_type.unquantized() {
        match model.outlet_fact(*zero_point)?.datum_type.unquantized() {
            DatumType::U8 => {
                *zero_point = model.wire_node(
                    format!("{model_name}.offset_{zero_point_name}_as_i8"),
                    ops::quant::offset_u8_as_i8(),
                    &[*zero_point],
                )?[0];
            }
            DatumType::I32 => {
                let zp_rank = model.outlet_fact(*zero_point)?.rank();
                let cst = model.add_const(
                    format!("{model_name}.offset_{zero_point_name}_as_i8.min"),
                    tensor0(-128i32).broadcast_into_rank(zp_rank)?.into_arc_tensor(),
                )?;
                *zero_point = model.wire_node(
                    format!("{model_name}.offset_{zero_point_name}_as_i8"),
                    ops::math::add(),
                    &[*zero_point, cst],
                )?[0];
            }
            _ => (),
        }
        Ok(model.wire_node(
            format!("{model_name}.offset_{matrix_name}_as_i8"),
            ops::quant::offset_u8_as_i8(),
            &[matrix],
        )?[0])
    } else {
        Ok(matrix)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn wire_matmul_quant(
    model: &mut TypedModel,
    name: &str,
    a: OutletId,
    b: OutletId,
    bias: Option<OutletId>,
    axes: MatMulAxes,
    mut result: OutletId,
    output_type: DatumType,
    params: &[OutletId],
) -> TractResult<OutletId> {
    let b_fact = model.outlet_fact(b)?.clone();
    // TODO: assumed c_rank == b_rank (== a_rank)

    if let Some(mut bias) = bias {
        // bias is scalar -> ok
        // bias is vec, m is right in C -> broadcast will add left side axes to bias
        // bias is vec, m is not right in C -> we must append in C axes to the right to align them
        let bias_rank = model.outlet_fact(bias)?.rank();
        if bias_rank == 1 && axes.c_m < b_fact.rank() - 1 {
            for i in 0..(b_fact.rank() - axes.c_m - 1) {
                bias = model.wire_node(
                    format!("{name}.axis_rank_fix.{i}"),
                    AxisOp::Add(bias_rank + i),
                    &[bias],
                )?[0]
            }
        }
        result = wire_with_rank_broadcast(
            &format!("{}.add_bias", &name),
            model,
            ops::math::add(),
            &[result, bias],
        )?[0];
    }

    let k = model.outlet_fact(a)?.shape[axes.a_k].clone();

    let abc_scale = combine_scales(model, name, params[1], params[3], params[5])?;

    let a_i32 =
        model.wire_node(format!("{name}.a_as_i32"), ops::cast::cast(i32::datum_type()), &[a])?[0];
    let b_i32 =
        model.wire_node(format!("{name}.b_as_i32"), ops::cast::cast(i32::datum_type()), &[b])?[0];
    let sum_a = model.wire_node(
        format!("{name}.sum_a"),
        ops::nn::Reduce::new(tvec!(axes.a_k), ops::nn::Reducer::Sum),
        &[a_i32],
    )?[0];
    let sum_a =
        model.wire_node(format!("{name}.sum_a_rm_k_axis"), AxisOp::Rm(axes.a_k), &[sum_a])?[0];
    let sum_a =
        model.wire_node(format!("{name}.sum_a_add_n_axis"), AxisOp::Add(axes.c_n), &[sum_a])?[0];
    let sum_b = model.wire_node(
        format!("{name}.sum_b"),
        ops::nn::Reduce::new(tvec!(axes.b_k), ops::nn::Reducer::Sum),
        &[b_i32],
    )?[0];
    let sum_b =
        model.wire_node(format!("{name}.sum_b_rm_k_axis"), AxisOp::Rm(axes.b_k), &[sum_b])?[0];
    let sum_b =
        model.wire_node(format!("{name}.sum_a_add_m_axis"), AxisOp::Add(axes.c_m), &[sum_b])?[0];
    let result =
        compensate_zero_points(model, name, result, k, params[0], params[2], sum_a, sum_b)?;
    requant(model, name, result, output_type, abc_scale, params[4])
}

pub(crate) fn combine_scales(
    model: &mut TypedModel,
    name: &str,
    a_scale: OutletId,
    b_scale: OutletId,
    c_scale: OutletId,
) -> TractResult<OutletId> {
    let ab_scale = wire_with_rank_broadcast(
        &format!("{name}.ab_scale"),
        model,
        ops::math::mul(),
        &[a_scale, b_scale],
    )?[0];
    let abc_scale = wire_with_rank_broadcast(
        &format!("{name}.abc_scales"),
        model,
        ops::math::div(),
        &[ab_scale, c_scale],
    )?[0];
    Ok(abc_scale)
}

#[allow(clippy::too_many_arguments)]
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
    let output_rank = model.outlet_fact(result)?.rank();
    ensure!(model.outlet_fact(sum_a)?.rank() == output_rank);
    ensure!(model.outlet_fact(sum_b)?.rank() == output_rank);

    let a0 =
        model.wire_node(format!("{name}.cast_a0"), ops::cast::cast(i32::datum_type()), &[a0])?[0];

    let b0 =
        model.wire_node(format!("{name}.cast_b0"), ops::cast::cast(i32::datum_type()), &[b0])?[0];

    let k = model.add_const(format!("{name}.k"), rctensor0(k))?;
    let k = model.wire_node(format!("{name}.cast_k"), ops::cast::cast(i32::datum_type()), &[k])?[0];

    let a0_sum_b = wire_with_rank_broadcast(
        &format!("{name}.a0_sum_b"),
        model,
        ops::math::mul(),
        &[a0, sum_b],
    )?[0];

    let b0_sum_a = wire_with_rank_broadcast(
        &format!("{name}.b0_sum_a"),
        model,
        ops::math::mul(),
        &[b0, sum_a],
    )?[0];

    let a0_k =
        wire_with_rank_broadcast(&format!("{name}.a0_k"), model, ops::math::mul(), &[a0, k])?[0];

    let a0_k_b0 =
        wire_with_rank_broadcast(&format!("{name}.a0_k_b0"), model, ops::math::mul(), &[a0_k, b0])?
            [0];

    let result = wire_with_rank_broadcast(
        &format!("{}.minus_a0_B", &name),
        model,
        ops::math::sub(),
        &[result, a0_sum_b],
    )?[0];
    let result = wire_with_rank_broadcast(
        &format!("{}.minus_b0_A", &name),
        model,
        ops::math::sub(),
        &[result, b0_sum_a],
    )?[0];

    let result = wire_with_rank_broadcast(
        &format!("{}.plus_a0_k_b0", &name),
        model,
        ops::math::add(),
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
    let wire = wire_with_rank_broadcast(
        &format!("{name}.scale"),
        model,
        ops::quant::scale(),
        &[scale, wire],
    )?[0];

    let zero_point = model.wire_node(
        format!("{name}.cast_c0"),
        ops::cast::cast(i32::datum_type()),
        &[zero_point],
    )?[0];

    let wire = wire_with_rank_broadcast(
        &format!("{name}.zeropoint"),
        model,
        ops::math::add(),
        &[wire, zero_point],
    )?[0];

    clamp_and_cast_to(model, name, dt, wire)
}

pub(crate) fn clamp_and_cast_to(
    model: &mut TypedModel,
    name: &str,
    dt: DatumType,
    wire: OutletId,
) -> TractResult<OutletId> {
    if dt == i32::datum_type() {
        return Ok(wire);
    }
    let rank = model.outlet_fact(wire)?.rank();
    let inf = dt
        .unquantized()
        .min_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let inf = model.add_const(format!("{name}.min.const"), inf)?;
    let sup = dt
        .unquantized()
        .max_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let sup = model.add_const(format!("{name}.max.const"), sup)?;
    let wire = model.wire_node(format!("{name}.min"), ops::math::min(), &[wire, sup])?;
    let wire = model.wire_node(format!("{name}.max"), ops::math::max(), &[wire[0], inf])?;
    let wire = model.wire_node(format!("{name}.cast"), ops::cast::cast(dt), &wire)?;
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
        fn prop_i8_i8_i8(pb in any::<QMatMulProblemI8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_i8_u8(pb in any::<QMatMulProblemI8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_i8(pb in any::<QMatMulProblemI8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_i8(pb in any::<QMatMulProblemU8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_u8(pb in any::<QMatMulProblemI8U8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_u8(pb in any::<QMatMulProblemU8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_i8(pb in any::<QMatMulProblemU8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_u8(pb in any::<QMatMulProblemU8U8U8>()) {
            pb.check();
        }
    }

    #[test]
    fn c0() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn b_scale() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
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
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            bias: tensor0(0i32),
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
        QMatMulProblemI8I8I8 {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
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
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            bias: tensor0(0i32),
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
        QMatMulProblemI8I8I8 {
            a: arr2(&[[47], [0]]),
            b: arr2(&[[1, 0, 30]]),
            bias: tensor0(0i32),
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
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-30]]),
            b: arr2(&[[0, 107, 0]]),
            bias: tensor0(0i32),
            a0: -59,
            b0: 117,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.15,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn onnx_test_matmulinteger() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            bias: tensor0(0i32),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    fn round_ties_to_right(x: f32) -> i32 {
        (x + 0.5).floor() as i32
    }

    fn scale() -> BoxedStrategy<f32> {
        prop_oneof![Just(1.0), (1i32..=20).prop_map(|x| x as f32 / 20.0)].boxed()
    }

    macro_rules! impl_qmmp {
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
            }

            impl $name {
                fn check(&self) {
                    let check_with = |r: &Array2<$c>, opt: bool, qp: bool| {
                        let t = self.tract(opt, qp);
                        assert!(
                            r.iter().zip(t.iter()).all(|(r, t)| r.max(t) - r.min(t) <= 1),
                            "mismatch! optimized plan: {}, dynamic qparams: {}, reference: {:?}, tract: {:?}",
                            opt,
                            qp,
                            r,
                            t,
                            );
                    };

                    let r = self.reference();
                    check_with(&r, false, false);
                    check_with(&r, false, true);
                    check_with(&r, true, false);
                    check_with(&r, true, true);
                }

                fn reference(&self) -> Array2<$c> {
                    let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
                    let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
                    let c = a.dot(&b);
                    let c = c.map(|&x| round_ties_to_right(x / self.c_scale) + self.c0 as i32);
                    c.map(|&x| x.max(<$c>::MIN as i32).min(<$c>::MAX as i32) as $c)
                }

                fn tract(&self, opt: bool, qp: bool) -> Array2<$c> {
                    let mut model = TypedModel::default();
                    let mut inputs = tvec![];
                    inputs.push(
                        model
                        .add_source("a", <$a>::fact( &[self.a.nrows(), self.a.ncols()]))
                        .unwrap(),
                        );
                    inputs.push(
                        model
                        .add_source("b", <$b>::fact(&[self.b.nrows(), self.b.ncols()]))
                        .unwrap(),
                        );
                    inputs.push(
                        model
                        .add_source("bias", i32::fact(self.bias.shape()))
                        .unwrap(),
                        );
                    let qparams = if qp {
                        inputs.push(model.add_source("a0", TypedFact::scalar::<$a>()).unwrap());
                        inputs
                            .push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("b0", TypedFact::scalar::<$b>()).unwrap());
                        inputs
                            .push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("c0", TypedFact::scalar::<$c>()).unwrap());
                        inputs
                            .push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
                        MatMulQParams::all_dynamic(3)
                    } else {
                        MatMulQParams {
                            a0: rctensor0::<$a>(self.a0).into(),
                            a_scale: rctensor0::<f32>(self.a_scale).into(),
                            b0: rctensor0::<$b>(self.b0).into(),
                            b_scale:rctensor0::<f32>(self.b_scale).into(),
                            c0: rctensor0::<$c>(self.c0).into(),
                            c_scale:rctensor0::<f32>(self.c_scale).into(),
                        }
                    };
                    let result = model
                        .wire_node(
                            "qmm",
                            QMatMul::new(MatMulAxes::default(), <$c>::datum_type(), qparams),
                            &inputs,
                            )
                        .unwrap();
                    model.set_output_outlets(&result).unwrap();

                    let inputs = if qp {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                            self.a0.into(),
                            self.a_scale.into(),
                            self.b0.into(),
                            self.b_scale.into(),
                            self.c0.into(),
                            self.c_scale.into(),
                        ]
                    } else {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                        ]
                    };
                    let inputs = inputs.into_iter().map(|t| t.into_tvalue()).collect();
                    let mut outputs = if opt { model.into_optimized().unwrap() } else { model }
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
                                )
                        })
                    .prop_map(|((m, k, n), a, b, a0, b0, c0, a_scale, b_scale, c_scale)| {
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
                        }
                    })
                    .boxed()
                }
            }
        };
    }

    impl_qmmp! { QMatMulProblemI8I8I8, i8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8I8U8, i8, i8, u8 }
    impl_qmmp! { QMatMulProblemI8U8I8, i8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8I8I8, u8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8U8U8, i8, u8, u8 }
    impl_qmmp! { QMatMulProblemU8I8U8, u8, i8, u8 }
    impl_qmmp! { QMatMulProblemU8U8I8, u8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8U8U8, u8, u8, u8 }

    #[test]
    fn test_qmmp_i8_i8_i8() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_i8_u8() {
        QMatMulProblemI8I8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_i8_u8_i8() {
        QMatMulProblemI8U8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_i8_i8() {
        QMatMulProblemU8I8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_u8_u8() {
        QMatMulProblemI8U8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_i8_u8() {
        QMatMulProblemU8I8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_u8_i8() {
        QMatMulProblemU8U8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_u8_u8() {
        QMatMulProblemU8U8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    fn setup_qparams(inputs: [usize; 6]) -> ([OutletId; 9], MatMulQParams, TypedModel, OutletId) {
        let mut model = TypedModel::default();
        let ids = [
            model.add_source("a", i8::fact([2, 3])).unwrap(),
            model.add_source("b", i8::fact([3, 4])).unwrap(),
            model.add_const("bias", Tensor::zero_scalar::<i32>().unwrap()).unwrap(),
            model.add_const("a0", Tensor::zero_scalar::<i8>().unwrap()).unwrap(),
            model.add_const("a_scale", Tensor::zero_scalar::<f32>().unwrap()).unwrap(),
            model.add_source("b0", i8::scalar_fact()).unwrap(),
            model.add_source("b_scale", f32::scalar_fact()).unwrap(),
            model.add_const("c0", Tensor::zero_scalar::<i8>().unwrap()).unwrap(),
            model.add_const("c_scale", Tensor::zero_scalar::<f32>().unwrap()).unwrap(),
        ];
        let indices = inputs
            .iter()
            .enumerate()
            .sorted_by(|(_, this), (_, other)| this.cmp(other))
            .map(|(idx, _)| idx + 3)
            .collect::<Vec<_>>();
        let qparams = MatMulQParams {
            a0: indices[0].into(),
            a_scale: indices[1].into(),
            b0: indices[2].into(),
            b_scale: indices[3].into(),
            c0: indices[4].into(),
            c_scale: indices[5].into(),
        };
        let op = QMatMul {
            axes: MatMulAxes::default(),
            output_type: i8::datum_type(),
            params: qparams.clone(),
        };
        let node = model
            .wire_node(
                "qmatmul",
                op,
                &[
                    ids[0],
                    ids[1],
                    ids[2],
                    ids[inputs[0]],
                    ids[inputs[1]],
                    ids[inputs[2]],
                    ids[inputs[3]],
                    ids[inputs[4]],
                    ids[inputs[5]],
                ],
            )
            .unwrap()[0];

        (ids, qparams, model, node)
    }

    #[test]
    fn test_qparams_inline_ascending() {
        let (ids, qparams, model, node) = setup_qparams([3, 4, 5, 6, 7, 8]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[5], ids[6]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: AttrOrInput::Attr(_),
                a_scale: AttrOrInput::Attr(_),
                b0: AttrOrInput::Input(3),
                b_scale: AttrOrInput::Input(4),
                c0: AttrOrInput::Attr(_),
                c_scale: AttrOrInput::Attr(_),
            },
        ));
    }

    #[test]
    fn test_qparams_inline_descending() {
        let (ids, qparams, model, node) = setup_qparams([8, 7, 6, 5, 4, 3]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[6], ids[5]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: AttrOrInput::Attr(_),
                a_scale: AttrOrInput::Attr(_),
                b0: AttrOrInput::Input(4),
                b_scale: AttrOrInput::Input(3),
                c0: AttrOrInput::Attr(_),
                c_scale: AttrOrInput::Attr(_),
            },
        ));
    }

    #[test]
    fn test_qparams_inline_mixed() {
        let (ids, qparams, model, node) = setup_qparams([5, 3, 8, 4, 7, 6]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[5], ids[6]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: AttrOrInput::Attr(_),
                a_scale: AttrOrInput::Attr(_),
                b0: AttrOrInput::Input(3),
                b_scale: AttrOrInput::Input(4),
                c0: AttrOrInput::Attr(_),
                c_scale: AttrOrInput::Attr(_),
            },
        ));
    }
}
