use tract_data::itertools::izip;

use crate::internal::*;
use crate::ops::einsum::{Axis, EinSum, Expr};
use crate::ops::matmul::*;

/// The binary op. It will declutter to MatMulUnary if either A or B is constant.
#[derive(Debug, Clone, Default, Hash)]
pub struct MatMul {
    pub axes: MatMulAxes,
}



impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], inputs[1]);
        }
        Ok(tvec!(eval(&inputs[0], &inputs[1], self.axes)?.into()))
    }
}

impl TypedOp for MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let (_m, _k, _n, c_shape) = compute_shape(&inputs[0].shape, &inputs[1].shape, self.axes)?;
        Ok(tvec!(output_type(inputs[0].datum_type).fact(c_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;

        assert!(a_fact.rank() == b_fact.rank());

        let k_axis = Axis::new('k').input(0, self.axes.a_k).input(1, self.axes.b_k);
        let m_axis = Axis::new('m').input(0, self.axes.a_m).result(self.axes.c_m);
        let n_axis = Axis::new('n').input(1, self.axes.b_n).result(self.axes.c_n);
        let rank = a_fact.rank();
        fn remaining(rank: usize, skip1: usize, skip2: usize) -> impl Iterator<Item = usize> {
            (0..rank).filter(move |&i| i != skip1 && i != skip2)
        }
        let remain_a = remaining(rank, self.axes.a_k, self.axes.a_m);
        let remain_b = remaining(rank, self.axes.b_k, self.axes.b_n);
        let remain_c = remaining(rank, self.axes.c_m, self.axes.c_n);
        let alphabet = ('a'..).filter(|&c| c != 'k' && c != 'm' && c != 'n');
        let extra_axes = izip!(alphabet, remain_a, remain_b, remain_c)
            .map(|(letter, a, b, c)| Axis::new(letter).input(0, a).input(1, b).result(c));
        let expr: Expr = extra_axes.chain([k_axis, m_axis, n_axis].into_iter()).collect();
        TypedModelPatch::replace_single_op(model, node, &node.inputs, EinSum::new(expr, output_type(a_fact.datum_type))).map(Some)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        super::cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.axes,
        )
    }

    as_op!();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        //           0
        //           1
        //           2
        //
        // 0 1 2     5
        // 3 4 5    14
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a.into(), b.into())).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul { axes: MatMulAxes::default().transposing(true, true, true) };
        let c_found = op.eval(tvec!(b.into(), a.into())).unwrap().pop().unwrap();

        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn batch_input() -> TractResult<()> {
        crate::setup_test_logger();
        let (batch, len, ci, co) = (2, 3, 4, 5);
        let mut model = TypedModel::default();
        let input_shape = tvec!(batch, len, ci);
        let mut wire = tvec!(model.add_source("s", f32::fact(&*input_shape))?);
        let mut a = Tensor::zero::<f32>(&[1, ci, co])?;
        a.as_slice_mut::<f32>().unwrap()[0] = 1.0;
        let a = a.into_arc_tensor();
        wire = model.wire_node(
            "m",
            MatMulUnary { a, axes: MatMulAxes::default_for_rank(3).transposing(true, true, true) },
            &wire,
        )?;
        let mut b = Tensor::zero::<f32>(&[1, 1, co])?;
        b.as_slice_mut::<f32>().unwrap()[0] = 1.0;
        let b = b.into_arc_tensor();
        let b = model.add_const("b", b)?;
        wire = model.wire_node("a", crate::ops::math::add(), &[wire[0], b])?;
        model.set_output_outlets(&wire)?;
        let input = Tensor::zero::<f32>(&input_shape)?.into_tvalue();
        trace!("running mir");
        model.clone().into_runnable()?.run(tvec!(input.clone()))?;
        trace!("running optimized");
        model.into_decluttered()?.into_optimized()?.into_runnable()?.run(tvec!(input))?;
        Ok(())
    }
}
