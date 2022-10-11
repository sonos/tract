use crate::internal::*;
use crate::ops::matmul::*;

/// The binary op. It will declutter to MatMulUnary if either A or B is constant.
///
/// TODO: implemnent TypedOp fully to play nice with optimizer.
/// TODO: codegen fails if A and B are variable inputs.
#[derive(Debug, Clone, Default, Hash)]
pub struct MatMul {
    pub axes: MatMulAxes,
}

impl_dyn_hash!(MatMul);

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

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], inputs[1]);
        }
        Ok(tvec!(eval(&inputs[0], &inputs[1], self.axes)?.into_arc_tensor()))
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

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        let konst_ix = if a_fact.konst.is_some() {
            0
        } else if b_fact.konst.is_some() {
            1
        } else {
            return Ok(None);
        };

        let var_ix = 1 - konst_ix;
        let flip = konst_ix == 1;

        let var_fact: &TypedFact = [a_fact, b_fact][var_ix];
        let konst_fact = [a_fact, b_fact][konst_ix];

        let b_shape = if let Some(var_shape) = var_fact.shape.as_concrete() {
            var_shape
        } else {
            return Ok(None);
        };

        let axes = if flip {
            MatMulAxes {
                a_m: self.axes.b_n,
                a_k: self.axes.b_k,
                b_n: self.axes.a_m,
                b_k: self.axes.a_k,
                c_m: self.axes.c_n,
                c_n: self.axes.c_m,
            }
        } else {
            self.axes.clone()
        };

        let konst = konst_fact.konst.as_ref().unwrap();
        crate::ops::matmul::mir_unary::new_mat_mul_unary_finite(
            model,
            node,
            konst,
            b_shape,
            var_fact.datum_type,
            &axes,
        )
        .map(Some)
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
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a, b)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul { axes: MatMulAxes::default().transposing(true, true, true) };
        let c_found = op.eval(tvec!(b, a)).unwrap().pop().unwrap();
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
        wire = model.wire_node("a", crate::ops::math::add::unary(b), &wire)?;
        model.set_output_outlets(&wire)?;
        let input = Tensor::zero::<f32>(&input_shape)?;
        trace!("running mir");
        model.clone().into_runnable()?.run(tvec!(input.clone()))?;
        trace!("running optimized");
        model.into_decluttered()?.into_optimized()?.into_runnable()?.run(tvec!(input))?;
        Ok(())
    }
}
