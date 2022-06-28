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

/*
impl MatMul {
    pub fn with_a_trans(self, a_trans: bool) -> MatMul {
        std::mem::swap(&mut self.axes.a_k, &mut self.axes.a_m);
        self
    }

    pub fn with_b_trans(self, b_trans: bool) -> MatMul {
        std::mem::swap(&mut self.axes.b_k, &mut self.axes.b_n);
        self
    }

    pub fn with_c_trans(self, c_trans: bool) -> MatMul {
        std::mem::swap(&mut self.axes.c_m, &mut self.axes.c_n);
        self
    }
}
*/

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if &inputs[0].rank() != &inputs[1].rank() {
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

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        return Ok(None);
        //         let a_fact = model.outlet_fact(node.inputs[0])?;
        //         let b_fact = model.outlet_fact(node.inputs[1])?;
        //         let konst_ix = if a_fact.konst.is_some() {
        //             0
        //         } else if b_fact.konst.is_some() {
        //             1
        //         } else {
        //             return Ok(None);
        //         };
        //
        //         let var_ix = 1 - konst_ix;
        //         let flip = konst_ix == 1;
        //         let t_konst = [self.a_trans, self.b_trans][konst_ix] ^ flip;
        //         let t_var = [self.b_trans, self.a_trans][konst_ix] ^ flip;
        //         let konst = model.outlet_fact(node.inputs[konst_ix])?.konst.clone().unwrap();
        //         TypedModelPatch::replace_single_op(
        //             model,
        //             node,
        //             &node.inputs[var_ix..][..1],
        //             MatMulUnary::new(konst, t_konst, t_var, self.c_trans ^ flip),
        //         )
        //         .map(Some)
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
        let op = MatMul::default().with_a_trans(true).with_b_trans(true).with_c_trans(true);
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
            MatMulUnary { a, axes: MatMulAxes::default().transposing(true, true, true) },
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
