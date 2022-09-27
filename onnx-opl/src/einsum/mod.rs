use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_ndarray::{Axis, Dimension};
use tract_nnef::tract_num_traits::{One, Zero};

mod expr;
pub use expr::Expr;
mod to_matmul;

#[derive(Debug, Clone, Hash)]
pub struct EinSum {
    pub expr: Expr,
}

impl_dyn_hash!(EinSum);

impl Op for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.expr)])
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EinSum {
    fn new(expr: Expr) -> EinSum {
        EinSum { expr }
    }

    fn output_shape<D: DimLike>(&self, inputs: &[&[D]]) -> TVec<D> {
        self.expr
            .index
            .iter()
            .sorted_by_key(|axis| axis.result.unwrap())
            .map(|axis| {
                axis.inputs
                    .iter()
                    .enumerate()
                    .flat_map(|(input_id, positions)| {
                        positions.iter().map(move |p| inputs[input_id][*p].clone())
                    })
                    .find(|x| x != &1.into())
                    .unwrap_or_else(|| 1.into())
            })
            .collect()
    }

    fn eval_t<T: Datum + Zero + One>(
        &self,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let shapes: TVec<_> = inputs.iter().map(|t| t.shape()).collect();
        let output_shape = self.output_shape(&shapes);
        let inputs: TVec<tract_ndarray::ArrayViewD<T>> =
            inputs.iter().map(|t| t.to_array_view::<T>()).collect::<TractResult<_>>()?;
        let summing_shape: TVec<usize> = self
            .expr
            .sum
            .iter()
            .map(|axis| {
                axis.inputs
                    .iter()
                    .enumerate()
                    .find_map(|(input_id, positions)| {
                        if positions.len() > 0 {
                            Some(inputs[input_id].shape()[positions[0]])
                        } else {
                            None
                        }
                    })
                    .unwrap()
            })
            .collect();
        let output = tract_ndarray::ArrayD::<T>::from_shape_fn(&*output_shape, |coords| {
            let coords = coords.as_array_view();
            let mut views = inputs.clone();
            for (axis, x) in
                self.expr.index.iter().sorted_by_key(|axis| axis.result.unwrap()).zip(coords)
            {
                for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                    for position in input_axis_positions {
                        let x = if views[input_id].shape()[*position] == 1 { 0 } else { *x };
                        views[input_id]
                            .slice_axis_inplace(tract_ndarray::Axis(*position), (x..=x).into());
                    }
                }
            }
            let mut sum: T = T::zero();
            for sum_coords in tract_ndarray::indices(&*summing_shape) {
                let mut views = views.clone();
                let sum_coords = sum_coords.as_array_view();
                for (axis, x) in self.expr.sum.iter().zip(&sum_coords) {
                    for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                        for position in input_axis_positions {
                            views[input_id].slice_axis_inplace(Axis(*position), (*x..=*x).into())
                        }
                    }
                }
                let mut product = T::one();
                for v in &views {
                    debug_assert_eq!(v.len(), 1);
                    product = product * v.iter().next().unwrap().clone();
                }
                sum = sum + product;
            }
            sum
        });
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl EvalOp for EinSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_numbers!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl TypedOp for EinSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.iter().enumerate().all(|(ix, fact)| fact.rank() == self.expr.input_rank(ix)));
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, self.output_shape(&*shapes))))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        to_matmul::declutter(self, model, node)
    }

    as_op!();
}

pub fn parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().array().named("inputs"), TypeName::String.named("expr")]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let einsum = node.op_as::<EinSum>().unwrap();
    let inputs = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect();
    Ok(Some(invocation(
        "tract_onnx_einsum",
        &[Arc::new(RValue::Array(inputs))],
        &[("expr", string(einsum.expr.to_string()))],
    )))
}

pub fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let expr = invocation.named_arg_as::<String>(builder, "expr")?.parse::<Expr>()?;
    let einsum = EinSum { expr };
    let inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    builder.wire(einsum, &inputs)
}
