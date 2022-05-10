use std::iter::FromIterator;
use std::str::FromStr;

use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_ndarray::{Axis, Dimension};
use tract_hir::tract_num_traits::{One, Zero};

pub fn einsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let expr = node.get_attr::<String>("equation")?.parse()?;
    Ok((Box::new(EinSum { expr }), vec![]))
}

#[derive(Debug, Clone, PartialEq, Default, Hash)]
struct AxisSym {
    result: Option<usize>,
    inputs: TVec<TVec<usize>>,
}

impl AxisSym {
    fn result(axis: usize) -> AxisSym {
        AxisSym { result: Some(axis), inputs: tvec!() }
    }

    fn no_result() -> AxisSym {
        AxisSym { result: None, inputs: tvec!() }
    }

    fn set_result(&mut self, axis: usize) {
        self.result = Some(axis)
    }

    #[allow(dead_code)]
    fn input(mut self, input_id: usize, axis: usize) -> AxisSym {
        self.add_input(input_id, axis);
        self
    }

    fn add_input(&mut self, input_id: usize, axis: usize) {
        if self.inputs.len() <= input_id {
            self.inputs.resize(input_id + 1, tvec!())
        }
        self.inputs[input_id].push(axis);
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
struct Expr {
    index: TVec<AxisSym>,
    sum: TVec<AxisSym>,
}

impl Expr {
    fn iter_all_axes(&self) -> impl Iterator<Item = &AxisSym> {
        self.index.iter().chain(self.sum.iter())
    }

    fn n_inputs(&self) -> usize {
        self.iter_all_axes().map(|axis| axis.inputs.len()).max().unwrap()
    }
}

impl FromIterator<AxisSym> for Expr {
    fn from_iter<T: IntoIterator<Item = AxisSym>>(iter: T) -> Self {
        let (index, sum) = iter.into_iter().partition(|ax| ax.result.is_some());
        Expr { index, sum }
    }
}

impl<I: IntoIterator<Item = AxisSym>> From<I> for Expr {
    fn from(it: I) -> Self {
        it.into_iter().collect()
    }
}

impl FromStr for Expr {
    type Err = TractError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        assert!(!s.contains("..."));
        let s = s.replace(" ", "");
        let (inputs, result) =
            if let Some((i, r)) = s.split_once("->") { (i, Some(r)) } else { (&*s, None) };
        let inputs: TVec<&str> = inputs.split(",").collect();
        let mut axes = HashMap::<char, AxisSym>::default();
        if let Some(result) = result {
            for (ix, axis) in result.chars().enumerate() {
                axes.insert(axis, AxisSym::result(ix));
            }
        }
        for (input_ix, input) in inputs.iter().enumerate() {
            for (ix, axis) in input.chars().enumerate() {
                axes.entry(axis).or_insert(AxisSym::no_result()).add_input(input_ix, ix);
            }
        }
        if result.is_none() {
            axes.iter_mut()
                .sorted_by_key(|(k, _)| *k)
                .filter(|(_, v)| v.inputs.iter().map(|input| input.len()).sum::<usize>() == 1)
                .enumerate()
                .for_each(|(ix, (_, v))| v.set_result(ix))
        }
        Ok(axes.into_iter().sorted_by_key(|(k, _)| *k).map(|(_, v)| v).collect::<Expr>())
    }
}

#[derive(Debug, Clone, Hash)]
struct EinSum {
    expr: Expr,
}

impl_dyn_hash!(EinSum);

impl Op for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.expr)])
    }

    op_onnx!();
    op_as_typed_op!();
}

impl EinSum {
    fn output_shape<D: DimLike>(&self, inputs: &[&[D]]) -> TVec<D> {
        self.expr
            .index
            .iter()
            .sorted_by_key(|axis| axis.result.unwrap())
            .map(|axis| {
                axis.inputs
                    .iter()
                    .enumerate()
                    .find_map(|(input_id, positions)| {
                        if positions.len() > 0 {
                            Some(inputs[input_id][positions[0]].clone())
                        } else {
                            None
                        }
                    })
                    .unwrap()
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
                        views[input_id]
                            .slice_axis_inplace(tract_ndarray::Axis(*position), (*x..=*x).into());
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
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, self.output_shape(&*shapes))))
    }

    as_op!();
}

impl InferenceRulesOp for EinSum {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, self.expr.n_inputs())?;
        check_output_arity(outputs, 1)?;
        for i in inputs {
            s.equals(&i.datum_type, &outputs[0].datum_type)?;
        }
        for (input_id, input) in inputs.iter().enumerate() {
            let rank = self
                .expr
                .iter_all_axes()
                .flat_map(|axis| axis.inputs.get(input_id).map(|v| &**v).unwrap_or(&[]).iter())
                .max()
                .unwrap();
            s.equals(1 + *rank as i64, &input.rank)?;
        }
        let output_rank = self.expr.index.len();
        s.equals(output_rank as i64, &outputs[0].rank)?;
        for axis in self.expr.iter_all_axes() {
            let mut axes = vec![];
            if let Some(result) = axis.result {
                axes.push(outputs[0].shape[result].bex())
            }
            for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                for position in input_axis_positions {
                    axes.push(inputs[input_id].shape[*position].bex());
                }
            }
            s.equals_all(axes)?;
        }
        Ok(())
    }

    to_typed!();
    as_op!();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_expr_builder() {
        assert_eq!(
            Expr::from(tvec![AxisSym::result(0).input(0, 1), AxisSym::result(1).input(0, 0)]),
            Expr {
                index: tvec!(AxisSym::result(0).input(0, 1), AxisSym::result(1).input(0, 0)),
                sum: tvec!(),
            }
        )
    }

    #[test]
    fn test_parse_transpose() {
        assert_eq!(
            "ij->ji".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(1).input(0, 0), AxisSym::result(0).input(0, 1)]),
        )
    }

    #[test]
    fn test_parse_diag() {
        assert_eq!(
            "ii->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0).input(0, 1)]),
        )
    }

    #[test]
    fn test_parse_adamar_product_explicit() {
        assert_eq!(
            "i,i->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0).input(1, 0)]),
        )
    }

    #[test]
    fn test_parse_inner_product_implicit() {
        assert_eq!("i,i".parse::<Expr>().unwrap(), "i,i->".parse::<Expr>().unwrap(),)
    }

    #[test]
    fn test_parse_batch_matmul() {
        assert_eq!(
            "bij , bjk -> bik ".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0).input(1, 0),
                AxisSym::result(1).input(0, 1),
                AxisSym::no_result().input(0, 2).input(1, 1),
                AxisSym::result(2).input(1, 2)
            ])
        )
    }

    #[test]
    fn test_parse_outer_product() {
        assert_eq!(
            "i,j->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0), AxisSym::result(1).input(1, 0)]),
        )
    }

    #[test]
    fn test_parse_bilinear() {
        assert_eq!(
            "ik,jkl,il->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0).input(2, 0),
                AxisSym::result(1).input(1, 0),
                AxisSym::no_result().input(0, 1).input(1, 1),
                AxisSym::no_result().input(1, 2).input(2, 1)
            ]),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction() {
        assert_eq!(
            "pqrs,tuqvr->pstuv".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0),
                AxisSym::no_result().input(0, 1).input(1, 2),
                AxisSym::no_result().input(0, 2).input(1, 4),
                AxisSym::result(1).input(0, 3),
                AxisSym::result(2).input(1, 0),
                AxisSym::result(3).input(1, 1),
                AxisSym::result(4).input(1, 3),
            ]),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction_implicit() {
        assert_eq!(
            "pqrs,tuqvr".parse::<Expr>().unwrap(),
            "pqrs,tuqvr->pstuv".parse::<Expr>().unwrap(),
        )
    }
}
