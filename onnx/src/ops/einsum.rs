use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::infer::rules::expr::ConstantExp;
use tract_hir::internal::*;

// use tract_onnx_opl::einsum::EinSum;
use tract_onnx_opl::einsum::Expr;

pub fn einsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let expr = node.get_attr::<String>("equation")?;
    let expr = expr.replace("...", "*").parse()?;
    Ok((expand(EinSum { expr }), vec![]))
}

#[derive(Debug, Clone, Hash)]
pub struct EinSum {
    pub expr: Expr,
}

impl_dyn_hash!(EinSum);

impl Expansion for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    op_onnx!();

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        todo!()
    }

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
        let elipsis_rank = IntFactoid::default();
        for (input_id, input) in inputs.iter().enumerate() {
            let raw_rank =
                self.expr.iter_all_axes().flat_map(|axis| &axis.inputs[input_id]).max().unwrap()
                    + 1;
            let rank = if self
                .expr
                .iter_all_axes()
                .any(|axis| axis.repr == '*' && axis.inputs[input_id].len() == 1)
            {
                elipsis_rank + raw_rank as i64 + (-1i64)
            } else {
                GenericFactoid::Only(raw_rank as i64)
            };
            s.equals(rank, &input.rank)?;
        }
        let raw_output_rank = self.expr.output_rank();
        let output_rank = if self.expr.index().iter().any(|axis| axis.repr == '*') {
            elipsis_rank + raw_output_rank as i64 + (-1i64)
        } else {
            GenericFactoid::Only(raw_output_rank as i64)
        };
        s.equals(output_rank, &outputs[0].rank)?;

        if !self.expr.iter_all_axes().any(|axis| axis.repr == '*') {
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
        } else {
            s.given(elipsis_rank, move |s, elipsis_rank| {
                let output_elipsis_position =
                    self.expr.index().iter().position(|axis| axis.repr == '*');
                let input_elipsis_positions: Vec<Option<usize>> = (0..self.expr.n_inputs())
                    .map(|input_id| {
                        self.expr
                            .iter_all_axes()
                            .position(|axis| axis.repr == '*' && axis.inputs[input_id].len() == 1)
                    })
                    .collect();
                for axis in self.expr.iter_all_axes() {
                    if axis.repr == '*' {
                        todo!()
                    } else {
                        let mut axes = vec![];
                        if let Some(mut result) = axis.result {
                            if let Some(pos) = output_elipsis_position {
                                if pos < result {
                                    result = result + elipsis_rank as usize - 1;
                                }
                            }
                            axes.push(outputs[0].shape[result].bex())
                        }
                        for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                            for position in input_axis_positions {
                                let mut position = *position;
                                if let Some(elipsis_pos) = input_elipsis_positions[input_id] {
                                    if elipsis_pos < position {
                                        position = position + elipsis_rank as usize - 1;
                                    }
                                }
                                axes.push(inputs[input_id].shape[position].bex());
                            }
                        }
                        s.equals_all(axes)?;
                    }
                }
                Ok(())
            })?;
        }

        /*
        let output_elipsis_position = self.expr.index().iter().position(|axis| axis.repr == '*');
        let input_elipsis_positions: Vec<Option<usize>> = (0..self.expr.n_inputs())
            .map(|input_id| {
                self.expr
                    .iter_all_axes()
                    .position(|axis| axis.repr == '*' && axis.inputs[input_id].len() == 1)
            })
            .collect();
        for axis in self.expr.iter_all_axes() {
            if axis.repr == '*' {
                todo!()
            } else {
                let mut axes = vec![];
                if let Some(mut result) = axis.result {
                    if let Some(pos) = output_elipsis_position {
                        if pos < result {
                            s.given(elipsis_rank, move |s, elipsis_rank| {
                                result = result + elipsis_rank as usize - 1;
                                axes.push(outputs[0].shape[result].bex());
                                Ok(())
                            })?
                        }
                    }
                    axes.push(outputs[0].shape[result].bex())
                }
                for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                    for position in input_axis_positions {
                        let mut position = *position;
                        if let Some(elipsis_pos) = input_elipsis_positions[input_id] {
                            if elipsis_pos < position {
                                position = position + elipsis_rank as usize - 1;
                            }
                        }
                        axes.push(inputs[input_id].shape[position].bex());
                    }
                }
                s.equals_all(axes)?;
            }
            Ok(())
        }*/
        Ok(())
    }
}

fn resolve_ellipsis(expr: &Expr, elipsis_rank: usize) -> Expr {
    todo!()
}
