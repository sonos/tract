use crate::internal::*;

mod eval;
mod expr;
pub use expr::Axis;
pub use expr::Expr;
use tract_data::itertools::Itertools;
mod to_matmul;

#[derive(Debug, Clone, Hash)]
pub struct EinSum {
    pub expr: Expr,
}



impl Op for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.expr)])
    }

    op_as_typed_op!();
}

impl EinSum {
    pub fn new(expr: Expr) -> EinSum {
        EinSum { expr }
    }
}

impl EvalOp for EinSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        dispatch_numbers!(eval::eval_t(inputs[0].datum_type())(&self.expr, inputs))
    }
}

impl TypedOp for EinSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs
            .iter()
            .enumerate()
            .all(|(ix, fact)| fact.rank() == self.expr.input_rank(ix)));
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            eval::output_shape(&self.expr, &shapes)
        )))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let inv = self
            .expr
            .iter_all_axes()
            .filter_map(|axis| {
                // if axis is used twice, don't even dare do anything
                if axis.inputs.iter().any(|input| input.len() > 1) {
                    None
                } else {
                    let i = (0..inputs.len()).map(|i| axis.inputs[i].get(0).cloned()).collect();
                    let o = axis.result;
                    Some(AxisInfo { inputs: i, outputs: tvec!(o), period: 1, disposable: true })
                }
            })
            .collect();
        Ok(inv)
    }

    #[allow(unused_variables)]
    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let (mut inputs, mut result) = self.expr.to_strs();
        let interface: &mut String = match io {
            InOut::In(i) => &mut inputs[i],
            InOut::Out(o) => &mut result,
        };
        let mut axes: Vec<char> = interface.chars().collect();
        match change {
            AxisOp::Rm(rm) => {
                axes.remove(*rm);
            }
            AxisOp::Add(add) => axes.insert(*add, self.expr.available_label()),
            AxisOp::Move(from, to) => {
                let c = axes.remove(*from);
                axes.insert(*to, c);
            }
            _ => return Ok(None),
        };
        *interface = axes.into_iter().collect();
        let expr = Expr::from_strs(&inputs, Some(&result));
        return Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(EinSum::new(expr))),
            wire_changes: tvec!((io, change.clone())),
        }));
    }

    /*
    fn declutter(
    &self,
    model: &TypedModel,
    node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
    to_matmul::declutter(self, model, node)
    }
    */

    as_op!();
}
