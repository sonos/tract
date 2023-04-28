use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn einsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let expr = node.get_attr::<String>("equation")?;
    let expr: AxesMapping = expr.replace("...", "*").parse()?;
    Ok((expand(EinSum { expr }), vec![]))
}

#[derive(Debug, Clone, Hash)]
pub struct EinSum {
    pub expr: AxesMapping,
}

impl Expansion for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let ranks = inputs
            .iter()
            .map(|o| model.outlet_fact(*o).map(|f| f.rank()))
            .collect::<TractResult<TVec<_>>>()?;
        let expr = resolve_ellipsis(&self.expr, &ranks)?;
        let operating_dt = model.outlet_fact(inputs[0])?.datum_type;
        model.wire_node(
            prefix,
            tract_core::ops::einsum::EinSum { axes: expr, operating_dt, q_params: None },
            inputs,
        )
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, self.expr.input_count())?;
        check_output_arity(outputs, 1)?;
        for (ix, input) in inputs.iter().enumerate() {
            s.equals(&input.datum_type, &outputs[0].datum_type)?;
            // if no elipsis in input spec then rank is known
            // onnx specifies that all ellipsis usage must have the same rank, but this rule is
            // broken by pytorch exporter which assumes numpy convention
            if !self.expr.iter_all_axes().any(|axis| axis.repr == '*' && axis.inputs[ix].len() == 1)
            {
                let rank =
                    self.expr.iter_all_axes().map(|axis| axis.inputs[ix].len()).sum::<usize>();
                s.equals(rank as i64, &input.rank)?;
            }
        }

        let ranks: Vec<_> = inputs.iter().map(|i| &i.rank).collect();
        s.given_all(ranks, move |s, ranks| {
            let ranks = ranks.iter().map(|r| *r as usize).collect::<TVec<_>>();
            let expr = resolve_ellipsis(&self.expr, &ranks)?;
            s.equals(&outputs[0].rank, expr.rank(InOut::Out(0)) as i64)?;
            for axis in expr.iter_all_axes() {
                let mut axes = vec![];
                if let Some(result) = axis.outputs[0].first() {
                    axes.push(outputs[0].shape[*result].bex())
                }
                for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
                    for position in input_axis_positions {
                        axes.push(inputs[input_id].shape[*position].bex());
                    }
                }
                s.equals_all(axes)?;
            }
            Ok(())
        })
    }
}

fn resolve_ellipsis(expr: &AxesMapping, ranks: &[usize]) -> TractResult<AxesMapping> {
    if expr.axis('*').is_err() {
        return Ok(expr.clone());
    }
    let elipsed_axes: TVec<usize> = ranks
        .iter()
        .enumerate()
        .filter_map(|(ix, rank)| {
            if expr.axis_positions(InOut::In(ix), '*').is_ok() {
                Some(rank + 1 - expr.rank(InOut::In(ix)))
            } else {
                None
            }
        })
        .collect();
    let max_axes = *elipsed_axes.iter().max().unwrap();
    let axis_resolved: String = ('a'..)
        .filter(|l| expr.iter_all_axes().all(|axis| *l != axis.repr))
        .take(max_axes)
        .collect();
    //let mut resolved = expr.iter_all_axes().filter(|axis| axis.repr != '*').collect();
    // lol.
    let mut resolved = expr.to_string();
    for axes in elipsed_axes {
        resolved = resolved.replacen(
            '*',
            &axis_resolved.chars().skip(max_axes - axes).collect::<String>(),
            1,
        );
    }
    // replace in output
    resolved = resolved.replacen('*', &axis_resolved, 1);
    resolved.parse()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_resolve_ellipsis_0() {
        assert_eq!(
            resolve_ellipsis(&"*ii->*i".parse().unwrap(), &[6]).unwrap().to_string(),
            "abcdii->abcdi"
        )
    }

    #[test]
    fn test_resolve_ellipsis_1() {
        assert_eq!(
            resolve_ellipsis(&"*mk,*kn->*mn".parse().unwrap(), &[4, 4]).unwrap().to_string(),
            "abmk,abkn->abmn"
        )
    }

    #[test]
    fn test_resolve_ellipsis_2() {
        assert_eq!(
            resolve_ellipsis(&"*ab,*bc->*ac".parse().unwrap(), &[4, 4]).unwrap().to_string(),
            "deab,debc->deac"
        )
    }

    #[test]
    fn test_resolve_numpy_ellipsis_1() -> TractResult<()> {
        let expr: AxesMapping = "*gi,*gih->*gh".parse()?;
        let resolved = resolve_ellipsis(&expr, &[4, 3])?;
        assert_eq!(resolved, "abgi,gih->abgh".parse().unwrap());
        Ok(())
    }
}
