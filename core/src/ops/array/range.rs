use crate::ops::cast::Cast;
use tract_num_traits::AsPrimitive;
use tract_num_traits::Zero;

use crate::internal::*;

use super::Slice;

#[derive(Debug, Default, Clone, new, Hash)]
pub struct Range {
    len: TDim,
}

impl Op for Range {
    fn name(&self) -> StaticName {
        "Range".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Range {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (start, end, step) = args_3!(inputs);
        Ok(tvec!(self.make(&start, &end, &step, &session.resolved_symbols)?.into_tvalue()))
    }
}

impl Range {
    fn make_t<T: Datum + for<'a> std::ops::Add<&'a T, Output = T>>(
        start: &Tensor,
        step: &Tensor,
        len: usize,
    ) -> TractResult<Tensor> {
        unsafe {
            let mut result = Tensor::uninitialized::<T>(&[len])?;
            let mut v = start.try_as_dense()?.to_scalar::<T>()?.clone();
            let step = step.try_as_dense()?.to_scalar::<T>()?;
            {
                let mut result_dense = result.try_as_dense_mut()?;
                for i in 0..len {
                    result_dense.as_slice_mut_unchecked::<T>()[i] = v.clone();
                    v = v + step;
                }
            }
            Ok(result)
        }
    }

    fn make(
        &self,
        start: &Tensor,
        end: &Tensor,
        step: &Tensor,
        values: &SymbolValues,
    ) -> TractResult<Tensor> {
        if start.datum_type() == TDim::datum_type() {
            let start = start.try_as_dense()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let step = step.try_as_dense()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let len = {
                let end = end.try_as_dense()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
                #[allow(clippy::cast_abs_to_unsigned)]
                ((end - start).abs() as usize).divceil(step.abs() as usize)
            };
            Self::make_t::<i64>(&tensor0(start), &tensor0(step), len)
        } else {
            let len = dispatch_numbers!(Self::len_for_numbers(start.datum_type())(
                self, start, end, step
            ))?;
            dispatch_numbers!(Self::make_t(start.datum_type())(start, step, len))
        }
    }

    fn len_for_numbers<T: Datum + AsPrimitive<f64>>(
        &self,
        start: &Tensor,
        end: &Tensor,
        step: &Tensor,
    ) -> TractResult<usize> {
        let start = start.try_as_dense()?.to_scalar::<T>()?;
        let end = end.try_as_dense()?.to_scalar::<T>()?;
        let step = step.try_as_dense()?.to_scalar::<T>()?;
        Ok(((end.as_() - start.as_()) / (step.as_())).ceil() as usize)
    }
}

impl TypedOp for Range {
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        rule_if_some!(succ = model.single_succ(node.id)?);
        rule_if_some!(slice = succ.op_as::<Slice>());
        rule_if!(slice.start.is_zero());
        rule_if!(slice.end.is_zero());

        let mut patch = TypedModelPatch::default();
        let mut wire = patch.tap_model(model, node.inputs[0])?;
        if model.outlet_fact(node.inputs[0])?.datum_type.is_tdim() {
            wire = patch.wire_node(
                format!("{}.cast-tdim", node.name),
                Cast { to: DatumType::I64 },
                &[wire],
            )?[0];
        }
        let wire = patch.wire_node(&node.name, AxisOp::Add(0), &[wire])?;
        patch.shunt_outside(model, succ.id.into(), wire[0])?;
        Ok(Some(patch))
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [start, end, step] = inputs else {
            bail!("Expects three inputs");
        };
        ensure!(start.datum_type() == end.datum_type());
        ensure!(start.datum_type() == step.datum_type());
        ensure!(start.shape.volume().is_one());
        ensure!(end.shape.volume().is_one());
        ensure!(step.shape.volume().is_one());
        if let (Some(start), Some(end), Some(step)) = (&start.konst, &end.konst, &step.konst) {
            if start.datum_type() == TDim::datum_type() {
                let start_tdim = start.try_as_dense()?.to_scalar::<TDim>()?.clone();
                let end_tdim = end.try_as_dense()?.to_scalar::<TDim>()?;
                let step = step.cast_to_scalar::<i64>()?;
                let len = if step < 0 {
                    (start_tdim.clone() - end_tdim).divceil(-step as usize)
                } else {
                    (end_tdim.clone() - start_tdim.clone()).divceil(step as usize)
                };
                let mut fact = DatumType::I64.fact([len]);
                if let Some(scope) = start_tdim.find_scope().or_else(|| end_tdim.find_scope()) {
                    let x0 = TDim::Sym(scope.coord_sym(0));
                    let term = if step == 1 { x0 } else { TDim::MulInt(step, Box::new(x0)) };
                    fact.uniform_tdim = Some((start_tdim + term).reduce());
                }
                Ok(tvec!(fact))
            } else {
                let len = dispatch_numbers!(Self::len_for_numbers(start.datum_type())(
                    self, start, end, step
                ))?
                .to_dim();
                Ok(tvec!(start.datum_type().fact([len])))
            }
        } else {
            let mut fact = start.datum_type.fact(std::slice::from_ref(&self.len));
            if let (Some(s), Some(k)) = (&start.uniform_tdim, &step.uniform_tdim) {
                if let Some(scope) = self.len.find_scope() {
                    let x0 = TDim::Sym(scope.coord_sym(0));
                    let term = match k {
                        TDim::Val(1) => x0,
                        TDim::Val(v) => TDim::MulInt(*v, Box::new(x0)),
                        other => TDim::Mul(vec![other.clone(), x0]),
                    };
                    fact.uniform_tdim = Some((s.clone() + term).reduce());
                }
            }
            Ok(tvec!(fact))
        }
    }

    as_op!();
}
