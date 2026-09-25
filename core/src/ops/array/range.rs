use crate::ops::cast::Cast;
use tract_num_traits::AsPrimitive;
use tract_num_traits::Zero;

use crate::internal::*;

use super::Slice;

#[derive(Debug, Default, Clone, new, Hash, PartialEq, Eq)]
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
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (start, end, step) = args_3!(inputs);
        Ok(tvec!(self.make(&start, &end, &step, ctx.symbols)?.into_tvalue()))
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
            let mut v = start.try_as_plain_ram()?.to_scalar::<T>()?.clone();
            let step = step.try_as_plain_ram()?.to_scalar::<T>()?;
            {
                let mut result_plain = result.try_as_plain_ram_mut()?;
                let slots = result_plain.as_slice_mut_unchecked::<T>().as_mut_ptr();
                for i in 0..len {
                    std::ptr::write(slots.add(i), v.clone());
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
            // TDim inputs always yield an i64 tensor: both branches of
            // output_facts declare i64 for TDim inputs (the symbolic value
            // rides in uniform_tdim), and the plan's fact-vs-value assertions
            // check that.
            let start = start.try_as_plain_ram()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let step = step.try_as_plain_ram()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let end = end.try_as_plain_ram()?.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let len = {
                // i128 intermediate: end - start can overflow i64
                let span = usize::try_from((end as i128 - start as i128).unsigned_abs()).ok();
                let step = usize::try_from(step.unsigned_abs()).ok();
                match (span, step) {
                    (Some(span), Some(step)) if step > 0 => span.divceil(step),
                    _ => bail!("Range span or step out of usize bounds"),
                }
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
        let start = start.try_as_plain_ram()?.to_scalar::<T>()?;
        let end = end.try_as_plain_ram()?.to_scalar::<T>()?;
        let step = step.try_as_plain_ram()?.to_scalar::<T>()?;
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
                let start_tdim = start.try_as_plain_ram()?.to_scalar::<TDim>()?.clone();
                let end_tdim = end.try_as_plain_ram()?.to_scalar::<TDim>()?;
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
            // TDim inputs yield i64, like the konst branch above: the
            // symbolic per-element value rides in uniform_tdim, and
            // Range::make materializes i64 regardless of the input datum
            // type (a contract pulse-opl relies on).
            let dt = if start.datum_type.is_tdim() { i64::datum_type() } else { start.datum_type };
            let mut fact = dt.fact(std::slice::from_ref(&self.len));
            if let (Some(s), Some(k)) = (&start.uniform_tdim, &step.uniform_tdim)
                && let Some(scope) = self.len.find_scope()
            {
                let x0 = TDim::Sym(scope.coord_sym(0));
                let term = match k {
                    TDim::Val(1) => x0,
                    TDim::Val(v) => TDim::MulInt(*v, Box::new(x0)),
                    other => TDim::Mul(vec![other.clone(), x0]),
                };
                fact.uniform_tdim = Some((s.clone() + term).reduce());
            }
            Ok(tvec!(fact))
        }
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tdims_model(model: &mut TypedModel, end: OutletId) -> TractResult<TVec<OutletId>> {
        let start = model.wire_node(
            "start",
            crate::ops::konst::Const::new(tensor0(TDim::Val(0)).into_arc_tensor())?,
            &[],
        )?;
        let step = model.wire_node(
            "step",
            crate::ops::konst::Const::new(tensor0(TDim::Val(1)).into_arc_tensor())?,
            &[],
        )?;
        let s = model.symbols.sym("S");
        model.wire_node("range", Range::new(s.to_dim()), &[start[0], end, step[0]])
    }

    // sdpa-style: all-konst TDim scalars (the konst branch of output_facts).
    // Both the fact and the evaluated tensor must be i64. S is bound at
    // runtime through the definer source' shape.
    #[test]
    fn konst_tdim_inputs_yield_i64() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let definer = model.add_source("definer", f32::datum_type().fact([s.to_dim()]))?;
        let end = model.wire_node(
            "end",
            crate::ops::konst::Const::new(tensor0(TDim::Sym(s)).into_arc_tensor())?,
            &[],
        )?;
        let range = tdims_model(&mut model, end[0])?;
        model.select_output_outlets(&[range[0], definer])?;
        assert_eq!(model.outlet_fact(range[0])?.datum_type, i64::datum_type());
        let found = crate::internal::TypedSimplePlan::new(model)?
            .run(tvec!(tensor1(&[0f32; 5]).into_tvalue()))?;
        assert_eq!(*found[0], tensor1(&[0i64, 1, 2, 3, 4]));
        Ok(())
    }

    // dynamic end (the else branch of output_facts): i64 fact, i64 values,
    // symbolic per-element value still tracked as uniform_tdim.
    #[test]
    fn dynamic_tdim_end_yields_i64() -> TractResult<()> {
        let mut model = TypedModel::default();
        let end = model.add_source("T_dyn", TDim::datum_type().scalar_fact())?;
        let range = tdims_model(&mut model, end)?;
        model.select_output_outlets(&range)?;
        let fact = model.outlet_fact(range[0])?;
        assert_eq!(fact.datum_type, i64::datum_type());
        assert!(fact.uniform_tdim.is_some());
        let found = crate::internal::TypedSimplePlan::new(model)?
            .run(tvec!(tensor0(TDim::Val(5)).into_tvalue()))?;
        assert_eq!(*found[0], tensor1(&[0i64, 1, 2, 3, 4]));
        Ok(())
    }
}
