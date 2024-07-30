use tract_num_traits::AsPrimitive;

use crate::internal::*;

#[derive(Debug, Default, Clone, new, Hash)]
pub struct Range {
    len: TDim,
}

impl Op for Range {
    fn name(&self) -> Cow<str> {
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
        session: &SessionState,
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
            let mut v = start.to_scalar::<T>()?.clone();
            let step = step.to_scalar::<T>()?;
            for i in 0..len {
                result.as_slice_mut_unchecked::<T>()[i] = v.clone();
                v = v + step;
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
            let len = {
                let start = start.to_scalar::<TDim>()?.eval(values).to_i64()?;
                let end = end.to_scalar::<TDim>()?.eval(values).to_i64()?;
                let step = step.to_scalar::<TDim>()?.eval(values).to_i64()?;
                #[allow(clippy::cast_abs_to_unsigned)]
                ((end - start).abs() as usize).divceil(step.abs() as usize)
            };
            Self::make_t::<TDim>(start, step, len)
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
        let start = start.to_scalar::<T>()?;
        let end = end.to_scalar::<T>()?;
        let step = step.to_scalar::<T>()?;
        Ok(((end.as_() - start.as_()) / (step.as_())).ceil() as usize)
    }
}

impl TypedOp for Range {
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
            let len = if start.datum_type() == TDim::datum_type() {
                let start = start.to_scalar::<TDim>()?;
                let end = end.to_scalar::<TDim>()?;
                let step = step.cast_to_scalar::<i64>()?;
                if step < 0 {
                    (start.clone() - end).divceil(-step as usize)
                } else {
                    (end.clone() - start).divceil(step as usize)
                }
            } else {
                dispatch_numbers!(Self::len_for_numbers(start.datum_type())(
                    self, start, end, step
                ))?
                .to_dim()
            };
            Ok(tvec!(start.datum_type().fact([len])))
        } else {
            Ok(tvec!(start.datum_type.fact(&[self.len.clone()])))
        }
    }

    as_op!();
}
