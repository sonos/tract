use tract_num_traits::AsPrimitive;

use crate::internal::*;

#[derive(Debug, Default, Clone, new, Hash)]
pub struct Range {
    pub start: Tensor,
    pub end: Tensor,
    pub step: Tensor,
}

impl_dyn_hash!(Range);

impl Op for Range {
    fn name(&self) -> Cow<str> {
        "Range".into()
    }

    op_core!();
    op_as_typed_op!();
}

impl EvalOp for Range {
    fn is_stateless(&self) -> bool {
        self.start.datum_type() != TDim::datum_type()
            || (self.start.to_scalar::<TDim>().unwrap().to_i64().is_ok()
                && self.end.to_scalar::<TDim>().unwrap().to_i64().is_ok()
                && self.step.to_scalar::<TDim>().unwrap().to_i64().is_ok())
    }

    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let tensor = self.make(None)?;
        Ok(tvec!(tensor.into_arc_tensor()))
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        if self.is_stateless() {
            Ok(None)
        } else {
            Ok(Some(Box::new(self.clone())))
        }
    }
}

impl OpState for Range {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        _inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(self.make(Some(&session.resolved_symbols))?.into_arc_tensor()))
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

    fn make(&self, values: Option<&SymbolValues>) -> TractResult<Tensor> {
        if self.start.datum_type() == TDim::datum_type() {
            let none = SymbolValues::default();
            let values = values.unwrap_or(&none);
            let start = self.start.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let end = self.end.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let step = self.step.to_scalar::<TDim>()?.eval(values).to_i64()?;
            let len = ((end - start).abs() as usize).divceil(step.abs() as usize);
            return Self::make_t::<TDim>(&self.start, &self.step, len);
        } else {
            let len = dispatch_numbers!(Self::len_for_numbers(self.start.datum_type())(self))?;
            dispatch_numbers!(Self::make_t(self.start.datum_type())(&self.start, &self.step, len))
        }
    }

    fn len_for_numbers<T: Datum + AsPrimitive<f64>>(&self) -> TractResult<usize> {
        let start = self.start.to_scalar::<T>()?;
        let end = self.end.to_scalar::<T>()?;
        let step = self.step.to_scalar::<T>()?;
        Ok(((end.as_() - start.as_()) / (step.as_())).ceil() as usize)
    }
}

impl TypedOp for Range {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(self.start.datum_type() == self.end.datum_type());
        ensure!(self.start.datum_type() == self.step.datum_type());
        let len = if self.start.datum_type() == TDim::datum_type() {
            let start = self.start.to_scalar::<TDim>()?;
            let end = self.end.to_scalar::<TDim>()?;
            let step = self.step.to_scalar::<TDim>()?.to_i64()?;
            (end.clone() - start).divceil(step as usize)
        } else {
            dispatch_numbers!(Self::len_for_numbers(self.start.datum_type())(self))?.into()
        };
        Ok(tvec!(self.start.datum_type().fact(&[len])))
    }
    as_op!();
}
