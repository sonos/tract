use crate::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct QSumB {
    pub r: usize,
    pub n: TDim,
    pub k: usize,
}

impl Op for QSumB {
    fn name(&self) -> Cow<str> {
        "QSumB".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("r:{}, n:{}, k:{}", self.r, self.n, self.k)])
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for QSumB {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let n = self.n.eval_to_i64(&session.resolved_symbols)? as usize;
        self.eval(inputs, n)
    }
}

impl TypedOp for QSumB {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape: ShapeFact = inputs[0].shape.clone();
        shape.set(shape.rank() - 1, self.n.to_dim());
        Ok(tvec!(i32::fact(shape)))
    }
}

impl QSumB {
    fn eval(&self, inputs: TVec<TValue>, n: usize) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let mut shape: TVec<usize> = input.shape().into();
        shape[input.rank() - 1] = n;
        let mut output = ArrayD::zeros(&*shape);
        match input.datum_type().unquantized() {
            DatumType::I8 => self.eval_t::<i8>(&input, &mut output.view_mut(), n)?,
            DatumType::U8 => self.eval_t::<u8>(&input, &mut output.view_mut(), n)?,
            dt => bail!("Unsupported input type in quantized operation ({:?})", dt),
        }
        Ok(tvec!(output.into_tvalue()))
    }

    fn eval_t<T: Datum + tract_num_traits::AsPrimitive<i32>>(
        &self,
        input: &Tensor,
        output: &mut ArrayViewMutD<i32>,
        n: usize,
    ) -> TractResult<()> {
        for prefix in tract_ndarray::indices(&output.shape()[0..output.ndim() - 1]) {
            let mut panel = input.to_array_view::<T>()?;
            let mut output = output.view_mut();
            for d in prefix.slice() {
                panel.index_axis_inplace(Axis(0), *d);
                output.index_axis_inplace(Axis(0), *d);
            }
            let panel = panel.as_slice().unwrap();
            for p in 0..(n.divceil(self.r)) {
                let mut vec = vec![0i32; self.r];
                for k in 0..self.k {
                    for r in 0..self.r {
                        vec[r] += panel[(p * self.k + k) * self.r + r].as_();
                    }
                }
                let r_slice = &mut output.as_slice_mut().unwrap()[p * self.r..];
                let len = r_slice.len().min(self.r);
                r_slice[..len].copy_from_slice(&vec[..len]);
            }
        }
        Ok(())
    }
}
