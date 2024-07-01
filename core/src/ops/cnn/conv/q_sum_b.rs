use crate::internal::*;
use tract_linalg::mmm::MMMInputValue;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct QSumB {
    pub dt: DatumType,
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
        let mut shape: TVec<TDim> = inputs[0].shape.to_tvec();
        shape.push(self.n.to_dim());
        Ok(tvec!(i32::fact(shape)))
    }
}

impl QSumB {
    fn eval(&self, inputs: TVec<TValue>, n: usize) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let payloads = input.to_array_view::<Opaque>()?;
        let mut shape: TVec<usize> = input.shape().into();
        shape.push(n);
        let mut output = ArrayD::<i32>::zeros(&*shape);
        for b in 0..shape[0] {
            let mut output_view = output.index_axis_mut(Axis(0), b);
            for g in 0..shape[1] {
                let mut output_view = output_view.index_axis_mut(Axis(0), g);
                let output_slice = output_view.as_slice_mut().unwrap();
                let payload = payloads[[b, g]]
                    .downcast_ref::<Box<dyn MMMInputValue>>()
                    .context("Expected MMMInputs")?;
                match self.dt.unquantized() {
                    DatumType::I8 => self.eval_t::<i8>(&**payload, output_slice)?,
                    DatumType::U8 => self.eval_t::<u8>(&**payload, output_slice)?,
                    dt => bail!("Unsupported input type in quantized operation ({:?})", dt),
                }
            }
        }
        Ok(tvec!(output.into_tvalue()))
    }

    fn eval_t<T: Datum + tract_num_traits::AsPrimitive<i32>>(
        &self,
        input: &dyn MMMInputValue,
        output: &mut [i32],
    ) -> TractResult<()> {
        let (r, k, n) = (input.r(), input.k(), input.mn());
        let panels = n.divceil(r);
        for ipanel in 0..panels {
            let panel = input.panel_bytes(ipanel, None)?;
            let panel: &[T] = unsafe { std::slice::from_raw_parts(panel as *const T, r * k) };
            let mut vec = vec![0i32; r];
            for ik in 0..k {
                for ir in 0..r {
                    vec[ir] += panel[ik * r + ir].as_();
                }
            }
            let len = r.min(n - r * ipanel);
            output[r * ipanel..][..len].copy_from_slice(&vec[..len]);
        }
        Ok(())
    }
}
