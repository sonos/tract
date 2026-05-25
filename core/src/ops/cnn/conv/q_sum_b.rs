use crate::internal::*;
use tract_linalg::mmm::{MMMInputValue, PackedMatrixStorage};
use tract_linalg::pack::PackedI8K4;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct QSumB {
    pub dt: DatumType,
    pub r: usize,
    pub n: TDim,
    pub k: usize,
}

impl Op for QSumB {
    fn name(&self) -> StaticName {
        "QSumB".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("r:{}, n:{}, k:{}", self.r, self.n, self.k)])
    }

    op_as_typed_op!();
}

impl EvalOp for QSumB {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
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
        let storage = input
            .try_storage_as::<PackedMatrixStorage>()
            .context("Expected PackedMatrixStorage")?;
        let batch_shape = storage.batch_shape();
        let mut shape: TVec<usize> = batch_shape.into();
        shape.push(n);
        let mut output = ArrayD::<i32>::zeros(&*shape);
        for b in 0..batch_shape[0] {
            let mut output_view = output.index_axis_mut(Axis(0), b);
            for g in 0..batch_shape[1] {
                let mut output_view = output_view.index_axis_mut(Axis(0), g);
                let output_slice = output_view.as_slice_mut().unwrap();
                let payload = storage.value_at(&[b, g]);
                match self.dt.unquantized() {
                    DatumType::I8 => self.eval_t::<i8>(payload, output_slice)?,
                    DatumType::U8 => self.eval_t::<u8>(payload, output_slice)?,
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
        let (r, k, n) = (input.format().r(), input.k(), input.mn());
        // PackedI8K4 is K=4-inner: element (ik, ir) at (ik/4)*r*4 + ir*4 + ik%4,
        // and the panel is k padded up to a multiple of 4. PackedFormat is K-major.
        let is_k4 = input.format().downcast_ref::<PackedI8K4>().is_some();
        let panel_len = if is_k4 { r * k.div_ceil(4) * 4 } else { r * k };
        let panels = n.divceil(r);
        for ipanel in 0..panels {
            let panel = input.panel_bytes(ipanel, None)?;
            let panel: &[T] = unsafe { std::slice::from_raw_parts(panel as *const T, panel_len) };
            let mut vec = vec![0i32; r];
            if is_k4 {
                for ik in 0..k {
                    let kbase = (ik / 4) * r * 4 + ik % 4;
                    for ir in 0..r {
                        vec[ir] += panel[kbase + ir * 4].as_();
                    }
                }
            } else {
                for ik in 0..k {
                    for ir in 0..r {
                        vec[ir] += panel[ik * r + ir].as_();
                    }
                }
            }
            let len = r.min(n - r * ipanel);
            output[r * ipanel..][..len].copy_from_slice(&vec[..len]);
        }
        Ok(())
    }
}
