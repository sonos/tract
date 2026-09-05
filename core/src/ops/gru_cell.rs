use crate::internal::*;
use tract_linalg::element_wise::ElementWise;
use tract_linalg::routines::Func;

/// Fused GRU cell epilogue.
///
/// Given `xh = Xt·Wᵀ + Wb` and `rh = Ht-1·Rᵀ + Rb`, both `[batch, 3*hidden]` in
/// ONNX gate order z, r, h, and the previous hidden state `h_prev`
/// `[batch, hidden]`, computes the new hidden state in a single fused pass:
///
/// ```text
/// zt = sigmoid(xh[..h]    + rh[..h])
/// rt = sigmoid(xh[h..2h]  + rh[h..2h])
/// ht = tanh   (xh[2h..3h] + rt (.) rh[2h..3h])
/// Ht = ht + zt (.) (h_prev - ht)
/// ```
///
/// This is the ONNX GRU with `linear_before_reset != 0`, where the reset gate
/// scales the already-biased recurrent product so the cell needs no second
/// matmul. With `linear_before_reset == 0` the reset gate applies to `Ht-1`
/// before that product and no such epilogue exists; the importer keeps the
/// decomposed form there, as it does for non-standard activations (`f` must be
/// sigmoid, `g` must be tanh) and for a symbolic hidden size.
///
/// Activations use tract's vectorised `sigmoid`/`tanh` linalg kernels over
/// contiguous gate slices, so they match the decomposed path; the gate sums are
/// associated differently, so results differ by rounding. Runs in `f32` or
/// `f16`, matching the dtype the precision transform settled the graph on.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GruEpilogue {
    pub hidden: usize,
}

impl Op for GruEpilogue {
    fn name(&self) -> StaticName {
        "GruEpilogue".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("hidden={}", self.hidden)])
    }

    op_as_typed_op!();
}

impl EvalOp for GruEpilogue {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        match inputs[0].datum_type().unquantized() {
            DatumType::F32 => {
                self.eval_t::<f32>(inputs, Func::Sigmoid.ew_f32()?, Func::Tanh.ew_f32()?)
            }
            DatumType::F16 => {
                self.eval_t::<f16>(inputs, Func::Sigmoid.ew_f16()?, Func::Tanh.ew_f16()?)
            }
            dt => bail!("GruEpilogue only supports f32 and f16 preactivations, got {dt:?}"),
        }
    }
}

/// The GRU cell's row loop, over plain slices.
///
/// Exposed so a whole-sequence op can run the same arithmetic per timestep without
/// paying a tensor box per gate: `xh` is consumed in place as the accumulator, and
/// `h_out` receives the new hidden state. Keeping one implementation is what makes
/// the fused form bit-identical to the `Scan` it replaces.
#[allow(clippy::too_many_arguments)]
pub fn gru_cell_rows<T>(
    h: usize,
    rows: usize,
    xh: &mut [T],
    rh: &[T],
    h_prev: &[T],
    h_out: &mut [T],
    sigmoid: &dyn ElementWise<T>,
    tanh: &dyn ElementWise<T>,
) -> TractResult<()>
where
    T: Datum
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
{
    for row in 0..rows {
        let gb = row * 3 * h;
        let hb = row * h;
        let g = &mut xh[gb..gb + 3 * h];
        let r = &rh[gb..gb + 3 * h];
        for j in 0..2 * h {
            g[j] = g[j] + r[j];
        }
        sigmoid.run(&mut g[0..2 * h])?;
        for j in 0..h {
            g[2 * h + j] = g[2 * h + j] + g[h + j] * r[2 * h + j];
        }
        tanh.run(&mut g[2 * h..3 * h])?;
        for j in 0..h {
            let cand = g[2 * h + j];
            h_out[hb + j] = cand + g[j] * (h_prev[hb + j] - cand);
        }
    }
    Ok(())
}

impl GruEpilogue {
    fn eval_t<T>(
        &self,
        inputs: TVec<TValue>,
        sigmoid: Box<dyn ElementWise<T>>,
        tanh: Box<dyn ElementWise<T>>,
    ) -> TractResult<TVec<TValue>>
    where
        T: Datum
            + Copy
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>,
    {
        let h = self.hidden;
        let h_prev = &inputs[2];
        let hp = unsafe { h_prev.as_slice_unchecked::<T>() };
        // Rows come from the state, which also sizes the output: a gate operand
        // left broadcast on the batch axis would otherwise under-fill it.
        let rows = hp.len() / h;
        ensure!(
            inputs[0].len() == rows * 3 * h && inputs[1].len() == rows * 3 * h,
            "GruEpilogue expects xh and rh shaped [{rows}, 3*{h}], got {:?} and {:?}",
            inputs[0].shape(),
            inputs[1].shape()
        );
        let rh = unsafe { inputs[1].as_slice_unchecked::<T>() };
        let mut acc_t = inputs[0].clone().into_tensor();
        let acc = unsafe { acc_t.as_slice_mut_unchecked::<T>() };
        let mut ht = unsafe { Tensor::uninitialized_dt(T::datum_type(), h_prev.shape())? };
        {
            let hs = unsafe { ht.as_slice_mut_unchecked::<T>() };
            gru_cell_rows(h, rows, acc, rh, hp, hs, &*sigmoid, &*tanh)?;
        }
        Ok(tvec!(ht.into_tvalue()))
    }
}

impl TypedOp for GruEpilogue {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "GruEpilogue expects [xh, rh, h_prev]");
        ensure!(
            inputs[0].datum_type == inputs[2].datum_type,
            "GruEpilogue gate and state datum types differ: {:?} and {:?}",
            inputs[0].datum_type,
            inputs[2].datum_type
        );
        let h_prev = inputs[2];
        Ok(tvec!(h_prev.datum_type.fact(h_prev.shape.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fused epilogue must match a scalar reference GRU cell with
    // linear_before_reset = 1. The reference keeps the ONNX form of the output
    // update, (1-zt)*ht + zt*Ht-1, so the rearranged form is checked too.
    // Tolerance covers the rational sigmoid/tanh approximation vs the exact
    // reference. Multi-row exercises batch.
    #[test]
    fn epilogue_matches_scalar_reference() {
        let h = 4usize;
        let batch = 3usize;
        let xh: Vec<f32> =
            (0..batch * 3 * h).map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.25).collect();
        let rh: Vec<f32> =
            (0..batch * 3 * h).map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.3).collect();
        let hprev: Vec<f32> = (0..batch * h).map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.2).collect();
        let xh_t = Tensor::from_shape(&[batch, 3 * h], &xh).unwrap();
        let rh_t = Tensor::from_shape(&[batch, 3 * h], &rh).unwrap();
        let hprev_t = Tensor::from_shape(&[batch, h], &hprev).unwrap();
        let op = GruEpilogue { hidden: h };
        let out = op
            .eval(
                &EvalContext::out_of_plan(),
                tvec!(xh_t.into_tvalue(), rh_t.into_tvalue(), hprev_t.into_tvalue()),
            )
            .unwrap();
        let got = unsafe { out[0].as_slice_unchecked::<f32>() };

        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        for r in 0..batch {
            for j in 0..h {
                let p = r * 3 * h; // gate order on the 3*h axis: z, r, h
                let zt = sig(xh[p + j] + rh[p + j]);
                let rt = sig(xh[p + h + j] + rh[p + h + j]);
                let ht = (xh[p + 2 * h + j] + rt * rh[p + 2 * h + j]).tanh();
                let h_ref = (1.0 - zt) * ht + zt * hprev[r * h + j];
                assert!((got[r * h + j] - h_ref).abs() < 1e-3, "Ht mismatch at ({r},{j})");
            }
        }
    }
}
