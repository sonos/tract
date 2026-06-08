use crate::internal::*;
use tract_linalg::element_wise::ElementWise;

/// Fused LSTM cell epilogue.
///
/// Given the combined gate pre-activations
/// `preact = Xt·Wᵀ + Ht-1·Rᵀ + bias` of shape `[batch, 4*hidden]` (ONNX gate
/// order i, o, f, c) and the previous cell state `c_prev` `[batch, hidden]`,
/// computes the new hidden `Ht` and cell `Ct` in a SINGLE fused pass.
///
/// This collapses the per-gate `Sigmoid`/`Tanh` + elementwise `Mul`/`Add`
/// chain (≈ 15 separately-dispatched ops, each materialising an intermediate
/// tensor) into one op — the dominant non-matmul cost for streaming LSTM
/// inference. Standard activations only (`f = sigmoid`, `g = h = tanh`) and no
/// peepholes; the importer falls back to the decomposed form otherwise.
///
/// Activations use tract's vectorised `sigmoid`/`tanh` linalg kernels
/// (NEON on aarch64) applied to contiguous gate slices, so the output is
/// numerically identical to the decomposed Sigmoid/Tanh path while collapsing
/// the per-gate dispatch into one op. Runs in either `f32` or `f16`, matching
/// the dtype the precision transform settled the surrounding graph on.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LstmEpilogue {
    pub hidden: usize,
}

impl Op for LstmEpilogue {
    fn name(&self) -> StaticName {
        "LstmEpilogue".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("hidden={}", self.hidden)])
    }

    op_as_typed_op!();
}

impl EvalOp for LstmEpilogue {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        // Dispatch on the dtype the precision transform left the graph in. The
        // ONNX LSTM is f32-native, but `FloatPrecisionTranslator` rewrites the
        // whole float graph (including this op's inputs) to f16, so we must run
        // in whichever float type actually arrives — reading an f16 buffer as
        // f32 would walk off the end of the allocation.
        let ops = tract_linalg::ops();
        match inputs[0].datum_type().unquantized() {
            DatumType::F32 => self.eval_t::<f32>(inputs, (ops.sigmoid_f32)(), (ops.tanh_f32)()),
            DatumType::F16 => self.eval_t::<f16>(inputs, (ops.sigmoid_f16)(), (ops.tanh_f16)()),
            dt => bail!("LstmEpilogue only supports f32 and f16 preactivations, got {dt:?}"),
        }
    }
}

impl LstmEpilogue {
    fn eval_t<T>(
        &self,
        inputs: TVec<TValue>,
        sigmoid: Box<dyn ElementWise<T>>,
        tanh: Box<dyn ElementWise<T>>,
    ) -> TractResult<TVec<TValue>>
    where
        T: Datum + Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        let h = self.hidden;
        let c_prev = &inputs[1]; // [.., h]
        let cp = unsafe { c_prev.as_slice_unchecked::<T>() };
        let rows = inputs[0].len() / (4 * h); // any leading-dim layout, row-major
        // Mutable copy of preact so the activation kernels run in place.
        let mut pre_t = inputs[0].clone().into_tensor();
        let pre = unsafe { pre_t.as_slice_mut_unchecked::<T>() };
        let mut ht = unsafe { Tensor::uninitialized_dt(T::datum_type(), c_prev.shape())? };
        let mut ct = unsafe { Tensor::uninitialized_dt(T::datum_type(), c_prev.shape())? };
        {
            let hs = unsafe { ht.as_slice_mut_unchecked::<T>() };
            let cs = unsafe { ct.as_slice_mut_unchecked::<T>() };
            for r in 0..rows {
                let pb = r * 4 * h;
                let cb = r * h;
                let row = &mut pre[pb..pb + 4 * h];
                // gate order i,o,f,c: sigmoid the i,o,f block, tanh the c block
                sigmoid.run(&mut row[0..3 * h])?;
                tanh.run(&mut row[3 * h..4 * h])?;
                // Ct = ft*c_prev + it*cc  (it=row[j], ot=row[h+j], ft=row[2h+j], cc=row[3h+j])
                for j in 0..h {
                    cs[cb + j] = row[2 * h + j] * cp[cb + j] + row[j] * row[3 * h + j];
                }
                // Ht = ot * tanh(Ct): stage tanh(Ct) in hs, then scale by ot
                hs[cb..cb + h].copy_from_slice(&cs[cb..cb + h]);
                tanh.run(&mut hs[cb..cb + h])?;
                for j in 0..h {
                    hs[cb + j] = hs[cb + j] * row[h + j];
                }
            }
        }
        Ok(tvec!(ht.into_tvalue(), ct.into_tvalue()))
    }
}

impl TypedOp for LstmEpilogue {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2, "LstmEpilogue expects [preact, c_prev]");
        // Ht and Ct share c_prev's shape and dtype ([.., hidden]).
        let c_prev = inputs[1];
        let fact = c_prev.datum_type.fact(c_prev.shape.clone());
        Ok(tvec!(fact.clone(), fact))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fused epilogue must match a scalar reference LSTM cell (catches gate-order
    // and cell/hidden-formula bugs). Tolerance covers the rational sigmoid/tanh
    // approximation (~1e-7) vs the exact reference. Multi-row exercises batch.
    #[test]
    fn epilogue_matches_scalar_reference() {
        let h = 6usize;
        let batch = 3usize;
        let preact: Vec<f32> =
            (0..batch * 4 * h).map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.25).collect();
        let cprev: Vec<f32> = (0..batch * h).map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.2).collect();
        let pre_t = Tensor::from_shape(&[batch, 4 * h], &preact).unwrap();
        let cprev_t = Tensor::from_shape(&[batch, h], &cprev).unwrap();
        let op = LstmEpilogue { hidden: h };
        let out = op.eval(tvec!(pre_t.into_tvalue(), cprev_t.into_tvalue())).unwrap();
        let ht = unsafe { out[0].as_slice_unchecked::<f32>() };
        let ct = unsafe { out[1].as_slice_unchecked::<f32>() };

        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        for r in 0..batch {
            for j in 0..h {
                let p = r * 4 * h; // gate order on the 4*h axis: i, o, f, c
                let it = sig(preact[p + j]);
                let ot = sig(preact[p + h + j]);
                let ft = sig(preact[p + 2 * h + j]);
                let cc = preact[p + 3 * h + j].tanh();
                let c_ref = ft * cprev[r * h + j] + it * cc;
                let h_ref = ot * c_ref.tanh();
                assert!((ct[r * h + j] - c_ref).abs() < 1e-3, "Ct mismatch at ({r},{j})");
                assert!((ht[r * h + j] - h_ref).abs() < 1e-3, "Ht mismatch at ({r},{j})");
            }
        }
    }
}
