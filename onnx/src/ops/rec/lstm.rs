use tract_core::ndarray::*;
use crate::pb::NodeProto;
use tract_core::ops as core_ops;
use tract_core::ops::prelude::*;

pub fn lstm(pb: &NodeProto) -> TractResult<Box<Op>> {
    Ok(Box::new(LSTM::default()))
}

#[derive(Debug, Clone, new)]
pub struct LSTM {
    f: Box<StatelessOp>,
    g: Box<StatelessOp>,
    h: Box<StatelessOp>,
}

impl Default for LSTM {
    fn default() -> LSTM {
        LSTM {
            f: Box::new(core_ops::nn::Sigmoid::new(f32::datum_type().into())),
            g: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
            h: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
        }
    }
}

impl Op for LSTM {
    fn name(&self) -> Cow<str> {
        "LSTM".into()
    }

    fn rounding_errors(&self) -> bool {
        true
    }
}

impl StatefullOp for LSTM {
    fn state(&self) -> TractResult<Option<Box<OpState>>> {
        Ok(Some(Box::new(LSTMState {
            h_c: None,
        })))
    }
}

impl InferenceRulesOp for LSTM {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p[TensorProxy],
        outputs: &'p[TensorProxy],
    ) -> InferenceResult {
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[1].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?; // num_directions
        s.equals(&inputs[1].shape[1], &inputs[2].shape[1])?; // 4*hidden_size
        s.equals(&inputs[2].shape[1], 4 * inputs[2].shape[2].bex())?; // hidden_size
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?; // seq_lentgh
        s.equals(&outputs[0].shape[1], &inputs[1].shape[0])?; // num_directions
        s.equals(&outputs[0].shape[2], &inputs[0].shape[1])?; // batch_size
        s.equals(&outputs[0].shape[3], &inputs[1].shape[2])?; // hidden_size
        s.equals(&outputs[1].rank, 3)?;
        s.equals(&outputs[1].shape[0], &inputs[1].shape[0])?; // num_directions
        s.equals(&outputs[1].shape[1], &inputs[0].shape[1])?; // batch_size
        s.equals(&outputs[1].shape[2], &inputs[2].shape[2])?; // hidden_size
        if inputs.len() > 3 {
            // bias
            s.equals(&inputs[3].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[3].rank, 2)?;
            s.equals(&inputs[3].shape[0], &inputs[0].shape[0])?; // num_directions
            s.equals(&inputs[3].shape[1],  8 * inputs[2].shape[2].bex())?; // 8 * hidden_size
        }
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct LSTMState {
    h_c: Option<(Tensor, Tensor)>,
}

impl OpState for LSTMState {
    fn eval(&mut self, op: &Op, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let op: &LSTM = op.downcast_ref::<LSTM>().ok_or("LSTM state passed wrong op")?;
        let x:ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w:ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, input_size]
        let r:ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, hidden_size]

        let bias = if let Some(bias) = inputs.get(3) {
            Some(bias.to_array_view::<f32>()?.into_dimensionality::<Ix2>()?) // [num_directions, 8*hidden_size]
        } else {
            None
        };

        let seq_length = x.shape()[0];
        let batch_size = x.shape()[1];
        let num_directions = w.shape()[0];
        let hidden_size = r.shape()[2];

        if num_directions != 1 {
            bail!("Only forward LSTM implemented");
        }
        let w = w.index_axis_move(Axis(0), 0);
        let r = r.index_axis_move(Axis(0), 0);

        if self.h_c.is_none() {
            self.h_c = Some(
             (Array2::<f32>::zeros((batch_size, hidden_size)).into(),
             Array2::<f32>::zeros((batch_size, hidden_size)).into()));
        }
        let (ht, ct) = self.h_c.as_mut().unwrap();
        let mut ht: ArrayViewMut2<f32> = ht.to_array_view_mut::<f32>()?.into_dimensionality()?;
        let mut ct: ArrayViewMut2<f32> = ct.to_array_view_mut::<f32>()?.into_dimensionality()?;

        let mut ht_list: Array3<f32> = Array3::zeros((seq_length, batch_size, hidden_size));

        for (ix, x) in x.outer_iter().enumerate() {
            // x -> batch_size x input_size
            // Wt -> k=input_size x n=4*hidden_size
            // gates -> batch_size x 4 * hidden_size
            let mut gates = x.dot(&w.t()) + ht.dot(&r.t()); // batch_size x 4*hidden_size
            if let Some(bias) = bias {
                gates += &bias.slice(s!(0, 0..4*hidden_size));
                gates += &bias.slice(s!(0, 4*hidden_size..8*hidden_size));
            }
            let gates = gates.into_shape((batch_size, hidden_size, 4))?;
            dbg!(gates.shape());
            let i = op.f.eval(tvec!(gates.slice_axis(Axis(2), (0..=0).into()).to_owned().into()))?;
            let o = op.f.eval(tvec!(gates.slice_axis(Axis(2), (1..=1).into()).to_owned().into()))?;
            let f = op.f.eval(tvec!(gates.slice_axis(Axis(2), (2..=2).into()).to_owned().into()))?;
            let c = op.g.eval(tvec!(gates.slice_axis(Axis(2), (3..=3).into()).to_owned().into()))?;
            let i = i[0].to_array_view::<f32>()?.to_owned().into_dimensionality::<Ix3>()?.into_shape((batch_size, hidden_size))?;
            let o = o[0].to_array_view::<f32>()?.to_owned().into_dimensionality::<Ix3>()?.into_shape((batch_size, hidden_size))?;

            let f = f[0].to_array_view::<f32>()?.to_owned().into_dimensionality::<Ix3>()?.into_shape((batch_size, hidden_size))?;

            let c = c[0].to_array_view::<f32>()?.to_owned().into_dimensionality::<Ix3>()?.into_shape((batch_size, hidden_size))?;

            let big_c = f * &ct + i * c;
            let big_h = o * op.h.eval(tvec!(big_c.clone().into()))?[0].to_array_view::<f32>()?.to_owned().into_dimensionality::<Ix2>()?;
            ht_list.slice_axis_mut(Axis(0), (ix..=ix).into()).assign(&ht);
            ht.assign(&big_h);
            ct.assign(&big_c);
        }
        let ht_list = ht_list.into_shape((seq_length, 1, batch_size, hidden_size))?;
        let hto = ht.to_owned().into_shape((1, batch_size, hidden_size))?;

        Ok(tvec!(ht_list.into(), hto.into()))
    }

}

