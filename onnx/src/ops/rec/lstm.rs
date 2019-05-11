use crate::pb::NodeProto;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;
use tract_core::internal::*;

pub fn lstm(_pb: &NodeProto) -> TractResult<Box<Op>> {
    Ok(Box::new(LSTM::default()))
}

#[derive(Debug, Clone, new)]
pub struct LSTM {
    pub f: Box<StatelessOp>,
    pub g: Box<StatelessOp>,
    pub h: Box<StatelessOp>,
    pub initial_c: Option<Tensor>,
    pub initial_h: Option<Tensor>,
}

impl Default for LSTM {
    fn default() -> LSTM {
        LSTM {
            f: Box::new(core_ops::nn::Sigmoid::new(f32::datum_type().into())),
            g: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
            h: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
            initial_c: None,
            initial_h: None,
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
    fn state(&self, _session: &mut SessionState) -> TractResult<Option<Box<OpState>>> {
        Ok(Some(Box::new(LSTMState { h_c: None })))
    }
}

impl InferenceRulesOp for LSTM {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
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
            s.equals(&inputs[3].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[3].shape[1], 8 * inputs[2].shape[2].bex())?; // 8 * hidden_size
        }
        if outputs.len() == 3 {
            s.equals(&outputs[2].datum_type, &outputs[1].datum_type)?;
            s.equals(&outputs[2].shape, &outputs[1].shape)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct LSTMState {
    h_c: Option<(Tensor, Tensor)>,
}

impl OpState for LSTMState {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        op: &Op,
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let op: &LSTM = op.downcast_ref::<LSTM>().ok_or("LSTM state passed wrong op")?;
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, hidden_size]

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
            let h = op
                .initial_h
                .clone()
                .unwrap_or_else(|| Array2::<f32>::zeros((batch_size, hidden_size)).into());
            let c = op
                .initial_c
                .clone()
                .unwrap_or_else(|| Array2::<f32>::zeros((batch_size, hidden_size)).into());
            self.h_c = Some((h, c))
        }
        let (ht, ct) = self.h_c.as_mut().unwrap();
        let mut ht: ArrayViewMut2<f32> = ht.to_array_view_mut::<f32>()?.into_dimensionality()?;
        let mut ct: ArrayViewMut2<f32> = ct.to_array_view_mut::<f32>()?.into_dimensionality()?;

        // dbg!(&ct);
        let mut ht_list: Array3<f32> = Array3::zeros((seq_length, batch_size, hidden_size));

        for (ix, x) in x.outer_iter().enumerate() {
            // x -> batch_size x input_size
            // Wt -> k=input_size x n=4*hidden_size
            // iofc -> batch_size x 4 * hidden_size
            //dbg!(&x);
            //dbg!(&ht);
            let mut iofc = x.dot(&w.t()) + ht.dot(&r.t()); // batch_size x 4*hidden_size
            if let Some(bias) = bias {
                iofc += &bias.slice(s!(0, 0..4 * hidden_size));
                iofc += &bias.slice(s!(0, 4 * hidden_size..8 * hidden_size));
            }
            //  dbg!(&iofc);
            let iofc = iofc.into_shape((batch_size, 4, hidden_size))?;
            // dbg!(&iofc);
            let i = op.f.eval(tvec!(iofc.slice_axis(Axis(1), (0..=0).into()).to_owned().into_arc_tensor()))?;
            // dbg!(&i);
            let o = op.f.eval(tvec!(iofc.slice_axis(Axis(1), (1..=1).into()).to_owned().into_arc_tensor()))?;
            let f = op.f.eval(tvec!(iofc.slice_axis(Axis(1), (2..=2).into()).to_owned().into_arc_tensor()))?;
            // dbg!(&iofc.slice_axis(Axis(1), (3..=3).into()));

            let c = op.g.eval(tvec!(iofc.slice_axis(Axis(1), (3..=3).into()).to_owned().into_arc_tensor()))?;
            // dbg!(&c);
            let i = i[0]
                .to_array_view::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix3>()?
                .into_shape((batch_size, hidden_size))?;
            let o = o[0]
                .to_array_view::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix3>()?
                .into_shape((batch_size, hidden_size))?;
            // dbg!(&o);

            let f = f[0]
                .to_array_view::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix3>()?
                .into_shape((batch_size, hidden_size))?;

            // dbg!(&c);
            let c = c[0]
                .to_array_view::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix3>()?
                .into_shape((batch_size, hidden_size))?;
            /*
                        dbg!(&f);
                        dbg!(&ct);
                        dbg!(&i);
                        dbg!(&c);
            */
            let big_c = f * &ct + i * c;
            // dbg!(&big_c);
            let big_h = o * op.h.eval(tvec!(big_c.clone().into_arc_tensor()))?[0]
                .to_array_view::<f32>()?
                .to_owned()
                .into_dimensionality::<Ix2>()?;
            ht.assign(&big_h);
            // dbg!(&big_h);
            ht_list.slice_axis_mut(Axis(0), (ix..=ix).into()).assign(&ht);
            // dbg!(&ht_list);
            ct.assign(&big_c);
        }
        let ht_list = ht_list.into_shape((seq_length, 1, batch_size, hidden_size))?;
        let hto = ht.to_owned().into_shape((1, batch_size, hidden_size))?;

        Ok(tvec!(ht_list.into_arc_tensor(), hto.into_arc_tensor()))
    }
}
