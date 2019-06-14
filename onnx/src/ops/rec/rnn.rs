use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn rnn(_ctx: &ParsingContext, pb: &NodeProto) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let mut rnn = RNN::default();
    rnn.want_output_0_y = pb.get_output().get(0).map(|s| !s.is_empty()).unwrap_or(false);
    rnn.want_output_1_y_h = pb.get_output().get(1).map(|s| !s.is_empty()).unwrap_or(false);
    Ok((Box::new(rnn), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct RNN {
    pub want_output_0_y: bool,
    pub want_output_1_y_h: bool,
    pub fore: Box<StatelessOp>,
    pub back: Box<StatelessOp>,
    pub initial_h: Option<Tensor>,
}

impl Default for RNN {
    fn default() -> RNN {
        RNN {
            want_output_0_y: false,
            want_output_1_y_h: false,
            fore: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
            back: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
            initial_h: None,
        }
    }
}

impl Op for RNN {
    fn name(&self) -> Cow<str> {
        "RNN".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }
}

impl InferenceRulesOp for RNN {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[2].rank, 3)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?; // num_directions
        s.equals(&inputs[1].shape[1], &inputs[2].shape[1])?; // hidden_size
        s.equals(&inputs[1].shape[1], &inputs[2].shape[2])?; // hidden_size
        if inputs.len() > 3 {
            // bias
            s.equals(&inputs[3].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[3].rank, 2)?;
            s.equals(&inputs[3].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[3].shape[1], 2 * inputs[2].shape[2].bex())?; // 2 * hidden_size
        }
        if outputs.len() > 0 {
            s.equals(&outputs[0].rank, 4)?;
            s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?; // seq_lentgh
            s.equals(&outputs[0].shape[1], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[0].shape[2], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[0].shape[3], &inputs[2].shape[2])?; // hidden_size
        }
        if outputs.len() > 1 {
            s.equals(&inputs[0].datum_type, &outputs[1].datum_type)?;
            s.equals(&outputs[1].rank, 3)?;
            s.equals(&outputs[1].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[1].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[1].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        Ok(())
    }

    inference_op_as_op!();
}

impl StatelessOp for RNN {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, hidden_size, hidden_size]

        let bias = if let Some(bias) = inputs.get(3) {
            Some(bias.to_array_view::<f32>()?.into_dimensionality::<Ix2>()?) // [num_directions, 2*hidden_size]
        } else {
            None
        };

        let seq_length = x.shape()[0];
        let batch_size = x.shape()[1];
        let num_directions = w.shape()[0];
        let hidden_size = r.shape()[2];

        let mut output_0_y = if self.want_output_0_y {
            Some(Array4::<f32>::zeros((seq_length, num_directions, batch_size, hidden_size)))
        } else {
            None
        };

        let mut output_1_y_h = if self.want_output_1_y_h {
            Some(Array3::<f32>::zeros((num_directions, batch_size, hidden_size)))
        } else {
            None
        };

        for dir in 0..num_directions {
            let w = w.index_axis_move(Axis(0), dir);
            let r = r.index_axis_move(Axis(0), dir);

            let mut ht = if let Some(ref init) = self.initial_h {
                init.to_array_view::<f32>()?
                    .index_axis_move(Axis(0), dir)
                    .to_owned()
                    .into_dimensionality()?
            } else {
                Array2::<f32>::zeros((batch_size, hidden_size)).into()
            };

            for ix in 0..seq_length {
                let ix = if dir == 0 { ix } else { seq_length - 1 - ix };
                let x = x.index_axis_move(Axis(0), ix);

                let mut ht1 = x.dot(&w.t()) + ht.dot(&r.t()); // batch_size x 4*hidden_size
                if let Some(bias) = bias {
                    ht1 += &bias.slice(s!(dir, 0..hidden_size));
                    ht1 += &bias.slice(s!(dir, hidden_size..2 * hidden_size));
                }

                let ht1 = self.fore.eval(tvec!(ht1.into_arc_tensor()))?[0]
                    .to_array_view::<f32>()?
                    .to_owned()
                    .into_dimensionality::<Ix2>()?;
                ht.assign(&ht1);

                if let Some(ref mut o) = output_0_y {
                    o.index_axis_mut(Axis(0), ix).index_axis_move(Axis(0), dir).assign(&ht);
                }
            }
            if let Some(ref mut o) = output_1_y_h {
                o.index_axis_mut(Axis(0), dir).assign(&ht);
            }
        }

        let mut outputs = tvec!(
            output_0_y.map(|o| o.into_arc_tensor()),
            output_1_y_h.map(|o| o.into_arc_tensor()),
        );
        while outputs.len() > 0 && outputs.last().unwrap().is_none() {
            outputs.pop();
        }
        Ok(outputs.into_iter()
            .map(|t| t.unwrap_or(Tensor::from(0.0f32).into_arc_tensor()))
            .collect())
    }
}
