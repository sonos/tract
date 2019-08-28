use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn rnn(_ctx: &ParsingContext, pb: &NodeProto) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut rnn = RNN::default();

    let mut real_input_count = 3;
    let mut options = (3..6).map(|i| {
        pb.get_input().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_input_count += 1;
            real_input_count - 1
        })
    });
    rnn.optional_bias_input = options.next().unwrap();
    rnn.optional_sequence_lens_input = options.next().unwrap();
    rnn.optional_initial_h_input = options.next().unwrap();

    let mut real_output_count = 0;
    let mut options = (0..2).map(|i| {
        pb.get_output().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_output_count += 1;
            real_output_count - 1
        })
    });
    rnn.optional_y_output = options.next().unwrap();
    rnn.optional_y_h_output = options.next().unwrap();

    Ok((Box::new(rnn), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct RNN {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub fore: Box<dyn StatelessOp>,
    pub back: Box<dyn StatelessOp>,
}

impl Default for RNN {
    fn default() -> RNN {
        RNN {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            fore: Box::new(core_ops::math::Tanh::new(f32::datum_type().into())),
            back: Box::new(core_ops::math::Tanh::new(f32::datum_type().into())),
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
        let input_count = 3
            + self.optional_bias_input.is_some() as usize
            + self.optional_sequence_lens_input.is_some() as usize
            + self.optional_initial_h_input.is_some() as usize;
        check_input_arity(&inputs, input_count)?;
        let output_count =
            self.optional_y_output.is_some() as usize + self.optional_y_h_output.is_some() as usize;
        check_output_arity(&outputs, output_count)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[2].rank, 3)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?; // num_directions
        s.equals(&inputs[1].shape[1], &inputs[2].shape[1])?; // hidden_size
        s.equals(&inputs[1].shape[1], &inputs[2].shape[2])?; // hidden_size
        if let Some(bias) = self.optional_bias_input {
            s.equals(&inputs[bias].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[bias].rank, 2)?;
            s.equals(&inputs[bias].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[bias].shape[1], 2 * inputs[2].shape[2].bex())?; // 6 * hidden_size
        }
        if let Some(seq_len) = self.optional_sequence_lens_input {
            s.equals(&inputs[seq_len].rank, 1)?;
            s.equals(&inputs[seq_len].shape[0], &inputs[0].shape[1])?; // batch_size
        }
        if let Some(initial_h) = self.optional_initial_h_input {
            s.equals(&inputs[initial_h].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[initial_h].rank, 3)?;
            s.equals(&inputs[initial_h].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[initial_h].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&inputs[initial_h].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(y) = self.optional_y_output {
            s.equals(&outputs[y].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y].rank, 4)?;
            s.equals(&outputs[y].shape[0], &inputs[0].shape[0])?; // seq_lenght
            s.equals(&outputs[y].shape[1], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y].shape[2], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[y].shape[3], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(y_h) = self.optional_y_h_output {
            s.equals(&outputs[y_h].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y_h].rank, 3)?;
            s.equals(&outputs[y_h].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y_h].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[y_h].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}



impl StatelessOp for RNN {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, hidden_size, hidden_size]

        let bias = if let Some(ix) = self.optional_bias_input {
            Some(inputs[ix].to_array_view::<f32>()?.into_dimensionality::<Ix2>()?) // [num_directions, 6*hidden_size]
        } else {
            None
        };

        let seq_length = x.shape()[0];
        let batch_size = x.shape()[1];
        let num_directions = w.shape()[0];
        let hidden_size = r.shape()[2];

        let mut output_y = self.optional_y_output.map(|_| Array4::<f32>::zeros((seq_length, num_directions, batch_size, hidden_size)));
        let mut output_y_h = self.optional_y_h_output.map(|_| Array3::<f32>::zeros((num_directions, batch_size, hidden_size)));

        for dir in 0..num_directions {
            let w = w.index_axis_move(Axis(0), dir);
            let r = r.index_axis_move(Axis(0), dir);

            let mut ht = if let Some(init) = self.optional_initial_h_input {
                inputs[init].to_array_view::<f32>()?
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

                if let Some(ref mut o) = output_y {
                    o.index_axis_mut(Axis(0), ix).index_axis_move(Axis(0), dir).assign(&ht);
                }
            }
            if let Some(ref mut o) = output_y_h {
                o.index_axis_mut(Axis(0), dir).assign(&ht);
            }
        }

        let mut outputs = tvec!();
        outputs.extend(output_y.into_iter().map(|t| t.into_arc_tensor()));
        outputs.extend(output_y_h.into_iter().map(|t| t.into_arc_tensor()));
        Ok(outputs)
    }
}

impl TypedOp for RNN {
    typed_op_as_op!();

    fn output_facts(
        &self,
        inputs: &[&TypedTensorInfo],
    ) -> TractResult<TVec<TypedTensorInfo>> {
        let dt = inputs[0].datum_type;
        let seq_length = inputs[0].shape.dim(0);
        let num_directions = inputs[1].shape.dim(0);
        let batch_size = inputs[0].shape.dim(1);
        let hidden_size = inputs[2].shape.dim(2);
        let mut outputs = tvec!();
        if let Some(_) = self.optional_y_output {
            outputs.push(TypedTensorInfo::dt_shape(
                dt,
                [seq_length, num_directions.clone(), batch_size.clone(), hidden_size.clone()]
                    .as_ref(),
            )?)
        }
        if let Some(_) = self.optional_y_h_output {
            outputs.push(TypedTensorInfo::dt_shape(
                dt,
                [num_directions.clone(), batch_size.clone(), hidden_size.clone()].as_ref(),
            )?)
        }
        Ok(outputs)
    }
}
