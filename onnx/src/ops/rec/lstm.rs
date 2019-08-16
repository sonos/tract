use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn lstm(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut lstm = LSTM::default();

    let mut real_input_count = 3;
    let mut options = (3..8).map(|i| {
        pb.get_input().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_input_count += 1;
            real_input_count - 1
        })
    });
    lstm.optional_bias_input = options.next().unwrap();
    lstm.optional_sequence_lens_input = options.next().unwrap();
    lstm.optional_initial_h_input = options.next().unwrap();
    lstm.optional_p_input = options.next().unwrap();

    let mut real_output_count = 0;
    let mut options = (0..3).map(|i| {
        pb.get_output().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_output_count += 1;
            real_output_count - 1
        })
    });
    lstm.optional_y_output = options.next().unwrap();
    lstm.optional_y_h_output = options.next().unwrap();
    lstm.optional_y_c_output = options.next().unwrap();

    Ok((Box::new(lstm), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct LSTM {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_p_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub optional_y_c_output: Option<usize>,
    pub f: Box<dyn StatelessOp>,
    pub g: Box<dyn StatelessOp>,
    pub h: Box<dyn StatelessOp>,
    pub initial_c: Option<Tensor>,
    pub initial_h: Option<Tensor>,
}

impl Default for LSTM {
    fn default() -> LSTM {
        LSTM {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_p_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            optional_y_c_output: None,
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

    fn validation(&self) -> Validation {
        Validation::Rounding
    }
}

impl InferenceRulesOp for LSTM {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let input_count = 3
            + self.optional_bias_input.is_some() as usize
            + self.optional_sequence_lens_input.is_some() as usize
            + self.optional_initial_h_input.is_some() as usize
            + self.optional_p_input.is_some() as usize;
        check_input_arity(&inputs, input_count)?;
        let output_count = self.optional_y_output.is_some() as usize
            + self.optional_y_h_output.is_some() as usize
            + self.optional_y_c_output.is_some() as usize;
        check_output_arity(&outputs, output_count)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[2].rank, 3)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?; // num_directions
        s.equals(&inputs[1].shape[1], &inputs[2].shape[1])?; // 4*hidden_size
        s.equals(&inputs[2].shape[1], 4 * inputs[2].shape[2].bex())?; // hidden_size
        if let Some(b) = self.optional_bias_input {
            // bias
            s.equals(&inputs[b].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[b].rank, 2)?;
            s.equals(&inputs[b].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[b].shape[1], 8 * inputs[2].shape[2].bex())?; // 8 * hidden_size
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
        if let Some(p) = self.optional_p_input {
            s.equals(&inputs[p].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[p].rank, 2)?;
            s.equals(&inputs[p].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[p].shape[1], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(y) = self.optional_y_output {
            s.equals(&outputs[y].rank, 4)?;
            s.equals(&outputs[y].shape[0], &inputs[0].shape[0])?; // seq_lentgh
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
        if let Some(y_c) = self.optional_y_h_output {
            s.equals(&outputs[y_c].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y_c].rank, 3)?;
            s.equals(&outputs[y_c].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y_c].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[y_c].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        Ok(())
    }

    inference_op_as_op!();
}

impl StatelessOp for LSTM {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, hidden_size]

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
        let mut output_y_c = self.optional_y_c_output.map(|_| Array3::<f32>::zeros((num_directions, batch_size, hidden_size)));

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

            let mut ct = if let Some(ref init) = self.initial_c {
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
                // x -> batch_size x input_size
                // Wt -> k=input_size x n=4*hidden_size
                // iofc -> batch_size x 4 * hidden_size

                let mut iofc = x.dot(&w.t()) + ht.dot(&r.t()); // batch_size x 4*hidden_size
                if let Some(bias) = bias {
                    iofc += &bias.slice(s!(dir, 0..4 * hidden_size));
                    iofc += &bias.slice(s!(dir, 4 * hidden_size..8 * hidden_size));
                }

                let iofc = iofc.into_shape((batch_size, 4, hidden_size))?;

                let i = self.f.eval(tvec!(iofc
                    .slice_axis(Axis(1), (0..=0).into())
                    .to_owned()
                    .into_arc_tensor()))?;

                let o = self.f.eval(tvec!(iofc
                    .slice_axis(Axis(1), (1..=1).into())
                    .to_owned()
                    .into_arc_tensor()))?;
                let f = self.f.eval(tvec!(iofc
                    .slice_axis(Axis(1), (2..=2).into())
                    .to_owned()
                    .into_arc_tensor()))?;

                let c = self.g.eval(tvec!(iofc
                    .slice_axis(Axis(1), (3..=3).into())
                    .to_owned()
                    .into_arc_tensor()))?;

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

                let f = f[0]
                    .to_array_view::<f32>()?
                    .to_owned()
                    .into_dimensionality::<Ix3>()?
                    .into_shape((batch_size, hidden_size))?;

                let c = c[0]
                    .to_array_view::<f32>()?
                    .to_owned()
                    .into_dimensionality::<Ix3>()?
                    .into_shape((batch_size, hidden_size))?;

                let big_c = f * &ct + i * c;

                let big_h = o * self.h.eval(tvec!(big_c.clone().into_arc_tensor()))?[0]
                    .to_array_view::<f32>()?
                    .to_owned()
                    .into_dimensionality::<Ix2>()?;
                ht.assign(&big_h);

                if let Some(ref mut o) = output_y {
                    o.index_axis_mut(Axis(0), ix).index_axis_move(Axis(0), dir).assign(&ht);
                }
                ct.assign(&big_c);
            }
            if let Some(ref mut o) = output_y_h {
                o.index_axis_mut(Axis(0), dir).assign(&ht);
            }
            if let Some(ref mut o) = output_y_c {
                o.index_axis_mut(Axis(0), dir).assign(&ct);
            }
        }

        let mut outputs = tvec!();
        outputs.extend(output_y.into_iter().map(|t| t.into_arc_tensor()));
        outputs.extend(output_y_h.into_iter().map(|t| t.into_arc_tensor()));
        outputs.extend(output_y_c.into_iter().map(|t| t.into_arc_tensor()));
        Ok(outputs)
    }
}
