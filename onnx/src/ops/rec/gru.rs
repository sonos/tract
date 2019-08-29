use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn gru(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut gru = GRU::default();

    let mut real_input_count = 3;
    let mut options = (3..6).map(|i| {
        pb.get_input().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_input_count += 1;
            real_input_count - 1
        })
    });
    gru.optional_bias_input = options.next().unwrap();
    gru.optional_sequence_lens_input = options.next().unwrap();
    gru.optional_initial_h_input = options.next().unwrap();

    let mut real_output_count = 0;
    let mut options = (0..2).map(|i| {
        pb.get_output().get(i).filter(|s| !s.is_empty()).map(|_| {
            real_output_count += 1;
            real_output_count - 1
        })
    });
    gru.optional_y_output = options.next().unwrap();
    gru.optional_y_h_output = options.next().unwrap();

    Ok((Box::new(gru), vec![]))
}

#[derive(Debug, Clone, new)]
pub struct GRU {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub f: Box<dyn StatelessOp>,
    pub g: Box<dyn StatelessOp>,
    pub linear_before_reset: bool,
}

impl Default for GRU {
    fn default() -> GRU {
        GRU {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            f: Box::new(core_ops::nn::Sigmoid::new(f32::datum_type().into())),
            g: Box::new(core_ops::math::Tanh::new(f32::datum_type().into())),
            linear_before_reset: false,
        }
    }
}

impl Op for GRU {
    fn name(&self) -> Cow<str> {
        "GRU".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl InferenceRulesOp for GRU {
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
        s.equals(&inputs[1].shape[1], &inputs[2].shape[1])?; // 4*hidden_size
        s.equals(&inputs[2].shape[1], 3 * inputs[2].shape[2].bex())?; // hidden_size
        if let Some(bias) = self.optional_bias_input {
            s.equals(&inputs[bias].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[bias].rank, 2)?;
            s.equals(&inputs[bias].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[bias].shape[1], 6 * inputs[2].shape[2].bex())?; // 6 * hidden_size
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

impl TypedOp for GRU {
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
                [num_directions, batch_size, hidden_size].as_ref(),
            )?)
        }
        Ok(outputs)
    }
}

impl StatelessOp for GRU {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 3*hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 3*hidden_size, hidden_size]

        let bias = if let Some(ix) = self.optional_bias_input {
            Some(inputs[ix].to_array_view::<f32>()?.into_dimensionality::<Ix2>()?)
        // [num_directions, 6*hidden_size]
        } else {
            None
        };

        let seq_length = x.shape()[0];
        let batch_size = x.shape()[1];
        let num_directions = w.shape()[0];
        let hidden_size = r.shape()[2];

        let mut output_y = self
            .optional_y_output
            .map(|_| Array4::<f32>::zeros((seq_length, num_directions, batch_size, hidden_size)));
        let mut output_y_h = self
            .optional_y_h_output
            .map(|_| Array3::<f32>::zeros((num_directions, batch_size, hidden_size)));

        for dir in 0..num_directions {
            let w = w.index_axis_move(Axis(0), dir);
            let r = r.index_axis_move(Axis(0), dir);

            let mut ht = if let Some(ix) = self.optional_initial_h_input {
                inputs[ix]
                    .to_array_view::<f32>()?
                    .index_axis_move(Axis(0), dir)
                    .to_owned()
                    .into_dimensionality()?
            } else {
                Array2::<f32>::zeros((batch_size, hidden_size)).into()
            };

            for ix in 0..seq_length {
                let ix = if dir == 0 { ix } else { seq_length - 1 - ix };
                let x = x.index_axis_move(Axis(0), ix);

                // Xt*W_zrh^T + Wb_zrh
                let mut x_zrh = x.dot(&w.t()); // batch_size x 3*hidden_size
                if let Some(bias) = bias {
                    x_zrh += &bias.slice(s!(dir, 0..3 * hidden_size));
                }

                // Ht-1*R_zr
                let h_zr = ht.dot(&r.slice_axis(Axis(0), (0..2 * hidden_size).into()).t()); // batch_size x 3*hidden_size

                let x_zrh: Array3<f32> = x_zrh.into_shape((batch_size, 3, hidden_size))?;
                let h_zrh = h_zr.into_shape((batch_size, 2, hidden_size))?;

                let mut zt = x_zrh.index_axis(Axis(1), 0).to_owned() + h_zrh.index_axis(Axis(1), 0);
                if let Some(bias) = bias {
                    zt += &bias.slice(s!(dir, 3 * hidden_size..4 * hidden_size));
                }
                let zt: Array2<f32> = self
                    .f
                    .eval(tvec!(zt.into_arc_tensor()))?
                    .remove(0)
                    .into_tensor()
                    .into_array::<f32>()?
                    .into_dimensionality()?;

                let mut rt = x_zrh.index_axis(Axis(1), 1).to_owned() + h_zrh.index_axis(Axis(1), 1);
                if let Some(bias) = bias {
                    rt += &bias.slice(s!(dir, 4 * hidden_size..5 * hidden_size));
                }
                let rt = self
                    .f
                    .eval(tvec!(rt.into_arc_tensor()))?
                    .remove(0)
                    .into_tensor()
                    .into_array::<f32>()?;

                let ht1: Array2<f32> = if self.linear_before_reset {
                    let mut ht = ht.dot(&r.slice_axis(Axis(1), (2 * hidden_size..).into()).t());
                    if let Some(bias) = bias {
                        ht += &bias.slice(s!(dir, 5 * hidden_size..6 * hidden_size));
                    }
                    ht * rt + x_zrh.index_axis(Axis(1), 2)
                } else {
                    let mut ht =
                        ht.dot(&r.slice_axis(Axis(0), (2 * hidden_size..).into()).t()) * rt;
                    if let Some(bias) = bias {
                        ht += &bias.slice(s!(dir, 5 * hidden_size..6 * hidden_size));
                    }
                    ht + x_zrh.index_axis(Axis(1), 2)
                };
                let ht1 = self
                    .g
                    .eval(tvec!(ht1.into_arc_tensor()))?
                    .remove(0)
                    .into_tensor()
                    .into_array::<f32>()?;;

                ht = (1.0 - &zt) * ht1 + ht * &zt;

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
