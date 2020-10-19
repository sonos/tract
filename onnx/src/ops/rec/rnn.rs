use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;

pub fn rnn(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut rnn = RNN::default();

    let mut options = crate::model::optional_inputs(pb).skip(3);
    rnn.optional_bias_input = options.next().unwrap();
    rnn.optional_sequence_lens_input = options.next().unwrap();
    rnn.optional_initial_h_input = options.next().unwrap();

    let mut options = crate::model::optional_outputs(pb);
    rnn.optional_y_output = options.next().unwrap();
    rnn.optional_y_h_output = options.next().unwrap();

    Ok((expand(rnn), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
pub struct RNN {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub fore: Box<dyn TypedOp>,
    pub back: Box<dyn TypedOp>,
}

tract_data::impl_dyn_hash!(RNN);

impl Default for RNN {
    fn default() -> RNN {
        RNN {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            fore: Box::new(ops::math::tanh()),
            back: Box::new(ops::math::tanh()),
        }
    }
}

impl Expansion for RNN {
    fn name(&self) -> Cow<str> {
        "RNN".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
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
            s.equals(&inputs[bias].shape[1], 2 * inputs[2].shape[2].bex())?; // 2 * hidden_size
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

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.optional_y_output.is_some() as usize + self.optional_y_h_output.is_some() as usize)
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::tract_core::ops::array::TypedConcat;
        let fore = self.wire_one_side(prefix, target, inputs, 0)?;
        let w_fact = target.outlet_fact(inputs[1])?;
        if w_fact.shape[0] == 2.into() {
            let back = self.wire_one_side(&format!("{}.back", prefix), target, inputs, 1)?;
            let mut outputs = tvec!(0.into(); self.nboutputs()?);
            if let Some(ix) = self.optional_y_output {
                outputs[ix] = target.wire_node(
                    format!("{}.merge_y_output", prefix),
                    TypedConcat::concat_vars(1, 2),
                    &[fore[ix], back[ix]],
                )?[0];
            }
            if let Some(ix) = self.optional_y_h_output {
                outputs[ix] = target.wire_node(
                    format!("{}.merge_y_h_output", prefix),
                    TypedConcat::concat_vars(0, 2),
                    &[fore[ix], back[ix]],
                )?[0];
            }
            Ok(outputs)
        } else {
            Ok(fore)
        }
    }
}

impl RNN {
    #[allow(non_snake_case)]
    fn wire_one_side(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
        dir: usize,
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::{array, math, matmul, scan};

        let x_fact = target.outlet_fact(inputs[0])?.clone();
        let r_fact = target.outlet_fact(inputs[2])?;

        let b_size = x_fact.shape[1].to_usize().unwrap();
        let h_size = r_fact.shape[2].to_usize().unwrap();

        let chunk = if dir == 0 { 1 } else { -1 };

        let mut body = TypedModel::default();
        let mut outer_inputs = vec![];
        let mut input_mapping = vec![];

        macro_rules! target_wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = target.wire_node(
                    format!("{}.{}", prefix, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = body.wire_node(
                    format!("{}.{}", prefix, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        // X: onnx interface: [seq_length, batch_size, input_size]
        // scan outer interface: idem
        // scann inner interface: [chunk=1, batch_size, input_size]
        // onnx inner interface: [batch_size, input_size]
        outer_inputs.push(inputs[0]);
        input_mapping.push(scan::InputMapping::Scan { slot: 0, axis: 0, chunk });
        let mut x_source_fact = x_fact.without_value();
        x_source_fact.shape[0] = 1.to_dim();
        let x_source = body.add_source("x_source", x_source_fact)?.into();
        wire!(Xt = AxisOp::Rm(0), x_source);

        // W: onnx interface: [num_directions, 3*hidden_size, input_size]
        // scan interfaces: [3*hidden_size, input_size]
        target_wire!(w = AxisOp::Rm(0), inputs[1]);
        outer_inputs.push(w);
        input_mapping.push(scan::InputMapping::Full { slot: 1 });
        let W = body.add_source("w", target.outlet_fact(w)?.clone())?.into();

        // R: onnx interface: [num_directions, 3*hidden_size, hidden_size]
        // scan interfaces: [3*hidden_size, hidden_size]
        target_wire!(r_dir = array::Slice::new(0, dir, dir + 1), inputs[2]);
        target_wire!(r = AxisOp::Rm(0), r_dir);
        outer_inputs.push(r);
        input_mapping.push(scan::InputMapping::Full { slot: 2 });
        let R = body.add_source("r", target.outlet_fact(r)?.clone())?.into();

        // B: onnx interface: [num_directions, 6*hidden_size]
        let b = if let Some(slot) = self.optional_bias_input {
            target_wire!(b_dir = array::Slice::new(0, dir, dir + 1), inputs[slot]);
            outer_inputs.push(b_dir);
            input_mapping.push(scan::InputMapping::Full { slot });
            let b = body.add_source("b", target.outlet_fact(b_dir)?.clone())?.into();
            Some(b)
        } else {
            None
        };

        if let Some(slot) = self.optional_sequence_lens_input {
            outer_inputs.push(inputs[slot]);
        }

        // initial h, optional: onnx: [num_directions, batch_size, hidden_size]
        // scan outer: [chunk=1, batch_size, hidden_size]
        // scan inner: [chunk=1, batch_size, hidden_size]
        // onnx inner: [batch_size, hidden_size]
        let initializer = if let Some(initial_h_input) = self.optional_initial_h_input {
            target_wire!(h_dir = array::Slice::new(0, dir, dir+1), inputs[initial_h_input]);
            target_wire!(h = AxisOp::Rm(0), h_dir);
            target_wire!(h_chunk = AxisOp::Add(0), h);
            outer_inputs.push(h_chunk);
            scan::StateInitializer::FromInput(initial_h_input)
        } else {
            scan::StateInitializer::Value(
                tract_ndarray::Array3::<f32>::zeros((1, b_size, h_size)).into_arc_tensor(),
            )
        };
        input_mapping.push(scan::InputMapping::State { initializer });
        let h_source = body
            .add_source(
                "h_source",
                TypedFact::dt_shape(x_fact.datum_type, [1, b_size, h_size].as_ref())?,
            )?
            .into();

        wire!(Ht_1 = AxisOp::Rm(0), h_source);

        let bias = if let Some(b) = b {
            wire!(Wbi = array::Slice::new(1, 0 * h_size, 1 * h_size), b);
            wire!(Rbi = array::Slice::new(1, 1 * h_size, 2 * h_size), b);
            wire!(bi = math::add::bin_typed(), Wbi, Rbi);
            Some(bi)
        } else {
            None
        };

        // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        wire!(Xt_WiT = matmul::MatMul::default().with_b_trans(true), Xt, W);
        wire!(Ht_1_RiT = matmul::MatMul::default().with_b_trans(true), Ht_1, R);

        wire!(ht0 = math::add::bin_typed(), Xt_WiT, Ht_1_RiT);
        let mut ht0 = ht0;
        if let Some(bias) = bias {
            wire!(ht_bias = math::add::bin_typed(), ht0, bias);
            ht0 = ht_bias;
        }
        wire!(Ht = self.fore.clone(), ht0);

        wire!(y_h = AxisOp::Add(0), Ht);
        body.set_output_outlets(&[y_h])?;

        let output_mapping = scan::OutputMapping {
            state: true,
            axis: 0,
            chunk,
            full_dim_hint: None,
            last_value_slot: self.optional_y_h_output,
            full_slot: self.optional_y_output,
        };

        let scan_outputs = target.wire_node(
            prefix,
            scan::Scan::new(
                body,
                input_mapping,
                vec![output_mapping],
                self.optional_sequence_lens_input,
                0,
            )?,
            &outer_inputs,
        )?;

        let mut result = tvec!();
        if let Some(slot) = self.optional_y_output {
            target_wire!(y = AxisOp::Add(1), scan_outputs[slot]);
            result.push(y);
        }
        if let Some(slot) = self.optional_y_h_output {
            result.push(scan_outputs[slot]);
        }

        Ok(result)
    }
}
