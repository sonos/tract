use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn lstm(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut lstm = LSTM::default();

    let mut options = crate::model::optional_inputs(pb).skip(3);
    lstm.optional_bias_input = options.next().unwrap();
    lstm.optional_sequence_lens_input = options.next().unwrap();
    lstm.optional_initial_h_input = options.next().unwrap();
    lstm.optional_initial_c_input = options.next().unwrap();
    lstm.optional_p_input = options.next().unwrap();

    let mut options = crate::model::optional_outputs(pb);
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
    pub optional_initial_c_input: Option<usize>,
    pub optional_p_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub optional_y_c_output: Option<usize>,
    pub f: Box<dyn TypedOp>,
    pub g: Box<dyn TypedOp>,
    pub h: Box<dyn TypedOp>,
}

impl Default for LSTM {
    fn default() -> LSTM {
        LSTM {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_initial_c_input: None,
            optional_p_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            optional_y_c_output: None,
            f: Box::new(core_ops::nn::sigmoid()),
            g: Box::new(core_ops::math::tanh()),
            h: Box::new(core_ops::math::tanh()),
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

    not_a_typed_op!();
}

impl InferenceRulesOp for LSTM {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        let input_count = 3
            + self.optional_bias_input.is_some() as usize
            + self.optional_sequence_lens_input.is_some() as usize
            + self.optional_initial_h_input.is_some() as usize
            + self.optional_initial_c_input.is_some() as usize
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
        if let Some(initial_c) = self.optional_initial_c_input {
            s.equals(&inputs[initial_c].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[initial_c].rank, 3)?;
            s.equals(&inputs[initial_c].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[initial_c].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&inputs[initial_c].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(p) = self.optional_p_input {
            s.equals(&inputs[p].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[p].rank, 2)?;
            s.equals(&inputs[p].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[p].shape[1], 3 * inputs[2].shape[2].bex())?; // hidden_size
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
        if let Some(y_c) = self.optional_y_c_output {
            s.equals(&outputs[y_c].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y_c].rank, 3)?;
            s.equals(&outputs[y_c].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y_c].shape[1], &inputs[0].shape[1])?; // batch_size
            s.equals(&outputs[y_c].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        Ok(())
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.optional_y_output.is_some() as usize
            + self.optional_y_h_output.is_some() as usize
            + self.optional_y_c_output.is_some() as usize)
    }

    as_op!();

    #[allow(non_snake_case)]
    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::{array, math, matmul, scan};

        let x_fact = target.outlet_fact(mapping[&node.inputs[0]])?.clone();
        let r_fact = target.outlet_fact(mapping[&node.inputs[2]])?;

        let b_size = x_fact.shape.dim(1).to_integer().unwrap() as usize;
        let h_size = r_fact.shape.dim(2).to_integer().unwrap() as usize;

        // FIXME: bidi

        let mut body = TypedModel::default();
        let mut outer_inputs = vec![];
        let mut input_mapping = vec![];

        macro_rules! target_wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = target.wire_node(
                    format!("{}-{}", node.name, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = body.wire_node(
                    format!("{}-{}", node.name, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        // X: onnx interface: [seq_length, batch_size, input_size]
        // scan outer interface: idem
        // scann inner interface: [chunk=1, batch_size, input_size]
        // onnx inner interface: [batch_size, input_size]
        outer_inputs.push(mapping[&node.inputs[0]]);
        input_mapping.push(scan::InputMapping::Scan { slot: 0, axis: 0, chunk: 1.to_dim() });
        let mut x_source_fact = x_fact.clone();
        x_source_fact.shape.set_dim(0, 1.to_dim())?;
        let x_source = body.add_source("x_source", x_source_fact)?.into();
        wire!(Xt = AxisOp::Rm(0), x_source);

        // W: onnx interface: [num_directions, 4*hidden_size, input_size]
        // scan interfaces: [4*hidden_size, input_size]
        target_wire!(w = AxisOp::Rm(0), mapping[&node.inputs[1]]);
        outer_inputs.push(w);
        input_mapping.push(scan::InputMapping::Full { slot: 1 });
        let W = body.add_source("w", target.outlet_fact(w)?.clone())?.into();

        // R: onnx interface: [num_directions, 4*hidden_size, hidden_size]
        // scan interfaces: [4*hidden_size, hidden_size]
        target_wire!(r = AxisOp::Rm(0), mapping[&node.inputs[2]]);
        outer_inputs.push(r);
        input_mapping.push(scan::InputMapping::Full { slot: 2 });
        let R = body.add_source("r", target.outlet_fact(r)?.clone())?.into();

        // B: onnx interface: [num_directions, 8*hidden_size]
        let b = if let Some(slot) = self.optional_bias_input {
            target_wire!(b = AxisOp::Rm(0), mapping[&node.inputs[slot]]);
            outer_inputs.push(b);
            input_mapping.push(scan::InputMapping::Full { slot });
            let b = body.add_source("b", target.outlet_fact(b)?.clone())?.into();
            Some(b)
        } else {
            None
        };

        if let Some(slot) = self.optional_sequence_lens_input {
            outer_inputs.push(mapping[&node.inputs[slot]]);
        }

        // initial h, optional: onnx: [num_directions, batch_size, hidden_size]
        // scan outer: [chunk=1, batch_size, hidden_size]
        // scan inner: [chunk=1, batch_size, hidden_size]
        // onnx inner: [batch_size, hidden_size]
        let initializer = if let Some(initial_h_input) = self.optional_initial_h_input {
            target_wire!(h = AxisOp::Rm(0), mapping[&node.inputs[initial_h_input]]);
            target_wire!(h_chunk = AxisOp::Add(0), h);
            outer_inputs.push(h_chunk);
            scan::StateInitializer::FromInput(initial_h_input)
        } else {
            scan::StateInitializer::Value(
                ndarray::Array3::<f32>::zeros((1, b_size, h_size)).into_arc_tensor(),
            )
        };
        input_mapping.push(scan::InputMapping::State { initializer });
        let h_source = body
            .add_source(
                "h_source",
                TypedFact::dt_shape(x_fact.datum_type, [1, b_size, h_size].as_ref())?,
            )?
            .into();

        let initializer = if let Some(initial_c_input) = self.optional_initial_c_input {
            target_wire!(c = AxisOp::Rm(0), mapping[&node.inputs[initial_c_input]]);
            target_wire!(c_chunk = AxisOp::Add(0), c);
            outer_inputs.push(c_chunk);
            scan::StateInitializer::FromInput(initial_c_input)
        } else {
            scan::StateInitializer::Value(
                ndarray::Array3::<f32>::zeros((1, b_size, h_size)).into_arc_tensor(),
            )
        };
        input_mapping.push(scan::InputMapping::State { initializer });
        let c_source = body
            .add_source(
                "c_source",
                TypedFact::dt_shape(x_fact.datum_type, [1, b_size, h_size].as_ref())?,
            )?
            .into();

        // P: onnx [num_directions, 3*hidde_size]
        let p = if let Some(slot) = self.optional_p_input {
            target_wire!(p = AxisOp::Rm(0), mapping[&node.inputs[slot]]);
            outer_inputs.push(p);
            input_mapping.push(scan::InputMapping::Full { slot });
            let p = body.add_source("p", target.outlet_fact(p)?.clone())?.into();
            Some(p)
        } else {
            None
        };

        wire!(Ht_1 = AxisOp::Rm(0), h_source);
        wire!(Ct_1 = AxisOp::Rm(0), c_source);

        wire!(Wi = array::Slice::new(0, 0 * h_size, 1 * h_size), W);
        wire!(Wo = array::Slice::new(0, 1 * h_size, 2 * h_size), W);
        wire!(Wf = array::Slice::new(0, 2 * h_size, 3 * h_size), W);
        wire!(Wc = array::Slice::new(0, 3 * h_size, 4 * h_size), W);

        wire!(Ri = array::Slice::new(0, 0 * h_size, 1 * h_size), R);
        wire!(Ro = array::Slice::new(0, 1 * h_size, 2 * h_size), R);
        wire!(Rf = array::Slice::new(0, 2 * h_size, 3 * h_size), R);
        wire!(Rc = array::Slice::new(0, 3 * h_size, 4 * h_size), R);

        let biases = if let Some(b) = b {
            wire!(Wbi = array::Slice::new(0, 0 * h_size, 1 * h_size), b);
            wire!(Wbo = array::Slice::new(0, 1 * h_size, 2 * h_size), b);
            wire!(Wbf = array::Slice::new(0, 2 * h_size, 3 * h_size), b);
            wire!(Wbc = array::Slice::new(0, 3 * h_size, 4 * h_size), b);

            wire!(Rbi = array::Slice::new(0, 4 * h_size, 5 * h_size), b);
            wire!(Rbo = array::Slice::new(0, 5 * h_size, 6 * h_size), b);
            wire!(Rbf = array::Slice::new(0, 6 * h_size, 7 * h_size), b);
            wire!(Rbc = array::Slice::new(0, 7 * h_size, 8 * h_size), b);

            wire!(bi = math::add::bin_typed(), Wbi, Rbi);
            wire!(bo = math::add::bin_typed(), Wbo, Rbo);
            wire!(bf = math::add::bin_typed(), Wbf, Rbf);
            wire!(bc = math::add::bin_typed(), Wbc, Rbc);

            Some((bi, bo, bf, bc))
        } else {
            None
        };

        let peepholes = if let Some(p) = p {
            wire!(pi = array::Slice::new(0, 0 * h_size, 1 * h_size), p);
            wire!(po = array::Slice::new(0, 1 * h_size, 2 * h_size), p);
            wire!(pf = array::Slice::new(0, 2 * h_size, 3 * h_size), p);
            Some((pi, po, pf))
        } else {
            None
        };

        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        wire!(Xt_WiT = matmul::MatMul::default().with_b_trans(true), Xt, Wi);
        wire!(Ht_1_RiT = matmul::MatMul::default().with_b_trans(true), Ht_1, Ri);
        wire!(it0 = math::add::bin_typed(), Xt_WiT, Ht_1_RiT);
        let mut it0 = it0;
        if let Some(biases) = biases {
            wire!(it_bias = math::add::bin_typed(), it0, biases.0);
            it0 = it_bias;
        };
        if let Some(peephole) = peepholes {
            wire!(Pi_Ct_1 = math::mul::bin_typed(), peephole.0, Ct_1);
            wire!(it_peep = math::add::bin_typed(), Pi_Ct_1, it0);
            it0 = it_peep;
        }
        wire!(it = self.f.clone(), it0);

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        wire!(Xt_WfT = matmul::MatMul::default().with_b_trans(true), Xt, Wf);
        wire!(Ht_1_RfT = matmul::MatMul::default().with_b_trans(true), Ht_1, Rf);
        wire!(ft0 = math::add::bin_typed(), Xt_WfT, Ht_1_RfT);
        let mut ft0 = ft0;
        if let Some(biases) = biases {
            wire!(ft_bias = math::add::bin_typed(), ft0, biases.2);
            ft0 = ft_bias;
        };
        if let Some(peephole) = peepholes {
            wire!(Pf_Ct_1 = math::mul::bin_typed(), peephole.2, Ct_1);
            wire!(ft_peep = math::add::bin_typed(), Pf_Ct_1, ft0);
            ft0 = ft_peep;
        }
        wire!(ft = self.f.clone(), ft0);

        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        wire!(Xt_WcT = matmul::MatMul::default().with_b_trans(true), Xt, Wc);
        wire!(Ht_1_RcT = matmul::MatMul::default().with_b_trans(true), Ht_1, Rc);
        wire!(ct0 = math::add::bin_typed(), Xt_WcT, Ht_1_RcT);
        let mut ct0 = ct0;
        if let Some(biases) = biases {
            wire!(ct_bias = math::add::bin_typed(), ct0, biases.3);
            ct0 = ct_bias
        };
        wire!(ct = self.g.clone(), ct0);

        // Ct = ft (.) Ct-1 + it (.) ct
        wire!(ft_Ct_1 = math::mul::bin_typed(), ft, Ct_1);
        wire!(it_ct = math::mul::bin_typed(), it, ct);
        wire!(Ct = math::add::bin_typed(), ft_Ct_1, it_ct);

        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        wire!(Xt_WoT = matmul::MatMul::default().with_b_trans(true), Xt, Wo);
        wire!(Ht_1_RoT = matmul::MatMul::default().with_b_trans(true), Ht_1, Ro);
        wire!(ot0 = math::add::bin_typed(), Xt_WoT, Ht_1_RoT);
        let mut ot0 = ot0;
        if let Some(biases) = biases {
            wire!(ot_bias = math::add::bin_typed(), ot0, biases.1);
            ot0 = ot_bias
        };
        if let Some(peephole) = peepholes {
            wire!(Po_Ct = math::mul::bin_typed(), peephole.1, Ct);
            wire!(ot_peep = math::add::bin_typed(), Po_Ct, ot0);
            ot0 = ot_peep;
        }
        wire!(ot = self.f.clone(), ot0);

        // Ht = ot (.) h(Ct)
        wire!(h_Ct = self.h.clone(), Ct);
        wire!(Ht = math::mul::bin_typed(), ot, h_Ct);

        wire!(Ht_fixed = AxisOp::Add(0), Ht);
        wire!(Ct_fixed = AxisOp::Add(0), Ct);
        body.set_output_outlets(&[Ht_fixed, Ct_fixed])?;

        let h_mapping = scan::OutputMapping {
            state: true,
            axis: 0,
            chunk: 1.to_dim(),
            full_dim_hint: None,
            last_value_slot: self.optional_y_h_output,
            full_slot: self.optional_y_output,
        };
        let c_mapping = scan::OutputMapping {
            state: true,
            axis: 0,
            chunk: 1.to_dim(),
            full_dim_hint: None,
            last_value_slot: self.optional_y_c_output,
            full_slot: None,
        };

        let scan_outputs = target.wire_node(
            &*node.name,
            scan::TypedScan::new(
                body,
                input_mapping,
                vec![h_mapping, c_mapping],
                self.optional_sequence_lens_input,
            )?,
            &outer_inputs,
        )?;

        let mut result = tvec!();
        if let Some(slot) = self.optional_y_output {
            target_wire!(y = AxisOp::Add(0), scan_outputs[slot]);
            result.push(y);
        }
        if let Some(slot) = self.optional_y_h_output {
            result.push(scan_outputs[slot]);
        }
        if let Some(slot) = self.optional_y_c_output {
            result.push(scan_outputs[slot]);
        }

        Ok(result)
    }
}

impl StatelessOp for LSTM {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x: ArrayView3<f32> = inputs[0].to_array_view::<f32>()?.into_dimensionality()?; // [seq_length, batch_size, input_size]
        let w: ArrayView3<f32> = inputs[1].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, input_size]
        let r: ArrayView3<f32> = inputs[2].to_array_view::<f32>()?.into_dimensionality()?; // [num_directions, 4*hidden_size, hidden_size]

        let seq_length = x.shape()[0];
        let batch_size = x.shape()[1];
        let num_directions = w.shape()[0];
        let hidden_size = r.shape()[2];

        let bias = if let Some(ix) = self.optional_bias_input {
            Some(inputs[ix].to_array_view::<f32>()?.into_dimensionality::<Ix2>()?)
        // [num_directions, 6*hidden_size]
        } else {
            None
        };

        let peephole = if let Some(ix) = self.optional_p_input {
            Some(inputs[ix].to_array_view::<f32>()?.into_shape((num_directions, 3, hidden_size))?)
        } else {
            None
        };

        let mut output_y = self
            .optional_y_output
            .map(|_| Array4::<f32>::zeros((seq_length, num_directions, batch_size, hidden_size)));
        let mut output_y_h = self
            .optional_y_h_output
            .map(|_| Array3::<f32>::zeros((num_directions, batch_size, hidden_size)));
        let mut output_y_c = self
            .optional_y_c_output
            .map(|_| Array3::<f32>::zeros((num_directions, batch_size, hidden_size)));

        for dir in 0..num_directions {
            let w = w.index_axis_move(Axis(0), dir);
            let r = r.index_axis_move(Axis(0), dir);

            let mut ht = if let Some(init) = self.optional_initial_h_input {
                inputs[init]
                    .to_array_view::<f32>()?
                    .index_axis_move(Axis(0), dir)
                    .to_owned()
                    .into_dimensionality()?
            } else {
                Array2::<f32>::zeros((batch_size, hidden_size)).into()
            };

            let mut ct = if let Some(init) = self.optional_initial_c_input {
                inputs[init]
                    .to_array_view::<f32>()?
                    .index_axis_move(Axis(0), dir)
                    .to_owned()
                    .into_dimensionality()?
            } else {
                Array2::<f32>::zeros((batch_size, hidden_size)).into()
            };

            let peephole = peephole.map(|p| p.index_axis_move(Axis(0), dir));

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

                let mut i = iofc.index_axis(Axis(1), 0).to_owned();
                if let Some(peephole) = peephole {
                    i += &(ct.to_owned() * peephole.index_axis(Axis(0), 0));
                }
                let mut i = self.f.as_stateless().unwrap().eval(tvec!(i.into_arc_tensor()))?;
                let i = i
                    .pop()
                    .unwrap()
                    .into_tensor()
                    .into_array::<f32>()?
                    .into_shape((batch_size, hidden_size))?;

                let mut f = iofc.index_axis(Axis(1), 2).to_owned();
                if let Some(peephole) = peephole {
                    f += &(ct.to_owned() * peephole.index_axis(Axis(0), 2));
                }
                let mut f = self.f.as_stateless().unwrap().eval(tvec!(f.into_arc_tensor()))?;
                let f = f
                    .pop()
                    .unwrap()
                    .into_tensor()
                    .into_array::<f32>()?
                    .into_shape((batch_size, hidden_size))?;

                let c = iofc.index_axis(Axis(1), 3).to_owned();
                let mut c = self.g.as_stateless().unwrap().eval(tvec!(c.into_arc_tensor()))?;

                let c = c
                    .pop()
                    .unwrap()
                    .into_tensor()
                    .into_array::<f32>()?
                    .into_shape((batch_size, hidden_size))?;

                let big_c = f * &ct + i * c;

                let mut o = iofc.index_axis(Axis(1), 1).to_owned();
                if let Some(peephole) = peephole {
                    o += &(big_c.to_owned() * peephole.index_axis(Axis(0), 1));
                }
                let mut o = self.f.as_stateless().unwrap().eval(tvec!(o.into_arc_tensor()))?;
                let o = o
                    .pop()
                    .unwrap()
                    .into_tensor()
                    .into_array::<f32>()?
                    .into_shape((batch_size, hidden_size))?;

                let big_h = o * self
                    .h
                    .as_stateless()
                    .unwrap()
                    .eval(tvec!(big_c.clone().into_arc_tensor()))?[0]
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
