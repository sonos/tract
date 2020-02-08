use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::*;
use tract_core::ops as core_ops;

pub fn gru(
    _ctx: &ParsingContext,
    pb: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut gru = GRU::default();

    let mut options = crate::model::optional_inputs(pb).skip(3);
    gru.optional_bias_input = options.next().unwrap();
    gru.optional_sequence_lens_input = options.next().unwrap();
    gru.optional_initial_h_input = options.next().unwrap();

    let mut options = crate::model::optional_outputs(pb);
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
    pub f: Box<dyn TypedOp>,
    pub g: Box<dyn TypedOp>,
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
            f: Box::new(core_ops::nn::sigmoid()),
            g: Box::new(core_ops::math::tanh()),
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

    not_a_typed_op!();
}

impl InferenceRulesOp for GRU {
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

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.optional_y_output.is_some() as usize + self.optional_y_h_output.is_some() as usize)
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

        // W: onnx interface: [num_directions, 3*hidden_size, input_size]
        // scan interfaces: [3*hidden_size, input_size]
        target_wire!(w = AxisOp::Rm(0), mapping[&node.inputs[1]]);
        outer_inputs.push(w);
        input_mapping.push(scan::InputMapping::Full { slot: 1 });
        let W = body.add_source("w", target.outlet_fact(w)?.clone())?.into();

        // R: onnx interface: [num_directions, 3*hidden_size, hidden_size]
        // scan interfaces: [3*hidden_size, hidden_size]
        target_wire!(r = AxisOp::Rm(0), mapping[&node.inputs[2]]);
        outer_inputs.push(r);
        input_mapping.push(scan::InputMapping::Full { slot: 2 });
        let R = body.add_source("r", target.outlet_fact(r)?.clone())?.into();

        // B: onnx interface: [num_directions, 6*hidden_size]
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

        wire!(Ht_1 = AxisOp::Rm(0), h_source);

        wire!(Rz = array::Slice::new(0, 0 * h_size, 1 * h_size), R);
        wire!(Rr = array::Slice::new(0, 1 * h_size, 2 * h_size), R);
        wire!(Rh = array::Slice::new(0, 2 * h_size, 3 * h_size), R);

        wire!(Wz = array::Slice::new(0, 0 * h_size, 1 * h_size), W);
        wire!(Wr = array::Slice::new(0, 1 * h_size, 2 * h_size), W);
        wire!(Wh = array::Slice::new(0, 2 * h_size, 3 * h_size), W);

        // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        wire!(Xt_WzT = matmul::MatMul::default().with_b_trans(true), Xt, Wz);
        wire!(Ht_1_RzT = matmul::MatMul::default().with_b_trans(true), Ht_1, Rz);
        wire!(zt0 = math::add::bin_typed(), Xt_WzT, Ht_1_RzT);
        let mut zt0 = zt0;
        if let Some(b) = b {
            wire!(Wbz = array::Slice::new(0, 0 * h_size, 1 * h_size), b);
            wire!(Rbz = array::Slice::new(0, 3 * h_size, 4 * h_size), b);
            wire!(Wbz_Rbz = math::add::bin_typed(), Wbz, Rbz);
            wire!(zt0_biased = math::add::bin_typed(), zt0, Wbz_Rbz);
            zt0 = zt0_biased
        };
        wire!(zt = self.f.clone(), zt0);

        // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        wire!(Xt_WrT = matmul::MatMul::default().with_b_trans(true), Xt, Wr);
        wire!(Ht_1_RrT = matmul::MatMul::default().with_b_trans(true), Ht_1, Rr);
        wire!(rt0 = math::add::bin_typed(), Xt_WrT, Ht_1_RrT);
        let mut rt0 = rt0;
        if let Some(b) = b {
            wire!(Wbr = array::Slice::new(0, 1 * h_size, 2 * h_size), b);
            wire!(Rbr = array::Slice::new(0, 4 * h_size, 5 * h_size), b);
            wire!(Wbr_Rbr = math::add::bin_typed(), Wbr, Rbr);
            wire!(rt0_biased = math::add::bin_typed(), rt0, Wbr_Rbr);
            rt0 = rt0_biased
        };
        wire!(rt = self.f.clone(), rt0);

        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        wire!(Xt_WhT = matmul::MatMul::default().with_b_trans(true), Xt, Wh);
        let rt_Ht_1_RhT = if self.linear_before_reset {
            wire!(Ht_1_RhT = matmul::MatMul::default().with_b_trans(true), Ht_1, Rh);
            wire!(rt_Ht_1_RhT = math::mul::bin_typed(), rt, Ht_1_RhT);
            rt_Ht_1_RhT
        } else {
            wire!(rt_Ht_1 = math::mul::bin_typed(), rt, Ht_1);
            wire!(rt_Ht_1_RhT = matmul::MatMul::default().with_b_trans(true), rt_Ht_1, Rh);
            rt_Ht_1_RhT
        };
        wire!(ht0 = math::add::bin_typed(), Xt_WhT, rt_Ht_1_RhT);
        let mut ht0 = ht0;
        if let Some(b) = b {
            wire!(Wbh = array::Slice::new(0, 2 * h_size, 3 * h_size), b);
            wire!(Rbh = array::Slice::new(0, 5 * h_size, 6 * h_size), b);
            wire!(Wbh_Rbh = math::add::bin_typed(), Wbh, Rbh);
            wire!(ht0_biased = math::add::bin_typed(), ht0, Wbh_Rbh);
            ht0 = ht0_biased
        }
        wire!(ht = self.g.clone(), ht0);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        let one: OutletId = body.add_const("one", tensor0(1f32))?.into();
        wire!(one_sub_zt = math::sub::bin_typed(), one, zt);
        wire!(one_sub_zt_ht = math::mul::bin_typed(), one_sub_zt, ht);
        wire!(zt_Ht_1 = math::mul::bin_typed(), zt, Ht_1);
        wire!(Ht = math::add::bin_typed(), one_sub_zt_ht, zt_Ht_1);

        wire!(y_h = AxisOp::Add(0), Ht);
        body.set_output_outlets(&[y_h])?;

        let output_mapping = scan::OutputMapping {
            state: true,
            axis: 0,
            chunk: 1.to_dim(),
            full_dim_hint: None,
            last_value_slot: self.optional_y_h_output,
            full_slot: self.optional_y_output,
        };

        let scan_outputs = target.wire_node(
            &*node.name,
            scan::TypedScan::new(
                body,
                input_mapping,
                vec![output_mapping],
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

        Ok(result)
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
                    .as_stateless()
                    .unwrap()
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
                    .as_stateless()
                    .unwrap()
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
                    .as_stateless()
                    .unwrap()
                    .eval(tvec!(ht1.into_arc_tensor()))?
                    .remove(0)
                    .into_tensor()
                    .into_array::<f32>()?;

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
