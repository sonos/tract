use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;

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

    Ok((expand(gru), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
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

tract_data::impl_dyn_hash!(GRU);

impl Default for GRU {
    fn default() -> GRU {
        GRU {
            optional_bias_input: None,
            optional_sequence_lens_input: None,
            optional_initial_h_input: None,
            optional_y_output: None,
            optional_y_h_output: None,
            f: Box::new(ops::nn::sigmoid()),
            g: Box::new(ops::math::tanh()),
            linear_before_reset: false,
        }
    }
}

impl Expansion for GRU {
    fn name(&self) -> Cow<str> {
        "GRU".into()
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

impl GRU {
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
        target_wire!(w_dir = array::Slice::new(0, dir, dir + 1), inputs[1]);
        target_wire!(w = AxisOp::Rm(0), w_dir);
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
            target_wire!(h = AxisOp::Rm(0), inputs[initial_h_input]);
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
            wire!(Wbz = array::Slice::new(1, 0 * h_size, 1 * h_size), b);
            wire!(Rbz = array::Slice::new(1, 3 * h_size, 4 * h_size), b);
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
            wire!(Wbr = array::Slice::new(1, 1 * h_size, 2 * h_size), b);
            wire!(Rbr = array::Slice::new(1, 4 * h_size, 5 * h_size), b);
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
            wire!(Wbh = array::Slice::new(1, 2 * h_size, 3 * h_size), b);
            wire!(Rbh = array::Slice::new(1, 5 * h_size, 6 * h_size), b);
            wire!(Wbh_Rbh = math::add::bin_typed(), Wbh, Rbh);
            wire!(ht0_biased = math::add::bin_typed(), ht0, Wbh_Rbh);
            ht0 = ht0_biased
        }
        wire!(ht = self.g.clone(), ht0);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        let one: OutletId = body.add_const("one", tensor2(&[[1f32]]))?.into();
        wire!(one_sub_zt = math::sub::bin_typed(), one, zt);
        wire!(one_sub_zt_ht = math::mul::bin_typed(), one_sub_zt, ht);
        wire!(zt_Ht_1 = math::mul::bin_typed(), zt, Ht_1);
        wire!(Ht = math::add::bin_typed(), one_sub_zt_ht, zt_Ht_1);

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
            &*prefix,
            ops::scan::Scan::new(
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
