use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;
use tract_hir::tract_core::ops::matmul::MatMulAxes;
use tract_hir::tract_core::ops::scan::ScanInfo;

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

    gru.linear_before_reset = pb.get_attr("linear_before_reset").unwrap_or(false);

    Ok((expand(gru), vec![]))
}

#[derive(Debug, Clone)]
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
        check_input_arity(inputs, input_count)?;
        let output_count =
            self.optional_y_output.is_some() as usize + self.optional_y_h_output.is_some() as usize;
        check_output_arity(outputs, output_count)?;
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
        let w_fact = target.outlet_fact(inputs[1])?;
        if w_fact.shape[0] == 2.into() {
            let fore = self.wire_one_side(&format!("{prefix}.fore"), target, inputs, 0)?;
            let back = self.wire_one_side(&format!("{prefix}.back"), target, inputs, 1)?;
            let mut outputs = tvec!(0.into(); self.nboutputs()?);
            if let Some(ix) = self.optional_y_output {
                outputs[ix] = target.wire_node(
                    format!("{prefix}.merge_y_output"),
                    TypedConcat::new(1),
                    &[fore[ix], back[ix]],
                )?[0];
            }
            if let Some(ix) = self.optional_y_h_output {
                outputs[ix] = target.wire_node(
                    format!("{prefix}.merge_y_h_output"),
                    TypedConcat::new(0),
                    &[fore[ix], back[ix]],
                )?[0];
            }
            if outputs.len() == 1 {
                target.set_outlet_label(outputs[0], prefix.into())?;
            }
            Ok(outputs)
        } else {
            self.wire_one_side(prefix, target, inputs, 0)
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
        let r_fact = target.outlet_fact(inputs[2])?.clone();

        let b_size = &x_fact.shape[1];
        let h_size = &r_fact.shape[2];

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
        }

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = body.wire_node(
                    format!("{}.{}", prefix, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        }

        // move batch first
        target_wire!(x_batch_first = AxisOp::Move(1, 0), inputs[0]);
        // X: onnx interface: [batch_size, seq_length, input_size]
        // scan outer interface: idem
        // scann inner interface: [chunk=1, batch_size, input_size]
        // onnx inner interface: [batch_size, input_size]
        outer_inputs.push(x_batch_first);
        input_mapping.push(scan::InputMapping::Scan(ScanInfo { slot: 0, axis: 1, chunk }));
        let mut x_source_fact = target.outlet_fact(x_batch_first)?.without_value();
        x_source_fact.shape.set(1, 1.to_dim());
        let x_source = body.add_source("x_source", x_source_fact)?;
        wire!(Xt = AxisOp::Rm(1), x_source);

        // W: onnx interface: [num_directions, 3*hidden_size, input_size]
        // scan interfaces: [3*hidden_size, input_size]
        target_wire!(w_dir = array::Slice::new(0, dir, dir + 1), inputs[1]);
        target_wire!(w = AxisOp::Rm(0), w_dir);
        outer_inputs.push(w);
        input_mapping.push(scan::InputMapping::Full { slot: 1 });
        let W = body.add_source("w", target.outlet_fact(w)?.clone())?;

        // R: onnx interface: [num_directions, 3*hidden_size, hidden_size]
        // scan interfaces: [3*hidden_size, hidden_size]
        target_wire!(r_dir = array::Slice::new(0, dir, dir + 1), inputs[2]);
        target_wire!(r = AxisOp::Rm(0), r_dir);
        outer_inputs.push(r);
        input_mapping.push(scan::InputMapping::Full { slot: 2 });
        let R = body.add_source("r", target.outlet_fact(r)?.clone())?;

        // B: onnx interface: [num_directions, 6*hidden_size]
        let b = if let Some(slot) = self.optional_bias_input {
            target_wire!(b_dir = array::Slice::new(0, dir, dir + 1), inputs[slot]);
            outer_inputs.push(b_dir);
            input_mapping.push(scan::InputMapping::Full { slot });
            let b = body.add_source("b", target.outlet_fact(b_dir)?.clone())?;
            Some(b)
        } else {
            None
        };

        if let Some(slot) = self.optional_sequence_lens_input {
            outer_inputs.push(inputs[slot]);
        }

        // initial h, optional: onnx: [num_directions, batch_size, hidden_size]
        // scan outer: [batch_size, chunk=1, hidden_size]
        // scan inner: [batch_size, chunk=1, hidden_size]
        // onnx inner: [batch_size, hidden_size]
        let initializer = if let Some(initial_h_input) = self.optional_initial_h_input {
            target_wire!(h_dir = array::Slice::new(0, dir, dir + 1), inputs[initial_h_input]);
            target_wire!(h = AxisOp::Rm(0), h_dir);
            target_wire!(h_chunk_ = AxisOp::Add(0), h);
            target_wire!(h_chunk = AxisOp::Move(1, 0), h_chunk_);
            outer_inputs.push(h_chunk);
            scan::StateInitializer::FromInput(initial_h_input)
        } else {
            scan::StateInitializer::Value(
                tensor0(0.0f32)
                    .broadcast_scalar_to_shape(&[
                        b_size.to_usize().unwrap(),
                        1,
                        h_size.to_usize().unwrap(),
                    ])?
                    .into_arc_tensor(),
            )
        };
        input_mapping.push(scan::InputMapping::State { initializer });
        let h_source = body.add_source(
            "h_source",
            x_fact.datum_type.fact(&[b_size.clone(), 1.to_dim(), h_size.clone()]),
        )?;

        wire!(Ht_1 = AxisOp::Rm(1), h_source);

        wire!(Rz = array::Slice::new(0, 0.to_dim() * h_size, 1.to_dim() * h_size), R);
        wire!(Rr = array::Slice::new(0, 1.to_dim() * h_size, 2.to_dim() * h_size), R);
        wire!(Rh = array::Slice::new(0, 2.to_dim() * h_size, 3.to_dim() * h_size), R);

        wire!(Wz = array::Slice::new(0, 0.to_dim() * h_size, 1.to_dim() * h_size), W);
        wire!(Wr = array::Slice::new(0, 1.to_dim() * h_size, 2.to_dim() * h_size), W);
        wire!(Wh = array::Slice::new(0, 2.to_dim() * h_size, 3.to_dim() * h_size), W);

        let matmul_t = matmul::MatMul { axes: MatMulAxes::default().transposing_b() };

        // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        wire!(Xt_WzT = matmul_t.clone(), Xt, Wz);
        wire!(Ht_1_RzT = matmul_t.clone(), Ht_1, Rz);
        wire!(zt0 = math::add(), Xt_WzT, Ht_1_RzT);
        let mut zt0 = zt0;
        if let Some(b) = b {
            wire!(Wbz = array::Slice::new(1, 0.to_dim() * h_size, 1.to_dim() * h_size), b);
            wire!(Rbz = array::Slice::new(1, 3.to_dim() * h_size, 4.to_dim() * h_size), b);
            wire!(Wbz_Rbz = math::add(), Wbz, Rbz);
            wire!(zt0_biased = math::add(), zt0, Wbz_Rbz);
            zt0 = zt0_biased
        };
        wire!(zt = self.f.clone(), zt0);

        // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        wire!(Xt_WrT = matmul_t.clone(), Xt, Wr);
        wire!(Ht_1_RrT = matmul_t.clone(), Ht_1, Rr);
        wire!(rt0 = math::add(), Xt_WrT, Ht_1_RrT);
        let mut rt0 = rt0;
        if let Some(b) = b {
            wire!(Wbr = array::Slice::new(1, 1.to_dim() * h_size, 2.to_dim() * h_size), b);
            wire!(Rbr = array::Slice::new(1, 4.to_dim() * h_size, 5.to_dim() * h_size), b);
            wire!(Wbr_Rbr = math::add(), Wbr, Rbr);
            wire!(rt0_biased = math::add(), rt0, Wbr_Rbr);
            rt0 = rt0_biased
        };
        wire!(rt = self.f.clone(), rt0);

        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        wire!(Xt_WhT = matmul_t.clone(), Xt, Wh);
        let rt_Ht_1_RhT_Rbh = if self.linear_before_reset {
            // rt (.) (Ht-1*(Rh^T) + Rbh)
            wire!(Ht_1_RhT = matmul_t, Ht_1, Rh);
            let Ht_1_RhT_Rbh = if let Some(b) = b {
                wire!(Rbh = array::Slice::new(1, 5.to_dim() * h_size, 6.to_dim() * h_size), b);
                wire!(Ht_1_RhT_Rbh = math::add(), Ht_1_RhT, Rbh);
                Ht_1_RhT_Rbh
            } else {
                Ht_1_RhT
            };
            wire!(rt_Ht_1_RhT_Rbh = math::mul(), rt, Ht_1_RhT_Rbh);
            rt_Ht_1_RhT_Rbh
        } else {
            // (rt (.) Ht-1)*(Rh^T) + Rbh
            wire!(rt_Ht_1 = math::mul(), rt, Ht_1);
            wire!(rt_Ht_1_RhT = matmul_t, rt_Ht_1, Rh);
            if let Some(b) = b {
                wire!(Rbh = array::Slice::new(1, 5.to_dim() * h_size, 6.to_dim() * h_size), b);
                wire!(rt_Ht_1_RhT_Rbh = math::add(), rt_Ht_1_RhT, Rbh);
                rt_Ht_1_RhT_Rbh
            } else {
                rt_Ht_1_RhT
            }
        };
        wire!(ht0 = math::add(), Xt_WhT, rt_Ht_1_RhT_Rbh);
        let mut ht0 = ht0;
        if let Some(b) = b {
            wire!(Wbh = array::Slice::new(1, 2.to_dim() * h_size, 3.to_dim() * h_size), b);
            wire!(ht0_biased = math::add(), ht0, Wbh);
            ht0 = ht0_biased
        }
        wire!(ht = self.g.clone(), ht0);

        // Ht = (1 - zt) (.) ht + zt (.) Ht-1
        let one: OutletId = body.add_const("one", tensor2(&[[1f32]]))?;
        wire!(one_sub_zt = math::sub(), one, zt);
        wire!(one_sub_zt_ht = math::mul(), one_sub_zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(Ht = math::add(), one_sub_zt_ht, zt_Ht_1);

        /*
        // Ht = ht + (- (zt (.) ht) + zt (.) Ht-1)
        wire!(zt_ht = math::mul(), zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(zt_Ht_1_sub_zt_ht = math::sub(), zt_Ht_1, zt_ht);
        wire!(Ht = math::add(), ht, zt_Ht_1_sub_zt_ht);
        */

        // Ht = ht - (zt (.) ht) + zt (.) Ht-1)
        /*
        wire!(zt_ht = math::mul(), zt, ht);
        wire!(zt_Ht_1 = math::mul(), zt, Ht_1);
        wire!(ht_zt_ht = math::sub(), ht, zt_ht);
        wire!(Ht = math::add(), ht_zt_ht, zt_Ht_1);
        */

        wire!(y_h = AxisOp::Add(1), Ht);
        body.set_output_outlets(&[y_h])?;

        let output_mapping = scan::OutputMapping {
            state: true,
            full_dim_hint: None,
            last_value_slot: self.optional_y_h_output,
            scan: self.optional_y_output.map(|slot| ScanInfo { slot, axis: 1, chunk }),
        };

        let scan_outputs = target.wire_node(
            prefix,
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
            target_wire!(y_batch_middle = AxisOp::Move(1, 0), scan_outputs[slot]);
            target_wire!(y = AxisOp::Add(1), y_batch_middle);
            result.push(y);
        }
        if let Some(slot) = self.optional_y_h_output {
            target_wire!(y_h_batch_middle = AxisOp::Move(1, 0), scan_outputs[slot]);
            result.push(y_h_batch_middle);
        }

        Ok(result)
    }
}
