use std::fmt::Debug;

use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::tract_core::dyn_clone::{clone_trait_object, DynClone};
use tract_hir::tract_core::ops::scan::ScanInfo;

pub trait WireBody: Debug + DynClone + Send + Sync {
    fn name(&self) -> &'static str;
    fn wire_body(&self, prefix: &str, body: &mut TypedModel) -> TractResult<()>;
    fn w_b_multipliers(&self) -> (usize, usize);
    fn have_extra_c_state(&self) -> bool;
}

clone_trait_object!(WireBody);

#[derive(Debug, Clone)]
pub struct CommonRec {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_initial_c_input: Option<usize>,
    pub optional_p_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub optional_y_c_output: Option<usize>,
    pub batch_first: bool,
    pub body: Box<dyn WireBody>,
}

impl CommonRec {
    pub fn from_node_and_options(
        pb: &NodeProto,
        fixed_input: usize,
        fixed_outputs: usize,
        body: Box<dyn WireBody>,
    ) -> TractResult<Self> {
        let mut inputs = crate::model::optional_inputs(pb).skip(fixed_input);
        let mut outputs = crate::model::optional_outputs(pb).skip(fixed_outputs);
        Ok(Self {
            optional_bias_input: inputs.next().unwrap(),
            optional_sequence_lens_input: inputs.next().unwrap(),
            optional_initial_h_input: inputs.next().unwrap(),
            optional_initial_c_input: inputs.next().unwrap(),
            optional_p_input: inputs.next().unwrap(),

            optional_y_output: outputs.next().unwrap(),
            optional_y_h_output: outputs.next().unwrap(),
            optional_y_c_output: outputs.next().unwrap(),

            batch_first: pb.get_attr_opt("layout")?.unwrap_or(0) == 1,
            body,
        })
    }

    #[allow(non_snake_case)]
    fn wire_one_side(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
        dir: usize,
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::{array, scan};

        let x_fact = target.outlet_fact(inputs[0])?.clone();
        let r_fact = target.outlet_fact(inputs[2])?.clone();

        if let Some(seqlen) = self.optional_sequence_lens_input {
            let Some(seqlen) = &target.outlet_fact(inputs[seqlen])?.konst else {
                bail!("Non constant seq_len is not supported");
            };
            let Some(seqlen) = seqlen.as_uniform() else {
                bail!("Non uniform seq_len is not supported");
            };
            let seqlen = seqlen.cast_to::<TDim>()?;
            if seqlen.to_scalar::<TDim>()? != &x_fact.shape[self.batch_first as usize] {
                bail!("seq_len only supported for trivial noop case");
            };
        }

        let b_size = &x_fact.shape[1 - self.batch_first as usize];
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
                #[allow(unused_variables)]
                let $name = body.wire_node(
                    stringify!($name),
                    $op, [$($param),*].as_ref())?[0];
            }
        }

        // X: onnx interface: [batch_size, seq_length, input_size]
        // move batch first
        let x_batch_first = if self.batch_first {
            inputs[0]
        } else {
            target_wire!(x_batch_first = AxisOp::Move(1, 0), inputs[0]);
            x_batch_first
        };
        // scan outer interface: idem
        // scann inner interface: [chunk=1, batch_size, input_size]
        // onnx inner interface: [batch_size, input_size]
        outer_inputs.push(x_batch_first);
        input_mapping.push(scan::InputMapping::Scan(ScanInfo { axis: 1, chunk }));
        let mut x_source_fact = target.outlet_fact(x_batch_first)?.without_value();
        x_source_fact.shape.set(1, 1.to_dim());
        let x_source = body.add_source("x_source", x_source_fact)?;
        wire!(Xt = AxisOp::Rm(1), x_source);

        // W: onnx interface: [num_directions, 3*hidden_size, input_size]
        // scan interfaces: [3*hidden_size, input_size]
        target_wire!(w_dir = array::Slice::new(0, dir, dir + 1), inputs[1]);
        target_wire!(w = AxisOp::Rm(0), w_dir);
        outer_inputs.push(w);
        input_mapping.push(scan::InputMapping::Full);
        body.add_source("W", target.outlet_fact(w)?.clone())?;

        // R: onnx interface: [num_directions, 3*hidden_size, hidden_size]
        // scan interfaces: [3*hidden_size, hidden_size]
        target_wire!(r_dir = array::Slice::new(0, dir, dir + 1), inputs[2]);
        target_wire!(r = AxisOp::Rm(0), r_dir);
        outer_inputs.push(r);
        input_mapping.push(scan::InputMapping::Full);
        body.add_source("R", target.outlet_fact(r)?.clone())?;

        // B: onnx interface: [num_directions, 6*hidden_size]
        if let Some(slot) = self.optional_bias_input {
            target_wire!(b_dir = array::Slice::new(0, dir, dir + 1), inputs[slot]);
            outer_inputs.push(b_dir);
            input_mapping.push(scan::InputMapping::Full);
            let b = body.add_source("b", target.outlet_fact(b_dir)?.clone())?;
            Some(b)
        } else {
            None
        };

        // initial h, optional: onnx: [num_directions, batch_size, hidden_size]
        // scan outer: [batch_size, chunk=1, hidden_size]
        // scan inner: [batch_size, chunk=1, hidden_size]
        // onnx inner: [batch_size, hidden_size]
        let initializer = if let Some(initial_h_input) = self.optional_initial_h_input {
            let mut input = inputs[initial_h_input];
            if self.batch_first {
                target_wire!(h_batch_first = AxisOp::Move(1, 0), input);
                input = h_batch_first;
            };
            target_wire!(h_dir = array::Slice::new(0, dir, dir + 1), input);
            target_wire!(h = AxisOp::Rm(0), h_dir);
            target_wire!(h_chunk_ = AxisOp::Add(0), h);
            target_wire!(h_chunk = AxisOp::Move(1, 0), h_chunk_);
            h_chunk
        } else {
            target.add_const(
                format!("{prefix}.h0"),
                tensor0(0.0f32)
                    .broadcast_scalar_to_shape(&[
                        b_size.to_usize().unwrap(),
                        1,
                        h_size.to_usize().unwrap(),
                    ])?
                    .into_arc_tensor(),
            )?
        };
        outer_inputs.push(initializer);
        input_mapping.push(scan::InputMapping::State);

        let h_source = body.add_source(
            "h_source",
            x_fact.datum_type.fact(&[b_size.clone(), 1.to_dim(), h_size.clone()]),
        )?;
        wire!(Ht_1 = AxisOp::Rm(1), h_source);

        if self.body.have_extra_c_state() {
            let initializer = if let Some(initial_c_input) = self.optional_initial_c_input {
                let mut input = inputs[initial_c_input];
                if self.batch_first {
                    target_wire!(c_batch_first = AxisOp::Move(1, 0), input);
                    input = c_batch_first;
                };
                target_wire!(c_dir = array::Slice::new(0, dir, dir + 1), input);
                target_wire!(c = AxisOp::Rm(0), c_dir);
                target_wire!(c_chunk_ = AxisOp::Add(0), c);
                target_wire!(c_chunk = AxisOp::Move(1, 0), c_chunk_);
                c_chunk
            } else {
                target.add_const(
                    format!("{prefix}.c0"),
                    tensor0(0.0f32)
                        .broadcast_scalar_to_shape(&[
                            b_size.to_usize().unwrap(),
                            1,
                            h_size.to_usize().unwrap(),
                        ])?
                        .into_arc_tensor(),
                )?
            };
            outer_inputs.push(initializer);
            input_mapping.push(scan::InputMapping::State);
            let c_source = body.add_source(
                "c_source",
                x_fact.datum_type.fact(&[b_size.clone(), 1.to_dim(), h_size.clone()]),
            )?;
            wire!(Ct_1 = AxisOp::Rm(1), c_source);
        }

        // P: onnx [num_directions, 3*hidde_size]
        if let Some(slot) = self.optional_p_input {
            target_wire!(p = array::Slice::new(0, dir, dir + 1), inputs[slot]);
            outer_inputs.push(p);
            input_mapping.push(scan::InputMapping::Full);
            body.add_source("peepholes", target.outlet_fact(p)?.clone())?;
        };

        self.body.wire_body(prefix, &mut body).context("Wiring body")?;

        let mut output_mapping = vec![scan::OutputMapping {
            state: true,
            full_dim_hint: None,
            last_value_slot: self.optional_y_h_output,
            scan: self.optional_y_output.map(|slot| (slot, ScanInfo { axis: 1, chunk })),
        }];
        if self.body.have_extra_c_state() {
            output_mapping.push(scan::OutputMapping {
                state: true,
                full_dim_hint: None,
                last_value_slot: self.optional_y_c_output,
                scan: None,
            });
        }

        let scan_outputs = target.wire_node(
            prefix,
            tract_core::ops::scan::Scan::new(body, input_mapping, output_mapping, 0)?,
            &outer_inputs,
        )?;

        let mut result = tvec!();
        if let Some(slot) = self.optional_y_output {
            // scan: [batch_size, seq_len, hidden_size]
            if self.batch_first {
                // onnx: Y.shape = [batch_size, seq_length, num_directions, hidden_size]
                target_wire!(y = AxisOp::Add(2), scan_outputs[slot]);
                result.push(y);
            } else {
                // onnx: Y.shape = [seq_length, num_directions, batch_size, hidden_size]
                target_wire!(y_batch_middle = AxisOp::Move(1, 0), scan_outputs[slot]);
                target_wire!(y = AxisOp::Add(1), y_batch_middle);
                result.push(y);
            }
        }
        if let Some(slot) = self.optional_y_h_output {
            if self.batch_first {
                result.push(scan_outputs[slot]);
            } else {
                target_wire!(y_h_batch_middle = AxisOp::Move(1, 0), scan_outputs[slot]);
                result.push(y_h_batch_middle);
            }
        }
        if let Some(slot) = self.optional_y_c_output {
            if self.batch_first {
                result.push(scan_outputs[slot]);
            } else {
                target_wire!(y_c_batch_middle = AxisOp::Move(1, 0), scan_outputs[slot]);
                result.push(y_c_batch_middle);
            }
        }

        Ok(result)
    }
}

impl Expansion for CommonRec {
    fn name(&self) -> Cow<str> {
        self.body.name().into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("batch_first: {:?}", self.batch_first)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(self.optional_y_output.is_some() as usize
            + self.optional_y_h_output.is_some() as usize
            + self.optional_y_c_output.is_some() as usize)
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
            + self.optional_initial_h_input.is_some() as usize
            + self.optional_initial_c_input.is_some() as usize
            + self.optional_p_input.is_some() as usize;
        check_input_arity(inputs, input_count)?;
        let output_count = self.optional_y_output.is_some() as usize
            + self.optional_y_h_output.is_some() as usize
            + self.optional_y_c_output.is_some() as usize;
        check_output_arity(outputs, output_count)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[2].rank, 3)?;

        /* If 0
         *      X.shape = [seq_length, batch_size, input_size],
         *      Y.shape = [seq_length, num_directions, batch_size, hidden_size],
         *      initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
         *  If 1,
         *      X.shape = [batch_size, seq_length, input_size],
         *      Y.shape = [batch_size, seq_length, num_directions, hidden_size],
         *      initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
         */

        let b = if self.batch_first { 0 } else { 1 };
        let b_in_y = if self.batch_first { 0 } else { 2 };
        let seq_len = if self.batch_first { 1 } else { 0 };
        let dirs = if self.batch_first { 1 } else { 0 };
        let dirs_in_y = if self.batch_first { 2 } else { 1 };

        let (w_mul, b_mul) = self.body.w_b_multipliers();

        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?; // num_directions
        s.equals(&inputs[1].shape[1], (w_mul as i64) * inputs[2].shape[2].bex())?; // hidden_size
        s.equals(&inputs[2].shape[1], (w_mul as i64) * inputs[2].shape[2].bex())?; // hidden_size
        if let Some(bias) = self.optional_bias_input {
            s.equals(&inputs[bias].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[bias].rank, 2)?;
            s.equals(&inputs[bias].shape[0], &inputs[2].shape[0])?; // num_directions
            s.equals(&inputs[bias].shape[1], (b_mul as i64) * inputs[2].shape[2].bex())?;
            // 6 * hidden_size
        }
        if let Some(seq_len) = self.optional_sequence_lens_input {
            s.equals(&inputs[seq_len].rank, 1)?;
            s.equals(&inputs[seq_len].shape[0], &inputs[0].shape[b])?; // batch_size
        }
        if let Some(initial_h) = self.optional_initial_h_input {
            s.equals(&inputs[initial_h].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[initial_h].rank, 3)?;
            s.equals(&inputs[initial_h].shape[dirs], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[initial_h].shape[b], &inputs[0].shape[b])?; // batch_size
            s.equals(&inputs[initial_h].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(initial_c) = self.optional_initial_c_input {
            s.equals(&inputs[initial_c].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[initial_c].rank, 3)?;
            s.equals(&inputs[initial_c].shape[dirs], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[initial_c].shape[b], &inputs[0].shape[b])?; // batch_size
            s.equals(&inputs[initial_c].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(p) = self.optional_p_input {
            s.equals(&inputs[p].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[p].rank, 2)?;
            s.equals(&inputs[p].shape[0], &inputs[1].shape[0])?; // num_directions
            s.equals(&inputs[p].shape[1], 3 * inputs[2].shape[2].bex())?; // hidden_size
        }
        if let Some(y) = self.optional_y_output {
            s.equals(&outputs[y].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y].rank, 4)?;
            s.equals(&outputs[y].shape[seq_len], &inputs[0].shape[seq_len])?; // seq_lenght
            s.equals(&outputs[y].shape[dirs_in_y], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y].shape[b_in_y], &inputs[0].shape[b])?; // batch_size
            s.equals(&outputs[y].shape[3], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(y_h) = self.optional_y_h_output {
            s.equals(&outputs[y_h].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y_h].rank, 3)?;
            s.equals(&outputs[y_h].shape[dirs], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y_h].shape[b], &inputs[0].shape[b])?; // batch_size
            s.equals(&outputs[y_h].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        if let Some(y_c) = self.optional_y_c_output {
            s.equals(&outputs[y_c].datum_type, &inputs[0].datum_type)?;
            s.equals(&outputs[y_c].rank, 3)?;
            s.equals(&outputs[y_c].shape[dirs], &inputs[1].shape[0])?; // num_directions
            s.equals(&outputs[y_c].shape[b], &inputs[0].shape[b])?; // batch_size
            s.equals(&outputs[y_c].shape[2], &inputs[2].shape[2])?; // hidden_size
        }
        Ok(())
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
            if let Some(ix) = self.optional_y_c_output {
                outputs[ix] = target.wire_node(
                    format!("{prefix}.merge_y_c_output"),
                    TypedConcat::new(0),
                    &[fore[ix], back[ix]],
                )?[0];
            }
            Ok(outputs)
        } else {
            Ok(fore)
        }
    }
}
