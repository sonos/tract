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

mod helpers {
    use super::*;
    use tract_core::ops::{array, math};

    pub fn transpose() -> Box<dyn InferenceOp> {
        Box::new(array::PermuteAxes::new(Some(vec![1, 0])))
    }

    pub fn slice(axis: usize, range: std::ops::Range<usize>) -> Box<dyn InferenceOp> {
        Box::new(array::Slice::new(vec![axis], vec![range.start], vec![range.end]))
    }

    pub fn dot() -> Box<dyn InferenceOp> {
        Box::new(math::MatMul::new())
    }

    pub fn add() -> Box<dyn InferenceOp> {
        Box::new(math::Add::default())
    }

    pub fn sub() -> Box<dyn InferenceOp> {
        Box::new(math::Sub::default())
    }

    pub fn mul() -> Box<dyn InferenceOp> {
        Box::new(math::Mul::default())
    }
}

#[derive(Debug, Clone, new)]
pub struct GRU {
    pub optional_bias_input: Option<usize>,
    pub optional_sequence_lens_input: Option<usize>,
    pub optional_initial_h_input: Option<usize>,
    pub optional_y_output: Option<usize>,
    pub optional_y_h_output: Option<usize>,
    pub f: Box<dyn InferenceOp>,
    pub g: Box<dyn InferenceOp>,
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
            g: Box::new(core_ops::nn::Tanh::new(f32::datum_type().into())),
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

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        use helpers::*;
        use tract_core::ops::scan;

        let x_fact = model.outlet_fact(node.inputs[0])?;
        let b_size = x_fact
            .shape
            .dim(1)
            .unwrap()
            .concretize()
            .ok_or("Incomplete analysis, can not incorporate.")?
            .to_integer()
            .unwrap() as usize;
        let r_fact = model.outlet_fact(node.inputs[2])?;
        let h_size = r_fact
            .shape
            .dim(2)
            .unwrap()
            .concretize()
            .ok_or("Incomplete analysis, can not incorporate.")?
            .to_integer()
            .unwrap() as usize;

        let mut body = InferenceModel::default();
        let mut input_mapping = vec![];
        let mut output_mapping = vec![];

        let _ = body.add_source("x", TensorFact::shape(ShapeFact::open(tvec!(1.to_dim().into()))))?;
        input_mapping.push(scan::InputMapping::Scan { slot: 0, axis: 0, chunk: () });
        let x = body.chain_default(
            format!("{}-x-rm-chunk-dim", &*node.name),
            tract_core::ops::array::RmDims::new(vec![0]),
        )?;

        let w = body.add_source_default("w")?;
        input_mapping.push(scan::InputMapping::Full { slot: 1 });

        let r = body.add_source_default("r")?;
        input_mapping.push(scan::InputMapping::Full { slot: 2 });

        let _ = body.add_source_default("ht")?;
        let ht_prev = body.chain_default(
            format!("{}-ht-rm-chunk-dim", &*node.name),
            tract_core::ops::array::RmDims::new(vec![0]),
        )?;
        let initializer = if let Some(initial_h_input) = self.optional_initial_h_input {
            scan::StateInitializer::FromInput(initial_h_input)
        } else {
            scan::StateInitializer::Value(
                ndarray::Array3::<f32>::zeros((1, b_size, h_size)).into_arc_tensor(),
            )
        };
        input_mapping.push(scan::InputMapping::State { initializer });

        let b = if let Some(slot) = self.optional_bias_input {
            let b = body.add_source_default("b")?;
            input_mapping.push(scan::InputMapping::Full { slot });
            Some(b)
        } else {
            None
        };

        let rz = body.add_node_simple_default("rz", slice(0, 0..h_size), tvec!(r))?;
        let rr = body.add_node_simple_default("rr", slice(0, h_size..2 * h_size), tvec!(r))?;
        let rh = body.add_node_simple_default("rh", slice(0, 2 * h_size..3 * h_size), tvec!(r))?;

        let rz_t = body.add_node_simple_default("rz_t", transpose(), tvec!(rz))?;
        let rr_t = body.add_node_simple_default("rr_t", transpose(), tvec!(rr))?;
        let rh_t = body.add_node_simple_default("rh_t", transpose(), tvec!(rh))?;

        let ht_rz_t = body.add_node_simple_default("ht_rz_t", dot(), tvec!(ht_prev, rz_t))?;
        let ht_rr_t = body.add_node_simple_default("ht_rr_t", dot(), tvec!(ht_prev, rr_t))?;

        let wz = body.add_node_simple_default("wz", slice(0, 0..h_size), tvec!(w))?;
        let wr = body.add_node_simple_default("wr", slice(0, h_size..2 * h_size), tvec!(w))?;
        let wh = body.add_node_simple_default("wh", slice(0, 2 * h_size..3 * h_size), tvec!(w))?;

        let wz_t = body.add_node_simple_default("wz_t", transpose(), tvec!(wz))?;
        let wr_t = body.add_node_simple_default("wr_t", transpose(), tvec!(wr))?;
        let wh_t = body.add_node_simple_default("wh_t", transpose(), tvec!(wh))?;

        let xt_wz_t = body.add_node_simple_default("xt_wz_t", dot(), tvec!(x, wz_t))?;
        let xt_wr_t = body.add_node_simple_default("xt_wr_t", dot(), tvec!(x, wr_t))?;
        let xt_wh_t = body.add_node_simple_default("xt_wh_t", dot(), tvec!(x, wh_t))?;

        let mut zt0 = body.add_node_simple_default("zt0", add(), tvec!(xt_wz_t, ht_rz_t))?;
        if let Some(b) = b {
            let wbz = body.add_node_simple_default("wbz", slice(0, 0..h_size), tvec!(b))?;
            let rbz = body.add_node_simple_default("rbz", slice(0, 3*h_size..4*h_size), tvec!(b))?;
            let wbz_rbz = body.add_node_simple_default("wbz_rbz", add(), tvec!(wbz, rbz))?;
            zt0 = body.add_node_simple_default("zt0_biased", add(), tvec!(zt0, wbz_rbz))?;
        };
        let zt = body.add_node_simple_default("zt", self.f.clone(), tvec!(zt0))?;
        let mut rt0 = body.add_node_simple_default("rt0", add(), tvec!(xt_wr_t, ht_rr_t))?;
        if let Some(b) = b {
            let wbr = body.add_node_simple_default("wbr", slice(0, h_size..2*h_size), tvec!(b))?;
            let rbr = body.add_node_simple_default("rbr", slice(0, 4*h_size..5*h_size), tvec!(b))?;
            let wbr_rbr = body.add_node_simple_default("wbr_rbr", add(), tvec!(wbr, rbr))?;
            rt0 = body.add_node_simple_default("rt0_biased", add(), tvec!(rt0, wbr_rbr))?;
        };
        let rt = body.add_node_simple_default("rt", self.f.clone(), tvec!(rt0))?;

        let rt_ht_rht = if self.linear_before_reset {
            let ht_rht = body.add_node_simple_default("ht_rht", dot(), tvec!(ht_prev, rh_t))?;
            body.add_node_simple_default("rt_ht_rht", mul(), tvec!(rt, ht_rht))?
        } else {
            let rt_ht = body.add_node_simple_default("rt_ht", mul(), tvec!(rt, ht_prev))?;
            body.add_node_simple_default("rt_ht_rht", dot(), tvec!(rt_ht, rh_t))?
        };
        let mut ht0 = body.add_node_simple_default("ht0", add(), tvec!(xt_wh_t, rt_ht_rht))?;
        if let Some(b) = b {
            let wbh = body.add_node_simple_default("wbh", slice(0, 2*h_size..3*h_size), tvec!(b))?;
            let rbh = body.add_node_simple_default("rbh", slice(0, 5*h_size..6*h_size), tvec!(b))?;
            let wbh_rbh = body.add_node_simple_default("wbh_rbh", add(), tvec!(wbh, rbh))?;
            ht0 = body.add_node_simple_default("ht0_biased", add(), tvec!(ht0, wbh_rbh))?;
        };
        let ht = body.add_node_simple_default("ht", self.g.clone(), tvec!(ht0))?;

        let one = body.add_const("one", tensor0(1f32))?;
        let one_sub_zt = body.add_node_simple_default("one_sub_zt", sub(), tvec!(one, zt))?;

        let next_ht_left =
            body.add_node_simple_default("next_ht_left", mul(), tvec!(one_sub_zt, ht))?;
        let next_ht_right =
            body.add_node_simple_default("next_ht_right", mul(), tvec!(zt, ht_prev))?;
        let next_ht =
            body.add_node_simple_default("next_ht", add(), tvec!(next_ht_left, next_ht_right))?;

        let y_h = body.add_node_simple_default(
            "y_h",
            tract_core::ops::array::AddDims::new(vec![0]),
            tvec!(next_ht),
        )?;

        let y_h_dup = body.add_node_simple_default(
            "y_h_dup",
            tract_core::ops::identity::Identity,
            tvec!(y_h),
        )?;

        let mut scan_facts = tvec!();
        let mut output_outlets = tvec!();
        if let Some(_) = self.optional_y_output {
            output_mapping.push(scan::OutputMapping::Scan {
                slot: output_mapping.len(),
                axis: 0,
                chunk: (),
                full_dim_hint: None,
            });
            output_outlets.push(OutletId::new(y_h, scan_facts.len()));
            scan_facts.push(TensorFact::default());
        }

        if let Some(_) = self.optional_y_h_output {
            output_mapping.push(scan::OutputMapping::State { slot: Some(output_mapping.len()) });
            output_outlets.push(OutletId::new(y_h_dup, scan_facts.len()));
            scan_facts.push(TensorFact::default());
        }
        body.set_output_outlets(&*output_outlets)?;

        let mut patch = InferenceModelPatch::default();
        let scan = patch.add_node(
            &*node.name,
            scan::Inference::new(body, input_mapping, output_mapping),
            scan_facts,
        )?;

        let x = patch.tap_model(model, node.inputs[0])?;
        patch.add_edge(x, InletId::new(scan, 0))?;

        let _ = patch.tap_model(model, node.inputs[1])?;
        let w = patch.chain_default(
            format!("{}-w-rm-dir-dim", &*node.name),
            tract_core::ops::array::RmDims::new(vec![0]),
        )?;
        patch.add_edge(OutletId::new(w, 0), InletId::new(scan, 1))?;

        let _ = patch.tap_model(model, node.inputs[2])?;
        let r = patch.chain_default(
            format!("{}-r-rm-dir-dim", &*node.name),
            tract_core::ops::array::RmDims::new(vec![0]),
        )?;
        patch.add_edge(OutletId::new(r, 0), InletId::new(scan, 2))?;

        if let Some(slot) = self.optional_bias_input {
            let _ = patch.tap_model(model, node.inputs[slot])?;
            let b = patch.chain_default(
                format!("{}-b-rm-dir-dim", &*node.name),
                tract_core::ops::array::RmDims::new(vec![0]),
            )?;
            patch.add_edge(OutletId::new(b, 0), InletId::new(scan, slot))?;
        }

        if let Some(slot) = self.optional_y_h_output {
            let rm_dim = patch.add_node_default(
                format!("{}-y_h-rm-seq-dim", &*node.name),
                tract_core::ops::array::RmDims::new(vec![0])
            )?;
            patch.add_edge(OutletId::new(scan, slot), InletId::new(rm_dim, 0))?;
            let add_dim = patch.chain(
                format!("{}-y_h-add-dir-dim", &*node.name),
                tract_core::ops::array::AddDims::new(vec![0]),
                tvec!(node.outputs[slot].fact.clone())
            )?;
            patch.shunt_outside(OutletId::new(node.id, slot), OutletId::new(add_dim, 0))?;
        }

        Ok(Some(patch))
    }
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
