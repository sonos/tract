use tract_hir::internal::*;
use tract_ndarray::prelude::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn block_lstm(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let forget_bias = node.get_attr_opt_float("forget_bias")?.unwrap_or(1.0);
    let cell_clip = node.get_attr_opt_float("cell_clip")?.unwrap_or(3.0);
    let t = node.get_attr_datum_type("T")?;
    let use_peephole = node.get_attr_opt_bool("use_peephole")?.unwrap_or(false);
    if use_peephole {
        unimplemented!("Block LSTM peeplholes");
    }
    Ok(Box::new(BlockLSTM::new(forget_bias, cell_clip, t, use_peephole)))
}

#[derive(Clone, Debug, new, Educe, Serialize, Deserialize)]
#[educe(Hash)]
pub struct BlockLSTM {
    #[educe(Hash(method = "hash_f32"))]
    forget_bias: f32,
    #[educe(Hash(method = "hash_f32"))]
    cell_clip: f32,
    t: DatumType,
    use_peephole: bool,
}

tract_linalg::impl_dyn_hash!(BlockLSTM);

impl Op for BlockLSTM {
    fn name(&self) -> Cow<str> {
        "BlockLSTM".into()
    }

    op_tf!();
    op_as_typed_op!();
}

impl StatelessOp for BlockLSTM {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let len = *inputs[0].cast_to::<i32>()?.to_scalar::<i32>()? as usize;

        let x = inputs[1].to_array_view::<f32>()?.into_dimensionality::<Ix3>()?;
        let cell_size = x.shape()[2];
        let cs_prev = inputs[2].to_array_view::<f32>()?;
        let h_prev = inputs[3].to_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
        let w = inputs[4].to_array_view::<f32>()?.into_dimensionality::<Ix2>()?;
        let bias = inputs[8].to_array_view::<f32>()?;

        let outputs_shape = x.shape();
        let mut i = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut cs = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut f = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut o = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut ci = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut co = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut h = unsafe { ArrayD::<f32>::uninitialized(&*outputs_shape) };
        let mut h_prev = h_prev.to_owned();
        let mut cs_prev = cs_prev.to_owned();

        let sigmoid = (tract_linalg::ops().sigmoid_f32)();
        let sigmoid_f32 = |f: f32| -> f32 {
            let mut f = [f];
            sigmoid.run(&mut f);
            f[0]
        };

        let tanh = (tract_linalg::ops().tanh_f32)();
        let tanh_f32 = |f: f32| -> f32 {
            let mut f = [f];
            tanh.run(&mut f);
            f[0]
        };

        for n in 0..len {
            let x = x.index_axis(Axis(0), n);
            let mut i = i.index_axis_mut(Axis(0), n);
            let mut cs = cs.index_axis_mut(Axis(0), n);
            let mut f = f.index_axis_mut(Axis(0), n);
            let mut o = o.index_axis_mut(Axis(0), n);
            let mut ci = ci.index_axis_mut(Axis(0), n);
            let mut co = co.index_axis_mut(Axis(0), n);
            let mut h = h.index_axis_mut(Axis(0), n);

            let xh = tract_ndarray::stack(Axis(1), &[x, h_prev.view()])?;

            let i_ci_f_o = xh.dot(&w) + &bias;

            i.assign(&i_ci_f_o.slice_axis(Axis(1), (0..cell_size).into()));
            i.mapv_inplace(sigmoid_f32); // TODO: peepholes
                                         // dbg!(&i);

            f.assign(&i_ci_f_o.slice_axis(Axis(1), (2 * cell_size..3 * cell_size).into()));
            f.mapv_inplace(|x| sigmoid_f32(x + self.forget_bias)); // TODO: peepholes

            ci.assign(&i_ci_f_o.slice_axis(Axis(1), (cell_size..2 * cell_size).into()));
            ci.mapv_inplace(tanh_f32);

            cs_prev *= &f;
            cs_prev += &(ci.to_owned() * &i);
            // TODO: clip cs
            cs.assign(&cs_prev);

            o.assign(&i_ci_f_o.slice_axis(Axis(1), (3 * cell_size..4 * cell_size).into()));
            o.mapv_inplace(sigmoid_f32); // TODO: peephole

            co.assign(&cs);
            co.mapv_inplace(tanh_f32);

            h_prev.assign(&co);
            h_prev *= &o;
            h.assign(&h_prev);
        }

        if x.shape()[0] > len as usize {
            i.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            cs.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            f.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            o.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            ci.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            co.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            h.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
        }
        Ok(tvec!(
            i.into_arc_tensor(),
            cs.into_arc_tensor(),
            f.into_arc_tensor(),
            o.into_arc_tensor(),
            ci.into_arc_tensor(),
            co.into_arc_tensor(),
            h.into_arc_tensor()
        ))
    }
}

impl InferenceRulesOp for BlockLSTM {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 9)?;
        check_input_arity(&outputs, 7)?;

        s.equals(&inputs[0].rank, 0)?; // seq_len_max
        s.equals(&inputs[0].datum_type, i64::datum_type())?;

        // other inputs and outputs are consistent float-like
        s.equals_all((1..=7).map(move |i| (&inputs[i].datum_type).bex()).collect())?;

        s.equals(&inputs[1].rank, 3)?; // x:  [ time, batch, cell_size ]
        s.equals(&inputs[2].rank, 2)?; // cs_prev: [batch, cell_size]
        s.equals(&inputs[3].rank, 2)?; // h_prev: [batch, cell_size]
        s.equals(&inputs[4].rank, 2)?; // w: []
        s.equals(&inputs[5].rank, 1)?; // peephole input
        s.equals(&inputs[6].rank, 1)?; // peephole forget
        s.equals(&inputs[7].rank, 1)?; // peephole output
        s.equals(&inputs[8].rank, 1)?; // bias: [ 4*cell_size ]
        s.equals(&inputs[8].shape[0], 4 * inputs[1].shape[2].bex())?; // bias: [ 4*cell_size ]

        // i, cs, f, o, ci, co, h
        for i in 0..7 {
            s.equals(&inputs[1].datum_type, &outputs[i].datum_type)?;
            s.equals(&outputs[i].shape, &inputs[1].shape)?;
        }

        Ok(())
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(7)
    }

    as_op!();
    to_typed!();
}

macro_rules! some_or_else {
    ($exp: expr) => {
        if let Some(it) = $exp {
            it
        } else {
            return Ok(None);
        }
    };
}

#[typetag::serde]
impl TypedOp for BlockLSTM {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(std::iter::repeat(inputs[1].clone()).take(7).collect())
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use tract_hir::tract_core::ops::{array, math, matmul, nn, scan};

        let mut body = TypedModel::default();
        let mut patch = TypedModelPatch::default();
        let mut outer_inputs = vec![];
        let mut input_mapping = vec![];
        let mut output_mapping = vec![];

        /*
        dbg!(&model.node_input_facts(node.id));
        let cs_initializer = some_or_else!(self.inline_var_assign(model, node, 2, 1, &mut patch)?);
        dbg!(&cs_initializer);
        let h_initializer = some_or_else!(self.inline_var_assign(model, node, 3, 6, &mut patch)?);
        dbg!(&h_initializer);
        */
        let w = some_or_else!(model.outlet_fact(node.inputs[4])?.konst.clone());
        let b = some_or_else!(model.outlet_fact(node.inputs[8])?.konst.clone());
        let cell_size = w.shape()[1] / 4;
        let mut b = b.into_tensor();
        b.insert_axis(0)?;

        macro_rules! wire {
            ($name: ident = $op: expr, $($param: expr),*) => {
                let $name = body.wire_node(
                    format!("{}-{}", node.name, stringify!($name)),
                    $op, [$($param),*].as_ref())?[0];
            }
        };

        let seq_len = patch.tap_model(model, node.inputs[0])?;
        outer_inputs.push(seq_len);

        // X: body input 0: X, new outside input 0 (was 1)
        let outside_x = patch.tap_model(model, node.inputs[1])?;
        outer_inputs.push(outside_x);
        input_mapping.push(scan::InputMapping::Scan { slot: 1, axis: 0, chunk: 1.to_dim() });
        let mut x_source_fact = model.outlet_fact(node.inputs[1])?.clone();
        x_source_fact.shape.set_dim(0, 1.to_dim())?;
        let x_source = body.add_source("x_source", x_source_fact)?.into();
        wire!(x = AxisOp::Rm(0), x_source);

        // CS: body input 1
        let cs = patch.tap_model(model, node.inputs[2])?;
        let cs = patch.wire_node(format!("{}-cs-axis", node.name), AxisOp::Add(0), &[cs])?[0];
        outer_inputs.push(cs);
        let cs_fact = patch.outlet_fact(cs)?.clone();
        let cs_source = body.add_source("cs_source", cs_fact)?;
        input_mapping
            .push(scan::InputMapping::State { initializer: scan::StateInitializer::FromInput(2) });
        wire!(cs_prev = AxisOp::Rm(0), cs_source);

        // H: body input 2
        let h = patch.tap_model(model, node.inputs[3])?;
        let h = patch.wire_node(format!("{}-h-axis", node.name), AxisOp::Add(0), &[h])?[0];
        outer_inputs.push(h);
        let h_fact = patch.outlet_fact(h)?.clone();
        let h_source = body.add_source("h_source", h_fact)?;
        input_mapping
            .push(scan::InputMapping::State { initializer: scan::StateInitializer::FromInput(3) });
        wire!(h_prev = AxisOp::Rm(0), h_source);

        wire!(xh = array::TypedConcat::concat_vars(1, 2), x, h_prev);

        wire!(i_ci_f_o_1 = matmul::mir::MatMulUnary::new(w, true, true, true, None), xh);
        wire!(i_ci_f_o = math::add::unary(b.into_arc_tensor()), i_ci_f_o_1);

        wire!(i_1 = array::Slice::new(1, 0, cell_size), i_ci_f_o);
        wire!(i = nn::sigmoid(), i_1);

        wire!(f_1 = array::Slice::new(1, 2 * cell_size, 3 * cell_size), i_ci_f_o);
        wire!(f_2 = math::add::unary(rctensor2(&[[self.forget_bias]])), f_1);
        wire!(f = nn::sigmoid(), f_2);

        wire!(ci_1 = array::Slice::new(1, cell_size, 2 * cell_size), i_ci_f_o);
        wire!(ci = math::tanh(), ci_1);

        wire!(o_1 = array::Slice::new(1, 3 * cell_size, 4 * cell_size), i_ci_f_o);
        wire!(o = nn::sigmoid(), o_1);

        wire!(ci_i = math::mul::bin_typed(), ci, i);
        wire!(cs_1 = math::mul::bin_typed(), cs_prev, f);
        wire!(cs = math::add::bin_typed(), cs_1, ci_i);

        wire!(co = math::tanh(), cs);
        wire!(h = math::mul::bin_typed(), co, o);

        wire!(i_ = AxisOp::Add(0), i);
        wire!(cs_ = AxisOp::Add(0), cs);
        wire!(f_ = AxisOp::Add(0), f);
        wire!(o_ = AxisOp::Add(0), o);
        wire!(ci_ = AxisOp::Add(0), ci);
        wire!(co_ = AxisOp::Add(0), co);
        wire!(h_ = AxisOp::Add(0), h);
        body.set_output_outlets(&[i_, cs_, f_, o_, ci_, co_, h_])?;
        for ix in 0..7 {
            output_mapping.push(scan::OutputMapping::<TDim, TDim> {
                state: ix == 1 || ix == 6,
                axis: 0,
                chunk: 1.into(),
                full_dim_hint: None,
                last_value_slot: None,
                full_slot: Some(ix),
            })
        }

        let scan = scan::Scan::new(body, input_mapping, output_mapping, Some(0))?;
        let scan = patch.wire_node(&*node.name, scan, &*outer_inputs)?;

        for (ix, o) in scan.iter().enumerate() {
            patch.shunt_outside(model, (node.id, ix).into(), *o)?
        }

        Ok(Some(patch))
    }
}

/* 
// TODO: rewrite this logic as a tf.Assign declutter ?
impl BlockLSTM {
    fn inline_var_assign(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        input_id: usize,
        output_id: usize,
        patch: &mut TypedModelPatch,
    ) -> TractResult<Option<Arc<Tensor>>> {
        let var_2 = model.node(node.inputs[input_id].node);
        let var_2_op = if let Some(op) = var_2.op_as::<crate::ops::vars::VariableV2>() {
            op
        } else {
            return Ok(None);
        };
        if var_2.outputs[0].successors.len() != 2 {
            return Ok(None);
        }
        let assign = if let Some(assign_node) = var_2.outputs[0]
            .successors
            .iter()
            .map(|s| model.node(s.node))
            .filter(|s| s.op_is::<crate::ops::vars::Assign>())
            .next()
        {
            assign_node
        } else {
            return Ok(None);
        };
        let rm_axis_node = model.node(assign.inputs[1].node);
        let rm_axis_op = if let Some(op) = rm_axis_node.op_as::<tract_hir::internal::AxisOp>() {
            op
        } else {
            return Ok(None);
        };
        if rm_axis_op != &tract_hir::internal::AxisOp::Rm(0) {
            return Ok(None);
        }
        let slice_node = model.node(rm_axis_node.inputs[0].node);
        let slice_op = if let Some(op) = slice_node.op_as::<tract_hir::ops::array::Slice<usize>>() {
            op
        } else {
            return Ok(None);
        };
        if slice_node.inputs[0] != (node.id, output_id).into() {
            return Ok(None);
        }
        let lstm_output_fact = model.outlet_fact(slice_node.inputs[0])?;
        if slice_op.axis != 0
            || slice_op.end != slice_op.start + 1
            || slice_op.end.to_dim() != lstm_output_fact.shape.dim(0)
        {
            return Ok(None);
        }
        let tap = patch.tap_model(model, rm_axis_node.id.into())?;
        patch.shunt_outside(model, assign.id.into(), tap)?;
        Ok(var_2_op.initializer.clone())
    }
}
*/
