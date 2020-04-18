use tract_hir::internal::*;
use tract_ndarray::prelude::*;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn block_lstm(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let forget_bias = node.get_attr_opt_float("forget_bias")?.unwrap_or(1.0);
    let cell_clip = node.get_attr_opt_float("cell_clip")?.unwrap_or(3.0);
    let t = node.get_attr_datum_type("T")?;
    let use_peephole = node.get_attr_opt_bool("use_peephole")?.unwrap_or(false);
    Ok(Box::new(BlockLSTM::new(forget_bias, cell_clip, t, use_peephole)))
}

#[derive(Clone, Debug, new, Educe)]
#[educe(Hash)]
pub struct BlockLSTM {
    #[educe(Hash(method="hash_f32"))]
    forget_bias: f32,
    #[educe(Hash(method="hash_f32"))]
    cell_clip: f32,
    t: DatumType,
    use_peephole: bool,
}

tract_linalg::impl_dyn_hash!(BlockLSTM);

impl Op for BlockLSTM {
    fn name(&self) -> Cow<str> {
        "tf.BlockLSTM".into()
    }

    op_as_typed_op!();
}

impl StatelessOp for BlockLSTM {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        eprintln!("entering eval");
        eprintln!("{:?}", inputs);
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

        eprintln!("before loop");
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

        eprintln!("after loop");
        if x.shape()[0] > len as usize {
            i.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            cs.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            f.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            o.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            ci.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            co.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
            h.slice_axis_mut(Axis(0), (len..).into()).fill(0.0);
        }
        eprintln!("after assigns");
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

impl TypedOp for BlockLSTM {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(std::iter::repeat(inputs[1].clone()).take(7).collect())
    }
}
