use crate::tfpb::*;
use crate::tfpb::node_def::NodeDef;
use tract_core::ops::prelude::*;

pub fn block_lstm(node: &NodeDef) -> TractResult<Box<Op>> {
    let forget_bias = node.get_attr_opt_float("forget_bias")?.unwrap_or(1.0);
    let cell_clip = node.get_attr_opt_float("cell_clip")?.unwrap_or(3.0);
    let t = node.get_attr_datum_type("T")?;
    let use_peephole = node.get_attr_opt_bool("use_peephole")?.unwrap_or(false);
    Ok(Box::new(BlockLSTM::new(forget_bias, cell_clip, t, use_peephole)))
}

#[derive(Clone, Debug, new)]
struct BlockLSTM {
    forget_bias: f32,
    cell_clip: f32,
    t: DatumType,
    use_peephole: bool,
}

impl Op for BlockLSTM {
    fn name(&self) -> Cow<str> {
        "tf.BlockLSTM".into()
    }
}

impl StatelessOp for BlockLSTM {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        unimplemented!()
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

        // seq_len_max
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&inputs[0].datum_type, i64::datum_type())?;

        s.equals_all((1..=7).map(move |i| (&inputs[i].datum_type).bex()).collect())?;
        s.equals_all((0..=6).map(move |i| (&outputs[i].datum_type).bex()).collect())?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;

        s.equals(&inputs[1].rank, 3)?;
        s.equals(&inputs[2].rank, 2)?;
        s.equals(&inputs[3].rank, 2)?;
        s.equals(&inputs[4].rank, 2)?;
        s.equals(&inputs[5].rank, 1)?;
        s.equals(&inputs[6].rank, 1)?;
        s.equals(&inputs[7].rank, 1)?;
        s.equals(&inputs[8].rank, 1)?;
        Ok(())
    }
}
