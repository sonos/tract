use tract_core::internal::*;
use tract_core::ops::gru_seq::GruSeq;
use tract_nnef::internal::*;

fn round_trip(model: &TypedModel) -> TractResult<TypedModel> {
    let nnef = tract_nnef::nnef();
    let mut buffer = vec![];
    nnef.write_to_tar(model, &mut buffer)?;
    nnef.model_for_read(&mut &*buffer)
}

fn model_with(op: GruSeq) -> TractResult<TypedModel> {
    let (batch, t, input, h) = (1usize, 5usize, 3usize, op.hidden);
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([batch, t, input]))?;
    let w = model.add_const("w", Tensor::zero::<f32>(&[3 * h, input])?)?;
    let r = model.add_const("r", Tensor::zero::<f32>(&[3 * h, h])?)?;
    let mut inputs = tvec!(x, w, r);
    if op.has_bias {
        inputs.push(model.add_const("b", Tensor::zero::<f32>(&[1, 6 * h])?)?);
    }
    inputs.push(model.add_source("h0", f32::fact([batch, 1usize, h]))?);
    let outs = model.wire_node("gru", op, &inputs)?;
    model.select_output_outlets(&outs)?;
    Ok(model)
}

/// The op's parameters -- including the state contract -- have to survive a dump
/// and reload, or a fused model silently changes behaviour on the way through NNEF.
#[test]
fn gru_seq_round_trips_its_attributes() -> TractResult<()> {
    for has_bias in [false, true] {
        for reset_every_turn in [false, true] {
            for chunk in [1isize, -1] {
                let op = GruSeq { hidden: 4, has_bias, chunk, reset_every_turn };
                let reloaded = round_trip(&model_with(op.clone())?)?;
                let got = reloaded
                    .nodes()
                    .iter()
                    .find_map(|n| n.op_as::<GruSeq>())
                    .with_context(|| format!("GruSeq missing after round trip: {op:?}"))?;
                assert_eq!(got, &op, "attributes changed across the NNEF round trip");
            }
        }
    }
    Ok(())
}
