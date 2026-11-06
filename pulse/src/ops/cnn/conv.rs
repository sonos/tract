use crate::internal::*;
use crate::ops::cnn::pools::pulsify_pooled_input;
use tract_core::num_traits::Zero;
use tract_core::ops::array::Pad;
use tract_core::ops::array::PadMode;
use tract_core::ops::cnn::Conv;

register_all!(Conv: pulsify);

/// Left edge padding on the streaming `axis` that the `Pad` producing `wire`
/// hands over to the conv it feeds.
///
/// Holds when `wire` comes out of an edge-mode `Pad` that only pads `axis`, on
/// the left, and whose single consumer is a `Conv` (as data input) with no
/// padding of its own on that axis. The `Pad` pulsifier then forwards its input
/// untouched and the `Conv` pulsifier pads its own overlapping window instead,
/// the way a streaming conv seeds its history with the first frame: unlike a
/// standalone pulsed edge `Pad`, this costs no delay and accepts any pulse.
pub(crate) fn absorbed_edge_pad(
    source: &TypedModel,
    wire: OutletId,
    axis: usize,
) -> TractResult<Option<usize>> {
    let node = source.node(wire.node);
    rule_if_some!(pad = node.op_as::<Pad>());
    rule_if!(wire.slot == 0 && pad.mode == PadMode::Edge);
    rule_if!(pad.pads.iter().enumerate().all(|(ax, &(b, a))| a == 0 && (ax == axis || b == 0)));
    let before = pad.pads[axis].0;
    rule_if!(before > 0);
    let successors = &node.outputs[0].successors;
    rule_if!(successors.len() == 1 && successors[0].slot == 0);
    let consumer = source.node(successors[0].node);
    rule_if_some!(conv = consumer.op_as::<Conv>());
    let spec = &conv.pool_spec;
    let shape = spec.data_format.shape(&*node.outputs[0].fact.shape)?;
    rule_if!(axis >= shape.h_axis() && axis < shape.h_axis() + shape.hw_rank());
    let geo_axis = axis - shape.h_axis();
    let own = spec.padding.compute_one(
        geo_axis,
        &shape.hw_dims()[geo_axis],
        spec.kernel_shape[geo_axis],
        spec.dilation(geo_axis),
        spec.stride(geo_axis),
    );
    rule_if!(own.pad_before.is_zero() && own.pad_after.is_zero());
    Ok(Some(before))
}

fn pulsify(
    op: &Conv,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
    let zero = Tensor::zero_scalar_dt(fact.datum_type)?;
    let edge_before = if let Some(stream) = &fact.stream {
        absorbed_edge_pad(source, node.inputs[0], stream.axis)?.unwrap_or(0)
    } else {
        0
    };
    if let Some((wire, pool_spec)) =
        pulsify_pooled_input(&op.pool_spec, source, node, target, mapping, Some(zero), edge_before)?
    {
        let mut wires: TVec<_> = node.inputs.iter().map(|i| mapping[i]).collect();
        wires[0] = wire;
        Ok(Some(target.wire_node(&node.name, Conv { pool_spec, ..op.clone() }, &wires)?))
    } else {
        Ok(None)
    }
}

impl PulsedOp for Conv {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let dt = self.q_params.unwrap_or(inputs[0].datum_type);
        super::pools::pulsed_output_facts(&self.pool_spec, inputs, dt)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
