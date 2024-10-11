use tract_linalg::frame::block_quant::{BlockQuant, Q4_0};

use crate::internal::*;
use crate::ops::einsum::optimize::{ensure_mkn_axes, AxesOrPatch};
use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use crate::transform::ModelTransform;

#[derive(Clone, Hash)]
pub struct BlockQuantFact {
    pub format: Box<dyn BlockQuant>,
    pub shape: ShapeFact,
}

impl std::fmt::Debug for BlockQuantFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:?})", self.format, self.shape)
    }
}

impl OpaqueFact for BlockQuantFact {

    fn mem_size(&self) -> TDim {
        self.shape.volume() * self.format.block_bytes()
    }
}

#[derive(Clone, Hash)]
pub struct BlockQuantValue {
    pub fact: BlockQuantFact,
    pub value: Blob,
}

impl OpaquePayload for BlockQuantValue {}

impl std::fmt::Debug for BlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.fact, self.value)
    }
}

impl std::fmt::Display for BlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug)]
pub struct BlockQuantTransform;

impl ModelTransform for BlockQuantTransform {
    fn name(&self) -> Cow<str> {
        "BlockQuantTransform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::<()>::default()
            .with_rule_for("block_quant_einsum_weights", block_quant_einsum_weights)
            .rewrite(&(), model)
    }
}

fn block_quant_einsum_weights(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    let &[a, b] = &*model.node_input_facts(node.id)? else { return Ok(None) };
    if b.konst.is_some() {
        let mut new_axes = op.axes.clone().with_extra_input(2)?;
        for (ix, axis) in op.axes.axes(InOut::In(0)).enumerate() {
            new_axes = new_axes.with_extra_axis_occurency(axis.repr, InOut::In(2), ix)?;
        }
        new_axes = new_axes.remove_slot(InOut::In(0))?;
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &[node.inputs[1], node.inputs[0]],
            EinSum { axes: new_axes, ..op.clone() },
        )?));
    }
    if a.konst.is_none() || a.rank() != 2 {
        return Ok(None);
    }
    let AxesOrPatch::Annotated(op) = ensure_mkn_axes(op, model, node)? else { return Ok(None) };
    if op.a_m() == 1 && op.a_k() == 0 {
        let a: &Tensor = a.konst.as_ref().unwrap();
        let mut patch = TypedModelPatch::default();
        let konst =
            patch.add_const(&model.node(node.inputs[0].node).name, a.clone().move_axis(1, 0)?)?;
        let axes = op
            .op
            .axes
            .clone()
            .with_extra_axis_occurency(op.k_axis, InOut::In(0), 2)?
            .remove_axis_occurency(InOut::In(0), 0)?;
        let tap = patch.tap_model(model, node.inputs[1])?;
        let output = patch.wire_node(prefix, EinSum { axes, ..op.op.clone() }, &[konst, tap])?;
        patch.shunt_outside(model, node.id.into(), output[0])?;
        return Ok(Some(patch));
    }
    let format = Q4_0;
    let mut patch = TypedModelPatch::default();
    let weights = if a.datum_type == f16::datum_type() {
        format.quant_f16(a.konst.as_ref().unwrap().as_slice::<f16>()?)?
    } else {
        format.quant_f32(a.konst.as_ref().unwrap().cast_to::<f32>()?.as_slice::<f32>()?)?
    };
    let name = &model.node(node.inputs[0].node).name;
    let fact = BlockQuantFact { format: Box::new(format), shape: a.shape.clone() };
    let value = BlockQuantValue { fact: fact.clone(), value: weights };
    let weights = patch.wire_node(
        format!("{name}.bq"),
        Const::new_with_opaque_fact(rctensor0(Opaque(Arc::new(value))), Box::new(fact)),
        &[],
    )?;
    let tap = patch.tap_model(model, node.inputs[1])?;
    let wire = patch.wire_node(prefix, op.op.clone(), &[weights[0], tap])?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}
