use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::tract_core::ops::binary::TypedBinOp;
use tract_hir::tract_core::ops::element_wise::ElementWiseOp;

use crate::tflite::{BuiltinOperator, Model, Operator, SubGraph};

#[derive(Default)]
pub struct Registry {
    //    pub primitives: HashMap<Identifier, PrimitiveDecl>,
    //    pub unit_element_wise_ops: Vec<(Identifier, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(BuiltinOperator, Box<dyn ElementWiseMiniOp>)>,
    pub binary_ops: Vec<(BuiltinOperator, TypedBinOp)>,
    pub to_tract: HashMap<BuiltinOperator, ToTract>,
    //    pub from_tract: HashMap<TypeId, FromTract>,
}

type ToTract = fn(
    model: &Model,
    subgraph: &SubGraph,
    prefix: &str,
    flat: &Operator,
    target: &mut TypedModel,
    inputs: &[OutletId],
) -> TractResult<TVec<OutletId>>;

impl Registry {
    pub fn op(
        &self,
        model: &Model,
        subgraph: &SubGraph,
        flat: &Operator,
        target: &mut TypedModel,
        mapping: &mut HashMap<i32, OutletId>,
    ) -> TractResult<()> {
        let mut inputs = tvec!();
        for input in flat.inputs().unwrap() {
            inputs.push(mapping[&input]);
        }
        let tensors = subgraph.tensors().unwrap();
        let name = tensors.get(flat.outputs().unwrap().get(0) as usize).name().unwrap();
        let opcode_index = flat.opcode_index();
        let opcode = model.operator_codes().unwrap().get(opcode_index as _).builtin_code();
        if let Some(ew) =
            self.element_wise_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            inputs = target.wire_node(name, ElementWiseOp(ew.clone()), &inputs)?;
        } else if let Some(bin) =
            self.binary_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            inputs = wire_with_rank_broadcast(name, target, bin, &inputs)?;
        } else if let Some(op) = self.to_tract.get(&opcode) {
            inputs = (op)(model, subgraph, name, flat, target, &inputs)?
        } else {
            let facts =
                inputs.iter().map(|o| target.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            bail!("Unsupported operator {opcode:?}, inputs: {facts:#?}")
        }
        for (flat, wire) in flat.outputs().unwrap().iter().zip(inputs.iter()) {
            mapping.insert(flat, *wire);
        }
        Ok(())
    }
}
