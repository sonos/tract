use std::any::TypeId;

use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::tract_core::ops::binary::TypedBinOp;
use tract_hir::tract_core::ops::element_wise::ElementWiseOp;

use crate::ser::SubgraphBuilder;
use crate::tflite::{BuiltinOperator, Model, Operator, SubGraph};

pub type ToTract = fn(op_ctx: &mut DeserOp) -> TractResult<TVec<OutletId>>;
pub type ToTflite = fn(&mut SubgraphBuilder, &TypedModel, &TypedNode) -> TractResult<()>;

#[derive(Default)]
pub struct Registry {
    //    pub primitives: HashMap<Identifier, PrimitiveDecl>,
    //    pub unit_element_wise_ops: Vec<(Identifier, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(BuiltinOperator, Box<dyn ElementWiseMiniOp>)>,
    pub binary_ops: Vec<(BuiltinOperator, TypedBinOp)>,
    pub to_tract: HashMap<BuiltinOperator, ToTract>,
    pub to_tflite: HashMap<TypeId, ToTflite>,
}

pub struct DeserContext<'ctx> {
    pub model: &'ctx Model<'ctx>,
    pub subgraph: &'ctx SubGraph<'ctx>,
    pub target: &'ctx mut TypedModel,
}

pub struct DeserOp<'op> {
    pub ctx: DeserContext<'op>,
    pub prefix: &'op str,
    pub flat: &'op Operator<'op>,
    pub inputs: &'op [OutletId],
}

impl<'op> DeserOp<'op> {
    pub fn facts(&self) -> TractResult<TVec<TypedFact>> {
        self.inputs
            .iter()
            .map(|o| self.ctx.target.outlet_fact(*o).cloned())
            .collect::<TractResult<TVec<_>>>()
    }
}

impl Registry {
    pub fn reg_to_tflite<T: 'static>(&mut self, tflite: ToTflite) {
        self.to_tflite.insert(std::any::TypeId::of::<T>(), tflite);
    }

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
        let prefix = tensors.get(flat.outputs().unwrap().get(0) as usize).name().unwrap();
        let opcode_index = flat.opcode_index();
        let opcode = model.operator_codes().unwrap().get(opcode_index as _).builtin_code();
        let ctx = DeserContext { model, subgraph, target };
        if let Some(ew) =
            self.element_wise_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            inputs = target.wire_node(prefix, ElementWiseOp(ew.clone()), &inputs)?;
        } else if let Some(bin) =
            self.binary_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            inputs = wire_with_rank_broadcast(prefix, target, bin, &inputs)?;
        } else if let Some(op) = self.to_tract.get(&opcode) {
            inputs = (op)(&mut DeserOp { ctx, prefix, flat, inputs: &inputs })?
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
