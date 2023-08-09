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
    pub element_wise_ops: Vec<(i32, Box<dyn ElementWiseMiniOp>)>,
    pub binary_ops: Vec<(i32, TypedBinOp)>,
    pub to_tract: HashMap<i32, ToTract>,
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
    pub output_facts: &'op [TypedFact],
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

    pub fn reg_to_tract(&mut self, op: BuiltinOperator, to: ToTract) {
        self.to_tract.insert(op.0, to);
    }

    pub fn reg_element_wise(&mut self, tflite: BuiltinOperator, tract: Box<dyn ElementWiseMiniOp>) {
        self.element_wise_ops.push((tflite.0, tract));
    }

    pub fn reg_binary(&mut self, tflite: BuiltinOperator, tract: TypedBinOp) {
        self.binary_ops.push((tflite.0, tract));
    }

    pub fn op(
        &self,
        model: &Model,
        subgraph: &SubGraph,
        flat: &Operator,
        target: &mut TypedModel,
        mapping: &mut HashMap<i32, OutletId>,
    ) -> TractResult<()> {
        let inputs: TVec<OutletId> = flat.inputs().unwrap().iter().map(|o| mapping[&o]).collect();
        let tensors = subgraph.tensors().unwrap();
        let prefix = tensors.get(flat.outputs().unwrap().get(0) as usize).name().unwrap();
        let opcode_index = flat.opcode_index();
        let operator_code = model.operator_codes().unwrap().get(opcode_index as _);
        let opcode = if operator_code.deprecated_builtin_code() as i32
            == BuiltinOperator::PLACEHOLDER_FOR_GREATER_OP_CODES.0
        {
            operator_code.builtin_code().0
        } else {
            operator_code.deprecated_builtin_code() as i32
        };
        let ctx = DeserContext { model, subgraph, target };
        let results = if let Some(ew) =
            self.element_wise_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            target.wire_node(prefix, ElementWiseOp(ew.clone()), &inputs)?
        } else if let Some(bin) =
            self.binary_ops.iter().find(|bin| bin.0 == opcode).map(|pair| pair.1.clone())
        {
            wire_with_rank_broadcast(prefix, target, bin, &inputs)?
        } else if let Some(op) = self.to_tract.get(&opcode) {
            let output_facts = flat
                .outputs()
                .unwrap()
                .iter()
                .map(|t| Ok(crate::tensors::flat_tensor_to_tract_fact(model, subgraph, t)?.0))
                .collect::<TractResult<TVec<TypedFact>>>()?;
            (op)(&mut DeserOp { ctx, prefix, flat, inputs: &inputs, output_facts: &output_facts })
                .with_context(|| format!("Opcode is {operator_code:#?}"))?
        } else {
            let facts =
                inputs.iter().map(|o| target.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            bail!("Unsupported: {operator_code:#?}, inputs: {facts:#?}")
        };
        for (flat, wire) in flat.outputs().unwrap().iter().zip(results.iter()) {
            mapping.insert(flat, *wire);
        }
        Ok(())
    }
}
