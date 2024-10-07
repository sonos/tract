use std::any::TypeId;

use tract_core::internal::*;

use crate::ser::SubgraphBuilder;
use crate::tflite::{BuiltinOperator, Model, Operator, SubGraph};

pub type ToTract = Box<dyn Fn(&mut DeserOp) -> TractResult<TVec<OutletId>> + Send + Sync + 'static>;
pub type ToTflite<T> = fn(&mut SubgraphBuilder, &TypedModel, &TypedNode, &T) -> TractResult<()>;
pub type ToTfliteRaw = Box<
    dyn Fn(&mut SubgraphBuilder, &TypedModel, &TypedNode) -> TractResult<()>
        + Send
        + Sync
        + 'static,
>;

#[derive(Default)]
pub struct Registry {
    pub to_tract: HashMap<i32, ToTract>,
    pub to_tflite: HashMap<TypeId, ToTfliteRaw>,
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

impl DeserOp<'_> {
    pub fn facts(&self) -> TractResult<TVec<TypedFact>> {
        self.inputs
            .iter()
            .map(|o| self.ctx.target.outlet_fact(*o).cloned())
            .collect::<TractResult<TVec<_>>>()
    }
}

impl Registry {
    pub fn reg_to_tflite<T: Op>(&mut self, tflite: ToTflite<T>) {
        self.to_tflite.insert(
            std::any::TypeId::of::<T>(),
            Box::new(move |b, m, n| tflite(b, m, n, n.op_as::<T>().unwrap())),
        );
    }

    pub fn reg_to_tract<T>(&mut self, op: BuiltinOperator, to: T)
    where
        T: Fn(&mut DeserOp) -> TractResult<TVec<OutletId>> + Send + Sync + 'static,
    {
        self.to_tract.insert(op.0, Box::new(to));
    }

    pub fn deser_op(
        &self,
        model: &Model,
        subgraph: &SubGraph,
        flat_op: &Operator,
        target: &mut TypedModel,
        mapping: &mut HashMap<i32, OutletId>,
    ) -> TractResult<()> {
        let inputs: TVec<OutletId> =
            flat_op.inputs().unwrap().iter().map(|o| mapping[&o]).collect();
        let tensors = subgraph.tensors().unwrap();
        let prefix = tensors.get(flat_op.outputs().unwrap().get(0) as usize).name().unwrap();
        let opcode_index = flat_op.opcode_index();
        let operator_code = model.operator_codes().unwrap().get(opcode_index as _);
        let opcode = if operator_code.deprecated_builtin_code() as i32
            == BuiltinOperator::PLACEHOLDER_FOR_GREATER_OP_CODES.0
        {
            operator_code.builtin_code().0
        } else {
            operator_code.deprecated_builtin_code() as i32
        };
        let ctx = DeserContext { model, subgraph, target };
        let results = if let Some(op) = self.to_tract.get(&opcode) {
            let output_facts = flat_op
                .outputs()
                .unwrap()
                .iter()
                .map(|t| Ok(crate::tensors::flat_tensor_to_tract_fact(model, subgraph, t)?.0))
                .collect::<TractResult<TVec<TypedFact>>>()?;
            (op)(&mut DeserOp {
                ctx,
                prefix,
                flat: flat_op,
                inputs: &inputs,
                output_facts: &output_facts,
            })
            .with_context(|| format!("Opcode is {operator_code:#?}"))?
        } else {
            let facts =
                inputs.iter().map(|o| target.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            bail!("Unsupported: {operator_code:#?}, inputs: {facts:#?}")
        };
        for (flat, wire) in flat_op.outputs().unwrap().iter().zip(results.iter()) {
            mapping.insert(flat, *wire);
        }
        Ok(())
    }
}
