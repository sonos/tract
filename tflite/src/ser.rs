use tract_hir::internal::*;
use tract_hir::ops::konst::Const;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops::source::TypedSource;

use crate::registry::Registry;
use crate::tflite::{
    Buffer, BufferArgs, BuiltinOperator, BuiltinOptions, CustomOptionsFormat, Model, ModelArgs,
    Operator, OperatorArgs, OperatorCode, OperatorCodeArgs, QuantizationDetails,
    QuantizationParameters, QuantizationParametersArgs, SubGraph, SubGraphArgs, Tensor, TensorArgs,
};
use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};

#[derive(Debug, PartialEq, Copy, Clone, new)]
pub struct BuiltinOp {
    deprecated_builtin_code: i8,
    version: i32,
    code: BuiltinOperator,
    options_type: BuiltinOptions,
}

pub struct ModelBuilder<'f, 'b> {
    pub registry: &'b Registry,
    pub builder: &'b mut FlatBufferBuilder<'f>,
    pub op_codes: &'b mut Vec<BuiltinOp>,
    pub buffers: &'b mut Vec<WIPOffset<Buffer<'f>>>,
}

impl<'f, 'b> ModelBuilder<'f, 'b> {
    pub fn write_model(&mut self, model: &TypedModel) -> TractResult<()> {
        let mut subgraph = SubgraphBuilder::new(self);
        subgraph.write_subgraph(model)?;
        let subgraph = subgraph.finish(model)?;
        let subgraphs = vec![subgraph];
        let subgraphs = self.builder.create_vector(&subgraphs);
        let buffers = self.builder.create_vector(self.buffers);
        let operator_codes = self
            .op_codes
            .iter()
            .map(|code| {
                OperatorCode::create(
                    self.builder,
                    &OperatorCodeArgs {
                        deprecated_builtin_code: code.deprecated_builtin_code,
                        custom_code: None,
                        version: code.version,
                        builtin_code: code.code,
                    },
                )
            })
            .collect_vec();
        let operator_codes = self.builder.create_vector(&operator_codes);
        let model = Model::create(
            self.builder,
            &ModelArgs {
                version: 3,
                operator_codes: Some(operator_codes),
                subgraphs: Some(subgraphs),
                description: None,
                buffers: Some(buffers),
                metadata_buffer: None,
                metadata: None,
                signature_defs: None,
            },
        );
        self.builder.finish(model, Some("TFL3"));
        Ok(())
    }

    fn operator_code_index(&mut self, builtin: BuiltinOp) -> u32 {
        if let Some(found) = self.op_codes.iter().position(|op| op == &builtin) {
            found as u32
        } else {
            self.op_codes.push(builtin);
            self.op_codes.len() as u32 - 1
        }
    }
}

pub struct SubgraphBuilder<'f, 'b, 'mb> {
    pub model: &'mb mut ModelBuilder<'f, 'b>,
    pub tensors: Vec<WIPOffset<Tensor<'f>>>,
    pub operators: Vec<WIPOffset<Operator<'f>>>,
    pub outlets_to_tensors: HashMap<OutletId, i32>,
}

impl<'f, 'b, 'mb> SubgraphBuilder<'f, 'b, 'mb> {
    fn new(model: &'mb mut ModelBuilder<'f, 'b>) -> SubgraphBuilder<'f, 'b, 'mb> {
        SubgraphBuilder {
            model,
            tensors: vec![],
            operators: vec![],
            outlets_to_tensors: HashMap::new(),
        }
    }

    pub fn fb<'short>(&'short mut self) -> &'short mut FlatBufferBuilder<'f>
    where
        'f: 'short,
    {
        self.model.builder
    }

    pub fn write_fact(
        &mut self,
        name: impl AsRef<str>,
        fact: impl Into<TypedFact>,
    ) -> TractResult<i32> {
        let fact = fact.into();
        let buffer = if let Some(k) = &fact.konst {
            let data = self.fb().create_vector(unsafe { k.as_bytes() });
            let buffer = Buffer::create(self.fb(), &BufferArgs { data: Some(data) });
            self.model.buffers.push(buffer);
            self.model.buffers.len() as u32 - 1
        } else {
            0
        };
        let shape = fact.shape.as_concrete().unwrap().iter().map(|d| *d as i32).collect_vec();
        let shape = self.fb().create_vector(&shape);
        let name = self.fb().create_string(name.as_ref());
        let quantization = if let Some(qp) = fact.datum_type.qparams() {
            let zero_point = self.fb().create_vector(&[qp.zp_scale().0 as i64]);
            let scale = self.fb().create_vector(&[qp.zp_scale().1]);
            let qp = QuantizationParameters::create(
                self.fb(),
                &QuantizationParametersArgs {
                    min: None,
                    max: None,
                    zero_point: Some(zero_point),
                    scale: Some(scale),
                    details: None,
                    details_type: QuantizationDetails::NONE,
                    quantized_dimension: 0,
                },
            );
            Some(qp)
        } else {
            None
        };
        let tensor = Tensor::create(
            self.fb(),
            &TensorArgs {
                name: Some(name),
                buffer,
                is_variable: false,
                quantization,
                shape: Some(shape),
                type_: fact.datum_type.try_into()?,
                sparsity: None,
                shape_signature: None,
                has_rank: true,
                variant_tensors: None,
            },
        );
        self.tensors.push(tensor);
        Ok(self.tensors.len() as i32 - 1)
    }

    fn write_subgraph(&mut self, model: &TypedModel) -> TractResult<()> {
        for &node_id in &model.eval_order()? {
            let node = &model.nodes[node_id];
            // will serialize constants at the demand of operators only
            if node.op_is::<Const>() {
                continue;
            }
            // create fb tensors for all outputs
            for (slot, output) in node.outputs.iter().enumerate() {
                let name = model
                    .outlet_labels
                    .get(&OutletId::new(node.id, slot))
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(format!("outlet_{}_{}", node_id, slot)));
                let tensor = self.write_fact(name.as_str(), &output.fact)?;
                let outlet = OutletId::new(node.id, slot);
                self.outlets_to_tensors.insert(outlet, tensor);
            }
            // Source inputs are not reified
            if node.op_is::<TypedSource>() {
                continue;
            } else if let Some(to_tflite) =
                self.model.registry.to_tflite.get(&(*(node.op)).type_id())
            {
                to_tflite(self, model, node)?;
            } else {
                bail!("Unsupported op: {}", node)
            };
        }
        Ok(())
    }

    pub fn write_op_with_options(
        &mut self,
        inputs: &[i32],
        outputs: &[i32],
        op: BuiltinOp,
        builtin_options: WIPOffset<UnionWIPOffset>,
    ) -> TractResult<()> {
        let opcode_index = self.model.operator_code_index(op);
        let inputs = self.fb().create_vector(inputs);
        let outputs = self.fb().create_vector(outputs);
        let operator = Operator::create(
            self.fb(),
            &OperatorArgs {
                inputs: Some(inputs),
                outputs: Some(outputs),
                opcode_index,
                builtin_options: Some(builtin_options),
                builtin_options_type: op.options_type,
                custom_options: None,
                custom_options_format: CustomOptionsFormat::FLEXBUFFERS,
                mutating_variable_inputs: None,
                intermediates: None,
            },
        );
        self.operators.push(operator);
        Ok(())
    }

    fn finish(self, model: &TypedModel) -> TractResult<WIPOffset<SubGraph<'f>>> {
        let Self { model: ModelBuilder { builder, .. }, tensors, operators, outlets_to_tensors } =
            self;
        let inputs = model.inputs.iter().map(|i| outlets_to_tensors[i]).collect_vec();
        let outputs = model.outputs.iter().map(|i| outlets_to_tensors[i]).collect_vec();
        let inputs = builder.create_vector(&inputs);
        let outputs = builder.create_vector(&outputs);
        let tensors = builder.create_vector(&tensors);
        let operators = builder.create_vector(&operators);

        Ok(SubGraph::create(
            builder,
            &SubGraphArgs {
                name: None,
                tensors: Some(tensors),
                inputs: Some(inputs),
                outputs: Some(outputs),
                operators: Some(operators),
            },
        ))
    }
}
