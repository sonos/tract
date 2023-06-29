use tract_hir::internal::*;
use tract_hir::ops::cnn::ConvUnary;
use tract_hir::ops::konst::Const;
use tract_hir::ops::nn::DataFormat;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops::source::TypedSource;

use crate::tflite::{
    ActivationFunctionType, Buffer, BufferArgs, BuiltinOperator, BuiltinOptions, Conv2DOptions,
    Conv2DOptionsArgs, CustomOptionsFormat, Operator, OperatorArgs, Padding, SubGraph,
    SubGraphArgs, Tensor, TensorArgs, TensorType, OperatorCode, OperatorCodeArgs, Model, ModelArgs,
};
use flatbuffers::{FlatBufferBuilder, WIPOffset};

pub(crate) struct TfliteBuilder<'f, 'b> {
    pub builder: &'b mut FlatBufferBuilder<'f>,
    pub op_codes: &'b mut Vec<BuiltinOperator>,
    pub buffers: &'b mut Vec<WIPOffset<Buffer<'f>>>,
}

impl<'f, 'b> TfliteBuilder<'f, 'b> {
    fn write_fact(
        &mut self,
        tensors: &mut Vec<WIPOffset<Tensor<'f>>>,
        name: &str,
        fact: &TypedFact,
    ) -> TractResult<i32> {
        let buffer = if let Some(k) = &fact.konst {
            let data = self.builder.create_vector(unsafe { k.as_bytes() });
            let buffer = Buffer::create(&mut self.builder, &BufferArgs { data: Some(data) });
            self.buffers.push(buffer);
            self.buffers.len() as u32 - 1
        } else {
            0
        };
        let shape = fact.shape.as_concrete().unwrap().iter().map(|d| *d as i32).collect_vec();
        let shape = self.builder.create_vector(&shape);
        let name = self.builder.create_string(name);
        let tensor = Tensor::create(
            &mut self.builder,
            &TensorArgs {
                name: Some(name),
                buffer,
                is_variable: false,
                quantization: None,
                shape: Some(shape),
                type_: TensorType::FLOAT32,
                sparsity: None,
                shape_signature: None,
                has_rank: true,
                variant_tensors: None,
            },
        );
        tensors.push(tensor);
        Ok(tensors.len() as i32 - 1)
    }

    fn operator_code_index(op_codes: &mut Vec<BuiltinOperator>, builtin: BuiltinOperator) -> u32 {
        if let Some(found) = op_codes.iter().position(|op| op == &builtin) {
            found as u32
        } else {
            op_codes.push(builtin);
            op_codes.len() as u32 - 1
        }
    }

    fn write_subgraph(&mut self, model: &TypedModel) -> TractResult<WIPOffset<SubGraph<'f>>> {
        let mut tensors: Vec<WIPOffset<Tensor<'f>>> = vec![];
        let mut operators: Vec<WIPOffset<Operator>> = vec![];
        let mut outlets_to_tensors = HashMap::new();
        for &node_id in &model.eval_order()? {
            let node = &model.nodes[node_id];
            let node_name = &node.name;
            for (slot, output) in node.outputs.iter().enumerate() {
                let name = model
                    .outlet_labels
                    .get(&OutletId::new(node.id, slot))
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(format!("outlet_{}_{}", node_id, slot)));
                let tensor = self.write_fact(&mut tensors, name.as_str(), &output.fact)?;
                let outlet = OutletId::new(node.id, slot);
                outlets_to_tensors.insert(outlet, tensor);
            }
            if node.op_is::<TypedSource>() || node.op_is::<Const>() {
                continue;
            }
            let mut inputs = node.inputs.iter().map(|o| outlets_to_tensors[o]).collect_vec();
            let outputs = (0..node.outputs.len())
                .map(|o| outlets_to_tensors[&OutletId::new(node_id, o)])
                .collect_vec();
            let (op_code, options, options_type) = if let Some(conv) = node.op_as::<ConvUnary>() {
                inputs.push(self.write_fact(
                    &mut tensors,
                    &format!("{node_name}.weights"),
                    &conv.kernel.clone().into(),
                )?);
                inputs.push(
                    self.write_fact(
                        &mut tensors,
                        &format!("{node_name}.bias"),
                        &conv
                            .bias
                            .clone()
                            .unwrap_or_else(|| {
                                rctensor1(&vec![
                                    0f32;
                                    conv.pool_spec.output_channel_override.unwrap()
                                ])
                            })
                            .into(),
                    )?,
                );
                ensure!(conv.pool_spec.data_format == DataFormat::NHWC);
                ensure!(model.node_input_facts(node.id)?[0].rank() == 4);
                let options = Conv2DOptions::create(
                    &mut self.builder,
                    &Conv2DOptionsArgs {
                        padding: Padding::VALID,
                        stride_w: 1,
                        stride_h: 1,
                        dilation_w_factor: 1,
                        dilation_h_factor: 1,
                        fused_activation_function: ActivationFunctionType::NONE,
                    },
                );
                (BuiltinOperator::CONV_2D, options.as_union_value(), BuiltinOptions::Conv2DOptions)
            } else {
                bail!("Unsupported op: {}", node)
            };
            let opcode_index = Self::operator_code_index(&mut self.op_codes, op_code);
            let inputs = self.builder.create_vector(&inputs);
            let outputs = self.builder.create_vector(&outputs);
            let operator = Operator::create(
                &mut self.builder,
                &OperatorArgs {
                    inputs: Some(inputs),
                    outputs: Some(outputs),
                    opcode_index,
                    builtin_options: Some(options),
                    builtin_options_type: options_type,
                    custom_options: None,
                    custom_options_format: CustomOptionsFormat::FLEXBUFFERS,
                    mutating_variable_inputs: None,
                    intermediates: None,
                },
            );
            operators.push(operator)
        }
        let inputs = model.inputs.iter().map(|i| outlets_to_tensors[i]).collect_vec();
        let outputs = model.outputs.iter().map(|i| outlets_to_tensors[i]).collect_vec();

        let inputs = self.builder.create_vector(&inputs);
        let outputs = self.builder.create_vector(&outputs);
        let tensors = self.builder.create_vector(&tensors);
        let operators = self.builder.create_vector(&operators);

        Ok(SubGraph::create(
            &mut self.builder,
            &SubGraphArgs {
                name: None,
                tensors: Some(tensors),
                inputs: Some(inputs),
                outputs: Some(outputs),
                operators: Some(operators),
            },
        ))
    }

    pub fn write_model(&mut self, model: &TypedModel) -> TractResult<()> {
        let subgraphs = vec![self.write_subgraph(model)?];
        let subgraphs = self.builder.create_vector(&subgraphs);
        let buffers = self.builder.create_vector(&mut self.buffers);
        let operator_codes = self
            .op_codes
            .iter()
            .map(|code| {
                OperatorCode::create(
                    &mut self.builder,
                    &OperatorCodeArgs {
                        deprecated_builtin_code: 0,
                        custom_code: None,
                        version: 1,
                        builtin_code: *code,
                    },
                )
            })
            .collect_vec();
        let operator_codes = self.builder.create_vector(&operator_codes);
        let model = Model::create(
            &mut self.builder,
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
        self.builder.finish_minimal(model);
        Ok(())
    }
}
