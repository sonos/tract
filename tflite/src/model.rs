use std::collections::hash_map::Entry;

use flatbuffers::{FlatBufferBuilder, Push, Vector, WIPOffset};
use tract_hir::internal::*;
use tract_hir::ops::cnn::ConvUnary;
use tract_hir::ops::konst::Const;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops::source::TypedSource;

use crate::registry::Registry;
use crate::tflite::{
    self, Buffer, BufferArgs, BuiltinOperator, BuiltinOptions, CustomOptionsFormat, Model,
    ModelArgs, ModelBuilder, Operator, OperatorArgs, OperatorCode, OperatorCodeArgs, SubGraph,
    SubGraphArgs, SubGraphBuilder, Tensor, TensorArgs, TensorType, Conv2DOptions, Conv2DOptionsArgs, ActivationFunctionType, Padding,
};

pub struct Tflite(Registry);

impl Default for Tflite {
    fn default() -> Self {
        let mut registry = Registry::default();
        crate::ops::register_all(&mut registry);
        Tflite(registry)
    }
}

#[derive(Clone, Debug)]
pub struct TfliteProtoModel(Vec<u8>);

impl TfliteProtoModel {
    fn new(buf: Vec<u8>) -> TractResult<TfliteProtoModel> {
        let _ = tflite::root_as_model(&buf)?;
        Ok(TfliteProtoModel(buf))
    }

    pub fn root(&self) -> tflite::Model {
        unsafe { tflite::root_as_model_unchecked(&self.0) }
        //        tflite::model::Model::from_buffer(&self.0).context("Failed to read flat buffer model")
    }
}

fn write_fact<'f>(
    builder: &mut FlatBufferBuilder<'f>,
    buffers: &mut Vec<WIPOffset<Buffer<'f>>>,
    tensors: &mut Vec<WIPOffset<Tensor<'f>>>,
    name: &str,
    fact: &TypedFact,
) -> TractResult<i32> {
    let buffer = if let Some(k) = &fact.konst {
        let data = builder.create_vector(unsafe { k.as_bytes() });
        let buffer = Buffer::create(builder, &BufferArgs { data: Some(data) });
        buffers.push(buffer);
        buffers.len() as u32 - 1
    } else {
        0
    };
    let shape = fact.shape.as_concrete().unwrap().iter().map(|d| *d as i32).collect_vec();
    let shape = builder.create_vector(&shape);
    let name = builder.create_string(name);
    let tensor = Tensor::create(
        builder,
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

fn operator_code_index<'f>(op_codes: &mut Vec<BuiltinOperator>, builtin: BuiltinOperator) -> u32 {
    if let Some(found) = op_codes.iter().position(|op| op == &builtin) {
        found as u32
    } else {
        op_codes.push(builtin);
        op_codes.len() as u32 - 1
    }
}

fn write_subgraph<'f>(
    builder: &mut FlatBufferBuilder<'f>,
    op_codes: &mut Vec<BuiltinOperator>,
    buffers: &mut Vec<WIPOffset<Buffer<'f>>>,
    model: &TypedModel,
) -> TractResult<WIPOffset<SubGraph<'f>>> {
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
            let tensor = write_fact(builder, buffers, &mut tensors, name.as_str(), &output.fact)?;
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
            inputs.push(write_fact(
                builder,
                buffers,
                &mut tensors,
                &format!("{node_name}.weights"),
                &conv.kernel.clone().into(),
            )?);
            inputs.push(write_fact(
                builder,
                buffers,
                &mut tensors,
                &format!("{node_name}.bias"),
                &conv
                    .bias
                    .clone()
                    .unwrap_or_else(|| {
                        rctensor1(&vec![0f32; conv.pool_spec.output_channel_override.unwrap()])
                    })
                    .into(),
            )?);
            let options = Conv2DOptions::create(builder, &Conv2DOptionsArgs {
                padding: Padding::VALID,
                stride_w: 1,
                stride_h: 1,
                dilation_w_factor: 1,
                dilation_h_factor: 1,
                fused_activation_function: ActivationFunctionType::NONE,
            });
            (BuiltinOperator::CONV_2D, options.as_union_value(), BuiltinOptions::Conv2DOptions)
        } else {
            bail!("Unsupported op")
        };
        let opcode_index = operator_code_index(op_codes, op_code);
        let inputs = builder.create_vector(&inputs);
        let outputs = builder.create_vector(&outputs);
        let operator = Operator::create(
            builder,
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

fn write_model(model: &TypedModel) -> TractResult<FlatBufferBuilder> {
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let mut op_codes = vec![];
    let sentinel = Buffer::create(&mut builder, &BufferArgs { data: None });
    let mut buffers = vec![sentinel];
    let subgraph = write_subgraph(&mut builder, &mut op_codes, &mut buffers, model)?;
    let subgraphs = vec![subgraph];
    let subgraphs = builder.create_vector(&subgraphs);
    let buffers = builder.create_vector(&buffers);
    let operator_codes = op_codes
        .into_iter()
        .map(|code| {
            OperatorCode::create(
                &mut builder,
                &OperatorCodeArgs {
                    deprecated_builtin_code: 0,
                    custom_code: None,
                    version: 1,
                    builtin_code: code,
                },
            )
        })
        .collect_vec();
    let operator_codes = builder.create_vector(&operator_codes);
    let model = Model::create(
        &mut builder,
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
    builder.finish_minimal(model);
    Ok(builder)
}

impl Tflite {
    pub fn write(&self, model: &TypedModel, mut w: impl std::io::Write) -> TractResult<()> {
        let builder = write_model(model)?;
        w.write_all(builder.finished_data())?;
        Ok(())
    }
}

impl Framework<TfliteProtoModel, TypedModel> for Tflite {
    fn proto_model_for_read(
        &self,
        reader: &mut dyn std::io::Read,
    ) -> tract_hir::prelude::TractResult<TfliteProtoModel> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        TfliteProtoModel::new(buf)
    }

    fn model_for_proto_model_with_symbols(
        &self,
        proto: &TfliteProtoModel,
        _symbols: &SymbolTable,
    ) -> TractResult<TypedModel> {
        let root = proto.root();
        let main = &root.subgraphs().context("No subgraphs in Tflite model")?.get(0);
        let mut target = TypedModel::default();
        let mut mapping = HashMap::new();
        for input in main.inputs().context("No inputs in Tflite model")? {
            let (fact, name) = crate::tensors::flat_tensor_to_tract_fact(&root, main, input)?;
            let it = target.add_source(name, fact)?;
            mapping.insert(input, it);
        }
        for op in main.operators().context("No operators in Tflite model")? {
            for input in op.inputs().context("No input in Tflite  operator")? {
                if let Entry::Vacant(slot) = mapping.entry(input) {
                    let (fact, name) =
                        crate::tensors::flat_tensor_to_tract_fact(&root, main, input)?;
                    let value = fact.konst.with_context(|| format!("Error in TF file for operator {:?}. No prior computation nor constant for input {}", op, input))?;
                    let konst = target.add_const(name, value)?;
                    slot.insert(konst);
                }
            }
            self.0.op(&root, main, &op, &mut target, &mut mapping)?;
        }
        let outputs: TVec<_> = main
            .outputs()
            .context("No outputs in Tflite model")?
            .iter()
            .map(|o| mapping[&o])
            .collect();
        target.set_output_outlets(&outputs)?;
        Ok(target)
    }
}
