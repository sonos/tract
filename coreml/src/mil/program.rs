//! Helpers for assembling MIL `Program` / `Function` / `Block` protos.

use std::collections::HashMap;

use crate::proto::core_ml::specification::mil_spec as mil;

/// The opset name to embed in `Function::opset` and the `block_specializations`
/// key. CoreML 7 (iOS 17 / macOS 14) is the minimum we target — see
/// `notes/phase-0-recon.md` for rationale.
pub const DEFAULT_OPSET: &str = "CoreML7";

/// Build a single-function single-block-spec MIL Program.
///
/// This is the typical shape for a fused subgraph: one entry function called
/// `"main"`, one block specialization keyed by the opset name (so all ops in
/// the block must be valid in that opset), and the function output is the
/// block output.
pub fn single_function_program(
    inputs: Vec<mil::NamedValueType>,
    outputs: Vec<String>,
    operations: Vec<mil::Operation>,
) -> mil::Program {
    let block = mil::Block { inputs: vec![], outputs, operations, attributes: HashMap::new() };
    let mut block_specs = HashMap::new();
    block_specs.insert(DEFAULT_OPSET.to_string(), block);

    let function = mil::Function {
        inputs,
        opset: DEFAULT_OPSET.to_string(),
        block_specializations: block_specs,
        attributes: HashMap::new(),
    };

    let mut functions = HashMap::new();
    functions.insert("main".to_string(), function);

    mil::Program { version: 1, functions, doc_string: String::new(), attributes: HashMap::new() }
}
