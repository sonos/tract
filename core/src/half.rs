use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::binary::UnaryOp;
use crate::ops::cnn::ConvUnary;
use crate::ops::matmul::MatMulUnary;
use crate::ops::scan::{InputMapping, Scan, StateInitializer};
use crate::ops::source::TypedSource;

#[derive(Debug)]
pub struct HalfTranslator;

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for HalfTranslator {
    fn translate_node(
        &self,
        _source: &Graph<TypedFact, Box<dyn TypedOp>>,
        node: &Node<TypedFact, Box<dyn TypedOp>>,
        target: &mut Graph<TypedFact, Box<dyn TypedOp>>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let new_op = if let Some(source) = node.op_as::<TypedSource>() {
            Box::new(TypedSource::new(fact_f32_to_f16(&source.fact)))
        } else if let Some(op) = node.op_as::<ConvUnary>() {
            Box::new(ConvUnary {
                kernel: tensor_f32_to_f16(&op.kernel),
                bias: op.bias.as_ref().map(tensor_f32_to_f16),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<MatMulUnary>() {
            Box::new(MatMulUnary { a: tensor_f32_to_f16(&op.a), ..op.clone() })
        } else if let Some(op) = node.op_as::<UnaryOp>() {
            let mut new = op.clone();
            new.a = tensor_f32_to_f16(&op.a);
            Box::new(new)
        } else if let Some(op) = node.op_as::<Scan>() {
            let mut new = op.clone();
            new.body = HalfTranslator.translate_model(&op.body)?;
            for im in &mut new.input_mapping {
                if let InputMapping::State { initializer: StateInitializer::Value(v) } = im {
                    *v = tensor_f32_to_f16(v)
                }
            }
            Box::new(new)
        } else {
            node.op.clone()
        };
        target.wire_node(
            &node.name,
            new_op,
            &node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>(),
        )
    }
}

fn fact_f32_to_f16(t: &TypedFact) -> TypedFact {
    if t.datum_type == f32::datum_type() {
        let mut t = t.clone();
        t.datum_type = f16::datum_type();
        t
    } else {
        t.clone()
    }
}

fn tensor_f32_to_f16(t: &Arc<Tensor>) -> Arc<Tensor> {
    if t.datum_type() == f32::datum_type() {
        t.cast_to::<f16>().unwrap().into_owned().into_arc_tensor()
    } else {
        Arc::clone(t)
    }
}
