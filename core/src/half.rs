use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::binary::UnaryOp;
use crate::ops::cnn::ConvUnary;
use crate::ops::matmul::MatMulUnary;
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
            let mut fact = source.fact.clone();
            if fact.datum_type == f32::datum_type() {
                fact.datum_type = f16::datum_type();
            }
            Box::new(TypedSource::new(fact))
        } else if let Some(op) = node.op_as::<ConvUnary>() {
            Box::new(ConvUnary {
                kernel: op.kernel.cast_to::<f16>()?.into_owned().into_arc_tensor(),
                bias: op
                    .bias
                    .as_ref()
                    .map(|b| b.cast_to::<f16>())
                    .transpose()?
                    .map(|t| t.into_owned().into_arc_tensor()),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<MatMulUnary>() {
            Box::new(MatMulUnary {
                a: op.a.cast_to::<f16>()?.into_owned().into_arc_tensor(),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<UnaryOp>() {
            let mut new = op.clone();
            new.a = op.a.cast_to::<f16>()?.into_owned().into_arc_tensor();
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
