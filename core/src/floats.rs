use tract_num_traits::Float;

use crate::internal::translator::Translate;
use crate::internal::*;
use crate::ops::array::{Pad, PadMode};
use crate::ops::cnn::{ConvUnary, DeconvUnary};
use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use crate::ops::scan::Scan;
use crate::ops::source::TypedSource;

#[derive(Debug, Default)]
pub struct FloatPrecisionTranslator<T1: Datum + Float, T2: Datum + Float>(PhantomData<(T1, T2)>);

impl<T1: Datum + Float, T2: Datum + Float>
    Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>>
    for FloatPrecisionTranslator<T1, T2>
{
    fn translate_node(
        &self,
        _source: &Graph<TypedFact, Box<dyn TypedOp>>,
        node: &Node<TypedFact, Box<dyn TypedOp>>,
        target: &mut Graph<TypedFact, Box<dyn TypedOp>>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let new_op = if let Some(source) = node.op_as::<TypedSource>() {
            Box::new(TypedSource::new(fact_float_precision_conversion::<T1, T2>(&source.fact)))
        } else if let Some(konst) = node.op_as::<Const>() {
            Box::new(Const(tensor_float_precision_conversion::<T1, T2>(&konst.0)))
        } else if let Some(op) = node.op_as::<ConvUnary>() {
            Box::new(ConvUnary {
                kernel: tensor_float_precision_conversion::<T1, T2>(&op.kernel),
                bias: op.bias.as_ref().map(tensor_float_precision_conversion::<T1, T2>),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<Scan>() {
            let body = FloatPrecisionTranslator::<T1, T2>::default().translate_model(&op.body)?;
            Box::new(Scan { body, ..op.clone() })
        } else if let Some(op) = node.op_as::<EinSum>() {
            Box::new(EinSum {
                operating_dt: dt_float_precision_conversion::<T1, T2>(op.operating_dt),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<DeconvUnary>() {
            Box::new(DeconvUnary {
                kernel: tensor_float_precision_conversion::<T1, T2>(&op.kernel),
                bias: op.bias.as_ref().map(tensor_float_precision_conversion::<T1, T2>),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<Pad>() {
            if let PadMode::Constant(t) = &op.mode {
                Box::new(Pad {
                    mode: PadMode::Constant(tensor_float_precision_conversion::<T1, T2>(t)),
                    ..op.clone()
                })
            } else {
                Box::new(op.clone())
            }
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

fn dt_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(dt: DatumType) -> DatumType {
    if dt == T1::datum_type() {
        T2::datum_type()
    } else {
        dt
    }
}

fn fact_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(
    t: &TypedFact,
) -> TypedFact {
    if t.datum_type == T1::datum_type() {
        let mut t = t.clone();
        t.datum_type = T2::datum_type();
        t
    } else {
        t.clone()
    }
}

fn tensor_float_precision_conversion<T1: Datum + Float, T2: Datum + Float>(
    t: &Arc<Tensor>,
) -> Arc<Tensor> {
    if t.datum_type() == T1::datum_type() {
        t.cast_to::<T2>().unwrap().into_owned().into_arc_tensor()
    } else {
        Arc::clone(t)
    }
}
