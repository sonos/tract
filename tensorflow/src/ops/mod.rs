use tract_core::internal::*;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;

#[macro_use]
mod macros;

pub mod array;
pub mod logic;
pub mod math;
pub mod nn;
pub mod quant;
pub mod rec;
pub mod vars;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    array::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    rec::register_all_ops(reg);
    vars::register_all_ops(reg);
    reg.insert("Cast", cast);
    reg.insert("Const", konst);
    reg.insert("Identity", |_| Ok(Box::new(Identity::default())));
    reg.insert("NoOp", |_| Ok(Box::new(Noop)));
    reg.insert("Placeholder", placeholder);
}

pub fn cast(node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let dtype = node.get_attr_datum_type("DstT")?;
    Ok(Box::new(::tract_core::ops::cast::Cast::new(dtype)))
}

pub fn konst(node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!("Const node {:?} doesn't have the expected {:?} type.", mat, dtype);
    }

    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(mat)))
}

pub fn placeholder(_node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    Ok(Box::new(::tract_core::ops::source::Source::new()))
}

#[derive(Clone, Debug, new)]
pub struct Noop;

impl Op for Noop {
    fn name(&self) -> Cow<str> {
        "tf.Noop".into()
    }
}

impl StatelessOp for Noop {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(Tensor::from(false).into()))
    }
}

impl InferenceRulesOp for Noop {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, bool::datum_type())?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }

    inference_op_as_op!();
}

#[derive(Clone, Debug, new, Default)]
struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "tf.Identity".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if node.inputs.len() == 1 {
            Ok(Some(TypedModelPatch::single_unary_op(
                model,
                node,
                tract_core::ops::identity::Identity::default(),
            )?))
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for Identity {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(inputs.remove(0)))
    }
}

impl InferenceRulesOp for Identity {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        Ok(())
    }

    inference_op_as_op!();
}
