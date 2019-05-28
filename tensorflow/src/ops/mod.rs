use tract_core::internal::*;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;
use crate::model::ParsingContext;

#[macro_use]
mod macros;

pub mod array;
pub mod logic;
pub mod math;
pub mod nn;
pub mod quant;
pub mod random;
pub mod rec;
pub mod vars;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    array::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    random::register_all_ops(reg);
    rec::register_all_ops(reg);
    vars::register_all_ops(reg);
    reg.insert("Cast", cast);
    reg.insert("Const", konst);
    reg.insert("Identity", |_, _| Ok(Box::new(tract_core::ops::identity::Identity)));
    reg.insert("NoOp", |_, _| Ok(Box::new(Noop)));
    reg.insert("Placeholder", |_, _| Ok(Box::new(::tract_core::ops::source::Source::new())));
}

fn cast(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let dtype = node.get_attr_datum_type("DstT")?;
    Ok(Box::new(::tract_core::ops::cast::Cast::new(dtype)))
}

fn konst(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<InferenceOp>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!("Const node {:?} doesn't have the expected {:?} type.", mat, dtype);
    }

    Ok(Box::new(::tract_core::ops::konst::Const::for_tensor(mat)))
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
