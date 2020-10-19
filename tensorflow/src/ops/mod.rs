use tract_hir::internal::*;

use crate::model::ParsingContext;
use crate::model::TfOpRegister;
use crate::tfpb::tensorflow::NodeDef;

macro_rules! op_tf {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["tf"]
        }
    };
}

pub mod array;
pub mod control_flow;
pub mod logic;
pub mod math;
pub mod nn;
pub mod quant;
pub mod random;
pub mod rec;
pub mod vars;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    array::register_all_ops(reg);
    control_flow::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    random::register_all_ops(reg);
    rec::register_all_ops(reg);
    vars::register_all_ops(reg);
    reg.insert("Cast", cast);
    reg.insert("Const", konst);
    reg.insert("Identity", |_, _| Ok(Box::new(tract_hir::ops::identity::Identity)));
    reg.insert("NoOp", |_, _| Ok(Box::new(Noop)));
    reg.insert("Placeholder", |_, _| Ok(Box::new(tract_hir::ops::source::Source::new())));
}

fn cast(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = node.get_attr_datum_type("DstT")?;
    Ok(Box::new(::tract_hir::ops::cast(dtype)))
}

fn konst(_ctx: &ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = node.get_attr_datum_type("dtype")?;
    let mat = node.get_attr_tensor("value")?;

    if mat.datum_type() != dtype {
        bail!("Const node {:?} doesn't have the expected {:?} type.", mat, dtype);
    }

    Ok(Box::new(::tract_hir::ops::konst::Const(mat.into())))
}

#[derive(Clone, Debug, new, Hash)]
pub struct Noop;

tract_data::impl_dyn_hash!(Noop);

impl Op for Noop {
    fn name(&self) -> Cow<str> {
        "Noop".into()
    }

    op_tf!();
    op_as_typed_op!();
}

impl EvalOp for Noop {
    fn is_stateless(&self) -> bool {
        true
    }

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

    as_op!();
    to_typed!();
}

impl TypedOp for Noop {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(bool::datum_type(), ())?))
    }

    as_op!();
}
