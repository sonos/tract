use crate::model::ParsingContext;
use crate::tfpb::node_def::NodeDef;
use ndarray::prelude::*;
use num_traits::AsPrimitive;
use std::ops::{Add, Div, Mul, Sub};
use tract_core::internal::*;

#[derive(Debug, Clone, new)]
pub struct Range {
    dtype: DatumType,
}

pub fn range(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("Tidx")?;
    Ok(Box::new(Range::new(dtype)))
}

impl Range {
    fn eval_t<T>(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Datum
            + AsPrimitive<usize>
            + Add<T, Output = T>
            + Div<T, Output = T>
            + Mul<T, Output = T>
            + Sub<T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let (start, limit, delta) = args_3!(inputs);
        let start = *start.to_scalar::<T>()?;
        let limit = *limit.to_scalar::<T>()?;
        let delta = *delta.to_scalar::<T>()?;
        let value =
            Array1::from_shape_fn(((limit - start) / delta).as_(), |ix| ix.as_() * delta + start);
        Ok(tvec![value.into_arc_tensor()])
    }
}

impl Op for Range {
    fn name(&self) -> Cow<str> {
        "tf.Range".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Range {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_numbers!(Self::eval_t(self.dtype)(self, inputs))
    }
}

impl InferenceRulesOp for Range {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[1].datum_type, self.dtype)?;
        s.equals(&inputs[2].datum_type, self.dtype)?;
        s.equals(&outputs[0].datum_type, self.dtype)?;
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&outputs[0].rank, 1)?;
        Ok(())
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(start), Some(limit), Some(delta)) = (
            target.outlet_fact(mapping[&node.inputs[0]])?.konst.as_ref(),
            target.outlet_fact(mapping[&node.inputs[1]])?.konst.as_ref(),
            target.outlet_fact(mapping[&node.inputs[2]])?.konst.as_ref(),
        ) {
            let mut value = dispatch_numbers!(Self::eval_t(start.datum_type())(
                self,
                tvec!(start.clone(), limit.clone(), delta.clone())
            ))?;
            Ok(tvec!(target.add_const(&*node.name, value.remove(0))?))
        } else {
            bail!("Can not type Fill op")
        }
    }
}
