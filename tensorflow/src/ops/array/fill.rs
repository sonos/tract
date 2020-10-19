use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Fill {
    dt: DatumType,
}

tract_data::impl_dyn_hash!(Fill);

pub fn fill(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(Box::new(Fill::new(dtype)))
}

impl Fill {
    fn eval_t<T: Datum>(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (shape, value) = args_2!(inputs);
        let value = value.to_scalar::<T>()?;
        let shape = shape.cast_to::<i32>()?;
        let shape = shape.to_array_view::<i32>()?;
        let array = tract_ndarray::Array::from_shape_fn(
            shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(),
            |_| value.clone(),
        );
        Ok(tvec![array.into_arc_tensor()])
    }
}

impl Op for Fill {
    fn name(&self) -> Cow<str> {
        "Fill".into()
    }

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for Fill {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_datum!(Self::eval_t(self.dt)(self, inputs))
    }
}

impl InferenceRulesOp for Fill {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&inputs[1].datum_type, self.dt)?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(outputs[0].rank.bex().to_dim(), &inputs[0].shape[0])?;
        s.given(&outputs[0].rank, move |s, rank| {
            for dim in 0..(rank as usize) {
                s.equals(&outputs[0].shape[dim], inputs[0].value[dim].bex().to_dim())?;
            }
            Ok(())
        })
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(shape), Some(value)) = (
            target.outlet_fact(mapping[&node.inputs[0]])?.konst.as_ref(),
            target.outlet_fact(mapping[&node.inputs[1]])?.konst.as_ref(),
        ) {
            let mut value = dispatch_datum!(Self::eval_t(value.datum_type())(
                self,
                tvec!(shape.clone(), value.clone())
            ))?;
            let id = target.add_const(&*node.name, value.remove(0))?;
            Ok(tvec!(id))
        } else {
            bail!("Can not type Fill op")
        }
    }
}
