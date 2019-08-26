use tract_core::internal::*;
use crate::tfpb::node_def::NodeDef;
use crate::model::ParsingContext;

#[derive(Debug, Clone, new)]
pub struct Reshape<T: Datum>(PhantomData<T>);

pub fn reshape(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Reshape(dtype)()))
}

impl<T: Datum> Op for Reshape<T> {
    fn name(&self) -> Cow<str> {
        "tf.Reshape".into()
    }
}

impl<T: Datum> StatelessOp for Reshape<T> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, dims) = args_2!(inputs);

        let input = input.into_tensor().into_array::<T>()?;
        let dims = true_dims(dims.as_slice::<i32>()?, input.len());
        let output = input.into_shape(&*dims)?.into_dyn();
        Ok(tvec![output.into_arc_tensor()])
    }
}

impl<T: Datum> InferenceRulesOp for Reshape<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[1].datum_type, DatumType::I32)?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[1].rank, 1)?;
        s.given_2(&inputs[0].shape, &inputs[1].value, move |solver, shape, dims| {
            let dims = dims.as_slice::<i32>().unwrap(); // checked
            if shape.iter().all(|d| !d.is_stream()) {
                let len = shape.iter().map(|d| d.as_const().unwrap() as usize).product();
                let shape = true_dims(dims, len);
                solver.equals(&outputs[0].shape, ShapeFact::from(shape))?;
            }
            Ok(())
        })
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref dims) = target.outlet_fact(mapping[&node.inputs[1]])?.konst {
            let input_shape = target.outlet_fact(mapping[&node.inputs[0]])?;
            let output_shape = if dims.as_slice::<i32>()?.iter().all(|&x| x >= 0) {
                dims.as_slice::<i32>()?.iter().map(|d| *d as usize).collect()
            } else if input_shape.shape.stream_info.is_none() {
                let input_len = input_shape.shape.iter().fold(1, |a,b| a*b.to_integer().unwrap());
                let dims = dims.cast_to::<i32>()?;
                true_dims(dims.as_slice::<i32>()?, input_len as usize)
            } else {
                bail!("Not enough information to infer fixed output shape")
            };
            let op = tract_core::ops::array::IntoShape::new(output_shape);
            target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref())
        } else {
            bail!("Need axes to be const")
        }
    }
}

fn true_dims(dims: &[i32], input_length: usize) -> TVec<usize> {
    let prod: usize = dims.iter().filter(|a| **a != -1).map(|&a| a as usize).product();
    dims.iter().map(|&a| if a == -1 { input_length / prod } else { a as usize }).collect()
}
