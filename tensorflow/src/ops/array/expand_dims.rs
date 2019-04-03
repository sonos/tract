use tract_core::internal::*;

pub fn build(_pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    Ok(Box::new(ExpandDims))
}

#[derive(Debug, Clone)]
pub struct ExpandDims;

impl ExpandDims {
    fn eval_t<T: Datum + Copy>(
        &self,
        data: SharedTensor,
        shape: &[usize],
    ) -> TractResult<TVec<SharedTensor>> {
        let data = data.to_array::<T>()?;
        Ok(tvec![Tensor::from(data.into_shape(&*shape)?).into()])
    }
}

impl Op for ExpandDims {
    fn name(&self) -> Cow<str> {
        "tf.ExpandDims".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut inputs = model.node_input_facts(node.id)?;
        let (_, dims) = args_2!(inputs);
        if let Some(ref dims) = dims.konst {
            let dims = dims.cast_to::<i64>()?;
            let op = ::tract_core::ops::array::AddDims::new(
                dims.to_array_view::<i64>()?.iter().map(|&i| i as usize).collect(),
            );
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }
}

impl StatelessOp for ExpandDims {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (data, dims) = args_2!(inputs);
        let dims = dims.to_array_view::<i32>()?;
        let mut shape: TVec<usize> = data.shape().into();
        for d in dims.iter() {
            let d = if *d >= 0 { *d } else { *d + 1 + data.shape().len() as i32 } as usize;
            shape.insert(d, 1);
        }
        dispatch_copy!(Self::eval_t(data.datum_type())(self, data, &*shape))
    }
}

impl InferenceRulesOp for ExpandDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let data = &inputs[0];
        let dims = &inputs[1];
        let output = &outputs[0];

        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&dims.datum_type, DatumType::I32)?;
        s.equals(&data.datum_type, &output.datum_type)?;
        s.equals(data.rank.bex() + 1, &output.rank)?;
        s.given_2(&dims.value, &data.rank, move |s, index, rank| {
            let mut index = *(index.to_scalar::<i32>()?);
            if index < 0 {
                index += rank
            }
            let index = index as usize;

            for i in 0..index {
                s.equals(&output.shape[i], &data.shape[i])?;
            }

            s.equals(output.shape[index].bex(), 1i32.to_dim().bex())?;

            s.given(&data.rank, move |s, rank| {
                for i in index..(rank as usize) {
                    s.equals(&output.shape[i + 1], &data.shape[i])?;
                }
                Ok(())
            })
        })
    }
}
