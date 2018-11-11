use tract_core::ops::prelude::*;

pub fn build(_pb: &::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    Ok(Box::new(ExpandDims))
}

#[derive(Debug, Clone)]
pub struct ExpandDims;

impl Op for ExpandDims {
    fn name(&self) -> &str {
        "tf.ExpandDims"
    }

    fn reduce(
        &self,
        mut inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
    ) -> TractResult<Option<ReducedOpRewire>> {
        let (_, dims) = args_2!(inputs);
        if let Some(dims) = dims.concretize() {
            let dims = dims.cast_to::<i64>()?;
            Ok(Some(ReducedOpRewire {
                new_op: Box::new(::tract_core::ops::array::AddDims::new(
                    dims.to_array_view::<i64>()?.iter().map(|&i| i as usize).collect(),
                )),
                rewired: tvec!(0),
            }))
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for ExpandDims {
    fn eval(&self, mut inputs: TVec<Tensor>) -> TractResult<TVec<Tensor>> {
        let (data, dims) = args_2!(inputs);
        let data = data.to_array::<f32>()?;
        let dims = dims.to_array_view::<i32>()?;
        let mut shape = data.shape().to_vec();
        for d in dims.iter() {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(tvec![DtArray::from(data.into_shape(shape)?).into()])
    }
}

impl InferenceRulesOp for ExpandDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        let data = &inputs[0];
        let dims = &inputs[1];
        let output = &outputs[0];

        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&dims.datum_type, DatumType::I32)?;
        s.equals(&dims.rank, 0)?;
        s.equals(&data.datum_type, &output.datum_type)?;
        s.equals_zero(data.rank.bex() + 1 - &output.rank)?;
        s.given(&dims.value, move |s, index| {
            let index = index.to_scalar::<i32>()? as usize;

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
