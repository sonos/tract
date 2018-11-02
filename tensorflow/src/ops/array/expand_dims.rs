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
            let dims = dims.cast_to_array::<i64>()?;
            Ok(Some(ReducedOpRewire {
                new_op: Box::new(::tract_core::ops::array::AddDims::new(
                    dims.view().iter().map(|&i| i as usize).collect(),
                )),
                rewired: tvec!(0),
            }))
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for ExpandDims {
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let (data, dims) = args_2!(inputs);
        let data = data
            .into_tensor()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let dims = dims.as_i32s().ok_or("Expected a i32 matrix")?;
        let mut shape = data.shape().to_vec();
        for d in dims.iter() {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(tvec![Tensor::from(data.into_shape(shape)?).into()])
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
        s.given(&dims.value, move |s, index: Tensor| {
            let index = index.as_i32().unwrap() as usize; // enforced

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
