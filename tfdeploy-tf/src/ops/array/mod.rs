use ndarray::prelude::*;
use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;
use ops::OpRegister;

mod concatv2;
mod fill;
mod pack;
mod pad;
mod reshape;
mod squeeze;
mod strided_slice;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", concatv2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity",
               |_| Ok(Box::new(::tfdeploy::ops::identity::Identity::default())));
    reg.insert("Fill", fill::fill);
    reg.insert("Pack", pack::pack);
    reg.insert("Pad", pad::pad);
    reg.insert("Reshape", reshape::reshape);
    reg.insert("Shape", Shape::build);
    reg.insert("Squeeze", squeeze::squeeze);
    reg.insert("StridedSlice", strided_slice::build);
}

#[derive(Debug, Clone)]
pub struct ExpandDims;

impl ExpandDims {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
        Ok(Box::new(ExpandDims))
    }
}

impl Op for ExpandDims {
    fn name(&self) -> &str {
        "tf.ExpandDims"
    }
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
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

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        _: &mut Box<OpBuffer>,
    ) -> TfdResult<Option<TVec<Value>>> {
        let (data, dims) = args_2!(inputs);

        let dims = if let StepValue::Const(dims) = dims {
            dims
        } else {
            bail!("Dims input should not be streamed.")
        };

        let data = data.into_stream().ok_or("Data input should be streamed.")?;

        match data.chunk {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(tvec![tv, dims])?)),
        }
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


#[derive(Debug, Clone)]
pub struct Shape;

impl Shape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
        Ok(Box::new(Shape))
    }
}

impl Op for Shape {
    fn name(&self) -> &str {
        "tf.Shape"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let shape: Vec<i32> = data.shape().into_iter().map(|s| *s as i32).collect();
        Ok(tvec![Tensor::from(Array1::from_vec(shape)).into()])
    }
}

impl InferenceRulesOp for Shape {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, DatumType::TDim)?;
        s.equals(&outputs[0].rank, 1)?;
        s.given(&inputs[0].rank, move |s, r| {
            s.equals(&outputs[0].shape[0], r.to_dim())
        })?;
        s.given(&outputs[0].shape[0], move |s, r| {
            if let Ok(d) = r.to_integer() {
                s.equals(&inputs[0].rank, d)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].shape, move |s, shape: Vec<TDim>| {
            let array1: Array1<TDim> = Array1::from_vec(shape);
            let tensor: Tensor = Tensor::from(array1);
            s.equals(&outputs[0].value, tensor)
        })?;
        s.given(&outputs[0].value, move |s, shape: Tensor| {
            let shape = shape.take_dims().unwrap(); // checked
            s.equals(
                &inputs[0].shape,
                shape.into_iter().cloned().collect::<Vec<TDim>>(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tfpb::node;

    #[test]
    fn shape_inference_1() {
        let input = TensorFact {
            datum_type: typefact!(DatumType::F32),
            shape: shapefact![1, _, _; ..],
            value: valuefact!(_),
        };

        let output = TensorFact {
            datum_type: typefact!(DatumType::TDim),
            shape: shapefact![_],
            value: valuefact!(_),
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_2() {
        let input = TensorFact {
            datum_type: typefact!(DatumType::F32),
            shape: shapefact![1, _, _],
            value: valuefact!(_),
        };

        let output = TensorFact {
            datum_type: typefact!(DatumType::TDim),
            shape: shapefact![3],
            value: valuefact!(_),
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_3() {
        let input = TensorFact {
            datum_type: typefact!(DatumType::F32),
            shape: shapefact![1, 2, 3],
            value: valuefact!(_),
        };

        let output = TensorFact {
            datum_type: typefact!(DatumType::TDim),
            shape: shapefact![3],
            value: valuefact!(Tensor::dims(&[3], &[1.to_dim(), 2.to_dim(), 3.to_dim()]).unwrap()),
        };

        assert_forward!(Shape::build(&node()).unwrap(), input, output);
    }

    #[test]
    fn shape_inference_4() {
        let input = TensorFact {
            datum_type: typefact!(_),
            shape: shapefact![1, 2, 3],
            value: valuefact!(_),
        };

        let output = TensorFact {
            datum_type: typefact!(DatumType::TDim),
            shape: shapefact![3],
            value: valuefact!(Tensor::dims(&[3], &[1.to_dim(), 2.to_dim(), 3.to_dim()]).unwrap()),
        };

        assert_backward!(Shape::build(&node()).unwrap(), input, output);
    }
}
