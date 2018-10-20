use ndarray::prelude::*;

use ops::prelude::*;

#[derive(Debug, Clone, Default, new)]
pub struct Shape;

impl Op for Shape {
    fn name(&self) -> &str {
        "Shape"
    }
}

impl StatelessOp for Shape {
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

        assert_forward!(Shape, input, output);
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

        assert_forward!(Shape, input, output);
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

        assert_forward!(Shape, input, output);
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

        assert_backward!(Shape, input, output);
    }
}
