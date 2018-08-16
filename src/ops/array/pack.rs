use std::collections::HashMap;
use analyser::interface::*;
use ops::prelude::*;
use tensor::Datum;
use Result;

pub fn pack(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let n = pb.get_input().len();
    let axis = pb.get_attr_int("axis")?;

    Ok(Box::new(Pack::new(dtype, n, axis)))
}

#[derive(Debug, Clone, new)]
pub struct Pack {
    t: DatumType,
    n: usize, // The number of inputs
    axis: usize,
}

impl Pack {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T:Datum>(&self, inputs: Vec<Value>) -> Result<Vec<Value>> {
        use ndarray::Axis;
        let arrays = inputs
            .iter()
            .map(|m| Ok(T::tensor_cast_to_array(&*m)?))
            .collect::<Result<Vec<_>>>()?;
        let views:Vec<_> = arrays.iter().map(|v| v.view().insert_axis(Axis(self.axis))).collect();
        let array = ::ndarray::stack(Axis(self.axis), &*views)?;
        Ok(vec![T::array_into_tensor(array).into()])
    }
}

impl Op for Pack {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: Vec<Value>) -> Result<Vec<Value>> {
        let dt = DatumType::super_type_for(inputs.iter().map(|dt| dt.datum_type()))
            .ok_or("Could not find a supertype")?;
        match dt {
            DatumType::TDim => self.eval_t::<TDim>(inputs),
            DatumType::I32 => self.eval_t::<i32>(inputs),
            DatumType::F32 => self.eval_t::<f32>(inputs),
            _ => panic!("unsupported type"),
        }
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "T"    => Attr::DatumType(self.t),
            "n"    => Attr::Usize(self.n),
            "axis" => Attr::Usize(self.axis),
        }
    }
}

impl InferenceRulesOp for Pack {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        let n = self.n;
        let axis = self.axis;
        solver
            .equals(&inputs.len, n as isize)
            .equals(&outputs.len, 1)
            .equals(&outputs[0].rank, inputs[0].rank.bex() + 1)
            .equals_all((0..n).map(|i| inputs[i].rank.bex()).collect())
            .given_all((0..n).map(move |i| &inputs[i].datum_type), move |solver, dts|{
                if let Some(dt) = DatumType::super_type_for(dts) {
                    solver.equals(&outputs[0].datum_type, dt);
                }
            })
            .given(&inputs[0].rank, move |solver, r: isize| {
                (0..r as usize).for_each(|d| {
                    solver.equals_all((0..n).map(|i| inputs[i].shape[d].bex()).collect());
                })
            })
            .given(&inputs[0].rank, move |solver, r: isize| {
                (0..axis).for_each(|d| {
                    solver.equals(&outputs[0].shape[d], &inputs[0].shape[d]);
                });
                if r > 0 {
                    (axis..(r as usize - 1)).for_each(|d| {
                        solver.equals(&outputs[0].shape[d + 1], &inputs[0].shape[d]);
                    });
                }
            })
            .equals(&outputs[0].shape[axis], n.to_dim());
            ;
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use num::Zero;
    use ops::InferenceOp;
    use super::*;
    use ndarray::prelude::*;
    use Tensor;

    #[test]
    fn pack_0() {
        let inputs = vec![
            Tensor::i32s(&[2], &[1, 4]).unwrap().into(),
            Tensor::i32s(&[2], &[2, 5]).unwrap().into(),
            Tensor::i32s(&[2], &[3, 6]).unwrap().into(),
        ];
        assert_eq!(
            Pack::new(DatumType::I32, 3, 0)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .into_tensor(),
            Tensor::from(arr2(&[[1, 4], [2, 5], [3, 6]]))
        );
        assert_eq!(
            Pack::new(DatumType::I32, 3, 1)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .into_tensor(),
            Tensor::from(arr2(&[[1, 2, 3], [4, 5, 6]]))
        );
    }

    #[test]
    fn pack_1() {
        let pack = Pack::new(DatumType::I32, 3, 0);
        let input = Tensor::i32s(&[0], &[]).unwrap();
        let exp: Tensor = Tensor::i32s(&[1, 0], &[]).unwrap();
        let found = pack.eval(vec![input.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0], false),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
        )
    }

    #[test]
    fn inference_1() {
        let pack = Pack::new(DatumType::I32, 2, 0);
        let a = TensorFact::from(Tensor::from(0i32));
        let b = TensorFact::from(Tensor::from(TDim::zero()));
        let (_, output_facts) = pack.infer(vec![a,b], vec![TensorFact::default()]).unwrap();
        assert_eq!(output_facts, vec![TensorFact::dt_shape(DatumType::TDim, vec!(2))]);

    }

    #[test]
    fn inference_2() {
        let pack = Pack::new(DatumType::I32, 2, 0);
        let a = TensorFact::from(Tensor::from(0i32));
        let b = TensorFact::from(Tensor::from(TDim::zero()));
        let (_, output_facts) = pack.infer_and_propagate(vec![a,b], vec![TensorFact::default()]).unwrap();
        assert_eq!(output_facts, vec![
           Tensor::from(arr1(&[TDim::zero(), TDim::zero()])).into()
        ]);
    }

}
