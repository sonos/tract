use tract_core::ops::prelude::*;
use tract_core::TractResult;

pub fn pack(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
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
    fn eval_t<T: Copy + Datum>(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        use ndarray::Axis;
        let arrays = inputs
            .iter()
            .map(|m| Ok(m.cast_to::<T>()?))
            .collect::<TractResult<Vec<_>>>()?;
        let views: Vec<_> = arrays
            .iter()
            .map(|v| v.to_array_view::<T>().unwrap().insert_axis(Axis(self.axis)))
            .collect();
        let array = ::ndarray::stack(Axis(self.axis), &*views)?;
        Ok(tvec![array.into()])
    }
}

impl Op for Pack {
    fn name(&self) -> Cow<str> {
        "tf.Pack".into()
    }
}

impl StatelessOp for Pack {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let dt = DatumType::super_type_for(inputs.iter().map(|dt| dt.datum_type()))
            .ok_or("Could not find a supertype")?;
        match dt {
            DatumType::TDim => self.eval_t::<TDim>(inputs),
            DatumType::I32 => self.eval_t::<i32>(inputs),
            DatumType::F32 => self.eval_t::<f32>(inputs),
            _ => panic!("unsupported type"),
        }
    }
}

impl InferenceRulesOp for Pack {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let axis = self.axis;
        check_input_arity(&inputs, self.n)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, inputs[0].rank.bex() + 1)?;
        s.equals_all((0..self.n).map(|i| inputs[i].rank.bex()).collect())?;
        s.given_all((0..self.n).map(move |i| &inputs[i].datum_type), move |s, dts| {
            if let Some(dt) = DatumType::super_type_for(dts) {
                s.equals(&outputs[0].datum_type, dt)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..r as usize {
                s.equals_all((0..self.n).map(|i| inputs[i].shape[d].bex()).collect())?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..axis {
                s.equals(&outputs[0].shape[d], &inputs[0].shape[d])?;
            }
            if r > 0 {
                for d in axis..(r as usize - 1) {
                    s.equals(&outputs[0].shape[d + 1], &inputs[0].shape[d])?
                }
            }
            Ok(())
        })?;
        s.equals(&outputs[0].shape[axis], self.n.to_dim())
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::prelude::*;
    use num_traits::Zero;
    use tract_core::ops::InferenceOp;
    use tract_core::Tensor;

    #[test]
    fn pack_0() {
        let inputs = tvec![
            arr1(&[1, 4]).into(),
            arr1(&[2, 5]).into(),
            arr1(&[3, 6]).into(),
        ];
        assert_eq!(
            Pack::new(DatumType::I32, 3, 0)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .to_tensor(),
            Tensor::from(arr2(&[[1, 4], [2, 5], [3, 6]]))
        );
        assert_eq!(
            Pack::new(DatumType::I32, 3, 1)
                .eval(inputs.clone())
                .unwrap()
                .remove(0)
                .to_tensor(),
            Tensor::from(arr2(&[[1, 2, 3], [4, 5, 6]]))
        );
    }

    #[test]
    fn pack_1() {
        let pack = Pack::new(DatumType::I32, 3, 0);
        let input = arr1::<i32>(&[]).into();
        let exp: SharedTensor = arr2::<i32, _>(&[[]]).into();
        let found = pack.eval(tvec![input]).unwrap();

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
        let any = TensorFact::default();
        let (_, output_facts) = pack.infer_facts(tvec![&a, &b], tvec![&any]).unwrap();
        let exp: TVec<TensorFact> = tvec!(TensorFact::dt_shape(DatumType::TDim, vec![2usize]));
        assert_eq!(output_facts, exp)
    }

    #[test]
    fn inference_2() {
        let pack = Pack::new(DatumType::I32, 2, 0);
        let a = TensorFact::from(Tensor::from(0i32));
        let b = TensorFact::from(Tensor::from(TDim::zero()));
        let any = TensorFact::default();
        let (_, output_facts) = pack.infer(tvec![&a, &b], tvec![&any]).unwrap();
        let exp: TVec<TensorFact> = tvec!(Tensor::from(arr1(&[TDim::zero(), TDim::zero()])).into());
        assert_eq!(output_facts, exp);
    }

}
