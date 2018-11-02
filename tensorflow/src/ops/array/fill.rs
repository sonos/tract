use std::marker::PhantomData;

use tract_core::ops::prelude::*;

#[derive(Debug, Clone, Default, new)]
pub struct Fill<T: Datum> {
    _phantom: PhantomData<T>,
}

pub fn fill(pb: &::tfpb::node_def::NodeDef) -> TfdResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(Fill(dtype)()))
}

impl<T> Op for Fill<T>
where
    T: Datum,
{
    fn name(&self) -> &str {
        "tf.Fill"
    }
}

impl<T: Datum> StatelessOp for Fill<T> {
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (shape, value) = args_2!(inputs);
        let value = value.to_array_view()?;
        let value: T = value[[]];
        let shape = shape.to_array_view::<i32>()?;
        let array = ::ndarray::Array::from_elem(
            shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(),
            value,
        );
        Ok(tvec![array.into()])
    }
}

impl<T: Datum> InferenceRulesOp for Fill<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
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
}
