use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct Tile {
    multipliers: TVec<usize>,
}

impl Tile {
    fn eval_t<T: Datum>(&self, data: &Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let data = data.to_array_view::<T>()?;
        let output_shape: TVec<usize> = data
            .shape()
            .iter()
            .zip(self.multipliers.iter())
            .map(|(&d, &m)| d * m as usize)
            .collect();
        let output = ndarray::ArrayD::from_shape_fn(&*output_shape, |coords| {
            let coords: TVec<usize> =
                coords.slice().iter().zip(data.shape().iter()).map(|(&x, &d)| x % d).collect();
            data[&*coords].clone()
        });

        Ok(output.into_arc_tensor())
    }
}

impl Op for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Tile {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let result = dispatch_datum!(Self::eval_t(inputs[0].datum_type())(self, &inputs[0]))?;
        Ok(tvec!(result))
    }
}

impl TypedOp for Tile {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = inputs[0]
            .shape
            .iter()
            .zip(self.multipliers.iter())
            .map(|(a, &b)| a.clone() * b)
            .collect::<TVec<_>>();
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*shape)?))
    }
}
