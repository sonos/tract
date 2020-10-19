use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Tile {
    pub multipliers: TVec<usize>,
}

tract_data::impl_dyn_hash!(Tile);

impl Tile {
    fn eval_t<T: Datum>(&self, data: &Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let view = unsafe { data.to_array_view_unchecked::<T>() };
        let output_shape: TVec<usize> = view
            .shape()
            .iter()
            .zip(self.multipliers.iter())
            .map(|(&d, &m)| d * m as usize)
            .collect();
        let output = ndarray::ArrayD::from_shape_fn(&*output_shape, |coords| {
            let coords: TVec<usize> =
                coords.slice().iter().zip(data.shape().iter()).map(|(&x, &d)| x % d).collect();
            view[&*coords].clone()
        });
        let mut output = output.into_tensor();
        unsafe {
            output.set_datum_type(data.datum_type());
        }

        Ok(output.into_arc_tensor())
    }
}

impl Op for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for Tile {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let result =
            dispatch_datum_by_size!(Self::eval_t(inputs[0].datum_type())(self, &inputs[0]))?;
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
