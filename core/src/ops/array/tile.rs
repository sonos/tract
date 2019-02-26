use crate::ops::prelude::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct Tile;

impl Tile {
    fn eval_t<T: Datum + Copy>(
        &self,
        data: &SharedTensor,
        indices: &[usize],
    ) -> TractResult<SharedTensor> {
        let data = data.to_array_view::<T>()?;
        let output_shape: TVec<usize> = data
            .shape()
            .iter()
            .zip(indices.iter())
            .map(|(&d, &m)| d * m as usize)
            .collect();
        let output = ndarray::ArrayD::from_shape_fn(&*output_shape, |coords| {
            let coords: Vec<usize> = coords
                .slice()
                .iter()
                .zip(data.shape().iter())
                .map(|(&x, &d)| x % d)
                .collect();
            data[&*coords]
        });

        Ok(output.into())
    }
}

impl Op for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }
}

impl StatelessOp for Tile {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (data, multipliers) = args_2!(inputs);
        let multipliers: TVec<usize> = multipliers
            .cast_to::<i32>()?
            .to_array_view::<i32>()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        Ok(tvec!(dispatch_numbers!(Self::eval_t(data.datum_type())(
            &self,
            &data,
            &*multipliers
        ))?))
    }
}

impl InferenceRulesOp for Tile {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape[0], inputs[0].rank.bex().to_dim())?;
        s.given(&inputs[1].value, move |s, mult| {
            for (ix, &m) in mult.cast_to::<i32>()?.as_slice::<i32>()?.iter().enumerate() {
                s.equals(m * inputs[0].shape[ix].bex(), &outputs[0].shape[ix])?;
            }
            Ok(())
        })?;
        // TODO i32 and dim
        /*
        s.given(&inputs[0].rank, |s, rank| {
            for d in 0..(rank as usize) {
                s.equals(inputs[1].value[d].bex() * &inputs[0].shape[d], &outputs[0].shape[d])?;
            }
            Ok(())
        })?;
        */
        Ok(())
    }
}
