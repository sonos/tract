use crate::internal::*;
use crate::infer::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct Tile;

impl Op for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Tile {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, multipliers) = args_2!(inputs);
        let multipliers: TVec<usize> = multipliers
            .cast_to::<i32>()?
            .to_array_view::<i32>()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        TypedTile::new(multipliers).eval(tvec!(data))
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

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref mult) = source.outlet_fact(node.inputs[1])?.value.concretize() {
            let mult: TVec<usize> =
                mult.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|i| *i as usize).collect();
            let input = mapping[&node.inputs[0]];
            let op = TypedTile::new(mult);
            let facts = op.output_facts(&[target.outlet_fact(input)?])?;
            let id = target.add_node(&*node.name, op, facts)?;
            target.add_edge(mapping[&node.inputs[0]], InletId::new(id, 0))?;
            return Ok(tvec!(OutletId::new(id, 0)));
        }
        bail!("shape input is variable")
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone, new, Default)]
pub struct TypedTile {
    multipliers: TVec<usize>,
}

impl TypedTile {
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

impl Op for TypedTile {
    fn name(&self) -> Cow<str> {
        "TypedTile".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for TypedTile {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let result = dispatch_datum!(Self::eval_t(inputs[0].datum_type())(self, &inputs[0]))?;
        Ok(tvec!(result))
    }
}

impl TypedOp for TypedTile {
    typed_op_as_op!();

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
