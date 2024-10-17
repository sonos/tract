use crate::internal::*;
use ndarray::*;

use super::MultiBroadcastTo;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Tile {
    pub multipliers: TVec<TDim>,
}

impl Op for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("multipliers: {:?}", self.multipliers)])
    }

    op_as_typed_op!();
}

impl EvalOp for Tile {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let multipliers: TVec<usize> = self
            .multipliers
            .iter()
            .map(|m| m.eval(&session.resolved_symbols).to_usize())
            .collect::<TractResult<_>>()?;
        let result =
            dispatch_datum_by_size!(eval_t(inputs[0].datum_type())(&inputs[0], &multipliers))?;
        Ok(tvec!(result))
    }
}

impl TypedOp for Tile {
    as_op!();

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let multipliers = self.multipliers.iter().map(|m| m.eval(values)).collect();
        target.wire_node(&node.name, Self { multipliers }, &[mapping[&node.inputs[0]]])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact
            .shape
            .iter()
            .zip(self.multipliers.iter())
            .all(|(i, m)| i.is_one() || m.is_one())
        {
            let output_fact = self.output_facts(&[input_fact])?.remove(0);
            TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs[0..1],
                MultiBroadcastTo { shape: output_fact.shape },
            )
            .map(Some)
        } else {
            Ok(None)
        }
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = inputs[0]
            .shape
            .iter()
            .zip(self.multipliers.iter())
            .map(|(a, b)| a.clone() * b)
            .collect::<TVec<_>>();
        Ok(tvec!(inputs[0].datum_type.fact(shape)))
    }
}

#[derive(Debug, Clone, Hash)]
pub struct DynTile {
    pub multiplier_placeholders: TVec<TDim>,
}

impl DynTile {
    pub fn new(scope: &SymbolScope, rank: usize) -> DynTile {
        let multiplier_placeholders =
            (0..rank).map(|_| scope.new_with_prefix("_tile_mult_").to_dim()).collect();
        DynTile { multiplier_placeholders }
    }
}

impl Op for DynTile {
    fn name(&self) -> Cow<str> {
        "DynTile".into()
    }

    op_as_typed_op!();
}

impl EvalOp for DynTile {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let multipliers = inputs[1].cast_to::<TDim>()?;
        let multipliers: TVec<usize> = multipliers
            .as_slice::<TDim>()?
            .iter()
            .map(|m| Ok(m.eval_to_i64(&session.resolved_symbols)? as usize))
            .collect::<TractResult<_>>()?;
        let result =
            dispatch_datum_by_size!(eval_t(inputs[0].datum_type())(&inputs[0], &multipliers))?;
        Ok(tvec!(result))
    }
}

impl TypedOp for DynTile {
    as_op!();

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(mult) = &model.outlet_fact(node.inputs[1])?.konst {
            let multipliers = mult.cast_to::<TDim>()?.as_slice::<TDim>()?.iter().cloned().collect();
            return TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Tile { multipliers },
            )
            .map(Some);
        }
        Ok(None)
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let multipliers = if let Some(k) = &inputs[1].konst {
            k.cast_to::<TDim>()?.as_slice::<TDim>()?.iter().cloned().collect()
        } else {
            self.multiplier_placeholders.clone()
        };
        let shape =
            inputs[0].shape.iter().zip(multipliers).map(|(a, b)| b * a).collect::<TVec<_>>();
        Ok(tvec!(inputs[0].datum_type.fact(shape)))
    }
}

fn eval_t<T: Datum>(data: &TValue, multipliers: &[usize]) -> TractResult<TValue> {
    let view = unsafe { data.to_array_view_unchecked::<T>() };
    let output_shape: TVec<usize> =
        view.shape().iter().zip(multipliers.iter()).map(|(&d, &m)| d * m).collect();
    let output = ndarray::ArrayD::from_shape_fn(&*output_shape, |coords| {
        let coords: TVec<usize> =
            coords.slice().iter().zip(data.shape().iter()).map(|(&x, &d)| x % d).collect();
        view[&*coords].clone()
    });
    let mut output = output.into_tensor();
    unsafe {
        output.set_datum_type(data.datum_type());
    }

    Ok(output.into_tvalue())
}
