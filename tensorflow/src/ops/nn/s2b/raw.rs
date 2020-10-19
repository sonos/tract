use tract_hir::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, new, Hash)]
pub struct SpaceToBatch {
    datum_type: DatumType,
}

tract_data::impl_dyn_hash!(SpaceToBatch);

impl Op for SpaceToBatch {
    fn name(&self) -> Cow<str> {
        "SpaceToBatch".into()
    }

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for SpaceToBatch {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let block_shape = block_shape.cast_to::<i32>()?;
        let block_shape = block_shape.to_array_view::<i32>()?.into_dimensionality()?;
        let paddings = paddings.cast_to::<i32>()?;
        let paddings = paddings.to_array_view::<i32>()?.into_dimensionality()?;
        let r = dispatch_numbers!(super::space_to_batch(input.datum_type())(
            input,
            &block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl InferenceRulesOp for SpaceToBatch {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        rules(s, self.datum_type, &outputs[0], &inputs[0], &inputs[1], &inputs[2])
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(block_shape), Some(paddings)) = (
            target.outlet_fact(mapping[&node.inputs[1]])?.konst.clone(),
            target.outlet_fact(mapping[&node.inputs[2]])?.konst.clone(),
        ) {
            let paddings = paddings.cast_to::<TDim>()?;
            let paddings_view = paddings.to_array_view::<TDim>()?.into_dimensionality::<Ix2>()?;
            let mut paddings = tvec![];
            for p in paddings_view.outer_iter() {
                let pad = match (p[0].to_usize(), p[1].to_usize()) {
                    (Ok(bef), Ok(aft)) => {
                        super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize)
                    }
                    (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                    (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                    _ => bail!("Failed to unarize SpaceToBatch because of padding"),
                };
                paddings.push(pad);
            }
            let op = super::unary::SpaceToBatchUnary::new(
                self.datum_type,
                target.outlet_fact(mapping[&node.inputs[0]])?.shape.to_tvec(),
                node.outputs[0]
                    .fact
                    .shape
                    .concretize()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect::<TVec<_>>(),
                block_shape.clone().into_tensor().into_array::<i32>()?.into_dimensionality()?,
                paddings,
            );
            target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref())
        } else {
            bail!("Need fixed block shape and padding")
        }
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct BatchToSpace {
    datum_type: DatumType,
}

tract_data::impl_dyn_hash!(BatchToSpace);

impl Op for BatchToSpace {
    fn name(&self) -> Cow<str> {
        "BatchToSpace".into()
    }

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for BatchToSpace {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, block_shape, crops) = args_3!(inputs);
        let block_shape = block_shape.cast_to::<i32>()?;
        let block_shape = block_shape.to_array_view::<i32>()?.into_dimensionality()?;
        let crops = crops.cast_to::<i32>()?;
        let crops = crops.to_array_view::<i32>()?.into_dimensionality()?;
        let r = dispatch_numbers!(super::batch_to_space(input.datum_type())(
            input,
            &block_shape.view(),
            &crops.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl InferenceRulesOp for BatchToSpace {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        rules(s, self.datum_type, &inputs[0], &outputs[0], &inputs[1], &inputs[2])
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(block_shape), Some(paddings)) = (
            target.outlet_fact(mapping[&node.inputs[1]])?.konst.clone(),
            target.outlet_fact(mapping[&node.inputs[2]])?.konst.clone(),
        ) {
            let paddings = paddings.cast_to::<TDim>()?;
            let paddings = paddings.to_array_view::<TDim>()?.into_dimensionality::<Ix2>()?;
            let paddings = paddings
                .outer_iter()
                .map(|p| {
                    Ok(match (p[0].to_usize(), p[1].to_usize()) {
                        (Ok(bef), Ok(aft)) => {
                            super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize)
                        }
                        (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                        (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                        _ => bail!("Failed to unarize SpaceToBatch because of padding"),
                    })
                })
                .collect::<TractResult<_>>()?;
            let op = super::unary::BatchToSpaceUnary::new(
                self.datum_type,
                target.outlet_fact(mapping[&node.inputs[0]])?.shape.to_tvec(),
                node.outputs[0]
                    .fact
                    .shape
                    .concretize()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect::<TVec<_>>(),
                block_shape.clone().into_tensor().into_array::<i32>()?.into_dimensionality()?,
                paddings,
            );
            target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref())
        } else {
            bail!("Need fixed block shape and padding")
        }
    }
    as_op!();
}

fn rules<'r, 'p: 'r>(
    s: &mut Solver<'r>,
    datum_type: DatumType,
    batch: &'p TensorProxy,
    space: &'p TensorProxy,
    block_shape: &'p TensorProxy,
    paddings: &'p TensorProxy,
) -> InferenceResult {
    s.equals(&batch.datum_type, datum_type)?;
    s.equals(&batch.datum_type, &space.datum_type)?;
    s.equals(&block_shape.datum_type, DatumType::I32)?;
    s.equals(&batch.rank, &space.rank)?;
    s.equals(&block_shape.rank, 1)?;
    s.equals(&paddings.rank, 2)?;
    s.equals(&block_shape.shape[0], &paddings.shape[0])?;
    s.given(&block_shape.value, move |s, block_shape| {
        let block_shape = block_shape.into_tensor().into_array::<i32>()?;
        let block_shape_prod = block_shape.iter().map(|s| *s as usize).product::<usize>();
        s.equals(&batch.shape[0], (block_shape_prod as i64) * space.shape[0].bex())?;
        s.given(&paddings.value, move |s, paddings| {
            let paddings = paddings.cast_to::<TDim>()?;
            let paddings = paddings.to_array_view::<TDim>()?.into_dimensionality()?;
            for d in 0..block_shape.len() {
                s.equals(
                    space.shape[1 + d].bex() + &paddings[(d, 0)] + &paddings[(d, 1)],
                    (block_shape[d] as i64) * batch.shape[1 + d].bex(),
                )?;
            }
            Ok(())
        })
    })?;
    s.given(&block_shape.value, move |s, block_shape| {
        let block_shape = block_shape.into_tensor().into_array::<i32>()?;
        s.given(&space.rank, move |s, rank: i64| {
            for d in block_shape.len() + 1..(rank as usize) {
                s.equals(&space.shape[d], &batch.shape[d])?
            }
            Ok(())
        })
    })
}
