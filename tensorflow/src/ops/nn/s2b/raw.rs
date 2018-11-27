use ndarray::prelude::*;
use tract_core::ops::prelude::*;

use tract_core::analyser::rules::SharedTensorProxy;

#[derive(Debug, Clone, new)]
pub struct SpaceToBatch {
    datum_type: DatumType,
}

impl Op for SpaceToBatch {
    fn name(&self) -> Cow<str> {
        "SpaceToBatch".into()
    }

    fn reduce(
        &self,
        mut inputs: TVec<&TensorFact>,
        mut outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase == ReductionPhase::Normalize {
            let (input, block_shape, paddings) = args_3!(inputs);
            let output = args_1!(outputs);
            if let (Some(input_shape), Some(block_shape), Some(paddings), Some(output_shape)) = (
                input.shape.concretize(),
                block_shape.value.concretize(),
                paddings.value.concretize(),
                output.shape.concretize(),
            ) {
                let paddings = paddings.cast_to::<TDim>()?;
                let paddings_view = paddings.to_array_view::<TDim>()?.into_dimensionality::<Ix2>()?;
                let mut paddings = tvec![];
                for p in paddings_view.outer_iter() {
                    let pad = match (p[0].to_integer(), p[1].to_integer()) {
                        (Ok(bef), Ok(aft)) => {
                            super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize)
                        }
                        (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                        (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                        _ => {
                            info!("Failed to unarize SpaceToBatch because of padding");
                            return Ok(None);
                        }
                    };
                    paddings.push(pad);
                }
                let op = super::unary::SpaceToBatchUnary::new(
                    self.datum_type,
                    input_shape,
                    output_shape,
                    block_shape.to_array::<i32>()?.into_dimensionality()?,
                    paddings,
                );
                return Ok(Some(ReducedOpRewire::new(Box::new(op), tvec!(0))))
            }
        }
        Ok(None)
    }
}

impl StatelessOp for SpaceToBatch {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 3)?;
        s.equals(&outputs.len, 1)?;
        rules(
            s,
            self.datum_type,
            &outputs[0],
            &inputs[0],
            &inputs[1],
            &inputs[2],
        )
    }
}

#[derive(Debug, Clone, new)]
pub struct BatchToSpace {
    datum_type: DatumType,
}

impl Op for BatchToSpace {
    fn name(&self) -> Cow<str> {
        "BatchToSpace".into()
    }

    fn reduce(
        &self,
        mut inputs: TVec<&TensorFact>,
        mut outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase == ReductionPhase::Normalize {
            let (input, block_shape, paddings) = args_3!(inputs);
            let output = args_1!(outputs);
            if let (Some(input_shape), Some(block_shape), Some(paddings), Some(output_shape)) = (
                input.shape.concretize(),
                block_shape.value.concretize(),
                paddings.value.concretize(),
                output.shape.concretize(),
            ) {
                let paddings = paddings.cast_to::<TDim>()?;
                let paddings = paddings.to_array_view::<TDim>()?.into_dimensionality::<Ix2>()?;
                let paddings = paddings
                    .outer_iter()
                    .map(|p| {
                        Ok(match (p[0].to_integer(), p[1].to_integer()) {
                            (Ok(bef), Ok(aft)) => {
                                super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize)
                            }
                            (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                            (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                            _ => bail!("Failed to unarize SpaceToBatch because of padding"),
                        })
                    }).collect::<TractResult<_>>()?;
                let op = super::unary::BatchToSpaceUnary::new(
                    self.datum_type,
                    input_shape,
                    output_shape,
                    block_shape.to_array::<i32>()?.into_dimensionality()?,
                    paddings,
                );
                return Ok(Some(ReducedOpRewire::new(Box::new(op), tvec!(0))))
            }
        }
        Ok(None)
    }
}

impl StatelessOp for BatchToSpace {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 3)?;
        s.equals(&outputs.len, 1)?;
        rules(
            s,
            self.datum_type,
            &inputs[0],
            &outputs[0],
            &inputs[1],
            &inputs[2],
        )
    }
}

fn rules<'r, 'p: 'r>(
    s: &mut Solver<'r>,
    datum_type: DatumType,
    batch: &'p SharedTensorProxy,
    space: &'p SharedTensorProxy,
    block_shape: &'p SharedTensorProxy,
    paddings: &'p SharedTensorProxy,
) -> InferenceResult {
    s.equals(&batch.datum_type, datum_type)?;
    s.equals(&batch.datum_type, &space.datum_type)?;
    s.equals(&block_shape.datum_type, DatumType::I32)?;
    s.equals(&batch.rank, &space.rank)?;
    s.equals(&block_shape.rank, 1)?;
    s.equals(&paddings.rank, 2)?;
    s.equals(&block_shape.shape[0], &paddings.shape[0])?;
    s.given(&block_shape.value, move |s, block_shape| {
        let block_shape = block_shape.to_array::<i32>()?;
        let block_shape_prod = block_shape.iter().map(|s| *s as usize).product::<usize>();
        s.equals(
            &batch.shape[0],
            (block_shape_prod as i32) * space.shape[0].bex(),
        )?;
        s.given(&paddings.value, move |s, paddings| {
            let paddings = paddings.cast_to::<TDim>()?;
            let paddings = paddings.to_array_view::<TDim>()?.into_dimensionality()?;
            for d in 0..block_shape.len() {
                s.equals(
                    space.shape[1 + d].bex() + paddings[(d, 0)] + paddings[(d, 1)],
                    (block_shape[d] as i32) * batch.shape[1 + d].bex(),
                )?;
            }
            Ok(())
        })
    })?;
    s.given(&block_shape.value, move |s, block_shape| {
        let block_shape = block_shape.to_array::<i32>()?;
        s.given(&space.rank, move |s, rank: i32| {
            for d in block_shape.len() + 1..(rank as usize) {
                s.equals(&space.shape[d], &batch.shape[d])?
            }
            Ok(())
        })
    })
}
