use ndarray::prelude::*;
use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct SpaceToBatch {
    datum_type: DatumType,
}

impl Op for SpaceToBatch {
    fn name(&self) -> &str {
        "SpaceToBatch"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let block_shape = block_shape
            .cast_to_array()?
            .into_owned()
            .into_dimensionality()?;
        let paddings = paddings
            .cast_to_array()?
            .into_owned()
            .into_dimensionality()?;
        let r = dispatch_numbers!(super::space_to_batch(input.datum_type())(
            input,
            &block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }

    fn reduce(
        &self,
        mut inputs: TVec<&TensorFact>,
        mut outputs: TVec<&TensorFact>,
    ) -> TfdResult<Option<ReducedOpRewire>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let output = args_1!(outputs);
        if let (Some(input_shape), Some(block_shape), Some(paddings), Some(output_shape)) = (
            input.shape.concretize(),
            block_shape.value.concretize(),
            paddings.value.concretize(),
            output.shape.concretize(),
        ) {
            let paddings = paddings.cast_to_array::<TDim>()?.into_owned().into_dimensionality::<Ix2>()?;
            let paddings = paddings.outer_iter().map(|p| {
                Ok(match (p[0].to_integer(), p[1].to_integer()) {
                    (Ok(bef), Ok(aft)) => super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize),
                    (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                    (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                    _ => {
                        bail!("Failed to unarize SpaceToBatch because of padding")
                    }
                })
            }).collect::<TfdResult<_>>()?;
            let op = super::unary::SpaceToBatchUnary::new(
                self.datum_type,
                input_shape,
                output_shape,
                block_shape.into_array::<i32>()?.into_dimensionality()?,
                paddings
            );
            Ok(Some(ReducedOpRewire::new(Box::new(op), tvec!(0))))
        } else {
            Ok(None)
        }
    }
}

impl InferenceRulesOp for SpaceToBatch {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
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
    fn name(&self) -> &str {
        "BatchToSpace"
    }
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (input, block_shape, crops) = args_3!(inputs);
        let block_shape = block_shape
            .cast_to_array()?
            .into_owned()
            .into_dimensionality()?;
        let crops = crops.cast_to_array()?.into_owned().into_dimensionality()?;
        let r = dispatch_numbers!(super::batch_to_space(input.datum_type())(
            input,
            &block_shape.view(),
            &crops.view()
        ))?;
        Ok(tvec!(r))
    }

    fn reduce(
        &self,
        mut inputs: TVec<&TensorFact>,
        mut outputs: TVec<&TensorFact>,
    ) -> TfdResult<Option<ReducedOpRewire>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let output = args_1!(outputs);
        if let (Some(input_shape), Some(block_shape), Some(paddings), Some(output_shape)) = (
            input.shape.concretize(),
            block_shape.value.concretize(),
            paddings.value.concretize(),
            output.shape.concretize(),
        ) {
            let paddings = paddings.cast_to_array::<TDim>()?.into_owned().into_dimensionality::<Ix2>()?;
            let paddings = paddings.outer_iter().map(|p| {
                Ok(match (p[0].to_integer(), p[1].to_integer()) {
                    (Ok(bef), Ok(aft)) => super::unary::PaddingStrat::FixedFixed(bef as usize, aft as usize),
                    (_, Ok(aft)) => super::unary::PaddingStrat::FlexFixed(aft as usize),
                    (Ok(bef), _) => super::unary::PaddingStrat::FixedFlex(bef as usize),
                    _ => {
                        bail!("Failed to unarize SpaceToBatch because of padding")
                    }
                })
            }).collect::<TfdResult<_>>()?;
            let op = super::unary::BatchToSpaceUnary::new(
                self.datum_type,
                input_shape,
                output_shape,
                block_shape.into_array::<i32>()?.into_dimensionality()?,
                paddings
            );
            Ok(Some(ReducedOpRewire::new(Box::new(op), tvec!(0))))
        } else {
            Ok(None)
        }
    }
}

impl InferenceRulesOp for BatchToSpace {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
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
    s.given(&block_shape.value, move |s, block_shape: Tensor| {
        let block_shape: ArrayD<i32> = block_shape.take_i32s().unwrap();
        let block_shape_prod = block_shape.iter().map(|s| *s as usize).product::<usize>();
        s.equals(
            &batch.shape[0],
            (block_shape_prod as i64) * space.shape[0].bex(),
        )?;
        s.given(&paddings.value, move |s, paddings: Tensor| {
            let paddings = TDim::tensor_cast_to_array(&paddings).unwrap(); // FIXMEa
            let paddings = paddings.view().into_dimensionality().unwrap();
            for d in 0..block_shape.len() {
                s.equals(
                    space.shape[1 + d].bex() + paddings[(d, 0)] + paddings[(d, 1)],
                    (block_shape[d] as i64) * batch.shape[1 + d].bex(),
                )?;
            }
            Ok(())
        })
    })?;
    s.given(&block_shape.value, move |s, block_shape: Tensor| {
        let block_shape: ArrayD<i32> = block_shape.take_i32s().unwrap();
        s.given(&space.rank, move |s, rank: i64| {
            for d in block_shape.len() + 1..(rank as usize) {
                s.equals(&space.shape[d], &batch.shape[d])?
            }
            Ok(())
        })
    })
}
