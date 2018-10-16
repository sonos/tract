use ndarray::*;
use tfdeploy::analyser::rules::prelude::*;
use tfdeploy::ops::prelude::*;

#[derive(Debug, Copy, Clone)]
pub enum PaddingStrat {
    FlexFixed(usize),
    FixedFlex(usize),
    FixedFixed(usize, usize),
}

#[derive(Debug, Clone, new)]
pub struct SpaceToBatchUnary {
    pub datum_type: DatumType,
    pub space_shape: Vec<TDim>,
    pub batch_shape: Vec<TDim>,
    pub block_shape: Array1<i32>,
    pub pad: Vec<PaddingStrat>,
}

impl Op for SpaceToBatchUnary {
    fn name(&self) -> &str {
        "SpaceToBatchUnary"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let mut paddings = unsafe { Array2::uninitialized((self.block_shape.len(), 2)) };
        for (ax, &strat) in self.pad.iter().enumerate() {
            let spread = (self.batch_shape[2 + ax] * self.block_shape[ax]
                - self.space_shape[2 + ax])
                .to_integer()? as usize;
            let (bef, aft) = match strat {
                PaddingStrat::FlexFixed(f) => (spread - f, f),
                PaddingStrat::FixedFlex(f) => (f, spread - f),
                PaddingStrat::FixedFixed(a, b) => (a, b),
            };
            paddings[(ax, 0)] = bef as i32;
            paddings[(ax, 1)] = aft as i32;
        }
        let r = dispatch_numbers!(super::space_to_batch(input.datum_type())(
            input,
            &self.block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl InferenceRulesOp for SpaceToBatchUnary {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, self.datum_type)?;
        s.equals(&outputs[0].datum_type, self.datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&outputs[0].shape, self.batch_shape.to_vec())?;
        s.equals(&inputs[0].shape, self.space_shape.to_vec())?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct BatchToSpaceUnary {
    datum_type: DatumType,
    batch_shape: Vec<TDim>,
    space_shape: Vec<TDim>,
    block_shape: Array1<i32>,
    pad: Vec<PaddingStrat>,
}

impl Op for BatchToSpaceUnary {
    fn name(&self) -> &str {
        "BatchToSpaceUnary"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        let mut paddings = unsafe { Array2::uninitialized((self.block_shape.len(), 2)) };
        for (ax, &strat) in self.pad.iter().enumerate() {
            let spread = (self.batch_shape[2 + ax] * self.block_shape[ax]
                - self.space_shape[2 + ax])
                .to_integer()? as usize;
            let (bef, aft) = match strat {
                PaddingStrat::FlexFixed(f) => (spread - f, f),
                PaddingStrat::FixedFlex(f) => (f, spread - f),
                PaddingStrat::FixedFixed(a, b) => (a, b),
            };
            paddings[(ax, 0)] = bef as i32;
            paddings[(ax, 1)] = aft as i32;
        }
        let r = dispatch_numbers!(super::batch_to_space(input.datum_type())(
            input,
            &self.block_shape.view(),
            &paddings.view()
        ))?;
        Ok(tvec!(r))
    }
}

impl InferenceRulesOp for BatchToSpaceUnary {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, self.datum_type)?;
        s.equals(&outputs[0].datum_type, self.datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].shape, self.batch_shape.to_vec())?;
        s.equals(&outputs[0].shape, self.space_shape.to_vec())?;
        Ok(())
    }
}
