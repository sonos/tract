use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct LayerHardmax {
    axis: usize,
    //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl LayerHardmax {
    fn eval_t<D: Datum + ::num::Float + ::num::FromPrimitive>(&self, input: Value) -> TfdResult<TVec<Value>> {
        let array = input.into_array::<D>()?;
        let shape = array.shape().to_vec();
        let first_dim: usize = array.shape()[0..self.axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            let max = layer
                .iter()
                .enumerate()
                .rev()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(b.0.cmp(&a.0)))
                .map(|(ix, _)| ix)
                .unwrap_or(0);
            layer
                .iter_mut()
                .enumerate()
                .for_each(|(ix, r)| *r = D::from_usize((ix == max) as usize).unwrap());
        });
        Ok(tvec!(array.into_shape(shape)?.into()))
    }
}

impl Op for LayerHardmax {
    fn name(&self) -> &str {
        "LayerHardmax"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerHardmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct LayerLogSoftmax {
    axis: usize,
    //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl LayerLogSoftmax {
    fn eval_t<D: Datum + ::num::Float + ::num::FromPrimitive + ::std::iter::Sum>(
        &self,
        input: Value,
    ) -> TfdResult<TVec<Value>> {
        let array = input.into_array::<D>()?;
        let shape = array.shape().to_vec();
        let first_dim: usize = array.shape()[0..self.axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            // https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
            let max:Option<D> = layer.iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
                .cloned();
            layer.mapv_inplace(|x| (x-max.unwrap()).exp());
            let divisor = layer.iter().cloned().sum();
            layer.mapv_inplace(|x| (x / divisor).ln());
        });
        Ok(tvec!(array.into_shape(shape)?.into()))
    }
}

impl Op for LayerLogSoftmax {
    fn name(&self) -> &str {
        "LayerLogSoftmax"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerLogSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct LayerSoftmax {
    axis: usize,
    //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl LayerSoftmax {
    fn eval_t<D: Datum + ::num::Float + ::num::FromPrimitive + ::std::iter::Sum>(
        &self,
        input: Value,
    ) -> TfdResult<TVec<Value>> {
        let array = input.into_array::<D>()?;
        let shape = array.shape().to_vec();
        let first_dim: usize = array.shape()[0..self.axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            // https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
            let max:Option<D> = layer.iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
                .cloned();
            layer.mapv_inplace(|x| (x-max.unwrap()).exp());
            let divisor = layer.iter().cloned().sum();
            layer.mapv_inplace(|x| x / divisor);
        });
        Ok(tvec!(array.into_shape(shape)?.into()))
    }
}

impl Op for LayerSoftmax {
    fn name(&self) -> &str {
        "LayerSoftmax"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p TensorsProxy,
    outputs: &'p TensorsProxy,
) -> InferenceResult {
    s.equals(&outputs.len, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape, &inputs[0].shape)?;
    Ok(())
}
