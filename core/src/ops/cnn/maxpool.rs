use crate::internal::*;
use ndarray::prelude::*;
use num_traits::Float;

use crate::ops::cnn::pools::PoolSpec;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    pool_spec: PoolSpec,
    with_index_outputs: Option<DatumType>,
}

impl MaxPool {
    fn to_fixed<T: Datum + Float>(&self, input_shape: &[usize]) -> TractResult<Box<Op>> {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_shape);
        let op = MaxPoolFixed::<T>::new(patch, input_shape, output_shape, self.with_index_outputs);
        Ok(Box::new(op))
    }
}

impl Op for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(shape) = inputs[0].shape.as_finite() {
            let dt = inputs[0].datum_type;
            let op = dispatch_floatlike!(MaxPool::to_fixed(dt)(self, shape))?;
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }
}

impl StatelessOp for MaxPool {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let op = dispatch_floatlike!(MaxPool::to_fixed(inputs[0].datum_type())(
            self,
            inputs[0].shape()
        ))?;
        op.as_stateless().unwrap().eval(inputs)
    }
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1 + self.with_index_outputs.is_some() as usize)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if let Some(idt) = self.with_index_outputs {
            s.equals(&outputs[1].datum_type, idt)?;
            s.equals(&outputs[1].shape, &outputs[0].shape)?;
        }
        self.pool_spec.rules_for_shape(s, inputs, outputs)
    }
}

#[derive(Debug, Clone, new)]
pub struct MaxPoolFixed<T: Datum + Float> {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    with_index_outputs: Option<DatumType>,
    _casper: PhantomData<T>,
}

impl<T: Datum + Float> Op for MaxPoolFixed<T> {
    fn name(&self) -> Cow<str> {
        format!("MaxPool::Fixed<{:?}>", T::datum_type()).into()
    }
}

impl<T: Datum + Float> StatelessOp for MaxPoolFixed<T> {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<T> = input.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninitialized(&*self.output_shape.shape) })
        } else {
            None
        };
        unsafe {
            self.patch.visit_output(|visitor| {
                for n in 0..self.input_shape.n() {
                    let input_offset = self.input_shape.n_stride() * n;
                    let output_offset = self.output_shape.n_stride() * n;
                    for c in 0..self.input_shape.c() {
                        let input_offset = input_offset + self.input_shape.c_stride() * c;
                        let output_offset = output_offset + self.output_shape.c_stride() * c;
                        let max = visitor
                            .valid_offsets()
                            .map(|v| (v, *input_ptr.offset(v + input_offset as isize)))
                            .fold((0, T::min_value()), |acc, v| if acc.1 < v.1 { v } else { acc });
                        *values
                            .as_mut_ptr()
                            .offset(output_offset as isize + visitor.output_offset) = max.1;
                        if let Some(ref mut indices) = indices {
                            *indices
                                .as_mut_ptr()
                                .offset(output_offset as isize + visitor.output_offset) =
                                max.0 as i32 / self.patch.spec.output_inner_stride as i32;
                        }
                    }
                }
            });
        }
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into_arc_tensor(),
                indices.unwrap().into_tensor().cast_to_dt(dt)?.into_owned().into_arc_tensor()
            ))
        } else {
            Ok(tvec!(values.into_arc_tensor()))
        }
    }
}

impl<T: Datum + Float> InferenceRulesOp for MaxPoolFixed<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!()
    }
}
