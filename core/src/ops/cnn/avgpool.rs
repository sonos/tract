use crate::internal::*;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Float};
use std::iter::Sum;

use crate::ops::cnn::pools::PoolSpec;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

// TODO check why AvgPool need to be typed

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    pool_spec: PoolSpec,
    count_include_pad: bool,
}

impl AvgPool {
    fn to_fixed<T: Datum + Float + Sum>(
        &self,
        input_shape: &[usize],
    ) -> TractResult<Box<dyn TypedOp>>
    where
        usize: AsPrimitive<T>,
    {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_shape);
        let op = AvgPoolFixed::<T>::new(patch, input_shape, output_shape, self.count_include_pad);
        Ok(Box::new(op))
    }
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AvgPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for AvgPool {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let op = dispatch_floatlike!(AvgPool::to_fixed(inputs[0].datum_type())(
            self,
            inputs[0].shape()
        ))?;
        op.as_stateless().unwrap().eval(inputs)
    }
}

impl InferenceRulesOp for AvgPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        self.pool_spec.rules_for_shape(s, inputs, outputs)
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for AvgPool {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        self.pool_spec.output_facts(inputs)
    }

    fn pulsify(
        &self,
        source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        self.pool_spec.pulsify(source, node, target, mapping)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(shape) = inputs[0].shape.as_finite() {
            let dt = inputs[0].datum_type;
            let op = dispatch_floatlike!(AvgPool::to_fixed(dt)(self, shape))?;
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }
}

#[derive(Debug, Clone, new)]
pub struct AvgPoolFixed<T: Datum + Float + Sum>
where
    usize: AsPrimitive<T>,
{
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    count_include_pad: bool,
    _casper: PhantomData<T>,
}

impl<T: Datum + Float + Sum> Op for AvgPoolFixed<T>
where
    usize: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        format!("AvgPool::Fixed<{:?}>", T::datum_type()).into()
    }

    op_as_typed_op!();
}

impl<T: Datum + Float + Sum> StatelessOp for AvgPoolFixed<T>
where
    usize: AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<T> = input.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        unsafe {
            self.patch.visit_output(|visitor| {
                let div = if self.count_include_pad {
                    self.patch.standard_layout_data_field.len()
                } else {
                    visitor.valid_count()
                };
                let div = div.as_().recip();
                for n in 0..*self.input_shape.n() {
                    let input_offset = self.input_shape.n_stride() * n;
                    let output_offset = self.output_shape.n_stride() * n;
                    for c in 0..*self.input_shape.c() {
                        let input_offset = input_offset + self.input_shape.c_stride() * c;
                        let output_offset = output_offset + self.output_shape.c_stride() * c;
                        let sum = visitor
                            .valid_offsets()
                            .map(|v| *input_ptr.offset(v + input_offset as isize))
                            .sum::<T>();

                        *values
                            .as_mut_ptr()
                            .offset(output_offset as isize + visitor.output_offset) = sum * div;
                    }
                }
            });
        }
        Ok(tvec!(values.into_arc_tensor()))
    }
}

impl<T: Datum + Float + Sum> TypedOp for AvgPoolFixed<T>
where
    usize: AsPrimitive<T>,
{
    typed_op_as_op!();

    fn output_facts(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(T::datum_type(), &*self.output_shape.shape)?))
    }
}
