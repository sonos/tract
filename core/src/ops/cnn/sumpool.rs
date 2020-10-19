use crate::internal::*;
use num_traits::AsPrimitive;
use std::iter::Sum;

use crate::ops::cnn::pools::PoolSpec;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct SumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
}

impl SumPool {
    fn to_fixed(
        &self,
        datum_type: DatumType,
        input_shape: &[usize],
    ) -> TractResult<Box<dyn TypedOp>> {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_shape)?;
        let op = SumPoolFixed::new(
            patch,
            input_shape,
            output_shape,
            datum_type,
            self.count_include_pad,
            self.normalize,
        );
        Ok(Box::new(op))
    }
}

impl Op for SumPool {
    fn name(&self) -> Cow<str> {
        "SumPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_core_mir!();
    op_as_typed_op!();
}

tract_data::impl_dyn_hash!(SumPool);

impl EvalOp for SumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.to_fixed(inputs[0].datum_type(), inputs[0].shape())?.eval(inputs)
    }
}

impl TypedOp for SumPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(shape) = inputs[0].shape.as_finite() {
            let op = self.to_fixed(inputs[0].datum_type, &*shape)?;
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }

    as_op!();
}

#[derive(Debug, Clone, new, Hash)]
pub struct SumPoolFixed {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    datum_type: DatumType,
    count_include_pad: bool,
    normalize: bool,
}

tract_data::impl_dyn_hash!(SumPoolFixed);

impl SumPoolFixed {
    fn eval_t<T: Copy + Datum + num_traits::Float + Sum>(
        &self,
        input: &Tensor,
        values_ptr: *mut T,
    ) -> TractResult<()>
    where
        usize: AsPrimitive<T>,
    {
        let input_ptr = input.as_ptr::<T>()?;

        let n = *self.input_shape.n().unwrap_or(&1);
        let n_stride_i = self.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = self.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            self.patch.visit_output(|visitor| {
                let div: Option<T> = if self.normalize {
                    Some(
                        if self.count_include_pad {
                            self.patch.standard_layout_data_field.len().as_()
                        } else {
                            visitor.valid_count().as_()
                        }
                        .recip(),
                    )
                } else {
                    None
                };
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*self.input_shape.c() {
                        let input_offset = input_offset + self.input_shape.c_stride() * c;
                        let output_offset = output_offset + self.output_shape.c_stride() * c;
                        let sum = visitor
                            .valid_offsets()
                            .map(|v| *input_ptr.offset(v + input_offset as isize))
                            .sum::<T>();

                        if let Some(div) = div {
                            *values_ptr.offset(output_offset as isize + visitor.output_offset) =
                                sum * div;
                        }
                    }
                }
            });
        }
        Ok(())
    }
}

impl Op for SumPoolFixed {
    fn name(&self) -> Cow<str> {
        "SumPool".into()
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl EvalOp for SumPoolFixed {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut values =
            unsafe { Tensor::uninitialized_dt(self.datum_type, &*self.output_shape.shape)? };
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, &*input, values.as_ptr_mut()?))?;
        Ok(tvec!(values.into_arc_tensor()))
    }
}

impl TypedOp for SumPoolFixed {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape.shape)?))
    }

    as_op!();
}
