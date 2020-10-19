use crate::internal::*;
use ndarray::prelude::*;
use num_traits::Float;

use crate::ops::cnn::pools::PoolSpec;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct MaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
}

tract_data::impl_dyn_hash!(MaxPool);

impl MaxPool {
    fn to_fixed<T: Datum + Float>(&self, input_shape: &[usize]) -> TractResult<Box<dyn TypedOp>> {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(input_shape)?;
        let op = MaxPoolFixed::new(patch, input_shape, output_shape, self.with_index_outputs);
        Ok(Box::new(op))
    }
}

impl Op for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let op = dispatch_floatlike!(MaxPool::to_fixed(inputs[0].datum_type())(
            self,
            inputs[0].shape()
        ))?;
        op.eval(inputs)
    }
}

impl TypedOp for MaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.with_index_outputs.is_some()
            && node.outputs[1].successors.len() == 0
            && !model.output_outlets()?.contains(&OutletId::new(node.id, 1))
        {
            let op = MaxPool { pool_spec: self.pool_spec.clone(), with_index_outputs: None };
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, node.inputs[0])?;
            wire = patch.wire_node(&node.name, op, &[wire])?[0];
            patch.shunt_outside(model, node.id.into(), wire)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(shape) = inputs[0].shape.as_finite() {
            let dt = inputs[0].datum_type;
            let op = dispatch_floatlike!(MaxPool::to_fixed(dt)(self, &*shape))?;
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }

    as_op!();
}

tract_data::impl_dyn_hash!(MaxPoolFixed);

#[derive(Debug, Clone, new, Hash)]
pub struct MaxPoolFixed {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    with_index_outputs: Option<DatumType>,
}

impl Op for MaxPoolFixed {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl MaxPoolFixed {
    fn eval_t<T: Datum + Copy + num_traits::Bounded + PartialOrd>(
        &self,
        input: &Tensor,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let input: ArrayViewD<T> = input.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninitialized(&*self.output_shape.shape) })
        } else {
            None
        };
        let n = *self.input_shape.n().unwrap_or(&1);
        let n_stride_i = self.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = self.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            self.patch.visit_output(|visitor| {
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*self.input_shape.c() {
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

impl EvalOp for MaxPoolFixed {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, &*input))
    }
}

impl TypedOp for MaxPoolFixed {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts =
            tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape.shape)?);
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }
}
