use crate::internal::*;
use num_traits::AsPrimitive;
use std::iter::Sum;

use crate::ops::cnn::pools::{ConcreteGeometry, PoolSpec};

#[derive(Debug, Clone, new, Default, Hash)]
pub struct SumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
    pub concrete_geometry: Option<ConcreteGeometry>,
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

impl_dyn_hash!(SumPool);

impl EvalOp for SumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let geo = if let Some(geo) = &self.concrete_geometry {
            Cow::Borrowed(geo)
        } else {
            Cow::Owned(self.pool_spec.compute_geo(&input.shape())?)
        };
        let mut values =
            unsafe { Tensor::uninitialized_dt(input.datum_type(), &*geo.output_shape.shape)? };
        dispatch_floatlike!(Self::eval_t(input.datum_type())(
            self,
            &*input,
            values.as_ptr_mut()?,
            geo.as_ref()
        ))?;
        Ok(tvec!(values.into_arc_tensor()))
    }
}

impl TypedOp for SumPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let (Some(shape), None) = (inputs[0].shape.as_concrete(), &self.concrete_geometry) {
            let op = Self {
                concrete_geometry: Some(self.pool_spec.compute_geo(shape)?),
                ..self.clone()
            };
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }

    as_op!();
}

impl SumPool {
    fn eval_t<T: Copy + Datum + num_traits::Float + Sum>(
        &self,
        input: &Tensor,
        values_ptr: *mut T,
        geo: &ConcreteGeometry,
    ) -> TractResult<()>
    where
        usize: AsPrimitive<T>,
    {
        let input_ptr = input.as_ptr::<T>()?;

        let n = *geo.input_shape.n().unwrap_or(&1);
        let n_stride_i = geo.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = geo.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            geo.patch.visit_output(|visitor| {
                let div: Option<T> = if self.normalize {
                    Some(
                        if self.count_include_pad {
                            geo.patch.standard_layout_data_field.len().as_()
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
                    for c in 0..*geo.input_shape.c() {
                        let input_offset = input_offset + geo.input_shape.c_stride() * c;
                        let output_offset = output_offset + geo.output_shape.c_stride() * c;
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
