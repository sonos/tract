use crate::internal::*;
use num_traits::AsPrimitive;
use std::iter::Sum;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};

#[derive(Debug, Clone, new, Hash)]
pub struct SumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
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

    op_as_typed_op!();
}

impl EvalOp for SumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape: TVec<TDim> = inputs[0].shape().iter().map(|d| d.to_dim()).collect();
        self.to_optimized(&shape)?.eval(inputs)
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
        let fact = model.outlet_fact(node.inputs[0])?;
        if let Some(pool_spec) = self.pool_spec.declutter(&fact.shape)? {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Self { pool_spec, ..self.clone() },
            )?));
        }
        Ok(None)
    }

    as_op!();
}

impl SumPool {
    fn to_optimized(&self, input_shape: &[TDim]) -> TractResult<OptSumPool> {
        Ok(OptSumPool {
            pool_spec: self.pool_spec.clone(),
            count_include_pad: self.count_include_pad,
            normalize: self.normalize,
            geometry: self.pool_spec.compute_geo(input_shape)?,
        })
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct OptSumPool {
    pub pool_spec: PoolSpec,
    pub count_include_pad: bool,
    pub normalize: bool,
    pub geometry: PoolGeometry,
}

impl Op for OptSumPool {
    fn name(&self) -> Cow<str> {
        "OptSumPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for OptSumPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let geo = self.geometry.to_concrete(input.shape())?;
        let values = if input.datum_type().is_float() {
            let mut values =
                unsafe { Tensor::uninitialized_dt(input.datum_type(), &geo.output_shape.shape)? };
            dispatch_floatlike!(Self::eval_t(input.datum_type())(
                self,
                &*input,
                values.as_ptr_mut()?,
                geo.as_ref()
            ))?;
            values
        } else {
            let mut values =
                unsafe { Tensor::uninitialized_dt(DatumType::F32, &geo.output_shape.shape)? };
            let input_f32 = input.cast_to_dt(DatumType::F32)?;
            self.eval_t::<f32>(input_f32.as_ref(), values.as_ptr_mut()?, geo.as_ref())?;
            values.cast_to_dt(input.datum_type())?.into_owned()
        };

        Ok(tvec!(values.into_tvalue()))
    }
}

impl TypedOp for OptSumPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.pool_spec.output_facts(inputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let fact = model.outlet_fact(node.inputs[0])?;
        if let Some(pool_spec) = self.pool_spec.declutter(&fact.shape)? {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                Self { pool_spec, ..self.clone() },
            )?));
        }
        Ok(None)
    }

    as_op!();
}

impl OptSumPool {
    fn eval_t<T: Copy + Datum + Sum + num_traits::Float>(
        &self,
        input: &Tensor,
        values_ptr: *mut T,
        geo: &ConcretePoolGeometry,
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
