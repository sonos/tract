use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry, PoolSpec};

#[derive(Debug, Clone, new, Hash)]
pub struct MaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
}

impl Op for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for MaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape: TVec<TDim> = inputs[0].shape().iter().map(|d| d.to_dim()).collect();
        self.to_optimized(&shape)?.eval(inputs)
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
            let op = Self { with_index_outputs: None, ..self.clone() };
            let mut patch = TypedModelPatch::default();
            let mut wire = patch.tap_model(model, node.inputs[0])?;
            wire = patch.wire_node(&node.name, op, &[wire])?[0];
            patch.shunt_outside(model, node.id.into(), wire)?;
            return Ok(Some(patch));
        }
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

impl MaxPool {
    fn to_optimized(&self, input_shape: &[TDim]) -> TractResult<OptMaxPool> {
        Ok(OptMaxPool {
            pool_spec: self.pool_spec.clone(),
            with_index_outputs: self.with_index_outputs,
            geometry: self.pool_spec.compute_geo(input_shape)?,
        })
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct OptMaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Option<DatumType>,
    pub geometry: PoolGeometry,
}

impl Op for OptMaxPool {
    fn name(&self) -> Cow<str> {
        "OptMaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalOp for OptMaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let geo = self.geometry.to_concrete(input.shape())?;
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, &*input, geo.as_ref()))
    }
}

impl TypedOp for OptMaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    as_op!();
}

impl OptMaxPool {
    fn eval_t<T: Datum + Copy + num_traits::Bounded + PartialOrd>(
        &self,
        input: &Tensor,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<TVec<TValue>> {
        let input_dt = input.datum_type();
        let input: ArrayViewD<T> = input.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninit(&*geo.output_shape.shape).assume_init() };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninit(&*geo.output_shape.shape).assume_init() })
        } else {
            None
        };
        let n = *geo.input_shape.n().unwrap_or(&1);
        let n_stride_i = geo.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = geo.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            geo.patch.visit_output(|visitor| {
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*geo.input_shape.c() {
                        let input_offset = input_offset + geo.input_shape.c_stride() * c;
                        let output_offset = output_offset + geo.output_shape.c_stride() * c;
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
                                max.0 as i32 / geo.patch.spec.output_inner_stride as i32;
                        }
                    }
                }
            });
        }
        let mut values = values.into_tensor();
        unsafe {
            values.set_datum_type(input_dt);
        }
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into_tvalue(),
                indices.unwrap().into_tensor().cast_to_dt(dt)?.into_owned().into_tvalue()
            ))
        } else {
            Ok(tvec!(values.into_tvalue()))
        }
    }
}
