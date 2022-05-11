use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Gather {
    pub axis: usize,
}
impl_dyn_hash!(Gather);

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl Gather {
    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        let mut output_shape = tvec![];
        for (idx, dim) in input_shape.iter().enumerate() {
            if idx != self.axis {
                output_shape.push(dim.clone());
            } else {
                for idx2 in indices_shape {
                    output_shape.push(idx2.clone());
                }
            }
        }
        Ok(output_shape)
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        data: Arc<Tensor>,
        indices: &Arc<Tensor>,
    ) -> TractResult<Arc<Tensor>> {
        let data_view = data.to_array_view_unchecked::<T>();
        let indices = indices.cast_to::<i64>()?;
        if indices.shape().len() == 0 {
            let mut index = *indices.to_scalar::<i64>()?;
            if index < 0 {
                index += data_view.shape()[0] as i64;
            }
            let mut tensor =
                data_view.index_axis(Axis(self.axis), index as usize).to_owned().into_tensor();
            tensor.set_datum_type(data.datum_type());
            return Ok(tensor.into_arc_tensor());
        }

        let mut output = Tensor::uninitialized_dt(
            data.datum_type(),
            &*self.compute_output_shape(data.shape(), indices.shape())?,
        )?;
        let mut view = output.to_array_view_mut_unchecked::<T>();
        for (indices_coords, indices_value) in indices.to_array_view::<i64>()?.indexed_iter() {
            let mut to_update = view.index_axis_mut(Axis(self.axis), indices_coords[0]);
            for idx in 1..indices_coords.ndim() {
                to_update = to_update.index_axis_move(Axis(0), indices_coords[idx]);
            }
            let index_value = if *indices_value >= 0 {
                *indices_value
            } else {
                indices_value + data_view.shape()[self.axis] as i64
            } as usize;
            to_update.assign(&data_view.index_axis(Axis(self.axis), index_value));
        }
        Ok(output.into_arc_tensor())
    }
}

impl TypedOp for Gather {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(
            &*self
                .compute_output_shape(&*inputs[0].shape.to_tvec(), &*inputs[1].shape.to_tvec())?
        )))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let indices_fact = model.outlet_fact(node.inputs[1])?;
        if let Some(indices) = indices_fact.konst.as_ref() {
            if indices.len() == 1 {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                let index = indices.cast_to_scalar::<i64>()?;
                let index = if index < 0 {
                    let data_fact = model.outlet_fact(node.inputs[0])?;
                    data_fact.shape[self.axis].clone() + index.to_dim()
                } else {
                    index.to_dim()
                };
                wire = patch.wire_node(
                    format!("{}.slice", node.name),
                    crate::ops::array::Slice {
                        axis: self.axis,
                        start: index.clone(),
                        end: index + 1,
                    },
                    &[wire],
                )?[0];
                wire = patch.wire_node(
                    format!("{}.rm_axis", node.name),
                    crate::ops::change_axes::AxisOp::Rm(self.axis),
                    &[wire],
                )?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

impl EvalOp for Gather {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, indices) = args_2!(inputs);
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                &self, data, &indices
            ))?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_gather_scalar_index() {
        let data = Tensor::from(arr1(&[1i64, 2, 3]));
        let gatherer = Gather::new(0);
        for idx in 2..3 {
            let index = Tensor::from(arr0(idx as i64));
            let outputs = gatherer.eval(tvec![data.clone().into(), index.into()]).unwrap();
            let output = &outputs[0];
            assert_eq!(output.shape().len(), 0);
            assert_eq!(*output.to_scalar::<i64>().unwrap(), idx + 1);
        }
    }
}
