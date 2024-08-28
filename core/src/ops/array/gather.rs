use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Gather {
    pub axis: usize,
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    op_as_typed_op!();
}

impl Gather {
    pub fn compute_output_shape<D: DimLike>(
        &self,
        input_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        ensure!(input_shape.len() > self.axis);
        let mut output_shape: TVec<D> = input_shape[..self.axis].into();
        output_shape.extend(indices_shape.iter().cloned());
        output_shape.extend(input_shape[self.axis + 1..].iter().cloned());
        Ok(output_shape)
    }

    unsafe fn eval_t<T: Datum>(&self, data: TValue, indices: &TValue) -> TractResult<TValue> {
        let data_view = data.to_array_view_unchecked::<T>();
        let indices = indices.to_array_view::<i64>()?;
        let output_shape = &*self.compute_output_shape(data.shape(), indices.shape())?;
        let mut output = Tensor::uninitialized::<T>(output_shape)?;
        let mut output_view = output.to_array_view_mut::<T>()?;
        for coords in tract_ndarray::indices(output_shape) {
            let ocoords = coords.as_array_view();
            let ocoords = ocoords.as_slice().unwrap();
            let mut icoords: TVec<usize> = ocoords[0..self.axis].into();
            let kcoords = &ocoords[self.axis..][..indices.ndim()];
            let k = indices[kcoords];
            let k = if k < 0 { k + data_view.shape()[self.axis] as i64 } else { k } as usize;
            icoords.push(k);
            icoords.extend(ocoords[self.axis + indices.ndim()..].iter().copied());
            output_view[ocoords] = data_view.get(&*icoords).context("Invalid gather")?.clone();
        }
        unsafe { output.set_datum_type(data.datum_type()) };
        Ok(output.into_tvalue())
    }
}

impl TypedOp for Gather {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs[1].datum_type == i64::datum_type());
        Ok(tvec!(inputs[0].datum_type.fact(
            &*self.compute_output_shape(&inputs[0].shape.to_tvec(), &inputs[1].shape.to_tvec())?
        )))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let indices_fact = model.outlet_fact(node.inputs[1])?;
        if let Some(indices) = indices_fact.konst.as_ref() {
            if indices.rank() == 1 && indices.len() == 1 {
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

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices
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
            let index = Tensor::from(arr0(idx));
            let outputs =
                gatherer.eval(tvec![data.clone().into_tvalue(), index.into_tvalue()]).unwrap();
            let output = &outputs[0];
            assert_eq!(output.shape().len(), 0);
            assert_eq!(*output.to_scalar::<i64>().unwrap(), idx + 1);
        }
    }
}
