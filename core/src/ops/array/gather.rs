use crate::internal::*;
use crate::ops::einsum::block_quant_aware_input_shape;
use crate::ops::matmul::pack::OptSimpleMatMulPack;
use ndarray::*;
use tract_linalg::block_quant::BlockQuantValue;
use tract_linalg::mmm::MMMInputValue;

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Gather {
    pub axis: usize,
    pub output_type: Option<DatumType>,
}

impl Op for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl Gather {
    pub fn new(axis: usize) -> Gather {
        Gather { axis, output_type: None }
    }

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

    fn eval_t<T: Datum>(&self, data: TValue, indices: &TValue) -> TractResult<Tensor> {
        let data_view = unsafe { data.to_array_view_unchecked::<T>() }; // copy only
        let indices = indices.to_array_view::<i64>()?;
        let output_shape = &*self.compute_output_shape(data.shape(), indices.shape())?;
        let mut output = unsafe { Tensor::uninitialized::<T>(output_shape)? };
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
        Ok(output)
    }

    fn eval_bq<F: Datum>(&self, data: &BlockQuantValue, indices: &TValue) -> TractResult<Tensor> {
        ensure!(self.axis == 0);
        ensure!(data.fact.shape().len() == 2);
        let data_shape = &data.fact.shape();
        let output_shape = &*self.compute_output_shape(data_shape, indices.shape())?;
        let mut output = unsafe { Tensor::uninitialized::<F>(output_shape)? };
        let indices_slice = indices.as_slice::<i64>()?;
        let vector_len = data_shape[1];

        let block_len = data.fact.format.block_len();
        let block_bytes = data.fact.format.block_bytes();
        if F::datum_type() == f16::datum_type() {
            let output_slice = output.as_slice_mut::<f16>()?;
            for (pos, ix) in indices_slice.iter().enumerate() {
                let slice = &mut output_slice[pos * vector_len..][..vector_len];
                for i in (0..vector_len).step_by(block_len) {
                    let offset = data_shape[1] * *ix as usize + i;
                    let block_id = offset / block_len;
                    data.fact.format.dequant_block_f16(
                        &data.value[block_id * block_bytes..][..block_bytes],
                        &mut slice[i..i + block_len],
                    );
                }
            }
        } else {
            let output_slice = output.as_slice_mut::<f32>()?;
            for (pos, ix) in indices_slice.iter().enumerate() {
                let slice = &mut output_slice[pos * vector_len..][..vector_len];
                for i in (0..vector_len).step_by(block_len) {
                    let offset = data_shape[1] * *ix as usize + i;
                    let block_id = offset / block_len;
                    data.fact.format.dequant_block_f32(
                        &data.value[block_id * block_bytes..][..block_bytes],
                        &mut slice[i..i + block_len],
                    );
                }
            }
        }
        Ok(output)
    }

    fn eval_input_store<F: Datum>(
        &self,
        data: &dyn MMMInputValue,
        indices: &TValue,
    ) -> TractResult<Tensor> {
        ensure!(self.axis == 0);
        let data_shape = &[data.mn(), data.k()];
        let output_shape = &*self.compute_output_shape(data_shape, indices.shape())?;
        let mut output = unsafe { Tensor::uninitialized::<F>(output_shape)? };
        let indices_slice = indices.as_slice::<i64>()?;
        let vector_len = data_shape[1];
        if F::datum_type() == f16::datum_type() {
            let output_slice = output.as_slice_mut::<f16>()?;
            for (pos, m) in indices_slice.iter().enumerate() {
                let slice = &mut output_slice[pos * vector_len..][..vector_len];
                data.extract_at_mn_f16(*m as usize, slice)?;
            }
        } else {
            let output_slice = output.as_slice_mut::<f32>()?;
            for (pos, m) in indices_slice.iter().enumerate() {
                let slice = &mut output_slice[pos * vector_len..][..vector_len];
                data.extract_at_mn_f32(*m as usize, slice)?;
            }
        }
        Ok(output)
    }
}

impl TypedOp for Gather {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if let Some(dt) = self.output_type {
            ensure!(
                inputs[0].datum_type.is_opaque() || inputs[0].datum_type == dt,
                "Inconsistent datum_type in Gather: attribute is {:?}, but inputs[0] is {:?}",
                dt,
                inputs[0].datum_type
            );
        } else {
            ensure!(!inputs[0].datum_type.is_opaque(),
                "Gather applied to compressed data requires an explicit datum_type attribute for its output");
        }
        ensure!(inputs[1].datum_type == i64::datum_type());
        if inputs[0].datum_type.is_opaque() {
            let data_shape = block_quant_aware_input_shape(inputs[0])?;
            Ok(tvec!(self
                .output_type
                .unwrap()
                .fact(&*self.compute_output_shape(&data_shape, &inputs[1].shape)?)))
        } else {
            Ok(tvec!(inputs[0]
                .datum_type
                .fact(&*self.compute_output_shape(&inputs[0].shape, &inputs[1].shape)?)))
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let (input_fact, indices_fact) = args_2!(model.node_input_facts(node.id)?);
        if let Some(indices) = indices_fact.konst.as_ref() {
            if indices.rank() == 1 && indices.len() == 1 && input_fact.datum_type.is_number() {
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
        if input_fact.konst.is_some() {
            // look for a OptSimpleMatMulPack *sibling*
            if let Some(sibling) = model
                .outlet_successors(node.inputs[0])
                .iter()
                .find(|o| o.node != node.id && model.node(o.node).op_is::<OptSimpleMatMulPack>())
            {
                let mut patch = TypedModelPatch::default();
                let mut taps = patch.taps(model, &node.inputs)?;
                taps[0] = patch.tap_model(model, sibling.node.into())?;
                let wire = patch.wire_node(&node.name, self.clone(), &taps)?[0];
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
        let result = if let Ok(opaque) = data.to_scalar::<Opaque>() {
            let dt = self.output_type.unwrap();
            if let Some(data) = opaque.downcast_ref::<BlockQuantValue>() {
                dispatch_floatlike!(Self::eval_bq(dt)(self, data, &indices))?
            } else if let Some(data) = opaque.downcast_ref::<Box<dyn MMMInputValue>>() {
                dispatch_floatlike!(Self::eval_input_store(dt)(self, &**data, &indices))?
            } else {
                bail!("Can't use Gather on {:?} input", data);
            }
        } else {
            dispatch_datum_by_size!(Self::eval_t(data.datum_type())(self, data, &indices))?
        };
        Ok(tvec!(result.into_tvalue()))
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
