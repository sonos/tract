use crate::internal::*;
use tract_ndarray::prelude::*;

#[derive(Debug, Clone, new, Hash)]
pub struct GatherNd {
    pub batch_dims: usize,
}



impl GatherNd {
    fn compute_shape<D: DimLike>(
        &self,
        data_shape: &[D],
        indices_shape: &[D],
    ) -> TractResult<TVec<D>> {
        let mut shape: TVec<D> = indices_shape.into();
        let n = shape.pop().unwrap().to_usize()?;
        shape.extend(data_shape[n + self.batch_dims..].iter().cloned());
        Ok(shape)
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        output: &mut Tensor,
        data: &Tensor,
        indices: &ArrayViewD<i32>,
    ) {
        let batch_dims = self.batch_dims;
        assert_eq!(output.shape()[..batch_dims], data.shape()[..batch_dims]);
        assert_eq!(output.shape()[..batch_dims], indices.shape()[..batch_dims]);
        let batch_size = data.shape().iter().take(batch_dims).product();
        let n = indices.shape()[indices.ndim() - 1];

        let remaining = indices.shape().iter().skip(batch_dims).rev().skip(1).product();
        let indices_shape_op = tvec!(batch_size, remaining, n);
        let reshaped_indices: ArrayViewD<i32> =
            indices.view().into_shape_with_order(&*indices_shape_op).unwrap();

        let mut data_shape_op: TVec<usize> =
            data.shape().iter().skip(batch_dims).copied().collect();
        data_shape_op.insert(0, batch_size);
        let reshaped_data =
            data.to_array_view_unchecked::<T>().into_shape_with_order(&*data_shape_op).unwrap();

        let mut output_shape_op: TVec<usize> =
            data.shape().iter().skip(n + batch_dims).copied().collect();
        output_shape_op.insert(0, batch_size * remaining);
        let mut output =
            output.to_array_view_mut_unchecked::<T>().into_shape_with_order(&*output_shape_op).unwrap();

        for b in 0..batch_size {
            let mut i = reshaped_data.view();
            i.index_axis_inplace(Axis(0), b);
            let mut coords = reshaped_indices.view();
            coords.index_axis_inplace(Axis(0), b);

            for ix in 0..remaining {
                let mut coords = coords.view();
                coords.index_axis_inplace(Axis(0), ix);

                let mut i = i.view();
                for x in coords {
                    i.index_axis_inplace(Axis(0), *x as usize);
                }

                let mut o = output.view_mut();
                o.index_axis_inplace(Axis(0), b * remaining + ix);
                o.assign(&i);
            }
        }
    }
}

impl Op for GatherNd {
    fn name(&self) -> Cow<str> {
        "GatherNd".into()
    }

    op_as_typed_op!();
}

impl EvalOp for GatherNd {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let shape = self.compute_shape(data.shape(), indices.shape())?;
        let indices = indices.cast_to::<i32>()?;
        let indices = indices.to_array_view::<i32>()?;
        unsafe {
            let mut output = Tensor::uninitialized_dt(data.datum_type(), &shape)?;
            dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self,
                &mut output,
                &data,
                &indices
            ));
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl TypedOp for GatherNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = self.compute_shape(&inputs[0].shape.to_tvec(), &inputs[1].shape.to_tvec())?;
        Ok(tvec!(inputs[0].datum_type.fact(&shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(indices) = &model.outlet_fact(node.inputs[1])?.konst {
            if indices.rank() == 2 && indices.shape()[0] == 1 {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                for (axis, &i) in indices.cast_to::<i32>()?.as_slice::<i32>()?.iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}-slice-axis-{}", node.name, axis),
                        crate::ops::array::Slice::new(axis, i as usize, (i + 1) as usize),
                        &[wire],
                    )?[0];
                }
                for i in (0..indices.shape()[1]).rev() {
                    wire = patch.wire_node(
                        format!("{}-remove_axis_{}", node.name, i),
                        crate::ops::change_axes::AxisOp::Rm(i),
                        &[wire],
                    )?[0];
                }
                wire = patch.wire_node(
                    format!("{}-add_axis", node.name),
                    crate::ops::change_axes::AxisOp::Add(0),
                    &[wire],
                )?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}
