use crate::internal::*;
use ndarray::*;

/// For every coordinate of `indices`, reads `data` at that same coordinate with
/// `axis` replaced by the index value found there. The output has `indices`'
/// shape. A negative index counts from the end of `axis`. `data` and `indices`
/// must have the same rank; off `axis`, `indices` may be smaller than `data`.
#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct GatherElements {
    pub axis: usize,
}

impl Op for GatherElements {
    fn name(&self) -> StaticName {
        "GatherElements".into()
    }

    op_as_typed_op!();
}

impl GatherElements {
    /// Gathers when `axis` is the last one and the leading dimensions of both
    /// operands are identical. Every leading coordinate then addresses one whole
    /// row of both operands, so the op is a flat per-row lookup and none of the
    /// per-element dynamic-rank stride arithmetic of the generic path is needed.
    /// `Ok(None)` means the operands do not qualify and the caller must use the
    /// generic path.
    ///
    /// Index resolution is identical to the generic path, but an out-of-range
    /// index is reported as an error instead of panicking inside `ndarray`.
    fn eval_contiguous_last_axis<T: Datum>(
        &self,
        data: &ArrayViewD<T>,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<Option<ArrayD<T>>> {
        let rank = data.ndim();
        let Some(last_axis) = rank.checked_sub(1) else { return Ok(None) };
        if self.axis != last_axis
            || indices.ndim() != rank
            || data.shape()[..last_axis] != indices.shape()[..last_axis]
        {
            return Ok(None);
        }
        // Both views come from a plain tract tensor, so they are contiguous and
        // this never declines; it is handled rather than asserted.
        let (Some(data_slice), Some(index_slice)) = (data.as_slice(), indices.as_slice()) else {
            return Ok(None);
        };
        let row_len = data.shape()[last_axis];
        let gathered_len = indices.shape()[last_axis];
        // A zero gathered length still has leading coordinates to walk, and
        // walking them would be pure waste for an empty output.
        let rows = if indices.is_empty() { 0 } else { indices.len() / gathered_len };
        let mut output = Vec::with_capacity(indices.len());
        for row in 0..rows {
            let data_row = &data_slice[row * row_len..][..row_len];
            for &index in &index_slice[row * gathered_len..][..gathered_len] {
                let resolved = if index < 0 { index + row_len as i64 } else { index };
                let value = usize::try_from(resolved)
                    .ok()
                    .and_then(|resolved| data_row.get(resolved))
                    .with_context(|| {
                        format!(
                            "Invalid GatherElements index {index} in row {row} on axis of len {row_len}"
                        )
                    })?;
                output.push(value.clone());
            }
        }
        Ok(Some(ArrayD::from_shape_vec(indices.shape(), output)?))
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<TValue> {
        let data_plain = data.try_as_plain()?;
        let data_view = unsafe { data_plain.to_array_view_unchecked::<T>() };
        let output = match self.eval_contiguous_last_axis::<T>(&data_view, indices)? {
            Some(output) => output,
            None => ArrayD::<T>::from_shape_fn(indices.shape(), |mut coords| {
                let index = indices[&coords];
                coords[self.axis] =
                    if index < 0 { index + data_view.shape()[self.axis] as i64 } else { index }
                        as usize;
                data_view[coords].clone()
            }),
        };
        let mut tensor = output.into_tensor();
        unsafe { tensor.set_datum_type(data.datum_type()) };
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for GatherElements {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs[0].rank() == inputs[1].rank(),
            "GatherElements data and indices must have the same rank, got {} and {}",
            inputs[0].rank(),
            inputs[1].rank()
        );
        ensure!(
            self.axis < inputs[0].rank(),
            "GatherElements axis {} is out of range for rank {}",
            self.axis,
            inputs[0].rank()
        );
        Ok(tvec!(inputs[0].datum_type.fact(&*inputs[1].shape)))
    }
}

impl EvalOp for GatherElements {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_plain_array_view::<i64>()?;
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices
            ))?))
        }
    }
}

/// The out-of-range branches are only reachable on the contiguous last-axis path
/// (the generic path panics inside `ndarray` instead), so they cannot be covered
/// by the output-comparing `suite-unit` cases.
#[cfg(test)]
mod tests {
    use super::*;

    fn gather(data_shape: &[usize], indices: &[i64]) -> TractResult<TValue> {
        let len = data_shape.iter().product::<usize>();
        let data = Tensor::from_shape(data_shape, &(0..len).map(|i| i as f32).collect::<Vec<_>>())?;
        let indices = Tensor::from_shape(&[1, indices.len()], indices)?;
        let mut outputs = GatherElements::new(data_shape.len() - 1)
            .eval(&EvalContext::out_of_plan(), tvec!(data.into_tvalue(), indices.into_tvalue()))?;
        Ok(outputs.remove(0))
    }

    #[test]
    fn last_axis_resolves_negative_indices() {
        let output = gather(&[1, 4], &[-1, 0, -4, 2]).unwrap();
        assert_eq!(output.try_as_plain().unwrap().as_slice::<f32>().unwrap(), [3., 0., 0., 2.]);
    }

    #[test]
    fn last_axis_rejects_index_past_the_end() {
        assert!(gather(&[1, 4], &[0, 4]).is_err());
    }

    #[test]
    fn last_axis_rejects_index_before_the_start() {
        assert!(gather(&[1, 4], &[0, -5]).is_err());
    }

    #[test]
    fn last_axis_rejects_any_index_into_an_empty_axis() {
        assert!(gather(&[1, 0], &[0]).is_err());
    }
}
