use crate::internal::*;

#[derive(Debug, Clone, Hash)]
pub struct StridedSlice {
    pub optional_axes_input: Option<usize>,
    pub optional_steps_input: Option<usize>,
    pub begin_mask: i64,
    pub end_mask: i64,
    pub shrink_axis_mask: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dim {
    // position of the first element to return
    pub begin: TDim,
    // position of the first element not to return
    pub end: TDim,
    pub stride: i32,
    pub shrink: bool,
}

impl Dim {
    pub fn soft_len(&self) -> TractResult<TDim> {
        if let Ok(len) = (self.end.clone() - &self.begin).to_isize() {
            Ok((((self.stride.abs() - 1) + len.abs() as i32) / self.stride.abs()).to_dim())
        } else if self.stride == 1 {
            Ok(self.end.clone() - &self.begin)
        } else {
            bail!("Streaming dimensions with strides are not supported for now")
        }
    }
}

impl StridedSlice {
    fn must_shrink(&self, ix: usize) -> bool {
        self.shrink_axis_mask & (1 << ix) != 0
    }
    fn ignore_begin(&self, ix: usize) -> bool {
        self.begin_mask & (1 << ix) != 0
    }
    fn ignore_end(&self, ix: usize) -> bool {
        self.end_mask & (1 << ix) != 0
    }
    pub fn prepare_one_dim(
        &self,
        ix: usize,
        dim: &TDim,
        begin: &Tensor,
        end: &Tensor,
        strides: &[i32],
    ) -> TractResult<Dim> {
        // cast bouds to Option<Dim>, dealing with ignore from mask, and spec shorted than dim
        // also for end, magic values in onnx :/
        let mut begin: Option<TDim> = if ix >= begin.len() {
            None
        } else {
            let begin = begin.cast_to::<TDim>()?;
            begin.as_slice::<TDim>()?.get(ix).cloned()
        };

        let mut end: Option<TDim> = if self.ignore_end(ix) || ix >= end.len() {
            None
        } else if end.datum_type() == i64::datum_type() {
            let end = *end.as_slice::<i64>()?.get(ix).unwrap();
            if end == i64::MAX || end == i64::MIN || end == i64::MIN + 1 || end == (i32::MAX as i64)
            {
                None
            } else {
                Some(end.to_dim())
            }
        } else {
            let end = end.cast_to::<TDim>()?;
            end.as_slice::<TDim>()?.get(ix).cloned()
        };

        let stride = strides.get(ix).cloned().unwrap_or(1);

        // deal with negative indexing
        fn fix_negative(bound: &mut TDim, dim: &TDim) {
            let neg = if let Ok(b) = bound.to_isize() {
                b < 0
            } else {
                #[allow(clippy::mutable_key_type)]
                let symbols = bound.symbols();
                if symbols.len() == 1 {
                    let sym = symbols.into_iter().next().unwrap();
                    let values = SymbolValues::default().with(&sym, 100_000_000);
                    bound.eval(&values).to_isize().unwrap() < 0
                } else {
                    false
                }
            };
            if neg {
                *bound = bound.clone() + dim;
            }
        }
        if let Some(begin) = begin.as_mut() {
            fix_negative(begin, dim)
        }
        if let Some(end) = end.as_mut() {
            fix_negative(end, dim)
        }

        if self.must_shrink(ix) {
            return Ok(Dim {
                begin: begin.clone().unwrap_or_else(|| 0.to_dim()),
                end: begin.unwrap_or_else(|| 0.to_dim()) + 1,
                stride: 1,
                shrink: true,
            });
        }

        // must happen after dealing with must_shrink :/
        if self.ignore_begin(ix) {
            begin = None;
        }

        let mut begin =
            begin.unwrap_or_else(|| if stride > 0 { 0.to_dim() } else { dim.clone() - 1 });
        if begin.to_isize().map(|b| b < 0).unwrap_or(false) {
            if stride < 0 {
                return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
            } else {
                begin = 0.to_dim();
            }
        }
        if let (Ok(b), Ok(d)) = (begin.to_isize(), dim.to_isize()) {
            if b > d - 1 {
                if stride > 0 {
                    return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
                } else {
                    begin = (d - 1).to_dim()
                }
            }
        }

        let mut end = end.unwrap_or_else(|| if stride > 0 { dim.clone() } else { (-1).to_dim() });
        if end.to_isize().map(|e| e < 0).unwrap_or(false) {
            if stride > 0 {
                return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
            } else {
                end = (-1).to_dim();
            }
        }
        if let (Ok(e), Ok(d)) = (end.to_isize(), dim.to_isize()) {
            if e > d - 1 {
                if stride > 0 {
                    end = d.to_dim()
                } else {
                    return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
                }
            }
        }
        Ok(Dim { begin, end, stride, shrink: false })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let params: TVec<Option<Arc<Tensor>>> = inputs[1..]
            .iter()
            .map(|i| Ok(target.outlet_fact(*i)?.konst.clone()))
            .collect::<TractResult<_>>()?;
        let input_shape = target.outlet_fact(inputs[0])?.shape.clone();
        let strides: TVec<i32> = if let Some(i) = self.optional_steps_input {
            let strides = params[i - 1]
                .as_ref()
                .context("StridedSlice is typable only if stride is a const")?
                .cast_to::<i32>()?;
            strides.as_slice::<i32>()?.into()
        } else {
            tvec![1; input_shape.rank()]
        };
        let axes: TVec<usize> = if let Some(i) = self.optional_axes_input {
            let axes = params[i - 1]
                .as_ref()
                .context("StridedSlice is typable only if axis is a const")?
                .cast_to::<i32>()?;
            axes.as_slice::<i32>()?
                .iter()
                .map(|&i| if i < 0 { input_shape.rank() as i32 + i } else { i } as usize)
                .collect()
        } else {
            (0..input_shape.rank()).collect()
        };
        let mut wire = inputs[0];
        let begin = params[0].as_ref();
        let end = params[1].as_ref();
        for (ix, &axis) in axes.iter().enumerate() {
            if let (Some(begin), Some(end)) = (begin, end) {
                let d = &input_shape[axis];
                let preped = self.prepare_one_dim(ix, d, begin, end, &strides)?;
                let (left, right) = if preped.stride > 0 {
                    (preped.begin, preped.end)
                } else {
                    (preped.end + 1, preped.begin + 1)
                };
                wire = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}"),
                    crate::ops::array::Slice::new(axis, left, right),
                    [wire].as_ref(),
                )?[0];
                if preped.stride != 1 {
                    wire = target.wire_node(
                        format!("{prefix}.stride-axis-{axis}"),
                        crate::ops::downsample::Downsample::new(axis, preped.stride as isize, 0),
                        [wire].as_ref(),
                    )?[0];
                }
            } else if strides[ix] == 1 {
                let left = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}-start"),
                    crate::ops::array::Slice::new(0, ix, ix + 1),
                    &[inputs[1]],
                )?;
                let left = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}-start-rm-axis"),
                    AxisOp::Rm(0),
                    &left,
                )?[0];
                let right = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}-end"),
                    crate::ops::array::Slice::new(0, ix, ix + 1),
                    &[inputs[2]],
                )?;
                let right = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}-end-rm-axis"),
                    AxisOp::Rm(0),
                    &right,
                )?[0];
                let sym = target.symbols.new_with_prefix("l");
                wire = target.wire_node(
                    format!("{prefix}.slice-axis-{axis}"),
                    crate::ops::array::DynSlice::new(axis, sym.to_dim()),
                    &[wire, left, right],
                )?[0];
            }
        }
        let mut shrink = input_shape
            .iter()
            .enumerate()
            .filter(|(ix, _d)| self.must_shrink(*ix))
            .map(|pair| pair.0)
            .collect::<Vec<_>>();
        shrink.sort();
        for axis in shrink.iter().rev() {
            wire = target.wire_node(
                format!("{prefix}.RmDim-{axis}"),
                AxisOp::Rm(*axis),
                [wire].as_ref(),
            )?[0];
        }
        target.rename_node(wire.node, prefix)?;
        Ok(tvec!(wire))
    }
}

impl Op for StridedSlice {
    fn name(&self) -> Cow<str> {
        "StridedSlice".into()
    }

    op_as_typed_op!();
}

impl EvalOp for StridedSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut model = TypedModel::default();
        let mut source = tvec!();
        for (ix, input) in inputs.iter().enumerate() {
            source.push(model.add_source(
                format!("adhoc_input.{ix}"),
                input.clone().into_arc_tensor().into(),
            )?);
        }
        let output = self.wire("adhoc", &mut model, &source)?;
        model.set_output_outlets(&output)?;
        model.into_runnable()?.run(inputs)
    }
}

impl TypedOp for StridedSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut model = TypedModel::default();
        let mut source = tvec!();
        for (ix, input) in inputs.iter().enumerate() {
            source.push(model.add_source(format!("adhoc_input.{ix}"), (*input).clone())?);
        }
        let output = self.wire("adhoc", &mut model, &source)?;
        model.set_output_outlets(&output)?;
        Ok(tvec!(model.outlet_fact(output[0])?.clone()))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();
        let mut source = tvec!();
        for &input in &node.inputs {
            source.push(patch.tap_model(model, input)?);
        }
        let output = self.wire(&node.name, &mut patch, &source)?;
        patch.shunt_outside(model, node.id.into(), output[0])?;
        Ok(Some(patch))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(
        input: &[i32],
        start: Option<isize>,
        end: Option<isize>,
        stride: Option<isize>,
    ) -> TValue {
        // [0,1,2,3,4,5][::2] => [0, 2, 4]
        let op = StridedSlice {
            optional_axes_input: None,
            optional_steps_input: if stride.is_some() { Some(3) } else { None },
            begin_mask: if start.is_some() { 0 } else { 1 },
            end_mask: if end.is_some() { 0 } else { 1 },
            shrink_axis_mask: 0,
        };
        let mut inputs = tvec!(
            tensor1(input).into(),
            tensor1(&[start.unwrap_or(0) as i32]).into(),
            tensor1(&[end.unwrap_or(0) as i32]).into(),
        );
        if let Some(stride) = stride {
            inputs.push(tensor1(&[stride as i32]).into());
        }
        op.eval(inputs).unwrap().remove(0)
    }

    #[test]
    fn numpy_pos_stride() {
        // [0,1,2,3][::2] => [0, 2]
        assert_eq!(apply(&[0, 1, 2, 3], None, None, Some(2)), tensor1(&[0, 2]).into());
    }

    #[test]
    fn numpy_neg_stride() {
        // [0,1,2,3][::-2] => [3, 1]
        assert_eq!(apply(&[0, 1, 2, 3], None, None, Some(-2)), tensor1(&[3, 1]).into());
    }

    #[test]
    fn numpy_neg_stride_with_start_even() {
        // [0,1,2,3][-1::-2] => [3, 1]
        assert_eq!(apply(&[0, 1, 2, 3], Some(-1), None, Some(-2)), tensor1(&[3, 1]).into());
    }

    #[test]
    fn numpy_neg_stride_with_start_odd() {
        // [0,1,2,3][-1::-2] => [3, 1]
        assert_eq!(apply(&[0, 1, 2, 3, 4], Some(-1), None, Some(-2)), tensor1(&[4, 2, 0]).into());
    }
}
