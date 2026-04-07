use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_itertools::Itertools;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuAxisOp {
    pub inner: AxisOp,
}

impl GpuAxisOp {
    pub fn new(inner: AxisOp) -> Self {
        Self { inner }
    }

    pub fn simplify_axis_op(op: AxisOp, dims: &[TDim]) -> Self {
        let inner = match op {
            AxisOp::Move(from, to) if from.abs_diff(to) == 1 => {
                if [&dims[from], &dims[to]].contains(&&1usize.into()) {
                    if from < to {
                        AxisOp::Reshape(
                            from,
                            tvec![dims[from].clone(), dims[to].clone()],
                            tvec![dims[to].clone(), dims[from].clone()],
                        )
                    } else {
                        AxisOp::Reshape(
                            to,
                            tvec![dims[to].clone(), dims[from].clone()],
                            tvec![dims[from].clone(), dims[to].clone()],
                        )
                    }
                } else {
                    op
                }
            }
            AxisOp::Move(from, to) if dims[from] == TDim::Val(1) => {
                let (start, end) = if from < to { (from, to) } else { (to, from) };
                let mut out_dims = dims[start..=end].to_vec();

                if from < to {
                    let tmp = out_dims.remove(0);
                    out_dims.push(tmp);
                } else {
                    let tmp = out_dims.pop().unwrap();
                    out_dims.insert(0, tmp);
                }

                AxisOp::Reshape(start, dims[start..=end].into(), out_dims.into())
            }
            _ => op,
        };
        Self { inner }
    }

    pub fn from_tract_core_with_fact(op: AxisOp, fact: &TypedFact) -> Self {
        let dims = fact.shape.dims();
        Self::simplify_axis_op(op, dims)
    }
}

impl Op for GpuAxisOp {
    fn name(&self) -> StaticName {
        format!("Gpu{}", self.inner.name()).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.inner.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuAxisOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let tensor = args_1!(inputs).into_tensor();
        let input = tensor.to_device_tensor()?;
        let shape = input.shape();

        let simplified = Self::simplify_axis_op(
            self.inner.clone(),
            &shape.iter().map(|s| s.into()).collect_vec(),
        );

        let new_shape = match &simplified.inner {
            AxisOp::Move(from, to) => {
                let mut permutation: Vec<usize> = (0..input.rank()).collect();
                permutation.remove(*from);
                permutation.insert(*to, *from);

                let out_shape = permute_output_shape(input.shape(), &permutation)?;
                let output = crate::session_handler::make_tensor_for_node(
                    session,
                    node_id,
                    input.datum_type(),
                    &out_shape,
                )?;
                // Compute permuted input strides
                let permuted_strides: TVec<isize> =
                    permutation.iter().map(|&i| input.strides()[i]).collect();
                let ctx = crate::device::get_context()?;
                ctx.copy_nd(
                    input,
                    0,
                    &permuted_strides,
                    &output,
                    0,
                    output.shape(),
                    output.strides(),
                )?;
                return Ok(tvec!(output.into_tensor().into_tvalue()));
            }
            AxisOp::Reshape(skip, from, to) => {
                let from = from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                let to = to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                let mut shape: TVec<usize> = input.shape().into();
                AxisOp::Reshape(*skip, from, to).change_shape_array(&mut shape, false)?;
                shape
            }
            _ => {
                let mut shape: TVec<usize> = input.shape().into();
                self.inner.change_shape_array(&mut shape, false)?;
                shape
            }
        };

        // Memcpy path (Reshape/Add/Rm) — flat copy, treat as 1D
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &new_shape,
        )?;
        let flat_len = input.len();
        let ctx = crate::device::get_context()?;
        ctx.copy_nd(input, 0, &[1], &output, 0, &[flat_len], &[1])?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuAxisOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let ref_inputs = crate::utils::get_device_facts(inputs, |facts| Ok(facts.to_vec()))?;
        let ref_outputs = crate::utils::get_device_facts(outputs, |facts| Ok(facts.to_vec()))?;
        self.inner.axes_mapping(&ref_inputs, &ref_outputs)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let inner = if let AxisOp::Reshape(axis, from, to) = &self.inner {
            AxisOp::Reshape(
                *axis,
                from.iter().map(|d| d.eval(values)).collect(),
                to.iter().map(|d| d.eval(values)).collect(),
            )
        } else {
            self.inner.clone()
        };
        let op = GpuAxisOp { inner };
        target.wire_node(&node.name, op, &[mapping[&node.inputs[0]]])
    }

    as_op!();
}

pub fn permute_output_shape(shape: &[usize], permutation: &[usize]) -> TractResult<TVec<usize>> {
    ensure!(shape.len() == permutation.len());
    Ok(permutation.iter().map(|&i| shape[i]).collect())
}
