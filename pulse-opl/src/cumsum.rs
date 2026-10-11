use tract_nnef::internal::*;
use tract_nnef::tract_core::tract_data::internal::tract_num_traits::Zero;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_pulse_cumsum",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Logical.named("exclusive"),
            TypeName::Integer.named("skip"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser,
    );
    registry.register_dumper(ser)
}

fn ser(ast: &mut IntoAst, node: &TypedNode, op: &PulsedCumSum) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let params = vec![
        ("axis", numeric(op.axis)),
        ("exclusive", logical(op.exclusive)),
        ("skip", numeric(op.skip)),
    ];
    Ok(Some(invocation("tract_pulse_cumsum", &[wire], &params)))
}

fn deser(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let exclusive = invocation.named_arg_as(builder, "exclusive")?;
    let skip = invocation.named_arg_as(builder, "skip")?;
    builder.wire(PulsedCumSum { axis, exclusive, skip }, &[wire])
}

/// Forward cumulative sum along the streaming `axis`, carried across pulses.
///
/// The first `skip` frames of the stream are the delay ahead of valid data: they are
/// left out of the running sum and come out as zero. Frames are counted from the first
/// pulse; the state holds one stream, so the op is not lane-aware.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct PulsedCumSum {
    pub axis: usize,
    pub exclusive: bool,
    pub skip: usize,
}

#[derive(Debug, Clone, Default)]
struct PulsedCumSumState {
    sum: Option<Tensor>,
    pos: usize,
}

impl PulsedCumSumState {
    fn eval_t<T: Datum + Copy + std::ops::AddAssign + Zero>(
        &mut self,
        op: &PulsedCumSum,
        input: &mut Tensor,
    ) -> TractResult<()> {
        let shape = input.shape().to_vec();
        let pulse = shape[op.axis];
        let inner: usize = shape[op.axis + 1..].iter().product();
        let outer: usize = shape[..op.axis].iter().product();
        if self.sum.is_none() {
            self.sum = Some(Tensor::zero::<T>(&[outer * inner])?);
        }
        let mut sum_plain = self.sum.as_mut().unwrap().try_as_plain_ram_mut()?;
        let sum = sum_plain.as_slice_mut::<T>()?;
        ensure!(sum.len() == outer * inner, "PulsedCumSum pulse shape changed between turns");
        let mut plain = input.try_as_plain_ram_mut()?;
        let data = plain.as_slice_mut::<T>()?;
        if pulse > 0 && inner > 0 {
            for (block, acc) in
                data.chunks_exact_mut(pulse * inner).zip(sum.chunks_exact_mut(inner))
            {
                for (frame, row) in block.chunks_exact_mut(inner).enumerate() {
                    if self.pos + frame < op.skip {
                        row.fill(T::zero());
                    } else if op.exclusive {
                        for (x, a) in row.iter_mut().zip(acc.iter_mut()) {
                            let value = *x;
                            *x = *a;
                            *a += value;
                        }
                    } else {
                        for (x, a) in row.iter_mut().zip(acc.iter_mut()) {
                            *a += *x;
                            *x = *a;
                        }
                    }
                }
            }
        }
        self.pos += pulse;
        Ok(())
    }
}

impl OpState for PulsedCumSumState {
    fn eval(
        &mut self,
        _ctx: &EvalContext,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulsedCumSum>().context("Wrong Op type")?;
        let mut input = args_1!(inputs).into_tensor();
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, op, &mut input))?;
        Ok(tvec!(input.into_tvalue()))
    }

    fn reset_lanes(&mut self, _lanes: &[LaneId]) -> TractResult<()> {
        bail!("PulsedCumSum is not lane-aware: its running sum has no lane axis")
    }
}

impl Op for PulsedCumSum {
    fn name(&self) -> StaticName {
        "PulsedCumSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {} exclusive: {} skip: {}", self.axis, self.exclusive, self.skip)])
    }

    op_as_typed_op!();
}

impl EvalOp for PulsedCumSum {
    not_out_of_plan!();

    fn state(&self, _ctx: &EvalContext) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::<PulsedCumSumState>::default()))
    }
}

impl TypedOp for PulsedCumSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.clone())))
    }

    as_op!();
}
