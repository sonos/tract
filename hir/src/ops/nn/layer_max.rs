use crate::infer::*;
use crate::internal::*;

// TODO tricky to re-express in "core" because of the multiple hot point... do
// we need one more reduce ?
#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerHardmax {
    axis: isize,
}

tract_data::impl_dyn_hash!(LayerHardmax);

impl LayerHardmax {
    fn eval_t<D: Datum + tract_num_traits::Float + tract_num_traits::FromPrimitive>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.into_tensor().into_array::<D>()?;
        let shape = array.shape().to_vec();
        let axis =
            if self.axis < 0 { shape.len() as isize + self.axis } else { self.axis } as usize;
        let first_dim: usize = array.shape()[0..axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            let max = layer
                .iter()
                .enumerate()
                .rev()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(b.0.cmp(&a.0)))
                .map(|(ix, _)| ix)
                .unwrap_or(0);
            layer
                .iter_mut()
                .enumerate()
                .for_each(|(ix, r)| *r = D::from_usize((ix == max) as usize).unwrap());
        });
        Ok(tvec!(array.into_shape(shape)?.into_arc_tensor()))
    }
}

impl Op for LayerHardmax {
    fn name(&self) -> Cow<str> {
        "LayerHardmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_hir!();
    op_as_typed_op!();
}

impl InferenceRulesOp for LayerHardmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();
    to_typed!();
}

impl EvalOp for LayerHardmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl TypedOp for LayerHardmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerLogSoftmax {
    axis: isize,
}

tract_data::impl_dyn_hash!(LayerLogSoftmax);

impl Expansion for LayerLogSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerLogSoftmax".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let softmax = LayerSoftmax { axis: self.axis }.wire(name, target, inputs)?;
        target.wire_node(format!("{}.logsoftmax", name), tract_core::ops::math::ln(), &softmax)
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerSoftmax {
    axis: isize,
}

tract_data::impl_dyn_hash!(LayerSoftmax);

impl Expansion for LayerSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerSoftmax".into()
    }
    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::{math, nn};
        let input = inputs[0];
        let rank = target.outlet_fact(input)?.rank();
        let axis = if self.axis < 0 { rank as isize + self.axis } else { self.axis } as usize;
        let reducing_axes = (axis..rank).collect::<TVec<usize>>();
        let maxes = target.wire_node(
            format!("{}.max", name),
            nn::Reduce::new(reducing_axes.clone(), nn::Reducer::Max),
            &[input],
        )?[0];
        let normed = target.wire_node(
            format!("{}.normed", name),
            math::sub::bin_typed(),
            &[input, maxes],
        )?[0];
        let exp =
            target.wire_node(format!("{}.exp", name), tract_core::ops::math::exp(), &[normed])?[0];
        let sum = target.wire_node(
            format!("{}.sum", name),
            nn::Reduce::new(reducing_axes, nn::Reducer::Sum),
            &[exp],
        )?[0];
        target.wire_node(
            format!("{}.softmax", name),
            tract_core::ops::math::div::bin_typed(),
            &[exp, sum],
        )
    }
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_output_arity(&outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape, &inputs[0].shape)?;
    Ok(())
}
