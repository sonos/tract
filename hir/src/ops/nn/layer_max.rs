use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash, Serialize, Deserialize)]
pub struct LayerHardmax {
    axis: isize,
}

tract_linalg::impl_dyn_hash!(LayerHardmax);

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

impl StatelessOp for LayerHardmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

#[typetag::serde]
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

tract_linalg::impl_dyn_hash!(LayerLogSoftmax);

impl LayerLogSoftmax {
    fn eval_t<
        T: Datum + tract_num_traits::Float + tract_num_traits::FromPrimitive + ::std::iter::Sum,
    >(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<Tensor> {
        let mut softmax = LayerSoftmax::new(self.axis).eval_t::<T>(input)?;
        softmax.as_slice_mut::<T>()?.iter_mut().for_each(|x| *x = x.ln());
        Ok(softmax)
    }
}

impl Op for LayerLogSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerLogSoftmax".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }
    fn validation(&self) -> Validation {
        Validation::Rounding
    }
    op_hir!();
    not_a_typed_op!();
}

impl StatelessOp for LayerLogSoftmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let t = dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl InferenceRulesOp for LayerLogSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let softmax =
            InferenceOp::to_typed(&LayerSoftmax::new(self.axis), source, node, target, mapping)?[0];
        target.wire_node(
            format!("{}-logsoftmax", node.name),
            tract_core::ops::math::ln(),
            &[softmax],
        )
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerSoftmax {
    axis: isize,
}

tract_linalg::impl_dyn_hash!(LayerSoftmax);

impl LayerSoftmax {
    fn eval_t<
        T: Datum + tract_num_traits::Float + tract_num_traits::FromPrimitive + ::std::iter::Sum,
    >(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<Tensor> {
        let array = input.into_tensor().into_array::<T>()?;
        let shape = array.shape().to_vec();
        let axis =
            if self.axis < 0 { shape.len() as isize + self.axis } else { self.axis } as usize;
        let first_dim: usize = array.shape()[0..axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            // https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
            let max: Option<T> = layer
                .iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
                .cloned();
            layer.mapv_inplace(|x| (x - max.unwrap()).exp());
            let divisor = layer.iter().cloned().sum();
            layer.mapv_inplace(|x| x / divisor);
        });
        Ok(array.into_shape(shape)?.into_tensor())
    }
}

impl Op for LayerSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerSoftmax".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }
    fn validation(&self) -> Validation {
        Validation::Rounding
    }
    op_hir!();
    not_a_typed_op!();
}

impl StatelessOp for LayerSoftmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let t = dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl InferenceRulesOp for LayerSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::nn;
        let input = mapping[&node.inputs[0]];
        let rank = target.outlet_fact(input)?.rank();
        let axis = if self.axis < 0 { rank as isize + self.axis } else { self.axis } as usize;
        let reducing_axes = (axis..rank).collect::<TVec<usize>>();
        let maxes = target.wire_node(
            format!("{}-max", node.name),
            nn::Reduce::new(reducing_axes.clone(), nn::Reducer::Max),
            &[input],
        )?[0];
        let normed = target.wire_node(
            format!("{}-normed", node.name),
            tract_core::ops::math::sub::bin_typed(),
            &[input, maxes],
        )?[0];
        let exp = target.wire_node(
            format!("{}-exp", node.name),
            tract_core::ops::math::exp(),
            &[normed],
        )?[0];
        let sum = target.wire_node(
            format!("{}-sum", node.name),
            nn::Reduce::new(reducing_axes, nn::Reducer::Sum),
            &[exp],
        )?[0];
        target.wire_node(
            format!("{}-softmax", node.name),
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
