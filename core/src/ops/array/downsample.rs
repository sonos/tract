use crate::internal::*;
use crate::ops;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Downsample {
    axis: usize,
    stride: usize,
    modulo: usize,
}

impl Downsample {
    fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Arc<Tensor>> {
        let input = input.to_array_view::<T>()?;
        let sampled = if self.modulo < input.shape()[self.axis] {
            input
                .slice_axis(
                    Axis(self.axis),
                    ndarray::Slice::new(self.modulo as isize, None, self.stride as isize),
                )
                .to_owned()
                .into_arc_tensor()
        } else {
            let mut shape = input.shape().to_vec();
            shape[self.axis] = 0;
            unsafe { Tensor::uninitialized::<T>(&shape)?.into_arc_tensor() }
        };
        Ok(sampled)
    }
}

impl Op for Downsample {
    fn name(&self) -> Cow<str> {
        "Downsample".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.stride == 1 {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        }
        pull_downsample_up(model, node)
    }
}

impl StatelessOp for Downsample {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, &*input))?))
    }
}

impl InferenceRulesOp for Downsample {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].rank, move |s, r| {
            for i in 0..(r as usize) {
                if i == self.axis {
                    s.given(&inputs[0].shape[i], move |s, d| {
                        s.equals(
                            &outputs[0].shape[i],
                            (d - self.modulo).div_ceil(self.stride.to_dim()),
                        )
                    })?
                } else {
                    s.equals(&inputs[0].shape[i], &outputs[0].shape[i])?
                }
            }
            Ok(())
        })
    }

    inference_op_as_op!();
}

fn pull_downsample_up(
    model: &TypedModel,
    down_node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let down_op = down_node.op_as::<Downsample>().unwrap();
    if let Some(prec) = model.single_prec(down_node.id)? {
        if let Some(crop_op) = prec.op_as::<ops::array::Crop>() {
            return pull_downsample_over_crop(model, prec, crop_op, down_node, down_op);
        }
    }
    Ok(None)
}

fn pull_downsample_over_crop(
    model: &TypedModel,
    crop_node: &TypedNode,
    crop_op: &ops::array::Crop,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let modulo = (down_op.modulo + crop_op.prune[down_op.axis].0) % down_op.stride;
    let left = (down_op.modulo + crop_op.prune[down_op.axis].0) / down_op.stride;
    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, crop_node.inputs[0])?;
    let input_outlet = crop_node.inputs[0].clone();
    let input_fact = model.outlet_fact(input_outlet).unwrap();
    let input_len = input_fact.shape.dim(down_op.axis);
    let final_len = down_node.outputs[0].fact.shape.dim(down_op.axis);
    let mut downed = input_fact.clone();
    let midway_len = (input_len - modulo).div_ceil(down_op.stride.into());
    downed.shape.set_dim(down_op.axis, midway_len.clone())?;
    patch.chain(
        &down_node.name,
        Downsample::new(down_op.axis, down_op.stride, modulo),
        tvec!(downed),
    )?;
    let mut new_prunes = crop_op.prune.clone();
    new_prunes[down_op.axis].0 = left;
    new_prunes[down_op.axis].1 =
        (midway_len.to_dim() - final_len.to_dim() - left).to_integer()? as usize;
    let new_crop = patch.chain(
        &crop_node.name,
        ops::array::Crop::new(new_prunes),
        tvec!(down_node.outputs[0].fact.clone()),
    )?;
    patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(new_crop, 0))?;
    return Ok(Some(patch));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    fn crop_then_down_strat() -> BoxedStrategy<(usize, usize, usize, usize, usize)> {
        (1usize..5, 1usize..5)
            .prop_flat_map(|(cropped, stride)| {
                (Just(cropped), 0..=cropped, Just(stride), (cropped + 15)..=(cropped + 15))
            })
            .prop_flat_map(|(cropped, left, stride, len)| {
                (Just(len), Just(left), Just(cropped - left), Just(stride), 0..stride)
            })
            .boxed()
    }

    fn crop_then_down(
        len: usize,
        left: usize,
        right: usize,
        stride: usize,
        modulo: usize,
    ) -> TestCaseResult {
        let model = {
            let mut model = InferenceModel::default();
            model.add_source("input", TensorFact::dt_shape(i32::datum_type(), vec![len]))?;
            model.chain_default("crop", ops::array::Crop::new(vec![(left, right)]))?;
            model.chain_default("down", Downsample::new(0, stride, modulo))?;
            model.auto_outputs()?;
            model
        };
        prop_assert!(model.node(model.output_outlets().unwrap()[0].node).op_is::<Downsample>());
        let typed = model.into_typed()?;
        let input = tensor1(&(0i32..len as _).collect::<Vec<_>>());
        let expected = SimplePlan::new(&typed)?.run(tvec!(input.clone()))?;

        let typed = typed.declutter()?;
        prop_assert!(!typed.node(typed.output_outlets().unwrap()[0].node).op_is::<Downsample>());
        let found = SimplePlan::new(&typed)?.run(tvec!(input))?;
        prop_assert_eq!(found, expected);
        Ok(())
    }

    proptest! {
        #[test]
        fn crop_then_down_prop((len, left, right, stride, modulo) in crop_then_down_strat()) {
            crop_then_down(len, left, right, stride, modulo).unwrap()
        }
    }

    #[test]
    fn crop_then_down_1() {
        crop_then_down(1, 0, 0, 2, 0).unwrap()
    }

    #[test]
    fn crop_then_down_2() {
        crop_then_down(2, 0, 1, 2, 0).unwrap()
    }

    #[test]
    fn crop_then_down_3() {
        crop_then_down(0, 0, 0, 2, 1).unwrap()
    }

    #[test]
    fn crop_then_down_4() {
        crop_then_down(1, 0, 1, 2, 1).unwrap()
    }

    #[test]
    fn crop_then_down_5() {
        crop_then_down(16, 0, 1, 2, 1).unwrap()
    }
}
